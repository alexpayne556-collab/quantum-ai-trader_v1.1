from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Iterable

import pandas as pd
import yfinance as yf


@dataclass(frozen=True)
class EarningsStudyConfig:
    cache_dir: Path
    limit_events: int = 12
    horizon_days: int = 5
    use_cache: bool = True
    benchmark_ticker: str | None = None
    timing_aware: bool = True


def _flatten_yf(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    return df


def get_earnings_events(ticker: str, *, limit: int = 12) -> pd.DataFrame:
    t = yf.Ticker(ticker)
    ed = t.get_earnings_dates(limit=limit)
    if ed is None or len(ed) == 0:
        raise RuntimeError(f"No earnings dates returned for {ticker}")

    ed = ed.reset_index().rename(columns={"Earnings Date": "earnings_datetime"})
    ed["earnings_date"] = pd.to_datetime(ed["earnings_datetime"]).dt.date
    return ed


def get_daily_prices(ticker: str, *, start: str, end: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
    df = _flatten_yf(df)
    if df is None or len(df) == 0:
        raise RuntimeError(f"No daily prices for {ticker} from {start} to {end}")

    df = df.copy()
    df.index = pd.to_datetime(df.index).date
    return df


def classify_earnings_session(earnings_datetime) -> str:
    """Classify earnings timing using local timestamp heuristics.

    Returns: before_open | during_session | after_close | unknown
    """
    try:
        ts = pd.to_datetime(earnings_datetime)
    except Exception:
        return "unknown"
    if pd.isna(ts):
        return "unknown"

    # If no time component, treat as unknown.
    if ts.hour == 0 and ts.minute == 0 and ts.second == 0 and getattr(ts, "nanosecond", 0) == 0:
        return "unknown"

    # US equities regular session: 09:30-16:00 (local exchange time unknown here)
    if (ts.hour < 9) or (ts.hour == 9 and ts.minute < 30):
        return "before_open"
    if ts.hour >= 16:
        return "after_close"
    return "during_session"


def earnings_event_study(
    ticker: str,
    *,
    cache_dir: Path,
    limit_events: int = 12,
    horizon_days: int = 5,
    use_cache: bool = True,
    benchmark_ticker: str | None = None,
    timing_aware: bool = True,
) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Preserve historical cache naming unless a benchmark is explicitly requested.
    cache_path = (
        cache_dir / f"{ticker}_earnings_event_study_h{horizon_days}.csv"
        if not benchmark_ticker
        else cache_dir / f"{ticker}_earnings_event_study_bm{benchmark_ticker}_h{horizon_days}.csv"
    )

    if use_cache and cache_path.exists():
        return pd.read_csv(cache_path)

    events = get_earnings_events(ticker, limit=limit_events)
    min_dt = pd.to_datetime(events["earnings_datetime"]).min().date() - timedelta(days=10)
    max_dt = pd.to_datetime(events["earnings_datetime"]).max().date() + timedelta(days=30)

    px = get_daily_prices(ticker, start=min_dt.isoformat(), end=max_dt.isoformat())
    bm = None
    if benchmark_ticker:
        bm = get_daily_prices(benchmark_ticker, start=min_dt.isoformat(), end=max_dt.isoformat())
    trading_days = list(px.index)

    def prev_day(d):
        prevs = [x for x in trading_days if x < d]
        return prevs[-1] if prevs else None

    def next_day_ge(d):
        nexts = [x for x in trading_days if x >= d]
        return nexts[0] if nexts else None

    def next_day_gt(d):
        nexts = [x for x in trading_days if x > d]
        return nexts[0] if nexts else None

    def plus_n(d, n):
        if d not in trading_days:
            return None
        i = trading_days.index(d)
        j = i + n
        return trading_days[j] if j < len(trading_days) else None

    rows: list[dict] = []
    for _, r in events.iterrows():
        edate = r["earnings_date"]
        prev = prev_day(edate)
        session = classify_earnings_session(r["earnings_datetime"])
        if timing_aware and session == "after_close":
            evt = next_day_gt(edate)
        else:
            evt = next_day_ge(edate)
        if not prev or not evt:
            continue

        evt_plus = plus_n(evt, horizon_days)
        if not evt_plus:
            continue

        prev_close = float(px.loc[prev, "Close"])
        evt_open = float(px.loc[evt, "Open"])
        evt_close = float(px.loc[evt, "Close"])
        h_close = float(px.loc[evt_plus, "Close"])

        gap = (evt_open / prev_close) - 1.0
        day1 = (evt_close / prev_close) - 1.0
        drift = (h_close / prev_close) - 1.0

        # Tradable-style returns (enter at event open, exit later)
        ret_open_to_close_1d = (evt_close / evt_open) - 1.0
        ret_open_to_close_hd = (h_close / evt_open) - 1.0

        bm_gap = bm_day1 = bm_drift = None
        ab_gap = ab_day1 = ab_drift = None
        if bm is not None:
            try:
                # Require benchmark dates to exist; otherwise leave abnormal as None.
                if prev in bm.index and evt in bm.index and evt_plus in bm.index:
                    bm_prev_close = float(bm.loc[prev, "Close"])
                    bm_evt_open = float(bm.loc[evt, "Open"])
                    bm_evt_close = float(bm.loc[evt, "Close"])
                    bm_h_close = float(bm.loc[evt_plus, "Close"])
                    bm_gap = (bm_evt_open / bm_prev_close) - 1.0
                    bm_day1 = (bm_evt_close / bm_prev_close) - 1.0
                    bm_drift = (bm_h_close / bm_prev_close) - 1.0
                    ab_gap = gap - bm_gap
                    ab_day1 = day1 - bm_day1
                    ab_drift = drift - bm_drift
            except Exception:
                pass

        eps_est = r.get("EPS Estimate", None)
        eps_act = r.get("Reported EPS", None)
        is_pos = None
        surprise = None
        try:
            if pd.notna(eps_est) and pd.notna(eps_act):
                eps_est_f = float(eps_est)
                eps_act_f = float(eps_act)
                surprise = eps_act_f - eps_est_f
                is_pos = surprise >= 0
        except Exception:
            pass

        rows.append(
            {
                "ticker": ticker,
                "benchmark": benchmark_ticker,
                "earnings_datetime": str(r["earnings_datetime"]),
                "earnings_date": str(edate),
                "earnings_session": session,
                "prev_trading_day": str(prev),
                "event_trading_day": str(evt),
                f"event_plus_{horizon_days}d": str(evt_plus),
                "prev_close": prev_close,
                "event_open": evt_open,
                "event_close": evt_close,
                f"event_plus_{horizon_days}d_close": h_close,
                "gap_prevclose_to_open": gap,
                "ret_prevclose_to_close_1d": day1,
                f"ret_prevclose_to_close_{horizon_days}d": drift,
                "ret_open_to_close_1d": ret_open_to_close_1d,
                f"ret_open_to_close_{horizon_days}d": ret_open_to_close_hd,
                "bm_gap_prevclose_to_open": bm_gap,
                "bm_ret_prevclose_to_close_1d": bm_day1,
                f"bm_ret_prevclose_to_close_{horizon_days}d": bm_drift,
                "ab_gap_prevclose_to_open": ab_gap,
                "ab_ret_prevclose_to_close_1d": ab_day1,
                f"ab_ret_prevclose_to_close_{horizon_days}d": ab_drift,
                "eps_estimate": eps_est,
                "eps_reported": eps_act,
                "eps_surprise": surprise,
                "positive_eps_surprise": is_pos,
            }
        )

    out = pd.DataFrame(rows)
    out.to_csv(cache_path, index=False)
    return out


def summarize_by_eps_surprise(events: pd.DataFrame, *, horizon_days: int, include_abnormal: bool = True) -> pd.DataFrame:
    if events is None or len(events) == 0:
        return pd.DataFrame()

    have_eps = events[events["positive_eps_surprise"].notna()].copy()
    if len(have_eps) == 0:
        return pd.DataFrame()

    cols = [
        "gap_prevclose_to_open",
        "ret_prevclose_to_close_1d",
        f"ret_prevclose_to_close_{horizon_days}d",
        "ret_open_to_close_1d",
        f"ret_open_to_close_{horizon_days}d",
    ]
    if include_abnormal:
        ab_cols = [
            "ab_gap_prevclose_to_open",
            "ab_ret_prevclose_to_close_1d",
            f"ab_ret_prevclose_to_close_{horizon_days}d",
        ]
        cols.extend([c for c in ab_cols if c in have_eps.columns])
    return have_eps.groupby("positive_eps_surprise")[cols].agg(["count", "mean", "median"])


def batch_event_study(
    tickers: Iterable[str],
    *,
    config: EarningsStudyConfig,
    progress_every: int = 25,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    failures: list[dict] = []

    tickers_list = list(tickers)
    total = len(tickers_list)

    for i, ticker in enumerate(tickers_list, start=1):
        try:
            df_t = earnings_event_study(
                ticker,
                cache_dir=config.cache_dir,
                limit_events=config.limit_events,
                horizon_days=config.horizon_days,
                use_cache=config.use_cache,
                benchmark_ticker=config.benchmark_ticker,
                timing_aware=config.timing_aware,
            )
            frames.append(df_t)
        except Exception as e:
            failures.append({"ticker": ticker, "error": str(e)})

        if progress_every and (i % progress_every == 0):
            print(f"... {i}/{total} tickers done")

    events = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    failures_df = pd.DataFrame(failures)
    return events, failures_df
