from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests
import yfinance as yf


@dataclass(frozen=True)
class EarningsCalendarConfig:
    days_ahead: int = 30
    finnhub_api_key: str | None = None
    request_timeout_s: int = 15


def _as_date(value) -> date | None:
    try:
        ts = pd.to_datetime(value)
        if pd.isna(ts):
            return None
        return ts.date()
    except Exception:
        return None


def fetch_upcoming_earnings_finnhub(
    *,
    from_date: date,
    to_date: date,
    finnhub_api_key: str,
    request_timeout_s: int = 15,
) -> pd.DataFrame:
    url = "https://finnhub.io/api/v1/calendar/earnings"
    params = {
        "from": from_date.isoformat(),
        "to": to_date.isoformat(),
        "token": finnhub_api_key,
    }
    resp = requests.get(url, params=params, timeout=request_timeout_s)
    resp.raise_for_status()
    data = resp.json() or {}
    rows = data.get("earningsCalendar", []) or []

    out = pd.DataFrame(rows)
    if len(out) == 0:
        return out

    # Normalize a minimal schema
    if "symbol" in out.columns:
        out = out.rename(columns={"symbol": "ticker"})
    if "date" in out.columns:
        out["earnings_date"] = out["date"].map(_as_date)
    else:
        out["earnings_date"] = None

    out["source"] = "finnhub"
    keep = [
        c
        for c in [
            "ticker",
            "earnings_date",
            "hour",
            "estimate",
            "actual",
            "quarter",
            "year",
            "revenueEstimate",
            "revenueActual",
            "source",
        ]
        if c in out.columns
    ]
    out = out[keep].copy()
    out["ticker"] = out["ticker"].astype(str).str.upper()
    out = out[out["ticker"].notna()]
    out = out.drop_duplicates(subset=["ticker", "earnings_date"]).reset_index(drop=True)
    return out


def fetch_next_earnings_yfinance(ticker: str) -> pd.DataFrame:
    """Best-effort next earnings date via yfinance per-ticker fields.

    This is a fallback when Finnhub isn't available.
    """
    t = yf.Ticker(ticker)

    # Try calendar
    try:
        cal = t.calendar
        if isinstance(cal, pd.DataFrame) and len(cal.columns) >= 1:
            if "Earnings Date" in cal.index:
                val = cal.loc["Earnings Date"].iloc[0]
                d = _as_date(val)
                if d:
                    return pd.DataFrame(
                        [{"ticker": ticker.upper(), "earnings_date": d, "source": "yfinance"}]
                    )
    except Exception:
        pass

    # Try get_earnings_dates (can include upcoming)
    try:
        ed = t.get_earnings_dates(limit=12)
        if ed is not None and len(ed) > 0:
            ed = ed.reset_index().rename(columns={"Earnings Date": "earnings_datetime"})
            ed["earnings_date"] = pd.to_datetime(ed["earnings_datetime"]).dt.date
            today = date.today()
            ed = ed[ed["earnings_date"].notna()].copy()
            ed = ed[ed["earnings_date"] >= today].sort_values("earnings_date")
            if len(ed):
                d = ed.iloc[0]["earnings_date"]
                dt = ed.iloc[0]["earnings_datetime"]
                return pd.DataFrame(
                    [
                        {
                            "ticker": ticker.upper(),
                            "earnings_date": d,
                            "earnings_datetime": str(dt),
                            "source": "yfinance",
                        }
                    ]
                )
    except Exception:
        pass

    # Try info
    try:
        info = t.info or {}
        # Yahoo sometimes returns a list of timestamps
        val = info.get("earningsDate")
        if isinstance(val, (list, tuple)) and val:
            d = _as_date(val[0])
        else:
            d = _as_date(val)
        if d:
            return pd.DataFrame([{"ticker": ticker.upper(), "earnings_date": d, "source": "yfinance"}])
    except Exception:
        pass

    return pd.DataFrame([])


def build_earnings_calendar(
    tickers: Iterable[str],
    *,
    config: EarningsCalendarConfig,
) -> pd.DataFrame:
    today = date.today()
    end = today + timedelta(days=config.days_ahead)

    if config.finnhub_api_key:
        cal = fetch_upcoming_earnings_finnhub(
            from_date=today,
            to_date=end,
            finnhub_api_key=config.finnhub_api_key,
            request_timeout_s=config.request_timeout_s,
        )
        # Filter to requested tickers (Finnhub returns market-wide calendar)
        wanted = {str(t).upper() for t in tickers}
        if len(cal) and wanted:
            cal = cal[cal["ticker"].isin(wanted)].reset_index(drop=True)
        return cal

    # Fallback: per-ticker yfinance next earnings
    frames: list[pd.DataFrame] = []
    for t in tickers:
        try:
            df = fetch_next_earnings_yfinance(str(t))
            if len(df):
                frames.append(df)
        except Exception:
            continue

    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame([])
    if len(out) and "earnings_date" in out.columns:
        out = out[out["earnings_date"].notna()].copy()
        out = out[(out["earnings_date"] >= today) & (out["earnings_date"] <= end)].reset_index(drop=True)
    return out


def save_calendar(df: pd.DataFrame, *, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
