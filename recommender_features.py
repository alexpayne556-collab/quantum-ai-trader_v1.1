import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple

EMA_SET = (8, 13, 21, 34, 55)

def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for span in EMA_SET:
        out[f"ema_{span}"] = _ema(out["Close"], span)
    out["rsi_14"] = _rsi(out["Close"], 14)
    out["atr_14"] = _atr(out, 14)
    out["vol_zscore"] = (out["Volume"] - out["Volume"].rolling(50).mean()) / (out["Volume"].rolling(50).std() + 1e-9)
    # Ribbon spread as max-min of EMA distances
    ema_cols = [f"ema_{s}" for s in EMA_SET]
    dists = [(out["Close"] - out[c]) / (out[c] + 1e-9) for c in ema_cols]
    out["ema_ribbon_spread"] = np.nanmax(np.vstack([d.values for d in dists]), axis=0) - np.nanmin(np.vstack([d.values for d in dists]), axis=0)
    # Trend slopes
    out["slope_short"] = out["ema_8"].diff()
    out["slope_long"] = out["ema_55"].diff()
    # Regime class (simple):
    regime = np.where(out["ema_8"] > out["ema_55"], "BULL", np.where(out["ema_8"] < out["ema_55"], "BEAR", "CHOP"))
    out["regime_class"] = regime
    return out

def _rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up, index=series.index).ewm(alpha=1/period, adjust=False).mean()
    roll_down = pd.Series(down, index=series.index).ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-12)
    return 100 - (100 / (1 + rs))

def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    high = df["High"]; low = df["Low"]; close = df["Close"]
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()

def build_feature_row(latest_df: pd.DataFrame, cached: Optional[Dict[str, Any]] = None, ai_trader_ctx: Optional[Dict[str, Any]] = None, pattern_flags: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    row = latest_df.iloc[-1]
    feats = {
        "ts": str(row.name),
        "price": float(row["Close"]),
        "ema": {f"ema_{s}": float(row.get(f"ema_{s}", np.nan)) for s in EMA_SET},
        "ema_ribbon_spread": float(row.get("ema_ribbon_spread", np.nan)),
        "rsi_14": float(row.get("rsi_14", np.nan)),
        "atr_14": float(row.get("atr_14", np.nan)),
        "vol_zscore": float(row.get("vol_zscore", np.nan)),
        "trend_features": {
            "slope_short": float(row.get("slope_short", 0.0)),
            "slope_long": float(row.get("slope_long", 0.0)),
            "regime_class": str(row.get("regime_class", "CHOP"))
        },
        "pattern_flags": pattern_flags or {},
        "pattern_stage": int(cached.get("pattern_stage", 0) if cached else 0),
        "ai_trader_action": (ai_trader_ctx or {}).get("action", "FLAT"),
        "ai_trader_score": float((ai_trader_ctx or {}).get("score", 0.0)),
        "position_context": (ai_trader_ctx or {}).get("position_context", {"current_pos": "flat", "size_fraction": 0.0, "unrealized_pnl_pct": 0.0})
    }
    # Optional higher timeframe context if present
    if "rsi_14_h1" in latest_df.columns or "regime_class_h1" in latest_df.columns:
        feats["context_h1"] = {
            "ema_ribbon_spread": float(row.get("ema_ribbon_spread_h1", np.nan)),
            "rsi_14": float(row.get("rsi_14_h1", np.nan)),
            "atr_14": float(row.get("atr_14_h1", np.nan)),
            "slope_short": float(row.get("slope_short_h1", 0.0)),
            "slope_long": float(row.get("slope_long_h1", 0.0)),
            "regime_class": str(row.get("regime_class_h1", "CHOP"))
        }
    if "rsi_14_d1" in latest_df.columns or "regime_class_d1" in latest_df.columns:
        feats["context_d1"] = {
            "ema_ribbon_spread": float(row.get("ema_ribbon_spread_d1", np.nan)),
            "rsi_14": float(row.get("rsi_14_d1", np.nan)),
            "atr_14": float(row.get("atr_14_d1", np.nan)),
            "slope_short": float(row.get("slope_short_d1", 0.0)),
            "slope_long": float(row.get("slope_long_d1", 0.0)),
            "regime_class": str(row.get("regime_class_d1", "CHOP"))
        }
    return feats

def build_label_row(future_window: pd.Series, atr: float, horizon_bars: int = 10) -> Dict[str, Any]:
    if len(future_window) == 0:
        return {"direction": 0, "ret_over_atr": 0.0}
    close0 = future_window.iloc[0]
    ret = (future_window.iloc[-1] - close0) / (close0 + 1e-12)
    direction = 1 if ret > 0 else 0
    ret_over_atr = float(ret / (atr + 1e-9))
    return {"direction": direction, "ret_over_atr": ret_over_atr}

# --- Intraday multi-timeframe utilities ---
def load_multi_timeframe_ohlc(
    ticker: str,
    period_5m: str = "60d",
    period_60m: str = "730d",
    period_1d: str = "5y"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Download 5m, 60m and 1d OHLCV using yfinance with safe default periods.
    Returns (df_5m, df_60m, df_1d). Empty frames are possible if unavailable.
    """
    try:
        import yfinance as yf  # local import to avoid hard dependency when unused
    except Exception:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    def _dl(interval: str, period: str) -> pd.DataFrame:
        try:
            return yf.download(ticker, interval=interval, period=period, auto_adjust=True, progress=False)
        except Exception:
            return pd.DataFrame()

    df_5m = _dl("5m", period_5m)
    df_60m = _dl("60m", period_60m)
    df_1d = _dl("1d", period_1d)
    return df_5m, df_60m, df_1d


def compute_intraday_with_context(
    ticker: str,
    period_5m: str = "60d",
    period_60m: str = "730d",
    period_1d: str = "5y"
) -> pd.DataFrame:
    """
    Build a 5m feature frame enriched with 1h and 1d context.
    Output columns include base indicators plus *_h1 and *_d1 suffixed context.
    """
    base5, h1, d1 = load_multi_timeframe_ohlc(ticker, period_5m, period_60m, period_1d)
    if base5.empty:
        return base5

    # Compute indicators on each timeframe independently
    base5i = compute_indicators(base5)
    h1i = compute_indicators(h1) if not h1.empty else pd.DataFrame(index=base5i.index)
    d1i = compute_indicators(d1) if not d1.empty else pd.DataFrame(index=base5i.index)

    # Select a compact set of context features to project down to 5m index
    def _select_ctx(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(index=base5i.index)
        cols = [
            "ema_ribbon_spread", "rsi_14", "atr_14", "slope_short", "slope_long", "regime_class",
        ]
        sub = df[[c for c in cols if c in df.columns]].copy()
        return sub

    base_index = base5i.index
    h1_ctx = _select_ctx(h1i)
    d1_ctx = _select_ctx(d1i)

    # Align context to 5m timeline with forward-fill semantics
    if not h1_ctx.empty:
        h1_ctx = h1_ctx.reindex(base_index, method="ffill")
        h1_ctx.columns = [f"{c}_h1" for c in h1_ctx.columns]
    else:
        h1_ctx = pd.DataFrame(index=base_index)

    if not d1_ctx.empty:
        d1_ctx = d1_ctx.reindex(base_index, method="ffill")
        d1_ctx.columns = [f"{c}_d1" for c in d1_ctx.columns]
    else:
        d1_ctx = pd.DataFrame(index=base_index)

    combined = pd.concat([base5i, h1_ctx, d1_ctx], axis=1)
    return combined

