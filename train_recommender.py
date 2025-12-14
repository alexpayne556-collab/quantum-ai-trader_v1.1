import os
import json
from datetime import datetime
from typing import List, Dict, Any

import numpy as np
import pandas as pd

try:
    import xgboost as xgb  # optional
    HAS_XGB = True
except Exception:
    HAS_XGB = False

from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.calibration import calibration_curve
import joblib

from COMPREHENSIVE_SYSTEM_TEST_AND_TRAINER import prepare_train_test_data, DEFAULT_TICKERS
from recommender_features import compute_indicators, build_feature_row, build_label_row, compute_intraday_with_context

OUTPUT_DIR = "models"
os.makedirs(OUTPUT_DIR, exist_ok=True)

HORIZON_BARS = 10  # default intraday horizon; works for daily too

FEATURE_KEYS_ORDER = [
    "price","ema_ribbon_spread","rsi_14","atr_14","vol_zscore",
    "trend_features.slope_short","trend_features.slope_long",
    # Encoded regime_class will be one-hot expanded
]

REGIME_CLASSES = ["BULL","BEAR","CHOP"]
REGIME_CLASSES_H1 = REGIME_CLASSES
REGIME_CLASSES_D1 = REGIME_CLASSES


def _extract_features(feat: Dict[str, Any]) -> List[float]:
    vals = [
        feat.get("price", np.nan),
        feat.get("ema_ribbon_spread", np.nan),
        feat.get("rsi_14", np.nan),
        feat.get("atr_14", np.nan),
        feat.get("vol_zscore", np.nan),
        feat.get("trend_features", {}).get("slope_short", 0.0),
        feat.get("trend_features", {}).get("slope_long", 0.0),
    ]
    # regime one-hot
    regime = feat.get("trend_features", {}).get("regime_class", "CHOP")
    vals.extend([1.0 if regime == r else 0.0 for r in REGIME_CLASSES])
    # Optional higher timeframe context if present
    ctx_h1 = feat.get("context_h1")
    if ctx_h1:
        vals.extend([
            ctx_h1.get("ema_ribbon_spread", np.nan),
            ctx_h1.get("rsi_14", np.nan),
            ctx_h1.get("atr_14", np.nan),
            ctx_h1.get("slope_short", 0.0),
            ctx_h1.get("slope_long", 0.0),
        ])
        r1 = ctx_h1.get("regime_class", "CHOP")
        vals.extend([1.0 if r1 == r else 0.0 for r in REGIME_CLASSES_H1])
    ctx_d1 = feat.get("context_d1")
    if ctx_d1:
        vals.extend([
            ctx_d1.get("ema_ribbon_spread", np.nan),
            ctx_d1.get("rsi_14", np.nan),
            ctx_d1.get("atr_14", np.nan),
            ctx_d1.get("slope_short", 0.0),
            ctx_d1.get("slope_long", 0.0),
        ])
        r2 = ctx_d1.get("regime_class", "CHOP")
        vals.extend([1.0 if r2 == r else 0.0 for r in REGIME_CLASSES_D1])
    return [0.0 if (v is None or np.isnan(v)) else float(v) for v in vals]


def build_dataset(train_df: pd.DataFrame) -> pd.DataFrame:
    df = compute_indicators(train_df)
    rows = []
    for i in range(len(df) - HORIZON_BARS - 1):
        window = df.iloc[: i + 1]
        feat = build_feature_row(window, cached=None, ai_trader_ctx=None, pattern_flags={})
        future = df["Close"].iloc[i + 1 : i + 1 + HORIZON_BARS]
        label = build_label_row(future, float(df["atr_14"].iloc[i]), horizon_bars=HORIZON_BARS)
        X = _extract_features(feat)
        rows.append({
            "ts": window.index[-1],
            "X": X,
            "direction": label["direction"],
            "ret_over_atr": label["ret_over_atr"]
        })
    return pd.DataFrame(rows)


def build_intraday_dataset_for_ticker(ticker: str, horizon_bars: int = HORIZON_BARS) -> pd.DataFrame:
    """Build intraday (5m) samples with 1h/daily context for a single ticker."""
    df = compute_intraday_with_context(ticker)
    if df is None or df.empty:
        raise RuntimeError(f"No intraday data for {ticker}")
    rows = []
    for i in range(len(df) - horizon_bars - 1):
        window = df.iloc[: i + 1]
        feat = build_feature_row(window, cached=None, ai_trader_ctx=None, pattern_flags={})
        future = df["Close"].iloc[i + 1 : i + 1 + horizon_bars]
        label = build_label_row(future, float(df["atr_14"].iloc[i]), horizon_bars=horizon_bars)
        X = _extract_features(feat)
        rows.append({
            "ts": window.index[-1],
            "X": X,
            "direction": label["direction"],
            "ret_over_atr": label["ret_over_atr"],
        })
    return pd.DataFrame(rows)


def train_models(tickers: List[str], intraday: bool = False) -> Dict[str, Any]:
    frames = []
    if intraday:
        # Build from fresh intraday downloads
        for t in tickers:
            try:
                ds = build_intraday_dataset_for_ticker(t)
                ds["ticker"] = t
                frames.append(ds)
            except Exception as e:
                print(f"Skip {t} (intraday): {e}")
        TRAIN_DATA = {t: None for t in tickers}  # metadata only
    else:
        TRAIN_DATA, TEST_DATA = prepare_train_test_data(tickers=tickers, train_days=365, test_days=90)
        for t in TRAIN_DATA:
            try:
                ds = build_dataset(TRAIN_DATA[t])
                ds["ticker"] = t
                frames.append(ds)
            except Exception as e:
                print(f"Skip {t}: {e}")
    if not frames:
        raise RuntimeError("No training data built")
    data = pd.concat(frames, ignore_index=True)

    # Train/val split (time-based 80/20)
    cutoff = int(len(data) * 0.8)
    X = np.vstack(data["X"].values)
    y_dir = data["direction"].values
    y_ret = data["ret_over_atr"].values

    X_train, X_val = X[:cutoff], X[cutoff:]
    y_dir_train, y_dir_val = y_dir[:cutoff], y_dir[cutoff:]
    y_ret_train, y_ret_val = y_ret[:cutoff], y_ret[cutoff:]

    # Direction model
    if HAS_XGB:
        clf = xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, n_jobs=2)
    else:
        clf = GradientBoostingClassifier(n_estimators=300, max_depth=3)
    clf.fit(X_train, y_dir_train)
    dir_pred = clf.predict(X_val)
    dir_proba = clf.predict_proba(X_val)[:,1] if hasattr(clf, "predict_proba") else (dir_pred * 1.0)
    dir_acc = float(accuracy_score(y_dir_val, dir_pred))

    # Return model
    if HAS_XGB:
        reg = xgb.XGBRegressor(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, n_jobs=2)
    else:
        reg = GradientBoostingRegressor(n_estimators=400, max_depth=3)
    reg.fit(X_train, y_ret_train)
    ret_pred = reg.predict(X_val)
    ret_mae = float(mean_absolute_error(y_ret_val, ret_pred))

    # Simple calibration (reliability curve stored for reference)
    prob_true, prob_pred = calibration_curve(y_dir_val, dir_proba, n_bins=10, strategy='uniform')
    calib = {"bins": prob_pred.tolist(), "empirical": prob_true.tolist()}

    # Persist
    joblib.dump(clf, os.path.join(OUTPUT_DIR, "recommender_dir_model.pkl"))
    joblib.dump(reg, os.path.join(OUTPUT_DIR, "recommender_ret_model.pkl"))

    meta = {
        "timestamp": datetime.utcnow().isoformat(),
        "tickers": list(TRAIN_DATA.keys()),
        "features": {
            "order": FEATURE_KEYS_ORDER,
            "regime_onehot": REGIME_CLASSES,
            "horizon_bars": HORIZON_BARS
        },
        "metrics": {
            "direction_val_acc": dir_acc,
            "ret_val_mae": ret_mae
        },
        "calibration": calib
    }
    with open(os.path.join(OUTPUT_DIR, "recommender_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return meta

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--intraday", action="store_true", help="Train on 5m with 1h/daily context")
    parser.add_argument("--tickers", nargs="*", default=DEFAULT_TICKERS[:20], help="Tickers to train")
    parser.add_argument("--horizon", type=int, default=HORIZON_BARS, help="Prediction horizon in bars")
    args = parser.parse_args()

    HORIZON_BARS = args.horizon
    meta = train_models(args.tickers, intraday=args.intraday)
    print("Saved models to", OUTPUT_DIR)
    print("Val acc:", meta["metrics"]["direction_val_acc"])