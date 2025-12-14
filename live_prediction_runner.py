"""
Live Prediction Runner
----------------------
Loads trained per-ticker models and produces live predictions with:
 - Regime-aware gating (via MarketRegimeManager + RiskManager)
 - Position sizing preview
 - Logging to CSV: live_performance_log.csv

Usage (Colab / local):
    python live_prediction_runner.py --tickers AAPL MSFT NVDA SPY QQQ --capital 25000

CSV Columns:
 timestamp,ticker,price,signal,confidence,regime,position_shares,position_risk
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timedelta
from typing import List, Dict

import pandas as pd
import yfinance as yf
from joblib import load

from market_regime_manager import MarketRegimeManager
from risk_manager import RiskManager, MarketRegime
from ai_recommender import AIRecommender

LOG_PATH = 'live_performance_log.csv'
MODELS_DIR = 'models'


def load_model_artifacts(ticker: str):
    path = os.path.join(MODELS_DIR, f"{ticker}_model.joblib")
    if not os.path.exists(path):
        return None
    try:
        bundle = load(path)
        return bundle
    except Exception:
        return None


def predict_for_ticker(ticker: str, bundle, regime: str, risk: RiskManager) -> Dict:
    start = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    df = yf.download(ticker, start=start, progress=False, auto_adjust=True)
    if len(df) < 50:
        return {'ticker': ticker, 'error': 'insufficient_data'}

    rec = AIRecommender()
    # Reuse engineered features pipeline
    X = rec.FeatureEngineer.engineer(df) if hasattr(rec, 'FeatureEngineer') else None  # fallback
    # Use last row transform via existing scaler + selector
    if bundle.get('scaler') and bundle.get('feature_selector'):
        latest = X.iloc[-1:]
        xs = bundle['scaler'].transform(latest)
        xs = bundle['feature_selector'].transform(xs)
        probs = bundle['model'].predict_proba(xs)[0]
        classes = bundle['model'].classes_
        mapping = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
        best_idx = int(classes[probs.argmax()])
        signal = mapping.get(best_idx, 'HOLD')
        confidence = float(probs.max())
    else:
        # Fallback simple rule
        close = df['Close'].values
        rsi = pd.Series(close).pct_change().rolling(14).std().iloc[-1]
        signal = 'BUY' if rsi < 0.01 else 'SELL'
        confidence = 0.5

    # Position sizing preview
    price = float(df['Close'].iloc[-1])
    shares_preview = 0
    risk_amount_preview = 0.0
    regime_enum = MarketRegime[regime] if regime in MarketRegime.__members__ else MarketRegime.UNKNOWN
    if risk.can_trade(regime_enum):
        # Use synthetic 2*ATR stop approximation
        atr = df['High'].rolling(14).max().iloc[-1] - df['Low'].rolling(14).min().iloc[-1]
        stop_loss = price - (atr * 0.5 if atr and atr > 0 else price * 0.02)
        shares_preview, risk_amount_preview = risk.calculate_position_size(price, stop_loss, regime_enum)

    return {
        'ticker': ticker,
        'price': price,
        'signal': signal,
        'confidence': confidence,
        'regime': regime,
        'position_shares': shares_preview,
        'position_risk': risk_amount_preview
    }


def append_log(row: Dict):
    df_row = pd.DataFrame([{
        'timestamp': datetime.utcnow().isoformat(),
        **row
    }])
    if not os.path.exists(LOG_PATH):
        df_row.to_csv(LOG_PATH, index=False)
    else:
        df_row.to_csv(LOG_PATH, mode='a', header=False, index=False)


def run_live(tickers: List[str], capital: float):
    mgr = MarketRegimeManager()
    regime_info = mgr.calculate_market_regime()
    regime = regime_info.get('regime', 'UNKNOWN')
    print(f"📊 Current Regime: {regime}")

    risk = RiskManager(initial_capital=capital)

    results = []
    for t in tickers:
        bundle = load_model_artifacts(t)
        if bundle is None:
            print(f"   ⚠️ Missing model for {t}, skipping")
            continue
        res = predict_for_ticker(t, bundle, regime, risk)
        if 'error' in res:
            print(f"   ⚠️ {t} error: {res['error']}")
            continue
        append_log(res)
        results.append(res)
        print(f"   {t}: {res['signal']} ({res['confidence']:.2f}) shares={res['position_shares']} risk=${res['position_risk']:.2f}")

    print("\n✅ Live prediction batch complete")
    return results


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tickers', nargs='+', required=True, help='Ticker list')
    ap.add_argument('--capital', type=float, default=10000.0, help='Initial capital for sizing preview')
    return ap.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run_live(args.tickers, args.capital)
