"""
Regime Retrain Scheduler
-----------------------
Monitors market regime + model performance and triggers selective retraining.
Rules:
 - If regime changes (e.g., BULL -> BEAR / CORRECTION -> CRASH) retrain all momentum-sensitive tickers.
 - If cv_mean < threshold for any ticker OR walk-forward Sharpe < 0.5 retrain that ticker only.

Usage:
    python regime_retrain_scheduler.py --tickers AAPL MSFT NVDA SPY QQQ MU APLD IONQ ANNX TSLA --capital 20000
Assumes models already trained and artifacts in /models.
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from typing import List, Dict, Any

from market_regime_manager import MarketRegimeManager
from ai_recommender import AIRecommender
try:
    from multi_ticker_trainer import train_single_ticker, MODELS_DIR
except Exception:
    # Fallback: inform user to add repo path in Colab
    train_single_ticker = None
    MODELS_DIR = 'models'
try:
    from config.improvement_loader import load_improvements
except Exception:
    def load_improvements(path=None):
        return {}, {}

PERFORMANCE_META_SUFFIX = '_meta.json'
CV_THRESHOLD = 0.38
WF_SHARPE_THRESHOLD = 0.5

MOMENTUM_SENSITIVE = { 'AAPL','MSFT','NVDA','TSLA','QQQ','SPY' }


def load_meta(ticker: str) -> Dict[str, Any]:
    path = os.path.join(MODELS_DIR, f"{ticker}{PERFORMANCE_META_SUFFIX}")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception:
        return {}


def detect_regime_change(history: List[str], current: str) -> bool:
    return len(history) > 0 and history[-1] != current


def should_retrain(meta: Dict[str, Any]) -> bool:
    if not meta:
        return True
    cv = meta.get('cv_mean', 0)
    wf_sharpe = meta.get('walk_forward_sharpe', 0)
    return cv < CV_THRESHOLD or (wf_sharpe is not None and wf_sharpe < WF_SHARPE_THRESHOLD)


def run_scheduler(tickers: List[str]) -> Dict[str, Any]:
    mgr = MarketRegimeManager()
    regime_info = mgr.calculate_market_regime()
    current_regime = regime_info.get('regime','UNKNOWN')

    # Load previous regime if exists
    prev_regime_path = 'last_regime.txt'
    prev_regime = None
    if os.path.exists(prev_regime_path):
        with open(prev_regime_path,'r') as f:
            prev_regime = f.read().strip()

    regime_changed = prev_regime != current_regime if prev_regime else False
    if regime_changed:
        print(f"⚠️ Regime change detected: {prev_regime} -> {current_regime}")
    else:
        print(f"Regime stable: {current_regime}")

    with open(prev_regime_path,'w') as f:
        f.write(current_regime)

    retrain_targets = []
    meta_summary = {}

    for t in tickers:
        meta = load_meta(t)
        meta_summary[t] = meta
        if regime_changed and t in MOMENTUM_SENSITIVE:
            retrain_targets.append(t)
        elif should_retrain(meta):
            retrain_targets.append(t)

    retrain_targets = sorted(set(retrain_targets))
    print(f"Retrain targets: {', '.join(retrain_targets) if retrain_targets else 'NONE'}")

    recommender = AIRecommender()
    retrain_results = {}
    for t in retrain_targets:
        if train_single_ticker is None:
            retrain_results[t] = {'error': 'multi_ticker_trainer not importable - ensure repo path is on sys.path'}
            continue
        try:
            weights, patterns = load_improvements()
            # Pass weights/patterns to trainer if supported
            try:
                res = train_single_ticker(t, recommender, walk_forward=True, weights=weights, pattern_features=patterns.get(t))
            except TypeError:
                res = train_single_ticker(t, recommender, walk_forward=True)
            retrain_results[t] = res.__dict__
        except Exception as exc:
            retrain_results[t] = {'error': str(exc)}

    result = {
        'timestamp': datetime.utcnow().isoformat(),
        'current_regime': current_regime,
        'regime_changed': regime_changed,
        'retrain_targets': retrain_targets,
        'retrain_results': retrain_results
    }

    with open('retrain_scheduler_result.json','w') as f:
        json.dump(result, f, indent=2)

    print("✅ Scheduler run complete")
    return result


def parse_args(argv: List[str] | None = None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--tickers', nargs='+', required=False, help='Ticker list')
    ap.add_argument('--default', action='store_true', help='Use default 10-ticker basket')
    ap.add_argument('--capital', type=float, default=10000.0, help='(unused placeholder) initial capital')
    return ap.parse_args(argv)

def _in_notebook() -> bool:
    try:
        get_ipython()
        return True
    except NameError:
        return False

if __name__ == '__main__':
    # In notebooks, avoid argparse hard failure; provide defaults
    if _in_notebook():
        # Ensure repo path is on sys.path in Colab
        import sys
        repo_path = '/content/drive/MyDrive/quantum-ai-trader-v1.1'
        if repo_path not in sys.path:
            sys.path.append(repo_path)
        default_tickers = ['AAPL','MSFT','NVDA','SPY','QQQ','MU','APLD','IONQ','ANNX','TSLA']
        run_scheduler(default_tickers)
    else:
        args = parse_args()
        if (not args.tickers) and args.default:
            args.tickers = ['AAPL','MSFT','NVDA','SPY','QQQ','MU','APLD','IONQ','ANNX','TSLA']
        if not args.tickers:
            print('❌ No tickers provided. Use --tickers or --default')
        else:
            run_scheduler(args.tickers)
