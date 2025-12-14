"""
🔬 OPTUNA BASELINE HYPERPARAMETER SEARCH
Uses Optuna to find optimal labeling strategy parameters

Searches:
- Profit target: 3% to 15%
- Stop loss: -3% to -12%
- Time horizon: 1 to 14 days
- Labeling method: Simple vs Triple Barrier

Goal: Maximize test accuracy while minimizing overfitting
Runtime: ~30 minutes for 100 trials
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import optuna
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

from src.ml.feature_engineer_56 import FeatureEngineer70


def create_labels(df, profit_target, stop_loss, horizon_days, method='simple'):
    """Create labels with given parameters"""
    close = df['close'].values
    labels = np.zeros(len(close))
    
    if method == 'simple':
        # Simple forward return
        for i in range(len(close) - horizon_days):
            future_prices = close[i+1:i+1+horizon_days]
            entry_price = close[i]
            
            returns = (future_prices - entry_price) / entry_price
            max_return = returns.max()
            min_return = returns.min()
            
            if max_return >= profit_target:
                labels[i] = 1
            elif min_return <= stop_loss:
                labels[i] = -1
            else:
                labels[i] = 0
    
    elif method == 'triple_barrier':
        # Triple barrier (institutional)
        for i in range(len(close) - horizon_days):
            future_prices = close[i+1:i+1+horizon_days]
            entry_price = close[i]
            
            hit_profit = False
            hit_stop = False
            
            for future_price in future_prices:
                ret = (future_price - entry_price) / entry_price
                
                if ret >= profit_target:
                    hit_profit = True
                    break
                elif ret <= stop_loss:
                    hit_stop = True
                    break
            
            if hit_profit:
                labels[i] = 1
            elif hit_stop:
                labels[i] = -1
            else:
                final_ret = (future_prices[-1] - entry_price) / entry_price
                labels[i] = 1 if final_ret > 0 else 0
    
    return labels


def load_sample_data():
    """Load cached sample data or fetch new"""
    cache_file = 'data/optuna_sample_cache.pkl'
    
    if os.path.exists(cache_file):
        print("📁 Loading cached sample data...")
        return pd.read_pickle(cache_file)
    
    print("🔍 Fetching sample data (20 tickers for speed)...")
    tickers = [
        'NVDA', 'AMD', 'TSLA', 'PLTR', 'HOOD', 'IONQ', 'PALI', 'ASTS',
        'MRVL', 'COIN', 'MARA', 'AVGO', 'AAPL', 'MSFT', 'GOOGL',
        'QS', 'LAZR', 'RKLB', 'CRIS', 'SMR'
    ]
    
    all_data = []
    for ticker in tickers:
        try:
            features = FeatureEngineer70.download_and_engineer(ticker, period='1y')
            if features is not None and len(features) >= 100:
                features['ticker'] = ticker
                all_data.append(features)
        except:
            pass
    
    df = pd.concat(all_data, ignore_index=True)
    
    # Cache for future runs
    os.makedirs('data', exist_ok=True)
    df.to_pickle(cache_file)
    print(f"   Cached {len(df)} rows to {cache_file}")
    
    return df


def objective(trial):
    """Optuna objective function"""
    
    # Hyperparameters to search
    profit_target = trial.suggest_float('profit_target', 0.03, 0.15, step=0.01)
    stop_loss = trial.suggest_float('stop_loss', -0.12, -0.03, step=0.01)
    horizon_days = trial.suggest_int('horizon_days', 1, 14)
    method = trial.suggest_categorical('method', ['simple', 'triple_barrier'])
    
    # Create labels
    labels = create_labels(DF_GLOBAL, profit_target, stop_loss, horizon_days, method)
    
    # Filter
    df_labeled = DF_GLOBAL.copy()
    df_labeled['label'] = labels
    df_labeled = df_labeled[df_labeled['label'] != 0].copy()
    
    if len(df_labeled) < 500:
        # Not enough data
        return 0.0
    
    # Check class balance
    unique, counts = np.unique(df_labeled['label'].values, return_counts=True)
    if len(unique) < 2:
        return 0.0
    
    min_class_size = counts.min()
    if min_class_size < 100:
        return 0.0
    
    # Prepare data
    feature_cols = [col for col in df_labeled.columns if col not in ['ticker', 'label']]
    X = df_labeled[feature_cols].values
    y = df_labeled['label'].values
    
    X = np.nan_to_num(X, nan=0.0, posinf=1e9, neginf=-1e9)
    
    # Train/test split
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
    except:
        return 0.0
    
    # Quick RF
    clf = RandomForestClassifier(
        n_estimators=50,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    
    clf.fit(X_train, y_train)
    
    # Score
    train_acc = clf.score(X_train, y_train)
    test_acc = clf.score(X_test, y_test)
    
    # Penalize overfitting
    overfit_penalty = abs(train_acc - test_acc) * 0.5
    
    # Final score: test accuracy - overfit penalty
    score = test_acc - overfit_penalty
    
    return score


if __name__ == "__main__":
    print("="*60)
    print("🔬 OPTUNA BASELINE HYPERPARAMETER SEARCH")
    print("="*60)
    
    # Load data once
    print("\n📊 Loading sample data...")
    DF_GLOBAL = load_sample_data()
    print(f"   Rows: {len(DF_GLOBAL)}")
    print(f"   Features: {len(DF_GLOBAL.columns) - 1}")
    
    # Create study
    print("\n🔍 Starting Optuna search (100 trials)...")
    print("   Searching: profit_target, stop_loss, horizon_days, method")
    
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    study.optimize(objective, n_trials=100, show_progress_bar=True)
    
    # Results
    print("\n" + "="*60)
    print("📊 SEARCH RESULTS")
    print("="*60)
    
    print(f"\n🏆 BEST TRIAL:")
    print(f"   Score: {study.best_value:.4f}")
    print(f"   Params:")
    for key, value in study.best_params.items():
        print(f"      {key:20s}: {value}")
    
    print(f"\n📈 TOP 5 TRIALS:")
    top_trials = sorted(study.trials, key=lambda t: t.value if t.value is not None else 0, reverse=True)[:5]
    
    for i, trial in enumerate(top_trials, 1):
        print(f"\n   #{i} - Score: {trial.value:.4f}")
        print(f"      Profit: {trial.params['profit_target']:.1%}")
        print(f"      Stop:   {trial.params['stop_loss']:.1%}")
        print(f"      Horizon: {trial.params['horizon_days']} days")
        print(f"      Method: {trial.params['method']}")
    
    # Save results
    results_file = 'data/optuna_best_params.json'
    import json
    with open(results_file, 'w') as f:
        json.dump(study.best_params, f, indent=2)
    
    print(f"\n💾 Best params saved to: {results_file}")
    print(f"\n✅ Search complete! Use these params for A100 training.")
