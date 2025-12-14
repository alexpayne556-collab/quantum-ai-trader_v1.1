"""
🎯 QUICK BASELINE VALIDATOR
Tests 71 features on sample data locally BEFORE committing to A100

Purpose:
- Sample 50 tickers × 1 year = ~12,500 rows
- Engineer all 71 features
- Test multiple labeling strategies
- Predict baseline WR before full training
- Runtime: <10 minutes locally

Expected: 75-80% baseline WR
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

from src.ml.feature_engineer_56 import FeatureEngineer70


# Sample tickers (diverse mix)
SAMPLE_TICKERS = [
    # Your proven winners
    'NVDA', 'AMD', 'TSLA', 'PLTR', 'HOOD', 'IONQ', 'PALI', 'ASTS', 'MRVL',
    # Future tech
    'QS', 'SLDP', 'LAZR', 'MVIS', 'RKLB', 'LUNR', 'COIN', 'MARA',
    # Biotech
    'CRIS', 'NTLA', 'BEAM', 'AVIR', 'RLAY', 'DGNX',
    # Nuclear/Energy
    'SMR', 'OKLO', 'CCJ', 'UEC', 'ENPH',
    # Semiconductors
    'AVGO', 'AMAT', 'MPWR', 'CRUS', 'ALGM',
    # Quantum
    'QBTS', 'RGTI', 'ARQQ',
    # Large caps (context)
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'V', 'JPM', 'WMT', 'KO', 'PG'
]


def create_simple_labels(df, profit_target=0.05, stop_loss=-0.08, horizon_days=7):
    """
    Simple forward-looking labels
    
    Args:
        df: DataFrame with 'close' column
        profit_target: +5% = BUY signal
        stop_loss: -8% = SELL signal
        horizon_days: Look forward N days
        
    Returns:
        Series with labels: 1=BUY, 0=HOLD, -1=SELL
    """
    close = df['close'].values
    labels = np.zeros(len(close))
    
    for i in range(len(close) - horizon_days):
        future_prices = close[i+1:i+1+horizon_days]
        entry_price = close[i]
        
        returns = (future_prices - entry_price) / entry_price
        max_return = returns.max()
        min_return = returns.min()
        
        if max_return >= profit_target:
            labels[i] = 1  # BUY
        elif min_return <= stop_loss:
            labels[i] = -1  # SELL
        else:
            labels[i] = 0  # HOLD
    
    return labels


def create_triple_barrier_labels(df, profit=0.05, stop=0.08, horizon_hours=24):
    """
    Triple barrier method (institutional)
    
    Look forward and see what hits first:
    - +5% profit target = WIN (label 1)
    - -8% stop loss = LOSS (label -1)
    - 24 hours timeout = based on final price (label 0 or 1)
    """
    close = df['close'].values
    labels = np.zeros(len(close))
    
    for i in range(len(close) - horizon_hours):
        future_prices = close[i+1:i+1+horizon_hours]
        entry_price = close[i]
        
        hit_profit = False
        hit_stop = False
        
        for future_price in future_prices:
            ret = (future_price - entry_price) / entry_price
            
            if ret >= profit:
                hit_profit = True
                break
            elif ret <= -stop:
                hit_stop = True
                break
        
        if hit_profit:
            labels[i] = 1  # WIN
        elif hit_stop:
            labels[i] = -1  # LOSS
        else:
            # Timeout - check final price
            final_ret = (future_prices[-1] - entry_price) / entry_price
            labels[i] = 1 if final_ret > 0 else 0
    
    return labels


def fetch_and_engineer_sample():
    """Fetch sample data and engineer features"""
    print("🔍 Fetching sample data (50 tickers × 1 year)...\n")
    
    all_data = []
    successes = 0
    failures = 0
    
    for ticker in SAMPLE_TICKERS[:50]:  # Limit to 50 for speed
        try:
            features = FeatureEngineer70.download_and_engineer(ticker, period='1y')
            
            if features is not None and len(features) >= 100:
                features['ticker'] = ticker
                all_data.append(features)
                successes += 1
                print(f"   ✅ {ticker}: {len(features)} rows")
            else:
                failures += 1
                print(f"   ❌ {ticker}: Insufficient data")
        except Exception as e:
            failures += 1
            print(f"   ❌ {ticker}: {str(e)[:50]}")
    
    print(f"\n📊 Collection Summary:")
    print(f"   Successes: {successes}")
    print(f"   Failures: {failures}")
    
    if not all_data:
        raise ValueError("No data collected!")
    
    # Combine all tickers
    df_combined = pd.concat(all_data, ignore_index=True)
    print(f"   Total rows: {len(df_combined)}")
    print(f"   Features: {len(df_combined.columns) - 1}")  # -1 for ticker
    
    return df_combined


def test_labeling_strategy(df, strategy_name, label_func, **kwargs):
    """Test a labeling strategy"""
    print(f"\n{'='*60}")
    print(f"🧪 TESTING: {strategy_name}")
    print(f"{'='*60}")
    
    # Create labels
    labels = label_func(df, **kwargs)
    df_labeled = df.copy()
    df_labeled['label'] = labels
    
    # Remove rows without labels (end of series)
    df_labeled = df_labeled[df_labeled['label'] != 0].copy()
    
    if len(df_labeled) < 100:
        print(f"❌ Not enough labeled data: {len(df_labeled)} rows")
        return None
    
    # Label distribution
    unique, counts = np.unique(labels[labels != 0], return_counts=True)
    print(f"\n📊 Label Distribution:")
    for val, count in zip(unique, counts):
        pct = count / len(labels[labels != 0]) * 100
        label_name = {1: 'BUY', -1: 'SELL', 0: 'HOLD'}.get(int(val), 'UNK')
        print(f"   {label_name:4s} ({int(val):2d}): {count:5d} ({pct:5.1f}%)")
    
    # Prepare features
    feature_cols = [col for col in df_labeled.columns if col not in ['ticker', 'label']]
    X = df_labeled[feature_cols].values
    y = df_labeled['label'].values
    
    # Replace any remaining NaNs/infs
    X = np.nan_to_num(X, nan=0.0, posinf=1e9, neginf=-1e9)
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"\n🔧 Training Quick Random Forest...")
    print(f"   Train samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    
    # Quick RF (not optimized, just baseline)
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    
    clf.fit(X_train, y_train)
    
    # Predictions
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)
    
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    
    print(f"\n📈 RESULTS:")
    print(f"   Train Accuracy: {train_acc:.1%}")
    print(f"   Test Accuracy:  {test_acc:.1%}")
    print(f"   Overfit Gap:    {(train_acc - test_acc):.1%}")
    
    # Feature importance (top 10)
    importances = clf.feature_importances_
    feature_names = [feature_cols[i] for i in range(len(feature_cols))]
    top_indices = np.argsort(importances)[-10:][::-1]
    
    print(f"\n🎯 Top 10 Features:")
    for idx in top_indices:
        print(f"   {feature_names[idx]:25s}: {importances[idx]:.4f}")
    
    # Check if institutional features are in top 10
    institutional_features = [
        'liquidity_impact', 'vol_accel', 'smart_money_score', 'wick_ratio', 'mom_accel',
        'fractal_efficiency', 'price_efficiency', 'rel_volume_50', 'trend_consistency'
    ]
    
    top_features = [feature_names[i] for i in top_indices]
    institutional_in_top = [f for f in top_features if f in institutional_features]
    
    print(f"\n💎 Institutional Features in Top 10: {len(institutional_in_top)}/10")
    for feat in institutional_in_top:
        print(f"      • {feat}")
    
    return {
        'strategy': strategy_name,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'samples': len(df_labeled),
        'institutional_in_top10': len(institutional_in_top),
        'top_features': top_features[:5]
    }


def main():
    """Run quick baseline validation"""
    print("="*60)
    print("🎯 QUICK BASELINE VALIDATOR - 71 FEATURES")
    print("="*60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Fetch data
    df = fetch_and_engineer_sample()
    
    # Test different labeling strategies
    results = []
    
    # Strategy 1: Simple 5%/-8% over 7 days
    result = test_labeling_strategy(
        df,
        "Simple 7-Day (5%/-8%)",
        create_simple_labels,
        profit_target=0.05,
        stop_loss=-0.08,
        horizon_days=7
    )
    if result:
        results.append(result)
    
    # Strategy 2: Aggressive 10%/-5% over 3 days
    result = test_labeling_strategy(
        df,
        "Aggressive 3-Day (10%/-5%)",
        create_simple_labels,
        profit_target=0.10,
        stop_loss=-0.05,
        horizon_days=3
    )
    if result:
        results.append(result)
    
    # Strategy 3: Conservative 3%/-10% over 14 days
    result = test_labeling_strategy(
        df,
        "Conservative 14-Day (3%/-10%)",
        create_simple_labels,
        profit_target=0.03,
        stop_loss=-0.10,
        horizon_days=14
    )
    if result:
        results.append(result)
    
    # Strategy 4: Triple Barrier (institutional)
    result = test_labeling_strategy(
        df,
        "Triple Barrier (5%/-8% @ 24h)",
        create_triple_barrier_labels,
        profit=0.05,
        stop=0.08,
        horizon_hours=24
    )
    if result:
        results.append(result)
    
    # Final summary
    print("\n" + "="*60)
    print("📊 FINAL SUMMARY - BASELINE COMPARISON")
    print("="*60)
    
    if results:
        summary_df = pd.DataFrame(results)
        print(f"\n{summary_df.to_string(index=False)}\n")
        
        best = summary_df.loc[summary_df['test_acc'].idxmax()]
        print(f"🏆 BEST STRATEGY: {best['strategy']}")
        print(f"   Test Accuracy: {best['test_acc']:.1%}")
        print(f"   Institutional Features in Top 10: {best['institutional_in_top10']}")
        print(f"   Top 5 Features: {', '.join(best['top_features'])}")
        
        if best['test_acc'] >= 0.75:
            print(f"\n✅ BASELINE TARGET MET! ({best['test_acc']:.1%} ≥ 75%)")
            print(f"🚀 READY FOR A100 GPU TRAINING!")
        else:
            print(f"\n⚠️  BASELINE BELOW TARGET ({best['test_acc']:.1%} < 75%)")
            print(f"💡 Consider adjusting labeling parameters or adding more features")
    
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
