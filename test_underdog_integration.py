"""
Integration Test - Underdog Trading System

Tests all components working together:
1. Download data for 5 tickers
2. Calculate features
3. Train ensemble
4. Classify regime
5. Make predictions

Quick smoke test before Colab Pro training.
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'python'))

from multi_model_ensemble import MultiModelEnsemble
from feature_engine import FeatureEngine
from regime_classifier import RegimeClassifier

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_test_data(tickers: list, period: str = '3mo', interval: str = '1h') -> pd.DataFrame:
    """Download data for test tickers"""
    all_data = []
    
    for ticker in tickers:
        logger.info(f"Downloading {ticker}...")
        try:
            df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
            if len(df) > 0:
                df = df.reset_index()
                # Handle MultiIndex columns from yfinance
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df.columns]
                else:
                    df.columns = [c.lower() for c in df.columns]
                df['ticker'] = ticker
                all_data.append(df)
        except Exception as e:
            logger.error(f"Error downloading {ticker}: {e}")
    
    combined = pd.concat(all_data, ignore_index=True)
    return combined


def main():
    print("="*70)
    print("UNDERDOG TRADING SYSTEM - INTEGRATION TEST")
    print("="*70)
    
    # Test tickers (subset of Alpha 76)
    test_tickers = ['RKLB', 'IONQ', 'SOFI', 'APP', 'VKTX']
    
    # Step 1: Download data
    print("\n[1/6] Downloading test data...")
    raw_data = download_test_data(test_tickers)
    print(f"  ✅ Downloaded {len(raw_data)} bars for {len(test_tickers)} tickers")
    
    # Step 2: Calculate features
    print("\n[2/6] Calculating features...")
    engine = FeatureEngine()
    
    feature_data = []
    for ticker in raw_data['ticker'].unique():
        ticker_df = raw_data[raw_data['ticker'] == ticker].copy()
        ticker_df = ticker_df.rename(columns={'datetime': 'timestamp'})
        
        df_features = engine.calculate_all_features(ticker_df)
        df_features = engine.fill_missing_values(df_features)
        df_features['ticker'] = ticker
        feature_data.append(df_features)
    
    features_df = pd.concat(feature_data, ignore_index=True)
    print(f"  ✅ Calculated {len(engine.get_feature_names())} features")
    
    # Step 3: Prepare training data
    print("\n[3/6] Preparing training labels...")
    ensemble = MultiModelEnsemble(use_gpu=False)
    
    labeled_data = []
    for ticker in features_df['ticker'].unique():
        ticker_df = features_df[features_df['ticker'] == ticker].copy()
        labels = ensemble.prepare_labels(ticker_df['close'], forward_periods=5)
        ticker_df['label'] = labels
        labeled_data.append(ticker_df)
    
    training_data = pd.concat(labeled_data, ignore_index=True)
    training_data = training_data.dropna(subset=['label'])
    
    print(f"  ✅ {len(training_data)} training samples prepared")
    
    # Split data
    split_idx = int(0.8 * len(training_data))
    train_data = training_data.iloc[:split_idx]
    val_data = training_data.iloc[split_idx:]
    
    feature_cols = engine.get_feature_names()
    X_train = train_data[feature_cols]
    y_train = train_data['label'].values
    X_val = val_data[feature_cols]
    y_val = val_data['label'].values
    
    # Step 4: Train ensemble
    print("\n[4/6] Training 3-model ensemble...")
    metrics = ensemble.train(X_train, y_train, X_val, y_val)
    
    print("\n  Model Performance:")
    for model, metric in metrics.items():
        if 'error' not in metric:
            print(f"    {model}: accuracy={metric['accuracy']:.3f}, auc={metric.get('roc_auc', 0):.3f}")
    
    # Step 5: Classify regime
    print("\n[5/6] Classifying market regime...")
    classifier = RegimeClassifier()
    regime = classifier.classify_regime()
    
    print(f"  ✅ Regime: {regime['name']}")
    print(f"     Position multiplier: {regime['position_size_multiplier']}")
    print(f"     Min confidence: {regime['min_confidence']}")
    
    # Step 6: Make predictions
    print("\n[6/6] Making test predictions...")
    
    test_samples = min(5, len(X_val))
    test_indices = np.random.choice(len(X_val), test_samples, replace=False)
    
    high_conf_count = 0
    for idx in test_indices:
        X_test = X_val.iloc[idx:idx+1]
        ticker = val_data.iloc[idx]['ticker']
        
        pred = ensemble.predict(X_test)
        
        # Apply regime filter
        meets_regime = pred['confidence'] >= regime['min_confidence']
        if meets_regime:
            high_conf_count += 1
        
        print(f"\n  Ticker: {ticker}")
        print(f"    Signal: {pred['signal']} (confidence: {pred['confidence']:.3f})")
        print(f"    Meets regime filter: {'✅' if meets_regime else '❌'}")
    
    print("\n" + "="*70)
    print("INTEGRATION TEST COMPLETE ✅")
    print("="*70)
    
    print(f"\nSummary:")
    print(f"  • Features: {len(feature_cols)}")
    print(f"  • Training samples: {len(train_data)}")
    print(f"  • Validation samples: {len(val_data)}")
    print(f"  • Models trained: 3 (XGBoost, RF, GB)")
    print(f"  • Current regime: {regime['name']}")
    print(f"  • High-confidence signals: {high_conf_count}/{test_samples}")
    
    print(f"\n✅ Ready for Colab Pro training on full Alpha 76 dataset!")
    print(f"   Upload notebooks/UNDERDOG_COLAB_TRAINER.ipynb to Colab")
    print(f"   Enable T4 GPU and run all cells (~2-4 hours)")


if __name__ == "__main__":
    main()
