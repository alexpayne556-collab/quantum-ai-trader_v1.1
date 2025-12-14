"""
📊 BUILD TRAINING DATASET (FAST VERSION)
Test with top 20 tickers first, then expand
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

# Suppress yfinance logging
import logging
logging.getLogger('yfinance').setLevel(logging.CRITICAL)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ml.feature_engineer_56 import FeatureEngineer56

# Top 20 high-quality tickers (known to have good data)
TOP_20_TICKERS = [
    'NVDA', 'AMD', 'TSLA', 'PLTR', 'HOOD', 'SOFI', 'COIN',
    'SNOW', 'CRWD', 'NET', 'DDOG', 'PANW', 'RKLB', 'IONQ',
    'ASTS', 'RIVN', 'MSTR', 'RIOT', 'SMCI', 'AVGO'
]

# Label thresholds
LABEL_THRESHOLDS = {
    'buy_threshold': 0.05,   # 5% gain = BUY
    'sell_threshold': -0.03,  # -3% loss = SELL
    'forward_days': 7,
}


def create_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Create labels from forward returns"""
    df['forward_return_7d'] = df['close'].pct_change(LABEL_THRESHOLDS['forward_days']).shift(-LABEL_THRESHOLDS['forward_days'])
    df['label'] = 0  # Default: HOLD
    df.loc[df['forward_return_7d'] >= LABEL_THRESHOLDS['buy_threshold'], 'label'] = 1  # BUY
    df.loc[df['forward_return_7d'] <= LABEL_THRESHOLDS['sell_threshold'], 'label'] = -1  # SELL
    return df


def build_training_dataset(tickers: list, output_path: str = 'data/training_dataset.csv'):
    """Build training dataset"""
    print("🏗️  BUILDING TRAINING DATASET")
    print(f"   Tickers: {len(tickers)}")
    print(f"   Features: 56")
    print(f"   Label thresholds: BUY >= +{LABEL_THRESHOLDS['buy_threshold']*100}%, SELL <= {LABEL_THRESHOLDS['sell_threshold']*100}%\n")
    
    all_data = []
    success_count = 0
    fail_count = 0
    
    for i, ticker in enumerate(tickers, 1):
        try:
            print(f"[{i}/{len(tickers)}] {ticker}...", end=' ', flush=True)
            
            # Download and engineer features
            features = FeatureEngineer56.download_and_engineer(ticker, period='2y')
            
            if features is None or len(features) < 200:
                print("❌ Insufficient data")
                fail_count += 1
                continue
            
            # Add metadata
            features['ticker'] = ticker
            features['date'] = features.index
            
            # Create labels
            features = create_labels(features)
            features = features.dropna(subset=['forward_return_7d'])
            
            # Count labels
            buy = (features['label'] == 1).sum()
            hold = (features['label'] == 0).sum()
            sell = (features['label'] == -1).sum()
            
            print(f"✅ {len(features)} samples (B:{buy} H:{hold} S:{sell})")
            
            all_data.append(features)
            success_count += 1
            
        except Exception as e:
            print(f"❌ Error: {str(e)[:50]}")
            fail_count += 1
            continue
    
    if not all_data:
        raise ValueError(f"No data collected! Succeeded: {success_count}, Failed: {fail_count}")
    
    # Combine
    print(f"\n📊 Combining {len(all_data)} tickers...")
    dataset = pd.concat(all_data, ignore_index=True)
    
    # Reorder columns
    feature_cols = [col for col in dataset.columns if col not in ['ticker', 'date', 'label', 'forward_return_7d']]
    column_order = feature_cols + ['ticker', 'date', 'forward_return_7d', 'label']
    dataset = dataset[column_order]
    
    # Summary
    total = len(dataset)
    buy_pct = (dataset['label'] == 1).sum() / total * 100
    hold_pct = (dataset['label'] == 0).sum() / total * 100
    sell_pct = (dataset['label'] == -1).sum() / total * 100
    
    print(f"\n✅ DATASET SUMMARY:")
    print(f"   Success: {success_count}/{len(tickers)} tickers")
    print(f"   Samples: {total:,}")
    print(f"   Features: {len(feature_cols)}")
    print(f"   Date range: {dataset['date'].min().date()} to {dataset['date'].max().date()}")
    print(f"\n   Labels:")
    print(f"     BUY:  {(dataset['label'] == 1).sum():,} ({buy_pct:.1f}%)")
    print(f"     HOLD: {(dataset['label'] == 0).sum():,} ({hold_pct:.1f}%)")
    print(f"     SELL: {(dataset['label'] == -1).sum():,} ({sell_pct:.1f}%)")
    
    # Save
    print(f"\n💾 Saving to {output_path}...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_path, index=False)
    
    file_size = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"   ✅ Saved ({file_size:.1f} MB)")
    
    return dataset


if __name__ == "__main__":
    # Test with top 20 first
    dataset = build_training_dataset(TOP_20_TICKERS, output_path='data/training_dataset.csv')
    
    print(f"\n🎯 READY FOR TRAINING!")
    print(f"   Upload: data/training_dataset.csv → Google Drive")
    print(f"   Then: Run COLAB_ULTIMATE_TRAINER.ipynb")
    print(f"\n🚀 LET'S GO!")
