"""
📊 BUILD TRAINING DATASET
Create high-quality training dataset with 56 features + labels
for Trident ensemble training

OUTPUT: training_dataset.csv with columns:
- 56 features (from feature_engineer_56.py)
- ticker: stock symbol
- date: timestamp
- label: BUY (1) / HOLD (0) / SELL (-1)
- forward_return_7d: actual 7-day return (for verification)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import sys
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ml.feature_engineer_56 import FeatureEngineer56
from config.legendary_tickers import get_legendary_tickers

# Label thresholds (evolved from gold findings)
LABEL_THRESHOLDS = {
    'buy_threshold': 0.05,   # 5% gain in 7 days = BUY
    'sell_threshold': -0.03,  # -3% loss in 7 days = SELL
    'forward_days': 7,        # Predict 7-day forward returns
}


def create_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create quality labels based on forward returns
    
    Args:
        df: DataFrame with 'close' column
        
    Returns:
        DataFrame with 'label' and 'forward_return_7d' columns
    """
    # Calculate forward returns
    df['forward_return_7d'] = df['close'].pct_change(LABEL_THRESHOLDS['forward_days']).shift(-LABEL_THRESHOLDS['forward_days'])
    
    # Create labels
    df['label'] = 0  # Default: HOLD
    df.loc[df['forward_return_7d'] >= LABEL_THRESHOLDS['buy_threshold'], 'label'] = 1  # BUY
    df.loc[df['forward_return_7d'] <= LABEL_THRESHOLDS['sell_threshold'], 'label'] = -1  # SELL
    
    return df


def build_training_dataset(tickers: list, output_path: str = 'data/training_dataset.csv'):
    """
    Build complete training dataset for all tickers
    
    Args:
        tickers: List of stock tickers
        output_path: Path to save CSV
        
    Returns:
        DataFrame with all features, labels, metadata
    """
    print("🏗️  BUILDING TRAINING DATASET")
    print(f"   Tickers: {len(tickers)}")
    print(f"   Features: 56")
    print(f"   Label thresholds: BUY >= {LABEL_THRESHOLDS['buy_threshold']*100}%, SELL <= {LABEL_THRESHOLDS['sell_threshold']*100}%")
    print(f"   Forward window: {LABEL_THRESHOLDS['forward_days']} days\n")
    
    all_data = []
    
    for i, ticker in enumerate(tickers, 1):
        try:
            print(f"[{i}/{len(tickers)}] Processing {ticker}...", end=' ')
            
            # Download and engineer features
            features = FeatureEngineer56.download_and_engineer(ticker, period='2y')
            
            if features is None or len(features) < 200:
                print("❌ Insufficient data")
                continue
            
            # Add ticker column
            features['ticker'] = ticker
            
            # Add date (already in index)
            features['date'] = features.index
            
            # Create labels
            features = create_labels(features)
            
            # Remove samples without labels (last 7 days)
            features = features.dropna(subset=['forward_return_7d'])
            
            # Count labels
            buy_count = (features['label'] == 1).sum()
            sell_count = (features['label'] == -1).sum()
            hold_count = (features['label'] == 0).sum()
            
            print(f"✅ {len(features)} samples (BUY: {buy_count}, HOLD: {hold_count}, SELL: {sell_count})")
            
            all_data.append(features)
            
        except Exception as e:
            print(f"❌ Error: {e}")
            continue
    
    if not all_data:
        raise ValueError("No data collected! Check tickers and data availability.")
    
    # Combine all data
    print(f"\n📊 Combining data from {len(all_data)} tickers...")
    dataset = pd.concat(all_data, ignore_index=True)
    
    # Reorder columns (features first, then metadata, then label)
    feature_cols = [col for col in dataset.columns if col not in ['ticker', 'date', 'label', 'forward_return_7d']]
    column_order = feature_cols + ['ticker', 'date', 'forward_return_7d', 'label']
    dataset = dataset[column_order]
    
    # Summary
    print(f"\n✅ DATASET SUMMARY:")
    print(f"   Total samples: {len(dataset):,}")
    print(f"   Features: {len(feature_cols)}")
    print(f"   Tickers: {dataset['ticker'].nunique()}")
    print(f"   Date range: {dataset['date'].min()} to {dataset['date'].max()}")
    print(f"\n   Label distribution:")
    print(f"     BUY (1):   {(dataset['label'] == 1).sum():,} ({(dataset['label'] == 1).sum() / len(dataset) * 100:.1f}%)")
    print(f"     HOLD (0):  {(dataset['label'] == 0).sum():,} ({(dataset['label'] == 0).sum() / len(dataset) * 100:.1f}%)")
    print(f"     SELL (-1): {(dataset['label'] == -1).sum():,} ({(dataset['label'] == -1).sum() / len(dataset) * 100:.1f}%)")
    
    # Save to CSV
    print(f"\n💾 Saving to {output_path}...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_path, index=False)
    
    # File size
    file_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"   ✅ Saved! ({file_size_mb:.1f} MB)")
    
    return dataset


if __name__ == "__main__":
    """Build training dataset"""
    
    # Use legendary tickers (76 total)
    tickers = get_legendary_tickers()
    
    # Build dataset
    dataset = build_training_dataset(tickers, output_path='data/training_dataset.csv')
    
    print(f"\n🎯 READY FOR TRAINING!")
    print(f"   Next step: Upload data/training_dataset.csv to Google Drive")
    print(f"   Then: Run COLAB_ULTIMATE_TRAINER.ipynb on Colab Pro GPU")
    print(f"\n🚀 LET'S GO!")
