"""
DATASET LOADER
==============
Load and prepare data from dataset_builder.py for Trident training.

Features:
- Load features (X), labels (y), tickers from dataset_builder output
- Compute ticker characteristics for clustering
- Handle data validation and preprocessing
- Split data for training/validation

Input: Output from dataset_builder.py (CSV or pickle)
Output: Ready-to-train data dictionary

Author: Quantum AI Trader
Date: December 10, 2025
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatasetLoader:
    """
    Load and prepare datasets for Trident ensemble training.
    """
    
    def __init__(self, data_dir: str = 'data/training'):
        """
        Args:
            data_dir: Directory containing training data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def load_from_csv(self, filepath: str) -> Dict:
        """
        Load dataset from CSV file.
        
        Expected CSV format:
        - Columns: ticker, date, feature_1, ..., feature_56, label
        - Labels: 0 (SELL/HOLD) or 1 (BUY)
        
        Args:
            filepath: Path to CSV file
            
        Returns:
            {
                'X': Features DataFrame (N × 56),
                'y': Labels Series (N,),
                'tickers': Ticker symbols Series (N,),
                'dates': Date Series (N,),
                'ticker_features': Ticker stats DataFrame (M × 5)
            }
        """
        logger.info(f"Loading dataset from {filepath}...")
        
        df = pd.read_csv(filepath, parse_dates=['date'] if 'date' in pd.read_csv(filepath, nrows=1).columns else False)
        
        logger.info(f"✅ Loaded {len(df)} samples")
        
        # Extract components
        tickers = df['ticker'] if 'ticker' in df.columns else pd.Series(['UNKNOWN'] * len(df))
        dates = df['date'] if 'date' in df.columns else pd.Series([pd.NaT] * len(df))
        
        # Features (all columns except ticker, date, label)
        feature_cols = [c for c in df.columns if c not in ['ticker', 'date', 'label', 'target']]
        X = df[feature_cols]
        
        # Labels
        if 'label' in df.columns:
            y = df['label']
        elif 'target' in df.columns:
            y = df['target']
        else:
            raise ValueError("No 'label' or 'target' column found in dataset")
        
        # Compute ticker features for clustering
        ticker_features = self.compute_ticker_features(df, tickers)
        
        logger.info(f"   Features shape: {X.shape}")
        logger.info(f"   Labels shape: {y.shape}")
        logger.info(f"   Unique tickers: {tickers.nunique()}")
        logger.info(f"   Label distribution: {dict(y.value_counts())}")
        
        return {
            'X': X,
            'y': y,
            'tickers': tickers,
            'dates': dates,
            'ticker_features': ticker_features
        }
    
    def compute_ticker_features(
        self,
        df: pd.DataFrame,
        tickers: pd.Series
    ) -> pd.DataFrame:
        """
        Compute ticker-level features for clustering.
        
        Features computed:
        1. volatility: 21-day std of returns
        2. avg_volume: Average daily volume
        3. avg_price: Average price
        4. price_range: Max - Min price (21 days)
        5. sector: Sector category (if available)
        
        Args:
            df: Full dataset
            tickers: Ticker symbols
            
        Returns:
            DataFrame with ticker stats (one row per ticker)
        """
        logger.info("Computing ticker features for clustering...")
        
        ticker_stats = []
        unique_tickers = tickers.unique()
        
        for ticker in unique_tickers:
            # Get data for this ticker
            ticker_mask = tickers == ticker
            ticker_data = df[ticker_mask]
            
            # Calculate features from existing data
            if 'close' in ticker_data.columns or 'Close' in ticker_data.columns:
                close_col = 'close' if 'close' in ticker_data.columns else 'Close'
                prices = ticker_data[close_col]
                
                # Volatility (std of returns)
                returns = prices.pct_change()
                volatility = returns.std() if len(returns) > 1 else 0.05
                
                # Price stats
                avg_price = prices.mean()
                price_range = (prices.max() - prices.min()) / avg_price if avg_price > 0 else 0
            else:
                # Estimate from feature columns if price not available
                volatility = 0.05  # Default medium volatility
                avg_price = 50.0   # Default price
                price_range = 0.3  # Default 30% range
            
            # Volume stats
            if 'volume' in ticker_data.columns or 'Volume' in ticker_data.columns:
                vol_col = 'volume' if 'volume' in ticker_data.columns else 'Volume'
                avg_volume = ticker_data[vol_col].mean()
            else:
                avg_volume = 1_000_000  # Default volume
            
            ticker_stats.append({
                'ticker': ticker,
                'volatility': volatility,
                'avg_volume': avg_volume,
                'avg_price': avg_price,
                'price_range': price_range,
                'sector': 'Unknown'  # Can be enhanced with sector data
            })
        
        ticker_features = pd.DataFrame(ticker_stats).set_index('ticker')
        
        logger.info(f"✅ Computed features for {len(ticker_features)} tickers")
        logger.info(f"   Avg volatility: {ticker_features['volatility'].mean():.3f}")
        logger.info(f"   Avg volume: {ticker_features['avg_volume'].mean():,.0f}")
        logger.info(f"   Avg price: ${ticker_features['avg_price'].mean():.2f}")
        
        return ticker_features
    
    def download_and_build_dataset(
        self,
        tickers: list,
        period: str = '2y',
        min_samples_per_ticker: int = 100
    ) -> Dict:
        """
        Download data and build training dataset from scratch.
        
        This is a simplified version - in production, you'd use dataset_builder.py
        
        Args:
            tickers: List of ticker symbols
            period: Historical period to download
            min_samples_per_ticker: Minimum samples required
            
        Returns:
            Dataset dictionary ready for training
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"BUILDING DATASET FROM SCRATCH")
        logger.info(f"{'='*60}")
        logger.info(f"Tickers: {len(tickers)}")
        logger.info(f"Period: {period}")
        
        all_data = []
        failed_tickers = []
        
        for ticker in tickers:
            try:
                logger.info(f"\n📥 Downloading {ticker}...")
                df = yf.download(ticker, period=period, progress=False)
                
                if len(df) < min_samples_per_ticker:
                    logger.warning(f"   ⚠️ Insufficient data: {len(df)} samples")
                    failed_tickers.append(ticker)
                    continue
                
                # Add basic features
                df['ticker'] = ticker
                df['returns_1d'] = df['Close'].pct_change()
                df['returns_5d'] = df['Close'].pct_change(5)
                df['returns_21d'] = df['Close'].pct_change(21)
                df['volume_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
                
                # Simple label: 1 if next 5-day return > 3%, else 0
                df['future_return_5d'] = df['Close'].pct_change(5).shift(-5)
                df['label'] = (df['future_return_5d'] > 0.03).astype(int)
                
                # Drop NaN
                df = df.dropna()
                
                if len(df) > 0:
                    all_data.append(df)
                    logger.info(f"   ✅ {len(df)} samples added")
                
            except Exception as e:
                logger.error(f"   ❌ Failed: {e}")
                failed_tickers.append(ticker)
        
        if not all_data:
            raise ValueError("No data collected from any ticker!")
        
        # Combine all data
        combined = pd.concat(all_data, ignore_index=True)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"DATASET SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"Total samples: {len(combined)}")
        logger.info(f"Successful tickers: {len(tickers) - len(failed_tickers)}")
        logger.info(f"Failed tickers: {len(failed_tickers)}")
        if failed_tickers:
            logger.info(f"   {', '.join(failed_tickers)}")
        
        # Prepare for training
        tickers_col = combined['ticker']
        dates_col = combined.index
        
        # Feature columns (simple version)
        feature_cols = ['returns_1d', 'returns_5d', 'returns_21d', 'volume_ratio']
        X = combined[feature_cols]
        y = combined['label']
        
        # Compute ticker features
        ticker_features = self.compute_ticker_features(combined, tickers_col)
        
        return {
            'X': X,
            'y': y,
            'tickers': tickers_col,
            'dates': dates_col,
            'ticker_features': ticker_features
        }
    
    def validate_dataset(self, dataset: Dict) -> Tuple[bool, str]:
        """
        Validate dataset before training.
        
        Checks:
        - No missing values in features
        - No infinite values
        - Labels are binary (0/1)
        - Sufficient samples per ticker
        - Class balance
        
        Args:
            dataset: Dataset dictionary
            
        Returns:
            (is_valid, error_message)
        """
        logger.info("\n🔍 Validating dataset...")
        
        X = dataset['X']
        y = dataset['y']
        tickers = dataset['tickers']
        
        # Check missing values
        if X.isnull().any().any():
            return False, "Features contain missing values"
        
        # Check infinite values
        if np.isinf(X.values).any():
            return False, "Features contain infinite values"
        
        # Check labels
        unique_labels = y.unique()
        if not set(unique_labels).issubset({0, 1}):
            return False, f"Labels must be 0 or 1, got: {unique_labels}"
        
        # Check class balance
        class_counts = y.value_counts()
        class_ratio = class_counts.min() / class_counts.max()
        if class_ratio < 0.1:
            logger.warning(f"⚠️ Severe class imbalance: {dict(class_counts)}")
            logger.warning(f"   Ratio: {class_ratio:.2%}")
        
        # Check samples per ticker
        samples_per_ticker = tickers.value_counts()
        min_samples = samples_per_ticker.min()
        if min_samples < 50:
            logger.warning(f"⚠️ Some tickers have < 50 samples (min: {min_samples})")
        
        logger.info("✅ Dataset validation passed")
        logger.info(f"   Samples: {len(X)}")
        logger.info(f"   Features: {X.shape[1]}")
        logger.info(f"   Tickers: {tickers.nunique()}")
        logger.info(f"   Class balance: {dict(class_counts)}")
        logger.info(f"   Min samples/ticker: {min_samples}")
        
        return True, "OK"
    
    def save_dataset(self, dataset: Dict, filepath: str):
        """Save dataset to file for later use."""
        output_path = Path(filepath)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Combine into single DataFrame
        df = dataset['X'].copy()
        df['ticker'] = dataset['tickers'].values
        df['label'] = dataset['y'].values
        
        if 'dates' in dataset and dataset['dates'] is not None:
            df['date'] = dataset['dates'].values
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        logger.info(f"✅ Dataset saved to {output_path}")
        
        # Save ticker features separately
        ticker_features_path = output_path.parent / 'ticker_features.csv'
        dataset['ticker_features'].to_csv(ticker_features_path)
        logger.info(f"✅ Ticker features saved to {ticker_features_path}")


def example_usage():
    """Example: Build dataset from legendary tickers."""
    logger.info("\n" + "="*60)
    logger.info("DATASET LOADER - Example Usage")
    logger.info("="*60 + "\n")
    
    # Load legendary tickers
    try:
        from config.legendary_tickers import get_legendary_tickers
        tickers = get_legendary_tickers()
        logger.info(f"✅ Loaded {len(tickers)} legendary tickers")
    except ImportError:
        logger.warning("⚠️ legendary_tickers not found, using sample tickers")
        tickers = ['NVDA', 'TSLA', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NFLX']
    
    # Initialize loader
    loader = DatasetLoader(data_dir='data/training')
    
    # Build dataset (simplified version - in production use dataset_builder.py)
    dataset = loader.download_and_build_dataset(
        tickers=tickers[:10],  # First 10 for demo
        period='1y',
        min_samples_per_ticker=100
    )
    
    # Validate
    is_valid, message = loader.validate_dataset(dataset)
    
    if is_valid:
        logger.info(f"\n✅ Dataset ready for training!")
        logger.info(f"   Samples: {len(dataset['X'])}")
        logger.info(f"   Features: {dataset['X'].shape[1]}")
        logger.info(f"   Tickers: {dataset['tickers'].nunique()}")
        
        # Save
        loader.save_dataset(dataset, 'data/training/demo_dataset.csv')
    else:
        logger.error(f"❌ Validation failed: {message}")


if __name__ == '__main__':
    example_usage()
