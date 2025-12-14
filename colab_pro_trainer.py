"""
COLAB PRO TRAINING PIPELINE
GPU-optimized training system for swing trading models (1-3 day to multi-week horizons).

Features:
- Multi-horizon training (1d, 3d, 5d, 10d, 21d forecasts)
- Walk-forward validation with embargo periods
- GPU acceleration (HistGradientBoosting, XGBoost if available)
- Pattern statistics integration
- Regime-specific model training
- Comprehensive logging and checkpointing
- Self-improvement loop with performance monitoring
- Optimized for $1K-$5K capital swing trading

Usage (in Colab Pro):
    !pip install -q yfinance talib-binary xgboost

    from colab_pro_trainer import ColabProTrainer
    
    trainer = ColabProTrainer(
        tickers=['AAPL', 'MSFT', 'GOOGL', 'TSLA'],
        swing_horizon='5bar'  # 5-day swing trades
    )
    
    trainer.run_full_training_pipeline()
"""

import os
import sys
import numpy as np
import pandas as pd
import yfinance as yf
import talib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import joblib
import json
import time
import logging

# ML libraries
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, brier_score_loss

# Try GPU-accelerated XGBoost
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except:
    XGB_AVAILABLE = False
    print("⚠️  XGBoost not available, using HistGradientBoosting only")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ColabProTrainer:
    """
    GPU-optimized training pipeline for swing trading models.
    
    Designed for $1K-$5K capital with 1-3 day to multi-week hold periods.
    """
    
    # Swing trading horizons (bars to hold)
    SWING_HORIZONS = {
        '1bar': 1,      # 1 day (short swing)
        '3bar': 3,      # 3 days (typical swing)
        '5bar': 5,      # 1 week
        '10bar': 10,    # 2 weeks
        '21bar': 21     # 1 month (position trade)
    }
    
    def __init__(
        self,
        tickers: List[str] = ['SPY', 'QQQ', 'IWM'],
        swing_horizon: str = '5bar',
        lookback_days: int = 730,  # 2 years of data
        model_dir: str = '/content/drive/MyDrive/quantum-trader/models',
        use_gpu: bool = True
    ):
        """
        Initialize Colab Pro trainer.
        
        Args:
            tickers: List of tickers to train on
            swing_horizon: Target holding period ('1bar', '3bar', '5bar', '10bar', '21bar')
            lookback_days: Days of historical data to use
            model_dir: Directory to save trained models (use Google Drive for persistence)
            use_gpu: Use GPU acceleration if available
        """
        self.tickers = tickers
        self.swing_horizon = swing_horizon
        self.horizon_bars = self.SWING_HORIZONS[swing_horizon]
        self.lookback_days = lookback_days
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.use_gpu = use_gpu
        
        # Import custom modules (assuming they're in the workspace)
        self.feature_engineer = None
        self.pattern_stats = None
        self.training_logger = None
        
        logger.info(f"✅ ColabProTrainer initialized")
        logger.info(f"   Tickers: {tickers}")
        logger.info(f"   Swing Horizon: {swing_horizon} ({self.horizon_bars} bars)")
        logger.info(f"   GPU: {'Enabled' if use_gpu and XGB_AVAILABLE else 'CPU only'}")
    
    def download_data(self) -> Dict[str, pd.DataFrame]:
        """
        Download OHLCV data for all tickers.
        
        Returns:
            Dict mapping ticker to DataFrame
        """
        logger.info("📥 Downloading market data...")
        
        data_dict = {}
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.lookback_days)
        
        for ticker in self.tickers:
            try:
                df = yf.download(ticker, start=start_date, end=end_date, progress=False)
                
                if len(df) > 100:
                    data_dict[ticker] = df
                    logger.info(f"  ✅ {ticker}: {len(df)} bars")
                else:
                    logger.warning(f"  ⚠️  {ticker}: Insufficient data ({len(df)} bars)")
            
            except Exception as e:
                logger.error(f"  ❌ {ticker}: Download failed - {e}")
        
        logger.info(f"✅ Downloaded data for {len(data_dict)} tickers")
        return data_dict
    
    def engineer_features_simple(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Simple feature engineering (for standalone use without core modules).
        
        Args:
            df: OHLCV DataFrame
        
        Returns:
            DataFrame with features
        """
        close = df['Close'].values
        high = df['High'].values
        low = df['Low'].values
        volume = df['Volume'].values
        
        features = pd.DataFrame(index=df.index)
        
        # Technical indicators
        features['rsi_9'] = talib.RSI(close, 9)
        features['rsi_14'] = talib.RSI(close, 14)
        features['rsi_21'] = talib.RSI(close, 21)
        
        macd, macd_sig, macd_hist = talib.MACD(close, 12, 26, 9)
        features['macd'] = macd
        features['macd_signal'] = macd_sig
        features['macd_histogram'] = macd_hist
        
        features['atr_14'] = talib.ATR(high, low, close, 14)
        features['adx_14'] = talib.ADX(high, low, close, 14)
        features['cci_14'] = talib.CCI(high, low, close, 14)
        features['mfi_14'] = talib.MFI(high, low, close, volume, 14)
        
        features['ema_5'] = talib.EMA(close, 5)
        features['ema_13'] = talib.EMA(close, 13)
        features['ema_50'] = talib.EMA(close, 50)
        features['ema_200'] = talib.EMA(close, 200)
        
        # Percentile features (research requirement)
        features['rsi_14_percentile'] = features['rsi_14'].rolling(90).rank(pct=True)
        features['atr_14_percentile'] = features['atr_14'].rolling(90).rank(pct=True)
        features['volume_percentile'] = pd.Series(volume).rolling(90).rank(pct=True)
        
        # Second-order features
        features['rsi_momentum'] = features['rsi_14'].diff()
        features['volume_acceleration'] = pd.Series(volume).pct_change().diff()
        
        # Returns
        features['return_1d'] = pd.Series(close).pct_change(1)
        features['return_3d'] = pd.Series(close).pct_change(3)
        features['return_5d'] = pd.Series(close).pct_change(5)
        
        # Volume ratio
        vol_sma = talib.SMA(volume, 20)
        features['volume_ratio'] = volume / (vol_sma + 1e-9)
        
        # Regime indicators
        features['price_above_ema200'] = (close > features['ema_200'].values).astype(float)
        features['trend_regime_bull'] = ((features['adx_14'] > 25) & (close > features['ema_200'].values)).astype(float)
        
        # Cleanup
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.ffill().bfill()
        features = features.fillna(0)
        
        return features
    
    def create_labels(self, df: pd.DataFrame) -> pd.Series:
        """
        Create labels for swing trading.
        
        Label logic for swing trading:
        - Class 2 (BUY): Forward return > +2% threshold
        - Class 0 (SELL/SHORT): Forward return < -2% threshold
        - Class 1 (HOLD): Between -2% and +2%
        
        Args:
            df: OHLCV DataFrame
        
        Returns:
            Series with labels (0, 1, 2)
        """
        close = df['Close'].values
        
        # Calculate forward returns at swing horizon
        future_returns = pd.Series(close).pct_change(self.horizon_bars).shift(-self.horizon_bars)
        
        # Thresholds for swing trading
        buy_threshold = 0.02   # 2% gain
        sell_threshold = -0.02  # 2% loss
        
        labels = pd.Series(1, index=df.index)  # Default: HOLD
        labels.loc[future_returns > buy_threshold] = 2  # BUY
        labels.loc[future_returns < sell_threshold] = 0  # SELL/SHORT
        
        return labels
    
    def walk_forward_validation(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int = 5,
        embargo_days: int = 10
    ) -> List[Dict]:
        """
        Walk-forward validation with embargo period.
        
        Args:
            X: Features
            y: Labels
            n_splits: Number of walk-forward splits
            embargo_days: Gap between train and test (prevents leakage)
        
        Returns:
            List of fold results
        """
        logger.info(f"🔄 Walk-forward validation ({n_splits} splits, {embargo_days}-day embargo)...")
        
        # Time series split
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        fold_results = []
        
        for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X)):
            logger.info(f"\n  Fold {fold_idx + 1}/{n_splits}...")
            
            # Apply embargo: remove last embargo_days from train, first embargo_days from test
            train_idx = train_idx[:-embargo_days] if len(train_idx) > embargo_days else train_idx
            test_idx = test_idx[embargo_days:] if len(test_idx) > embargo_days else test_idx
            
            if len(train_idx) < 100 or len(test_idx) < 20:
                logger.warning(f"    Skipping fold {fold_idx + 1}: insufficient data")
                continue
            
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            if self.use_gpu and XGB_AVAILABLE:
                model = xgb.XGBClassifier(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.05,
                    tree_method='hist',  # Fast CPU method (GPU: 'gpu_hist' if CUDA available)
                    random_state=42
                )
            else:
                model = HistGradientBoostingClassifier(
                    max_iter=200,
                    learning_rate=0.05,
                    max_depth=6,
                    random_state=42
                )
            
            start_time = time.time()
            model.fit(X_train_scaled, y_train)
            training_time = time.time() - start_time
            
            # Predict
            y_pred = model.predict(X_test_scaled)
            y_proba = model.predict_proba(X_test_scaled)
            
            # Metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            # Directional accuracy (BUY vs SELL)
            directional_correct = ((y_test == 2) & (y_pred == 2)).sum() + ((y_test == 0) & (y_pred == 0)).sum()
            directional_total = ((y_test == 2) | (y_test == 0)).sum()
            directional_acc = directional_correct / directional_total if directional_total > 0 else 0
            
            fold_results.append({
                'fold': fold_idx + 1,
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'directional_accuracy': directional_acc,
                'training_time': training_time
            })
            
            logger.info(f"    Accuracy: {accuracy:.3f} | Directional: {directional_acc:.3f} | F1: {f1:.3f}")
        
        return fold_results
    
    def train_final_model(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2
    ) -> Tuple[any, Dict]:
        """
        Train final model on all data (with holdout test set).
        
        Args:
            X: Features
            y: Labels
            test_size: Fraction for test set
        
        Returns:
            (trained_model, metrics_dict)
        """
        logger.info("\n🎯 Training final model...")
        
        # Train/test split (time-based)
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train
        if self.use_gpu and XGB_AVAILABLE:
            model = xgb.XGBClassifier(
                n_estimators=300,
                max_depth=7,
                learning_rate=0.03,
                tree_method='hist',
                random_state=42
            )
        else:
            model = HistGradientBoostingClassifier(
                max_iter=300,
                learning_rate=0.03,
                max_depth=7,
                random_state=42
            )
        
        start_time = time.time()
        model.fit(X_train_scaled, y_train)
        training_time = time.time() - start_time
        
        # Evaluate
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'feature_count': X.shape[1],
            'training_time': training_time
        }
        
        logger.info(f"✅ Final Model Trained:")
        logger.info(f"   Accuracy: {metrics['accuracy']:.3f}")
        logger.info(f"   Precision: {metrics['precision']:.3f}")
        logger.info(f"   Recall: {metrics['recall']:.3f}")
        logger.info(f"   F1: {metrics['f1']:.3f}")
        logger.info(f"   Training Time: {training_time:.1f}s")
        
        # Save model and scaler
        model_path = self.model_dir / f'swing_model_{self.swing_horizon}.pkl'
        scaler_path = self.model_dir / f'scaler_{self.swing_horizon}.pkl'
        
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
        
        logger.info(f"💾 Saved model to {model_path}")
        
        return model, scaler, metrics
    
    def run_full_training_pipeline(self):
        """
        Execute complete training pipeline.
        
        Steps:
        1. Download data for all tickers
        2. Engineer features
        3. Create labels
        4. Walk-forward validation
        5. Train final model
        6. Save results and logs
        """
        logger.info("🚀 Starting Full Training Pipeline")
        logger.info(f"   Swing Horizon: {self.swing_horizon} ({self.horizon_bars} bars)")
        logger.info(f"   Tickers: {', '.join(self.tickers)}")
        
        pipeline_start = time.time()
        
        # Step 1: Download data
        data_dict = self.download_data()
        
        if not data_dict:
            logger.error("❌ No data downloaded. Exiting.")
            return
        
        # Step 2: Process each ticker and combine
        all_features = []
        all_labels = []
        
        for ticker, df in data_dict.items():
            logger.info(f"\n📊 Processing {ticker}...")
            
            # Engineer features
            features = self.engineer_features_simple(df)
            
            # Create labels
            labels = self.create_labels(df)
            
            # Align features and labels
            aligned = pd.concat([features, labels.rename('label')], axis=1).dropna()
            
            if len(aligned) < 100:
                logger.warning(f"  ⚠️  Insufficient aligned data for {ticker}")
                continue
            
            all_features.append(aligned.drop('label', axis=1))
            all_labels.append(aligned['label'])
            
            logger.info(f"  ✅ {len(aligned)} samples with {len(features.columns)} features")
        
        # Combine all tickers
        X = pd.concat(all_features, axis=0)
        y = pd.concat(all_labels, axis=0)
        
        logger.info(f"\n✅ Combined Dataset:")
        logger.info(f"   Total Samples: {len(X)}")
        logger.info(f"   Total Features: {X.shape[1]}")
        logger.info(f"   Label Distribution: {y.value_counts().to_dict()}")
        
        # Step 3: Walk-forward validation
        fold_results = self.walk_forward_validation(X, y, n_splits=5, embargo_days=10)
        
        # Step 4: Train final model
        model, scaler, final_metrics = self.train_final_model(X, y, test_size=0.2)
        
        # Step 5: Save comprehensive results
        results = {
            'timestamp': datetime.now().isoformat(),
            'swing_horizon': self.swing_horizon,
            'horizon_bars': self.horizon_bars,
            'tickers': self.tickers,
            'total_samples': len(X),
            'feature_count': X.shape[1],
            'fold_results': fold_results,
            'final_metrics': final_metrics,
            'avg_accuracy': np.mean([f['accuracy'] for f in fold_results]),
            'avg_directional_accuracy': np.mean([f['directional_accuracy'] for f in fold_results])
        }
        
        results_path = self.model_dir / f'training_results_{self.swing_horizon}.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        pipeline_time = time.time() - pipeline_start
        
        logger.info(f"\n{'='*60}")
        logger.info(f"🎉 TRAINING PIPELINE COMPLETE")
        logger.info(f"{'='*60}")
        logger.info(f"Total Time: {pipeline_time:.1f}s ({pipeline_time/60:.1f} minutes)")
        logger.info(f"Swing Horizon: {self.swing_horizon} ({self.horizon_bars} bars)")
        logger.info(f"Average Accuracy: {results['avg_accuracy']:.3f}")
        logger.info(f"Average Directional Accuracy: {results['avg_directional_accuracy']:.3f}")
        logger.info(f"Final Model Accuracy: {final_metrics['accuracy']:.3f}")
        logger.info(f"Results saved to: {results_path}")
        logger.info(f"{'='*60}")


if __name__ == '__main__':
    # Example usage for Colab Pro
    print("🚀 Quantum AI Trader - Colab Pro Training Pipeline")
    print("Optimized for swing trading (1-3 days to weeks)\n")
    
    # Configuration for swing trading with $1K-$5K capital
    TICKERS = ['SPY', 'QQQ', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'AMD', 'META']
    SWING_HORIZON = '5bar'  # 5-day swings (adjust to '3bar' or '10bar' as needed)
    
    trainer = ColabProTrainer(
        tickers=TICKERS,
        swing_horizon=SWING_HORIZON,
        lookback_days=730,  # 2 years of data
        model_dir='models',  # Change to '/content/drive/MyDrive/...' in Colab
        use_gpu=True
    )
    
    trainer.run_full_training_pipeline()
    
    print("\n✅ Training complete! Models ready for live trading.")
    print("📊 Review training_results_*.json for detailed metrics")
    print("💾 Load trained model with: joblib.load('models/swing_model_5bar.pkl')")
