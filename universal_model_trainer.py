"""
UniversalTrader: Train single model on 30 assets
Pattern recognition across multiple tickers with walk-forward validation

Optimized for Colab T4 High-RAM (15GB+)
Trains universal model that works across all assets
"""

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from typing import Dict, List, Tuple, Optional
import warnings
import joblib
from datetime import datetime

warnings.filterwarnings('ignore')

# Try LightGBM, fallback to sklearn
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier
    HAS_LIGHTGBM = False
    print("⚠️ LightGBM not found, using sklearn GradientBoosting")

from ultimate_feature_engine import UltimateFeatureEngine


class UniversalTrader:
    """
    Universal trading model trained on multiple assets.
    
    Features:
    - Multi-asset training (30 tickers)
    - 50+ technical indicators
    - Walk-forward validation
    - Robust scaling for cross-asset compatibility
    - LightGBM for fast GPU training
    """
    
    def __init__(self, tickers: List[str], start_date: str = "2000-01-01", 
                 end_date: str = None, target_days: int = 5, 
                 target_threshold: float = 0.01):
        """
        Initialize UniversalTrader.
        
        Args:
            tickers: List of ticker symbols to train on
            start_date: Start date for historical data
            end_date: End date (default: today)
            target_days: Days to look ahead for return
            target_threshold: Minimum return to be classified as positive
        """
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date or datetime.now().strftime("%Y-%m-%d")
        self.target_days = target_days
        self.target_threshold = target_threshold
        
        self.model = None
        self.scalers: Dict[str, StandardScaler] = {}
        self.universal_scaler = RobustScaler()
        
        self.feature_names = []
        self.training_stats = {}
    
    def download_ticker_data(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        Download historical data for a single ticker.
        
        Args:
            ticker: Ticker symbol
            
        Returns:
            DataFrame with OHLCV data or None if failed
        """
        try:
            df = yf.download(
                ticker, 
                start=self.start_date, 
                end=self.end_date,
                progress=False,
                auto_adjust=True
            )
            
            if df.empty or len(df) < 252:  # Need at least 1 year
                print(f"  ⚠️ {ticker}: Insufficient data ({len(df)} rows)")
                return None
            
            # Handle multi-index columns
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            return df
            
        except Exception as e:
            print(f"  ❌ {ticker}: Download failed - {e}")
            return None
    
    def create_target(self, df: pd.DataFrame) -> pd.Series:
        """
        Create binary classification target.
        1 = price increases by threshold in target_days
        0 = otherwise
        """
        future_return = df['Close'].pct_change(self.target_days).shift(-self.target_days)
        target = (future_return > self.target_threshold).astype(int)
        return target
    
    def prepare_multi_asset_data(self, verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load and prepare data for all tickers.
        
        Returns:
            X_universal: Combined feature matrix
            y_universal: Combined target vector
        """
        all_X = []
        all_y = []
        all_tickers_processed = []
        
        print("\n" + "=" * 60)
        print("📊 LOADING MULTI-ASSET DATA")
        print("=" * 60)
        print(f"Tickers: {len(self.tickers)}")
        print(f"Period: {self.start_date} to {self.end_date}")
        print(f"Target: {self.target_days}-day return > {self.target_threshold:.1%}")
        print("=" * 60)
        
        for i, ticker in enumerate(self.tickers, 1):
            if verbose:
                print(f"\n[{i}/{len(self.tickers)}] Loading {ticker}...")
            
            # Download data
            df = self.download_ticker_data(ticker)
            if df is None:
                continue
            
            # Generate features
            try:
                engine = UltimateFeatureEngine(df)
                features = engine.compute_all_indicators()
                
                if features.empty or len(features) < 100:
                    print(f"  ⚠️ {ticker}: Not enough features ({len(features)} rows)")
                    continue
                
                # Store feature names (from first successful ticker)
                if not self.feature_names:
                    self.feature_names = list(features.columns)
                
                # Create target
                target = self.create_target(df)
                
                # Align features and target
                valid_idx = features.index.intersection(target.dropna().index)
                X = features.loc[valid_idx]
                y = target.loc[valid_idx]
                
                # Remove any remaining NaN
                valid_mask = ~(X.isna().any(axis=1) | y.isna())
                X = X[valid_mask]
                y = y[valid_mask]
                
                if len(X) < 100:
                    print(f"  ⚠️ {ticker}: Too few valid samples ({len(X)})")
                    continue
                
                # Create and fit scaler for this ticker
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                self.scalers[ticker] = scaler
                
                all_X.append(X_scaled)
                all_y.append(y.values)
                all_tickers_processed.append(ticker)
                
                if verbose:
                    pos_rate = y.mean() * 100
                    print(f"  ✓ {ticker}: {len(X):,} samples, {X.shape[1]} features, {pos_rate:.1f}% positive")
                
            except Exception as e:
                print(f"  ❌ {ticker}: Feature engineering failed - {e}")
                continue
        
        if len(all_X) == 0:
            raise ValueError("No valid data collected from any ticker!")
        
        # Combine all data
        X_universal = np.vstack(all_X)
        y_universal = np.hstack(all_y)
        
        # Fit universal scaler on combined data
        self.universal_scaler.fit(X_universal)
        X_universal = self.universal_scaler.transform(X_universal)
        
        # Store stats
        self.training_stats = {
            'tickers_processed': all_tickers_processed,
            'total_samples': len(X_universal),
            'total_features': X_universal.shape[1],
            'positive_rate': y_universal.mean() * 100,
            'tickers_count': len(all_tickers_processed)
        }
        
        print("\n" + "=" * 60)
        print("📈 DATA PREPARATION COMPLETE")
        print("=" * 60)
        print(f"✓ Tickers processed: {len(all_tickers_processed)}/{len(self.tickers)}")
        print(f"✓ Total samples: {X_universal.shape[0]:,}")
        print(f"✓ Total features: {X_universal.shape[1]}")
        print(f"✓ Positive rate: {y_universal.mean()*100:.1f}%")
        print("=" * 60)
        
        return X_universal, y_universal
    
    def train_universal_model(self, X: np.ndarray, y: np.ndarray, 
                               use_gpu: bool = True) -> object:
        """
        Train universal LightGBM model on combined data.
        
        Args:
            X: Feature matrix
            y: Target vector
            use_gpu: Whether to use GPU (for Colab T4)
            
        Returns:
            Trained model
        """
        print("\n" + "=" * 60)
        print("🤖 TRAINING UNIVERSAL MODEL")
        print("=" * 60)
        
        # Time-based split (80/20)
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        print(f"Training set: {len(X_train):,} samples")
        print(f"Test set: {len(X_test):,} samples")
        
        if HAS_LIGHTGBM:
            # LightGBM parameters optimized for T4 GPU
            params = {
                'n_estimators': 1000,
                'max_depth': 8,
                'learning_rate': 0.05,
                'num_leaves': 63,
                'min_child_samples': 100,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 0.1,
                'n_jobs': -1,
                'random_state': 42,
                'verbose': -1,
                'importance_type': 'gain'
            }
            
            # Add GPU params if available
            if use_gpu:
                try:
                    params['device'] = 'gpu'
                    params['gpu_platform_id'] = 0
                    params['gpu_device_id'] = 0
                except:
                    pass  # Fall back to CPU
            
            self.model = lgb.LGBMClassifier(**params)
            
            # Train with early stopping
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                callbacks=[
                    lgb.early_stopping(50, verbose=False),
                    lgb.log_evaluation(period=100)
                ]
            )
        else:
            # Fallback to sklearn
            self.model = GradientBoostingClassifier(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42,
                verbose=1
            )
            self.model.fit(X_train, y_train)
        
        # Evaluate
        train_acc = self.model.score(X_train, y_train)
        test_acc = self.model.score(X_test, y_test)
        
        # Get predictions for detailed metrics
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate additional metrics
        from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
        
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        try:
            auc = roc_auc_score(y_test, y_pred_proba)
        except:
            auc = 0.5
        
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        print("\n" + "=" * 60)
        print("📊 MODEL PERFORMANCE")
        print("=" * 60)
        print(f"✓ Train Accuracy: {train_acc:.4f}")
        print(f"✓ Test Accuracy:  {test_acc:.4f}")
        print(f"✓ AUC Score:      {auc:.4f}")
        print(f"✓ Precision:      {precision:.4f}")
        print(f"✓ Recall:         {recall:.4f}")
        print(f"✓ F1 Score:       {f1:.4f}")
        print("=" * 60)
        
        # Store metrics
        self.training_stats['train_accuracy'] = train_acc
        self.training_stats['test_accuracy'] = test_acc
        self.training_stats['auc'] = auc
        self.training_stats['precision'] = precision
        self.training_stats['recall'] = recall
        self.training_stats['f1'] = f1
        
        return self.model
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Get top N most important features."""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        if HAS_LIGHTGBM:
            importances = self.model.feature_importances_
        else:
            importances = self.model.feature_importances_
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return importance_df.head(top_n)
    
    def walk_forward_validation(self, X: np.ndarray, y: np.ndarray, 
                                 n_splits: int = 5) -> Dict:
        """
        Perform walk-forward validation (realistic for trading).
        
        Args:
            X: Feature matrix
            y: Target vector
            n_splits: Number of time-based splits
            
        Returns:
            Dictionary with validation results
        """
        print("\n" + "=" * 60)
        print("🔄 WALK-FORWARD VALIDATION")
        print("=" * 60)
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        results = {
            'fold_accuracy': [],
            'fold_auc': [],
            'fold_sharpe': []
        }
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Train fresh model for this fold
            if HAS_LIGHTGBM:
                model = lgb.LGBMClassifier(
                    n_estimators=500, max_depth=6, learning_rate=0.05,
                    num_leaves=31, n_jobs=-1, verbose=-1
                )
            else:
                model = GradientBoostingClassifier(
                    n_estimators=200, max_depth=5, learning_rate=0.05
                )
            
            model.fit(X_train, y_train)
            
            # Evaluate
            acc = model.score(X_test, y_test)
            y_proba = model.predict_proba(X_test)[:, 1]
            
            try:
                from sklearn.metrics import roc_auc_score
                auc = roc_auc_score(y_test, y_proba)
            except:
                auc = 0.5
            
            # Simulate trading returns
            signals = (y_proba > 0.5).astype(int)
            returns = np.where(signals == 1, 
                              np.where(y_test == 1, 0.02, -0.01),  # Simplified
                              0)
            sharpe = (returns.mean() / (returns.std() + 1e-8)) * np.sqrt(252)
            
            results['fold_accuracy'].append(acc)
            results['fold_auc'].append(auc)
            results['fold_sharpe'].append(sharpe)
            
            print(f"  Fold {fold}: Acc={acc:.4f}, AUC={auc:.4f}, Sharpe={sharpe:.2f}")
        
        # Summary
        results['mean_accuracy'] = np.mean(results['fold_accuracy'])
        results['mean_auc'] = np.mean(results['fold_auc'])
        results['mean_sharpe'] = np.mean(results['fold_sharpe'])
        
        print("\n" + "-" * 40)
        print(f"Average Accuracy: {results['mean_accuracy']:.4f}")
        print(f"Average AUC: {results['mean_auc']:.4f}")
        print(f"Average Sharpe: {results['mean_sharpe']:.2f}")
        print("=" * 60)
        
        return results
    
    def save_model(self, filepath: str = "universal_trader_model.pkl"):
        """Save trained model and scalers."""
        save_data = {
            'model': self.model,
            'scalers': self.scalers,
            'universal_scaler': self.universal_scaler,
            'feature_names': self.feature_names,
            'training_stats': self.training_stats,
            'tickers': self.tickers
        }
        joblib.dump(save_data, filepath)
        print(f"✓ Model saved to {filepath}")
    
    def load_model(self, filepath: str = "universal_trader_model.pkl"):
        """Load trained model and scalers."""
        save_data = joblib.load(filepath)
        self.model = save_data['model']
        self.scalers = save_data['scalers']
        self.universal_scaler = save_data['universal_scaler']
        self.feature_names = save_data['feature_names']
        self.training_stats = save_data['training_stats']
        self.tickers = save_data['tickers']
        print(f"✓ Model loaded from {filepath}")


# Quick test
if __name__ == "__main__":
    # Test with subset of tickers
    TEST_TICKERS = ['SPY', 'QQQ', 'AAPL', 'MSFT', 'GOOGL']
    
    print("🚀 UniversalTrader Test")
    print("=" * 60)
    
    trader = UniversalTrader(
        tickers=TEST_TICKERS,
        start_date="2020-01-01",
        target_days=5,
        target_threshold=0.01
    )
    
    # Prepare data
    X, y = trader.prepare_multi_asset_data()
    
    # Train model
    model = trader.train_universal_model(X, y, use_gpu=False)
    
    # Show feature importance
    print("\n📊 Top 10 Most Important Features:")
    importance = trader.get_feature_importance(10)
    print(importance.to_string(index=False))
    
    # Walk-forward validation
    results = trader.walk_forward_validation(X, y, n_splits=3)
    
    print("\n✅ UniversalTrader test complete!")
