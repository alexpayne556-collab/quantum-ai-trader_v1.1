"""
Multi-Model Ensemble - Layer 1 of Underdog Trading System

3-model voting system for high-confidence predictions:
- XGBoost (GPU-accelerated on Colab Pro T4)
- Random Forest (CPU)
- Gradient Boosting (CPU)

Voting Logic:
- 3/3 agree = STRONG_BUY (confidence: 1.0)
- 2/3 agree = BUY (confidence: 0.67-0.90)
- 1/3 agree = WEAK/SKIP (confidence: <0.67)

Intelligence edge, not speed edge.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
import pickle
import logging
from pathlib import Path

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    logging.warning("XGBoost not installed - GPU training unavailable")

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

logger = logging.getLogger(__name__)


class MultiModelEnsemble:
    """
    3-Model Ensemble for Trade Signals
    
    Each model votes on direction (BUY/SELL/HOLD)
    Confidence = agreement level + probability std
    """
    
    def __init__(self, use_gpu: bool = False):
        """
        Initialize ensemble models
        
        Args:
            use_gpu: Use XGBoost GPU acceleration (requires cuda)
        """
        self.use_gpu = use_gpu and XGB_AVAILABLE
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_trained = False
        
        # Initialize models
        self._init_models()
        
    def _init_models(self):
        """Initialize the 3 models"""
        
        # Model 1: XGBoost (GPU if available)
        if XGB_AVAILABLE:
            xgb_params = {
                'objective': 'multi:softprob',
                'num_class': 3,  # BUY=2, HOLD=1, SELL=0
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 200,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'tree_method': 'gpu_hist' if self.use_gpu else 'hist',
                'random_state': 42,
                'eval_metric': 'mlogloss'
            }
            self.models['xgboost'] = xgb.XGBClassifier(**xgb_params)
            logger.info(f"XGBoost initialized (GPU: {self.use_gpu})")
        else:
            logger.warning("XGBoost unavailable - using 2-model ensemble")
            
        # Model 2: Random Forest (always CPU)
        rf_params = {
            'n_estimators': 200,
            'max_depth': 10,
            'min_samples_split': 20,
            'min_samples_leaf': 10,
            'max_features': 'sqrt',
            'random_state': 42,
            'n_jobs': -1  # Use all CPU cores
        }
        self.models['random_forest'] = RandomForestClassifier(**rf_params)
        logger.info("Random Forest initialized")
        
        # Model 3: Gradient Boosting (always CPU)
        gb_params = {
            'n_estimators': 200,
            'max_depth': 5,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'random_state': 42
        }
        self.models['gradient_boosting'] = GradientBoostingClassifier(**gb_params)
        logger.info("Gradient Boosting initialized")
        
    def prepare_labels(self, prices: pd.Series, forward_periods: int = 5) -> np.ndarray:
        """
        Create labels from future price movement
        
        Args:
            prices: Price series (close prices)
            forward_periods: Periods ahead to calculate return
            
        Returns:
            labels: 0=SELL, 1=HOLD, 2=BUY
        """
        # Calculate forward returns
        future_returns = prices.pct_change(forward_periods).shift(-forward_periods)
        
        # Classify into 3 categories
        labels = np.zeros(len(future_returns))
        
        # BUY if return > 2%
        labels[future_returns > 0.02] = 2
        
        # SELL if return < -2%
        labels[future_returns < -0.02] = 0
        
        # HOLD otherwise
        labels[(future_returns >= -0.02) & (future_returns <= 0.02)] = 1
        
        return labels
        
    def train(self, 
              X_train: pd.DataFrame, 
              y_train: np.ndarray,
              X_val: Optional[pd.DataFrame] = None,
              y_val: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Train all 3 models
        
        Args:
            X_train: Training features
            y_train: Training labels (0=SELL, 1=HOLD, 2=BUY)
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            
        Returns:
            metrics: Training metrics for each model
        """
        logger.info(f"Training ensemble on {len(X_train)} samples, {X_train.shape[1]} features")
        
        # Store feature names
        self.feature_names = list(X_train.columns)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        if X_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
        
        metrics = {}
        
        # Train each model
        for name, model in self.models.items():
            logger.info(f"Training {name}...")
            
            try:
                # XGBoost needs eval_set for early stopping
                if name == 'xgboost' and X_val is not None:
                    model.fit(
                        X_train_scaled, y_train,
                        eval_set=[(X_val_scaled, y_val)],
                        verbose=False
                    )
                else:
                    model.fit(X_train_scaled, y_train)
                
                # Evaluate on validation set
                if X_val is not None:
                    y_pred = model.predict(X_val_scaled)
                    y_proba = model.predict_proba(X_val_scaled)
                    
                    acc = accuracy_score(y_val, y_pred)
                    precision = precision_score(y_val, y_pred, average='weighted', zero_division=0)
                    recall = recall_score(y_val, y_pred, average='weighted', zero_division=0)
                    
                    try:
                        auc = roc_auc_score(y_val, y_proba, multi_class='ovr', average='weighted')
                    except:
                        auc = 0.0
                    
                    metrics[name] = {
                        'accuracy': acc,
                        'precision': precision,
                        'recall': recall,
                        'roc_auc': auc
                    }
                    
                    logger.info(f"{name}: acc={acc:.3f}, prec={precision:.3f}, recall={recall:.3f}, auc={auc:.3f}")
                else:
                    metrics[name] = {'accuracy': 0.0}
                    
            except Exception as e:
                logger.error(f"Error training {name}: {e}")
                metrics[name] = {'error': str(e)}
        
        self.is_trained = True
        logger.info("Ensemble training complete")
        
        return metrics
        
    def predict(self, X: pd.DataFrame) -> Dict[str, any]:
        """
        Ensemble prediction with voting
        
        Args:
            X: Features (single row or batch)
            
        Returns:
            prediction: {
                'signal': 'BUY' | 'HOLD' | 'SELL',
                'confidence': 0.0 - 1.0,
                'votes': {'xgboost': 2, 'rf': 2, 'gb': 1},
                'probabilities': model probabilities
            }
        """
        if not self.is_trained:
            raise ValueError("Models not trained - call train() first")
        
        # Handle single row
        if len(X.shape) == 1 or X.shape[0] == 1:
            return self._predict_single(X)
        else:
            return self._predict_batch(X)
    
    def _predict_single(self, X: pd.DataFrame) -> Dict[str, any]:
        """Single prediction"""
        
        # Ensure correct feature order
        X_ordered = X[self.feature_names]
        X_scaled = self.scaler.transform(X_ordered.values.reshape(1, -1))
        
        votes = []
        probas = {}
        
        # Get each model's prediction
        for name, model in self.models.items():
            pred = model.predict(X_scaled)[0]
            proba = model.predict_proba(X_scaled)[0]
            
            votes.append(pred)
            probas[name] = proba.tolist()
        
        # Vote counting
        vote_counts = {0: 0, 1: 0, 2: 0}
        for vote in votes:
            vote_counts[vote] += 1
        
        # Final prediction = majority vote
        final_vote = max(vote_counts, key=vote_counts.get)
        signal_map = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
        final_signal = signal_map[final_vote]
        
        # Confidence = agreement + probability certainty
        agreement = vote_counts[final_vote] / len(votes)
        
        # Average probability across models for final class
        avg_proba = np.mean([probas[m][final_vote] for m in probas])
        
        confidence = (agreement * 0.7) + (avg_proba * 0.3)
        
        return {
            'signal': final_signal,
            'confidence': float(confidence),
            'votes': {name: int(vote) for name, vote in zip(self.models.keys(), votes)},
            'vote_counts': vote_counts,
            'probabilities': probas,
            'agreement': float(agreement)
        }
    
    def _predict_batch(self, X: pd.DataFrame) -> pd.DataFrame:
        """Batch predictions"""
        
        predictions = []
        
        for idx in range(len(X)):
            row = X.iloc[idx:idx+1]
            pred = self._predict_single(row)
            predictions.append(pred)
        
        return pd.DataFrame(predictions)
    
    def save(self, path: str):
        """Save ensemble to disk"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save each model
        for name, model in self.models.items():
            model_path = path / f"{name}.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
        
        # Save scaler
        scaler_path = path / "scaler.pkl"
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Save metadata
        metadata = {
            'feature_names': self.feature_names,
            'is_trained': self.is_trained,
            'use_gpu': self.use_gpu
        }
        metadata_path = path / "metadata.pkl"
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        logger.info(f"Ensemble saved to {path}")
    
    def load(self, path: str):
        """Load ensemble from disk"""
        path = Path(path)
        
        # Load metadata
        metadata_path = path / "metadata.pkl"
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        self.feature_names = metadata['feature_names']
        self.is_trained = metadata['is_trained']
        self.use_gpu = metadata['use_gpu']
        
        # Load scaler
        scaler_path = path / "scaler.pkl"
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        # Load each model
        self.models = {}
        for model_file in path.glob("*.pkl"):
            if model_file.stem not in ['scaler', 'metadata']:
                with open(model_file, 'rb') as f:
                    self.models[model_file.stem] = pickle.load(f)
        
        logger.info(f"Ensemble loaded from {path}")


def quick_test():
    """Quick test of ensemble"""
    print("Testing MultiModelEnsemble...")
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000
    n_features = 20
    
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    
    # Synthetic labels with some signal
    y = np.random.randint(0, 3, n_samples)
    # Add signal: if feature_0 > 1, more likely BUY
    y[X['feature_0'] > 1] = 2
    # If feature_0 < -1, more likely SELL
    y[X['feature_0'] < -1] = 0
    
    # Split
    split = int(0.8 * n_samples)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Train
    ensemble = MultiModelEnsemble(use_gpu=False)
    metrics = ensemble.train(X_train, y_train, X_test, y_test)
    
    print("\nTraining Metrics:")
    for model, metric in metrics.items():
        print(f"{model}: {metric}")
    
    # Test prediction
    test_row = X_test.iloc[0:1]
    pred = ensemble.predict(test_row)
    
    print(f"\nTest Prediction:")
    print(f"Signal: {pred['signal']}")
    print(f"Confidence: {pred['confidence']:.3f}")
    print(f"Votes: {pred['votes']}")
    print(f"Agreement: {pred['agreement']:.3f}")
    
    print("\n✅ MultiModelEnsemble test complete")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    quick_test()
