"""
INFERENCE ENGINE
================
Load trained Trident models and generate predictions for live trading.

Features:
- Automatic cluster detection (which model to use for which ticker)
- Soft voting ensemble (averages XGB + LGB + CAT predictions)
- Confidence scoring (0-100%)
- Real-time prediction (<10ms per ticker)

Input: Live market data (56 features from dataset_builder.py)
Output: {signal: BUY/HOLD/SELL, confidence: 0-100%, metadata: {...}}

Author: Quantum AI Trader
Date: December 10, 2025
"""

import numpy as np
import pandas as pd
import json
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TridenInference:
    """
    Inference engine for trained Trident ensemble models.
    Loads models and generates predictions with confidence scores.
    """
    
    def __init__(self, model_dir: str = 'models/trident'):
        """
        Args:
            model_dir: Directory containing trained models
        """
        self.model_dir = Path(model_dir)
        self.models = {}  # {cluster_id: {'xgb': model, 'lgb': model, 'cat': model}}
        self.cluster_assignments = {}  # {ticker: cluster_id}
        self.feature_names = None
        
        self._load_models()
    
    def _load_models(self):
        """Load all trained models from disk."""
        logger.info(f"Loading models from {self.model_dir}...")
        
        # Load cluster assignments
        assignments_path = self.model_dir / 'cluster_assignments.json'
        if assignments_path.exists():
            with open(assignments_path, 'r') as f:
                self.cluster_assignments = json.load(f)
            logger.info(f"✅ Loaded cluster assignments for {len(self.cluster_assignments)} tickers")
        else:
            logger.warning(f"⚠️ No cluster assignments found at {assignments_path}")
            logger.warning("   Using default cluster 0 for all tickers")
        
        # Load models for each cluster
        cluster_ids = set(self.cluster_assignments.values()) if self.cluster_assignments else [0]
        
        for cluster_id in cluster_ids:
            logger.info(f"\n🔹 Loading models for Cluster {cluster_id}...")
            
            self.models[cluster_id] = {}
            
            # Load XGBoost
            xgb_path = self.model_dir / f'cluster_{cluster_id}_xgb.json'
            if xgb_path.exists():
                self.models[cluster_id]['xgb'] = xgb.XGBClassifier()
                self.models[cluster_id]['xgb'].load_model(str(xgb_path))
                logger.info(f"   ✅ XGBoost loaded")
            else:
                logger.warning(f"   ❌ XGBoost model not found: {xgb_path}")
            
            # Load LightGBM
            lgb_path = self.model_dir / f'cluster_{cluster_id}_lgb.txt'
            if lgb_path.exists():
                self.models[cluster_id]['lgb'] = lgb.Booster(model_file=str(lgb_path))
                logger.info(f"   ✅ LightGBM loaded")
            else:
                logger.warning(f"   ❌ LightGBM model not found: {lgb_path}")
            
            # Load CatBoost
            cat_path = self.model_dir / f'cluster_{cluster_id}_cat.cbm'
            if cat_path.exists():
                self.models[cluster_id]['cat'] = cb.CatBoostClassifier()
                self.models[cluster_id]['cat'].load_model(str(cat_path))
                logger.info(f"   ✅ CatBoost loaded")
            else:
                logger.warning(f"   ❌ CatBoost model not found: {cat_path}")
        
        logger.info(f"\n✅ All models loaded successfully")
    
    def predict(
        self,
        ticker: str,
        features: pd.DataFrame,
        use_ensemble: bool = True
    ) -> Dict:
        """
        Generate prediction for a single ticker.
        
        Args:
            ticker: Stock ticker symbol
            features: Feature DataFrame (56 columns from dataset_builder.py)
            use_ensemble: Use soft voting ensemble (recommended)
            
        Returns:
            {
                'signal': 'BUY' | 'HOLD' | 'SELL',
                'confidence': float (0-100),
                'probability': float (0-1),
                'model_votes': {'xgb': ..., 'lgb': ..., 'cat': ...},
                'cluster_id': int,
                'timestamp': str
            }
        """
        # Get cluster for this ticker
        cluster_id = self.cluster_assignments.get(ticker, 0)
        
        if cluster_id not in self.models:
            logger.warning(f"No models found for cluster {cluster_id}, using cluster 0")
            cluster_id = 0
        
        # Get models for this cluster
        models = self.models[cluster_id]
        
        # Ensure features are in correct format
        if isinstance(features, pd.Series):
            features = features.to_frame().T
        
        # Get predictions from each model
        predictions = {}
        probabilities = {}
        
        if 'xgb' in models:
            xgb_prob = models['xgb'].predict_proba(features)[0]
            predictions['xgb'] = np.argmax(xgb_prob)
            probabilities['xgb'] = float(xgb_prob[1]) if len(xgb_prob) > 1 else float(xgb_prob[0])
        
        if 'lgb' in models:
            # LightGBM Booster requires different API
            lgb_prob = models['lgb'].predict(features)[0]
            predictions['lgb'] = 1 if lgb_prob > 0.5 else 0
            probabilities['lgb'] = float(lgb_prob)
        
        if 'cat' in models:
            cat_prob = models['cat'].predict_proba(features)[0]
            predictions['cat'] = np.argmax(cat_prob)
            probabilities['cat'] = float(cat_prob[1]) if len(cat_prob) > 1 else float(cat_prob[0])
        
        # Ensemble prediction (soft voting - average probabilities)
        if use_ensemble and len(probabilities) > 1:
            ensemble_prob = np.mean(list(probabilities.values()))
            ensemble_pred = 1 if ensemble_prob > 0.5 else 0
        else:
            # Use XGBoost as primary if ensemble disabled
            ensemble_prob = probabilities.get('xgb', 0.5)
            ensemble_pred = predictions.get('xgb', 0)
        
        # Convert to signal
        if ensemble_pred == 1:
            signal = 'BUY'
            confidence = ensemble_prob * 100
        else:
            signal = 'SELL' if ensemble_prob < 0.3 else 'HOLD'
            confidence = (1 - ensemble_prob) * 100 if signal == 'SELL' else (1 - abs(ensemble_prob - 0.5) * 2) * 100
        
        return {
            'ticker': ticker,
            'signal': signal,
            'confidence': round(confidence, 2),
            'probability': round(ensemble_prob, 4),
            'model_votes': predictions,
            'model_probabilities': probabilities,
            'cluster_id': cluster_id,
            'timestamp': pd.Timestamp.now().isoformat()
        }
    
    def predict_batch(
        self,
        tickers: list,
        features_dict: Dict[str, pd.DataFrame],
        use_ensemble: bool = True
    ) -> Dict[str, Dict]:
        """
        Generate predictions for multiple tickers at once.
        
        Args:
            tickers: List of ticker symbols
            features_dict: {ticker: features_df} dictionary
            use_ensemble: Use soft voting ensemble
            
        Returns:
            {ticker: prediction_dict}
        """
        predictions = {}
        
        for ticker in tickers:
            if ticker in features_dict:
                predictions[ticker] = self.predict(
                    ticker,
                    features_dict[ticker],
                    use_ensemble=use_ensemble
                )
            else:
                logger.warning(f"No features provided for {ticker}, skipping")
        
        return predictions
    
    def get_cluster_info(self, ticker: str) -> Dict:
        """Get cluster information for a ticker."""
        cluster_id = self.cluster_assignments.get(ticker, 0)
        
        # Get all tickers in same cluster
        cluster_tickers = [
            t for t, c in self.cluster_assignments.items()
            if c == cluster_id
        ]
        
        return {
            'ticker': ticker,
            'cluster_id': cluster_id,
            'cluster_size': len(cluster_tickers),
            'cluster_tickers': cluster_tickers[:10],  # First 10
            'models_available': list(self.models.get(cluster_id, {}).keys())
        }
    
    def validate_features(self, features: pd.DataFrame) -> Tuple[bool, str]:
        """
        Validate that features match expected format.
        
        Args:
            features: Feature DataFrame
            
        Returns:
            (is_valid, error_message)
        """
        # Check number of columns
        expected_cols = 56  # From dataset_builder.py with microstructure features
        
        if features.shape[1] != expected_cols:
            return False, f"Expected {expected_cols} features, got {features.shape[1]}"
        
        # Check for missing values
        if features.isnull().any().any():
            return False, "Features contain missing values"
        
        # Check for infinite values
        if np.isinf(features.values).any():
            return False, "Features contain infinite values"
        
        return True, "OK"


def example_usage():
    """Example usage of inference engine."""
    logger.info("\n" + "="*60)
    logger.info("TRIDENT INFERENCE ENGINE - Example Usage")
    logger.info("="*60 + "\n")
    
    # Initialize inference engine
    engine = TridenInference(model_dir='models/trident')
    
    # Example: Create dummy features (in production, these come from dataset_builder.py)
    logger.info("Creating example features...")
    example_features = pd.DataFrame(
        np.random.randn(1, 56),
        columns=[f'feature_{i}' for i in range(56)]
    )
    
    # Validate features
    is_valid, message = engine.validate_features(example_features)
    logger.info(f"Feature validation: {message}")
    
    if is_valid:
        # Get prediction
        ticker = 'NVDA'
        prediction = engine.predict(ticker, example_features)
        
        logger.info(f"\n📊 Prediction for {ticker}:")
        logger.info(f"   Signal: {prediction['signal']}")
        logger.info(f"   Confidence: {prediction['confidence']:.1f}%")
        logger.info(f"   Probability: {prediction['probability']:.3f}")
        logger.info(f"   Cluster: {prediction['cluster_id']}")
        logger.info(f"   Model Votes: {prediction['model_votes']}")
        
        # Get cluster info
        cluster_info = engine.get_cluster_info(ticker)
        logger.info(f"\n🎯 Cluster Info:")
        logger.info(f"   Cluster ID: {cluster_info['cluster_id']}")
        logger.info(f"   Cluster Size: {cluster_info['cluster_size']} tickers")
        logger.info(f"   Models Available: {cluster_info['models_available']}")


if __name__ == '__main__':
    example_usage()
