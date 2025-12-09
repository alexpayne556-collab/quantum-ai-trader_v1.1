"""
Feature Selection Pipeline (Perplexity Q7)
==========================================
Implements correlation filtering + Random Forest importance ranking
to select top N predictive features from 60+ engineered candidates.

Strategy:
1. Remove highly correlated features (r > 0.85) to reduce multicollinearity
2. Train Random Forest on remaining features
3. Select top N features by importance score
4. Expected: 60 candidates → 20 most predictive features

Author: Quantum AI Trader
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from typing import List, Tuple, Optional
import logging
import joblib
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureSelector:
    """
    Production-grade feature selection using correlation filtering + RF importance.
    
    Perplexity Q7 Implementation:
    - Correlation threshold: 0.85 (drop one feature from highly correlated pairs)
    - Importance method: Random Forest feature_importances_
    - Default selection: Top 20 features
    - Handles missing values, infinite values, and zero-variance features
    """
    
    def __init__(
        self,
        correlation_threshold: float = 0.85,
        importance_threshold: Optional[float] = None,
        n_features: int = 20,
        random_state: int = 42
    ):
        """
        Initialize feature selector.
        
        Args:
            correlation_threshold: Drop features with correlation > threshold (default 0.85)
            importance_threshold: Optional minimum importance score (use n_features if None)
            n_features: Number of top features to select (default 20)
            random_state: Random seed for reproducibility
        """
        self.correlation_threshold = correlation_threshold
        self.importance_threshold = importance_threshold
        self.n_features = n_features
        self.random_state = random_state
        
        # Trained components
        self.selected_features_: List[str] = []
        self.feature_importance_: pd.DataFrame = pd.DataFrame()
        self.correlation_matrix_: Optional[pd.DataFrame] = None
        self.dropped_correlated_: List[str] = []
        
        # Random Forest for importance ranking
        self.rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=random_state,
            n_jobs=-1
        )
        
        self.is_fitted = False
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'FeatureSelector':
        """
        Fit feature selector on training data.
        
        Pipeline:
        1. Remove zero-variance features
        2. Remove highly correlated features (keep first in pair)
        3. Train Random Forest on remaining features
        4. Rank features by importance
        5. Select top N features
        
        Args:
            X: Feature matrix (rows=samples, columns=features)
            y: Target labels (binary: 0=loss, 1=win)
            
        Returns:
            self for method chaining
        """
        logger.info("Starting feature selection pipeline...")
        logger.info(f"  Input: {X.shape[0]} samples, {X.shape[1]} features")
        
        # Validate input
        X_clean = self._validate_and_clean(X)
        
        # Step 1: Remove zero-variance features
        variances = X_clean.var()
        zero_var_features = variances[variances == 0].index.tolist()
        if zero_var_features:
            logger.info(f"  Removing {len(zero_var_features)} zero-variance features")
            X_clean = X_clean.drop(columns=zero_var_features)
        
        # Step 2: Remove correlated features
        X_clean, dropped = self._remove_correlated_features(X_clean)
        self.dropped_correlated_ = dropped
        
        logger.info(f"  After correlation filter: {X_clean.shape[1]} features remain")
        
        # Step 3: Train Random Forest for importance ranking
        logger.info("  Training Random Forest for importance ranking...")
        self.rf_model.fit(X_clean, y)
        
        # Step 4: Get feature importances
        importance_df = pd.DataFrame({
            'feature': X_clean.columns,
            'importance': self.rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        self.feature_importance_ = importance_df
        
        # Step 5: Select top N features
        if self.importance_threshold is not None:
            # Select by importance threshold
            selected = importance_df[
                importance_df['importance'] >= self.importance_threshold
            ]['feature'].tolist()
            logger.info(f"  Selected {len(selected)} features with importance >= {self.importance_threshold}")
        else:
            # Select top N features
            selected = importance_df.head(self.n_features)['feature'].tolist()
            logger.info(f"  Selected top {len(selected)} features by importance")
        
        self.selected_features_ = selected
        self.is_fitted = True
        
        # Log top features
        logger.info("\n  Top 10 Features by Importance:")
        for idx, row in importance_df.head(10).iterrows():
            logger.info(f"    {row['feature']:<30} {row['importance']:.4f}")
        
        if self.dropped_correlated_:
            logger.info(f"\n  Dropped {len(self.dropped_correlated_)} correlated features")
            logger.info(f"    Examples: {', '.join(self.dropped_correlated_[:5])}")
        
        logger.info("\n✅ Feature selection complete!")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data to selected features only.
        
        Args:
            X: Feature matrix to transform
            
        Returns:
            DataFrame with selected features only
        """
        if not self.is_fitted:
            raise ValueError("FeatureSelector must be fit before transform")
        
        # Keep only selected features
        missing_features = set(self.selected_features_) - set(X.columns)
        if missing_features:
            logger.warning(f"Missing features in transform: {missing_features}")
        
        available_features = [f for f in self.selected_features_ if f in X.columns]
        
        return X[available_features]
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Fit selector and transform in one call."""
        return self.fit(X, y).transform(X)
    
    def _validate_and_clean(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and clean input data.
        
        - Replace inf with large values
        - Fill NaN with median
        - Check for minimum samples
        """
        X_clean = X.copy()
        
        # Replace inf values
        X_clean = X_clean.replace([np.inf, -np.inf], np.nan)
        
        # Count and report NaN values
        nan_counts = X_clean.isnull().sum()
        features_with_nan = nan_counts[nan_counts > 0]
        
        if len(features_with_nan) > 0:
            logger.info(f"  Found NaN values in {len(features_with_nan)} features")
            # Fill NaN with median
            for col in features_with_nan.index:
                median_val = X_clean[col].median()
                X_clean[col] = X_clean[col].fillna(median_val)
        
        # Check minimum samples
        if len(X_clean) < 100:
            logger.warning(f"  Low sample count: {len(X_clean)} (recommend >= 100)")
        
        return X_clean
    
    def _remove_correlated_features(
        self,
        X: pd.DataFrame
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Remove highly correlated features.
        
        Strategy: For each pair with correlation > threshold,
        keep the first feature and drop the second.
        
        Args:
            X: Feature matrix
            
        Returns:
            (cleaned_df, list_of_dropped_features)
        """
        # Compute correlation matrix
        corr_matrix = X.corr().abs()
        self.correlation_matrix_ = corr_matrix
        
        # Find upper triangle (avoid duplicates)
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Find features with correlation > threshold
        to_drop = [
            column for column in upper_tri.columns
            if any(upper_tri[column] > self.correlation_threshold)
        ]
        
        # Drop correlated features
        X_reduced = X.drop(columns=to_drop)
        
        return X_reduced, to_drop
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get ranked feature importance scores.
        
        Returns:
            DataFrame with columns [feature, importance]
        """
        if not self.is_fitted:
            raise ValueError("FeatureSelector must be fit before getting importance")
        
        return self.feature_importance_.copy()
    
    def get_selected_features(self) -> List[str]:
        """Get list of selected feature names."""
        if not self.is_fitted:
            raise ValueError("FeatureSelector must be fit before getting features")
        
        return self.selected_features_.copy()
    
    def get_correlation_pairs(self, threshold: Optional[float] = None) -> pd.DataFrame:
        """
        Get pairs of highly correlated features.
        
        Args:
            threshold: Correlation threshold (uses self.correlation_threshold if None)
            
        Returns:
            DataFrame with columns [feature_1, feature_2, correlation]
        """
        if self.correlation_matrix_ is None:
            raise ValueError("FeatureSelector must be fit before getting correlations")
        
        threshold = threshold or self.correlation_threshold
        
        # Get upper triangle
        corr = self.correlation_matrix_
        upper_tri = corr.where(
            np.triu(np.ones(corr.shape), k=1).astype(bool)
        )
        
        # Find pairs above threshold
        pairs = []
        for col in upper_tri.columns:
            for idx in upper_tri.index:
                value = upper_tri.loc[idx, col]
                if pd.notna(value) and value > threshold:
                    pairs.append({
                        'feature_1': idx,
                        'feature_2': col,
                        'correlation': value
                    })
        
        return pd.DataFrame(pairs).sort_values('correlation', ascending=False)
    
    def save(self, filepath: str):
        """Save fitted selector to disk."""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted FeatureSelector")
        
        save_path = Path(filepath)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(self, filepath)
        logger.info(f"FeatureSelector saved to {filepath}")
    
    @staticmethod
    def load(filepath: str) -> 'FeatureSelector':
        """Load fitted selector from disk."""
        selector = joblib.load(filepath)
        logger.info(f"FeatureSelector loaded from {filepath}")
        return selector


# ============================================================================
# TEST HARNESS
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("FEATURE SELECTOR - TEST (Perplexity Q7)")
    print("=" * 80)
    
    # Generate synthetic data with known correlations
    np.random.seed(42)
    n_samples = 500
    
    # Base features (independent)
    base_features = {
        'rsi_14': np.random.normal(50, 20, n_samples),
        'macd_signal': np.random.normal(0, 1, n_samples),
        'volume_ratio': np.random.lognormal(0, 0.5, n_samples),
        'atr_percent': np.random.uniform(0, 5, n_samples),
        'obv_trend': np.random.normal(0, 1, n_samples),
    }
    
    # Create correlated features (should be dropped)
    correlated_features = {
        'rsi_14_smooth': base_features['rsi_14'] + np.random.normal(0, 2, n_samples),  # r > 0.95
        'macd_signal_lag': base_features['macd_signal'] + np.random.normal(0, 0.1, n_samples),  # r > 0.90
        'volume_ratio_ma': base_features['volume_ratio'] * np.random.normal(1, 0.05, n_samples),  # r > 0.85
    }
    
    # Additional independent features
    independent_features = {
        f'feature_{i}': np.random.normal(0, 1, n_samples)
        for i in range(15)
    }
    
    # Combine all features
    all_features = {**base_features, **correlated_features, **independent_features}
    X = pd.DataFrame(all_features)
    
    # Create target with some features being predictive
    y = (
        (X['rsi_14'] > 60).astype(int) * 0.4 +
        (X['macd_signal'] > 0).astype(int) * 0.3 +
        (X['volume_ratio'] > 1.5).astype(int) * 0.2 +
        np.random.binomial(1, 0.1, n_samples)  # Noise
    )
    y = (y > 0.5).astype(int)
    
    print(f"\n📊 Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"   Target distribution: {y.value_counts().to_dict()}")
    
    # Test feature selector
    print("\n" + "=" * 80)
    print("FITTING FEATURE SELECTOR")
    print("=" * 80)
    
    selector = FeatureSelector(
        correlation_threshold=0.85,
        n_features=10,
        random_state=42
    )
    
    X_selected = selector.fit_transform(X, y)
    
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    
    print(f"\n✅ Selected {len(selector.selected_features_)} features:")
    for feat in selector.selected_features_:
        print(f"   - {feat}")
    
    print(f"\n❌ Dropped {len(selector.dropped_correlated_)} correlated features:")
    for feat in selector.dropped_correlated_:
        print(f"   - {feat}")
    
    # Test correlation pairs
    print("\n" + "=" * 80)
    print("HIGH CORRELATION PAIRS")
    print("=" * 80)
    
    corr_pairs = selector.get_correlation_pairs()
    if len(corr_pairs) > 0:
        print(corr_pairs.head(10).to_string(index=False))
    else:
        print("No highly correlated pairs found")
    
    # Test save/load
    print("\n" + "=" * 80)
    print("PERSISTENCE TEST")
    print("=" * 80)
    
    test_path = "/tmp/test_feature_selector.joblib"
    selector.save(test_path)
    
    loaded_selector = FeatureSelector.load(test_path)
    print(f"✅ Loaded selector has {len(loaded_selector.selected_features_)} features")
    
    # Transform with loaded selector
    X_selected_loaded = loaded_selector.transform(X)
    assert X_selected.equals(X_selected_loaded), "Loaded selector produces different results!"
    print("✅ Transform results match original")
    
    print("\n" + "=" * 80)
    print("✅ Feature Selector implementation complete!")
    print("=" * 80)
