"""
Distribution Drift Detection (Perplexity Q12)
==============================================
Monitor feature distributions for model degradation using statistical tests.

Strategy (Perplexity Q12):
1. KS Test (Kolmogorov-Smirnov): Compare train vs recent 30d distributions
2. Threshold: p-value < 0.05 → feature drifted significantly
3. Trigger: if >20% features drift → retrain meta-learner
4. PSI (Population Stability Index) as alternative metric

Use Case:
- Production monitoring: detect when model assumptions break
- Retrain scheduling: automate retraining based on data drift
- Feature health: identify which features are degrading

Author: Quantum AI Trader
"""

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from typing import Dict, List, Tuple, Optional
import logging
import json
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DriftDetector:
    """
    Monitor feature distributions for drift using KS test and PSI.
    
    Perplexity Q12 Implementation:
    - KS test: non-parametric, works for any distribution
    - p-value threshold: 0.05 (95% confidence drift detection)
    - Retrain trigger: >20% features drifted
    - Monitoring window: recent 30 days vs training baseline
    """
    
    def __init__(
        self,
        ks_threshold: float = 0.05,
        psi_threshold: float = 0.1,
        drift_percentage_trigger: float = 0.20,
        monitoring_window: int = 30
    ):
        """
        Initialize drift detector.
        
        Args:
            ks_threshold: KS test p-value threshold (default 0.05)
            psi_threshold: PSI threshold for drift (default 0.1)
            drift_percentage_trigger: % features drifted to trigger retrain (default 0.20)
            monitoring_window: Days for recent window comparison (default 30)
        """
        self.ks_threshold = ks_threshold
        self.psi_threshold = psi_threshold
        self.drift_percentage_trigger = drift_percentage_trigger
        self.monitoring_window = monitoring_window
        
        # Training baseline (fitted during training)
        self.baseline_stats_: Dict[str, Dict] = {}
        self.feature_names_: List[str] = []
        self.is_fitted = False
        
        # Drift history
        self.drift_history_: List[Dict] = []
    
    def fit(self, X_train: pd.DataFrame) -> 'DriftDetector':
        """
        Fit detector on training baseline distribution.
        
        Stores statistics for each feature to compare against production data.
        
        Args:
            X_train: Training feature matrix
            
        Returns:
            self for method chaining
        """
        logger.info(f"Fitting drift detector on {X_train.shape[0]} training samples...")
        
        self.feature_names_ = X_train.columns.tolist()
        
        # Compute baseline statistics for each feature
        for col in self.feature_names_:
            feature_data = X_train[col].dropna()
            
            self.baseline_stats_[col] = {
                'mean': float(feature_data.mean()),
                'std': float(feature_data.std()),
                'min': float(feature_data.min()),
                'max': float(feature_data.max()),
                'q25': float(feature_data.quantile(0.25)),
                'q50': float(feature_data.quantile(0.50)),
                'q75': float(feature_data.quantile(0.75)),
                'distribution': feature_data.values  # Store full distribution for KS test
            }
        
        self.is_fitted = True
        logger.info(f"✅ Baseline computed for {len(self.feature_names_)} features")
        
        return self
    
    def detect_drift_ks(
        self,
        X_recent: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Detect drift using Kolmogorov-Smirnov test.
        
        KS test compares two empirical distributions:
        - H0: distributions are the same
        - H1: distributions are different
        - p-value < 0.05 → reject H0 → drift detected
        
        Args:
            X_recent: Recent feature matrix (last 30 days)
            
        Returns:
            DataFrame with columns [feature, ks_statistic, p_value, drifted]
        """
        if not self.is_fitted:
            raise ValueError("DriftDetector must be fit before detecting drift")
        
        results = []
        
        for col in self.feature_names_:
            if col not in X_recent.columns:
                logger.warning(f"Feature {col} missing from recent data")
                continue
            
            # Get baseline and recent distributions
            baseline_dist = self.baseline_stats_[col]['distribution']
            recent_dist = X_recent[col].dropna().values
            
            if len(recent_dist) < 10:
                logger.warning(f"Feature {col}: insufficient recent data ({len(recent_dist)} samples)")
                continue
            
            # Run KS test
            ks_stat, p_value = ks_2samp(baseline_dist, recent_dist)
            
            results.append({
                'feature': col,
                'ks_statistic': ks_stat,
                'p_value': p_value,
                'drifted': p_value < self.ks_threshold
            })
        
        return pd.DataFrame(results)
    
    def compute_psi(
        self,
        expected: np.ndarray,
        actual: np.ndarray,
        buckets: int = 10
    ) -> float:
        """
        Compute Population Stability Index (PSI).
        
        PSI measures distribution shift:
        - PSI < 0.1: No significant change
        - PSI 0.1-0.2: Moderate change, investigate
        - PSI > 0.2: Significant drift, retrain
        
        Formula: PSI = sum((actual% - expected%) * ln(actual% / expected%))
        
        Args:
            expected: Baseline distribution (training)
            actual: Recent distribution (production)
            buckets: Number of bins for discretization
            
        Returns:
            PSI score
        """
        # Remove NaN values
        expected = expected[~np.isnan(expected)]
        actual = actual[~np.isnan(actual)]
        
        if len(expected) < 10 or len(actual) < 10:
            return 0.0
        
        # Create bins based on expected distribution
        breakpoints = np.percentile(expected, np.linspace(0, 100, buckets + 1))
        breakpoints = np.unique(breakpoints)  # Remove duplicates
        
        if len(breakpoints) <= 2:
            return 0.0
        
        # Compute expected percentages
        expected_percents = np.histogram(expected, bins=breakpoints)[0] / len(expected)
        
        # Compute actual percentages
        actual_percents = np.histogram(actual, bins=breakpoints)[0] / len(actual)
        
        # Add small epsilon to avoid division by zero
        epsilon = 1e-10
        expected_percents = np.maximum(expected_percents, epsilon)
        actual_percents = np.maximum(actual_percents, epsilon)
        
        # Compute PSI
        psi_values = (actual_percents - expected_percents) * np.log(actual_percents / expected_percents)
        psi = np.sum(psi_values)
        
        return float(psi)
    
    def detect_drift_psi(
        self,
        X_recent: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Detect drift using Population Stability Index.
        
        Args:
            X_recent: Recent feature matrix
            
        Returns:
            DataFrame with columns [feature, psi, drifted]
        """
        if not self.is_fitted:
            raise ValueError("DriftDetector must be fit before detecting drift")
        
        results = []
        
        for col in self.feature_names_:
            if col not in X_recent.columns:
                continue
            
            baseline_dist = self.baseline_stats_[col]['distribution']
            recent_dist = X_recent[col].dropna().values
            
            if len(recent_dist) < 10:
                continue
            
            psi = self.compute_psi(baseline_dist, recent_dist)
            
            results.append({
                'feature': col,
                'psi': psi,
                'drifted': psi > self.psi_threshold
            })
        
        return pd.DataFrame(results)
    
    def check_retrain_trigger(
        self,
        X_recent: pd.DataFrame,
        method: str = 'ks'
    ) -> Dict:
        """
        Check if retraining should be triggered.
        
        Args:
            X_recent: Recent feature matrix
            method: Drift detection method ('ks' or 'psi')
            
        Returns:
            Dict with keys:
            - should_retrain: bool
            - drift_percentage: float (% features drifted)
            - drifted_features: List[str]
            - drift_details: DataFrame
        """
        if method == 'ks':
            drift_df = self.detect_drift_ks(X_recent)
        elif method == 'psi':
            drift_df = self.detect_drift_psi(X_recent)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Calculate drift percentage
        total_features = len(drift_df)
        drifted_features = drift_df[drift_df['drifted']]['feature'].tolist()
        drift_percentage = len(drifted_features) / total_features if total_features > 0 else 0
        
        # Should retrain?
        should_retrain = drift_percentage > self.drift_percentage_trigger
        
        # Log results
        logger.info(f"Drift Check ({method.upper()}):")
        logger.info(f"  Drifted features: {len(drifted_features)}/{total_features} ({drift_percentage*100:.1f}%)")
        logger.info(f"  Retrain trigger: {'YES ⚠️' if should_retrain else 'NO ✅'}")
        
        # Store in history
        self.drift_history_.append({
            'timestamp': datetime.now().isoformat(),
            'method': method,
            'drift_percentage': drift_percentage,
            'should_retrain': should_retrain,
            'drifted_features': drifted_features
        })
        
        return {
            'should_retrain': should_retrain,
            'drift_percentage': drift_percentage,
            'drifted_features': drifted_features,
            'drift_details': drift_df
        }
    
    def get_drift_summary(
        self,
        X_recent: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Get comprehensive drift summary (KS + PSI).
        
        Args:
            X_recent: Recent feature matrix
            
        Returns:
            DataFrame with both KS and PSI metrics
        """
        ks_df = self.detect_drift_ks(X_recent)
        psi_df = self.detect_drift_psi(X_recent)
        
        # Merge results
        summary = ks_df.merge(psi_df, on='feature', suffixes=('_ks', '_psi'))
        
        # Add consensus column (drift detected by both methods)
        summary['drift_consensus'] = summary['drifted_ks'] & summary['drifted_psi']
        
        # Sort by drift severity (KS p-value ascending)
        summary = summary.sort_values('p_value')
        
        return summary
    
    def save(self, filepath: str):
        """Save fitted detector to disk."""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted DriftDetector")
        
        save_path = Path(filepath)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy arrays to lists for JSON serialization
        baseline_stats_serializable = {}
        for col, stats in self.baseline_stats_.items():
            baseline_stats_serializable[col] = {
                k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in stats.items()
            }
        
        save_data = {
            'ks_threshold': self.ks_threshold,
            'psi_threshold': self.psi_threshold,
            'drift_percentage_trigger': self.drift_percentage_trigger,
            'monitoring_window': self.monitoring_window,
            'baseline_stats': baseline_stats_serializable,
            'feature_names': self.feature_names_,
            'drift_history': self.drift_history_
        }
        
        with open(filepath, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        logger.info(f"DriftDetector saved to {filepath}")
    
    @staticmethod
    def load(filepath: str) -> 'DriftDetector':
        """Load fitted detector from disk."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        detector = DriftDetector(
            ks_threshold=data['ks_threshold'],
            psi_threshold=data['psi_threshold'],
            drift_percentage_trigger=data['drift_percentage_trigger'],
            monitoring_window=data['monitoring_window']
        )
        
        # Restore baseline stats (convert lists back to numpy arrays)
        detector.baseline_stats_ = {}
        for col, stats in data['baseline_stats'].items():
            detector.baseline_stats_[col] = {
                k: (np.array(v) if k == 'distribution' else v)
                for k, v in stats.items()
            }
        
        detector.feature_names_ = data['feature_names']
        detector.drift_history_ = data['drift_history']
        detector.is_fitted = True
        
        logger.info(f"DriftDetector loaded from {filepath}")
        return detector


# ============================================================================
# TEST HARNESS
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("DRIFT DETECTOR - TEST (Perplexity Q12)")
    print("=" * 80)
    
    # Generate training data (baseline)
    np.random.seed(42)
    n_train = 500
    n_recent = 30
    
    # Training features (baseline distributions)
    X_train = pd.DataFrame({
        'rsi_14': np.random.normal(50, 15, n_train),
        'macd': np.random.normal(0, 2, n_train),
        'volume_ratio': np.random.lognormal(0, 0.5, n_train),
        'price_momentum': np.random.normal(0.001, 0.02, n_train),
        'volatility': np.random.gamma(2, 0.5, n_train)
    })
    
    print(f"\n📊 Training Baseline: {n_train} samples, {len(X_train.columns)} features")
    
    # Scenario 1: No drift (same distribution)
    print("\n" + "=" * 80)
    print("SCENARIO 1: NO DRIFT (same distribution)")
    print("=" * 80)
    
    X_recent_no_drift = pd.DataFrame({
        'rsi_14': np.random.normal(50, 15, n_recent),
        'macd': np.random.normal(0, 2, n_recent),
        'volume_ratio': np.random.lognormal(0, 0.5, n_recent),
        'price_momentum': np.random.normal(0.001, 0.02, n_recent),
        'volatility': np.random.gamma(2, 0.5, n_recent)
    })
    
    detector = DriftDetector(
        ks_threshold=0.05,
        psi_threshold=0.1,
        drift_percentage_trigger=0.20,
        monitoring_window=30
    )
    
    detector.fit(X_train)
    
    result_no_drift = detector.check_retrain_trigger(X_recent_no_drift, method='ks')
    print(f"\n  Drifted features: {result_no_drift['drifted_features']}")
    print(f"  Should retrain: {result_no_drift['should_retrain']}")
    
    # Scenario 2: Moderate drift (some features changed)
    print("\n" + "=" * 80)
    print("SCENARIO 2: MODERATE DRIFT (volatility regime shift)")
    print("=" * 80)
    
    X_recent_moderate = pd.DataFrame({
        'rsi_14': np.random.normal(50, 15, n_recent),  # Same
        'macd': np.random.normal(0, 2, n_recent),  # Same
        'volume_ratio': np.random.lognormal(0, 0.5, n_recent),  # Same
        'price_momentum': np.random.normal(0.001, 0.02, n_recent),  # Same
        'volatility': np.random.gamma(4, 0.8, n_recent)  # DRIFTED (higher volatility)
    })
    
    result_moderate = detector.check_retrain_trigger(X_recent_moderate, method='ks')
    print(f"\n  Drifted features: {result_moderate['drifted_features']}")
    print(f"  Should retrain: {result_moderate['should_retrain']}")
    
    # Scenario 3: Severe drift (>20% features changed)
    print("\n" + "=" * 80)
    print("SCENARIO 3: SEVERE DRIFT (market regime change)")
    print("=" * 80)
    
    X_recent_severe = pd.DataFrame({
        'rsi_14': np.random.normal(70, 10, n_recent),  # DRIFTED (overbought regime)
        'macd': np.random.normal(3, 1, n_recent),  # DRIFTED (strong bullish)
        'volume_ratio': np.random.lognormal(1, 0.8, n_recent),  # DRIFTED (high volume)
        'price_momentum': np.random.normal(0.001, 0.02, n_recent),  # Same
        'volatility': np.random.gamma(2, 0.5, n_recent)  # Same
    })
    
    result_severe = detector.check_retrain_trigger(X_recent_severe, method='ks')
    print(f"\n  Drifted features: {result_severe['drifted_features']}")
    print(f"  Should retrain: {result_severe['should_retrain']} ⚠️")
    
    # Comprehensive drift summary
    print("\n" + "=" * 80)
    print("COMPREHENSIVE DRIFT ANALYSIS (KS + PSI)")
    print("=" * 80)
    
    summary = detector.get_drift_summary(X_recent_severe)
    print("\n" + summary.to_string(index=False))
    
    # PSI breakdown
    print("\n" + "=" * 80)
    print("PSI DRIFT DETECTION")
    print("=" * 80)
    
    psi_result = detector.check_retrain_trigger(X_recent_severe, method='psi')
    print(f"\n  Drifted features (PSI): {psi_result['drifted_features']}")
    print(f"  Should retrain (PSI): {psi_result['should_retrain']}")
    
    # Drift history
    print("\n" + "=" * 80)
    print("DRIFT MONITORING HISTORY")
    print("=" * 80)
    
    for i, entry in enumerate(detector.drift_history_, 1):
        print(f"\n  Check #{i} ({entry['method'].upper()}):")
        print(f"    Timestamp: {entry['timestamp']}")
        print(f"    Drift: {entry['drift_percentage']*100:.1f}%")
        print(f"    Retrain: {entry['should_retrain']}")
    
    # Save/load test
    print("\n" + "=" * 80)
    print("PERSISTENCE TEST")
    print("=" * 80)
    
    test_path = "/tmp/test_drift_detector.json"
    detector.save(test_path)
    
    loaded_detector = DriftDetector.load(test_path)
    print(f"✅ Loaded detector has {len(loaded_detector.feature_names_)} features")
    print(f"✅ Loaded detector has {len(loaded_detector.drift_history_)} history entries")
    
    # Validation checks
    print("\n" + "=" * 80)
    print("VALIDATION CHECKS")
    print("=" * 80)
    
    checks_passed = 0
    checks_total = 0
    
    # No drift scenario should not trigger retrain
    checks_total += 1
    if not result_no_drift['should_retrain']:
        print(f"✅ No drift: retrain not triggered")
        checks_passed += 1
    else:
        print(f"❌ No drift: false positive retrain trigger")
    
    # Moderate drift (1 feature) should not trigger retrain
    checks_total += 1
    if not result_moderate['should_retrain'] and result_moderate['drift_percentage'] < 0.20:
        print(f"✅ Moderate drift: retrain not triggered ({result_moderate['drift_percentage']*100:.1f}% < 20%)")
        checks_passed += 1
    else:
        print(f"❌ Moderate drift: unexpected retrain trigger")
    
    # Severe drift (>20% features) should trigger retrain
    checks_total += 1
    if result_severe['should_retrain'] and result_severe['drift_percentage'] > 0.20:
        print(f"✅ Severe drift: retrain triggered ({result_severe['drift_percentage']*100:.1f}% > 20%)")
        checks_passed += 1
    else:
        print(f"❌ Severe drift: failed to trigger retrain")
    
    # Drifted features should be correct
    checks_total += 1
    expected_drifted = {'rsi_14', 'macd', 'volume_ratio'}
    detected_drifted = set(result_severe['drifted_features'])
    if expected_drifted.issubset(detected_drifted):
        print(f"✅ Detected expected drifted features: {detected_drifted}")
        checks_passed += 1
    else:
        print(f"❌ Failed to detect drifted features: expected {expected_drifted}, got {detected_drifted}")
    
    # PSI should identify drift
    checks_total += 1
    if psi_result['should_retrain']:
        print(f"✅ PSI method also detected severe drift")
        checks_passed += 1
    else:
        print(f"❌ PSI method failed to detect drift")
    
    # History tracking
    checks_total += 1
    if len(detector.drift_history_) == 5:  # 3 KS checks + 2 PSI checks
        print(f"✅ Drift history tracking: {len(detector.drift_history_)} entries")
        checks_passed += 1
    else:
        print(f"❌ Drift history incomplete: {len(detector.drift_history_)} entries")
    
    print(f"\n{'=' * 80}")
    print(f"✅ Drift Detector: {checks_passed}/{checks_total} checks passed")
    print("=" * 80)
