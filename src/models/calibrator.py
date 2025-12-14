"""
Probability Calibrator using Platt Scaling
===========================================

Implementation of Perplexity Q3 recommendation:
- Platt Scaling (Logistic Regression on raw outputs)
- ECE (Expected Calibration Error) target: <0.05
- Rolling window: Last 100 trades, updated daily

Why Platt Scaling:
- Robust for small samples (<1000 trades)
- Only 2 parameters (A, B) → less overfitting than Isotonic
- Fast calibration: P_cal = 1/(1 + exp(A*P_raw + B))

Author: Quantum AI Trader
Date: December 8, 2025
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from typing import List, Tuple
import logging
from collections import deque
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProbabilityCalibrator:
    """
    Platt Scaling calibrator for trading model probabilities.
    
    Converts raw model confidence (often miscalibrated) into
    true win-rate probabilities using logistic regression.
    
    Example Problem:
    - Model says 70% confidence → actual win-rate only 55% (overconfident)
    
    Solution:
    - Train logistic regression: P_calibrated = sigmoid(A * P_raw + B)
    - Ensures predicted 70% → actual 70% win-rate
    
    Usage:
    ------
    >>> cal = ProbabilityCalibrator(window_size=100)
    >>> 
    >>> # After each trade, update with result
    >>> cal.add_observation(raw_score=0.75, actual_outcome=1)  # Win
    >>> cal.add_observation(raw_score=0.60, actual_outcome=0)  # Loss
    >>> 
    >>> # Once enough data, fit calibrator
    >>> cal.fit()
    >>> 
    >>> # Use calibrated probabilities
    >>> raw_prob = 0.70
    >>> calibrated_prob = cal.calibrate(raw_prob)  # True win-rate estimate
    """
    
    def __init__(self, window_size: int = 100, min_samples: int = 30):
        """
        Initialize calibrator.
        
        Args:
            window_size: Rolling window size (last N trades)
            min_samples: Minimum trades before calibration (30 = ~15 per class)
        """
        self.window_size = window_size
        self.min_samples = min_samples
        
        # Rolling windows for observations
        self.raw_scores = deque(maxlen=window_size)
        self.actual_outcomes = deque(maxlen=window_size)
        
        # Platt Scaling model (Logistic Regression)
        self.calibrator = LogisticRegression(
            C=1.0,
            solver='lbfgs',
            max_iter=500
        )
        
        self.is_fitted = False
        self.calibration_metrics = {}
        
    def add_observation(self, raw_score: float, actual_outcome: int):
        """
        Add a new trade result to rolling window.
        
        Args:
            raw_score: Raw model probability (0-1)
            actual_outcome: Trade result (1 = win, 0 = loss)
        """
        self.raw_scores.append(raw_score)
        self.actual_outcomes.append(actual_outcome)
        
        # Auto-refit if window is full and previously fitted
        if len(self.raw_scores) == self.window_size and self.is_fitted:
            self.fit()
    
    def fit(self) -> bool:
        """
        Fit calibrator on current rolling window.
        
        Returns:
            bool: True if successful, False if insufficient data
        """
        if len(self.raw_scores) < self.min_samples:
            logger.warning(f"Insufficient data for calibration: {len(self.raw_scores)}/{self.min_samples}")
            return False
        
        # Convert to arrays
        X = np.array(self.raw_scores).reshape(-1, 1)
        y = np.array(self.actual_outcomes)
        
        # Check for class balance (need both wins and losses)
        n_wins = y.sum()
        n_losses = len(y) - n_wins
        
        if n_wins < 5 or n_losses < 5:
            logger.warning(f"Class imbalance: {n_wins} wins, {n_losses} losses. Calibration unreliable.")
            return False
        
        # Fit Platt Scaling
        self.calibrator.fit(X, y)
        self.is_fitted = True
        
        # Compute calibration metrics
        y_pred_calibrated = self.calibrator.predict_proba(X)[:, 1]
        
        # Expected Calibration Error (ECE)
        ece = self._compute_ece(y, y_pred_calibrated, n_bins=10)
        
        # Reliability metrics
        actual_win_rate = y.mean()
        predicted_win_rate = y_pred_calibrated.mean()
        
        self.calibration_metrics = {
            'ece': ece,
            'actual_win_rate': actual_win_rate,
            'predicted_win_rate': predicted_win_rate,
            'n_samples': len(y),
            'n_wins': int(n_wins),
            'n_losses': int(n_losses)
        }
        
        logger.info(f"Calibrator fitted:")
        logger.info(f"  ECE: {ece:.4f} (target <0.05)")
        logger.info(f"  Actual win-rate: {actual_win_rate:.2%}")
        logger.info(f"  Predicted win-rate: {predicted_win_rate:.2%}")
        
        return True
    
    def calibrate(self, raw_score: float) -> float:
        """
        Calibrate a raw model probability.
        
        Args:
            raw_score: Raw model output (0-1)
            
        Returns:
            float: Calibrated probability (true win-rate estimate)
        """
        if not self.is_fitted:
            logger.debug("Calibrator not fitted, returning raw score")
            return raw_score
        
        # Clamp input to [0, 1]
        raw_score = np.clip(raw_score, 0.0, 1.0)
        
        # Apply Platt Scaling
        calibrated_prob = self.calibrator.predict_proba([[raw_score]])[:, 1][0]
        
        # Clamp output to [0.05, 0.95] (avoid extreme predictions)
        calibrated_prob = np.clip(calibrated_prob, 0.05, 0.95)
        
        return float(calibrated_prob)
    
    def calibrate_batch(self, raw_scores: List[float]) -> np.ndarray:
        """
        Calibrate multiple raw scores at once.
        
        Args:
            raw_scores: List of raw model outputs
            
        Returns:
            np.ndarray: Calibrated probabilities
        """
        if not self.is_fitted:
            return np.array(raw_scores)
        
        raw_scores = np.clip(raw_scores, 0.0, 1.0).reshape(-1, 1)
        calibrated_probs = self.calibrator.predict_proba(raw_scores)[:, 1]
        calibrated_probs = np.clip(calibrated_probs, 0.05, 0.95)
        
        return calibrated_probs
    
    def _compute_ece(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        n_bins: int = 10
    ) -> float:
        """
        Compute Expected Calibration Error (ECE).
        
        ECE measures calibration quality:
        - ECE = 0: Perfect calibration
        - ECE < 0.05: Good calibration
        - ECE > 0.10: Poor calibration (overconfident/underconfident)
        
        Formula:
        ECE = Σ (n_k / n) * |acc_k - conf_k|
        
        Where:
        - n_k: samples in bin k
        - acc_k: actual accuracy in bin k
        - conf_k: average confidence in bin k
        """
        # Create bins for probabilities
        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(y_pred, bins) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        
        ece = 0.0
        n_total = len(y_true)
        
        for bin_idx in range(n_bins):
            # Get samples in this bin
            in_bin = bin_indices == bin_idx
            n_in_bin = in_bin.sum()
            
            if n_in_bin == 0:
                continue
            
            # Actual accuracy in bin
            acc_bin = y_true[in_bin].mean()
            
            # Average confidence in bin
            conf_bin = y_pred[in_bin].mean()
            
            # Weighted absolute difference
            ece += (n_in_bin / n_total) * abs(acc_bin - conf_bin)
        
        return ece
    
    def get_calibration_curve(
        self,
        n_bins: int = 10
    ) -> pd.DataFrame:
        """
        Get calibration curve for visualization.
        
        Returns:
            DataFrame with columns: predicted_prob, actual_win_rate, count
        """
        if not self.is_fitted or len(self.raw_scores) < self.min_samples:
            return pd.DataFrame()
        
        X = np.array(self.raw_scores).reshape(-1, 1)
        y = np.array(self.actual_outcomes)
        
        # Get calibrated predictions
        y_pred = self.calibrator.predict_proba(X)[:, 1]
        
        # Create bins
        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(y_pred, bins) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        
        curve_data = []
        for bin_idx in range(n_bins):
            in_bin = bin_indices == bin_idx
            if in_bin.sum() == 0:
                continue
            
            predicted_prob = y_pred[in_bin].mean()
            actual_win_rate = y.mean()
            count = in_bin.sum()
            
            curve_data.append({
                'predicted_prob': predicted_prob,
                'actual_win_rate': actual_win_rate,
                'count': count
            })
        
        return pd.DataFrame(curve_data)
    
    def reset(self):
        """Clear all observations and reset calibrator."""
        self.raw_scores.clear()
        self.actual_outcomes.clear()
        self.is_fitted = False
        self.calibration_metrics = {}
        logger.info("Calibrator reset")
    
    def save(self, filepath: str):
        """Save calibrator to disk."""
        if not self.is_fitted:
            logger.warning("Cannot save unfitted calibrator")
            return
        
        calibrator_data = {
            'calibrator': self.calibrator,
            'raw_scores': list(self.raw_scores),
            'actual_outcomes': list(self.actual_outcomes),
            'calibration_metrics': self.calibration_metrics,
            'window_size': self.window_size,
            'min_samples': self.min_samples
        }
        
        joblib.dump(calibrator_data, filepath)
        logger.info(f"Calibrator saved to {filepath}")
    
    def load(self, filepath: str):
        """Load calibrator from disk."""
        data = joblib.load(filepath)
        
        self.calibrator = data['calibrator']
        self.raw_scores = deque(data['raw_scores'], maxlen=self.window_size)
        self.actual_outcomes = deque(data['actual_outcomes'], maxlen=self.window_size)
        self.calibration_metrics = data['calibration_metrics']
        self.is_fitted = True
        
        logger.info(f"Calibrator loaded from {filepath}")
        logger.info(f"  ECE: {self.calibration_metrics['ece']:.4f}")


if __name__ == "__main__":
    # ===== EXAMPLE USAGE =====
    print("=" * 80)
    print("PROBABILITY CALIBRATOR - TEST")
    print("=" * 80)
    
    # Simulate 100 trades with overconfident model
    np.random.seed(42)
    
    # Model is overconfident: predicts 70% but actual win-rate is 55%
    true_win_rate = 0.55
    model_overconfidence = 0.15
    
    raw_scores = []
    actual_outcomes = []
    
    for i in range(100):
        # Model prediction (overconfident)
        raw_score = np.random.uniform(0.4, 0.9)
        
        # Actual outcome (based on true win-rate, not model confidence)
        actual_outcome = 1 if np.random.rand() < true_win_rate else 0
        
        raw_scores.append(raw_score)
        actual_outcomes.append(actual_outcome)
    
    print(f"\nSimulated 100 trades:")
    print(f"  Average raw score: {np.mean(raw_scores):.2%}")
    print(f"  Actual win-rate: {np.mean(actual_outcomes):.2%}")
    print(f"  Gap (overconfidence): {np.mean(raw_scores) - np.mean(actual_outcomes):.2%}")
    
    # ===== Train Calibrator =====
    print(f"\n" + "=" * 80)
    print("TRAINING CALIBRATOR")
    print("=" * 80)
    
    cal = ProbabilityCalibrator(window_size=100, min_samples=30)
    
    # Add observations
    for raw, actual in zip(raw_scores, actual_outcomes):
        cal.add_observation(raw, actual)
    
    # Fit
    success = cal.fit()
    
    if success:
        # ===== Test Calibration =====
        print(f"\n" + "=" * 80)
        print("CALIBRATION TEST")
        print("=" * 80)
        
        test_scores = [0.50, 0.60, 0.70, 0.80, 0.90]
        
        print(f"\n{'Raw Score':<12} {'Calibrated':<12} {'Adjustment'}")
        print("-" * 40)
        
        for raw in test_scores:
            calibrated = cal.calibrate(raw)
            adjustment = calibrated - raw
            print(f"{raw:<12.2%} {calibrated:<12.2%} {adjustment:+.2%}")
        
        # ===== Calibration Curve =====
        print(f"\n" + "=" * 80)
        print("CALIBRATION CURVE")
        print("=" * 80)
        
        curve_df = cal.get_calibration_curve(n_bins=5)
        if not curve_df.empty:
            print(curve_df.to_string(index=False))
        
        print("\n✅ Calibrator implementation complete!")
        print(f"   ECE: {cal.calibration_metrics['ece']:.4f} (target <0.05)")
