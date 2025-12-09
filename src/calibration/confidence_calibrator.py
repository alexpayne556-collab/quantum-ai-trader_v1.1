"""
Confidence Calibrator
Maps raw model confidence to actual win-rate using regime-aware calibration curves.

Research: Models often output 70% confidence but only win 55% of time.
Solution: Per-regime calibration curves (Platt scaling, isotonic regression, or neural net).
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy.optimize import minimize
from sklearn.isotonic import IsotonicRegression


@dataclass
class CalibrationConfig:
    """Configuration for confidence calibration"""
    method: str = "isotonic"  # "platt", "isotonic", "neural"
    min_samples_per_regime: int = 50  # Minimum trades to calibrate
    update_frequency: int = 20  # Recalibrate every N trades
    calibration_window: int = 200  # Use last N trades for calibration


class ConfidenceCalibrator:
    """
    Calibrate raw confidence scores to actual win-rates.
    
    Problem: Model says 70% → actual win-rate 55% (overconfident)
    Solution: Learn mapping function per regime: predicted → actual
    
    Usage:
        calibrator = ConfidenceCalibrator(config=CalibrationConfig())
        calibrator.fit(predicted_confidences, actual_outcomes, regimes)
        calibrated = calibrator.calibrate(raw_confidence=0.70, regime='BULL_LOW_VOL')
        # Returns: 0.63 (calibrated down)
    """
    
    def __init__(self, config: Optional[CalibrationConfig] = None):
        self.config = config or CalibrationConfig()
        
        # Calibration models per regime
        self.calibrators = {}  # {regime: calibration_model}
        
        # Historical data for calibration
        self.history = {
            'predicted': [],
            'actual': [],
            'regime': [],
        }
        
        # Calibration metadata
        self.last_update_count = 0
        self.calibration_scores = {}  # {regime: calibration_score}
    
    def fit(self, predicted: np.ndarray, actual: np.ndarray, regimes: np.ndarray):
        """
        Fit calibration curves from historical data.
        
        Args:
            predicted: Raw confidence scores [0, 1]
            actual: Actual outcomes (0 or 1)
            regimes: Regime labels for each prediction
        """
        unique_regimes = np.unique(regimes)
        
        for regime in unique_regimes:
            mask = regimes == regime
            regime_pred = predicted[mask]
            regime_actual = actual[mask]
            
            if len(regime_pred) < self.config.min_samples_per_regime:
                print(f"⚠️  Insufficient samples for {regime}: {len(regime_pred)} < {self.config.min_samples_per_regime}")
                continue
            
            # Fit calibration model
            if self.config.method == "platt":
                self.calibrators[regime] = self._fit_platt_scaling(regime_pred, regime_actual)
            elif self.config.method == "isotonic":
                self.calibrators[regime] = self._fit_isotonic_regression(regime_pred, regime_actual)
            elif self.config.method == "neural":
                self.calibrators[regime] = self._fit_neural_calibrator(regime_pred, regime_actual)
            else:
                raise ValueError(f"Unknown calibration method: {self.config.method}")
            
            # Calculate calibration score (how well calibrated)
            calibrated_pred = self.calibrators[regime].predict(regime_pred.reshape(-1, 1))
            self.calibration_scores[regime] = self._calculate_calibration_score(calibrated_pred, regime_actual)
        
        print(f"✅ Calibrated {len(self.calibrators)} regimes")
        for regime, score in self.calibration_scores.items():
            print(f"   {regime}: calibration error = {score:.3f}")
    
    def calibrate(self, raw_confidence: float, regime: str, sector: Optional[str] = None) -> float:
        """
        Calibrate raw confidence to actual win-rate.
        
        Args:
            raw_confidence: Model's raw confidence [0, 1]
            regime: Current market regime
            sector: Optional sector for sector-specific adjustment
        
        Returns:
            Calibrated confidence [0, 1]
        """
        if regime not in self.calibrators:
            # Fallback: no calibration available, return raw
            return raw_confidence
        
        calibrator = self.calibrators[regime]
        calibrated = calibrator.predict(np.array([[raw_confidence]]))[0]
        
        # Clip to [0, 1]
        calibrated = np.clip(calibrated, 0.0, 1.0)
        
        # Optional: sector-specific adjustment
        if sector is not None:
            calibrated = self._adjust_for_sector(calibrated, sector)
        
        return calibrated
    
    def update(self, predicted: float, actual: int, regime: str):
        """
        Update calibration with new trade result.
        
        Args:
            predicted: Predicted confidence
            actual: Actual outcome (0 or 1)
            regime: Regime at time of prediction
        """
        self.history['predicted'].append(predicted)
        self.history['actual'].append(actual)
        self.history['regime'].append(regime)
        
        # Check if time to recalibrate
        if len(self.history['predicted']) - self.last_update_count >= self.config.update_frequency:
            self._recalibrate()
            self.last_update_count = len(self.history['predicted'])
    
    # ========== CALIBRATION METHODS ==========
    
    @staticmethod
    def _fit_platt_scaling(predicted: np.ndarray, actual: np.ndarray):
        """
        Fit Platt scaling (logistic regression calibration).
        
        Model: P_calibrated = 1 / (1 + exp(A * P_raw + B))
        Find A, B that minimize log-loss
        """
        from sklearn.linear_model import LogisticRegression
        
        model = LogisticRegression(penalty=None, solver='lbfgs')
        model.fit(predicted.reshape(-1, 1), actual)
        return model
    
    @staticmethod
    def _fit_isotonic_regression(predicted: np.ndarray, actual: np.ndarray):
        """
        Fit isotonic regression (non-parametric, monotonic).
        
        Advantage: No assumptions about calibration curve shape
        Disadvantage: Can overfit with small samples
        """
        model = IsotonicRegression(out_of_bounds='clip')
        model.fit(predicted, actual)
        
        # Wrap in sklearn-compatible interface
        class IsotonicWrapper:
            def __init__(self, iso_model):
                self.iso_model = iso_model
            
            def predict(self, X):
                return self.iso_model.predict(X.flatten())
        
        return IsotonicWrapper(model)
    
    @staticmethod
    def _fit_neural_calibrator(predicted: np.ndarray, actual: np.ndarray):
        """
        Fit neural network calibrator (most flexible).
        
        Architecture: 1 → 16 → 8 → 1 (small network to avoid overfitting)
        """
        # TODO: Implement neural calibrator
        # For now, fallback to Platt scaling
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(penalty=None, solver='lbfgs')
        model.fit(predicted.reshape(-1, 1), actual)
        return model
    
    # ========== CALIBRATION EVALUATION ==========
    
    @staticmethod
    def _calculate_calibration_score(predicted: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
        """
        Calculate Expected Calibration Error (ECE).
        
        Perfect calibration: predicted = actual win-rate in each bin
        ECE = average absolute difference between predicted and actual
        
        Returns: ECE (lower is better, 0 = perfect calibration)
        """
        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(predicted, bins) - 1
        
        ece = 0.0
        for i in range(n_bins):
            mask = bin_indices == i
            if mask.sum() == 0:
                continue
            
            avg_predicted = predicted[mask].mean()
            avg_actual = actual[mask].mean()
            bin_weight = mask.sum() / len(predicted)
            
            ece += bin_weight * abs(avg_predicted - avg_actual)
        
        return ece
    
    def evaluate_calibration(self, predicted: np.ndarray, actual: np.ndarray, 
                            regime: str) -> Dict[str, float]:
        """
        Evaluate calibration quality for a regime.
        
        Returns:
            {
                'ece': Expected Calibration Error,
                'brier_score': Brier score (MSE of probabilities),
                'log_loss': Log-loss,
            }
        """
        calibrated = self.calibrate(predicted, regime)
        
        # Expected Calibration Error
        ece = self._calculate_calibration_score(calibrated, actual)
        
        # Brier Score (MSE)
        brier = np.mean((calibrated - actual) ** 2)
        
        # Log-Loss
        epsilon = 1e-15
        calibrated_clipped = np.clip(calibrated, epsilon, 1 - epsilon)
        log_loss = -np.mean(actual * np.log(calibrated_clipped) + (1 - actual) * np.log(1 - calibrated_clipped))
        
        return {
            'ece': ece,
            'brier_score': brier,
            'log_loss': log_loss,
        }
    
    # ========== SECTOR ADJUSTMENT ==========
    
    @staticmethod
    def _adjust_for_sector(calibrated: float, sector: str) -> float:
        """
        Apply sector-specific calibration adjustment.
        
        Research: AI sector (0.82 Sharpe) → boost confidence
                 Quantum sector (0.68 Sharpe) → reduce confidence
        """
        sector_multipliers = {
            'AI_INFRA': 1.05,  # Historically more accurate
            'QUANTUM': 0.95,   # Historically less accurate
            'ROBOTAXI': 0.92,
            'HEALTHCARE': 0.98,
            'ENERGY': 1.00,
        }
        
        multiplier = sector_multipliers.get(sector, 1.0)
        adjusted = calibrated * multiplier
        return np.clip(adjusted, 0.0, 1.0)
    
    # ========== RECALIBRATION ==========
    
    def _recalibrate(self):
        """Recalibrate using recent history window"""
        window = self.config.calibration_window
        
        predicted = np.array(self.history['predicted'][-window:])
        actual = np.array(self.history['actual'][-window:])
        regimes = np.array(self.history['regime'][-window:])
        
        if len(predicted) >= self.config.min_samples_per_regime:
            self.fit(predicted, actual, regimes)
            print(f"🔄 Recalibrated with {len(predicted)} recent trades")
    
    # ========== VISUALIZATION ==========
    
    def plot_calibration_curve(self, regime: str, predicted: np.ndarray, actual: np.ndarray):
        """
        Plot calibration curve: predicted confidence vs actual win-rate.
        
        Perfect calibration: diagonal line (y = x)
        """
        import matplotlib.pyplot as plt
        
        # Bin predictions
        n_bins = 10
        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(predicted, bins) - 1
        
        bin_predicted = []
        bin_actual = []
        
        for i in range(n_bins):
            mask = bin_indices == i
            if mask.sum() == 0:
                continue
            bin_predicted.append(predicted[mask].mean())
            bin_actual.append(actual[mask].mean())
        
        plt.figure(figsize=(8, 6))
        plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
        plt.scatter(bin_predicted, bin_actual, s=100, alpha=0.7, label=f'{regime} (ECE={self.calibration_scores.get(regime, 0):.3f})')
        plt.xlabel('Predicted Confidence')
        plt.ylabel('Actual Win-Rate')
        plt.title(f'Calibration Curve: {regime}')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()


# ========== MULTI-REGIME CALIBRATOR ==========

class MultiRegimeCalibrator:
    """
    Manage calibration across all 12 regimes + fallback for unseen regimes.
    
    Usage:
        calibrator = MultiRegimeCalibrator()
        calibrator.fit_all(historical_data)
        calibrated = calibrator.calibrate(raw=0.70, regime='BULL_LOW_VOL', sector='AI')
    """
    
    def __init__(self, config: Optional[CalibrationConfig] = None):
        self.config = config or CalibrationConfig()
        self.calibrators = {}  # {regime: ConfidenceCalibrator}
        self.fallback_calibrator = ConfidenceCalibrator(config)
    
    def fit_all(self, historical_predictions: pd.DataFrame):
        """
        Fit calibrators for all regimes from historical data.
        
        Args:
            historical_predictions: DataFrame with columns:
                ['predicted_confidence', 'actual_outcome', 'regime', 'sector']
        """
        for regime in historical_predictions['regime'].unique():
            regime_data = historical_predictions[historical_predictions['regime'] == regime]
            
            if len(regime_data) < self.config.min_samples_per_regime:
                print(f"⚠️  Skipping {regime}: only {len(regime_data)} samples")
                continue
            
            calibrator = ConfidenceCalibrator(self.config)
            calibrator.fit(
                regime_data['predicted_confidence'].values,
                regime_data['actual_outcome'].values,
                regime_data['regime'].values
            )
            self.calibrators[regime] = calibrator
        
        # Fit fallback calibrator on all data
        self.fallback_calibrator.fit(
            historical_predictions['predicted_confidence'].values,
            historical_predictions['actual_outcome'].values,
            historical_predictions['regime'].values
        )
    
    def calibrate(self, raw: float, regime: str, sector: Optional[str] = None) -> float:
        """Calibrate using regime-specific calibrator or fallback"""
        if regime in self.calibrators:
            return self.calibrators[regime].calibrate(raw, regime, sector)
        else:
            return self.fallback_calibrator.calibrate(raw, regime, sector)
