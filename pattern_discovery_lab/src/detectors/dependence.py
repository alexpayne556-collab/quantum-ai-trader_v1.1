"""
Dependence Detectors - Linear and Nonlinear

1. Linear: Autocorrelation decay (sum of squared ACF)
2. Nonlinear: Mutual information at lag 1
"""

import numpy as np
from typing import Dict, Any
from scipy import stats
from ..detectors import BaseDetector


class AutocorrelationDetector(BaseDetector):
    """
    Linear dependence via autocorrelation function.
    
    structure_score = sum of squared autocorrelations
    
    Higher score → stronger linear dependence.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: {
                'max_lag': int (default 20, maximum lag to compute)
            }
        """
        super().__init__(config)
        self.max_lag = config.get('max_lag', 20)
    
    def autocorrelation(self, data: np.ndarray, lag: int) -> float:
        """
        Compute autocorrelation at given lag.
        
        Args:
            data: Time-series
            lag: Lag
        
        Returns:
            Autocorrelation coefficient
        """
        n = len(data)
        if lag >= n:
            return 0.0
        
        mean = np.mean(data)
        c0 = np.sum((data - mean) ** 2) / n
        
        if c0 == 0:
            return 0.0
        
        c_lag = np.sum((data[:-lag] - mean) * (data[lag:] - mean)) / n
        return c_lag / c0
    
    def detect(self, data: np.ndarray) -> float:
        """
        Compute autocorrelation structure score.
        
        Args:
            data: Time-series
        
        Returns:
            structure_score: Sum of squared autocorrelations
        """
        acf_values = []
        for lag in range(1, min(self.max_lag + 1, len(data))):
            acf = self.autocorrelation(data, lag)
            acf_values.append(acf)
        
        # Sum of squared ACF (higher = more linear dependence)
        structure_score = np.sum(np.array(acf_values) ** 2)
        
        return structure_score
    
    def get_name(self) -> str:
        return "autocorrelation_dependence"


class MutualInformationDetector(BaseDetector):
    """
    Nonlinear dependence via mutual information.
    
    Computes MI between X_t and X_{t-lag}.
    
    Higher MI → stronger (possibly nonlinear) dependence.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: {
                'lag': int (default 1),
                'bins': int (default 10, for histogram estimation)
            }
        """
        super().__init__(config)
        self.lag = config.get('lag', 1)
        self.bins = config.get('bins', 10)
    
    def mutual_information(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Estimate mutual information via histogram.
        
        MI(X; Y) = sum p(x,y) log(p(x,y) / (p(x)p(y)))
        
        Args:
            x: First variable
            y: Second variable
        
        Returns:
            Mutual information (nats)
        """
        # Joint histogram
        hist_2d, x_edges, y_edges = np.histogram2d(x, y, bins=self.bins)
        
        # Normalize to probabilities
        pxy = hist_2d / np.sum(hist_2d)
        
        # Marginal probabilities
        px = np.sum(pxy, axis=1)
        py = np.sum(pxy, axis=0)
        
        # Outer product of marginals
        px_py = px[:, None] * py[None, :]
        
        # Mutual information (avoid log(0))
        nonzero = pxy > 0
        mi = np.sum(pxy[nonzero] * np.log(pxy[nonzero] / px_py[nonzero]))
        
        return mi
    
    def detect(self, data: np.ndarray) -> float:
        """
        Compute MI structure score.
        
        Args:
            data: Time-series
        
        Returns:
            structure_score: Mutual information at lag
        """
        if len(data) < self.lag + 1:
            return 0.0
        
        x = data[:-self.lag]
        y = data[self.lag:]
        
        mi = self.mutual_information(x, y)
        
        return mi
    
    def get_name(self) -> str:
        return f"mutual_information_lag{self.lag}"
