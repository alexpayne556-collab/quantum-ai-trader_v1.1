"""
Regime Structure Detector - Changepoint Detection

Uses Ruptures library to detect regime changes.

structure_score = number of changepoints detected
"""

import numpy as np
from typing import Dict, Any
from ..detectors import BaseDetector


class ChangepointDetector(BaseDetector):
    """
    Changepoint detection via Ruptures library.
    
    Detects regime shifts (breakpoints) in time-series.
    
    structure_score = number of changepoints detected
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: {
                'model': str (default 'rbf', kernel for Ruptures),
                'min_size': int (default 50, minimum segment length),
                'penalty': float (default None, auto-select via BIC)
            }
        """
        super().__init__(config)
        self.model = config.get('model', 'rbf')
        self.min_size = config.get('min_size', 50)
        self.penalty = config.get('penalty', None)
    
    def detect(self, data: np.ndarray) -> float:
        """
        Detect changepoints.
        
        Args:
            data: Time-series
        
        Returns:
            structure_score: Number of changepoints detected
        """
        try:
            import ruptures as rpt
        except ImportError:
            raise ImportError("ruptures library required. Install: pip install ruptures")
        
        # Reshape for ruptures (expects 2D)
        signal = data.reshape(-1, 1)
        
        # Choose algorithm
        if self.model == 'rbf':
            algo = rpt.Pelt(model='rbf', min_size=self.min_size)
        elif self.model == 'l2':
            algo = rpt.Pelt(model='l2', min_size=self.min_size)
        else:
            algo = rpt.Pelt(model=self.model, min_size=self.min_size)
        
        # Fit
        algo.fit(signal)
        
        # Detect (auto-select penalty if not provided)
        if self.penalty is None:
            # Use BIC-like penalty
            penalty = np.log(len(data)) * signal.shape[1]
        else:
            penalty = self.penalty
        
        breakpoints = algo.predict(pen=penalty)
        
        # Number of changepoints (excluding endpoint)
        n_changepoints = len(breakpoints) - 1
        
        return float(n_changepoints)
    
    def get_name(self) -> str:
        return f"changepoint_{self.model}"
