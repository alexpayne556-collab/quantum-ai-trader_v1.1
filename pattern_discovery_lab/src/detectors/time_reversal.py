"""
Time-Reversal Asymmetry Detector

Measures nonlinearity via time-reversal asymmetry.

For linear Gaussian processes: X_t and X_{-t} have same statistics.
For nonlinear processes: asymmetry emerges.

Metric: E[(X_t - X_{t-1})^3] (third-order moment of increments)
Expected ~0 for linear, nonzero for nonlinear (e.g., GARCH).
"""

import numpy as np
from typing import Dict, Any
from ..detectors import BaseDetector


class TimeReversalAsymmetryDetector(BaseDetector):
    """
    Time-reversal asymmetry via third moment of increments.
    
    structure_score = |E[(dX)^3]| / (std(dX)^3)
    
    Higher score → more nonlinear structure.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: {
                'normalize': bool (default True)
            }
        """
        super().__init__(config)
        self.normalize = config.get('normalize', True)
    
    def detect(self, data: np.ndarray) -> float:
        """
        Compute time-reversal asymmetry.
        
        Args:
            data: Time-series
        
        Returns:
            structure_score: Normalized third moment of increments
        """
        # Compute increments
        increments = np.diff(data)
        
        if len(increments) < 3:
            return 0.0
        
        # Third moment
        third_moment = np.mean(increments ** 3)
        
        if self.normalize:
            # Normalize by std^3
            std = np.std(increments)
            if std == 0:
                return 0.0
            structure_score = abs(third_moment) / (std ** 3)
        else:
            structure_score = abs(third_moment)
        
        return structure_score
    
    def get_name(self) -> str:
        return "time_reversal_asymmetry"
