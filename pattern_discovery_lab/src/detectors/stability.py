"""
Stability Detector - Window Shift Test

Tests if structure is stable across non-overlapping windows.

Splits data into windows, computes structure in each window,
measures consistency.

structure_score = 1 - coefficient_of_variation(window_scores)

Higher score → more stable structure across time.
"""

import numpy as np
from typing import Dict, Any
from ..detectors import BaseDetector


class WindowShiftDetector(BaseDetector):
    """
    Stability test via window shifts.
    
    Splits data into non-overlapping windows,
    computes a base detector's score in each window,
    measures stability (low CV = high stability).
    """
    
    def __init__(self, config: Dict[str, Any], base_detector: BaseDetector):
        """
        Args:
            config: {
                'n_windows': int (default 5, number of windows)
            }
            base_detector: Detector to apply in each window
        """
        super().__init__(config)
        self.n_windows = config.get('n_windows', 5)
        self.base_detector = base_detector
    
    def detect(self, data: np.ndarray) -> float:
        """
        Compute stability score.
        
        Args:
            data: Time-series
        
        Returns:
            structure_score: 1 - CV(window_scores)
                           Higher → more stable structure
        """
        n = len(data)
        window_size = n // self.n_windows
        
        if window_size < 10:
            # Too short for windowing
            return 0.0
        
        window_scores = []
        for i in range(self.n_windows):
            start = i * window_size
            end = start + window_size if i < self.n_windows - 1 else n
            window_data = data[start:end]
            
            if len(window_data) >= 10:
                score = self.base_detector.detect(window_data)
                window_scores.append(score)
        
        if len(window_scores) < 2:
            return 0.0
        
        # Coefficient of variation
        mean_score = np.mean(window_scores)
        std_score = np.std(window_scores)
        
        if mean_score == 0:
            return 0.0
        
        cv = std_score / mean_score
        
        # Stability score (lower CV = higher stability)
        stability_score = 1.0 / (1.0 + cv)
        
        return stability_score
    
    def get_name(self) -> str:
        base_name = self.base_detector.get_name()
        return f"stability_window_shift_{base_name}"
