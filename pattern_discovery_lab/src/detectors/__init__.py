"""
Pattern Discovery Lab - Base Detector Interface

All structure detectors must inherit from BaseDetector and implement:
- detect(): Compute structure score
- get_metadata(): Return detector configuration

Detectors output:
- structure_score: Numeric measure of detected structure
- metadata: Dict with detector params, window, etc.

No predictions. No forward returns. Only structure detection.
"""

import numpy as np
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class DetectorResult:
    """
    Standard output format for all detectors.
    
    Attributes:
        detector_name: Name of the detector
        structure_score: Numeric score (higher = more structure detected)
        metadata: Dict with detector params, window size, etc.
        execution_time: Time taken to compute (seconds)
    """
    detector_name: str
    structure_score: float
    metadata: Dict[str, Any]
    execution_time: float


class BaseDetector(ABC):
    """
    Abstract base class for all structure detectors.
    
    Detectors must be:
    - Deterministic (same input → same output)
    - Documented (clear metadata about params)
    - Calibrated (tested on pure noise, known structure, finance null)
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize detector with configuration.
        
        Args:
            config: Detector parameters (e.g., max_lag, n_bins, etc.)
        """
        self.config = config
        
    @abstractmethod
    def detect(self, data: np.ndarray) -> float:
        """
        Detect structure in time-series data.
        
        Args:
            data: 1D numpy array of time-series observations
        
        Returns:
            structure_score: Numeric score (higher = more structure)
        
        Notes:
            - Must be deterministic
            - Must handle edge cases (short series, constant values, etc.)
            - Should normalize score to comparable range when possible
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Return detector name."""
        pass
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Return detector metadata for reproducibility.
        
        Returns:
            Dict with config, version, description
        """
        return {
            'name': self.get_name(),
            'version': '0.1.0',
            'config': self.config,
            'description': self.__doc__
        }
    
    def validate_input(self, data: np.ndarray):
        """
        Validate input data.
        
        Args:
            data: Input time-series
        
        Raises:
            ValueError: If data is invalid
        """
        if not isinstance(data, np.ndarray):
            raise ValueError("Input must be numpy array")
        
        if data.ndim != 1:
            raise ValueError(f"Input must be 1D, got {data.ndim}D")
        
        if len(data) < 10:
            raise ValueError(f"Input too short: {len(data)} points (need >= 10)")
        
        if not np.isfinite(data).all():
            raise ValueError("Input contains inf or nan")
    
    def run(self, data: np.ndarray) -> DetectorResult:
        """
        Run detector with timing and validation.
        
        Args:
            data: Input time-series
        
        Returns:
            DetectorResult with score, metadata, timing
        """
        import time
        
        self.validate_input(data)
        
        start_time = time.time()
        structure_score = self.detect(data)
        execution_time = time.time() - start_time
        
        return DetectorResult(
            detector_name=self.get_name(),
            structure_score=structure_score,
            metadata=self.get_metadata(),
            execution_time=execution_time
        )


class MultiSeriesDetector(ABC):
    """
    Base class for detectors that analyze multiple time-series.
    
    Used for cross-series structure detection (e.g., Granger causality).
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    @abstractmethod
    def detect(self, data: Dict[str, np.ndarray]) -> float:
        """
        Detect cross-series structure.
        
        Args:
            data: Dict mapping series_id -> 1D numpy array
        
        Returns:
            structure_score: Numeric score (higher = more cross-structure)
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Return detector name."""
        pass
