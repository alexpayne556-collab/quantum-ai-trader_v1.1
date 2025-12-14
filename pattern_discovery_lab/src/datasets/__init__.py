"""
Pattern Discovery Lab - Dataset Base Classes
Defines common interface for all dataset generators.
"""

import numpy as np
from typing import Dict, Any, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
import json
import hashlib


@dataclass
class DatasetMetadata:
    """Metadata for reproducibility and versioning."""
    family: str  # pure_noise, known_structure, finance_null, real_data
    name: str
    version: str
    seed: int
    length: int
    config: Dict[str, Any]
    generated_at: str
    hash: str  # Hash of data for integrity


class BaseDataset(ABC):
    """
    Abstract base class for all datasets.
    
    All datasets must be:
    - Reproducible (same seed → same data)
    - Versioned (hash + metadata)
    - Documented (config saved alongside data)
    """
    
    def __init__(self, seed: int, config: Dict[str, Any]):
        """
        Initialize dataset generator.
        
        Args:
            seed: Random seed for reproducibility
            config: Configuration dict (length, params, etc.)
        """
        self.seed = seed
        self.config = config
        self.rng = np.random.RandomState(seed)
        self.metadata: Optional[DatasetMetadata] = None
        
    @abstractmethod
    def generate(self) -> np.ndarray:
        """
        Generate time-series data.
        
        Returns:
            1D numpy array of shape (length,)
        """
        pass
    
    @abstractmethod
    def get_true_structure(self) -> Dict[str, Any]:
        """
        Return ground-truth structure (for known-structure datasets).
        
        Returns:
            Dict with structure info (e.g., {'ar_coef': 0.7, 'garch_alpha': 0.1})
            For pure noise: return {}
        """
        pass
    
    def compute_hash(self, data: np.ndarray) -> str:
        """Compute hash of data for integrity checks."""
        return hashlib.sha256(data.tobytes()).hexdigest()[:16]
    
    def save_metadata(self, data: np.ndarray, output_path: str):
        """Save metadata JSON alongside dataset."""
        from datetime import datetime
        
        self.metadata = DatasetMetadata(
            family=self.get_family(),
            name=self.get_name(),
            version=self.get_version(),
            seed=self.seed,
            length=len(data),
            config=self.config,
            generated_at=datetime.utcnow().isoformat(),
            hash=self.compute_hash(data)
        )
        
        with open(output_path, 'w') as f:
            json.dump(self.metadata.__dict__, f, indent=2)
    
    @abstractmethod
    def get_family(self) -> str:
        """Return dataset family name."""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Return dataset name."""
        pass
    
    def get_version(self) -> str:
        """Return dataset version."""
        return "0.1.0"


class MultiSeriesDataset(ABC):
    """
    Base class for multi-series datasets (for cross-series structure detection).
    """
    
    def __init__(self, seed: int, config: Dict[str, Any]):
        self.seed = seed
        self.config = config
        self.rng = np.random.RandomState(seed)
        
    @abstractmethod
    def generate(self) -> Dict[str, np.ndarray]:
        """
        Generate multiple time-series.
        
        Returns:
            Dict mapping series_id -> 1D numpy array
        """
        pass
    
    @abstractmethod
    def get_true_cross_structure(self) -> Dict[str, Any]:
        """
        Return ground-truth cross-series structure.
        
        Returns:
            Dict with cross-structure info (e.g., {'lead_lag': {'A': 'B', 'lag': 2}})
        """
        pass
