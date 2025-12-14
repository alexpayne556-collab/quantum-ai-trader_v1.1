"""
Pure Noise World - Dataset Family 1

Generates time-series with NO structure by design.
Used to calibrate false positive rates.

Expected detector behavior:
- structure_score should be low
- should NOT pass robustness gates
- controls should show no difference
"""

import numpy as np
from typing import Dict, Any
from . import BaseDataset


class WhiteNoiseDataset(BaseDataset):
    """
    Pure white noise: IID Gaussian draws.
    NO autocorrelation, NO regime changes, NO structure.
    """
    
    def __init__(self, seed: int, config: Dict[str, Any]):
        """
        Args:
            config: {
                'length': int (default 1000),
                'mean': float (default 0.0),
                'std': float (default 1.0)
            }
        """
        super().__init__(seed, config)
        self.length = config.get('length', 1000)
        self.mean = config.get('mean', 0.0)
        self.std = config.get('std', 1.0)
    
    def generate(self) -> np.ndarray:
        """Generate pure white noise."""
        return self.rng.normal(self.mean, self.std, size=self.length)
    
    def get_true_structure(self) -> Dict[str, Any]:
        """No structure by design."""
        return {
            'has_structure': False,
            'description': 'IID Gaussian white noise',
            'expected_autocorr': [0.0] * 10,  # Should be ~0 at all lags
            'expected_regime_count': 1  # Single regime (stationary)
        }
    
    def get_family(self) -> str:
        return "pure_noise"
    
    def get_name(self) -> str:
        return "white_noise"


class RandomWalkDataset(BaseDataset):
    """
    Random walk: cumsum of white noise.
    
    Has apparent "trends" but they are spurious.
    Tests if detectors confuse non-stationarity with structure.
    """
    
    def __init__(self, seed: int, config: Dict[str, Any]):
        """
        Args:
            config: {
                'length': int (default 1000),
                'innovation_std': float (default 1.0),
                'drift': float (default 0.0)
            }
        """
        super().__init__(seed, config)
        self.length = config.get('length', 1000)
        self.innovation_std = config.get('innovation_std', 1.0)
        self.drift = config.get('drift', 0.0)
    
    def generate(self) -> np.ndarray:
        """Generate random walk."""
        innovations = self.rng.normal(self.drift, self.innovation_std, size=self.length)
        return np.cumsum(innovations)
    
    def get_true_structure(self) -> Dict[str, Any]:
        """Random walk has no exploitable structure."""
        return {
            'has_structure': False,
            'description': 'Random walk (cumsum of white noise)',
            'is_stationary': False,
            'has_predictable_increments': False,
            'expected_autocorr_returns': [0.0] * 10  # Returns are white noise
        }
    
    def get_family(self) -> str:
        return "pure_noise"
    
    def get_name(self) -> str:
        return "random_walk"


class IIDUniformDataset(BaseDataset):
    """
    IID uniform draws.
    
    Different distribution than Gaussian, tests if detectors are
    distribution-agnostic.
    """
    
    def __init__(self, seed: int, config: Dict[str, Any]):
        """
        Args:
            config: {
                'length': int (default 1000),
                'low': float (default 0.0),
                'high': float (default 1.0)
            }
        """
        super().__init__(seed, config)
        self.length = config.get('length', 1000)
        self.low = config.get('low', 0.0)
        self.high = config.get('high', 1.0)
    
    def generate(self) -> np.ndarray:
        """Generate IID uniform."""
        return self.rng.uniform(self.low, self.high, size=self.length)
    
    def get_true_structure(self) -> Dict[str, Any]:
        """No structure."""
        return {
            'has_structure': False,
            'description': 'IID uniform draws',
            'distribution': 'uniform',
            'expected_autocorr': [0.0] * 10
        }
    
    def get_family(self) -> str:
        return "pure_noise"
    
    def get_name(self) -> str:
        return "iid_uniform"


def get_pure_noise_dataset(name: str, seed: int, config: Dict[str, Any]) -> BaseDataset:
    """
    Factory function to create pure noise datasets.
    
    Args:
        name: 'white_noise', 'random_walk', or 'iid_uniform'
        seed: Random seed
        config: Dataset config
    
    Returns:
        BaseDataset instance
    """
    datasets = {
        'white_noise': WhiteNoiseDataset,
        'random_walk': RandomWalkDataset,
        'iid_uniform': IIDUniformDataset
    }
    
    if name not in datasets:
        raise ValueError(f"Unknown pure noise dataset: {name}. Choose from {list(datasets.keys())}")
    
    return datasets[name](seed, config)
