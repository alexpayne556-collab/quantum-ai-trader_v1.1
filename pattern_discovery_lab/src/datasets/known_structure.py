"""
Known-Structure World - Dataset Family 2

Generates time-series with DESIGNED, KNOWN structure.
Used to calibrate true positive rates and validate detectors.

Expected detector behavior:
- structure_score should be HIGH
- SHOULD pass robustness gates
- controls should destroy structure (time-shuffle, phase-randomization)
"""

import numpy as np
from typing import Dict, Any
from . import BaseDataset


class ARProcessDataset(BaseDataset):
    """
    AR(1) process with known coefficient.
    
    X_t = phi * X_{t-1} + epsilon_t
    
    Has linear dependence structure.
    """
    
    def __init__(self, seed: int, config: Dict[str, Any]):
        """
        Args:
            config: {
                'length': int (default 1000),
                'phi': float (default 0.7, must be in (-1, 1) for stationarity),
                'innovation_std': float (default 1.0)
            }
        """
        super().__init__(seed, config)
        self.length = config.get('length', 1000)
        self.phi = config.get('phi', 0.7)
        self.innovation_std = config.get('innovation_std', 1.0)
        
        if abs(self.phi) >= 1.0:
            raise ValueError(f"phi must be in (-1, 1) for stationarity. Got {self.phi}")
    
    def generate(self) -> np.ndarray:
        """Generate AR(1) process."""
        x = np.zeros(self.length)
        innovations = self.rng.normal(0, self.innovation_std, size=self.length)
        
        # Start from stationary distribution
        x[0] = self.rng.normal(0, self.innovation_std / np.sqrt(1 - self.phi**2))
        
        for t in range(1, self.length):
            x[t] = self.phi * x[t-1] + innovations[t]
        
        return x
    
    def get_true_structure(self) -> Dict[str, Any]:
        """Return known AR structure."""
        # Theoretical autocorrelation for AR(1)
        lags = np.arange(1, 11)
        theoretical_acf = self.phi ** lags
        
        return {
            'has_structure': True,
            'type': 'AR(1)',
            'phi': self.phi,
            'theoretical_autocorr': theoretical_acf.tolist(),
            'is_stationary': True,
            'is_predictable': True
        }
    
    def get_family(self) -> str:
        return "known_structure"
    
    def get_name(self) -> str:
        return f"ar1_phi{self.phi:.2f}"


class GARCHProcessDataset(BaseDataset):
    """
    GARCH(1,1) process with known parameters.
    
    r_t = sigma_t * z_t
    sigma_t^2 = omega + alpha * r_{t-1}^2 + beta * sigma_{t-1}^2
    
    Has volatility clustering (nonlinear dependence in squared returns).
    """
    
    def __init__(self, seed: int, config: Dict[str, Any]):
        """
        Args:
            config: {
                'length': int (default 1000),
                'omega': float (default 0.1),
                'alpha': float (default 0.1),
                'beta': float (default 0.85),
                Must have: omega > 0, alpha >= 0, beta >= 0, alpha + beta < 1
            }
        """
        super().__init__(seed, config)
        self.length = config.get('length', 1000)
        self.omega = config.get('omega', 0.1)
        self.alpha = config.get('alpha', 0.1)
        self.beta = config.get('beta', 0.85)
        
        if self.omega <= 0:
            raise ValueError("omega must be > 0")
        if self.alpha < 0 or self.beta < 0:
            raise ValueError("alpha and beta must be >= 0")
        if self.alpha + self.beta >= 1:
            raise ValueError("alpha + beta must be < 1 for stationarity")
    
    def generate(self) -> np.ndarray:
        """Generate GARCH(1,1) process."""
        r = np.zeros(self.length)
        sigma2 = np.zeros(self.length)
        
        # Start from unconditional variance
        sigma2[0] = self.omega / (1 - self.alpha - self.beta)
        
        z = self.rng.standard_normal(size=self.length)
        r[0] = np.sqrt(sigma2[0]) * z[0]
        
        for t in range(1, self.length):
            sigma2[t] = self.omega + self.alpha * r[t-1]**2 + self.beta * sigma2[t-1]
            r[t] = np.sqrt(sigma2[t]) * z[t]
        
        return r
    
    def get_true_structure(self) -> Dict[str, Any]:
        """Return known GARCH structure."""
        return {
            'has_structure': True,
            'type': 'GARCH(1,1)',
            'omega': self.omega,
            'alpha': self.alpha,
            'beta': self.beta,
            'has_vol_clustering': True,
            'returns_autocorr': [0.0] * 10,  # Returns are uncorrelated
            'squared_returns_autocorr': 'positive',  # Squared returns ARE correlated
            'is_predictable_volatility': True
        }
    
    def get_family(self) -> str:
        return "known_structure"
    
    def get_name(self) -> str:
        return f"garch_a{self.alpha:.2f}_b{self.beta:.2f}"


class RegimeShiftDataset(BaseDataset):
    """
    Two-regime process with known breakpoint.
    
    Regime 1: mean=0, std=1 (low volatility)
    Regime 2: mean=0, std=3 (high volatility)
    
    Has regime structure.
    """
    
    def __init__(self, seed: int, config: Dict[str, Any]):
        """
        Args:
            config: {
                'length': int (default 1000),
                'breakpoint': int (default 500, where regime changes),
                'regime1_std': float (default 1.0),
                'regime2_std': float (default 3.0)
            }
        """
        super().__init__(seed, config)
        self.length = config.get('length', 1000)
        self.breakpoint = config.get('breakpoint', 500)
        self.regime1_std = config.get('regime1_std', 1.0)
        self.regime2_std = config.get('regime2_std', 3.0)
        
        if self.breakpoint >= self.length or self.breakpoint <= 0:
            raise ValueError(f"breakpoint must be in (0, {self.length})")
    
    def generate(self) -> np.ndarray:
        """Generate regime-shift process."""
        x = np.zeros(self.length)
        
        # Regime 1
        x[:self.breakpoint] = self.rng.normal(0, self.regime1_std, size=self.breakpoint)
        
        # Regime 2
        x[self.breakpoint:] = self.rng.normal(0, self.regime2_std, size=self.length - self.breakpoint)
        
        return x
    
    def get_true_structure(self) -> Dict[str, Any]:
        """Return known regime structure."""
        return {
            'has_structure': True,
            'type': 'regime_shift',
            'num_regimes': 2,
            'breakpoint': self.breakpoint,
            'regime1_std': self.regime1_std,
            'regime2_std': self.regime2_std,
            'is_stationary': False,  # Globally non-stationary
            'is_detectable': True
        }
    
    def get_family(self) -> str:
        return "known_structure"
    
    def get_name(self) -> str:
        return f"regime_shift_bp{self.breakpoint}"


class MAProcessDataset(BaseDataset):
    """
    MA(1) process with known coefficient.
    
    X_t = epsilon_t + theta * epsilon_{t-1}
    
    Has short-term dependence.
    """
    
    def __init__(self, seed: int, config: Dict[str, Any]):
        """
        Args:
            config: {
                'length': int (default 1000),
                'theta': float (default 0.5),
                'innovation_std': float (default 1.0)
            }
        """
        super().__init__(seed, config)
        self.length = config.get('length', 1000)
        self.theta = config.get('theta', 0.5)
        self.innovation_std = config.get('innovation_std', 1.0)
    
    def generate(self) -> np.ndarray:
        """Generate MA(1) process."""
        innovations = self.rng.normal(0, self.innovation_std, size=self.length + 1)
        x = np.zeros(self.length)
        
        for t in range(self.length):
            x[t] = innovations[t] + self.theta * innovations[t-1]
        
        return x
    
    def get_true_structure(self) -> Dict[str, Any]:
        """Return known MA structure."""
        # Theoretical ACF for MA(1): rho_1 = theta / (1 + theta^2), rho_k = 0 for k > 1
        acf1 = self.theta / (1 + self.theta**2)
        theoretical_acf = [acf1] + [0.0] * 9
        
        return {
            'has_structure': True,
            'type': 'MA(1)',
            'theta': self.theta,
            'theoretical_autocorr': theoretical_acf,
            'memory_length': 1,  # Only 1-step dependence
            'is_stationary': True
        }
    
    def get_family(self) -> str:
        return "known_structure"
    
    def get_name(self) -> str:
        return f"ma1_theta{self.theta:.2f}"


def get_known_structure_dataset(name: str, seed: int, config: Dict[str, Any]) -> BaseDataset:
    """
    Factory function to create known-structure datasets.
    
    Args:
        name: 'ar1', 'garch', 'regime_shift', or 'ma1'
        seed: Random seed
        config: Dataset config
    
    Returns:
        BaseDataset instance
    """
    datasets = {
        'ar1': ARProcessDataset,
        'garch': GARCHProcessDataset,
        'regime_shift': RegimeShiftDataset,
        'ma1': MAProcessDataset
    }
    
    if name not in datasets:
        raise ValueError(f"Unknown known-structure dataset: {name}. Choose from {list(datasets.keys())}")
    
    return datasets[name](seed, config)
