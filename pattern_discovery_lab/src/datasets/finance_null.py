"""
Finance-Like Null World - Dataset Family 3

Generates time-series with "finance texture" but NO exploitable structure.

Finance texture includes:
- Volatility clustering (GARCH)
- Fat tails (Student-t innovations)
- Occasional jumps
- Autocorrelation in volatility (but NOT in returns)

This is the HARDEST test: does detector confuse "finance stylized facts"
with "exploitable structure"?

Expected detector behavior:
- MAY detect texture (volatility clustering, fat tails)
- Should NOT pass robustness gates
- Controls should show texture is not exploitable
"""

import numpy as np
from typing import Dict, Any
from scipy import stats
from . import BaseDataset


class GARCHWithJumpsDataset(BaseDataset):
    """
    GARCH + fat tails + occasional jumps.
    
    Looks like real financial returns, but has NO exploitable edge.
    """
    
    def __init__(self, seed: int, config: Dict[str, Any]):
        """
        Args:
            config: {
                'length': int (default 1000),
                'omega': float (default 0.1),
                'alpha': float (default 0.1),
                'beta': float (default 0.85),
                'dof': int (default 5, degrees of freedom for Student-t),
                'jump_prob': float (default 0.01, probability of jump per period),
                'jump_std': float (default 5.0, jump size std)
            }
        """
        super().__init__(seed, config)
        self.length = config.get('length', 1000)
        self.omega = config.get('omega', 0.1)
        self.alpha = config.get('alpha', 0.1)
        self.beta = config.get('beta', 0.85)
        self.dof = config.get('dof', 5)
        self.jump_prob = config.get('jump_prob', 0.01)
        self.jump_std = config.get('jump_std', 5.0)
    
    def generate(self) -> np.ndarray:
        """Generate GARCH + jumps + fat tails."""
        r = np.zeros(self.length)
        sigma2 = np.zeros(self.length)
        
        # Start from unconditional variance
        sigma2[0] = self.omega / (1 - self.alpha - self.beta)
        
        # Student-t innovations (fat tails)
        z = stats.t.rvs(df=self.dof, size=self.length, random_state=self.rng)
        z = z / np.sqrt(self.dof / (self.dof - 2))  # Standardize to variance=1
        
        r[0] = np.sqrt(sigma2[0]) * z[0]
        
        for t in range(1, self.length):
            # GARCH evolution
            sigma2[t] = self.omega + self.alpha * r[t-1]**2 + self.beta * sigma2[t-1]
            
            # Base return
            r[t] = np.sqrt(sigma2[t]) * z[t]
            
            # Occasional jumps
            if self.rng.rand() < self.jump_prob:
                jump = self.rng.normal(0, self.jump_std)
                r[t] += jump
        
        return r
    
    def get_true_structure(self) -> Dict[str, Any]:
        """Return known finance texture (but NO exploitable structure)."""
        return {
            'has_exploitable_structure': False,
            'has_finance_texture': True,
            'texture_components': [
                'volatility_clustering',
                'fat_tails',
                'occasional_jumps'
            ],
            'returns_autocorr': [0.0] * 10,  # Returns are UNPREDICTABLE
            'squared_returns_autocorr': 'positive',  # Vol clustering
            'distribution': 'student_t',
            'dof': self.dof,
            'jump_frequency': self.jump_prob,
            'is_predictable_returns': False,
            'is_predictable_volatility': True  # But can't trade on vol alone
        }
    
    def get_family(self) -> str:
        return "finance_null"
    
    def get_name(self) -> str:
        return "garch_jumps_fattails"


class StochasticVolatilityDataset(BaseDataset):
    """
    Stochastic volatility model.
    
    r_t = exp(h_t / 2) * epsilon_t
    h_t = mu + phi * (h_{t-1} - mu) + sigma_eta * eta_t
    
    Has volatility clustering but unpredictable returns.
    """
    
    def __init__(self, seed: int, config: Dict[str, Any]):
        """
        Args:
            config: {
                'length': int (default 1000),
                'mu': float (default 0.0, long-run log-vol),
                'phi': float (default 0.95, vol persistence),
                'sigma_eta': float (default 0.2, vol-of-vol)
            }
        """
        super().__init__(seed, config)
        self.length = config.get('length', 1000)
        self.mu = config.get('mu', 0.0)
        self.phi = config.get('phi', 0.95)
        self.sigma_eta = config.get('sigma_eta', 0.2)
        
        if abs(self.phi) >= 1.0:
            raise ValueError("phi must be in (-1, 1) for stationarity")
    
    def generate(self) -> np.ndarray:
        """Generate stochastic volatility process."""
        r = np.zeros(self.length)
        h = np.zeros(self.length)
        
        # Start from stationary distribution
        h[0] = self.rng.normal(self.mu, self.sigma_eta / np.sqrt(1 - self.phi**2))
        
        epsilon = self.rng.standard_normal(size=self.length)
        eta = self.rng.normal(0, self.sigma_eta, size=self.length)
        
        r[0] = np.exp(h[0] / 2) * epsilon[0]
        
        for t in range(1, self.length):
            h[t] = self.mu + self.phi * (h[t-1] - self.mu) + eta[t]
            r[t] = np.exp(h[t] / 2) * epsilon[t]
        
        return r
    
    def get_true_structure(self) -> Dict[str, Any]:
        """Return known SV texture (but NO exploitable structure)."""
        return {
            'has_exploitable_structure': False,
            'has_finance_texture': True,
            'type': 'stochastic_volatility',
            'mu': self.mu,
            'phi': self.phi,
            'sigma_eta': self.sigma_eta,
            'returns_autocorr': [0.0] * 10,
            'vol_is_persistent': True,
            'is_predictable_returns': False
        }
    
    def get_family(self) -> str:
        return "finance_null"
    
    def get_name(self) -> str:
        return "stochastic_volatility"


class LevyJumpDataset(BaseDataset):
    """
    Compound Poisson jumps + diffusion.
    
    Simulates rare large moves (crashes/rallies) but unpredictable timing.
    """
    
    def __init__(self, seed: int, config: Dict[str, Any]):
        """
        Args:
            config: {
                'length': int (default 1000),
                'lambda_jumps': float (default 0.05, jump intensity),
                'jump_mean': float (default 0.0),
                'jump_std': float (default 2.0),
                'diffusion_std': float (default 1.0)
            }
        """
        super().__init__(seed, config)
        self.length = config.get('length', 1000)
        self.lambda_jumps = config.get('lambda_jumps', 0.05)
        self.jump_mean = config.get('jump_mean', 0.0)
        self.jump_std = config.get('jump_std', 2.0)
        self.diffusion_std = config.get('diffusion_std', 1.0)
    
    def generate(self) -> np.ndarray:
        """Generate Levy jump process."""
        r = np.zeros(self.length)
        
        # Diffusion component
        diffusion = self.rng.normal(0, self.diffusion_std, size=self.length)
        
        # Jump component
        for t in range(self.length):
            r[t] = diffusion[t]
            
            # Poisson jump arrivals
            n_jumps = self.rng.poisson(self.lambda_jumps)
            if n_jumps > 0:
                jumps = self.rng.normal(self.jump_mean, self.jump_std, size=n_jumps)
                r[t] += jumps.sum()
        
        return r
    
    def get_true_structure(self) -> Dict[str, Any]:
        """Return known Levy structure (but NO predictability)."""
        return {
            'has_exploitable_structure': False,
            'has_finance_texture': True,
            'type': 'levy_jumps',
            'lambda_jumps': self.lambda_jumps,
            'has_fat_tails': True,
            'has_kurtosis': True,
            'is_predictable': False,
            'jump_timing': 'unpredictable'
        }
    
    def get_family(self) -> str:
        return "finance_null"
    
    def get_name(self) -> str:
        return "levy_jumps"


def get_finance_null_dataset(name: str, seed: int, config: Dict[str, Any]) -> BaseDataset:
    """
    Factory function to create finance-null datasets.
    
    Args:
        name: 'garch_jumps', 'stochastic_vol', or 'levy_jumps'
        seed: Random seed
        config: Dataset config
    
    Returns:
        BaseDataset instance
    """
    datasets = {
        'garch_jumps': GARCHWithJumpsDataset,
        'stochastic_vol': StochasticVolatilityDataset,
        'levy_jumps': LevyJumpDataset
    }
    
    if name not in datasets:
        raise ValueError(f"Unknown finance-null dataset: {name}. Choose from {list(datasets.keys())}")
    
    return datasets[name](seed, config)
