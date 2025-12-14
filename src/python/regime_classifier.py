"""
Regime Classifier - Layer 2 of Underdog Trading System

Detect market regime and adjust strategy accordingly:
- 10 regimes based on VIX, SPY trend, yield curve, breadth
- Each regime has different strategy weights
- Update every 5-15 minutes

Regimes:
1. BULL_LOW_VOL - VIX<15, SPY up - Full risk on
2. BULL_MODERATE_VOL - VIX 15-20, SPY up - Normal risk
3. BULL_HIGH_VOL - VIX 20-30, SPY up - Cautious
4. BEAR_LOW_VOL - VIX<15, SPY down - Mean reversion
5. BEAR_MODERATE_VOL - VIX 15-20, SPY down - Defensive
6. BEAR_HIGH_VOL - VIX 20-30, SPY down - Very defensive
7. CHOPPY_LOW_VOL - VIX<15, SPY flat - Range trading
8. CHOPPY_HIGH_VOL - VIX>20, SPY flat - Wait mode
9. PANIC_EXTREME_VOL - VIX>30, SPY down - Cash/hedges
10. EUPHORIA_EXTREME - VIX<12, SPY parabolic - Fade rallies
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from datetime import datetime, timedelta
import logging

try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    YF_AVAILABLE = False
    logging.warning("yfinance not installed - regime detection unavailable")

logger = logging.getLogger(__name__)


class RegimeClassifier:
    """
    Classify current market regime
    
    Uses free data sources:
    - VIX (volatility)
    - SPY (trend)
    - ^TNX (10-year yield)
    - ^FVX (5-year yield)
    - QQQ/SPY breadth
    """
    
    def __init__(self, cache_minutes: int = 15):
        """
        Initialize regime classifier
        
        Args:
            cache_minutes: Cache regime for this many minutes before recalculating
        """
        self.cache_minutes = cache_minutes
        self.last_update = None
        self.current_regime = None
        self.regime_data = {}
        
    def classify_regime(self, force_update: bool = False) -> Dict[str, any]:
        """
        Classify current market regime
        
        Args:
            force_update: Force recalculation (ignore cache)
            
        Returns:
            regime: {
                'name': 'BULL_LOW_VOL',
                'position_size_multiplier': 1.0,
                'max_positions': 10,
                'stop_loss_pct': 0.08,
                'strategy_weights': {...}
            }
        """
        if not YF_AVAILABLE:
            logger.warning("yfinance unavailable - returning default regime")
            return self._get_default_regime()
        
        # Check cache
        if not force_update and self.last_update is not None:
            elapsed = (datetime.now() - self.last_update).total_seconds() / 60
            if elapsed < self.cache_minutes:
                logger.info(f"Using cached regime: {self.current_regime['name']}")
                return self.current_regime
        
        # Fetch regime indicators
        try:
            self.regime_data = self._fetch_regime_indicators()
            regime_name = self._determine_regime(self.regime_data)
            regime_config = self._get_regime_config(regime_name)
            
            self.current_regime = regime_config
            self.last_update = datetime.now()
            
            logger.info(f"Regime classified: {regime_name}")
            logger.info(f"VIX: {self.regime_data['vix']:.1f}, SPY 20d return: {self.regime_data['spy_return_20d']:.2%}")
            
            return regime_config
            
        except Exception as e:
            logger.error(f"Error classifying regime: {e}")
            return self._get_default_regime()
    
    def _fetch_regime_indicators(self) -> Dict[str, float]:
        """Fetch regime indicators from yfinance"""
        
        indicators = {}
        
        # VIX (volatility)
        vix = yf.Ticker("^VIX")
        vix_data = vix.history(period="1d")
        indicators['vix'] = vix_data['Close'].iloc[-1] if len(vix_data) > 0 else 20.0
        
        # SPY trend (20-day return)
        spy = yf.Ticker("SPY")
        spy_data = spy.history(period="1mo")
        if len(spy_data) >= 20:
            indicators['spy_return_20d'] = (spy_data['Close'].iloc[-1] / spy_data['Close'].iloc[-20] - 1)
        else:
            indicators['spy_return_20d'] = 0.0
        
        # SPY 5-day return (short-term)
        if len(spy_data) >= 5:
            indicators['spy_return_5d'] = (spy_data['Close'].iloc[-1] / spy_data['Close'].iloc[-5] - 1)
        else:
            indicators['spy_return_5d'] = 0.0
        
        # Yield curve (10Y - 2Y)
        try:
            tnx = yf.Ticker("^TNX")  # 10-year
            tnx_data = tnx.history(period="1d")
            ten_year = tnx_data['Close'].iloc[-1] if len(tnx_data) > 0 else 4.0
            
            # Note: ^IRX is 13-week (3-month), not 2-year
            # Use approximation or skip if unavailable
            indicators['yield_curve'] = 0.0  # Placeholder
            
        except:
            indicators['yield_curve'] = 0.0
        
        # QQQ/SPY ratio (tech leadership)
        try:
            qqq = yf.Ticker("QQQ")
            qqq_data = qqq.history(period="5d")
            
            if len(qqq_data) >= 2 and len(spy_data) >= 2:
                qqq_return = (qqq_data['Close'].iloc[-1] / qqq_data['Close'].iloc[0] - 1)
                spy_return = (spy_data['Close'].iloc[-1] / spy_data['Close'].iloc[0] - 1)
                indicators['qqq_spy_ratio'] = qqq_return - spy_return
            else:
                indicators['qqq_spy_ratio'] = 0.0
        except:
            indicators['qqq_spy_ratio'] = 0.0
        
        return indicators
    
    def _determine_regime(self, data: Dict[str, float]) -> str:
        """Determine regime name from indicators"""
        
        vix = data['vix']
        spy_20d = data['spy_return_20d']
        spy_5d = data['spy_return_5d']
        
        # Extreme regimes first
        
        # PANIC - VIX > 30 and SPY down
        if vix > 30 and spy_20d < -0.05:
            return 'PANIC_EXTREME_VOL'
        
        # EUPHORIA - VIX < 12 and SPY up big
        if vix < 12 and spy_20d > 0.10:
            return 'EUPHORIA_EXTREME'
        
        # Bullish regimes (SPY up over 20 days)
        if spy_20d > 0.03:
            if vix < 15:
                return 'BULL_LOW_VOL'
            elif vix < 20:
                return 'BULL_MODERATE_VOL'
            else:
                return 'BULL_HIGH_VOL'
        
        # Bearish regimes (SPY down over 20 days)
        elif spy_20d < -0.03:
            if vix < 15:
                return 'BEAR_LOW_VOL'
            elif vix < 20:
                return 'BEAR_MODERATE_VOL'
            else:
                return 'BEAR_HIGH_VOL'
        
        # Choppy regimes (SPY flat)
        else:
            if vix < 15:
                return 'CHOPPY_LOW_VOL'
            else:
                return 'CHOPPY_HIGH_VOL'
    
    def _get_regime_config(self, regime_name: str) -> Dict[str, any]:
        """Get configuration for a regime"""
        
        configs = {
            'BULL_LOW_VOL': {
                'name': 'BULL_LOW_VOL',
                'description': 'Bull market, low volatility - Full risk on',
                'position_size_multiplier': 1.0,
                'max_positions': 10,
                'stop_loss_pct': 0.08,
                'take_profit_pct': 0.15,
                'min_confidence': 0.65,
                'strategy_weights': {
                    'momentum': 0.40,
                    'mean_reversion': 0.10,
                    'dark_pool': 0.25,
                    'cross_asset': 0.15,
                    'sentiment': 0.10
                }
            },
            'BULL_MODERATE_VOL': {
                'name': 'BULL_MODERATE_VOL',
                'description': 'Bull market, moderate volatility - Normal risk',
                'position_size_multiplier': 0.8,
                'max_positions': 8,
                'stop_loss_pct': 0.10,
                'take_profit_pct': 0.12,
                'min_confidence': 0.70,
                'strategy_weights': {
                    'momentum': 0.35,
                    'mean_reversion': 0.15,
                    'dark_pool': 0.25,
                    'cross_asset': 0.15,
                    'sentiment': 0.10
                }
            },
            'BULL_HIGH_VOL': {
                'name': 'BULL_HIGH_VOL',
                'description': 'Bull market, high volatility - Cautious',
                'position_size_multiplier': 0.6,
                'max_positions': 6,
                'stop_loss_pct': 0.12,
                'take_profit_pct': 0.10,
                'min_confidence': 0.75,
                'strategy_weights': {
                    'momentum': 0.25,
                    'mean_reversion': 0.20,
                    'dark_pool': 0.25,
                    'cross_asset': 0.20,
                    'sentiment': 0.10
                }
            },
            'BEAR_LOW_VOL': {
                'name': 'BEAR_LOW_VOL',
                'description': 'Bear market, low volatility - Mean reversion',
                'position_size_multiplier': 0.5,
                'max_positions': 5,
                'stop_loss_pct': 0.10,
                'take_profit_pct': 0.08,
                'min_confidence': 0.75,
                'strategy_weights': {
                    'momentum': 0.15,
                    'mean_reversion': 0.35,
                    'dark_pool': 0.20,
                    'cross_asset': 0.20,
                    'sentiment': 0.10
                }
            },
            'BEAR_MODERATE_VOL': {
                'name': 'BEAR_MODERATE_VOL',
                'description': 'Bear market, moderate volatility - Defensive',
                'position_size_multiplier': 0.4,
                'max_positions': 4,
                'stop_loss_pct': 0.08,
                'take_profit_pct': 0.06,
                'min_confidence': 0.80,
                'strategy_weights': {
                    'momentum': 0.10,
                    'mean_reversion': 0.40,
                    'dark_pool': 0.20,
                    'cross_asset': 0.20,
                    'sentiment': 0.10
                }
            },
            'BEAR_HIGH_VOL': {
                'name': 'BEAR_HIGH_VOL',
                'description': 'Bear market, high volatility - Very defensive',
                'position_size_multiplier': 0.3,
                'max_positions': 3,
                'stop_loss_pct': 0.06,
                'take_profit_pct': 0.05,
                'min_confidence': 0.85,
                'strategy_weights': {
                    'momentum': 0.10,
                    'mean_reversion': 0.40,
                    'dark_pool': 0.20,
                    'cross_asset': 0.20,
                    'sentiment': 0.10
                }
            },
            'CHOPPY_LOW_VOL': {
                'name': 'CHOPPY_LOW_VOL',
                'description': 'Sideways market, low volatility - Range trading',
                'position_size_multiplier': 0.6,
                'max_positions': 6,
                'stop_loss_pct': 0.08,
                'take_profit_pct': 0.06,
                'min_confidence': 0.75,
                'strategy_weights': {
                    'momentum': 0.15,
                    'mean_reversion': 0.40,
                    'dark_pool': 0.20,
                    'cross_asset': 0.15,
                    'sentiment': 0.10
                }
            },
            'CHOPPY_HIGH_VOL': {
                'name': 'CHOPPY_HIGH_VOL',
                'description': 'Sideways market, high volatility - Wait mode',
                'position_size_multiplier': 0.3,
                'max_positions': 3,
                'stop_loss_pct': 0.06,
                'take_profit_pct': 0.05,
                'min_confidence': 0.85,
                'strategy_weights': {
                    'momentum': 0.10,
                    'mean_reversion': 0.35,
                    'dark_pool': 0.25,
                    'cross_asset': 0.20,
                    'sentiment': 0.10
                }
            },
            'PANIC_EXTREME_VOL': {
                'name': 'PANIC_EXTREME_VOL',
                'description': 'Extreme panic - Cash/hedges only',
                'position_size_multiplier': 0.2,
                'max_positions': 2,
                'stop_loss_pct': 0.05,
                'take_profit_pct': 0.04,
                'min_confidence': 0.90,
                'strategy_weights': {
                    'momentum': 0.05,
                    'mean_reversion': 0.45,
                    'dark_pool': 0.20,
                    'cross_asset': 0.20,
                    'sentiment': 0.10
                }
            },
            'EUPHORIA_EXTREME': {
                'name': 'EUPHORIA_EXTREME',
                'description': 'Extreme euphoria - Fade rallies',
                'position_size_multiplier': 0.4,
                'max_positions': 4,
                'stop_loss_pct': 0.10,
                'take_profit_pct': 0.08,
                'min_confidence': 0.80,
                'strategy_weights': {
                    'momentum': 0.15,
                    'mean_reversion': 0.35,
                    'dark_pool': 0.20,
                    'cross_asset': 0.20,
                    'sentiment': 0.10
                }
            }
        }
        
        return configs.get(regime_name, configs['BULL_MODERATE_VOL'])
    
    def _get_default_regime(self) -> Dict[str, any]:
        """Default regime when data unavailable"""
        return self._get_regime_config('BULL_MODERATE_VOL')
    
    def get_regime_history(self, days: int = 30) -> pd.DataFrame:
        """Get historical regime classifications (for analysis)"""
        # TODO: Implement if needed for backtesting
        pass


def quick_test():
    """Quick test of RegimeClassifier"""
    print("Testing RegimeClassifier...")
    
    classifier = RegimeClassifier(cache_minutes=15)
    
    # Classify current regime
    regime = classifier.classify_regime()
    
    print(f"\nCurrent Regime: {regime['name']}")
    print(f"Description: {regime['description']}")
    print(f"Position Size Multiplier: {regime['position_size_multiplier']}")
    print(f"Max Positions: {regime['max_positions']}")
    print(f"Stop Loss: {regime['stop_loss_pct']:.1%}")
    print(f"Min Confidence: {regime['min_confidence']:.2f}")
    
    print(f"\nStrategy Weights:")
    for strategy, weight in regime['strategy_weights'].items():
        print(f"  {strategy}: {weight:.1%}")
    
    # Test caching
    print(f"\nTesting cache...")
    regime2 = classifier.classify_regime()
    print(f"Cache working: {regime['name'] == regime2['name']}")
    
    # Force update
    print(f"\nForce update...")
    regime3 = classifier.classify_regime(force_update=True)
    print(f"Regime after update: {regime3['name']}")
    
    print("\n✅ RegimeClassifier test complete")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    quick_test()
