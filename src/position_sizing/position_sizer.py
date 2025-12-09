"""
Position Sizer & Risk Manager
Translates forecaster confidence → optimal position size, stop-loss, take-profit.

Research: Kelly criterion with regime/sector adjustments + volatility targeting.
"""

from typing import Dict, Optional, Tuple
import numpy as np
from dataclasses import dataclass


@dataclass
class PositionSizingConfig:
    """Configuration for position sizing"""
    # Portfolio constraints
    max_position_pct: float = 0.05  # Max 5% per position
    min_position_pct: float = 0.005  # Min 0.5% per position
    max_sector_allocation: float = 0.30  # Max 30% per sector
    
    # Risk parameters
    max_risk_per_trade: float = 0.01  # Max 1% portfolio risk per trade
    target_portfolio_vol: float = 0.15  # Target 15% annual volatility
    
    # Kelly criterion
    kelly_fraction: float = 0.25  # Fractional Kelly (conservative)
    kelly_enabled: bool = True
    
    # Regime adjustments
    bull_regime_multiplier: float = 1.2
    bear_regime_multiplier: float = 0.7
    chop_regime_multiplier: float = 0.5


class PositionSizer:
    """
    Calculate optimal position sizes based on confidence, risk, and regime.
    
    Architecture:
        Input: calibrated_confidence, sector, regime, portfolio_state
        Processing: Kelly formula → volatility adjustment → regime adjustment → constraints
        Output: position_size_pct, stop_loss, take_profit
    
    Usage:
        sizer = PositionSizer(config=PositionSizingConfig())
        position = sizer.calculate_position_size(
            confidence=0.72,
            expected_return=0.052,
            sector='AI_INFRA',
            regime='BULL_LOW_VOL',
            ticker_volatility=0.30,
            current_sector_allocation=0.15
        )
        # Returns: {'size_pct': 0.025, 'stop_loss': -0.035, 'take_profit': 0.075}
    """
    
    def __init__(self, config: Optional[PositionSizingConfig] = None):
        self.config = config or PositionSizingConfig()
        
        # Sector-specific parameters
        self.sector_params = self._init_sector_params()
    
    def calculate_position_size(
        self,
        confidence: float,
        expected_return: float,
        sector: str,
        regime: str,
        ticker_volatility: float,
        current_sector_allocation: float = 0.0,
        portfolio_value: float = 100000.0
    ) -> Dict[str, float]:
        """
        Calculate optimal position size and risk levels.
        
        Args:
            confidence: Calibrated confidence [0, 1]
            expected_return: Expected return over horizon (e.g., 0.052 = +5.2%)
            sector: Sector label
            regime: Current market regime
            ticker_volatility: Historical volatility (annualized)
            current_sector_allocation: Current % of portfolio in this sector
            portfolio_value: Total portfolio value
        
        Returns:
            {
                'position_size_pct': 0.025,  # 2.5% of portfolio
                'position_size_dollars': 2500,
                'stop_loss_pct': -0.035,  # -3.5%
                'take_profit_pct': 0.075,  # +7.5%
                'risk_per_trade_pct': 0.008,  # 0.8% portfolio risk
                'hold_period_days': 8,
            }
        """
        
        # 1. Base Kelly sizing
        if self.config.kelly_enabled:
            kelly_size = self._calculate_kelly(confidence, expected_return, ticker_volatility)
        else:
            kelly_size = confidence * self.config.max_position_pct
        
        # 2. Adjust for regime
        regime_multiplier = self._get_regime_multiplier(regime)
        adjusted_size = kelly_size * regime_multiplier
        
        # 3. Adjust for sector
        sector_multiplier = self._get_sector_multiplier(sector)
        adjusted_size *= sector_multiplier
        
        # 4. Volatility targeting
        vol_adjustment = self.config.target_portfolio_vol / ticker_volatility
        vol_adjustment = np.clip(vol_adjustment, 0.5, 2.0)  # Cap adjustment
        adjusted_size *= vol_adjustment
        
        # 5. Apply constraints
        final_size = self._apply_constraints(
            adjusted_size,
            sector,
            current_sector_allocation
        )
        
        # 6. Calculate stop-loss and take-profit
        stop_loss = self._calculate_stop_loss(sector, regime, ticker_volatility)
        take_profit = self._calculate_take_profit(sector, regime, expected_return, stop_loss)
        
        # 7. Calculate actual risk per trade
        position_dollars = final_size * portfolio_value
        risk_per_trade = abs(stop_loss) * final_size
        
        # 8. Adaptive hold period
        hold_period = self._calculate_hold_period(regime, confidence, expected_return)
        
        return {
            'position_size_pct': final_size,
            'position_size_dollars': position_dollars,
            'stop_loss_pct': stop_loss,
            'take_profit_pct': take_profit,
            'risk_per_trade_pct': risk_per_trade,
            'hold_period_days': hold_period,
            'kelly_base': kelly_size,
            'regime_multiplier': regime_multiplier,
            'sector_multiplier': sector_multiplier,
            'vol_adjustment': vol_adjustment,
        }
    
    # ========== KELLY CRITERION ==========
    
    def _calculate_kelly(self, confidence: float, expected_return: float, volatility: float) -> float:
        """
        Calculate Kelly fraction: f* = (p * b - q) / b
        
        Where:
            p = probability of win (confidence)
            q = probability of loss (1 - confidence)
            b = odds received (win amount / loss amount)
        
        Returns: Position size as fraction of portfolio
        """
        p = confidence
        q = 1 - confidence
        
        # Estimate odds: assume symmetric loss/gain initially
        # b = (expected_gain / expected_loss)
        # Simplification: b ≈ expected_return / risk
        # For conservative Kelly, assume risk = 0.5 * expected_return
        b = abs(expected_return) / (0.5 * abs(expected_return)) if expected_return != 0 else 1.0
        
        kelly_fraction = (p * b - q) / b if b > 0 else 0.0
        
        # Apply fractional Kelly (reduce aggressiveness)
        kelly_fraction *= self.config.kelly_fraction
        
        # Clip to reasonable range
        kelly_fraction = np.clip(kelly_fraction, 0.0, self.config.max_position_pct)
        
        return kelly_fraction
    
    # ========== REGIME ADJUSTMENTS ==========
    
    def _get_regime_multiplier(self, regime: str) -> float:
        """Get position size multiplier for regime"""
        if 'BULL' in regime:
            return self.config.bull_regime_multiplier
        elif 'BEAR' in regime:
            return self.config.bear_regime_multiplier
        elif 'NEUTRAL' in regime or 'CHOP' in regime:
            return self.config.chop_regime_multiplier
        else:
            return 1.0
    
    # ========== SECTOR ADJUSTMENTS ==========
    
    def _init_sector_params(self) -> Dict[str, Dict]:
        """
        Initialize sector-specific parameters.
        
        Research: Different sectors have different optimal parameters
        - AI: Higher Sharpe (0.82) → larger positions
        - Quantum: Lower Sharpe (0.68) → smaller positions
        """
        return {
            'AI_INFRA': {
                'multiplier': 1.15,
                'stop_loss_pct': -0.03,  # Tighter stops (less volatile)
                'risk_reward_ratio': 2.5,
            },
            'QUANTUM': {
                'multiplier': 0.85,
                'stop_loss_pct': -0.08,  # Wider stops (more volatile)
                'risk_reward_ratio': 2.0,
            },
            'ROBOTAXI': {
                'multiplier': 0.90,
                'stop_loss_pct': -0.04,
                'risk_reward_ratio': 2.2,
            },
            'HEALTHCARE': {
                'multiplier': 1.00,
                'stop_loss_pct': -0.05,
                'risk_reward_ratio': 2.3,
            },
            'ENERGY': {
                'multiplier': 0.95,
                'stop_loss_pct': -0.06,
                'risk_reward_ratio': 2.0,
            },
            'DEFAULT': {
                'multiplier': 1.0,
                'stop_loss_pct': -0.05,
                'risk_reward_ratio': 2.0,
            }
        }
    
    def _get_sector_multiplier(self, sector: str) -> float:
        """Get position size multiplier for sector"""
        params = self.sector_params.get(sector, self.sector_params['DEFAULT'])
        return params['multiplier']
    
    # ========== STOP-LOSS & TAKE-PROFIT ==========
    
    def _calculate_stop_loss(self, sector: str, regime: str, volatility: float) -> float:
        """
        Calculate optimal stop-loss level.
        
        Approach: Sector-specific + regime-adjusted + volatility-scaled
        """
        params = self.sector_params.get(sector, self.sector_params['DEFAULT'])
        base_stop = params['stop_loss_pct']
        
        # Adjust for regime (wider stops in volatile regimes)
        if 'EXTREME' in regime or 'VOLATILE' in regime:
            base_stop *= 1.5
        elif 'ELEVATED' in regime:
            base_stop *= 1.2
        
        # Adjust for ticker volatility (wider stops for high-vol stocks)
        if volatility > 0.40:  # >40% annual vol
            base_stop *= 1.3
        elif volatility > 0.25:  # >25% annual vol
            base_stop *= 1.1
        
        # Ensure minimum stop
        base_stop = min(base_stop, -0.02)  # At least 2% stop
        
        return base_stop
    
    def _calculate_take_profit(self, sector: str, regime: str, expected_return: float, stop_loss: float) -> float:
        """
        Calculate optimal take-profit level.
        
        Approach: Risk-reward ratio (typically 2:1 or 2.5:1)
        """
        params = self.sector_params.get(sector, self.sector_params['DEFAULT'])
        risk_reward_ratio = params['risk_reward_ratio']
        
        # Base take-profit: risk-reward ratio * stop-loss
        take_profit = abs(stop_loss) * risk_reward_ratio
        
        # Don't exceed expected return by too much (be realistic)
        if expected_return > 0:
            take_profit = min(take_profit, expected_return * 1.5)
        
        # Adjust for regime (take profits faster in volatile regimes)
        if 'EXTREME' in regime:
            take_profit *= 0.8
        elif 'ELEVATED' in regime:
            take_profit *= 0.9
        
        return take_profit
    
    # ========== CONSTRAINTS ==========
    
    def _apply_constraints(self, size: float, sector: str, current_sector_allocation: float) -> float:
        """Apply portfolio-level constraints"""
        
        # 1. Max position size
        size = min(size, self.config.max_position_pct)
        
        # 2. Min position size (don't bother with tiny positions)
        if size < self.config.min_position_pct:
            return 0.0
        
        # 3. Sector allocation constraint
        if current_sector_allocation + size > self.config.max_sector_allocation:
            # Reduce to fit within sector limit
            available = self.config.max_sector_allocation - current_sector_allocation
            size = max(available, 0.0)
        
        return size
    
    # ========== HOLD PERIOD ==========
    
    @staticmethod
    def _calculate_hold_period(regime: str, confidence: float, expected_return: float) -> int:
        """
        Calculate adaptive hold period.
        
        Research: Higher confidence + bull regime → longer hold
                  Lower confidence + catalyst-driven → shorter hold
        """
        # Base hold period: 5-15 days
        base_hold = 10
        
        # Adjust for confidence (higher confidence → hold longer)
        confidence_adjustment = (confidence - 0.5) * 10  # ±5 days
        
        # Adjust for regime
        if 'BULL' in regime:
            regime_adjustment = 3
        elif 'BEAR' in regime:
            regime_adjustment = -2
        else:
            regime_adjustment = 0
        
        # Adjust for expected return magnitude (larger moves → longer hold)
        return_adjustment = min(abs(expected_return) * 50, 5)  # Up to +5 days
        
        hold_period = base_hold + confidence_adjustment + regime_adjustment + return_adjustment
        
        # Clip to reasonable range [3, 20] days
        return int(np.clip(hold_period, 3, 20))


# ========== PORTFOLIO-LEVEL RISK MANAGER ==========

class PortfolioRiskManager:
    """
    Manage portfolio-level risk across all positions.
    
    Responsibilities:
    - Enforce sector allocation limits
    - Cap total portfolio leverage
    - Monitor correlated positions
    - Adjust position sizes in high-risk environments
    """
    
    def __init__(self, config: Optional[PositionSizingConfig] = None):
        self.config = config or PositionSizingConfig()
        self.current_positions = {}  # {ticker: position_info}
    
    def check_new_position(self, ticker: str, sector: str, proposed_size: float) -> Tuple[bool, str]:
        """
        Check if new position is allowed given portfolio constraints.
        
        Returns: (allowed: bool, reason: str)
        """
        # 1. Check sector allocation
        sector_allocation = self._calculate_sector_allocation(sector)
        if sector_allocation + proposed_size > self.config.max_sector_allocation:
            return False, f"Sector {sector} at max allocation ({sector_allocation:.1%})"
        
        # 2. Check total portfolio allocation
        total_allocation = sum(pos['size_pct'] for pos in self.current_positions.values())
        if total_allocation + proposed_size > 1.0:
            return False, f"Portfolio fully allocated ({total_allocation:.1%})"
        
        # 3. Check correlation risk
        # TODO: Implement correlation checks
        
        return True, "OK"
    
    def _calculate_sector_allocation(self, sector: str) -> float:
        """Calculate current allocation to a sector"""
        return sum(
            pos['size_pct'] 
            for pos in self.current_positions.values() 
            if pos.get('sector') == sector
        )
