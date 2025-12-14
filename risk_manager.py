"""
Capital Protection Risk Manager
Prevents account blowup through position sizing and drawdown limits
"""

import logging
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class MarketRegime(Enum):
    CRASH = "crash"
    BEAR = "bear"
    CORRECTION = "correction"
    BULL = "bull"
    UNKNOWN = "unknown"

@dataclass
class RiskLimits:
    """Risk limits based on market regime"""
    max_position_pct: float  # Max % of capital per position
    max_open_trades: int     # Max concurrent positions
    can_trade: bool         # Whether trading is allowed

class RiskManager:
    """
    Capital protection system with regime-based position sizing
    Prevents account blowup through drawdown limits and position controls
    """

    def __init__(self,
                 initial_capital: float = 10000.0,
                 max_drawdown_pct: float = 0.10,  # 10% max loss
                 risk_per_trade_pct: float = 0.02,
                 account_balance: Optional[float] = None):  # Colab compatibility alias
        """Initialize RiskManager.

        Supports both `initial_capital` and legacy/Colab param `account_balance`.
        If `account_balance` is provided it will override `initial_capital`.
        """

        if account_balance is not None:
            # Backwards compatibility: allow passing account_balance instead of initial_capital
            initial_capital = account_balance

        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_drawdown_pct = max_drawdown_pct
        self.risk_per_trade_pct = risk_per_trade_pct

        # Track peak capital for drawdown calculation
        self.peak_capital = initial_capital

        # Regime-based risk limits
        self.regime_limits = {
            MarketRegime.CRASH: RiskLimits(max_position_pct=0.00, max_open_trades=0, can_trade=False),
            MarketRegime.BEAR: RiskLimits(max_position_pct=0.02, max_open_trades=1, can_trade=True),
            MarketRegime.CORRECTION: RiskLimits(max_position_pct=0.03, max_open_trades=2, can_trade=True),
            MarketRegime.BULL: RiskLimits(max_position_pct=0.05, max_open_trades=5, can_trade=True),
            MarketRegime.UNKNOWN: RiskLimits(max_position_pct=0.01, max_open_trades=1, can_trade=True)
        }

        logger.info(
            f"🛡️ Risk Manager initialized - Capital: ${initial_capital:,.2f}, Max DD: {max_drawdown_pct:.1%}, Risk/Trade: {risk_per_trade_pct:.1%}"
        )

    def update_current_capital(self, new_balance: float) -> None:
        """Alias method for external integrators using different naming."""
        self.update_capital(new_balance)

    def update_capital(self, new_capital: float) -> None:
        """Update current capital and track peak for drawdown calculation"""
        self.current_capital = new_capital
        self.peak_capital = max(self.peak_capital, new_capital)

    def check_max_drawdown(self) -> bool:
        """
        Check if current drawdown exceeds maximum allowed
        Returns True if trading should be halted
        """
        if self.peak_capital <= 0:
            return True

        current_drawdown = (self.peak_capital - self.current_capital) / self.peak_capital

        if current_drawdown > self.max_drawdown_pct:
            logger.warning(f"🚨 MAX DRAWDOWN EXCEEDED: {current_drawdown:.1%} > {self.max_drawdown_pct:.1%}")
            logger.warning(f"   Peak: ${self.peak_capital:,.2f}, Current: ${self.current_capital:,.2f}")
            return True

        return False

    def can_trade(self, regime: MarketRegime = MarketRegime.UNKNOWN) -> bool:
        """
        Master gate function - determines if trading is allowed
        Checks drawdown limits and regime restrictions
        """
        # First check drawdown
        if self.check_max_drawdown():
            logger.warning("🚫 TRADING HALTED: Maximum drawdown exceeded")
            return False

        # Then check regime limits
        limits = self.regime_limits.get(regime, self.regime_limits[MarketRegime.UNKNOWN])

        if not limits.can_trade:
            logger.warning(f"🚫 TRADING HALTED: {regime.value.upper()} regime - trading not allowed")
            return False

        return True

    def calculate_position_size(self,
                              entry_price: float,
                              stop_loss: float,
                              regime: MarketRegime = MarketRegime.UNKNOWN) -> Tuple[int, float]:
        """
        Calculate position size based on risk per trade and regime limits
        Returns: (shares, risk_amount)
        """
        if not self.can_trade(regime):
            return 0, 0.0

        # Get regime limits
        limits = self.regime_limits.get(regime, self.regime_limits[MarketRegime.UNKNOWN])

        # Calculate risk amount (regime-adjusted)
        max_risk_amount = self.current_capital * self.risk_per_trade_pct
        regime_adjusted_risk = max_risk_amount * limits.max_position_pct / 0.05  # Normalize to BULL regime
        risk_amount = min(max_risk_amount, regime_adjusted_risk)

        # Calculate price risk (entry to stop distance)
        price_risk = abs(entry_price - stop_loss)
        if price_risk <= 0:
            logger.warning("⚠️ Invalid stop loss - no price risk calculated")
            return 0, 0.0

        # Position size = risk amount / price risk
        shares = int(risk_amount / price_risk)

        # Ensure we don't exceed regime position limits
        max_position_value = self.current_capital * limits.max_position_pct
        max_shares_by_regime = int(max_position_value / entry_price)
        shares = min(shares, max_shares_by_regime)

        # Minimum position size check
        if shares < 1:
            logger.info(f"📊 Position too small for risk management: {shares} shares")
            return 0, 0.0

        actual_risk = shares * price_risk
        logger.info(f"📊 Position Size: {shares} shares, Risk: ${actual_risk:.2f} ({actual_risk/self.current_capital:.2%} of capital)")

        return shares, actual_risk

    def get_regime_limits(self, regime: MarketRegime) -> RiskLimits:
        """Get risk limits for a specific market regime"""
        return self.regime_limits.get(regime, self.regime_limits[MarketRegime.UNKNOWN])

    def get_current_drawdown(self) -> float:
        """Get current drawdown percentage"""
        if self.peak_capital <= 0:
            return 0.0
        return (self.peak_capital - self.current_capital) / self.peak_capital

    def reset_peak_capital(self) -> None:
        """Reset peak capital (use after successful recovery)"""
        self.peak_capital = self.current_capital
        logger.info(f"🔄 Peak capital reset to: ${self.peak_capital:,.2f}")

# Test functions for validation
def test_risk_manager():
    """Test risk manager with different scenarios"""
    print("🧪 Testing Risk Manager...")

    # Initialize with $10,000
    rm = RiskManager(initial_capital=10000.0)

    # Test CRASH regime
    print("\n1. Testing CRASH regime:")
    can_trade = rm.can_trade(MarketRegime.CRASH)
    shares, risk = rm.calculate_position_size(100.0, 95.0, MarketRegime.CRASH)
    print(f"   Can trade: {can_trade}, Shares: {shares}, Risk: ${risk:.2f}")

    # Test BULL regime
    print("\n2. Testing BULL regime:")
    can_trade = rm.can_trade(MarketRegime.BULL)
    shares, risk = rm.calculate_position_size(100.0, 95.0, MarketRegime.BULL)
    print(f"   Can trade: {can_trade}, Shares: {shares}, Risk: ${risk:.2f}")

    # Test drawdown limit
    print("\n3. Testing drawdown limit:")
    rm.update_capital(9000.0)  # 10% loss
    can_trade = rm.can_trade(MarketRegime.BULL)
    print(f"   After 10% loss, can trade: {can_trade}")

    rm.update_capital(8500.0)  # 15% loss
    can_trade = rm.can_trade(MarketRegime.BULL)
    print(f"   After 15% loss, can trade: {can_trade}")

    print("\n✅ Risk Manager tests completed")

if __name__ == "__main__":
    test_risk_manager()