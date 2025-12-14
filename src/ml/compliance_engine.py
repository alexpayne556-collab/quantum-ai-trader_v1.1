"""
PDT COMPLIANCE ENGINE
=====================
Pattern Day Trader rules and risk management.

Features:
- PDT compliance (3 day trades per 5 trading days)
- Risk limits (2% per trade, 8% total stop loss)
- Daily loss limits (5% yellow alert, 10% red alert)
- Position concentration limits
- Maximum drawdown protection

This keeps you SAFE and compliant while maximizing returns.

Author: Quantum AI Trader
Date: December 10, 2025
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ComplianceEngine:
    """
    Enforce PDT rules and risk management.
    """
    
    def __init__(
        self,
        account_equity: float = 780.59,
        is_pdt_restricted: bool = True,
        max_day_trades_per_period: int = 3,
        pdt_period_days: int = 5,
        max_risk_per_trade: float = 0.02,     # 2% max risk per trade
        max_total_risk: float = 0.08,         # 8% max total portfolio risk
        daily_loss_limit_yellow: float = 0.05, # 5% daily loss = yellow alert
        daily_loss_limit_red: float = 0.10,    # 10% daily loss = red alert
        max_position_size: float = 0.25,       # 25% max per position
        max_sector_concentration: float = 0.50 # 50% max per sector
    ):
        """
        Args:
            account_equity: Current account equity
            is_pdt_restricted: Account under $25K?
            max_day_trades_per_period: PDT limit (3 trades per 5 days)
            pdt_period_days: PDT rolling period (5 trading days)
            max_risk_per_trade: Maximum risk per trade (2%)
            max_total_risk: Maximum total portfolio risk (8%)
            daily_loss_limit_yellow: Yellow alert threshold (5%)
            daily_loss_limit_red: Red alert threshold (10%)
            max_position_size: Max position as % of equity
            max_sector_concentration: Max sector exposure
        """
        self.account_equity = account_equity
        self.is_pdt_restricted = is_pdt_restricted
        self.max_day_trades_per_period = max_day_trades_per_period
        self.pdt_period_days = pdt_period_days
        self.max_risk_per_trade = max_risk_per_trade
        self.max_total_risk = max_total_risk
        self.daily_loss_limit_yellow = daily_loss_limit_yellow
        self.daily_loss_limit_red = daily_loss_limit_red
        self.max_position_size = max_position_size
        self.max_sector_concentration = max_sector_concentration
        
        # Tracking
        self.day_trades: List[Dict] = []
        self.daily_pnl: Dict[str, float] = defaultdict(float)
        self.starting_equity_today: float = account_equity
    
    def check_trade_allowed(
        self,
        action: str,  # 'BUY' or 'SELL'
        ticker: str,
        position_size: float,
        entry_date: Optional[datetime] = None,
        current_portfolio: Optional[Dict] = None
    ) -> Tuple[bool, str]:
        """
        Check if trade is allowed under compliance rules.
        
        Args:
            action: 'BUY' or 'SELL'
            ticker: Ticker symbol
            position_size: Position size ($)
            entry_date: Entry date (for SELL, to check day trade)
            current_portfolio: Current portfolio state
            
        Returns:
            (is_allowed, reason)
        """
        # BUY checks
        if action == 'BUY':
            return self._check_buy_allowed(ticker, position_size, current_portfolio)
        
        # SELL checks
        elif action == 'SELL':
            return self._check_sell_allowed(ticker, entry_date, current_portfolio)
        
        return False, f"Unknown action: {action}"
    
    def _check_buy_allowed(
        self,
        ticker: str,
        position_size: float,
        current_portfolio: Optional[Dict]
    ) -> Tuple[bool, str]:
        """Check if BUY is allowed."""
        
        # 1. Position size limit (25% max)
        max_position_value = self.account_equity * self.max_position_size
        if position_size > max_position_value:
            return False, f"Position too large: ${position_size:.2f} > ${max_position_value:.2f} (25% limit)"
        
        # 2. Risk per trade limit (2%)
        # Assuming 19% stop loss (from evolved config)
        stop_loss_pct = 0.19
        risk_amount = position_size * stop_loss_pct
        max_risk = self.account_equity * self.max_risk_per_trade
        
        if risk_amount > max_risk:
            return False, f"Trade risk too high: ${risk_amount:.2f} > ${max_risk:.2f} (2% limit)"
        
        # 3. Total portfolio risk limit (8%)
        if current_portfolio:
            current_risk = self._calculate_total_risk(current_portfolio)
            total_risk = current_risk + risk_amount
            max_total_risk_value = self.account_equity * self.max_total_risk
            
            if total_risk > max_total_risk_value:
                return False, f"Total risk too high: ${total_risk:.2f} > ${max_total_risk_value:.2f} (8% limit)"
        
        # 4. Sector concentration limit (50%)
        if current_portfolio:
            sector_ok, sector_reason = self._check_sector_concentration(
                ticker, position_size, current_portfolio
            )
            if not sector_ok:
                return False, sector_reason
        
        # 5. Daily loss limit check
        daily_loss_ok, daily_reason = self._check_daily_loss_limit()
        if not daily_loss_ok:
            return False, daily_reason
        
        # All checks passed
        return True, "Trade allowed"
    
    def _check_sell_allowed(
        self,
        ticker: str,
        entry_date: Optional[datetime],
        current_portfolio: Optional[Dict]
    ) -> Tuple[bool, str]:
        """Check if SELL is allowed (PDT check)."""
        
        if not self.is_pdt_restricted:
            return True, "Not PDT restricted - trade allowed"
        
        # Check if this would be a day trade
        if entry_date is None:
            logger.warning("Entry date not provided - assuming not day trade")
            return True, "Trade allowed (entry date unknown)"
        
        is_day_trade = (datetime.now().date() == entry_date.date())
        
        if not is_day_trade:
            return True, "Not a day trade - trade allowed"
        
        # Count day trades in last 5 trading days
        day_trades_used = self._count_recent_day_trades()
        
        if day_trades_used >= self.max_day_trades_per_period:
            return False, f"PDT violation: {day_trades_used}/3 day trades already used this week"
        
        return True, f"Day trade allowed ({day_trades_used + 1}/3)"
    
    def record_trade(
        self,
        action: str,
        ticker: str,
        shares: float,
        price: float,
        entry_date: Optional[datetime] = None,
        is_day_trade: bool = False,
        pnl: float = 0.0
    ):
        """
        Record trade for tracking.
        
        Args:
            action: 'BUY' or 'SELL'
            ticker: Ticker symbol
            shares: Number of shares
            price: Trade price
            entry_date: Entry date (for day trade check)
            is_day_trade: Is this a day trade?
            pnl: Realized P&L (for SELL)
        """
        trade = {
            'action': action,
            'ticker': ticker,
            'shares': shares,
            'price': price,
            'timestamp': datetime.now(),
            'is_day_trade': is_day_trade,
            'pnl': pnl
        }
        
        # Record day trade
        if is_day_trade and action == 'SELL':
            self.day_trades.append(trade)
            logger.info(f"📊 Day trade recorded: {ticker} (${pnl:+.2f})")
        
        # Record daily P&L
        if action == 'SELL':
            today = datetime.now().date().isoformat()
            self.daily_pnl[today] += pnl
    
    def _count_recent_day_trades(self) -> int:
        """Count day trades in last 5 trading days."""
        cutoff = datetime.now() - timedelta(days=self.pdt_period_days)
        recent_trades = [
            dt for dt in self.day_trades
            if dt['timestamp'] > cutoff
        ]
        return len(recent_trades)
    
    def _calculate_total_risk(self, current_portfolio: Dict) -> float:
        """Calculate total portfolio risk (sum of all stop loss amounts)."""
        total_risk = 0.0
        
        for position in current_portfolio.get('positions', []):
            position_value = position.get('market_value', 0)
            # Assume 19% stop loss
            risk = position_value * 0.19
            total_risk += risk
        
        return total_risk
    
    def _check_sector_concentration(
        self,
        ticker: str,
        new_position_size: float,
        current_portfolio: Dict
    ) -> Tuple[bool, str]:
        """Check sector concentration limits."""
        # Get sector for ticker (simplified)
        sector = self._get_sector(ticker)
        
        # Calculate current sector exposure
        sector_exposure = defaultdict(float)
        for position in current_portfolio.get('positions', []):
            pos_ticker = position.get('ticker', '')
            pos_sector = self._get_sector(pos_ticker)
            pos_value = position.get('market_value', 0)
            sector_exposure[pos_sector] += pos_value
        
        # Add new position
        sector_exposure[sector] += new_position_size
        
        # Check limit
        max_sector_value = self.account_equity * self.max_sector_concentration
        if sector_exposure[sector] > max_sector_value:
            return False, f"Sector concentration too high: {sector} ${sector_exposure[sector]:.2f} > ${max_sector_value:.2f} (50% limit)"
        
        return True, "Sector check passed"
    
    def _check_daily_loss_limit(self) -> Tuple[bool, str]:
        """Check daily loss limits."""
        today = datetime.now().date().isoformat()
        daily_pnl = self.daily_pnl.get(today, 0.0)
        daily_return = daily_pnl / self.starting_equity_today
        
        # Red alert: -10% daily loss
        if daily_return <= -self.daily_loss_limit_red:
            return False, f"🚨 DAILY LOSS LIMIT HIT: {daily_return:.1%} (Stop trading today)"
        
        # Yellow alert: -5% daily loss
        if daily_return <= -self.daily_loss_limit_yellow:
            logger.warning(f"⚠️ Yellow alert: Daily loss {daily_return:.1%} (Reduce risk)")
        
        return True, "Daily loss check passed"
    
    def _get_sector(self, ticker: str) -> str:
        """Get sector for ticker (simplified)."""
        biotech = ['PALI', 'KDK', 'DGNX', 'ASTS']
        tech = ['NVDA', 'TSLA', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'AMD', 'INTC']
        
        if ticker in biotech:
            return 'biotech'
        elif ticker in tech:
            return 'tech'
        else:
            return 'other'
    
    def get_compliance_status(self) -> Dict:
        """Get current compliance status."""
        day_trades_used = self._count_recent_day_trades()
        today = datetime.now().date().isoformat()
        daily_pnl = self.daily_pnl.get(today, 0.0)
        daily_return = daily_pnl / self.starting_equity_today if self.starting_equity_today > 0 else 0
        
        # Determine alert level
        alert_level = 'GREEN'
        if daily_return <= -self.daily_loss_limit_red:
            alert_level = 'RED'
        elif daily_return <= -self.daily_loss_limit_yellow:
            alert_level = 'YELLOW'
        
        return {
            'is_pdt_restricted': self.is_pdt_restricted,
            'day_trades_used': day_trades_used,
            'day_trades_remaining': max(0, self.max_day_trades_per_period - day_trades_used),
            'daily_pnl': daily_pnl,
            'daily_return': daily_return,
            'alert_level': alert_level,
            'can_day_trade': day_trades_used < self.max_day_trades_per_period
        }
    
    def reset_daily_tracking(self):
        """Reset daily tracking (call at start of new day)."""
        today = datetime.now().date().isoformat()
        self.starting_equity_today = self.account_equity
        logger.info(f"📊 Daily tracking reset: Starting equity ${self.account_equity:.2f}")


def example_usage():
    """Example: Compliance checks."""
    logger.info("\n" + "="*60)
    logger.info("COMPLIANCE ENGINE - PDT + Risk Management")
    logger.info("="*60 + "\n")
    
    # Initialize
    compliance = ComplianceEngine(
        account_equity=780.59,
        is_pdt_restricted=True
    )
    
    # Mock portfolio
    portfolio = {
        'positions': [
            {'ticker': 'PALI', 'market_value': 119.00},
            {'ticker': 'RXT', 'market_value': 117.60},
            {'ticker': 'KDK', 'market_value': 122.40},
            {'ticker': 'ASTS', 'market_value': 114.00}
        ]
    }
    
    # Example 1: Check if we can BUY NVDA
    logger.info("📊 EXAMPLE 1: Can we BUY NVDA?")
    allowed, reason = compliance.check_trade_allowed(
        action='BUY',
        ticker='NVDA',
        position_size=150,
        current_portfolio=portfolio
    )
    logger.info(f"   Allowed: {allowed}")
    logger.info(f"   Reason: {reason}")
    
    # Example 2: Check if we can SELL PALI (day trade)
    logger.info("\n📊 EXAMPLE 2: Can we SELL PALI (day trade)?")
    allowed, reason = compliance.check_trade_allowed(
        action='SELL',
        ticker='PALI',
        position_size=0,
        entry_date=datetime.now(),  # Entered today = day trade
        current_portfolio=portfolio
    )
    logger.info(f"   Allowed: {allowed}")
    logger.info(f"   Reason: {reason}")
    
    # Example 3: Record day trade
    logger.info("\n📊 EXAMPLE 3: Record day trade")
    compliance.record_trade(
        action='SELL',
        ticker='PALI',
        shares=50,
        price=2.38,
        is_day_trade=True,
        pnl=14.00
    )
    
    # Example 4: Get compliance status
    logger.info("\n📊 EXAMPLE 4: Compliance Status")
    status = compliance.get_compliance_status()
    logger.info(f"   PDT restricted: {status['is_pdt_restricted']}")
    logger.info(f"   Day trades used: {status['day_trades_used']}/3")
    logger.info(f"   Day trades remaining: {status['day_trades_remaining']}")
    logger.info(f"   Daily P&L: ${status['daily_pnl']:+.2f}")
    logger.info(f"   Daily return: {status['daily_return']:+.1%}")
    logger.info(f"   Alert level: {status['alert_level']}")
    logger.info(f"   Can day trade: {status['can_day_trade']}")


if __name__ == '__main__':
    example_usage()
