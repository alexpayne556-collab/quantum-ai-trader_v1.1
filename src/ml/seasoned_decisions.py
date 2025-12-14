"""
SEASONED DECISIONS ENGINE
=========================
Your trading wisdom coded into the AI companion.

This module codifies YOUR proven trading patterns:
- 87% of your dips bounce within 2 hours
- Biotech dips work (PALI +13%, KDK +9%)
- Single-share positions don't work (too small)
- Don't panic sell winners during normal dips (0-8% down)
- Take profit at 15% (your sweet spot)
- Cut losses at -19% (evolved threshold)

This is YOUR EXPERIENCE in code - the AI learns from what YOU know works.

Author: Quantum AI Trader
Date: December 10, 2025
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SeasonedTrader:
    """
    Codified trading wisdom from real experience.
    """
    
    def __init__(self):
        """Initialize with proven trading rules."""
        # Your proven patterns
        self.patterns = {
            'dip_bounce_rate': 0.87,  # 87% of dips bounce
            'dip_bounce_hours': 2,    # Usually within 2 hours
            'biotech_edge': True,     # You have biotech edge
            'min_position_value': 100, # Minimum $100 position
            'sweet_spot_profit': 0.15, # 15% take profit
            'max_loss': -0.19,        # -19% stop loss
            'normal_dip_threshold': -0.08,  # -8% is normal dip
            'revenge_trading_cooldown': 1,  # 1 day cooldown after loss
        }
        
        # Sector expertise
        self.sector_edges = {
            'biotech': {'confidence_boost': 0.05, 'reason': 'Proven edge in biotech dips'},
            'tech': {'confidence_boost': 0.02, 'reason': 'Good tech pattern recognition'},
        }
        
        # Anti-patterns (what NOT to do)
        self.avoid_patterns = {
            'single_share_positions': 'Too small to matter, wastes mental energy',
            'revenge_trading': 'Leads to losses, take cooldown after big loss',
            'panic_selling_dips': '87% of dips bounce, trust the pattern',
            'holding_too_long': 'Max 32 days, take profit at 15%',
        }
    
    def should_enter_trade(
        self,
        ticker: str,
        signal: Dict,
        position_size: float,
        current_portfolio: Dict
    ) -> Tuple[bool, str, Dict]:
        """
        Should we enter this trade based on seasoned wisdom?
        
        Args:
            ticker: Ticker symbol
            signal: Model signal {confidence, signal, entry_quality, ...}
            position_size: Proposed position size ($)
            current_portfolio: Current portfolio state
            
        Returns:
            (should_enter, reason, adjustments)
        """
        adjustments = {}
        
        # 1. Minimum position size check
        if position_size < self.patterns['min_position_value']:
            return False, f"Position too small (${position_size:.2f} < ${self.patterns['min_position_value']})", adjustments
        
        # 2. Check for revenge trading
        if self._is_revenge_trading(current_portfolio):
            return False, "Revenge trading detected - take 1 day cooldown after big loss", adjustments
        
        # 3. Apply sector edge
        sector = self._get_sector(ticker)
        if sector in self.sector_edges:
            edge = self.sector_edges[sector]
            confidence_boost = edge['confidence_boost']
            adjustments['confidence_boost'] = confidence_boost
            adjustments['reason'] = edge['reason']
            logger.info(f"✨ Sector edge detected: {sector} (+{confidence_boost:.1%} confidence)")
        
        # 4. Check if ticker has biotech characteristics
        if self._is_biotech_dip(ticker, signal):
            adjustments['biotech_dip'] = True
            adjustments['confidence_boost'] = adjustments.get('confidence_boost', 0) + 0.05
            logger.info(f"✨ Biotech dip pattern - your proven edge!")
        
        # 5. Model confidence check (with boosts)
        adjusted_confidence = signal['confidence'] + adjustments.get('confidence_boost', 0) * 100
        if adjusted_confidence < 70:
            return False, f"Confidence too low ({adjusted_confidence:.1f}%)", adjustments
        
        # 6. Entry quality check
        if signal.get('entry_quality', 0) < 60:
            return False, f"Entry quality too low ({signal.get('entry_quality', 0):.1f}/100)", adjustments
        
        # All checks passed
        reason = f"Good entry: {adjusted_confidence:.1f}% confidence"
        if adjustments.get('biotech_dip'):
            reason += " + biotech dip edge"
        
        return True, reason, adjustments
    
    def should_exit_position(
        self,
        ticker: str,
        position: Dict,
        model_signal: Dict,
        market_data: Dict
    ) -> Tuple[bool, str]:
        """
        Should we exit this position based on seasoned wisdom?
        
        Args:
            ticker: Ticker symbol
            position: Position data {entry_price, current_price, unrealized_return, days_held, ...}
            model_signal: Latest model signal
            market_data: Current market data
            
        Returns:
            (should_exit, reason)
        """
        current_return = position['unrealized_return']
        days_held = position['days_held']
        
        # 1. Hard stop loss (-19%)
        if current_return <= self.patterns['max_loss']:
            return True, f"Stop loss hit ({current_return:.1%})"
        
        # 2. Take profit at sweet spot (15%+)
        if current_return >= self.patterns['sweet_spot_profit']:
            return True, f"Sweet spot profit ({current_return:.1%}) - take it!"
        
        # 3. Max hold time (32 days)
        if days_held > 32:
            if current_return > 0:
                return True, f"Max hold reached ({days_held} days) - lock in profit"
            else:
                return True, f"Max hold reached ({days_held} days) - cut loss"
        
        # 4. DON'T PANIC on normal dips!
        if self.patterns['normal_dip_threshold'] < current_return < 0:
            # You're down 0-8% - this is NORMAL
            if model_signal['signal'] in ['BUY', 'HOLD']:
                # 87% of these bounce within 2 hours
                return False, f"Normal dip ({current_return:.1%}) - 87% bounce, model still {model_signal['signal']}"
        
        # 5. Winning position - let it run if model still bullish
        if current_return > 0.05:
            if model_signal['signal'] in ['BUY', 'HOLD']:
                return False, f"Winning position ({current_return:.1%}) - let it run, model {model_signal['signal']}"
            elif model_signal['confidence'] > 80 and model_signal['signal'] == 'SELL':
                return True, f"Model strong SELL ({model_signal['confidence']:.0f}%) - exit winner"
        
        # 6. Biotech positions - extra patience
        if self._is_biotech_ticker(ticker):
            if current_return > -0.10 and model_signal['signal'] != 'SELL':
                return False, f"Biotech dip ({current_return:.1%}) - your edge, hold"
        
        # Default: HOLD
        return False, f"Hold - no exit criteria met ({current_return:+.1%})"
    
    def suggest_position_size(
        self,
        ticker: str,
        signal: Dict,
        account_equity: float,
        current_positions: int
    ) -> float:
        """
        Suggest position size based on seasoned wisdom.
        
        Args:
            ticker: Ticker symbol
            signal: Model signal
            account_equity: Account equity
            current_positions: Number of current positions
            
        Returns:
            Position size as fraction of equity
        """
        # Base: 21% per position (from evolved config)
        base_size = 0.21
        
        # Adjust for confidence
        if signal['confidence'] > 85:
            base_size *= 1.2  # 25% for high-confidence
        elif signal['confidence'] < 75:
            base_size *= 0.8  # 17% for lower-confidence
        
        # Adjust for number of positions (diversification)
        if current_positions >= 4:
            base_size *= 0.8  # Smaller positions when diversified
        
        # Biotech edge - slightly larger positions
        if self._is_biotech_ticker(ticker):
            base_size *= 1.1
        
        # Cap at 25% (don't go crazy)
        base_size = min(base_size, 0.25)
        
        # Ensure minimum $100 position
        min_size = self.patterns['min_position_value'] / account_equity
        base_size = max(base_size, min_size)
        
        return base_size
    
    def _is_revenge_trading(self, current_portfolio: Dict) -> bool:
        """Detect revenge trading pattern."""
        # Check if last trade was a big loss in last 24 hours
        history = current_portfolio.get('trade_history', [])
        if not history:
            return False
        
        last_trade = history[-1]
        last_exit = datetime.fromisoformat(last_trade['exit_date'])
        hours_since = (datetime.now() - last_exit).total_seconds() / 3600
        
        # If last trade was >-10% loss and <24 hours ago
        if last_trade['realized_return'] < -0.10 and hours_since < 24:
            logger.warning(f"⚠️ Revenge trading detected: Big loss {last_trade['realized_return']:.1%} {hours_since:.1f}h ago")
            return True
        
        return False
    
    def _is_biotech_dip(self, ticker: str, signal: Dict) -> bool:
        """Check if this is a biotech dip (your proven edge)."""
        # Simple check: Is ticker likely biotech and showing dip pattern?
        if not self._is_biotech_ticker(ticker):
            return False
        
        # Check for dip pattern (negative recent returns but model bullish)
        if signal['signal'] == 'BUY' and signal.get('entry_quality', 0) > 70:
            return True
        
        return False
    
    def _is_biotech_ticker(self, ticker: str) -> bool:
        """Simple biotech detection (can be enhanced with sector data)."""
        # Your biotech holdings: PALI, KDK
        biotech_tickers = ['PALI', 'KDK', 'DGNX', 'ASTS']  # Add known biotech
        return ticker in biotech_tickers
    
    def _get_sector(self, ticker: str) -> str:
        """Get sector for ticker (simplified - can use yfinance in production)."""
        if self._is_biotech_ticker(ticker):
            return 'biotech'
        
        tech_tickers = ['NVDA', 'TSLA', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'AMD', 'INTC']
        if ticker in tech_tickers:
            return 'tech'
        
        return 'unknown'
    
    def get_trading_wisdom(self) -> str:
        """Get summary of trading wisdom."""
        wisdom = """
        🧠 SEASONED TRADING WISDOM
        =========================
        
        ✅ DO:
        - Trust biotech dips (87% bounce rate)
        - Take profit at 15% (your sweet spot)
        - Position size: 21% base, adjust for confidence
        - Hold through normal dips (0-8% down)
        - Max hold: 32 days
        
        ❌ DON'T:
        - Trade single-share positions (<$100)
        - Revenge trade after big loss (24h cooldown)
        - Panic sell during 0-8% dips (87% bounce)
        - Hold past 15% profit (take it!)
        - Hold past -19% loss (cut it!)
        
        🎯 YOUR EDGES:
        - Biotech dips: +5% confidence boost
        - Tech patterns: +2% confidence boost
        - Dip timing: 87% bounce within 2 hours
        
        📊 PROVEN PATTERNS:
        - PALI: +13.16% (biotech dip worked)
        - KDK: +9.08% (biotech dip worked)
        - RXT: +12.15% (pattern recognition)
        - ASTS: +8.46% (dip entry)
        """
        return wisdom


def example_usage():
    """Example: Apply seasoned wisdom to trading decision."""
    logger.info("\n" + "="*60)
    logger.info("SEASONED TRADER - Your Wisdom in Code")
    logger.info("="*60 + "\n")
    
    trader = SeasonedTrader()
    
    # Print wisdom
    print(trader.get_trading_wisdom())
    
    # Example 1: Should we enter PALI?
    logger.info("\n📊 EXAMPLE 1: Entry Decision (PALI)")
    signal = {
        'ticker': 'PALI',
        'confidence': 72,
        'signal': 'BUY',
        'entry_quality': 85
    }
    
    portfolio = {
        'n_positions': 3,
        'trade_history': []
    }
    
    should_enter, reason, adjustments = trader.should_enter_trade(
        ticker='PALI',
        signal=signal,
        position_size=150,
        current_portfolio=portfolio
    )
    
    logger.info(f"   Decision: {'ENTER' if should_enter else 'SKIP'}")
    logger.info(f"   Reason: {reason}")
    if adjustments:
        logger.info(f"   Adjustments: {adjustments}")
    
    # Example 2: Should we exit winning PALI position?
    logger.info("\n📊 EXAMPLE 2: Exit Decision (PALI +13%)")
    position = {
        'ticker': 'PALI',
        'entry_price': 2.10,
        'current_price': 2.38,
        'unrealized_return': 0.1316,
        'days_held': 2
    }
    
    model_signal = {
        'signal': 'HOLD',
        'confidence': 75
    }
    
    should_exit, reason = trader.should_exit_position(
        ticker='PALI',
        position=position,
        model_signal=model_signal,
        market_data={}
    )
    
    logger.info(f"   Decision: {'EXIT' if should_exit else 'HOLD'}")
    logger.info(f"   Reason: {reason}")
    
    # Example 3: Position sizing
    logger.info("\n📊 EXAMPLE 3: Position Sizing (NVDA)")
    size = trader.suggest_position_size(
        ticker='NVDA',
        signal={'confidence': 87},
        account_equity=780.59,
        current_positions=4
    )
    
    logger.info(f"   Suggested size: {size:.1%} of equity")
    logger.info(f"   Dollar amount: ${780.59 * size:.2f}")


if __name__ == '__main__':
    example_usage()
