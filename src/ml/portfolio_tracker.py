"""
PORTFOLIO STATE TRACKER
========================
Track your portfolio positions, P&L, risk, and decision history.

Features:
- Real-time position tracking (shares, entry price, current P&L)
- PDT compliance (day trade counter, restrictions)
- Risk management (position sizing, stop losses, exposure)
- Trade history and learning (what worked, what didn't)
- Panic prevention (don't sell winners during normal dips)

This is YOUR COMPANION - it knows your portfolio state and helps you make better decisions.

Author: Quantum AI Trader
Date: December 10, 2025
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import json
import yfinance as yf

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Position:
    """Individual position in portfolio."""
    
    def __init__(
        self,
        ticker: str,
        shares: float,
        entry_price: float,
        entry_date: datetime,
        entry_confidence: float,
        cluster_id: int = None
    ):
        self.ticker = ticker
        self.shares = shares
        self.entry_price = entry_price
        self.entry_date = entry_date
        self.entry_confidence = entry_confidence
        self.cluster_id = cluster_id
        self.current_price = entry_price
        self.stop_loss_price = entry_price * 0.81  # -19% stop (from evolved_config)
        self.take_profit_price = entry_price * 1.15  # +15% target
        
    @property
    def market_value(self) -> float:
        """Current market value."""
        return self.shares * self.current_price
    
    @property
    def cost_basis(self) -> float:
        """Original cost."""
        return self.shares * self.entry_price
    
    @property
    def unrealized_pnl(self) -> float:
        """Unrealized profit/loss in dollars."""
        return self.market_value - self.cost_basis
    
    @property
    def unrealized_return(self) -> float:
        """Unrealized return as percentage."""
        return (self.current_price / self.entry_price - 1) if self.entry_price > 0 else 0
    
    @property
    def days_held(self) -> int:
        """Days since entry."""
        return (datetime.now() - self.entry_date).days
    
    @property
    def is_day_trade_candidate(self) -> bool:
        """Would selling today count as day trade?"""
        return (datetime.now().date() == self.entry_date.date())
    
    def update_price(self, new_price: float):
        """Update current price."""
        self.current_price = new_price
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for saving."""
        return {
            'ticker': self.ticker,
            'shares': self.shares,
            'entry_price': self.entry_price,
            'entry_date': self.entry_date.isoformat(),
            'entry_confidence': self.entry_confidence,
            'cluster_id': self.cluster_id,
            'current_price': self.current_price,
            'unrealized_pnl': self.unrealized_pnl,
            'unrealized_return': self.unrealized_return,
            'days_held': self.days_held
        }


class PortfolioTracker:
    """
    Track portfolio state and provide trading guidance.
    
    This is your AI companion's memory - it knows:
    - What you own
    - How much you're up/down
    - When you can trade (PDT)
    - What your past decisions were
    """
    
    def __init__(
        self,
        account_equity: float = 780.59,  # Your current equity
        buying_power: float = 186.10,    # Your current buying power
        is_pdt_restricted: bool = True,  # Under $25K
        max_position_size: float = 0.21, # 21% per position (from evolved_config)
        max_day_trades_per_week: int = 3,
        portfolio_file: str = 'data/portfolio_state.json'
    ):
        """
        Args:
            account_equity: Total account value
            buying_power: Available cash
            is_pdt_restricted: PDT restricted account?
            max_position_size: Max % per position
            max_day_trades_per_week: PDT limit
            portfolio_file: File to persist state
        """
        self.account_equity = account_equity
        self.buying_power = buying_power
        self.is_pdt_restricted = is_pdt_restricted
        self.max_position_size = max_position_size
        self.max_day_trades_per_week = max_day_trades_per_week
        self.portfolio_file = Path(portfolio_file)
        
        # Positions
        self.positions: Dict[str, Position] = {}
        
        # Day trade tracking (last 5 trading days)
        self.day_trades_this_week: List[Dict] = []
        
        # Trade history (for learning)
        self.trade_history: List[Dict] = []
        
        # Load from file if exists
        self._load_state()
    
    def add_position(
        self,
        ticker: str,
        shares: float,
        entry_price: float,
        entry_confidence: float,
        cluster_id: int = None
    ) -> Tuple[bool, str]:
        """
        Add new position.
        
        Returns:
            (success, message)
        """
        # Check if position already exists
        if ticker in self.positions:
            return False, f"Already holding {ticker} (use add_to_position instead)"
        
        # Check position size
        position_value = shares * entry_price
        max_position_value = self.account_equity * self.max_position_size
        
        if position_value > max_position_value:
            return False, f"Position too large (${position_value:.2f} > ${max_position_value:.2f} max)"
        
        # Check buying power
        if position_value > self.buying_power:
            return False, f"Insufficient buying power (${position_value:.2f} needed, ${self.buying_power:.2f} available)"
        
        # Create position
        position = Position(
            ticker=ticker,
            shares=shares,
            entry_price=entry_price,
            entry_date=datetime.now(),
            entry_confidence=entry_confidence,
            cluster_id=cluster_id
        )
        
        self.positions[ticker] = position
        self.buying_power -= position_value
        
        logger.info(f"✅ Added position: {ticker}")
        logger.info(f"   Shares: {shares}")
        logger.info(f"   Entry: ${entry_price:.2f}")
        logger.info(f"   Confidence: {entry_confidence:.1f}%")
        logger.info(f"   Position value: ${position_value:.2f}")
        
        self._save_state()
        
        return True, "Position added"
    
    def close_position(
        self,
        ticker: str,
        exit_price: float,
        reason: str = "Manual exit"
    ) -> Tuple[bool, str]:
        """
        Close position.
        
        Returns:
            (success, message)
        """
        if ticker not in self.positions:
            return False, f"No position in {ticker}"
        
        position = self.positions[ticker]
        
        # Calculate P&L
        exit_value = position.shares * exit_price
        realized_pnl = exit_value - position.cost_basis
        realized_return = (exit_price / position.entry_price - 1)
        
        # Check if day trade
        is_day_trade = position.is_day_trade_candidate
        
        if is_day_trade and self.is_pdt_restricted:
            day_trades_used = len([dt for dt in self.day_trades_this_week 
                                  if dt['date'] > datetime.now() - timedelta(days=5)])
            
            if day_trades_used >= self.max_day_trades_per_week:
                return False, f"PDT violation: Already used {day_trades_used}/3 day trades this week"
            
            # Record day trade
            self.day_trades_this_week.append({
                'ticker': ticker,
                'date': datetime.now(),
                'pnl': realized_pnl
            })
        
        # Record in history
        trade = {
            'ticker': ticker,
            'entry_date': position.entry_date.isoformat(),
            'exit_date': datetime.now().isoformat(),
            'entry_price': position.entry_price,
            'exit_price': exit_price,
            'shares': position.shares,
            'realized_pnl': realized_pnl,
            'realized_return': realized_return,
            'days_held': position.days_held,
            'entry_confidence': position.entry_confidence,
            'is_day_trade': is_day_trade,
            'reason': reason
        }
        
        self.trade_history.append(trade)
        
        # Update account
        self.buying_power += exit_value
        self.account_equity += realized_pnl
        
        # Remove position
        del self.positions[ticker]
        
        logger.info(f"✅ Closed position: {ticker}")
        logger.info(f"   Exit: ${exit_price:.2f}")
        logger.info(f"   P&L: ${realized_pnl:+.2f} ({realized_return:+.2%})")
        logger.info(f"   Days held: {trade['days_held']}")
        logger.info(f"   Reason: {reason}")
        if is_day_trade:
            logger.warning(f"   ⚠️ Day trade used ({len(self.day_trades_this_week)}/3 this week)")
        
        self._save_state()
        
        return True, f"Position closed: ${realized_pnl:+.2f}"
    
    def update_prices(self, price_data: Dict[str, float]):
        """
        Update current prices for all positions.
        
        Args:
            price_data: {ticker: current_price}
        """
        for ticker, position in self.positions.items():
            if ticker in price_data:
                position.update_price(price_data[ticker])
    
    def get_current_state(self) -> Dict:
        """Get current portfolio snapshot."""
        total_market_value = sum(p.market_value for p in self.positions.values())
        total_unrealized_pnl = sum(p.unrealized_pnl for p in self.positions.values())
        total_cost_basis = sum(p.cost_basis for p in self.positions.values())
        
        return {
            'account_equity': self.account_equity,
            'buying_power': self.buying_power,
            'positions_value': total_market_value,
            'cash': self.buying_power,
            'total_value': self.buying_power + total_market_value,
            'unrealized_pnl': total_unrealized_pnl,
            'n_positions': len(self.positions),
            'day_trades_used': len([dt for dt in self.day_trades_this_week 
                                   if dt['date'] > datetime.now() - timedelta(days=5)]),
            'positions': [p.to_dict() for p in self.positions.values()]
        }
    
    def should_exit_position(
        self,
        ticker: str,
        model_prediction: Dict
    ) -> Tuple[bool, str]:
        """
        Should we exit this position?
        
        Considers:
        - Stop loss hit?
        - Take profit hit?
        - Max hold time exceeded?
        - Model says SELL with high confidence?
        - Normal dip vs. real problem?
        
        Args:
            ticker: Ticker symbol
            model_prediction: Latest model prediction
            
        Returns:
            (should_exit, reason)
        """
        if ticker not in self.positions:
            return False, "No position"
        
        position = self.positions[ticker]
        current_return = position.unrealized_return
        
        # 1. Stop loss hit? (-19%)
        if current_return <= -0.19:
            return True, f"Stop loss hit ({current_return:.1%})"
        
        # 2. Take profit hit? (+15%)
        if current_return >= 0.15:
            return True, f"Take profit hit ({current_return:.1%})"
        
        # 3. Max hold time exceeded? (32 days)
        if position.days_held > 32:
            if current_return > 0:
                return True, f"Max hold exceeded ({position.days_held} days) - take profit"
            else:
                return True, f"Max hold exceeded ({position.days_held} days) - cut loss"
        
        # 4. Model says SELL with high confidence?
        if model_prediction['signal'] == 'SELL' and model_prediction['confidence'] > 80:
            return True, f"Model SELL signal ({model_prediction['confidence']:.0f}% confidence)"
        
        # 5. Don't panic on small dips!
        if -0.08 < current_return < 0:
            # You're down 0-8% - this is NORMAL
            # 87% of your dips bounce within 2 hours (from your trading pattern)
            if model_prediction['signal'] in ['BUY', 'HOLD']:
                return False, f"Normal dip ({current_return:.1%}) - model still {model_prediction['signal']}"
        
        # 6. Winning position - let it run if model still bullish
        if current_return > 0.05 and model_prediction['signal'] in ['BUY', 'HOLD']:
            return False, f"Winning position ({current_return:.1%}) - model still {model_prediction['signal']}"
        
        # Default: HOLD
        return False, f"HOLD - no exit criteria met ({current_return:+.1%})"
    
    def get_trading_summary(self) -> Dict:
        """
        Get summary of trading performance.
        
        Includes:
        - Win rate
        - Avg profit/loss
        - Best/worst trades
        - Trading patterns
        """
        if not self.trade_history:
            return {'message': 'No trades yet'}
        
        trades_df = pd.DataFrame(self.trade_history)
        
        wins = trades_df[trades_df['realized_pnl'] > 0]
        losses = trades_df[trades_df['realized_pnl'] <= 0]
        
        summary = {
            'total_trades': len(trades_df),
            'wins': len(wins),
            'losses': len(losses),
            'win_rate': len(wins) / len(trades_df) if len(trades_df) > 0 else 0,
            'total_pnl': trades_df['realized_pnl'].sum(),
            'avg_win': wins['realized_pnl'].mean() if len(wins) > 0 else 0,
            'avg_loss': losses['realized_pnl'].mean() if len(losses) > 0 else 0,
            'best_trade': trades_df.loc[trades_df['realized_pnl'].idxmax()].to_dict() if len(trades_df) > 0 else None,
            'worst_trade': trades_df.loc[trades_df['realized_pnl'].idxmin()].to_dict() if len(trades_df) > 0 else None,
            'avg_hold_time': trades_df['days_held'].mean(),
            'day_trades': len(trades_df[trades_df['is_day_trade']])
        }
        
        return summary
    
    def _save_state(self):
        """Save portfolio state to file."""
        state = {
            'account_equity': self.account_equity,
            'buying_power': self.buying_power,
            'positions': [p.to_dict() for p in self.positions.values()],
            'day_trades_this_week': [
                {**dt, 'date': dt['date'].isoformat()}
                for dt in self.day_trades_this_week
            ],
            'trade_history': self.trade_history,
            'last_updated': datetime.now().isoformat()
        }
        
        self.portfolio_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.portfolio_file, 'w') as f:
            json.dump(state, f, indent=2)
    
    def _load_state(self):
        """Load portfolio state from file."""
        if not self.portfolio_file.exists():
            logger.info("No saved state found - starting fresh")
            return
        
        with open(self.portfolio_file, 'r') as f:
            state = json.load(f)
        
        self.account_equity = state.get('account_equity', self.account_equity)
        self.buying_power = state.get('buying_power', self.buying_power)
        
        # Restore positions
        for pos_dict in state.get('positions', []):
            position = Position(
                ticker=pos_dict['ticker'],
                shares=pos_dict['shares'],
                entry_price=pos_dict['entry_price'],
                entry_date=datetime.fromisoformat(pos_dict['entry_date']),
                entry_confidence=pos_dict['entry_confidence'],
                cluster_id=pos_dict.get('cluster_id')
            )
            position.current_price = pos_dict.get('current_price', pos_dict['entry_price'])
            self.positions[pos_dict['ticker']] = position
        
        # Restore day trades
        self.day_trades_this_week = [
            {**dt, 'date': datetime.fromisoformat(dt['date'])}
            for dt in state.get('day_trades_this_week', [])
        ]
        
        # Restore history
        self.trade_history = state.get('trade_history', [])
        
        logger.info(f"✅ Loaded portfolio state: {len(self.positions)} positions, {len(self.trade_history)} trades in history")


def example_usage():
    """Example: Your real portfolio."""
    logger.info("\n" + "="*60)
    logger.info("PORTFOLIO TRACKER - Your Current Portfolio")
    logger.info("="*60 + "\n")
    
    # Initialize with YOUR current state
    portfolio = PortfolioTracker(
        account_equity=780.59,
        buying_power=186.10,
        is_pdt_restricted=True
    )
    
    # Add your current positions (example - replace with real data)
    portfolio.add_position('PALI', 50, 2.10, entry_confidence=85, cluster_id=2)
    portfolio.add_position('RXT', 30, 3.50, entry_confidence=78, cluster_id=1)
    portfolio.add_position('KDK', 40, 2.80, entry_confidence=82, cluster_id=2)
    portfolio.add_position('ASTS', 25, 4.20, entry_confidence=75, cluster_id=3)
    
    # Update with current prices
    portfolio.update_prices({
        'PALI': 2.38,  # +13.16%
        'RXT': 3.92,   # +12.15%
        'KDK': 3.06,   # +9.08%
        'ASTS': 4.56   # +8.46%
    })
    
    # Get current state
    state = portfolio.get_current_state()
    logger.info(f"📊 PORTFOLIO STATE")
    logger.info(f"   Total value: ${state['total_value']:.2f}")
    logger.info(f"   Cash: ${state['cash']:.2f}")
    logger.info(f"   Positions: {state['n_positions']}")
    logger.info(f"   Unrealized P&L: ${state['unrealized_pnl']:+.2f}")
    logger.info(f"   Day trades used: {state['day_trades_used']}/3")
    
    logger.info(f"\n📈 POSITIONS")
    for pos in state['positions']:
        logger.info(f"   {pos['ticker']}: {pos['shares']} shares @ ${pos['entry_price']:.2f}")
        logger.info(f"      Current: ${pos['current_price']:.2f} ({pos['unrealized_return']:+.2%})")
        logger.info(f"      P&L: ${pos['unrealized_pnl']:+.2f}")
        logger.info(f"      Days held: {pos['days_held']}")
    
    # Check exit decisions (example)
    logger.info(f"\n🤔 EXIT DECISIONS")
    for ticker in portfolio.positions.keys():
        # Simulate model prediction
        mock_prediction = {'signal': 'HOLD', 'confidence': 70}
        should_exit, reason = portfolio.should_exit_position(ticker, mock_prediction)
        logger.info(f"   {ticker}: {'EXIT' if should_exit else 'HOLD'} - {reason}")


if __name__ == '__main__':
    example_usage()
