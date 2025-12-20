#!/usr/bin/env python3
"""
=============================================================================
ALPACA PAPER TRADING INTEGRATION
=============================================================================

Connects the Quantum Portfolio Engine to Alpaca for paper trading.

Setup:
1. Get Alpaca API keys from https://app.alpaca.markets/
2. Set environment variables:
   export ALPACA_API_KEY="your-key"
   export ALPACA_SECRET_KEY="your-secret"
   
Or create a .env file with these values.

Run:
    python ALPACA_PAPER_TRADER.py
    
Commands:
    python ALPACA_PAPER_TRADER.py status    # Check current positions
    python ALPACA_PAPER_TRADER.py signals   # Check current signals
    python ALPACA_PAPER_TRADER.py trade     # Execute trades based on signals
    python ALPACA_PAPER_TRADER.py backtest  # Run backtest comparison
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np

# Try to import alpaca - will fail gracefully if not installed
try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import MarketOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    print("⚠️ Alpaca not installed. Run: pip install alpaca-py")

# Import our portfolio engine
from CRITICAL_FIXES_AND_PORTFOLIO_ENGINE import (
    QuantumPortfolioEngine, 
    CorrectedSignals,
    CorrectedValidation
)


# ============================================================================
# CONFIGURATION
# ============================================================================

# Default selected signals from orthogonal analysis
SELECTED_SIGNALS = ['H20', 'H128', 'H62', 'H19']

# Risk parameters
MAX_POSITION_PCT = 0.25      # Max 25% per signal
MIN_TRADE_VALUE = 100        # Minimum $100 per trade
MAX_TRADE_VALUE = 10000      # Maximum $10,000 per trade (paper)

# Trading parameters
TRADING_SYMBOL = 'SPY'       # Trade SPY
PAPER_TRADING = True         # Always paper trade


# ============================================================================
# ALPACA CLIENT
# ============================================================================

class AlpacaClient:
    """Wrapper for Alpaca API."""
    
    def __init__(self):
        self.api_key = os.getenv('ALPACA_API_KEY', '')
        self.secret_key = os.getenv('ALPACA_SECRET_KEY', '')
        
        if not self.api_key or not self.secret_key:
            print("\n⚠️ Alpaca API keys not found!")
            print("Set environment variables:")
            print("  export ALPACA_API_KEY='your-key'")
            print("  export ALPACA_SECRET_KEY='your-secret'")
            print("\nOr get keys from: https://app.alpaca.markets/")
            self.client = None
            self.data_client = None
            return
        
        # Initialize clients (paper trading)
        self.client = TradingClient(
            self.api_key, 
            self.secret_key, 
            paper=True
        )
        
        self.data_client = StockHistoricalDataClient(
            self.api_key,
            self.secret_key
        )
    
    def get_account(self):
        """Get account information."""
        if not self.client:
            return None
        return self.client.get_account()
    
    def get_positions(self):
        """Get current positions."""
        if not self.client:
            return []
        return self.client.get_all_positions()
    
    def get_spy_position(self):
        """Get SPY position specifically."""
        positions = self.get_positions()
        for pos in positions:
            if pos.symbol == 'SPY':
                return {
                    'qty': float(pos.qty),
                    'market_value': float(pos.market_value),
                    'avg_entry': float(pos.avg_entry_price),
                    'unrealized_pl': float(pos.unrealized_pl),
                    'unrealized_plpc': float(pos.unrealized_plpc),
                }
        return None
    
    def place_market_order(self, symbol: str, qty: float, side: str) -> dict:
        """Place a market order."""
        if not self.client:
            return {'error': 'No client'}
        
        order_side = OrderSide.BUY if side.lower() == 'buy' else OrderSide.SELL
        
        request = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=order_side,
            time_in_force=TimeInForce.DAY
        )
        
        order = self.client.submit_order(request)
        return {
            'id': order.id,
            'symbol': order.symbol,
            'qty': order.qty,
            'side': order.side,
            'status': order.status,
        }
    
    def get_latest_price(self, symbol: str) -> float:
        """Get latest price for a symbol."""
        if not self.data_client:
            return 0
        
        bars = self.data_client.get_stock_bars(
            StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Minute,
                start=datetime.now() - timedelta(minutes=5),
            )
        )
        
        if bars and symbol in bars:
            return float(bars[symbol][-1].close)
        return 0


# ============================================================================
# PAPER TRADING ENGINE
# ============================================================================

class AlpacaPaperTrader:
    """
    Paper trading engine using Quantum Portfolio signals.
    """
    
    def __init__(self):
        self.alpaca = AlpacaClient()
        self.engine = QuantumPortfolioEngine(SELECTED_SIGNALS)
        self.trade_log = []
        
    def check_status(self):
        """Print current account and position status."""
        print("\n" + "="*60)
        print("ALPACA PAPER TRADING STATUS")
        print("="*60)
        
        if not self.alpaca.client:
            print("❌ Alpaca not connected. Set API keys.")
            return
        
        account = self.alpaca.get_account()
        
        print(f"\n💰 Account Summary:")
        print(f"   Equity: ${float(account.equity):,.2f}")
        print(f"   Cash: ${float(account.cash):,.2f}")
        print(f"   Buying Power: ${float(account.buying_power):,.2f}")
        print(f"   Day Trade Count: {account.daytrade_count}")
        
        spy_pos = self.alpaca.get_spy_position()
        
        print(f"\n📊 SPY Position:")
        if spy_pos:
            print(f"   Shares: {spy_pos['qty']}")
            print(f"   Market Value: ${spy_pos['market_value']:,.2f}")
            print(f"   Avg Entry: ${spy_pos['avg_entry']:.2f}")
            print(f"   Unrealized P&L: ${spy_pos['unrealized_pl']:,.2f} ({spy_pos['unrealized_plpc']*100:.1f}%)")
        else:
            print("   No SPY position")
        
        # Current allocation
        equity = float(account.equity)
        spy_value = spy_pos['market_value'] if spy_pos else 0
        allocation = spy_value / equity if equity > 0 else 0
        
        print(f"\n📈 Allocation:")
        print(f"   SPY: {allocation:.1%}")
        print(f"   Cash: {1-allocation:.1%}")
    
    def check_signals(self):
        """Print current signal status."""
        alloc = self.engine.print_allocation()
        return alloc
    
    def calculate_target_position(self) -> dict:
        """Calculate target position based on signals."""
        alloc = self.engine.get_portfolio_allocation()
        
        if not self.alpaca.client:
            return {'error': 'No Alpaca connection'}
        
        account = self.alpaca.get_account()
        equity = float(account.equity)
        
        # Target SPY allocation
        target_allocation = alloc['total_exposure']
        target_value = equity * target_allocation
        
        # Current SPY position
        spy_pos = self.alpaca.get_spy_position()
        current_value = spy_pos['market_value'] if spy_pos else 0
        current_shares = spy_pos['qty'] if spy_pos else 0
        
        # Calculate trade needed
        trade_value = target_value - current_value
        spy_price = alloc['spy_price']
        trade_shares = int(trade_value / spy_price)
        
        return {
            'equity': equity,
            'target_allocation': target_allocation,
            'target_value': target_value,
            'current_value': current_value,
            'current_shares': current_shares,
            'trade_value': trade_value,
            'trade_shares': trade_shares,
            'spy_price': spy_price,
            'signals': alloc['signals'],
        }
    
    def execute_trades(self, confirm: bool = True):
        """Execute trades to reach target allocation."""
        print("\n" + "="*60)
        print("TRADE EXECUTION")
        print("="*60)
        
        if not self.alpaca.client:
            print("❌ Alpaca not connected.")
            return
        
        target = self.calculate_target_position()
        
        print(f"\n📊 Current State:")
        print(f"   Account Equity: ${target['equity']:,.2f}")
        print(f"   Current SPY: ${target['current_value']:,.2f} ({target['current_shares']:.0f} shares)")
        
        print(f"\n🎯 Target State:")
        print(f"   Target Allocation: {target['target_allocation']:.1%}")
        print(f"   Target SPY Value: ${target['target_value']:,.2f}")
        
        print(f"\n📝 Required Trade:")
        if target['trade_shares'] > 0:
            print(f"   Action: BUY {target['trade_shares']} shares SPY")
            print(f"   Value: ${target['trade_value']:,.2f}")
            side = 'buy'
        elif target['trade_shares'] < 0:
            print(f"   Action: SELL {abs(target['trade_shares'])} shares SPY")
            print(f"   Value: ${abs(target['trade_value']):,.2f}")
            side = 'sell'
        else:
            print(f"   Action: No trade needed")
            return
        
        # Check minimum trade size
        if abs(target['trade_value']) < MIN_TRADE_VALUE:
            print(f"\n⚠️ Trade value below minimum (${MIN_TRADE_VALUE}). Skipping.")
            return
        
        # Confirm before trading
        if confirm:
            response = input("\n⚠️ Execute trade? (yes/no): ")
            if response.lower() != 'yes':
                print("Trade cancelled.")
                return
        
        # Execute trade
        print(f"\n🚀 Executing {side.upper()} order...")
        
        try:
            order = self.alpaca.place_market_order(
                symbol='SPY',
                qty=abs(target['trade_shares']),
                side=side
            )
            
            print(f"✅ Order submitted!")
            print(f"   Order ID: {order['id']}")
            print(f"   Status: {order['status']}")
            
            # Log trade
            self.trade_log.append({
                'timestamp': datetime.now().isoformat(),
                'side': side,
                'shares': abs(target['trade_shares']),
                'signals': target['signals'],
                'order_id': order['id'],
            })
            
        except Exception as e:
            print(f"❌ Order failed: {e}")
    
    def run_daily_check(self):
        """Run daily signal check and optional trade."""
        print("\n" + "="*70)
        print(f"QUANTUM TRADER - DAILY CHECK ({datetime.now().strftime('%Y-%m-%d %H:%M')})")
        print("="*70)
        
        # Check signals
        alloc = self.check_signals()
        
        # Show target position
        target = self.calculate_target_position()
        
        print("\n" + "-"*60)
        print("TRADE RECOMMENDATION")
        print("-"*60)
        
        if target['trade_shares'] > 0:
            print(f"📈 BUY {target['trade_shares']} SPY shares (${target['trade_value']:,.2f})")
        elif target['trade_shares'] < 0:
            print(f"📉 SELL {abs(target['trade_shares'])} SPY shares (${abs(target['trade_value']):,.2f})")
        else:
            print("📊 No trade needed - position aligned with signals")
        
        return target


# ============================================================================
# SIMULATION MODE (No API Keys)
# ============================================================================

class SimulatedTrader:
    """
    Simulated trading when Alpaca API keys are not available.
    Tracks paper positions locally.
    """
    
    def __init__(self, starting_capital: float = 100000):
        self.capital = starting_capital
        self.cash = starting_capital
        self.spy_shares = 0
        self.spy_avg_cost = 0
        self.trade_log = []
        self.engine = QuantumPortfolioEngine(SELECTED_SIGNALS)
        
    def update_prices(self):
        """Get current prices from engine."""
        self.engine.update_data()
        self.spy_price = self.engine.data['SPY']['close'].iloc[-1]
        
    def get_portfolio_value(self) -> float:
        """Calculate total portfolio value."""
        return self.cash + (self.spy_shares * self.spy_price)
    
    def check_status(self):
        """Print simulated portfolio status."""
        self.update_prices()
        total = self.get_portfolio_value()
        spy_value = self.spy_shares * self.spy_price
        
        print("\n" + "="*60)
        print("SIMULATED PAPER TRADING STATUS")
        print("="*60)
        
        print(f"\n💰 Portfolio:")
        print(f"   Total Value: ${total:,.2f}")
        print(f"   Cash: ${self.cash:,.2f}")
        print(f"   SPY: {self.spy_shares:.0f} shares @ ${self.spy_price:.2f} = ${spy_value:,.2f}")
        
        allocation = spy_value / total if total > 0 else 0
        print(f"\n📊 Allocation:")
        print(f"   SPY: {allocation:.1%}")
        print(f"   Cash: {1-allocation:.1%}")
        
        # P&L
        pnl = total - self.capital
        pnl_pct = pnl / self.capital * 100
        print(f"\n📈 Performance:")
        print(f"   P&L: ${pnl:,.2f} ({pnl_pct:+.1f}%)")
    
    def execute_signal(self):
        """Execute trade based on current signals."""
        self.update_prices()
        alloc = self.engine.get_portfolio_allocation()
        
        total = self.get_portfolio_value()
        target_allocation = alloc['total_exposure']
        target_value = total * target_allocation
        
        current_value = self.spy_shares * self.spy_price
        trade_value = target_value - current_value
        trade_shares = int(trade_value / self.spy_price)
        
        print("\n" + "="*60)
        print("SIMULATED TRADE EXECUTION")
        print("="*60)
        
        if trade_shares > 0:
            # Buy
            cost = trade_shares * self.spy_price
            if cost <= self.cash:
                self.spy_shares += trade_shares
                self.cash -= cost
                print(f"✅ BOUGHT {trade_shares} SPY @ ${self.spy_price:.2f}")
                print(f"   Cost: ${cost:,.2f}")
            else:
                print(f"❌ Insufficient cash for {trade_shares} shares")
        elif trade_shares < 0:
            # Sell
            shares_to_sell = min(abs(trade_shares), self.spy_shares)
            if shares_to_sell > 0:
                proceeds = shares_to_sell * self.spy_price
                self.spy_shares -= shares_to_sell
                self.cash += proceeds
                print(f"✅ SOLD {shares_to_sell} SPY @ ${self.spy_price:.2f}")
                print(f"   Proceeds: ${proceeds:,.2f}")
            else:
                print(f"❌ No shares to sell")
        else:
            print("📊 No trade needed")
        
        # Log
        self.trade_log.append({
            'date': datetime.now().isoformat(),
            'action': 'buy' if trade_shares > 0 else 'sell' if trade_shares < 0 else 'hold',
            'shares': abs(trade_shares),
            'price': self.spy_price,
            'signals': alloc['signals'],
        })
        
        return alloc


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*70)
    print("QUANTUM PAPER TRADER")
    print("="*70)
    
    if ALPACA_AVAILABLE and os.getenv('ALPACA_API_KEY'):
        print("\n✓ Using Alpaca Paper Trading")
        trader = AlpacaPaperTrader()
    else:
        print("\n⚠️ Alpaca not configured - using simulation mode")
        print("   To use Alpaca, set ALPACA_API_KEY and ALPACA_SECRET_KEY")
        trader = SimulatedTrader(starting_capital=100000)
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == 'status':
            trader.check_status()
        elif command == 'signals':
            if hasattr(trader, 'check_signals'):
                trader.check_signals()
            else:
                trader.engine.print_allocation()
        elif command == 'trade':
            if isinstance(trader, AlpacaPaperTrader):
                trader.execute_trades(confirm=True)
            else:
                trader.execute_signal()
        elif command == 'auto':
            # Auto-trade without confirmation
            if isinstance(trader, AlpacaPaperTrader):
                trader.execute_trades(confirm=False)
            else:
                trader.execute_signal()
        else:
            print(f"Unknown command: {command}")
            print("Usage: python ALPACA_PAPER_TRADER.py [status|signals|trade|auto]")
    else:
        # Default: show signals and recommendation
        if isinstance(trader, AlpacaPaperTrader):
            trader.run_daily_check()
        else:
            trader.engine.print_allocation()
            trader.check_status()


if __name__ == "__main__":
    main()
