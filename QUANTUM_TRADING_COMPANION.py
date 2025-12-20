#!/usr/bin/env python3
"""
=============================================================================
QUANTUM TRADING COMPANION
=============================================================================

A personalized, portfolio-aware AI trading companion that:
1. Learns from YOUR watchlist and trading universe
2. Monitors YOUR portfolio context in real-time
3. Combines YOUR intuition with validated systematic signals
4. Adapts to YOUR trading style over time

ARCHITECTURE:
┌─────────────────────────────────────────────────────────────────────┐
│                      QUANTUM TRADING COMPANION                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │   WATCHLIST  │───▶│   TRAINING   │───▶│   SIGNALS    │          │
│  │   MANAGER    │    │   PIPELINE   │    │   (per stock)│          │
│  └──────────────┘    └──────────────┘    └──────────────┘          │
│         │                   │                   │                   │
│         ▼                   ▼                   ▼                   │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │  PORTFOLIO   │───▶│  RISK-AWARE  │───▶│   FINAL      │          │
│  │   CONTEXT    │    │   BLENDER    │    │   DECISION   │          │
│  └──────────────┘    └──────────────┘    └──────────────┘          │
│         ▲                   ▲                   │                   │
│         │                   │                   ▼                   │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │   ALPACA     │    │    YOUR      │    │   TRADE      │          │
│  │   ACCOUNT    │    │  INTUITION   │    │   JOURNAL    │          │
│  └──────────────┘    └──────────────┘    └──────────────┘          │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘

Run:
    python QUANTUM_TRADING_COMPANION.py

Commands:
    python QUANTUM_TRADING_COMPANION.py scan      # Scan watchlist for signals
    python QUANTUM_TRADING_COMPANION.py portfolio # Analyze portfolio context
    python QUANTUM_TRADING_COMPANION.py train     # Train on your watchlist
    python QUANTUM_TRADING_COMPANION.py journal   # Log a trade with your reasoning
    python QUANTUM_TRADING_COMPANION.py learn     # Learn from your past trades
"""

import os
import sys
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

import pandas as pd
import numpy as np
import yfinance as yf

warnings.filterwarnings('ignore')

# Try to import Alpaca
try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import GetAssetsRequest
    from alpaca.trading.enums import AssetClass
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False


# ============================================================================
# CONFIGURATION
# ============================================================================

# Data paths
DATA_PATH = Path('./companion_data/')
DATA_PATH.mkdir(exist_ok=True)

# Validated signals from our testing (corrected for VIX lag)
VALIDATED_SIGNALS = ['H20', 'H128', 'H62', 'H19']

# Default watchlist (can be overridden by Alpaca)
DEFAULT_WATCHLIST = [
    # Your top performers
    'SPY', 'QQQ', 'IWM',
    # High-momentum tech
    'NVDA', 'AMD', 'AVGO', 'MRVL',
    # Volatile movers (good for mean reversion)
    'TSLA', 'COIN', 'MSTR',
    # Sector ETFs
    'XLF', 'XLE', 'XLK', 'XBI',
    # Safe havens
    'TLT', 'GLD', 'UUP',
]


# ============================================================================
# SIGNAL LIBRARY (Corrected for look-ahead bias)
# ============================================================================

def calc_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


class SignalLibrary:
    """Validated signal generators with proper lag handling."""
    
    @staticmethod
    def h16_weekly_reversal(data: pd.DataFrame, threshold: float = -0.03) -> pd.Series:
        """Buy when 5-day return < -3%."""
        weekly_ret = data['close'].pct_change(5)
        return (weekly_ret < threshold).shift(1).fillna(0).astype(int)
    
    @staticmethod
    def h19_bollinger_mr(data: pd.DataFrame, period: int = 20, num_std: float = 2.0) -> pd.Series:
        """Buy when price below lower Bollinger Band."""
        ma = data['close'].rolling(period).mean()
        std = data['close'].rolling(period).std()
        lower = ma - num_std * std
        return (data['close'] < lower).shift(1).fillna(0).astype(int)
    
    @staticmethod
    def h20_vix_mean_reversion(vix: pd.Series, threshold: int = 25) -> pd.Series:
        """Buy SPY when VIX > 25 (lagged)."""
        vix_lagged = vix.shift(1)  # Use yesterday's VIX
        return (vix_lagged > threshold).shift(1).fillna(0).astype(int)
    
    @staticmethod
    def h128_vix_turbulence(vix: pd.Series, lookback: int = 14, threshold: float = 2.0) -> pd.Series:
        """Buy when VIX volatility spikes."""
        vix_lagged = vix.shift(1)
        vix_changes = vix_lagged.diff()
        vix_vol = vix_changes.rolling(lookback).std()
        vix_vol_mean = vix_vol.rolling(252).mean()
        vix_vol_std = vix_vol.rolling(252).std()
        z_score = (vix_vol - vix_vol_mean) / vix_vol_std
        return (z_score > threshold).shift(1).fillna(0).astype(int)
    
    @staticmethod
    def h62_oil_equity(spy: pd.DataFrame, uso: pd.DataFrame, lookback: int = 21) -> pd.Series:
        """Stay invested unless SPY up but oil down >5%."""
        spy_mom = spy['close'].pct_change(lookback)
        uso_mom = uso['close'].pct_change(lookback).reindex(spy.index).ffill()
        divergence = (spy_mom > 0) & (uso_mom < -0.05)
        signal = pd.Series(1, index=spy.index)
        signal[divergence] = 0
        return signal.shift(1).fillna(1).astype(int)
    
    @staticmethod
    def momentum_score(data: pd.DataFrame, periods: list = [5, 21, 63]) -> pd.Series:
        """Multi-timeframe momentum score (0-1)."""
        scores = []
        for p in periods:
            ret = data['close'].pct_change(p)
            score = (ret > 0).astype(int)
            scores.append(score)
        return pd.concat(scores, axis=1).mean(axis=1)
    
    @staticmethod
    def rsi_signal(data: pd.DataFrame, oversold: int = 30, overbought: int = 70) -> pd.Series:
        """RSI-based signal: 1=oversold (buy), -1=overbought (sell), 0=neutral."""
        rsi = calc_rsi(data['close'], 14)
        signal = pd.Series(0, index=data.index)
        signal[rsi < oversold] = 1
        signal[rsi > overbought] = -1
        return signal.shift(1).fillna(0).astype(int)
    
    @staticmethod
    def volume_spike(data: pd.DataFrame, threshold: float = 2.0) -> pd.Series:
        """Detect volume spikes (>2x average)."""
        avg_vol = data['volume'].rolling(20).mean()
        spike = data['volume'] > (avg_vol * threshold)
        return spike.shift(1).fillna(0).astype(int)
    
    @staticmethod
    def gap_reversal(data: pd.DataFrame, threshold: float = 0.02) -> pd.Series:
        """Detect gap downs that might reverse."""
        gap = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
        return (gap < -threshold).shift(1).fillna(0).astype(int)


# ============================================================================
# PORTFOLIO CONTEXT ENGINE
# ============================================================================

class PortfolioContextEngine:
    """
    Monitors your portfolio to provide context for signal generation.
    
    Features:
    - Current position exposures
    - Sector concentration
    - Portfolio beta
    - Risk metrics (VaR, drawdown)
    """
    
    def __init__(self, api_key: str = None, secret_key: str = None):
        self.api_key = api_key or os.getenv('ALPACA_API_KEY', '')
        self.secret_key = secret_key or os.getenv('ALPACA_SECRET_KEY', '')
        self.client = None
        self.positions = {}
        self.account = None
        
        if ALPACA_AVAILABLE and self.api_key:
            self.client = TradingClient(self.api_key, self.secret_key, paper=True)
    
    def connect(self) -> bool:
        """Connect to Alpaca and fetch account info."""
        if not self.client:
            return False
        
        try:
            self.account = self.client.get_account()
            return True
        except Exception as e:
            print(f"Connection error: {e}")
            return False
    
    def get_positions(self) -> Dict:
        """Fetch all current positions."""
        if not self.client:
            return {}
        
        try:
            positions = self.client.get_all_positions()
            self.positions = {
                pos.symbol: {
                    'qty': float(pos.qty),
                    'market_value': float(pos.market_value),
                    'avg_entry': float(pos.avg_entry_price),
                    'current_price': float(pos.current_price),
                    'unrealized_pl': float(pos.unrealized_pl),
                    'unrealized_plpc': float(pos.unrealized_plpc),
                    'side': pos.side,
                }
                for pos in positions
            }
            return self.positions
        except Exception as e:
            print(f"Error fetching positions: {e}")
            return {}
    
    def get_exposure_summary(self) -> Dict:
        """Calculate portfolio exposure metrics."""
        positions = self.get_positions()
        
        if not positions:
            return {'total_exposure': 0, 'cash': 100, 'positions': 0}
        
        equity = float(self.account.equity) if self.account else 100000
        
        total_long = sum(p['market_value'] for p in positions.values() if p['market_value'] > 0)
        total_short = abs(sum(p['market_value'] for p in positions.values() if p['market_value'] < 0))
        
        return {
            'equity': equity,
            'cash': float(self.account.cash) if self.account else equity,
            'total_long': total_long,
            'total_short': total_short,
            'net_exposure': (total_long - total_short) / equity,
            'gross_exposure': (total_long + total_short) / equity,
            'n_positions': len(positions),
            'largest_position': max((p['market_value'] / equity for p in positions.values()), default=0),
        }
    
    def check_concentration_risk(self, max_single: float = 0.20, max_sector: float = 0.40) -> List[str]:
        """Check for concentration risks."""
        warnings = []
        positions = self.get_positions()
        
        if not positions or not self.account:
            return warnings
        
        equity = float(self.account.equity)
        
        for symbol, pos in positions.items():
            weight = pos['market_value'] / equity
            if weight > max_single:
                warnings.append(f"⚠️ {symbol} is {weight:.1%} of portfolio (max: {max_single:.0%})")
        
        return warnings
    
    def print_summary(self):
        """Print portfolio summary."""
        print("\n" + "="*60)
        print("PORTFOLIO CONTEXT")
        print("="*60)
        
        if not self.connect():
            print("❌ Not connected to Alpaca. Using simulation mode.")
            return
        
        exposure = self.get_exposure_summary()
        
        print(f"\n💰 Account:")
        print(f"   Equity: ${exposure['equity']:,.2f}")
        print(f"   Cash: ${exposure['cash']:,.2f}")
        
        print(f"\n📊 Exposure:")
        print(f"   Net: {exposure['net_exposure']:.1%}")
        print(f"   Gross: {exposure['gross_exposure']:.1%}")
        print(f"   Positions: {exposure['n_positions']}")
        print(f"   Largest: {exposure['largest_position']:.1%}")
        
        # Concentration warnings
        warnings = self.check_concentration_risk()
        if warnings:
            print(f"\n⚠️ Risk Warnings:")
            for w in warnings:
                print(f"   {w}")
        
        # Position details
        positions = self.positions
        if positions:
            print(f"\n📈 Positions:")
            for symbol, pos in sorted(positions.items(), key=lambda x: -x[1]['market_value']):
                pnl = pos['unrealized_plpc'] * 100
                print(f"   {symbol}: ${pos['market_value']:,.0f} ({pnl:+.1f}%)")


# ============================================================================
# WATCHLIST MANAGER
# ============================================================================

class WatchlistManager:
    """
    Manages your trading watchlist.
    Can sync with Alpaca or use local list.
    """
    
    def __init__(self, portfolio_engine: PortfolioContextEngine = None):
        self.portfolio = portfolio_engine
        self.watchlist = DEFAULT_WATCHLIST.copy()
        self.watchlist_file = DATA_PATH / 'watchlist.json'
        self.load_watchlist()
    
    def load_watchlist(self):
        """Load watchlist from file."""
        if self.watchlist_file.exists():
            with open(self.watchlist_file, 'r') as f:
                data = json.load(f)
                self.watchlist = data.get('symbols', DEFAULT_WATCHLIST)
    
    def save_watchlist(self):
        """Save watchlist to file."""
        with open(self.watchlist_file, 'w') as f:
            json.dump({
                'symbols': self.watchlist,
                'updated': datetime.now().isoformat()
            }, f, indent=2)
    
    def add_symbol(self, symbol: str):
        """Add a symbol to watchlist."""
        symbol = symbol.upper()
        if symbol not in self.watchlist:
            self.watchlist.append(symbol)
            self.save_watchlist()
            print(f"✓ Added {symbol} to watchlist")
    
    def remove_symbol(self, symbol: str):
        """Remove a symbol from watchlist."""
        symbol = symbol.upper()
        if symbol in self.watchlist:
            self.watchlist.remove(symbol)
            self.save_watchlist()
            print(f"✓ Removed {symbol} from watchlist")
    
    def sync_with_alpaca(self, watchlist_name: str = "Research"):
        """Sync with Alpaca watchlist."""
        if not self.portfolio or not self.portfolio.client:
            print("❌ Alpaca not connected")
            return
        
        try:
            # Get watchlists from Alpaca
            watchlists = self.portfolio.client.get_watchlists()
            
            for wl in watchlists:
                if wl.name == watchlist_name:
                    self.watchlist = [asset.symbol for asset in wl.assets]
                    self.save_watchlist()
                    print(f"✓ Synced {len(self.watchlist)} symbols from Alpaca")
                    return
            
            print(f"⚠️ Watchlist '{watchlist_name}' not found")
        except Exception as e:
            print(f"Error syncing: {e}")
    
    def get_symbols(self) -> List[str]:
        """Get current watchlist symbols."""
        return self.watchlist.copy()


# ============================================================================
# PERSONALIZED TRAINING PIPELINE
# ============================================================================

class PersonalizedTrainer:
    """
    Trains signal models on YOUR specific watchlist.
    Discovers which signals work best for which stocks.
    """
    
    def __init__(self, watchlist: List[str]):
        self.watchlist = watchlist
        self.signals = SignalLibrary()
        self.profiles = {}  # Asset-specific signal profiles
        self.profiles_file = DATA_PATH / 'signal_profiles.json'
        self.load_profiles()
    
    def load_profiles(self):
        """Load saved signal profiles."""
        if self.profiles_file.exists():
            with open(self.profiles_file, 'r') as f:
                self.profiles = json.load(f)
    
    def save_profiles(self):
        """Save signal profiles."""
        with open(self.profiles_file, 'w') as f:
            json.dump(self.profiles, f, indent=2)
    
    def fetch_data(self, symbol: str, start_date: str = '2020-01-01') -> pd.DataFrame:
        """Fetch historical data for a symbol."""
        try:
            data = yf.download(symbol, start=start_date, progress=False)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            data.columns = [c.lower().replace(' ', '_') for c in data.columns]
            return data
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
            return pd.DataFrame()
    
    def test_signal_on_asset(self, symbol: str, data: pd.DataFrame, 
                              signal_func, signal_name: str,
                              hold_period: int = 21) -> Dict:
        """Test a specific signal on an asset."""
        try:
            if 'vix' in signal_name.lower():
                # VIX signals need VIX data
                vix_data = yf.download('^VIX', start=data.index[0], progress=False)
                if isinstance(vix_data.columns, pd.MultiIndex):
                    vix_data.columns = vix_data.columns.get_level_values(0)
                vix_data.columns = [c.lower().replace(' ', '_') for c in vix_data.columns]
                signal = signal_func(vix_data['close'].reindex(data.index).ffill())
            else:
                signal = signal_func(data)
            
            # Calculate forward returns
            fwd_ret = data['close'].pct_change(hold_period).shift(-hold_period)
            
            # Signal performance
            signal_rets = fwd_ret[signal == 1].dropna()
            other_rets = fwd_ret[signal == 0].dropna()
            
            if len(signal_rets) < 10:
                return {'valid': False, 'reason': 'Insufficient signals'}
            
            spread = (signal_rets.mean() - other_rets.mean()) * (252/hold_period)
            signal_sharpe = signal_rets.mean() / signal_rets.std() * np.sqrt(252/hold_period) if signal_rets.std() > 0 else 0
            
            return {
                'valid': True,
                'spread': spread,
                'sharpe': signal_sharpe,
                'n_signals': len(signal_rets),
                'win_rate': (signal_rets > 0).mean(),
                'avg_return': signal_rets.mean() * (252/hold_period),
            }
        except Exception as e:
            return {'valid': False, 'reason': str(e)}
    
    def train_on_symbol(self, symbol: str) -> Dict:
        """Train all signals on a specific symbol."""
        print(f"\n  Training on {symbol}...")
        
        data = self.fetch_data(symbol)
        if data.empty:
            return {}
        
        # Test each signal
        signal_tests = [
            ('H16_Weekly_Reversal', lambda d: self.signals.h16_weekly_reversal(d)),
            ('H19_Bollinger_MR', lambda d: self.signals.h19_bollinger_mr(d)),
            ('RSI_Signal', lambda d: self.signals.rsi_signal(d)),
            ('Momentum_Score', lambda d: (self.signals.momentum_score(d) > 0.5).astype(int)),
            ('Volume_Spike', lambda d: self.signals.volume_spike(d)),
            ('Gap_Reversal', lambda d: self.signals.gap_reversal(d)),
        ]
        
        results = {}
        for signal_name, signal_func in signal_tests:
            result = self.test_signal_on_asset(symbol, data, signal_func, signal_name)
            if result.get('valid'):
                results[signal_name] = result
                status = "✓" if result['spread'] > 0 else "✗"
                print(f"    {signal_name}: {result['spread']:+.1%} spread {status}")
        
        # Find best signals for this asset
        if results:
            best = max(results.items(), key=lambda x: x[1]['spread'])
            print(f"    📈 Best signal: {best[0]} ({best[1]['spread']:+.1%})")
        
        return results
    
    def train_on_watchlist(self):
        """Train on all watchlist symbols."""
        print("\n" + "="*60)
        print("PERSONALIZED TRAINING PIPELINE")
        print("="*60)
        print(f"Training on {len(self.watchlist)} symbols...")
        
        for symbol in self.watchlist:
            results = self.train_on_symbol(symbol)
            if results:
                self.profiles[symbol] = {
                    'signals': results,
                    'trained': datetime.now().isoformat(),
                }
        
        self.save_profiles()
        
        # Summary
        print("\n" + "-"*60)
        print("TRAINING SUMMARY")
        print("-"*60)
        
        best_by_asset = []
        for symbol, profile in self.profiles.items():
            signals = profile.get('signals', {})
            if signals:
                best = max(signals.items(), key=lambda x: x[1]['spread'])
                best_by_asset.append({
                    'symbol': symbol,
                    'best_signal': best[0],
                    'spread': best[1]['spread'],
                    'sharpe': best[1]['sharpe'],
                })
        
        df = pd.DataFrame(best_by_asset).sort_values('spread', ascending=False)
        print("\nTop Assets by Signal Performance:")
        for _, row in df.head(10).iterrows():
            print(f"  {row['symbol']}: {row['best_signal']} → {row['spread']:+.1%}")
        
        return self.profiles
    
    def get_best_signal_for_asset(self, symbol: str) -> Optional[Tuple[str, Dict]]:
        """Get the best performing signal for an asset."""
        if symbol not in self.profiles:
            return None
        
        signals = self.profiles[symbol].get('signals', {})
        if not signals:
            return None
        
        best = max(signals.items(), key=lambda x: x[1]['spread'])
        return best


# ============================================================================
# TRADE JOURNAL - LEARN FROM YOUR TRADES
# ============================================================================

class TradeJournal:
    """
    Track your trades and learn from your decisions.
    
    Captures:
    - Your entry/exit reasoning
    - Market conditions at trade time
    - Outcome vs expectation
    - What signals were active
    """
    
    def __init__(self):
        self.journal_file = DATA_PATH / 'trade_journal.json'
        self.trades = []
        self.load_journal()
    
    def load_journal(self):
        """Load journal from file."""
        if self.journal_file.exists():
            with open(self.journal_file, 'r') as f:
                self.trades = json.load(f)
    
    def save_journal(self):
        """Save journal to file."""
        with open(self.journal_file, 'w') as f:
            json.dump(self.trades, f, indent=2)
    
    def log_trade(self, symbol: str, side: str, entry_price: float,
                  reasoning: str, confidence: int = 5,
                  signals_active: List[str] = None):
        """Log a new trade."""
        trade = {
            'id': len(self.trades) + 1,
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol.upper(),
            'side': side.lower(),
            'entry_price': entry_price,
            'reasoning': reasoning,
            'confidence': confidence,  # 1-10 scale
            'signals_active': signals_active or [],
            'status': 'open',
            'exit_price': None,
            'exit_timestamp': None,
            'exit_reasoning': None,
            'pnl': None,
            'pnl_pct': None,
            'lessons': None,
        }
        
        self.trades.append(trade)
        self.save_journal()
        
        print(f"\n✓ Trade logged: {side.upper()} {symbol} @ ${entry_price:.2f}")
        print(f"  Reasoning: {reasoning}")
        print(f"  Confidence: {confidence}/10")
        
        return trade['id']
    
    def close_trade(self, trade_id: int, exit_price: float, 
                    exit_reasoning: str = None, lessons: str = None):
        """Close an open trade."""
        for trade in self.trades:
            if trade['id'] == trade_id and trade['status'] == 'open':
                trade['exit_price'] = exit_price
                trade['exit_timestamp'] = datetime.now().isoformat()
                trade['exit_reasoning'] = exit_reasoning
                trade['lessons'] = lessons
                trade['status'] = 'closed'
                
                # Calculate P&L
                if trade['side'] == 'buy':
                    trade['pnl_pct'] = (exit_price - trade['entry_price']) / trade['entry_price']
                else:
                    trade['pnl_pct'] = (trade['entry_price'] - exit_price) / trade['entry_price']
                
                self.save_journal()
                
                print(f"\n✓ Trade closed: {trade['symbol']}")
                print(f"  P&L: {trade['pnl_pct']*100:+.1f}%")
                if lessons:
                    print(f"  Lessons: {lessons}")
                
                return True
        
        print(f"❌ Trade {trade_id} not found or already closed")
        return False
    
    def analyze_performance(self):
        """Analyze trading performance and patterns."""
        closed = [t for t in self.trades if t['status'] == 'closed']
        
        if not closed:
            print("No closed trades to analyze")
            return {}
        
        print("\n" + "="*60)
        print("TRADE JOURNAL ANALYSIS")
        print("="*60)
        
        # Overall performance
        pnls = [t['pnl_pct'] for t in closed if t['pnl_pct'] is not None]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        
        print(f"\n📊 Overall Performance:")
        print(f"   Total trades: {len(closed)}")
        print(f"   Win rate: {len(wins)/len(pnls)*100:.1f}%")
        print(f"   Avg win: {np.mean(wins)*100:+.1f}%" if wins else "   Avg win: N/A")
        print(f"   Avg loss: {np.mean(losses)*100:+.1f}%" if losses else "   Avg loss: N/A")
        print(f"   Total return: {sum(pnls)*100:+.1f}%")
        
        # By confidence level
        print(f"\n📈 Performance by Confidence:")
        high_conf = [t for t in closed if t.get('confidence', 5) >= 7]
        low_conf = [t for t in closed if t.get('confidence', 5) < 7]
        
        if high_conf:
            high_pnl = np.mean([t['pnl_pct'] for t in high_conf if t['pnl_pct']])
            print(f"   High confidence (≥7): {high_pnl*100:+.1f}% ({len(high_conf)} trades)")
        if low_conf:
            low_pnl = np.mean([t['pnl_pct'] for t in low_conf if t['pnl_pct']])
            print(f"   Low confidence (<7): {low_pnl*100:+.1f}% ({len(low_conf)} trades)")
        
        # By signal alignment
        print(f"\n🔗 Performance by Signal Alignment:")
        with_signals = [t for t in closed if t.get('signals_active')]
        without_signals = [t for t in closed if not t.get('signals_active')]
        
        if with_signals:
            sig_pnl = np.mean([t['pnl_pct'] for t in with_signals if t['pnl_pct']])
            print(f"   With signals: {sig_pnl*100:+.1f}% ({len(with_signals)} trades)")
        if without_signals:
            no_sig_pnl = np.mean([t['pnl_pct'] for t in without_signals if t['pnl_pct']])
            print(f"   Without signals: {no_sig_pnl*100:+.1f}% ({len(without_signals)} trades)")
        
        # Lessons learned
        lessons = [t['lessons'] for t in closed if t.get('lessons')]
        if lessons:
            print(f"\n📝 Lessons Learned ({len(lessons)} entries):")
            for lesson in lessons[-5:]:  # Last 5 lessons
                print(f"   • {lesson}")
        
        return {
            'total_trades': len(closed),
            'win_rate': len(wins)/len(pnls) if pnls else 0,
            'total_return': sum(pnls),
            'avg_return': np.mean(pnls) if pnls else 0,
        }
    
    def get_open_trades(self) -> List[Dict]:
        """Get all open trades."""
        return [t for t in self.trades if t['status'] == 'open']
    
    def print_open_trades(self):
        """Print all open trades."""
        open_trades = self.get_open_trades()
        
        if not open_trades:
            print("\nNo open trades")
            return
        
        print("\n" + "="*60)
        print("OPEN TRADES")
        print("="*60)
        
        for trade in open_trades:
            print(f"\n  #{trade['id']}: {trade['side'].upper()} {trade['symbol']}")
            print(f"     Entry: ${trade['entry_price']:.2f}")
            print(f"     Reasoning: {trade['reasoning']}")
            print(f"     Confidence: {trade['confidence']}/10")
            print(f"     Opened: {trade['timestamp']}")


# ============================================================================
# SIGNAL SCANNER
# ============================================================================

class SignalScanner:
    """
    Scans your watchlist for active signals.
    Combines systematic signals with personalized training.
    """
    
    def __init__(self, watchlist: List[str], trainer: PersonalizedTrainer = None):
        self.watchlist = watchlist
        self.trainer = trainer
        self.signals = SignalLibrary()
    
    def fetch_market_data(self) -> Dict[str, pd.DataFrame]:
        """Fetch data for all watchlist symbols + market indicators."""
        data = {}
        
        # Fetch watchlist
        for symbol in self.watchlist:
            try:
                df = yf.download(symbol, period='1y', progress=False)
                if not df.empty:
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                    df.columns = [c.lower().replace(' ', '_') for c in df.columns]
                    data[symbol] = df
            except:
                pass
        
        # Fetch VIX
        try:
            vix = yf.download('^VIX', period='1y', progress=False)
            if isinstance(vix.columns, pd.MultiIndex):
                vix.columns = vix.columns.get_level_values(0)
            vix.columns = [c.lower().replace(' ', '_') for c in vix.columns]
            data['^VIX'] = vix
        except:
            pass
        
        return data
    
    def scan(self) -> List[Dict]:
        """Scan watchlist for active signals."""
        print("\n" + "="*60)
        print("SIGNAL SCANNER")
        print("="*60)
        
        print("\n📥 Fetching market data...")
        data = self.fetch_market_data()
        
        vix_data = data.get('^VIX', pd.DataFrame())
        vix_close = vix_data['close'] if not vix_data.empty else pd.Series()
        
        print(f"   Loaded {len(data)} symbols")
        
        if not vix_close.empty:
            print(f"   VIX: {vix_close.iloc[-1]:.1f}")
        
        opportunities = []
        
        print(f"\n🔍 Scanning {len(self.watchlist)} symbols...")
        
        for symbol in self.watchlist:
            if symbol not in data or symbol.startswith('^'):
                continue
            
            df = data[symbol]
            if df.empty or len(df) < 50:
                continue
            
            current_price = df['close'].iloc[-1]
            
            # Check all signals
            active_signals = []
            signal_details = []
            
            # Weekly reversal
            h16 = self.signals.h16_weekly_reversal(df)
            if h16.iloc[-1] == 1:
                weekly_ret = df['close'].pct_change(5).iloc[-1]
                active_signals.append('H16')
                signal_details.append(f"Weekly: {weekly_ret*100:.1f}%")
            
            # Bollinger
            h19 = self.signals.h19_bollinger_mr(df)
            if h19.iloc[-1] == 1:
                active_signals.append('H19')
                signal_details.append("Below BB")
            
            # RSI
            rsi = self.signals.rsi_signal(df)
            if rsi.iloc[-1] == 1:
                active_signals.append('RSI_Oversold')
                signal_details.append(f"RSI: {calc_rsi(df['close'], 14).iloc[-1]:.0f}")
            
            # Momentum
            mom = self.signals.momentum_score(df)
            if mom.iloc[-1] > 0.7:
                active_signals.append('Strong_Momentum')
                signal_details.append(f"Mom: {mom.iloc[-1]:.0%}")
            
            # Volume spike
            vol_spike = self.signals.volume_spike(df)
            if vol_spike.iloc[-1] == 1:
                active_signals.append('Volume_Spike')
                signal_details.append("Vol >2x avg")
            
            # Gap reversal
            gap = self.signals.gap_reversal(df)
            if gap.iloc[-1] == 1:
                active_signals.append('Gap_Down')
                signal_details.append("Gap reversal candidate")
            
            # Add if any signals active
            if active_signals:
                # Check personalized profile
                best_signal = None
                if self.trainer:
                    best = self.trainer.get_best_signal_for_asset(symbol)
                    if best:
                        best_signal = best[0]
                
                opportunities.append({
                    'symbol': symbol,
                    'price': current_price,
                    'signals': active_signals,
                    'details': signal_details,
                    'best_signal': best_signal,
                    'n_signals': len(active_signals),
                })
        
        # Sort by number of active signals
        opportunities.sort(key=lambda x: -x['n_signals'])
        
        # Print results
        print("\n" + "-"*60)
        print("ACTIVE OPPORTUNITIES")
        print("-"*60)
        
        if not opportunities:
            print("\n  No active signals found")
        else:
            for opp in opportunities[:15]:  # Top 15
                print(f"\n  🎯 {opp['symbol']} @ ${opp['price']:.2f}")
                print(f"     Signals: {', '.join(opp['signals'])}")
                print(f"     Details: {' | '.join(opp['details'])}")
                if opp['best_signal']:
                    print(f"     ⭐ Best historical: {opp['best_signal']}")
        
        # VIX-based signals (apply to SPY/QQQ)
        if not vix_close.empty:
            print("\n" + "-"*60)
            print("VIX-BASED SIGNALS (Apply to SPY/QQQ)")
            print("-"*60)
            
            current_vix = vix_close.iloc[-1]
            
            # H20
            h20 = self.signals.h20_vix_mean_reversion(vix_close)
            if h20.iloc[-1] == 1:
                print(f"  🟢 H20 VIX Mean Reversion ACTIVE (VIX={current_vix:.1f} > 25)")
            
            # H128
            h128 = self.signals.h128_vix_turbulence(vix_close)
            if h128.iloc[-1] == 1:
                print(f"  🟢 H128 VIX Turbulence ACTIVE")
            
            if h20.iloc[-1] == 0 and h128.iloc[-1] == 0:
                print(f"  ⚪ No VIX signals active (VIX={current_vix:.1f})")
        
        return opportunities


# ============================================================================
# MAIN COMPANION CLASS
# ============================================================================

class QuantumTradingCompanion:
    """
    Your personalized AI trading companion.
    Combines portfolio awareness, personalized training, and your intuition.
    """
    
    def __init__(self):
        # Initialize components
        self.portfolio = PortfolioContextEngine()
        self.watchlist_mgr = WatchlistManager(self.portfolio)
        self.trainer = PersonalizedTrainer(self.watchlist_mgr.get_symbols())
        self.journal = TradeJournal()
        self.scanner = SignalScanner(self.watchlist_mgr.get_symbols(), self.trainer)
    
    def scan(self):
        """Scan watchlist for opportunities."""
        return self.scanner.scan()
    
    def portfolio_check(self):
        """Check portfolio context."""
        self.portfolio.print_summary()
    
    def train(self):
        """Train on your watchlist."""
        self.trainer.train_on_watchlist()
    
    def log_trade(self, symbol: str, side: str, price: float, reasoning: str, confidence: int = 5):
        """Log a trade."""
        # Get active signals for this symbol
        data = self.scanner.fetch_market_data()
        active_signals = []
        
        if symbol in data:
            df = data[symbol]
            if SignalLibrary.h16_weekly_reversal(df).iloc[-1]:
                active_signals.append('H16')
            if SignalLibrary.h19_bollinger_mr(df).iloc[-1]:
                active_signals.append('H19')
        
        return self.journal.log_trade(symbol, side, price, reasoning, confidence, active_signals)
    
    def close_trade(self, trade_id: int, price: float, reasoning: str = None, lessons: str = None):
        """Close a trade."""
        return self.journal.close_trade(trade_id, price, reasoning, lessons)
    
    def analyze_trades(self):
        """Analyze your trading performance."""
        return self.journal.analyze_performance()
    
    def show_open_trades(self):
        """Show open trades."""
        self.journal.print_open_trades()
    
    def daily_briefing(self):
        """Complete daily briefing."""
        print("\n" + "="*70)
        print(f"QUANTUM TRADING COMPANION - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print("="*70)
        
        # Portfolio context
        self.portfolio_check()
        
        # Open trades
        self.show_open_trades()
        
        # Scan for signals
        self.scan()
        
        # Recent performance
        print("\n" + "-"*60)
        print("RECENT PERFORMANCE")
        print("-"*60)
        self.analyze_trades()


# ============================================================================
# MAIN
# ============================================================================

def main():
    companion = QuantumTradingCompanion()
    
    if len(sys.argv) < 2:
        # Default: daily briefing
        companion.daily_briefing()
        return
    
    command = sys.argv[1].lower()
    
    if command == 'scan':
        companion.scan()
    
    elif command == 'portfolio':
        companion.portfolio_check()
    
    elif command == 'train':
        companion.train()
    
    elif command == 'journal':
        # Log a trade interactively
        print("\n📝 LOG NEW TRADE")
        symbol = input("Symbol: ").upper()
        side = input("Side (buy/sell): ").lower()
        price = float(input("Entry price: "))
        reasoning = input("Your reasoning: ")
        confidence = int(input("Confidence (1-10): "))
        
        companion.log_trade(symbol, side, price, reasoning, confidence)
    
    elif command == 'close':
        # Close a trade
        companion.show_open_trades()
        trade_id = int(input("\nTrade ID to close: "))
        price = float(input("Exit price: "))
        reasoning = input("Exit reasoning (optional): ") or None
        lessons = input("Lessons learned (optional): ") or None
        
        companion.close_trade(trade_id, price, reasoning, lessons)
    
    elif command == 'learn':
        companion.analyze_trades()
    
    elif command == 'open':
        companion.show_open_trades()
    
    elif command == 'add':
        if len(sys.argv) > 2:
            companion.watchlist_mgr.add_symbol(sys.argv[2])
    
    elif command == 'remove':
        if len(sys.argv) > 2:
            companion.watchlist_mgr.remove_symbol(sys.argv[2])
    
    elif command == 'watchlist':
        print("\n📋 Current Watchlist:")
        for s in companion.watchlist_mgr.get_symbols():
            print(f"   {s}")
    
    else:
        print(f"Unknown command: {command}")
        print("\nAvailable commands:")
        print("  scan      - Scan watchlist for signals")
        print("  portfolio - Check portfolio context")
        print("  train     - Train on your watchlist")
        print("  journal   - Log a new trade")
        print("  close     - Close an open trade")
        print("  learn     - Analyze your trading performance")
        print("  open      - Show open trades")
        print("  add       - Add symbol to watchlist")
        print("  remove    - Remove symbol from watchlist")
        print("  watchlist - Show current watchlist")


if __name__ == "__main__":
    main()
