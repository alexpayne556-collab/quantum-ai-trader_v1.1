#!/usr/bin/env python3
"""
🔬 COMBINATION DISCOVERY ENGINE
================================
Find THE winning pattern combinations through exhaustive search

This runs all possible combinations of:
- Technical indicators
- Timeframes
- Entry/exit conditions
- Position sizing rules

And saves every winning pattern found.
"""

import os
import json
import itertools
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    print("pip install yfinance")


# =============================================================================
# PATTERN DEFINITIONS
# =============================================================================

@dataclass
class WinningPattern:
    """A discovered winning pattern"""
    name: str
    description: str
    entry_conditions: Dict
    exit_conditions: Dict
    backtest_results: Dict
    win_rate: float
    avg_return: float
    max_drawdown: float
    sharpe_ratio: float
    total_trades: int
    discovered_date: str
    tickers_tested: List[str]
    confidence_score: float


# =============================================================================
# INDICATOR COMBINATIONS TO TEST
# =============================================================================

# RSI Combinations
RSI_PERIODS = [7, 9, 14, 21]
RSI_OVERSOLD = [20, 25, 30, 35]
RSI_OVERBOUGHT = [65, 70, 75, 80]

# Moving Average Combinations
MA_FAST = [5, 8, 10, 13, 21]
MA_SLOW = [21, 34, 50, 100, 200]
MA_TYPES = ['sma', 'ema']

# MACD Combinations
MACD_FAST = [8, 12]
MACD_SLOW = [21, 26]
MACD_SIGNAL = [5, 9]

# ATR for Stops
ATR_PERIODS = [7, 14, 21]
ATR_MULTIPLIERS = [1.0, 1.5, 2.0, 2.5, 3.0]

# Bollinger Bands
BB_PERIODS = [10, 20, 50]
BB_STD = [1.5, 2.0, 2.5, 3.0]

# Volume Conditions
VOLUME_THRESHOLDS = [1.5, 2.0, 2.5, 3.0]

# Holding Periods
HOLDING_PERIODS = [1, 2, 3, 5, 7, 10, 14, 21]

# Profit Targets
PROFIT_TARGETS = [0.02, 0.03, 0.05, 0.08, 0.10, 0.15]

# Stop Losses
STOP_LOSSES = [-0.02, -0.03, -0.05, -0.08, -0.10]

# Drawdown Conditions (for dip buying)
DRAWDOWN_THRESHOLDS = [-0.05, -0.08, -0.10, -0.15, -0.20]


# =============================================================================
# STRATEGY TEMPLATES
# =============================================================================

STRATEGY_TEMPLATES = {
    'RSI_OVERSOLD_BOUNCE': {
        'description': 'Buy when RSI oversold, sell when recovered',
        'entry': ['rsi_oversold'],
        'exit': ['rsi_recovered', 'profit_target', 'stop_loss'],
    },
    
    'MA_CROSSOVER': {
        'description': 'Buy on MA crossover, sell on reverse crossover',
        'entry': ['ma_cross_up'],
        'exit': ['ma_cross_down', 'stop_loss'],
    },
    
    'MACD_MOMENTUM': {
        'description': 'Buy on MACD signal cross, sell on reverse',
        'entry': ['macd_cross_up', 'macd_positive'],
        'exit': ['macd_cross_down', 'profit_target'],
    },
    
    'DIP_BUY': {
        'description': 'Buy on significant dip in uptrend',
        'entry': ['price_drawdown', 'rsi_oversold', 'above_long_ma'],
        'exit': ['profit_target', 'stop_loss', 'time_stop'],
    },
    
    'BREAKOUT': {
        'description': 'Buy on volume breakout above resistance',
        'entry': ['above_resistance', 'volume_surge', 'rsi_mid'],
        'exit': ['profit_target', 'trailing_stop'],
    },
    
    'MEAN_REVERSION': {
        'description': 'Buy at lower BB, sell at upper BB',
        'entry': ['below_lower_bb', 'rsi_oversold'],
        'exit': ['above_upper_bb', 'profit_target', 'stop_loss'],
    },
    
    'MOMENTUM_RIDE': {
        'description': 'Ride strong momentum with trailing stop',
        'entry': ['strong_momentum', 'volume_confirm', 'trend_up'],
        'exit': ['trailing_stop', 'momentum_fade'],
    },
    
    'EARNINGS_DIP': {
        'description': 'Buy post-earnings dip for recovery',
        'entry': ['post_earnings_drop', 'rsi_oversold'],
        'exit': ['profit_target', 'time_stop'],
    },
}


# =============================================================================
# BACKTESTER
# =============================================================================

class PatternBacktester:
    """Fast backtester for pattern discovery"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self._prepare_features()
    
    def _prepare_features(self):
        """Compute all features once"""
        df = self.df
        
        # Returns
        df['returns_1d'] = df['Close'].pct_change()
        df['returns_5d'] = df['Close'].pct_change(5)
        df['returns_21d'] = df['Close'].pct_change(21)
        
        # RSI variants
        for period in RSI_PERIODS:
            delta = df['Close'].diff()
            gain = delta.where(delta > 0, 0).rolling(period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
            rs = gain / (loss + 1e-10)
            df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
        # Moving averages
        for period in MA_FAST + MA_SLOW:
            df[f'sma_{period}'] = df['Close'].rolling(period).mean()
            df[f'ema_{period}'] = df['Close'].ewm(span=period).mean()
        
        # MACD
        for fast, slow in itertools.product(MACD_FAST, MACD_SLOW):
            if fast < slow:
                ema_fast = df['Close'].ewm(span=fast).mean()
                ema_slow = df['Close'].ewm(span=slow).mean()
                df[f'macd_{fast}_{slow}'] = ema_fast - ema_slow
                for signal in MACD_SIGNAL:
                    df[f'macd_signal_{fast}_{slow}_{signal}'] = df[f'macd_{fast}_{slow}'].ewm(span=signal).mean()
        
        # Bollinger Bands
        for period in BB_PERIODS:
            mid = df['Close'].rolling(period).mean()
            std = df['Close'].rolling(period).std()
            for num_std in BB_STD:
                df[f'bb_upper_{period}_{num_std}'] = mid + num_std * std
                df[f'bb_lower_{period}_{num_std}'] = mid - num_std * std
        
        # ATR
        for period in ATR_PERIODS:
            high_low = df['High'] - df['Low']
            high_close = (df['High'] - df['Close'].shift()).abs()
            low_close = (df['Low'] - df['Close'].shift()).abs()
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df[f'atr_{period}'] = tr.rolling(period).mean()
        
        # Volume
        df['volume_sma_20'] = df['Volume'].rolling(20).mean()
        df['volume_ratio'] = df['Volume'] / (df['volume_sma_20'] + 1e-10)
        
        # Support/Resistance
        df['support_20'] = df['Low'].rolling(20).min()
        df['resistance_20'] = df['High'].rolling(20).max()
        
        df.ffill(inplace=True)
        df.bfill(inplace=True)
        self.df = df
    
    def test_strategy(self, 
                      entry_conditions: Dict,
                      exit_conditions: Dict,
                      initial_capital: float = 100000) -> Dict:
        """Backtest a specific strategy combination"""
        
        df = self.df.copy()
        n = len(df)
        
        # Generate signals
        entry_signal = np.ones(n, dtype=bool)
        exit_signal = np.zeros(n, dtype=bool)
        
        # Entry conditions
        if 'rsi_oversold' in entry_conditions:
            period = entry_conditions.get('rsi_period', 14)
            threshold = entry_conditions.get('rsi_oversold', 30)
            entry_signal &= (df[f'rsi_{period}'] < threshold).values
        
        if 'rsi_overbought' in entry_conditions:
            period = entry_conditions.get('rsi_period', 14)
            threshold = entry_conditions.get('rsi_overbought', 70)
            entry_signal &= (df[f'rsi_{period}'] > threshold).values
        
        if 'ma_cross_up' in entry_conditions:
            fast = entry_conditions.get('ma_fast', 8)
            slow = entry_conditions.get('ma_slow', 21)
            ma_type = entry_conditions.get('ma_type', 'ema')
            fast_ma = df[f'{ma_type}_{fast}']
            slow_ma = df[f'{ma_type}_{slow}']
            entry_signal &= ((fast_ma > slow_ma) & (fast_ma.shift() <= slow_ma.shift())).values
        
        if 'price_drawdown' in entry_conditions:
            threshold = entry_conditions.get('drawdown_threshold', -0.08)
            entry_signal &= (df['returns_21d'] < threshold).values
        
        if 'above_long_ma' in entry_conditions:
            period = entry_conditions.get('long_ma', 200)
            if f'sma_{period}' in df.columns:
                entry_signal &= (df['Close'] > df[f'sma_{period}']).values
        
        if 'volume_surge' in entry_conditions:
            threshold = entry_conditions.get('volume_threshold', 2.0)
            entry_signal &= (df['volume_ratio'] > threshold).values
        
        if 'below_lower_bb' in entry_conditions:
            period = entry_conditions.get('bb_period', 20)
            std = entry_conditions.get('bb_std', 2.0)
            entry_signal &= (df['Close'] < df[f'bb_lower_{period}_{std}']).values
        
        # Exit conditions
        if 'rsi_recovered' in exit_conditions:
            period = exit_conditions.get('rsi_period', 14)
            threshold = exit_conditions.get('rsi_exit', 50)
            exit_signal |= (df[f'rsi_{period}'] > threshold).values
        
        if 'ma_cross_down' in exit_conditions:
            fast = exit_conditions.get('ma_fast', 8)
            slow = exit_conditions.get('ma_slow', 21)
            ma_type = exit_conditions.get('ma_type', 'ema')
            fast_ma = df[f'{ma_type}_{fast}']
            slow_ma = df[f'{ma_type}_{slow}']
            exit_signal |= ((fast_ma < slow_ma) & (fast_ma.shift() >= slow_ma.shift())).values
        
        if 'above_upper_bb' in exit_conditions:
            period = exit_conditions.get('bb_period', 20)
            std = exit_conditions.get('bb_std', 2.0)
            exit_signal |= (df['Close'] > df[f'bb_upper_{period}_{std}']).values
        
        # Simulate trades
        trades = []
        in_position = False
        entry_price = 0
        entry_idx = 0
        
        profit_target = exit_conditions.get('profit_target', 0.08)
        stop_loss = exit_conditions.get('stop_loss', -0.05)
        max_hold = exit_conditions.get('max_hold', 21)
        
        for i in range(50, n):  # Skip warmup period
            price = df['Close'].iloc[i]
            
            if not in_position:
                if entry_signal[i]:
                    in_position = True
                    entry_price = price
                    entry_idx = i
            else:
                pnl = (price / entry_price) - 1
                hold_days = i - entry_idx
                
                # Check exit conditions
                should_exit = False
                
                if pnl >= profit_target:
                    should_exit = True
                elif pnl <= stop_loss:
                    should_exit = True
                elif hold_days >= max_hold:
                    should_exit = True
                elif exit_signal[i]:
                    should_exit = True
                
                if should_exit:
                    trades.append({
                        'entry_idx': entry_idx,
                        'exit_idx': i,
                        'entry_price': entry_price,
                        'exit_price': price,
                        'pnl': pnl,
                        'hold_days': hold_days
                    })
                    in_position = False
        
        # Calculate metrics
        if not trades:
            return {
                'win_rate': 0,
                'avg_return': 0,
                'total_return': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0,
                'total_trades': 0,
            }
        
        pnls = [t['pnl'] for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        
        win_rate = len(wins) / len(pnls) if pnls else 0
        avg_return = np.mean(pnls) if pnls else 0
        total_return = np.prod([1 + p for p in pnls]) - 1
        
        # Calculate drawdown
        equity = initial_capital
        peak = equity
        max_dd = 0
        for pnl in pnls:
            equity *= (1 + pnl)
            peak = max(peak, equity)
            dd = (peak - equity) / peak
            max_dd = max(max_dd, dd)
        
        # Sharpe ratio
        if len(pnls) > 1 and np.std(pnls) > 0:
            sharpe = np.mean(pnls) / np.std(pnls) * np.sqrt(252 / np.mean([t['hold_days'] for t in trades]))
        else:
            sharpe = 0
        
        return {
            'win_rate': win_rate,
            'avg_return': avg_return,
            'total_return': total_return,
            'max_drawdown': max_dd,
            'sharpe_ratio': sharpe,
            'total_trades': len(trades),
            'avg_hold_days': np.mean([t['hold_days'] for t in trades]),
            'avg_win': np.mean(wins) if wins else 0,
            'avg_loss': np.mean(losses) if losses else 0,
        }


# =============================================================================
# DISCOVERY ENGINE
# =============================================================================

class CombinationDiscoveryEngine:
    """Search for winning pattern combinations"""
    
    def __init__(self, tickers: List[str]):
        self.tickers = tickers
        self.winning_patterns: List[WinningPattern] = []
        self.all_results = []
    
    def load_data(self, period: str = '2y'):
        """Load data for all tickers"""
        print(f"📥 Loading data for {len(self.tickers)} tickers...")
        
        self.data = {}
        for ticker in self.tickers:
            try:
                df = yf.download(ticker, period=period, progress=False)
                if len(df) > 100:
                    self.data[ticker] = df.reset_index()
                    print(f"   ✓ {ticker}: {len(df)} days")
            except:
                pass
        
        print(f"   Loaded {len(self.data)} tickers")
    
    def generate_combinations(self) -> List[Dict]:
        """Generate all strategy combinations to test"""
        combinations = []
        
        # DIP BUY combinations
        for rsi_period in RSI_PERIODS:
            for rsi_oversold in RSI_OVERSOLD:
                for drawdown in DRAWDOWN_THRESHOLDS:
                    for profit in PROFIT_TARGETS:
                        for stop in STOP_LOSSES:
                            for hold in HOLDING_PERIODS:
                                combinations.append({
                                    'name': 'DIP_BUY',
                                    'entry': {
                                        'price_drawdown': True,
                                        'rsi_oversold': True,
                                        'drawdown_threshold': drawdown,
                                        'rsi_period': rsi_period,
                                        'rsi_oversold': rsi_oversold,
                                    },
                                    'exit': {
                                        'profit_target': profit,
                                        'stop_loss': stop,
                                        'max_hold': hold,
                                    }
                                })
        
        # RSI BOUNCE combinations
        for rsi_period in RSI_PERIODS:
            for rsi_oversold in RSI_OVERSOLD:
                for rsi_exit in [40, 50, 60]:
                    for profit in PROFIT_TARGETS:
                        for stop in STOP_LOSSES:
                            combinations.append({
                                'name': 'RSI_BOUNCE',
                                'entry': {
                                    'rsi_oversold': True,
                                    'rsi_period': rsi_period,
                                    'rsi_oversold': rsi_oversold,
                                },
                                'exit': {
                                    'rsi_recovered': True,
                                    'rsi_period': rsi_period,
                                    'rsi_exit': rsi_exit,
                                    'profit_target': profit,
                                    'stop_loss': stop,
                                }
                            })
        
        # MA CROSSOVER combinations
        for ma_type in MA_TYPES:
            for fast in MA_FAST:
                for slow in MA_SLOW:
                    if fast < slow:
                        for profit in PROFIT_TARGETS:
                            for stop in STOP_LOSSES:
                                combinations.append({
                                    'name': 'MA_CROSS',
                                    'entry': {
                                        'ma_cross_up': True,
                                        'ma_type': ma_type,
                                        'ma_fast': fast,
                                        'ma_slow': slow,
                                    },
                                    'exit': {
                                        'ma_cross_down': True,
                                        'ma_type': ma_type,
                                        'ma_fast': fast,
                                        'ma_slow': slow,
                                        'profit_target': profit,
                                        'stop_loss': stop,
                                    }
                                })
        
        # BOLLINGER BAND combinations
        for bb_period in BB_PERIODS:
            for bb_std in BB_STD:
                for rsi_period in [14]:
                    for rsi_oversold in RSI_OVERSOLD:
                        for profit in PROFIT_TARGETS:
                            for stop in STOP_LOSSES:
                                combinations.append({
                                    'name': 'BB_BOUNCE',
                                    'entry': {
                                        'below_lower_bb': True,
                                        'bb_period': bb_period,
                                        'bb_std': bb_std,
                                        'rsi_oversold': True,
                                        'rsi_period': rsi_period,
                                        'rsi_oversold': rsi_oversold,
                                    },
                                    'exit': {
                                        'above_upper_bb': True,
                                        'bb_period': bb_period,
                                        'bb_std': bb_std,
                                        'profit_target': profit,
                                        'stop_loss': stop,
                                    }
                                })
        
        # Volume breakout combinations
        for volume_thresh in VOLUME_THRESHOLDS:
            for ma_period in [50, 100, 200]:
                for profit in PROFIT_TARGETS:
                    for stop in STOP_LOSSES:
                        combinations.append({
                            'name': 'VOLUME_BREAKOUT',
                            'entry': {
                                'volume_surge': True,
                                'volume_threshold': volume_thresh,
                                'above_long_ma': True,
                                'long_ma': ma_period,
                            },
                            'exit': {
                                'profit_target': profit,
                                'stop_loss': stop,
                                'max_hold': 10,
                            }
                        })
        
        print(f"📊 Generated {len(combinations)} combinations to test")
        return combinations
    
    def run_discovery(self, min_win_rate: float = 0.55, min_trades: int = 10):
        """Run the discovery process"""
        combinations = self.generate_combinations()
        
        print(f"\n🔬 Running Discovery...")
        print(f"   Min Win Rate: {min_win_rate*100:.0f}%")
        print(f"   Min Trades: {min_trades}")
        print("=" * 60)
        
        total = len(combinations) * len(self.data)
        tested = 0
        found = 0
        
        for combo in combinations:
            combo_results = []
            
            for ticker, df in self.data.items():
                backtester = PatternBacktester(df)
                results = backtester.test_strategy(combo['entry'], combo['exit'])
                
                if results['total_trades'] >= min_trades and results['win_rate'] >= min_win_rate:
                    results['ticker'] = ticker
                    results['combo'] = combo
                    combo_results.append(results)
                
                tested += 1
                if tested % 1000 == 0:
                    print(f"   Progress: {tested}/{total} ({tested/total*100:.1f}%) | Found: {found}")
            
            # If works across multiple tickers, it's a winner
            if len(combo_results) >= 3:
                avg_win_rate = np.mean([r['win_rate'] for r in combo_results])
                avg_return = np.mean([r['avg_return'] for r in combo_results])
                avg_sharpe = np.mean([r['sharpe_ratio'] for r in combo_results])
                
                if avg_win_rate >= min_win_rate and avg_sharpe > 0.5:
                    pattern = WinningPattern(
                        name=combo['name'],
                        description=f"{combo['name']} with {combo['entry']}",
                        entry_conditions=combo['entry'],
                        exit_conditions=combo['exit'],
                        backtest_results={r['ticker']: r for r in combo_results},
                        win_rate=avg_win_rate,
                        avg_return=avg_return,
                        max_drawdown=np.mean([r['max_drawdown'] for r in combo_results]),
                        sharpe_ratio=avg_sharpe,
                        total_trades=sum(r['total_trades'] for r in combo_results),
                        discovered_date=datetime.now().isoformat(),
                        tickers_tested=list(self.data.keys()),
                        confidence_score=min(avg_win_rate, 1.0) * min(avg_sharpe, 2.0) / 2
                    )
                    self.winning_patterns.append(pattern)
                    found += 1
                    
                    print(f"\n   🎯 FOUND: {combo['name']}")
                    print(f"      Win Rate: {avg_win_rate*100:.1f}%")
                    print(f"      Avg Return: {avg_return*100:.2f}%")
                    print(f"      Sharpe: {avg_sharpe:.2f}")
        
        print("\n" + "=" * 60)
        print(f"✅ Discovery Complete!")
        print(f"   Combinations Tested: {len(combinations)}")
        print(f"   Winning Patterns Found: {len(self.winning_patterns)}")
    
    def save_results(self, filename: str = 'discovered_patterns.json'):
        """Save discovered patterns"""
        results = {
            'discovery_date': datetime.now().isoformat(),
            'tickers_tested': list(self.data.keys()),
            'total_patterns': len(self.winning_patterns),
            'patterns': [asdict(p) for p in sorted(
                self.winning_patterns, 
                key=lambda x: x.confidence_score, 
                reverse=True
            )]
        }
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n💾 Saved to: {filename}")
        
        # Also save top patterns summary
        print("\n🏆 TOP PATTERNS:")
        for i, p in enumerate(results['patterns'][:10]):
            print(f"\n{i+1}. {p['name']}")
            print(f"   Win Rate: {p['win_rate']*100:.1f}%")
            print(f"   Avg Return: {p['avg_return']*100:.2f}%")
            print(f"   Sharpe: {p['sharpe_ratio']:.2f}")
            print(f"   Entry: {p['entry_conditions']}")
            print(f"   Exit: {p['exit_conditions']}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run combination discovery"""
    
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║   🔬 COMBINATION DISCOVERY ENGINE                                    ║
║                                                                      ║
║   Finding THE winning pattern combinations                           ║
║   Testing ALL indicator combinations                                 ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
""")
    
    # Tickers to test
    tickers = [
        'APLD', 'SERV', 'MRVL', 'HOOD', 'LUNR', 'TSLA', 'AMD', 'NVDA', 
        'MU', 'SPY', 'QQQ', 'GOOGL', 'META', 'INTC', 'BA', 'LAC'
    ]
    
    # Run discovery
    engine = CombinationDiscoveryEngine(tickers)
    engine.load_data(period='2y')
    engine.run_discovery(min_win_rate=0.55, min_trades=10)
    engine.save_results('discovered_patterns.json')
    
    return engine


if __name__ == '__main__':
    engine = main()
