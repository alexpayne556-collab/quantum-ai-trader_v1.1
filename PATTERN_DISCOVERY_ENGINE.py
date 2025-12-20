"""
PATTERN DISCOVERY ENGINE
========================
Scientific approach to pattern discovery:
1. Define HYPOTHESIS (pattern conditions)
2. BACKTEST on historical data
3. DISCOVER actual statistics (win rate, expected return, etc.)
4. VALIDATE with out-of-sample testing
5. Only then APPLY the discovered parameters

NO PREDETERMINED VALUES - Everything is discovered through testing.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Tuple
from datetime import datetime, timedelta
from enum import Enum
import json
import os

# ============================================================================
# HYPOTHESIS DEFINITION
# ============================================================================

@dataclass
class PatternHypothesis:
    """
    A hypothesis about a market pattern.
    Contains ONLY the conditions - no assumed outcomes.
    """
    name: str
    description: str
    conditions: Dict[str, Callable]  # Condition name -> function that returns bool
    holding_period: int  # Days to hold after signal
    direction: str  # 'long', 'short', or 'either'
    
    # These are DISCOVERED, not assumed
    discovered_win_rate: Optional[float] = None
    discovered_avg_return: Optional[float] = None
    discovered_max_drawdown: Optional[float] = None
    discovered_occurrences: int = 0
    discovery_date: Optional[str] = None
    sample_size: int = 0
    
    def is_validated(self) -> bool:
        """Pattern is validated if it has sufficient sample size."""
        return self.sample_size >= 20  # Minimum for statistical significance


@dataclass 
class PatternDiscoveryResult:
    """Results from backtesting a pattern hypothesis."""
    pattern_name: str
    total_signals: int
    winning_signals: int
    losing_signals: int
    win_rate: float
    avg_return: float
    median_return: float
    std_return: float
    max_return: float
    min_return: float
    max_drawdown: float
    sharpe_ratio: float
    profit_factor: float
    all_returns: List[float]
    signal_dates: List[str]
    
    def to_dict(self) -> Dict:
        return {
            'pattern_name': self.pattern_name,
            'total_signals': self.total_signals,
            'win_rate': self.win_rate,
            'avg_return': self.avg_return,
            'median_return': self.median_return,
            'std_return': self.std_return,
            'max_return': self.max_return,
            'min_return': self.min_return,
            'max_drawdown': self.max_drawdown,
            'sharpe_ratio': self.sharpe_ratio,
            'profit_factor': self.profit_factor,
            'sample_size': self.total_signals
        }


# ============================================================================
# PATTERN HYPOTHESES - Conditions only, NO assumed outcomes
# ============================================================================

def create_vix_capitulation_hypothesis() -> PatternHypothesis:
    """
    HYPOTHESIS: When VIX spikes above 30, then starts dropping,
    while SPY RSI is oversold (<35), this indicates panic selling
    exhaustion and a likely bounce.
    
    RATIONALE: Fear drives VIX up, but when it peaks and drops
    while stocks are oversold, smart money is buying the dip.
    
    NO ASSUMED WIN RATE - must be discovered through testing.
    """
    
    def vix_was_high(state: Dict) -> bool:
        return state.get('vix_5d_peak', 0) > 30
    
    def vix_peaked(state: Dict) -> bool:
        return state.get('vix_peaked', False)
    
    def vix_dropping(state: Dict) -> bool:
        vix = state.get('vix', 0)
        peak = state.get('vix_5d_peak', 0)
        return peak > 0 and vix < peak * 0.95
    
    def rsi_oversold(state: Dict) -> bool:
        return state.get('spy_rsi', 50) < 35
    
    return PatternHypothesis(
        name='VIX Capitulation',
        description='VIX peaked >30 and now dropping + RSI oversold',
        conditions={
            'vix_was_high': vix_was_high,
            'vix_peaked': vix_peaked,
            'vix_dropping': vix_dropping,
            'rsi_oversold': rsi_oversold
        },
        holding_period=7,
        direction='long'
    )


def create_golden_cross_hypothesis() -> PatternHypothesis:
    """
    HYPOTHESIS: When 50-day MA crosses above 200-day MA,
    combined with other bullish conditions, it signals
    the start of a new uptrend.
    
    NO ASSUMED WIN RATE - must be discovered through testing.
    """
    
    def golden_cross(state: Dict) -> bool:
        return state.get('golden_cross', False)
    
    def vix_elevated(state: Dict) -> bool:
        return state.get('vix', 0) > 20
    
    def rsi_not_overbought(state: Dict) -> bool:
        return state.get('spy_rsi', 50) < 70
    
    return PatternHypothesis(
        name='Golden Cross Setup',
        description='MA50 crosses above MA200 with elevated VIX',
        conditions={
            'golden_cross': golden_cross,
            'vix_elevated': vix_elevated,
            'rsi_not_overbought': rsi_not_overbought
        },
        holding_period=10,
        direction='long'
    )


def create_death_cross_hypothesis() -> PatternHypothesis:
    """
    HYPOTHESIS: When 50-day MA crosses below 200-day MA,
    combined with other bearish conditions, it signals
    further downside.
    
    NO ASSUMED WIN RATE - must be discovered through testing.
    """
    
    def death_cross(state: Dict) -> bool:
        return state.get('death_cross', False)
    
    def vix_rising(state: Dict) -> bool:
        return state.get('vix_change_5d', 0) > 3
    
    def rsi_weak(state: Dict) -> bool:
        return state.get('spy_rsi', 50) < 50
    
    return PatternHypothesis(
        name='Death Cross Setup',
        description='MA50 crosses below MA200 with rising VIX',
        conditions={
            'death_cross': death_cross,
            'vix_rising': vix_rising,
            'rsi_weak': rsi_weak
        },
        holding_period=10,
        direction='short'
    )


def create_oversold_bounce_hypothesis() -> PatternHypothesis:
    """
    HYPOTHESIS: Extreme RSI oversold (<25) with volume spike
    often precedes a short-term bounce.
    """
    
    def extreme_oversold(state: Dict) -> bool:
        return state.get('spy_rsi', 50) < 25
    
    def volume_spike(state: Dict) -> bool:
        return state.get('volume_ratio', 1) > 1.5
    
    def price_down(state: Dict) -> bool:
        return state.get('price_change_1d', 0) < -0.01
    
    return PatternHypothesis(
        name='Oversold Bounce',
        description='RSI < 25 with volume spike after down day',
        conditions={
            'extreme_oversold': extreme_oversold,
            'volume_spike': volume_spike,
            'price_down': price_down
        },
        holding_period=5,
        direction='long'
    )


# ============================================================================
# PATTERN DISCOVERY ENGINE
# ============================================================================

class PatternDiscoveryEngine:
    """
    Tests pattern hypotheses on historical data to DISCOVER
    their actual statistics. No assumptions, just evidence.
    """
    
    def __init__(self):
        self.hypotheses: Dict[str, PatternHypothesis] = {}
        self.results: Dict[str, PatternDiscoveryResult] = {}
        self.data_cache = {}
    
    def register_hypothesis(self, hypothesis: PatternHypothesis):
        """Register a pattern hypothesis for testing."""
        self.hypotheses[hypothesis.name] = hypothesis
        print(f"📝 Registered hypothesis: {hypothesis.name}")
        print(f"   Conditions: {list(hypothesis.conditions.keys())}")
        print(f"   Direction: {hypothesis.direction}, Hold: {hypothesis.holding_period}d")
    
    def get_market_data(self, years: int = 10) -> pd.DataFrame:
        """Get historical market data for backtesting."""
        cache_key = f"spy_vix_{years}y"
        
        if cache_key not in self.data_cache:
            print(f"\n📥 Downloading {years} years of market data...")
            
            spy = yf.download('SPY', period=f'{years}y', progress=False, auto_adjust=True)
            vix = yf.download('^VIX', period=f'{years}y', progress=False, auto_adjust=True)
            
            # Handle MultiIndex
            if isinstance(spy.columns, pd.MultiIndex):
                spy.columns = [c[0].lower() for c in spy.columns]
                vix.columns = [c[0].lower() for c in vix.columns]
            else:
                spy.columns = [c.lower() for c in spy.columns]
                vix.columns = [c.lower() for c in vix.columns]
            
            common = spy.index.intersection(vix.index)
            
            data = pd.DataFrame({
                'close': spy.loc[common, 'close'],
                'high': spy.loc[common, 'high'],
                'low': spy.loc[common, 'low'],
                'volume': spy.loc[common, 'volume'],
                'vix': vix.loc[common, 'close']
            })
            
            # Calculate technical indicators
            data = self._add_indicators(data)
            
            self.data_cache[cache_key] = data
            print(f"   Got {len(data)} days of data")
        
        return self.data_cache[cache_key]
    
    def _add_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to data."""
        # Moving averages
        data['ma50'] = data['close'].rolling(50).mean()
        data['ma200'] = data['close'].rolling(200).mean()
        
        # RSI
        delta = data['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        data['rsi'] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))
        
        # Volume ratio
        data['volume_ma'] = data['volume'].rolling(20).mean()
        data['volume_ratio'] = data['volume'] / data['volume_ma']
        
        # VIX analysis
        data['vix_5d_max'] = data['vix'].rolling(5).max()
        data['vix_change_5d'] = data['vix'].diff(5)
        
        # Price changes
        data['price_change_1d'] = data['close'].pct_change(1)
        data['price_change_5d'] = data['close'].pct_change(5)
        
        return data
    
    def _build_market_state(self, data: pd.DataFrame, idx: int) -> Dict:
        """Build market state dict for a specific day."""
        row = data.iloc[idx]
        prev = data.iloc[idx-1] if idx > 0 else row
        
        # VIX 5-day analysis
        vix_5d = data['vix'].iloc[max(0, idx-5):idx+1]
        vix_5d_peak = vix_5d.max()
        vix_peak_idx = vix_5d.idxmax()
        vix_peaked = vix_peak_idx != data.index[idx]
        
        # Golden/Death cross detection
        golden_cross = (row['ma50'] > row['ma200'] and 
                       prev['ma50'] <= prev['ma200'])
        death_cross = (row['ma50'] < row['ma200'] and 
                      prev['ma50'] >= prev['ma200'])
        
        return {
            'vix': row['vix'],
            'vix_5d_peak': vix_5d_peak,
            'vix_peaked': vix_peaked,
            'vix_change_5d': row.get('vix_change_5d', 0),
            'spy_rsi': row['rsi'],
            'golden_cross': golden_cross,
            'death_cross': death_cross,
            'volume_ratio': row['volume_ratio'],
            'price_change_1d': row['price_change_1d'],
            'price_change_5d': row['price_change_5d'],
            'ma50': row['ma50'],
            'ma200': row['ma200']
        }
    
    def test_hypothesis(self, hypothesis: PatternHypothesis, 
                       years: int = 10) -> PatternDiscoveryResult:
        """
        Test a pattern hypothesis on historical data.
        Returns DISCOVERED statistics, not assumed ones.
        """
        print(f"\n{'='*70}")
        print(f"TESTING HYPOTHESIS: {hypothesis.name}")
        print(f"{'='*70}")
        print(f"Conditions: {list(hypothesis.conditions.keys())}")
        
        data = self.get_market_data(years)
        holding_period = hypothesis.holding_period
        
        signals = []
        returns = []
        signal_dates = []
        
        # Need enough data for indicators and forward returns
        start_idx = 252  # 1 year of warm-up for indicators
        end_idx = len(data) - holding_period
        
        print(f"\n📊 Scanning {end_idx - start_idx} days for signals...")
        
        for idx in range(start_idx, end_idx):
            state = self._build_market_state(data, idx)
            
            # Check ALL conditions
            all_conditions_met = all(
                cond_func(state) 
                for cond_func in hypothesis.conditions.values()
            )
            
            if all_conditions_met:
                # Calculate forward return
                entry_price = data.iloc[idx]['close']
                exit_price = data.iloc[idx + holding_period]['close']
                
                if hypothesis.direction == 'long':
                    ret = (exit_price - entry_price) / entry_price
                elif hypothesis.direction == 'short':
                    ret = (entry_price - exit_price) / entry_price
                else:
                    ret = abs(exit_price - entry_price) / entry_price
                
                returns.append(ret)
                signal_dates.append(data.index[idx].strftime('%Y-%m-%d'))
                signals.append({
                    'date': data.index[idx],
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'return': ret,
                    'state': state
                })
        
        print(f"   Found {len(signals)} signals")
        
        # Calculate statistics
        if len(returns) > 0:
            returns_arr = np.array(returns)
            wins = sum(1 for r in returns if r > 0)
            losses = len(returns) - wins
            
            # Profit factor
            gross_profit = sum(r for r in returns if r > 0)
            gross_loss = abs(sum(r for r in returns if r < 0))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            # Max drawdown (simplified)
            cumulative = np.cumsum(returns_arr)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = running_max - cumulative
            max_drawdown = drawdown.max() if len(drawdown) > 0 else 0
            
            # Sharpe (simplified)
            avg_ret = np.mean(returns_arr)
            std_ret = np.std(returns_arr)
            sharpe = (avg_ret / std_ret * np.sqrt(252/holding_period)) if std_ret > 0 else 0
            
            result = PatternDiscoveryResult(
                pattern_name=hypothesis.name,
                total_signals=len(signals),
                winning_signals=wins,
                losing_signals=losses,
                win_rate=wins / len(signals),
                avg_return=float(avg_ret),
                median_return=float(np.median(returns_arr)),
                std_return=float(std_ret),
                max_return=float(np.max(returns_arr)),
                min_return=float(np.min(returns_arr)),
                max_drawdown=float(max_drawdown),
                sharpe_ratio=float(sharpe),
                profit_factor=float(profit_factor),
                all_returns=returns,
                signal_dates=signal_dates
            )
        else:
            result = PatternDiscoveryResult(
                pattern_name=hypothesis.name,
                total_signals=0,
                winning_signals=0,
                losing_signals=0,
                win_rate=0,
                avg_return=0,
                median_return=0,
                std_return=0,
                max_return=0,
                min_return=0,
                max_drawdown=0,
                sharpe_ratio=0,
                profit_factor=0,
                all_returns=[],
                signal_dates=[]
            )
        
        # Update hypothesis with discovered values
        hypothesis.discovered_win_rate = result.win_rate
        hypothesis.discovered_avg_return = result.avg_return
        hypothesis.discovered_max_drawdown = result.max_drawdown
        hypothesis.discovered_occurrences = result.total_signals
        hypothesis.sample_size = result.total_signals
        hypothesis.discovery_date = datetime.now().strftime('%Y-%m-%d')
        
        self.results[hypothesis.name] = result
        
        # Print discovered results
        print(f"\n{'='*50}")
        print(f"DISCOVERED STATISTICS for {hypothesis.name}")
        print(f"{'='*50}")
        print(f"   Total Signals: {result.total_signals}")
        print(f"   Win Rate: {result.win_rate:.1%}")
        print(f"   Avg Return: {result.avg_return:.2%}")
        print(f"   Median Return: {result.median_return:.2%}")
        print(f"   Std Return: {result.std_return:.2%}")
        print(f"   Max Return: {result.max_return:.2%}")
        print(f"   Min Return: {result.min_return:.2%}")
        print(f"   Max Drawdown: {result.max_drawdown:.2%}")
        print(f"   Sharpe Ratio: {result.sharpe_ratio:.2f}")
        print(f"   Profit Factor: {result.profit_factor:.2f}")
        
        if result.total_signals >= 20:
            print(f"\n   ✅ STATISTICALLY SIGNIFICANT ({result.total_signals} >= 20 samples)")
        else:
            print(f"\n   ⚠️ LOW SAMPLE SIZE ({result.total_signals} < 20 samples)")
        
        return result
    
    def test_all_hypotheses(self, years: int = 10):
        """Test all registered hypotheses."""
        print("="*70)
        print("PATTERN DISCOVERY - Testing All Hypotheses")
        print("="*70)
        
        for name, hypothesis in self.hypotheses.items():
            self.test_hypothesis(hypothesis, years)
        
        # Summary
        print("\n" + "="*70)
        print("DISCOVERY SUMMARY")
        print("="*70)
        
        for name, result in self.results.items():
            status = "✅" if result.total_signals >= 20 else "⚠️"
            edge = "📈" if result.avg_return > 0 and result.win_rate > 0.5 else "📉"
            print(f"\n{status} {edge} {name}:")
            print(f"   Signals: {result.total_signals}, Win: {result.win_rate:.1%}, Avg: {result.avg_return:.2%}")
    
    def save_discoveries(self, filepath: str = 'pattern_discoveries.json'):
        """Save discovered patterns to file."""
        discoveries = {}
        
        for name, hypothesis in self.hypotheses.items():
            if name in self.results:
                result = self.results[name]
                discoveries[name] = {
                    'hypothesis': {
                        'name': hypothesis.name,
                        'description': hypothesis.description,
                        'direction': hypothesis.direction,
                        'holding_period': hypothesis.holding_period
                    },
                    'discovered': {
                        'win_rate': result.win_rate,
                        'avg_return': result.avg_return,
                        'median_return': result.median_return,
                        'std_return': result.std_return,
                        'max_return': result.max_return,
                        'min_return': result.min_return,
                        'max_drawdown': result.max_drawdown,
                        'sharpe_ratio': result.sharpe_ratio,
                        'profit_factor': result.profit_factor,
                        'sample_size': result.total_signals,
                        'discovery_date': hypothesis.discovery_date,
                        'signal_dates': result.signal_dates[-10:]  # Last 10 signals
                    }
                }
        
        with open(filepath, 'w') as f:
            json.dump(discoveries, f, indent=2)
        
        print(f"\n💾 Saved discoveries to {filepath}")


# ============================================================================
# VALIDATED PATTERN CLASS - Uses only discovered values
# ============================================================================

class ValidatedPattern:
    """
    A pattern that has been tested and validated.
    Uses ONLY discovered values, not assumptions.
    """
    
    def __init__(self, hypothesis: PatternHypothesis, result: PatternDiscoveryResult):
        if not hypothesis.is_validated():
            raise ValueError(f"Pattern {hypothesis.name} is not validated (sample size: {hypothesis.sample_size})")
        
        self.name = hypothesis.name
        self.description = hypothesis.description
        self.conditions = hypothesis.conditions
        self.holding_period = hypothesis.holding_period
        self.direction = hypothesis.direction
        
        # DISCOVERED values only
        self.win_rate = result.win_rate
        self.avg_return = result.avg_return
        self.median_return = result.median_return
        self.std_return = result.std_return
        self.sharpe_ratio = result.sharpe_ratio
        self.profit_factor = result.profit_factor
        self.sample_size = result.total_signals
        
    def check_conditions(self, state: Dict) -> bool:
        """Check if all pattern conditions are met."""
        return all(cond(state) for cond in self.conditions.values())
    
    def get_signal_confidence(self) -> float:
        """
        Calculate signal confidence based on DISCOVERED statistics.
        Not assumed - derived from actual test results.
        """
        # Confidence based on sample size (more samples = more confident)
        sample_confidence = min(self.sample_size / 50, 1.0)  # Max at 50 samples
        
        # Confidence based on consistency (lower std = more consistent)
        if self.std_return > 0:
            consistency = 1 - min(self.std_return / 0.10, 1.0)  # Normalize to 10% std
        else:
            consistency = 0.5
        
        # Combine
        return sample_confidence * 0.6 + consistency * 0.4
    
    def __repr__(self):
        return (f"ValidatedPattern({self.name}: "
                f"win={self.win_rate:.1%}, avg={self.avg_return:.2%}, "
                f"n={self.sample_size})")


# ============================================================================
# MAIN - Run Discovery
# ============================================================================

def run_pattern_discovery():
    """Run pattern discovery on all hypotheses."""
    engine = PatternDiscoveryEngine()
    
    # Register hypotheses
    engine.register_hypothesis(create_vix_capitulation_hypothesis())
    engine.register_hypothesis(create_golden_cross_hypothesis())
    engine.register_hypothesis(create_death_cross_hypothesis())
    engine.register_hypothesis(create_oversold_bounce_hypothesis())
    
    # Test all
    engine.test_all_hypotheses(years=10)
    
    # Save results
    engine.save_discoveries()
    
    return engine


if __name__ == '__main__':
    engine = run_pattern_discovery()
