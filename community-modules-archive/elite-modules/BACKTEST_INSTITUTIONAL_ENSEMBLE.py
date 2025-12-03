"""
📊 BACKTEST FRAMEWORK - INSTITUTIONAL ENSEMBLE
==============================================
Walk-forward validation of the ensemble system

Features:
✅ Historical data simulation
✅ Realistic execution (slippage, fees)
✅ Walk-forward validation (no look-ahead bias)
✅ Module-by-module performance tracking
✅ Ensemble vs individual comparison
✅ Regime-specific analysis
✅ Comprehensive metrics (Sharpe, Win Rate, Max DD)

Target: Validate 60-70% win rate before live trading
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import json
from tqdm import tqdm
import logging

from INSTITUTIONAL_ENSEMBLE_ENGINE import (
    InstitutionalEnsembleEngine,
    Signal,
    INITIAL_WEIGHTS
)

# Import DataOrchestrator for clean scalar handling
import sys
from pathlib import Path
sys.path.insert(0, str(Path('/content/drive/MyDrive/QuantumAI/backend/modules')))
from data_orchestrator import DataOrchestrator

logger = logging.getLogger("Backtest")
logging.basicConfig(level=logging.INFO)

# ============================================================================
# CONFIGURATION
# ============================================================================

BACKTEST_CONFIG = {
    'start_date': '2024-06-01',  # 6 months of data
    'end_date': '2024-12-31',
    'initial_capital': 10000,
    'max_positions': 5,
    'hold_days': 5,  # Target hold period
    'commission': 0.001,  # 0.1% per trade
    'slippage': 0.002,  # 0.2% slippage
    'universe': [
        # Meme/volatile stocks
        'GME', 'AMC', 'BBBY', 'PLTR', 'NIO', 'RIVN', 'LCID',
        # Tech
        'NVDA', 'AMD', 'TSLA', 'AAPL', 'MSFT', 'GOOGL', 'META', 'AMZN',
        # Growth
        'SHOP', 'ROKU', 'SNAP', 'UBER', 'ABNB', 'DASH', 'SNOW',
        # Volatile
        'MARA', 'RIOT', 'COIN', 'HOOD', 'UPST', 'AFRM',
    ]
}

# ============================================================================
# MOCK SIGNAL GENERATORS (Replace with your real modules)
# ============================================================================

class MockDarkPoolTracker:
    """Mock dark pool signals based on volume"""
    def __init__(self, orchestrator: DataOrchestrator):
        self.orchestrator = orchestrator
    
    def analyze_ticker(self, symbol: str, data: pd.DataFrame) -> Dict:
        if len(data) < 20:
            return {'signal': 'NEUTRAL', 'confidence': 0.5}
        
        # Use orchestrator for clean scalar values
        volume_ratio = self.orchestrator.get_volume_ratio(data, period=20)
        
        if volume_ratio > 1.5:
            return {
                'signal': 'BUY',
                'confidence': min(0.7 + (volume_ratio - 1.5) * 0.1, 0.9),
                'volume_ratio': volume_ratio,
                'at_bid_ratio': 0.4  # Mock
            }
        return {'signal': 'NEUTRAL', 'confidence': 0.5}

class MockInsiderTracker:
    """Mock insider signals based on price action"""
    def __init__(self, orchestrator: DataOrchestrator):
        self.orchestrator = orchestrator
    
    def analyze_ticker(self, symbol: str, data: pd.DataFrame) -> Dict:
        if len(data) < 10:
            return {'signal': 'NEUTRAL', 'confidence': 0.5}
        
        # Use orchestrator for clean returns calculation
        returns = self.orchestrator.get_returns(data, period=10) / 10  # Avg daily
        
        if returns > 0.01:  # 1% avg daily gain
            return {
                'signal': 'BUY',
                'confidence': min(0.65 + returns * 20, 0.85),
                'buy_value': 1_000_000,  # Mock
                'sell_value': 0
            }
        return {'signal': 'NEUTRAL', 'confidence': 0.5}

class MockShortSqueezeScanner:
    """Mock squeeze signals based on volatility"""
    def __init__(self, orchestrator: DataOrchestrator):
        self.orchestrator = orchestrator
    
    def analyze_ticker(self, symbol: str, data: pd.DataFrame) -> Dict:
        if len(data) < 20:
            return {'signal': 'LOW_SQUEEZE', 'confidence': 0.4}
        
        # Calculate volatility using orchestrator
        volatility = self.orchestrator.to_scalar(data['Close'].pct_change().std())
        
        if volatility > 0.05:  # High volatility
            return {
                'signal': 'HIGH_SQUEEZE',
                'confidence': 0.75,
                'short_float_pct': 25.0  # Mock
            }
        return {'signal': 'LOW_SQUEEZE', 'confidence': 0.4}

class MockPatternScanner:
    """Mock pattern signals based on momentum"""
    def __init__(self, name: str, orchestrator: DataOrchestrator):
        self.name = name
        self.orchestrator = orchestrator
    
    def scan(self, symbol: str, data: pd.DataFrame) -> Dict:
        if len(data) < 5:
            return {'signal': 'NEUTRAL', 'confidence': 0.5}
        
        # Use orchestrator for clean returns calculation
        returns_5d = self.orchestrator.get_returns(data, period=5)
        
        if returns_5d > 0.03:  # 3% gain in 5 days
            return {
                'signal': 'BUY',
                'confidence': min(0.6 + returns_5d * 5, 0.8)
            }
        return {'signal': 'NEUTRAL', 'confidence': 0.5}

class MockSentimentEngine:
    """Mock sentiment based on recent returns"""
    def __init__(self, orchestrator: DataOrchestrator):
        self.orchestrator = orchestrator
    
    def analyze(self, symbol: str, data: pd.DataFrame) -> float:
        if len(data) < 3:
            return 0.5
        
        # Use orchestrator for clean returns
        recent_return = self.orchestrator.get_returns(data, period=3)
        sentiment = 0.5 + recent_return * 5  # Convert return to sentiment
        return float(np.clip(sentiment, 0.0, 1.0))

class MockRegimeDetector:
    """Mock regime detection"""
    def __init__(self, orchestrator: DataOrchestrator):
        self.orchestrator = orchestrator
    
    def detect_regime(self, data: pd.DataFrame) -> Dict:
        if len(data) < 20:
            return {'regime': 'neutral', 'vol': 'med'}
        
        # Use orchestrator for clean scalar values
        ma_20 = self.orchestrator.get_ma(data, period=20, column='close')
        current = self.orchestrator.get_last_close(data)
        
        if current > ma_20 * 1.02:
            regime = 'bull'
        elif current < ma_20 * 0.98:
            regime = 'bear'
        else:
            regime = 'chop'
        
        return {'regime': regime, 'vol': 'med'}

# ============================================================================
# RANKING MODEL MOCK
# ============================================================================

class MockRankingModel:
    """
    Mock ranking model - predicts returns based on momentum
    Replace with your real 80% success model!
    """
    def __init__(self, orchestrator: DataOrchestrator):
        self.orchestrator = orchestrator
    
    def predict(self, symbols: List[str], data_dict: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        predictions = {}
        
        for symbol in symbols:
            if symbol not in data_dict:
                predictions[symbol] = 0.0
                continue
            
            data = data_dict[symbol]
            if len(data) < 20:
                predictions[symbol] = 0.0
                continue
            
            # Use orchestrator for clean returns calculation
            returns_20d = self.orchestrator.get_returns(data, period=20)
            # Predict mean reversion
            predictions[symbol] = float(returns_20d * 0.5)
        
        return predictions

# ============================================================================
# BACKTEST ENGINE
# ============================================================================

class BacktestEngine:
    """
    Walk-forward backtest of institutional ensemble
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or BACKTEST_CONFIG
        self.ensemble = InstitutionalEnsembleEngine()
        
        # Initialize DataOrchestrator for clean scalar handling
        self.orchestrator = DataOrchestrator()
        logger.info("✅ DataOrchestrator initialized for scalar handling")
        
        # Mock modules with orchestrator (replace with real ones)
        self.dark_pool = MockDarkPoolTracker(self.orchestrator)
        self.insider = MockInsiderTracker(self.orchestrator)
        self.squeeze = MockShortSqueezeScanner(self.orchestrator)
        self.pregainer = MockPatternScanner('pregainer', self.orchestrator)
        self.day_trading = MockPatternScanner('day_trading', self.orchestrator)
        self.opportunity = MockPatternScanner('opportunity', self.orchestrator)
        self.sentiment = MockSentimentEngine(self.orchestrator)
        self.regime = MockRegimeDetector(self.orchestrator)
        self.ranking_model = MockRankingModel(self.orchestrator)
        
        # Results
        self.trades = []
        self.daily_portfolio_value = []
        self.signals_by_module = {module: [] for module in [
            'dark_pool', 'insider_trading', 'short_squeeze',
            'pregainer', 'day_trading', 'opportunity', 'sentiment'
        ]}
        
        logger.info("🧪 Backtest Engine initialized")
    
    def download_data(self) -> Dict[str, pd.DataFrame]:
        """Download historical data for universe"""
        logger.info(f"📥 Downloading data for {len(self.config['universe'])} stocks...")
        
        data_dict = {}
        for symbol in tqdm(self.config['universe'], desc="Downloading"):
            try:
                data = yf.download(
                    symbol,
                    start=self.config['start_date'],
                    end=self.config['end_date'],
                    progress=False,
                    auto_adjust=True  # Explicitly set to suppress FutureWarning
                )
                
                if not data.empty and len(data) > 20:
                    data_dict[symbol] = data
                    logger.debug(f"  ✅ {symbol}: {len(data)} days")
            except Exception as e:
                logger.warning(f"  ❌ {symbol}: {str(e)}")
        
        logger.info(f"✅ Downloaded data for {len(data_dict)} stocks")
        return data_dict
    
    def run_backtest(self) -> Dict:
        """
        Run complete walk-forward backtest
        """
        logger.info("="*80)
        logger.info("🚀 STARTING BACKTEST")
        logger.info("="*80)
        
        # Download data
        data_dict = self.download_data()
        
        if not data_dict:
            logger.error("❌ No data available for backtest!")
            return {}
        
        # Get date range
        all_dates = sorted(set(date for df in data_dict.values() for date in df.index))
        
        # Initialize portfolio
        capital = self.config['initial_capital']
        positions = {}  # {symbol: {'entry_date', 'entry_price', 'shares', 'signals'}}
        
        logger.info(f"\n📅 Backtesting {len(all_dates)} trading days...")
        logger.info(f"💰 Initial Capital: ${capital:,.2f}")
        logger.info(f"📊 Max Positions: {self.config['max_positions']}")
        logger.info(f"⏱️  Hold Period: {self.config['hold_days']} days\n")
        
        # Walk forward through time
        for i, current_date in enumerate(tqdm(all_dates[20:], desc="Backtesting")):
            # Get data up to current date (no look-ahead bias!)
            current_data = {
                symbol: df[df.index <= current_date]
                for symbol, df in data_dict.items()
            }
            
            # Close expired positions
            to_close = []
            for symbol, pos in positions.items():
                days_held = (current_date - pos['entry_date']).days
                if days_held >= self.config['hold_days']:
                    to_close.append(symbol)
            
            for symbol in to_close:
                capital += self._close_position(symbol, current_date, positions, current_data)
            
            # Generate new signals if we have room
            if len(positions) < self.config['max_positions']:
                new_positions = self._generate_signals(current_date, current_data, capital, positions)
                
                for symbol, entry_price, signals in new_positions:
                    capital -= self._open_position(symbol, entry_price, current_date, 
                                                   signals, positions, capital)
            
            # Track portfolio value
            portfolio_value = capital + sum(
                pos['shares'] * float(current_data[symbol]['Close'].iloc[-1])
                for symbol, pos in positions.items()
                if symbol in current_data and len(current_data[symbol]) > 0
            )
            # Ensure portfolio_value is a scalar
            portfolio_value = float(portfolio_value) if not isinstance(portfolio_value, (int, float)) else portfolio_value
            
            self.daily_portfolio_value.append({
                'date': current_date,
                'value': portfolio_value,
                'cash': capital,
                'positions': len(positions)
            })
        
        # Close all remaining positions
        final_date = all_dates[-1]
        for symbol in list(positions.keys()):
            capital += self._close_position(symbol, final_date, positions, current_data)
        
        # Calculate results
        results = self._calculate_results(capital)
        
        return results
    
    def _generate_signals(self, date, data_dict, capital, current_positions) -> List[Tuple]:
        """Generate signals for current date"""
        # Run ranking model (Tier 1)
        ranking_predictions = self.ranking_model.predict(
            list(data_dict.keys()),
            data_dict
        )
        
        # Filter universe
        top_stocks = self.ensemble.filter_universe(ranking_predictions)
        
        # Exclude stocks we already have
        candidates = [s for s in top_stocks if s not in current_positions]
        
        new_positions = []
        
        for symbol in candidates:
            if len(new_positions) + len(current_positions) >= self.config['max_positions']:
                break
            
            data = data_dict[symbol]
            if len(data) < 20:
                continue
            
            # Gather signals (Tier 2)
            signals = []
            
            # Dark pool
            dp_result = self.dark_pool.analyze_ticker(symbol, data)
            if dp_result['signal'] == 'BUY':
                signals.append(Signal('dark_pool', 'BUY', dp_result['confidence'], dp_result))
                self.signals_by_module['dark_pool'].append((date, symbol, dp_result['confidence']))
            
            # Insider
            insider_result = self.insider.analyze_ticker(symbol, data)
            if insider_result['signal'] == 'BUY':
                signals.append(Signal('insider_trading', 'BUY', insider_result['confidence'], insider_result))
                self.signals_by_module['insider_trading'].append((date, symbol, insider_result['confidence']))
            
            # Patterns
            for scanner_name, scanner in [
                ('pregainer', self.pregainer),
                ('day_trading', self.day_trading),
                ('opportunity', self.opportunity)
            ]:
                pattern_result = scanner.scan(symbol, data)
                if pattern_result['signal'] == 'BUY':
                    signals.append(Signal(scanner_name, 'BUY', pattern_result['confidence']))
                    self.signals_by_module[scanner_name].append((date, symbol, pattern_result['confidence']))
            
            # Sentiment
            sentiment_score = self.sentiment.analyze(symbol, data)
            if sentiment_score > 0.6:
                signals.append(Signal('sentiment', 'BUY', sentiment_score))
                self.signals_by_module['sentiment'].append((date, symbol, sentiment_score))
            
            # Regime
            regime_result = self.regime.detect_regime(data)
            
            # Get ranking percentile
            ranking_percentile = self.ensemble.universe_filter.get_percentile(
                symbol, ranking_predictions
            )
            
            # Evaluate through ensemble
            decision = self.ensemble.evaluate_stock(
                symbol=symbol,
                signals=signals,
                ranking_percentile=ranking_percentile,
                regime=regime_result['regime']
            )
            
            # Only take BUY_FULL signals for backtest
            if decision['action'] == 'BUY_FULL' and len(signals) >= 2:
                entry_price = float(data['Close'].iloc[-1]) * (1 + self.config['slippage'])
                new_positions.append((symbol, entry_price, signals))
        
        return new_positions
    
    def _open_position(self, symbol, entry_price, date, signals, positions, capital) -> float:
        """Open a position"""
        # Ensure entry_price is a scalar
        entry_price = float(entry_price) if not isinstance(entry_price, (int, float)) else entry_price
        
        # Position size: equal weight across max positions
        position_size = capital / self.config['max_positions']
        shares = position_size / entry_price
        cost = shares * entry_price * (1 + self.config['commission'])
        
        positions[symbol] = {
            'entry_date': date,
            'entry_price': entry_price,
            'shares': shares,
            'signals': signals,
            'cost': cost
        }
        
        logger.debug(f"  🟢 OPEN {symbol} @ ${entry_price:.2f} ({shares:.2f} shares)")
        return cost
    
    def _close_position(self, symbol, date, positions, data_dict) -> float:
        """Close a position and log outcome"""
        pos = positions[symbol]
        
        if symbol not in data_dict or len(data_dict[symbol]) == 0:
            # Can't get exit price, assume entry price (no gain/loss)
            proceeds = pos['cost']
        else:
            exit_price = float(data_dict[symbol]['Close'].iloc[-1]) * (1 - self.config['slippage'])
            proceeds = pos['shares'] * exit_price * (1 - self.config['commission'])
        
        pnl = proceeds - pos['cost']
        pnl_pct = (pnl / pos['cost']) * 100
        
        days_held = (date - pos['entry_date']).days
        
        # Log trade
        self.trades.append({
            'symbol': symbol,
            'entry_date': pos['entry_date'],
            'exit_date': date,
            'entry_price': pos['entry_price'],
            'exit_price': proceeds / pos['shares'] if pos['shares'] > 0 else pos['entry_price'],
            'shares': pos['shares'],
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'days_held': days_held,
            'signals': [s.name for s in pos['signals']]
        })
        
        # Log to ensemble for RL
        self.ensemble.log_trade_outcome(
            symbol=symbol,
            signals=pos['signals'],
            outcome=pnl_pct,
            days_held=days_held
        )
        
        logger.debug(f"  🔴 CLOSE {symbol} @ ${proceeds/pos['shares']:.2f} "
                    f"({pnl_pct:+.2f}% in {days_held} days)")
        
        del positions[symbol]
        return proceeds
    
    def _calculate_results(self, final_capital) -> Dict:
        """Calculate comprehensive backtest results"""
        initial = self.config['initial_capital']
        total_return = ((final_capital - initial) / initial) * 100
        
        trades_df = pd.DataFrame(self.trades)
        
        if len(trades_df) == 0:
            logger.error("❌ No trades executed!")
            return {}
        
        wins = trades_df[trades_df['pnl_pct'] > 0]
        losses = trades_df[trades_df['pnl_pct'] <= 0]
        
        win_rate = len(wins) / len(trades_df)
        avg_win = wins['pnl_pct'].mean() if len(wins) > 0 else 0
        avg_loss = losses['pnl_pct'].mean() if len(losses) > 0 else 0
        
        # Sharpe ratio - with robust error handling
        daily_returns = pd.DataFrame(self.daily_portfolio_value)
        
        # Ensure 'value' column contains float values, not Series
        if 'value' in daily_returns.columns:
            # Convert to numeric, handling any nested Series
            daily_returns['value'] = daily_returns['value'].apply(
                lambda x: float(x) if np.isscalar(x) else (
                    float(x.iloc[0]) if hasattr(x, 'iloc') else (
                        float(x[0]) if hasattr(x, '__len__') else float(x)
                    )
                )
            )
            daily_returns['value'] = pd.to_numeric(daily_returns['value'], errors='coerce')
            daily_returns = daily_returns.dropna(subset=['value'])
        
        # Calculate returns
        if len(daily_returns) > 1:
            daily_returns['return'] = daily_returns['value'].pct_change()
            # Remove first row (NaN from pct_change)
            daily_returns = daily_returns.dropna(subset=['return'])
            
            # Calculate Sharpe ratio safely
            if len(daily_returns) > 1:
                mean_return = daily_returns['return'].mean()
                std_return = daily_returns['return'].std()
                
                if std_return != 0 and not np.isnan(std_return) and not np.isnan(mean_return):
                    sharpe = (mean_return / std_return) * np.sqrt(252)
                else:
                    sharpe = 0.0
            else:
                sharpe = 0.0
        else:
            sharpe = 0.0
        
        # Max drawdown
        rolling_max = daily_returns['value'].expanding().max()
        drawdown = (daily_returns['value'] - rolling_max) / rolling_max
        max_drawdown = drawdown.min() * 100
        
        results = {
            'total_trades': len(trades_df),
            'win_rate': win_rate,
            'total_return': total_return,
            'final_capital': final_capital,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': abs(avg_win / avg_loss) if avg_loss != 0 else 0,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'avg_days_held': trades_df['days_held'].mean(),
            'trades_df': trades_df,
            'ensemble_stats': self.ensemble.get_performance_stats()
        }
        
        return results
    
    def print_results(self, results: Dict):
        """Print beautiful results summary"""
        print("\n" + "="*80)
        print("📊 BACKTEST RESULTS")
        print("="*80)
        
        print(f"\n💰 RETURNS:")
        print(f"  Initial Capital:  ${self.config['initial_capital']:,.2f}")
        print(f"  Final Capital:    ${results['final_capital']:,.2f}")
        print(f"  Total Return:     {results['total_return']:+.2f}%")
        
        print(f"\n📈 PERFORMANCE:")
        print(f"  Total Trades:     {results['total_trades']}")
        print(f"  Win Rate:         {results['win_rate']:.1%} ⭐")
        print(f"  Sharpe Ratio:     {results['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown:     {results['max_drawdown']:.2f}%")
        print(f"  Avg Days Held:    {results['avg_days_held']:.1f} days")
        
        print(f"\n💵 WIN/LOSS:")
        print(f"  Avg Win:          {results['avg_win']:+.2f}%")
        print(f"  Avg Loss:         {results['avg_loss']:+.2f}%")
        print(f"  Profit Factor:    {results['profit_factor']:.2f}x")
        
        print(f"\n🎯 TARGET COMPARISON:")
        target_met = "✅" if results['win_rate'] >= 0.60 else "❌"
        print(f"  Win Rate Target:  60-70% {target_met}")
        print(f"  Actual Win Rate:  {results['win_rate']:.1%}")
        
        sharpe_met = "✅" if results['sharpe_ratio'] >= 1.5 else "❌"
        print(f"  Sharpe Target:    >1.5 {sharpe_met}")
        print(f"  Actual Sharpe:    {results['sharpe_ratio']:.2f}")
        
        dd_met = "✅" if results['max_drawdown'] > -15 else "❌"
        print(f"  Max DD Target:    <15% {dd_met}")
        print(f"  Actual Max DD:    {results['max_drawdown']:.2f}%")
        
        print("\n" + "="*80)
        
        # Module performance
        if 'ensemble_stats' in results and results['ensemble_stats']:
            stats = results['ensemble_stats']
            print("\n📊 MODULE PERFORMANCE:")
            for module, perf in stats.get('by_module', {}).items():
                if perf['trades'] > 0:
                    print(f"  {module:20s}: {perf['accuracy']:5.1%} ({perf['trades']} trades)")
        
        print("\n" + "="*80)
    
    def save_results(self, results: Dict, filename: str = 'backtest_results.json'):
        """Save results to file"""
        # Convert non-serializable objects
        save_data = {
            k: v for k, v in results.items()
            if k != 'trades_df' and k != 'ensemble_stats'
        }
        
        # Save trades separately
        if 'trades_df' in results:
            results['trades_df'].to_csv('backtest_trades.csv', index=False)
            logger.info("💾 Trades saved to backtest_trades.csv")
        
        with open(filename, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        logger.info(f"💾 Results saved to {filename}")

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("🧪 INSTITUTIONAL ENSEMBLE BACKTEST")
    print("="*80)
    print("\nThis will backtest the ensemble system on 6 months of data")
    print("Expected runtime: 5-10 minutes\n")
    
    # Run backtest
    backtest = BacktestEngine()
    results = backtest.run_backtest()
    
    if results:
        backtest.print_results(results)
        backtest.save_results(results)
        
        # Save learned weights
        backtest.ensemble.save_weights('backtest_learned_weights.json')
        
        print("\n✅ Backtest complete!")
        print("📁 Files created:")
        print("   - backtest_results.json")
        print("   - backtest_trades.csv")
        print("   - backtest_learned_weights.json")

