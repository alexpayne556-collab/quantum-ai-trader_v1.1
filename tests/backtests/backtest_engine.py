"""
Comprehensive Testing Framework
Backtesting, forward testing, ablation studies, scenario replay, drift monitoring.

Production-grade testing harness for validating forecaster before deployment.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import json


@dataclass
class BacktestConfig:
    """Configuration for backtesting"""
    start_date: str = "2021-01-01"
    end_date: str = "2025-12-08"
    initial_capital: float = 100000.0
    
    # Walk-forward validation
    train_period_months: int = 18  # Train on 18 months
    test_period_months: int = 3    # Test on 3 months
    
    # Regime-aware CV
    regime_stratified: bool = True
    min_trades_per_regime: int = 20
    
    # Execution assumptions
    slippage_bps: float = 5.0  # 5 bps slippage
    commission_per_trade: float = 1.0  # $1 per trade
    
    # Metrics to track
    track_sharpe: bool = True
    track_max_drawdown: bool = True
    track_win_rate: bool = True
    track_per_regime: bool = True
    track_per_sector: bool = True


class BacktestEngine:
    """
    Walk-forward backtesting with regime-aware cross-validation.
    
    Usage:
        engine = BacktestEngine(config=BacktestConfig())
        results = engine.run_backtest(
            forecaster=integrated_forecaster,
            data_dict=historical_data,
            tickers=ticker_list
        )
        engine.generate_report(results)
    """
    
    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()
        self.trades = []
        self.equity_curve = []
    
    def run_walk_forward_backtest(
        self,
        forecaster,
        data_dict: Dict[str, pd.DataFrame],
        tickers: List[str]
    ) -> Dict:
        """
        Run walk-forward backtest (train → test → retrain → test ...).
        
        Architecture:
        1. Split data into train/test windows
        2. Train forecaster on train window
        3. Test on test window (never seen before)
        4. Roll forward and repeat
        5. Aggregate results across all test windows
        """
        start = pd.Timestamp(self.config.start_date)
        end = pd.Timestamp(self.config.end_date)
        
        train_period = pd.DateOffset(months=self.config.train_period_months)
        test_period = pd.DateOffset(months=self.config.test_period_months)
        
        current_train_start = start
        results_per_window = []
        
        while current_train_start < end:
            # Define windows
            train_end = current_train_start + train_period
            test_start = train_end
            test_end = test_start + test_period
            
            if test_end > end:
                break
            
            print(f"\n=== Walk-Forward Window ===")
            print(f"Train: {current_train_start.date()} to {train_end.date()}")
            print(f"Test:  {test_start.date()} to {test_end.date()}")
            
            # 1. Train forecaster on train window
            train_data = self._slice_data(data_dict, current_train_start, train_end)
            forecaster.fit(train_data, tickers)
            
            # 2. Test on test window
            test_data = self._slice_data(data_dict, test_start, test_end)
            window_results = self._run_test_window(forecaster, test_data, tickers)
            results_per_window.append(window_results)
            
            # 3. Roll forward
            current_train_start += test_period
        
        # Aggregate across all windows
        aggregated = self._aggregate_results(results_per_window)
        
        return aggregated
    
    def _run_test_window(self, forecaster, data_dict: Dict[str, pd.DataFrame], tickers: List[str]) -> Dict:
        """Run backtest on a single test window"""
        
        capital = self.config.initial_capital
        positions = {}
        trades_in_window = []
        
        # Get all dates in window
        all_dates = sorted(set().union(*[set(data_dict[t].index) for t in tickers if t in data_dict]))
        
        for date in all_dates:
            # Daily loop: evaluate signals, open/close positions
            
            # 1. Close expired positions
            capital, closed_trades = self._close_expired_positions(positions, data_dict, date, capital)
            trades_in_window.extend(closed_trades)
            
            # 2. Get signals for all tickers
            for ticker in tickers:
                if ticker not in data_dict or date not in data_dict[ticker].index:
                    continue
                
                # Get forecast
                forecast = forecaster.forecast(ticker, data_dict, as_of_date=date)
                
                # Skip if below confidence threshold
                if forecast['confidence'] < 0.55:
                    continue
                
                # Open new position if signal
                if forecast['direction'] == 'UP' and ticker not in positions:
                    position = self._open_position(ticker, forecast, data_dict[ticker].loc[date], capital)
                    if position:
                        positions[ticker] = position
                        capital -= position['position_dollars']
        
        # Close all remaining positions at end of window
        for ticker in list(positions.keys()):
            if ticker in data_dict:
                exit_price = data_dict[ticker].iloc[-1]['Close']
                trade = self._close_position(positions[ticker], exit_price, data_dict[ticker].index[-1])
                trades_in_window.append(trade)
                capital += trade['exit_value']
        
        # Calculate metrics for this window
        window_metrics = self._calculate_metrics(trades_in_window, self.config.initial_capital, capital)
        
        return {
            'trades': trades_in_window,
            'metrics': window_metrics,
            'final_capital': capital,
        }
    
    def _calculate_metrics(self, trades: List[Dict], initial_capital: float, final_capital: float) -> Dict:
        """Calculate performance metrics"""
        
        if len(trades) == 0:
            return {'num_trades': 0}
        
        returns = [t['return_pct'] for t in trades]
        
        # Win rate
        wins = sum(1 for r in returns if r > 0)
        win_rate = wins / len(returns)
        
        # Average win/loss
        winning_trades = [r for r in returns if r > 0]
        losing_trades = [r for r in returns if r < 0]
        avg_win = np.mean(winning_trades) if winning_trades else 0
        avg_loss = np.mean(losing_trades) if losing_trades else 0
        
        # Sharpe ratio (annualized)
        returns_array = np.array(returns)
        sharpe = np.mean(returns_array) / np.std(returns_array) * np.sqrt(252) if np.std(returns_array) > 0 else 0
        
        # Max drawdown
        equity_curve = [initial_capital]
        for trade in trades:
            equity_curve.append(equity_curve[-1] * (1 + trade['return_pct']))
        max_drawdown = self._calculate_max_drawdown(equity_curve)
        
        # Total return
        total_return = (final_capital - initial_capital) / initial_capital
        
        return {
            'num_trades': len(trades),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'total_return': total_return,
            'profit_factor': abs(sum(winning_trades) / sum(losing_trades)) if losing_trades else 0,
        }
    
    @staticmethod
    def _calculate_max_drawdown(equity_curve: List[float]) -> float:
        """Calculate maximum drawdown from equity curve"""
        peak = equity_curve[0]
        max_dd = 0.0
        
        for value in equity_curve:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd
        
        return max_dd
    
    @staticmethod
    def _slice_data(data_dict: Dict[str, pd.DataFrame], start: pd.Timestamp, end: pd.Timestamp) -> Dict[str, pd.DataFrame]:
        """Slice data dict to date range"""
        return {
            ticker: df[(df.index >= start) & (df.index <= end)]
            for ticker, df in data_dict.items()
        }
    
    def _open_position(self, ticker: str, forecast: Dict, entry_bar: pd.Series, capital: float) -> Optional[Dict]:
        """Open a new position"""
        entry_price = entry_bar['Close']
        position_dollars = forecast['position_size'] * capital
        
        # Apply slippage
        slippage = entry_price * (self.config.slippage_bps / 10000)
        entry_price_after_slippage = entry_price + slippage
        
        shares = position_dollars / entry_price_after_slippage
        
        return {
            'ticker': ticker,
            'entry_date': entry_bar.name,
            'entry_price': entry_price_after_slippage,
            'shares': shares,
            'position_dollars': position_dollars,
            'stop_loss': entry_price_after_slippage * (1 + forecast['stop_loss']),
            'take_profit': entry_price_after_slippage * (1 + forecast['take_profit']),
            'hold_until': entry_bar.name + timedelta(days=forecast['hold_period_days']),
            'forecast': forecast,
        }
    
    def _close_position(self, position: Dict, exit_price: float, exit_date: pd.Timestamp) -> Dict:
        """Close a position and return trade record"""
        
        # Apply slippage
        slippage = exit_price * (self.config.slippage_bps / 10000)
        exit_price_after_slippage = exit_price - slippage
        
        exit_value = position['shares'] * exit_price_after_slippage
        pnl = exit_value - position['position_dollars']
        return_pct = pnl / position['position_dollars']
        
        return {
            'ticker': position['ticker'],
            'entry_date': position['entry_date'],
            'exit_date': exit_date,
            'entry_price': position['entry_price'],
            'exit_price': exit_price_after_slippage,
            'shares': position['shares'],
            'position_dollars': position['position_dollars'],
            'exit_value': exit_value,
            'pnl': pnl,
            'return_pct': return_pct,
            'hold_days': (exit_date - position['entry_date']).days,
            'forecast_confidence': position['forecast']['confidence'],
        }
    
    def _close_expired_positions(self, positions: Dict, data_dict: Dict, current_date: pd.Timestamp, capital: float) -> Tuple[float, List[Dict]]:
        """Close positions that hit stops, targets, or hold period"""
        closed_trades = []
        
        for ticker in list(positions.keys()):
            if ticker not in data_dict or current_date not in data_dict[ticker].index:
                continue
            
            position = positions[ticker]
            current_bar = data_dict[ticker].loc[current_date]
            current_price = current_bar['Close']
            
            # Check stop-loss
            if current_price <= position['stop_loss']:
                trade = self._close_position(position, position['stop_loss'], current_date)
                closed_trades.append(trade)
                capital += trade['exit_value']
                del positions[ticker]
                continue
            
            # Check take-profit
            if current_price >= position['take_profit']:
                trade = self._close_position(position, position['take_profit'], current_date)
                closed_trades.append(trade)
                capital += trade['exit_value']
                del positions[ticker]
                continue
            
            # Check hold period
            if current_date >= position['hold_until']:
                trade = self._close_position(position, current_price, current_date)
                closed_trades.append(trade)
                capital += trade['exit_value']
                del positions[ticker]
        
        return capital, closed_trades
    
    def _aggregate_results(self, results_per_window: List[Dict]) -> Dict:
        """Aggregate results across all walk-forward windows"""
        
        all_trades = []
        for window in results_per_window:
            all_trades.extend(window['trades'])
        
        # Overall metrics
        initial_capital = self.config.initial_capital
        final_capital = results_per_window[-1]['final_capital'] if results_per_window else initial_capital
        
        overall_metrics = self._calculate_metrics(all_trades, initial_capital, final_capital)
        
        # Per-regime metrics
        regime_metrics = self._calculate_per_regime_metrics(all_trades)
        
        # Per-sector metrics
        sector_metrics = self._calculate_per_sector_metrics(all_trades)
        
        return {
            'overall': overall_metrics,
            'per_regime': regime_metrics,
            'per_sector': sector_metrics,
            'all_trades': all_trades,
            'num_windows': len(results_per_window),
        }
    
    @staticmethod
    def _calculate_per_regime_metrics(trades: List[Dict]) -> Dict:
        """Calculate metrics broken down by regime"""
        # TODO: Group trades by regime and calculate metrics
        return {}
    
    @staticmethod
    def _calculate_per_sector_metrics(trades: List[Dict]) -> Dict:
        """Calculate metrics broken down by sector"""
        # TODO: Group trades by sector and calculate metrics
        return {}
    
    def generate_report(self, results: Dict, output_path: str = "backtest_report.json"):
        """Generate comprehensive backtest report"""
        
        report = {
            'config': {
                'start_date': self.config.start_date,
                'end_date': self.config.end_date,
                'initial_capital': self.config.initial_capital,
            },
            'overall_metrics': results['overall'],
            'per_regime_metrics': results['per_regime'],
            'per_sector_metrics': results['per_sector'],
            'num_trades': len(results['all_trades']),
            'num_windows': results['num_windows'],
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n✅ Backtest report saved to {output_path}")
        print(f"\n=== OVERALL RESULTS ===")
        for key, value in results['overall'].items():
            if isinstance(value, float):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")


# ========== ABLATION TESTING ==========

class AblationStudy:
    """
    Test incremental value of each component.
    
    Ablation matrix:
    1. Patterns only
    2. Patterns + regimes
    3. Patterns + regimes + research features
    4. Patterns + regimes + research + calibration
    5. Full system (+ position sizing)
    
    Measure: Delta Sharpe ratio at each step
    """
    
    def __init__(self, backtest_engine: BacktestEngine):
        self.engine = backtest_engine
    
    def run_ablation(self, forecaster_components: Dict, data_dict: Dict, tickers: List[str]) -> Dict:
        """
        Run ablation study to measure incremental value.
        
        Returns: {component: delta_sharpe}
        """
        # TODO: Implement ablation logic
        raise NotImplementedError("Ablation study coming soon")


# ========== SCENARIO REPLAY ==========

class ScenarioTester:
    """
    Replay historical scenarios (2020 crash, 2022 rates shock, 2023 AI melt-up).
    
    Validates forecaster robustness to known stress events.
    """
    
    SCENARIOS = {
        '2020_covid_crash': ('2020-02-01', '2020-04-30'),
        '2022_rates_shock': ('2022-01-01', '2022-06-30'),
        '2023_ai_meltup': ('2023-01-01', '2023-12-31'),
    }
    
    def __init__(self, backtest_engine: BacktestEngine):
        self.engine = backtest_engine
    
    def replay_scenario(self, scenario_name: str, forecaster, data_dict: Dict, tickers: List[str]) -> Dict:
        """Replay a specific historical scenario"""
        
        if scenario_name not in self.SCENARIOS:
            raise ValueError(f"Unknown scenario: {scenario_name}")
        
        start, end = self.SCENARIOS[scenario_name]
        
        # Override backtest config dates
        original_start = self.engine.config.start_date
        original_end = self.engine.config.end_date
        
        self.engine.config.start_date = start
        self.engine.config.end_date = end
        
        # Run backtest
        results = self.engine.run_walk_forward_backtest(forecaster, data_dict, tickers)
        
        # Restore original dates
        self.engine.config.start_date = original_start
        self.engine.config.end_date = original_end
        
        print(f"\n=== Scenario: {scenario_name} ===")
        print(f"Period: {start} to {end}")
        print(f"Sharpe: {results['overall']['sharpe_ratio']:.2f}")
        print(f"Max DD: {results['overall']['max_drawdown']:.2%}")
        
        return results
