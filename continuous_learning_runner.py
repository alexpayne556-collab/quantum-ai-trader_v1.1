"""
Continuous Learning Runner
==========================
Runs the paper trading system on a schedule.
Can be run as a background process or via cron/task scheduler.

Usage:
    # Run continuously (checks every 4 hours during market hours)
    python continuous_learning_runner.py --continuous
    
    # Run once
    python continuous_learning_runner.py --once
    
    # Simulate N days of learning on historical data
    python continuous_learning_runner.py --simulate 30
"""

import os
import sys
import json
import time
import schedule
import argparse
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path
import logging

# Add project root
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / 'core'))

from paper_trader import PaperTrader, ContinuousLearningScheduler

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('continuous_learning.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class MarketHoursChecker:
    """Check if market is open (US Eastern Time)."""
    
    @staticmethod
    def is_market_open() -> bool:
        from datetime import timezone
        now = datetime.now(timezone.utc)
        # Convert to ET (UTC-5 or UTC-4 during DST)
        # Simplified: market open 9:30 AM - 4:00 PM ET
        hour_utc = now.hour
        # Rough approximation: 14:30-21:00 UTC
        weekday = now.weekday()
        
        if weekday >= 5:  # Weekend
            return False
        if hour_utc < 14 or hour_utc >= 21:
            return False
        return True
        
    @staticmethod
    def next_market_open() -> datetime:
        """Get next market open time."""
        now = datetime.now()
        if now.weekday() >= 5:  # Weekend
            days_until_monday = 7 - now.weekday()
            next_open = now + timedelta(days=days_until_monday)
        else:
            next_open = now + timedelta(days=1)
        return next_open.replace(hour=9, minute=30, second=0, microsecond=0)


class HistoricalSimulator:
    """
    Simulate continuous learning on historical data.
    This lets you see how the system would have performed
    if it had been learning and adjusting over time.
    """
    
    def __init__(self, tickers: list = None):
        self.tickers = tickers or [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA',
            'META', 'TSLA', 'SPY', 'QQQ', 'AMD'
        ]
        # Use separate DB for simulation
        self.trader = PaperTrader(db_path='simulation.db')
        
    def download_historical_data(self, days: int = 365) -> dict:
        """Download historical data for all tickers."""
        logger.info(f"Downloading {days} days of historical data...")
        
        data = {}
        spy = yf.download('SPY', period=f'{days}d', progress=False)
        vix = yf.download('^VIX', period=f'{days}d', progress=False)
        
        # Flatten columns
        if hasattr(spy.columns, 'levels'):
            spy.columns = [col[0] if isinstance(col, tuple) else col for col in spy.columns]
        if hasattr(vix.columns, 'levels'):
            vix.columns = [col[0] if isinstance(col, tuple) else col for col in vix.columns]
            
        data['SPY'] = spy
        data['VIX'] = vix
        
        for ticker in self.tickers:
            if ticker not in ['SPY', '^VIX']:
                df = yf.download(ticker, period=f'{days}d', progress=False)
                if hasattr(df.columns, 'levels'):
                    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
                data[ticker] = df
                
        return data
        
    def simulate_day(self, day_idx: int, historical_data: dict, 
                     lookback: int = 120) -> dict:
        """
        Simulate one day of trading.
        Uses data up to day_idx to make predictions,
        then evaluates using day_idx+1.
        """
        results = {
            'day': day_idx,
            'predictions': [],
            'correct': 0,
            'total': 0
        }
        
        spy = historical_data['SPY']
        vix = historical_data['VIX']
        
        for ticker in self.tickers:
            if ticker not in historical_data:
                continue
                
            df = historical_data[ticker]
            
            # Need at least lookback days before and 1 day after
            if day_idx < lookback or day_idx >= len(df) - 1:
                continue
                
            # Data up to prediction day (not including)
            df_slice = df.iloc[day_idx-lookback:day_idx].copy()
            spy_slice = spy.iloc[day_idx-lookback:day_idx].copy()
            vix_slice = vix.iloc[day_idx-lookback:day_idx].copy()
            
            try:
                # Get prediction
                pred = self.trader.predictor.predict(df_slice, spy_slice, vix_slice)
                
                # Apply learned weights
                xgb_proba = np.array(pred['xgb_proba'])
                lgb_proba = np.array(pred['lgb_proba'])
                
                ensemble = (
                    self.trader.config['xgb_weight'] * xgb_proba +
                    self.trader.config['lgb_weight'] * lgb_proba
                )
                
                pred_class = np.argmax(ensemble)
                signal = ['HOLD', 'BUY', 'SELL'][pred_class]
                confidence = ensemble[pred_class]
                
                # Apply thresholds
                if signal == 'BUY' and confidence < self.trader.config['buy_threshold']:
                    signal = 'HOLD'
                elif signal == 'SELL' and confidence < self.trader.config['sell_threshold']:
                    signal = 'HOLD'
                    
                # Get actual outcome (next day return)
                price_today = float(df.iloc[day_idx]['Close'])
                price_tomorrow = float(df.iloc[day_idx + 1]['Close'])
                actual_return = (price_tomorrow - price_today) / price_today
                
                if actual_return > 0.005:
                    actual = 'UP'
                elif actual_return < -0.005:
                    actual = 'DOWN'
                else:
                    actual = 'FLAT'
                    
                # Check if correct
                correct = int(
                    (signal == 'BUY' and actual == 'UP') or
                    (signal == 'SELL' and actual == 'DOWN') or
                    (signal == 'HOLD' and actual == 'FLAT')
                )
                
                results['predictions'].append({
                    'ticker': ticker,
                    'signal': signal,
                    'confidence': float(confidence),
                    'actual': actual,
                    'return': actual_return,
                    'correct': correct
                })
                results['total'] += 1
                results['correct'] += correct
                
            except Exception as e:
                logger.debug(f"Error simulating {ticker} day {day_idx}: {e}")
                continue
                
        return results
        
    def run_simulation(self, days: int = 30, learning_frequency: int = 5) -> dict:
        """
        Run full simulation over N days.
        
        Args:
            days: Number of days to simulate
            learning_frequency: Learn and adjust every N days
        """
        logger.info(f"Starting {days}-day simulation with learning every {learning_frequency} days")
        
        # Download data
        historical_data = self.download_historical_data(days + 200)  # Extra for lookback
        
        # Get common date range
        dates = historical_data['SPY'].index
        start_idx = 150  # After lookback period
        
        all_results = []
        cumulative_correct = 0
        cumulative_total = 0
        
        # Track XGB vs LGB performance for learning
        xgb_scores = []
        lgb_scores = []
        
        for i, day_offset in enumerate(range(start_idx, min(start_idx + days, len(dates) - 1))):
            day_results = self.simulate_day(day_offset, historical_data)
            
            cumulative_correct += day_results['correct']
            cumulative_total += day_results['total']
            
            day_results['cumulative_accuracy'] = (
                cumulative_correct / cumulative_total * 100 if cumulative_total > 0 else 0
            )
            day_results['date'] = str(dates[day_offset].date())
            day_results['config'] = self.trader.config.copy()
            
            all_results.append(day_results)
            
            # Track individual model performance for learning
            for pred in day_results['predictions']:
                # Would need to track XGB/LGB individual predictions
                pass
                
            # Learn every N days
            if (i + 1) % learning_frequency == 0 and cumulative_total > 10:
                # Simulate learning from recent results
                recent_results = all_results[-learning_frequency:]
                recent_correct = sum(r['correct'] for r in recent_results)
                recent_total = sum(r['total'] for r in recent_results)
                recent_accuracy = recent_correct / recent_total if recent_total > 0 else 0.5
                
                # Adjust thresholds based on performance
                if recent_accuracy < 0.45:
                    # Increase thresholds (be more selective)
                    self.trader.config['buy_threshold'] = min(0.75, self.trader.config['buy_threshold'] + 0.02)
                    self.trader.config['sell_threshold'] = min(0.75, self.trader.config['sell_threshold'] + 0.02)
                    logger.info(f"Day {i+1}: Accuracy {recent_accuracy:.1%} - Increasing thresholds")
                elif recent_accuracy > 0.55:
                    # Decrease thresholds (be less selective)
                    self.trader.config['buy_threshold'] = max(0.50, self.trader.config['buy_threshold'] - 0.01)
                    self.trader.config['sell_threshold'] = max(0.50, self.trader.config['sell_threshold'] - 0.01)
                    logger.info(f"Day {i+1}: Accuracy {recent_accuracy:.1%} - Decreasing thresholds")
                    
            if (i + 1) % 10 == 0:
                logger.info(f"Day {i+1}/{days}: Cumulative accuracy {day_results['cumulative_accuracy']:.1f}%")
                
        # Final summary
        summary = {
            'total_days': len(all_results),
            'total_predictions': cumulative_total,
            'total_correct': cumulative_correct,
            'final_accuracy': cumulative_correct / cumulative_total * 100 if cumulative_total > 0 else 0,
            'final_config': self.trader.config.copy(),
            'daily_results': all_results
        }
        
        return summary


def run_scheduled():
    """Run on a schedule throughout the day."""
    trader = PaperTrader()
    scheduler_obj = ContinuousLearningScheduler(trader)
    
    # Schedule jobs
    # Full daily run at 5 PM ET (after market close)
    schedule.every().day.at("17:00").do(scheduler_obj.run_daily_full)
    
    # Intraday checks every 4 hours during market hours
    schedule.every(4).hours.do(
        lambda: scheduler_obj.run_intraday_check() if MarketHoursChecker.is_market_open() else None
    )
    
    # Weekly analysis on Sunday
    schedule.every().sunday.at("20:00").do(scheduler_obj.run_weekly_analysis)
    
    logger.info("Scheduler started. Press Ctrl+C to stop.")
    logger.info("Scheduled jobs:")
    logger.info("  - Daily full run at 5:00 PM")
    logger.info("  - Intraday checks every 4 hours during market")
    logger.info("  - Weekly analysis on Sundays at 8:00 PM")
    
    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute


def main():
    parser = argparse.ArgumentParser(description='Continuous Learning Runner')
    parser.add_argument('--continuous', action='store_true', help='Run continuously on schedule')
    parser.add_argument('--once', action='store_true', help='Run once and exit')
    parser.add_argument('--simulate', type=int, help='Simulate N days of historical learning')
    parser.add_argument('--stats', action='store_true', help='Show performance stats')
    
    args = parser.parse_args()
    
    if args.simulate:
        logger.info(f"Running {args.simulate}-day simulation...")
        simulator = HistoricalSimulator()
        results = simulator.run_simulation(days=args.simulate)
        
        print("\n" + "="*60)
        print("📊 SIMULATION RESULTS")
        print("="*60)
        print(f"Days simulated: {results['total_days']}")
        print(f"Total predictions: {results['total_predictions']}")
        print(f"Correct: {results['total_correct']}")
        print(f"Final Accuracy: {results['final_accuracy']:.1f}%")
        print(f"\nFinal Config (after learning):")
        for k, v in results['final_config'].items():
            print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
            
        # Save detailed results
        with open('simulation_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nDetailed results saved to simulation_results.json")
        
    elif args.continuous:
        run_scheduled()
        
    elif args.stats:
        trader = PaperTrader()
        stats = trader.get_performance_stats()
        print("\n" + "="*60)
        print("📈 PAPER TRADING STATS")
        print("="*60)
        print(json.dumps(stats, indent=2, default=str))
        
    else:  # Default: run once
        trader = PaperTrader()
        scheduler_obj = ContinuousLearningScheduler(trader)
        results = scheduler_obj.run_daily_full()
        print("\n" + "="*60)
        print("📊 DAILY RUN COMPLETE")
        print("="*60)
        print(f"Evaluated: {results['evaluation']['evaluated']} predictions")
        print(f"Correct: {results['evaluation']['correct']}")
        print(f"New predictions: {len(results['predictions'])}")
        
        if results['learning'].get('adjustments'):
            print("\nLearning adjustments made:")
            for adj in results['learning']['adjustments']:
                print(f"  {adj['metric']}: {adj['old']:.3f} → {adj['new']:.3f}")


if __name__ == '__main__':
    main()
