"""
Paper Trading Simulator with Continuous Learning
================================================
Tracks predictions vs actual outcomes, learns from mistakes,
and adjusts model weights/thresholds daily.

Run modes:
1. Live paper trading (makes predictions, logs them, evaluates next day)
2. Simulation mode (backtests continuous learning on historical data)
"""

import os
import json
import sqlite3
import pickle
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / 'core'))

from core.colab_predictor import ColabPredictor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PaperTrader:
    """
    Paper trading system with continuous learning.
    
    Features:
    - Logs all predictions with timestamps
    - Evaluates predictions against actual outcomes
    - Adjusts confidence thresholds based on performance
    - Reweights ensemble based on which model performs better
    - Tracks cumulative P&L
    """
    
    def __init__(self, db_path: str = 'paper_trading.db', 
                 initial_capital: float = 100000.0):
        self.db_path = db_path
        self.initial_capital = initial_capital
        self.predictor = ColabPredictor()
        
        # Learning parameters (will be adjusted based on performance)
        self.config = {
            'xgb_weight': 0.55,
            'lgb_weight': 0.45,
            'buy_threshold': 0.60,   # Minimum confidence for BUY
            'sell_threshold': 0.60,  # Minimum confidence for SELL
            'position_size': 0.10,   # 10% of portfolio per trade
            'stop_loss': 0.02,       # 2% stop loss
            'take_profit': 0.04,     # 4% take profit
            'learning_rate': 0.05,   # How fast to adjust weights
        }
        
        self._init_database()
        self._load_config()
        
    def _init_database(self):
        """Initialize SQLite database for tracking trades and performance."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Predictions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                ticker TEXT NOT NULL,
                signal TEXT NOT NULL,
                confidence REAL NOT NULL,
                xgb_proba TEXT,
                lgb_proba TEXT,
                price_at_prediction REAL,
                evaluated INTEGER DEFAULT 0,
                actual_outcome TEXT,
                actual_return REAL,
                correct INTEGER
            )
        ''')
        
        # Trades table (paper trades)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_id INTEGER,
                timestamp TEXT NOT NULL,
                ticker TEXT NOT NULL,
                action TEXT NOT NULL,
                shares REAL,
                price REAL,
                value REAL,
                closed INTEGER DEFAULT 0,
                close_timestamp TEXT,
                close_price REAL,
                pnl REAL,
                FOREIGN KEY (prediction_id) REFERENCES predictions (id)
            )
        ''')
        
        # Portfolio state
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS portfolio (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                cash REAL,
                positions TEXT,
                total_value REAL,
                daily_pnl REAL
            )
        ''')
        
        # Learning adjustments log
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS learning_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                metric TEXT,
                old_value REAL,
                new_value REAL,
                reason TEXT
            )
        ''')
        
        # Config table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS config (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
        
    def _load_config(self):
        """Load config from database if exists."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT key, value FROM config')
        rows = cursor.fetchall()
        conn.close()
        
        for key, value in rows:
            if key in self.config:
                self.config[key] = float(value)
                
    def _save_config(self):
        """Save config to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        for key, value in self.config.items():
            cursor.execute(
                'INSERT OR REPLACE INTO config (key, value) VALUES (?, ?)',
                (key, str(value))
            )
        conn.commit()
        conn.close()
        
    def get_prediction(self, ticker: str) -> Dict:
        """Get prediction for a ticker using current model weights."""
        # Download data
        df = yf.download(ticker, period='6mo', progress=False)
        spy = yf.download('SPY', period='6mo', progress=False)
        vix = yf.download('^VIX', period='6mo', progress=False)
        
        if len(df) == 0:
            return {'error': f'No data for {ticker}'}
        
        # Get prediction
        result = self.predictor.predict(df, spy, vix)
        
        # Apply our learned weights (override predictor's defaults)
        xgb_proba = np.array(result['xgb_proba'])
        lgb_proba = np.array(result['lgb_proba'])
        
        ensemble_proba = (
            self.config['xgb_weight'] * xgb_proba + 
            self.config['lgb_weight'] * lgb_proba
        )
        
        pred_class = np.argmax(ensemble_proba)
        confidence = ensemble_proba[pred_class]
        
        signal_map = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}
        signal = signal_map[pred_class]
        
        # Apply confidence thresholds
        if signal == 'BUY' and confidence < self.config['buy_threshold']:
            signal = 'HOLD'
        elif signal == 'SELL' and confidence < self.config['sell_threshold']:
            signal = 'HOLD'
            
        return {
            'ticker': ticker,
            'signal': signal,
            'confidence': float(confidence),
            'probabilities': {
                'HOLD': float(ensemble_proba[0]),
                'BUY': float(ensemble_proba[1]),
                'SELL': float(ensemble_proba[2])
            },
            'xgb_proba': xgb_proba.tolist(),
            'lgb_proba': lgb_proba.tolist(),
            'current_price': float(df['Close'].iloc[-1].iloc[0] if hasattr(df['Close'].iloc[-1], 'iloc') else df['Close'].iloc[-1]),
            'timestamp': datetime.now().isoformat()
        }
        
    def log_prediction(self, prediction: Dict) -> int:
        """Log a prediction to the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO predictions 
            (timestamp, ticker, signal, confidence, xgb_proba, lgb_proba, price_at_prediction)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            prediction['timestamp'],
            prediction['ticker'],
            prediction['signal'],
            prediction['confidence'],
            json.dumps(prediction['xgb_proba']),
            json.dumps(prediction['lgb_proba']),
            prediction['current_price']
        ))
        
        pred_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        logger.info(f"Logged prediction {pred_id}: {prediction['ticker']} - {prediction['signal']} ({prediction['confidence']:.1%})")
        return pred_id
        
    def evaluate_predictions(self, lookback_hours: int = 24) -> Dict:
        """
        Evaluate predictions made in the last N hours against actual outcomes.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get unevaluated predictions
        cutoff = (datetime.now() - timedelta(hours=lookback_hours)).isoformat()
        cursor.execute('''
            SELECT id, timestamp, ticker, signal, confidence, xgb_proba, lgb_proba, price_at_prediction
            FROM predictions
            WHERE evaluated = 0 AND timestamp < ?
        ''', (cutoff,))
        
        predictions = cursor.fetchall()
        
        results = {
            'evaluated': 0,
            'correct': 0,
            'wrong': 0,
            'xgb_correct': 0,
            'lgb_correct': 0,
            'details': []
        }
        
        for pred in predictions:
            pred_id, ts, ticker, signal, conf, xgb_str, lgb_str, price_at_pred = pred
            
            # Get current price
            try:
                df = yf.download(ticker, period='5d', progress=False)
                if len(df) < 2:
                    continue
                    
                # Flatten columns if needed
                if hasattr(df.columns, 'levels'):
                    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
                    
                current_price = float(df['Close'].iloc[-1])
                actual_return = (current_price - price_at_pred) / price_at_pred
                
                # Determine actual outcome
                if actual_return > 0.01:
                    actual_outcome = 'UP'
                elif actual_return < -0.01:
                    actual_outcome = 'DOWN'
                else:
                    actual_outcome = 'FLAT'
                    
                # Was prediction correct?
                correct = 0
                if signal == 'BUY' and actual_outcome == 'UP':
                    correct = 1
                elif signal == 'SELL' and actual_outcome == 'DOWN':
                    correct = 1
                elif signal == 'HOLD' and actual_outcome == 'FLAT':
                    correct = 1
                    
                # Check individual model performance
                xgb_proba = json.loads(xgb_str)
                lgb_proba = json.loads(lgb_str)
                
                xgb_signal = ['HOLD', 'BUY', 'SELL'][np.argmax(xgb_proba)]
                lgb_signal = ['HOLD', 'BUY', 'SELL'][np.argmax(lgb_proba)]
                
                xgb_correct = int(
                    (xgb_signal == 'BUY' and actual_outcome == 'UP') or
                    (xgb_signal == 'SELL' and actual_outcome == 'DOWN')
                )
                lgb_correct = int(
                    (lgb_signal == 'BUY' and actual_outcome == 'UP') or
                    (lgb_signal == 'SELL' and actual_outcome == 'DOWN')
                )
                
                # Update database
                cursor.execute('''
                    UPDATE predictions
                    SET evaluated = 1, actual_outcome = ?, actual_return = ?, correct = ?
                    WHERE id = ?
                ''', (actual_outcome, actual_return, correct, pred_id))
                
                results['evaluated'] += 1
                results['correct'] += correct
                results['wrong'] += (1 - correct)
                results['xgb_correct'] += xgb_correct
                results['lgb_correct'] += lgb_correct
                
                results['details'].append({
                    'ticker': ticker,
                    'signal': signal,
                    'actual': actual_outcome,
                    'return': actual_return,
                    'correct': correct
                })
                
            except Exception as e:
                logger.warning(f"Error evaluating {ticker}: {e}")
                continue
                
        conn.commit()
        conn.close()
        
        return results
        
    def learn_from_results(self) -> Dict:
        """
        Analyze recent performance and adjust model weights/thresholds.
        This is the continuous learning component.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get last 7 days of evaluated predictions
        week_ago = (datetime.now() - timedelta(days=7)).isoformat()
        cursor.execute('''
            SELECT signal, confidence, xgb_proba, lgb_proba, correct, actual_return
            FROM predictions
            WHERE evaluated = 1 AND timestamp > ?
        ''', (week_ago,))
        
        rows = cursor.fetchall()
        conn.close()
        
        if len(rows) < 10:
            logger.info("Not enough data to learn from (need at least 10 predictions)")
            return {'status': 'insufficient_data', 'count': len(rows)}
            
        adjustments = []
        
        # Analyze by signal type
        buys = [(r[1], r[4], r[5]) for r in rows if r[0] == 'BUY']
        sells = [(r[1], r[4], r[5]) for r in rows if r[0] == 'SELL']
        
        # Analyze XGB vs LGB performance
        xgb_correct = 0
        lgb_correct = 0
        total = 0
        
        for row in rows:
            signal, conf, xgb_str, lgb_str, correct, ret = row
            xgb_proba = json.loads(xgb_str)
            lgb_proba = json.loads(lgb_str)
            
            xgb_signal = ['HOLD', 'BUY', 'SELL'][np.argmax(xgb_proba)]
            lgb_signal = ['HOLD', 'BUY', 'SELL'][np.argmax(lgb_proba)]
            
            actual = 'UP' if ret > 0.01 else ('DOWN' if ret < -0.01 else 'FLAT')
            
            if (xgb_signal == 'BUY' and actual == 'UP') or (xgb_signal == 'SELL' and actual == 'DOWN'):
                xgb_correct += 1
            if (lgb_signal == 'BUY' and actual == 'UP') or (lgb_signal == 'SELL' and actual == 'DOWN'):
                lgb_correct += 1
            total += 1
            
        # Adjust weights based on relative performance
        if total > 0:
            xgb_accuracy = xgb_correct / total
            lgb_accuracy = lgb_correct / total
            
            # Only adjust if there's a meaningful difference
            if abs(xgb_accuracy - lgb_accuracy) > 0.05:
                old_xgb = self.config['xgb_weight']
                
                # Shift weight toward better performing model
                if xgb_accuracy > lgb_accuracy:
                    shift = min(0.05, (xgb_accuracy - lgb_accuracy) * self.config['learning_rate'])
                    self.config['xgb_weight'] = min(0.70, self.config['xgb_weight'] + shift)
                else:
                    shift = min(0.05, (lgb_accuracy - xgb_accuracy) * self.config['learning_rate'])
                    self.config['xgb_weight'] = max(0.30, self.config['xgb_weight'] - shift)
                    
                self.config['lgb_weight'] = 1.0 - self.config['xgb_weight']
                
                adjustments.append({
                    'metric': 'xgb_weight',
                    'old': old_xgb,
                    'new': self.config['xgb_weight'],
                    'reason': f'XGB accuracy: {xgb_accuracy:.1%}, LGB accuracy: {lgb_accuracy:.1%}'
                })
                
        # Adjust thresholds based on false positive rate
        if len(buys) > 5:
            buy_accuracy = sum(1 for b in buys if b[1] == 1) / len(buys)
            if buy_accuracy < 0.45:  # Too many false BUYs
                old_thresh = self.config['buy_threshold']
                self.config['buy_threshold'] = min(0.80, self.config['buy_threshold'] + 0.02)
                adjustments.append({
                    'metric': 'buy_threshold',
                    'old': old_thresh,
                    'new': self.config['buy_threshold'],
                    'reason': f'BUY accuracy only {buy_accuracy:.1%}'
                })
            elif buy_accuracy > 0.60 and self.config['buy_threshold'] > 0.55:
                old_thresh = self.config['buy_threshold']
                self.config['buy_threshold'] = max(0.50, self.config['buy_threshold'] - 0.02)
                adjustments.append({
                    'metric': 'buy_threshold',
                    'old': old_thresh,
                    'new': self.config['buy_threshold'],
                    'reason': f'BUY accuracy good at {buy_accuracy:.1%}, lowering threshold'
                })
                
        if len(sells) > 5:
            sell_accuracy = sum(1 for s in sells if s[1] == 1) / len(sells)
            if sell_accuracy < 0.45:  # Too many false SELLs
                old_thresh = self.config['sell_threshold']
                self.config['sell_threshold'] = min(0.80, self.config['sell_threshold'] + 0.02)
                adjustments.append({
                    'metric': 'sell_threshold',
                    'old': old_thresh,
                    'new': self.config['sell_threshold'],
                    'reason': f'SELL accuracy only {sell_accuracy:.1%}'
                })
                
        # Save updated config
        self._save_config()
        
        # Log adjustments
        if adjustments:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            for adj in adjustments:
                cursor.execute('''
                    INSERT INTO learning_log (timestamp, metric, old_value, new_value, reason)
                    VALUES (?, ?, ?, ?, ?)
                ''', (datetime.now().isoformat(), adj['metric'], adj['old'], adj['new'], adj['reason']))
            conn.commit()
            conn.close()
            
        return {
            'status': 'learned',
            'predictions_analyzed': len(rows),
            'xgb_accuracy': xgb_accuracy if total > 0 else None,
            'lgb_accuracy': lgb_accuracy if total > 0 else None,
            'adjustments': adjustments,
            'current_config': self.config.copy()
        }
        
    def run_daily_cycle(self, tickers: List[str]) -> Dict:
        """
        Run a complete daily cycle:
        1. Evaluate yesterday's predictions
        2. Learn from results
        3. Make today's predictions
        """
        logger.info("="*50)
        logger.info("Starting Daily Paper Trading Cycle")
        logger.info("="*50)
        
        # Step 1: Evaluate previous predictions
        logger.info("\n📊 Evaluating previous predictions...")
        eval_results = self.evaluate_predictions(lookback_hours=24)
        logger.info(f"Evaluated: {eval_results['evaluated']}, Correct: {eval_results['correct']}, Wrong: {eval_results['wrong']}")
        
        # Step 2: Learn from results
        logger.info("\n🧠 Learning from results...")
        learn_results = self.learn_from_results()
        if learn_results.get('adjustments'):
            for adj in learn_results['adjustments']:
                logger.info(f"  Adjusted {adj['metric']}: {adj['old']:.3f} → {adj['new']:.3f}")
                
        # Step 3: Make new predictions
        logger.info(f"\n🎯 Making predictions for {len(tickers)} tickers...")
        predictions = []
        for ticker in tickers:
            try:
                pred = self.get_prediction(ticker)
                if 'error' not in pred:
                    self.log_prediction(pred)
                    predictions.append(pred)
                    logger.info(f"  {ticker}: {pred['signal']} ({pred['confidence']:.1%})")
            except Exception as e:
                logger.warning(f"  {ticker}: Error - {e}")
                
        return {
            'evaluation': eval_results,
            'learning': learn_results,
            'predictions': predictions,
            'timestamp': datetime.now().isoformat()
        }
        
    def get_performance_stats(self) -> Dict:
        """Get overall performance statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Overall accuracy
        cursor.execute('''
            SELECT COUNT(*), SUM(correct) FROM predictions WHERE evaluated = 1
        ''')
        total, correct = cursor.fetchone()
        
        # By signal type
        cursor.execute('''
            SELECT signal, COUNT(*), SUM(correct), AVG(actual_return)
            FROM predictions WHERE evaluated = 1
            GROUP BY signal
        ''')
        by_signal = {row[0]: {'count': row[1], 'correct': row[2], 'avg_return': row[3]} 
                     for row in cursor.fetchall()}
        
        # Recent trend (last 7 days vs previous 7 days)
        cursor.execute('''
            SELECT 
                CASE WHEN timestamp > date('now', '-7 days') THEN 'recent' ELSE 'older' END as period,
                COUNT(*), SUM(correct)
            FROM predictions 
            WHERE evaluated = 1 AND timestamp > date('now', '-14 days')
            GROUP BY period
        ''')
        trend = {row[0]: {'count': row[1], 'correct': row[2]} for row in cursor.fetchall()}
        
        conn.close()
        
        return {
            'total_predictions': total or 0,
            'total_correct': correct or 0,
            'accuracy': (correct / total * 100) if total else 0,
            'by_signal': by_signal,
            'trend': trend,
            'current_config': self.config.copy()
        }


class ContinuousLearningScheduler:
    """
    Scheduler for running paper trading at regular intervals.
    Can run multiple times per day for intraday learning.
    """
    
    def __init__(self, paper_trader: PaperTrader):
        self.trader = paper_trader
        self.watchlist = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA',
            'META', 'TSLA', 'SPY', 'QQQ', 'AMD',
            'NFLX', 'CRM', 'ORCL', 'ADBE', 'INTC'
        ]
        
    def run_intraday_check(self) -> Dict:
        """
        Quick intraday check - evaluates recent predictions and logs new ones.
        Good for running every 4 hours during market hours.
        """
        # Only evaluate predictions older than 4 hours
        eval_results = self.trader.evaluate_predictions(lookback_hours=4)
        
        # Make new predictions
        predictions = []
        for ticker in self.watchlist[:5]:  # Quick check on top 5
            try:
                pred = self.trader.get_prediction(ticker)
                if 'error' not in pred:
                    self.trader.log_prediction(pred)
                    predictions.append(pred)
            except:
                continue
                
        return {
            'type': 'intraday',
            'evaluated': eval_results['evaluated'],
            'new_predictions': len(predictions)
        }
        
    def run_daily_full(self) -> Dict:
        """
        Full daily cycle with learning. Run once per day after market close.
        """
        return self.trader.run_daily_cycle(self.watchlist)
        
    def run_weekly_analysis(self) -> Dict:
        """
        Weekly deep analysis of performance.
        """
        stats = self.trader.get_performance_stats()
        learn = self.trader.learn_from_results()
        
        return {
            'type': 'weekly',
            'stats': stats,
            'learning': learn
        }


def main():
    """Main entry point for paper trading."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Paper Trading with Continuous Learning')
    parser.add_argument('--mode', choices=['daily', 'intraday', 'weekly', 'stats', 'predict'],
                       default='daily', help='Run mode')
    parser.add_argument('--tickers', nargs='+', help='Specific tickers to predict')
    
    args = parser.parse_args()
    
    trader = PaperTrader()
    scheduler = ContinuousLearningScheduler(trader)
    
    if args.mode == 'daily':
        results = scheduler.run_daily_full()
        print(json.dumps(results, indent=2, default=str))
        
    elif args.mode == 'intraday':
        results = scheduler.run_intraday_check()
        print(json.dumps(results, indent=2, default=str))
        
    elif args.mode == 'weekly':
        results = scheduler.run_weekly_analysis()
        print(json.dumps(results, indent=2, default=str))
        
    elif args.mode == 'stats':
        stats = trader.get_performance_stats()
        print("\n📈 Paper Trading Performance Stats")
        print("="*50)
        print(f"Total Predictions: {stats['total_predictions']}")
        print(f"Correct: {stats['total_correct']}")
        print(f"Accuracy: {stats['accuracy']:.1f}%")
        print(f"\nCurrent Config:")
        for k, v in stats['current_config'].items():
            print(f"  {k}: {v}")
            
    elif args.mode == 'predict':
        tickers = args.tickers or ['AAPL', 'MSFT', 'GOOGL']
        print(f"\n🎯 Predictions for {tickers}")
        print("="*50)
        for ticker in tickers:
            pred = trader.get_prediction(ticker)
            if 'error' not in pred:
                trader.log_prediction(pred)
                print(f"{ticker}: {pred['signal']} ({pred['confidence']:.1%})")
            else:
                print(f"{ticker}: {pred['error']}")


if __name__ == '__main__':
    main()
