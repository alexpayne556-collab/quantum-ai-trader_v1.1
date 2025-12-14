"""
Real-Time Data Collector & Continuous Learning System
=====================================================
Collects real price data twice daily, tracks predictions vs outcomes,
and retrains models when enough data is collected.

Schedule:
- 10:30 AM ET: Morning predictions (after market open settles)
- 4:30 PM ET: Evening evaluation + new predictions for next day

Data Collection:
- Stores actual prices at prediction time
- Evaluates at multiple horizons: 4hr, 1day, 3day, 5day
- Builds training dataset from prediction outcomes
"""

import os
import sys
import json
import sqlite3
import pickle
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging
import schedule
import time
import warnings

warnings.filterwarnings('ignore')

# Add project paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / 'core'))

from core.colab_predictor import ColabPredictor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('realtime_collector.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class RealTimeDataCollector:
    """
    Collects real market data twice daily and builds training dataset
    from prediction outcomes.
    """
    
    def __init__(self, db_path: str = 'realtime_predictions.db'):
        self.db_path = db_path
        self.predictor = ColabPredictor()
        
        # Watchlist - diverse set of liquid stocks
        self.watchlist = [
            # Mega caps
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA',
            # Tech
            'AMD', 'INTC', 'CRM', 'ORCL', 'ADBE', 'NFLX', 'PYPL',
            # ETFs
            'SPY', 'QQQ', 'IWM', 'DIA',
            # Financials
            'JPM', 'BAC', 'GS', 'V', 'MA',
            # Healthcare
            'JNJ', 'UNH', 'PFE', 'ABBV',
            # Energy
            'XOM', 'CVX',
            # Consumer
            'WMT', 'HD', 'MCD', 'KO', 'PEP'
        ]
        
        # Learning configuration
        self.config = {
            'xgb_weight': 0.55,
            'lgb_weight': 0.45,
            'buy_threshold': 0.60,
            'sell_threshold': 0.60,
            'min_predictions_to_retrain': 500,  # Retrain after 500 predictions evaluated
            'learning_rate': 0.05,
            'evaluation_horizons': [1, 3, 5],  # Days to evaluate
        }
        
        self._init_database()
        self._load_config()
        
    def _init_database(self):
        """Initialize comprehensive database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Main predictions table with multiple evaluation horizons
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                session TEXT NOT NULL,  -- 'morning' or 'evening'
                ticker TEXT NOT NULL,
                
                -- Prediction data
                signal TEXT NOT NULL,
                confidence REAL NOT NULL,
                hold_prob REAL,
                buy_prob REAL,
                sell_prob REAL,
                xgb_signal TEXT,
                lgb_signal TEXT,
                xgb_confidence REAL,
                lgb_confidence REAL,
                
                -- Price at prediction
                price_at_prediction REAL,
                spy_price REAL,
                vix_level REAL,
                
                -- Feature values (for analysis)
                rsi_14 REAL,
                macd_hist REAL,
                bb_position REAL,
                atr REAL,
                
                -- Evaluation flags
                eval_1d INTEGER DEFAULT 0,
                eval_3d INTEGER DEFAULT 0,
                eval_5d INTEGER DEFAULT 0,
                
                -- 1-day evaluation
                price_1d REAL,
                return_1d REAL,
                outcome_1d TEXT,
                correct_1d INTEGER,
                
                -- 3-day evaluation
                price_3d REAL,
                return_3d REAL,
                outcome_3d TEXT,
                correct_3d INTEGER,
                
                -- 5-day evaluation
                price_5d REAL,
                return_5d REAL,
                outcome_5d TEXT,
                correct_5d INTEGER,
                
                -- Metadata
                model_version TEXT,
                config_snapshot TEXT
            )
        ''')
        
        # Training data table (successful predictions for retraining)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS training_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_id INTEGER,
                timestamp TEXT,
                ticker TEXT,
                features TEXT,  -- JSON of feature values
                label INTEGER,  -- 0=HOLD, 1=BUY, 2=SELL (actual outcome)
                return_achieved REAL,
                FOREIGN KEY (prediction_id) REFERENCES predictions (id)
            )
        ''')
        
        # Performance metrics over time
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS daily_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT UNIQUE,
                total_predictions INTEGER,
                correct_1d INTEGER,
                accuracy_1d REAL,
                correct_3d INTEGER,
                accuracy_3d REAL,
                buy_accuracy REAL,
                sell_accuracy REAL,
                avg_return_on_buys REAL,
                avg_return_on_sells REAL,
                xgb_accuracy REAL,
                lgb_accuracy REAL,
                config_snapshot TEXT
            )
        ''')
        
        # Config history
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS config_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                key TEXT,
                old_value REAL,
                new_value REAL,
                reason TEXT
            )
        ''')
        
        # Retrain events
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS retrain_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                training_samples INTEGER,
                accuracy_before REAL,
                accuracy_after REAL,
                notes TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
        
    def _load_config(self):
        """Load config from database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS config (key TEXT PRIMARY KEY, value TEXT)
        ''')
        
        cursor.execute('SELECT key, value FROM config')
        for key, value in cursor.fetchall():
            if key in self.config:
                try:
                    self.config[key] = float(value) if '.' in value else int(value)
                except:
                    self.config[key] = value
        conn.close()
        
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
        
    def collect_predictions(self, session: str = 'morning') -> Dict:
        """
        Collect predictions for all watchlist tickers.
        Call twice daily: morning (10:30 AM) and evening (4:30 PM).
        """
        logger.info(f"="*60)
        logger.info(f"📊 Starting {session.upper()} data collection")
        logger.info(f"="*60)
        
        timestamp = datetime.now().isoformat()
        
        # Download market data once
        logger.info("Downloading market data...")
        spy = yf.download('SPY', period='6mo', progress=False)
        vix = yf.download('^VIX', period='6mo', progress=False)
        
        # Flatten columns
        if hasattr(spy.columns, 'levels'):
            spy.columns = [c[0] if isinstance(c, tuple) else c for c in spy.columns]
        if hasattr(vix.columns, 'levels'):
            vix.columns = [c[0] if isinstance(c, tuple) else c for c in vix.columns]
            
        spy_price = float(spy['Close'].iloc[-1])
        vix_level = float(vix['Close'].iloc[-1])
        
        results = {
            'session': session,
            'timestamp': timestamp,
            'spy_price': spy_price,
            'vix_level': vix_level,
            'predictions': [],
            'errors': []
        }
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for ticker in self.watchlist:
            try:
                # Download ticker data
                df = yf.download(ticker, period='6mo', progress=False)
                if len(df) < 100:
                    results['errors'].append(f"{ticker}: Insufficient data")
                    continue
                    
                # Flatten columns
                if hasattr(df.columns, 'levels'):
                    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
                    
                # Get prediction
                pred = self.predictor.predict(df, spy, vix)
                
                # Apply our learned ensemble weights
                xgb_proba = np.array(pred['xgb_proba'])
                lgb_proba = np.array(pred['lgb_proba'])
                
                ensemble = (
                    self.config['xgb_weight'] * xgb_proba +
                    self.config['lgb_weight'] * lgb_proba
                )
                
                pred_class = np.argmax(ensemble)
                confidence = ensemble[pred_class]
                signal = ['HOLD', 'BUY', 'SELL'][pred_class]
                
                # Apply thresholds
                original_signal = signal
                if signal == 'BUY' and confidence < self.config['buy_threshold']:
                    signal = 'HOLD'
                elif signal == 'SELL' and confidence < self.config['sell_threshold']:
                    signal = 'HOLD'
                    
                # Get individual model signals
                xgb_signal = ['HOLD', 'BUY', 'SELL'][np.argmax(xgb_proba)]
                lgb_signal = ['HOLD', 'BUY', 'SELL'][np.argmax(lgb_proba)]
                
                current_price = float(df['Close'].iloc[-1])
                
                # Extract key features for analysis
                features = self.predictor.engineer_features(df, spy, vix)
                rsi_14 = float(features['RSI_14'].iloc[-1]) if 'RSI_14' in features else None
                macd_hist = float(features['MACD_Hist'].iloc[-1]) if 'MACD_Hist' in features else None
                bb_pos = float(features['BB_Position'].iloc[-1]) if 'BB_Position' in features else None
                atr = float(features['ATR'].iloc[-1]) if 'ATR' in features else None
                
                # Store in database
                cursor.execute('''
                    INSERT INTO predictions (
                        timestamp, session, ticker, signal, confidence,
                        hold_prob, buy_prob, sell_prob,
                        xgb_signal, lgb_signal, xgb_confidence, lgb_confidence,
                        price_at_prediction, spy_price, vix_level,
                        rsi_14, macd_hist, bb_position, atr,
                        model_version, config_snapshot
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    timestamp, session, ticker, signal, float(confidence),
                    float(ensemble[0]), float(ensemble[1]), float(ensemble[2]),
                    xgb_signal, lgb_signal, float(max(xgb_proba)), float(max(lgb_proba)),
                    current_price, spy_price, vix_level,
                    rsi_14, macd_hist, bb_pos, atr,
                    'colab_v1', json.dumps(self.config)
                ))
                
                pred_data = {
                    'ticker': ticker,
                    'signal': signal,
                    'confidence': float(confidence),
                    'price': current_price,
                    'xgb': xgb_signal,
                    'lgb': lgb_signal
                }
                results['predictions'].append(pred_data)
                
                logger.info(f"  {ticker}: {signal} ({confidence:.1%}) @ ${current_price:.2f}")
                
            except Exception as e:
                results['errors'].append(f"{ticker}: {str(e)}")
                logger.warning(f"  {ticker}: Error - {e}")
                
        conn.commit()
        conn.close()
        
        logger.info(f"\n✅ Collected {len(results['predictions'])} predictions")
        logger.info(f"❌ {len(results['errors'])} errors")
        
        return results
        
    def evaluate_predictions(self, horizon_days: int = 1) -> Dict:
        """
        Evaluate predictions that are old enough for the given horizon.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Find unevaluated predictions old enough
        eval_col = f'eval_{horizon_days}d'
        cutoff = (datetime.now() - timedelta(days=horizon_days + 1)).isoformat()
        
        cursor.execute(f'''
            SELECT id, ticker, signal, confidence, price_at_prediction, timestamp,
                   xgb_signal, lgb_signal
            FROM predictions
            WHERE {eval_col} = 0 AND timestamp < ?
        ''', (cutoff,))
        
        predictions = cursor.fetchall()
        
        if not predictions:
            conn.close()
            return {'evaluated': 0, 'horizon': horizon_days}
            
        results = {
            'horizon': horizon_days,
            'evaluated': 0,
            'correct': 0,
            'xgb_correct': 0,
            'lgb_correct': 0,
            'by_signal': {'BUY': {'total': 0, 'correct': 0, 'returns': []},
                         'SELL': {'total': 0, 'correct': 0, 'returns': []},
                         'HOLD': {'total': 0, 'correct': 0}}
        }
        
        for pred in predictions:
            pred_id, ticker, signal, conf, price_at_pred, ts, xgb_sig, lgb_sig = pred
            
            try:
                # Get current price
                df = yf.download(ticker, period='10d', progress=False)
                if len(df) < 2:
                    continue
                    
                if hasattr(df.columns, 'levels'):
                    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
                    
                current_price = float(df['Close'].iloc[-1])
                ret = (current_price - price_at_pred) / price_at_pred
                
                # Determine outcome
                if ret > 0.01:
                    outcome = 'UP'
                elif ret < -0.01:
                    outcome = 'DOWN'
                else:
                    outcome = 'FLAT'
                    
                # Check correctness
                correct = int(
                    (signal == 'BUY' and outcome == 'UP') or
                    (signal == 'SELL' and outcome == 'DOWN') or
                    (signal == 'HOLD' and outcome == 'FLAT')
                )
                
                xgb_correct = int(
                    (xgb_sig == 'BUY' and outcome == 'UP') or
                    (xgb_sig == 'SELL' and outcome == 'DOWN')
                )
                lgb_correct = int(
                    (lgb_sig == 'BUY' and outcome == 'UP') or
                    (lgb_sig == 'SELL' and outcome == 'DOWN')
                )
                
                # Update database
                cursor.execute(f'''
                    UPDATE predictions SET
                        {eval_col} = 1,
                        price_{horizon_days}d = ?,
                        return_{horizon_days}d = ?,
                        outcome_{horizon_days}d = ?,
                        correct_{horizon_days}d = ?
                    WHERE id = ?
                ''', (current_price, ret, outcome, correct, pred_id))
                
                # Add to training data if clear outcome
                if abs(ret) > 0.01:
                    label = 1 if outcome == 'UP' else 2  # BUY=1, SELL=2
                    cursor.execute('''
                        INSERT INTO training_data (prediction_id, timestamp, ticker, label, return_achieved)
                        VALUES (?, ?, ?, ?, ?)
                    ''', (pred_id, ts, ticker, label, ret))
                
                results['evaluated'] += 1
                results['correct'] += correct
                results['xgb_correct'] += xgb_correct
                results['lgb_correct'] += lgb_correct
                
                if signal in results['by_signal']:
                    results['by_signal'][signal]['total'] += 1
                    results['by_signal'][signal]['correct'] += correct
                    if signal in ['BUY', 'SELL']:
                        results['by_signal'][signal]['returns'].append(ret)
                        
            except Exception as e:
                logger.debug(f"Error evaluating {ticker}: {e}")
                continue
                
        conn.commit()
        conn.close()
        
        return results
        
    def learn_and_adjust(self) -> Dict:
        """
        Analyze recent performance and adjust model weights/thresholds.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get last 7 days of evaluated predictions
        week_ago = (datetime.now() - timedelta(days=7)).isoformat()
        cursor.execute('''
            SELECT signal, confidence, correct_1d, xgb_signal, lgb_signal,
                   return_1d, outcome_1d
            FROM predictions
            WHERE eval_1d = 1 AND timestamp > ?
        ''', (week_ago,))
        
        rows = cursor.fetchall()
        conn.close()
        
        if len(rows) < 50:
            return {'status': 'insufficient_data', 'count': len(rows)}
            
        adjustments = []
        
        # Calculate accuracies
        total = len(rows)
        correct = sum(1 for r in rows if r[2] == 1)
        accuracy = correct / total
        
        # XGB vs LGB performance
        xgb_correct = sum(1 for r in rows if 
            (r[3] == 'BUY' and r[6] == 'UP') or (r[3] == 'SELL' and r[6] == 'DOWN'))
        lgb_correct = sum(1 for r in rows if 
            (r[4] == 'BUY' and r[6] == 'UP') or (r[4] == 'SELL' and r[6] == 'DOWN'))
            
        xgb_acc = xgb_correct / total
        lgb_acc = lgb_correct / total
        
        # Adjust weights based on performance
        if abs(xgb_acc - lgb_acc) > 0.03:
            old_xgb = self.config['xgb_weight']
            if xgb_acc > lgb_acc:
                shift = min(0.05, (xgb_acc - lgb_acc) * 0.5)
                self.config['xgb_weight'] = min(0.70, self.config['xgb_weight'] + shift)
            else:
                shift = min(0.05, (lgb_acc - xgb_acc) * 0.5)
                self.config['xgb_weight'] = max(0.30, self.config['xgb_weight'] - shift)
            self.config['lgb_weight'] = 1.0 - self.config['xgb_weight']
            
            adjustments.append({
                'param': 'xgb_weight',
                'old': old_xgb,
                'new': self.config['xgb_weight'],
                'reason': f'XGB acc: {xgb_acc:.1%}, LGB acc: {lgb_acc:.1%}'
            })
            
        # Adjust thresholds based on false positive rate
        buys = [r for r in rows if r[0] == 'BUY']
        sells = [r for r in rows if r[0] == 'SELL']
        
        if len(buys) > 10:
            buy_acc = sum(1 for b in buys if b[2] == 1) / len(buys)
            if buy_acc < 0.45:
                old = self.config['buy_threshold']
                self.config['buy_threshold'] = min(0.80, old + 0.03)
                adjustments.append({
                    'param': 'buy_threshold',
                    'old': old,
                    'new': self.config['buy_threshold'],
                    'reason': f'BUY accuracy only {buy_acc:.1%}'
                })
            elif buy_acc > 0.60 and self.config['buy_threshold'] > 0.55:
                old = self.config['buy_threshold']
                self.config['buy_threshold'] = max(0.50, old - 0.02)
                adjustments.append({
                    'param': 'buy_threshold',
                    'old': old,
                    'new': self.config['buy_threshold'],
                    'reason': f'BUY accuracy good at {buy_acc:.1%}'
                })
                
        if len(sells) > 10:
            sell_acc = sum(1 for s in sells if s[2] == 1) / len(sells)
            if sell_acc < 0.45:
                old = self.config['sell_threshold']
                self.config['sell_threshold'] = min(0.80, old + 0.03)
                adjustments.append({
                    'param': 'sell_threshold',
                    'old': old,
                    'new': self.config['sell_threshold'],
                    'reason': f'SELL accuracy only {sell_acc:.1%}'
                })
                
        # Save config
        self._save_config()
        
        # Log adjustments
        if adjustments:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            for adj in adjustments:
                cursor.execute('''
                    INSERT INTO config_history (timestamp, key, old_value, new_value, reason)
                    VALUES (?, ?, ?, ?, ?)
                ''', (datetime.now().isoformat(), adj['param'], adj['old'], adj['new'], adj['reason']))
            conn.commit()
            conn.close()
            
        return {
            'status': 'adjusted' if adjustments else 'no_changes',
            'predictions_analyzed': total,
            'overall_accuracy': accuracy,
            'xgb_accuracy': xgb_acc,
            'lgb_accuracy': lgb_acc,
            'adjustments': adjustments,
            'current_config': self.config.copy()
        }
        
    def check_retrain_needed(self) -> Tuple[bool, int]:
        """Check if we have enough new training data to retrain."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM training_data')
        count = cursor.fetchone()[0]
        
        cursor.execute('SELECT MAX(id) FROM retrain_log')
        last_retrain = cursor.fetchone()[0]
        
        if last_retrain:
            cursor.execute('''
                SELECT COUNT(*) FROM training_data WHERE id > ?
            ''', (last_retrain,))
            new_count = cursor.fetchone()[0]
        else:
            new_count = count
            
        conn.close()
        
        should_retrain = new_count >= self.config['min_predictions_to_retrain']
        return should_retrain, new_count
        
    def run_morning_session(self) -> Dict:
        """Morning session: collect predictions for the day."""
        logger.info("\n" + "="*70)
        logger.info("🌅 MORNING SESSION - 10:30 AM")
        logger.info("="*70)
        
        return self.collect_predictions(session='morning')
        
    def run_evening_session(self) -> Dict:
        """Evening session: evaluate predictions, learn, collect new predictions."""
        logger.info("\n" + "="*70)
        logger.info("🌙 EVENING SESSION - 4:30 PM")
        logger.info("="*70)
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'evaluations': {},
            'learning': {},
            'predictions': {}
        }
        
        # Evaluate at multiple horizons
        for horizon in [1, 3, 5]:
            eval_result = self.evaluate_predictions(horizon_days=horizon)
            results['evaluations'][f'{horizon}d'] = eval_result
            logger.info(f"  {horizon}-day: Evaluated {eval_result['evaluated']}, "
                       f"Correct: {eval_result.get('correct', 0)}")
            
        # Learn from results
        results['learning'] = self.learn_and_adjust()
        if results['learning'].get('adjustments'):
            for adj in results['learning']['adjustments']:
                logger.info(f"  📈 Adjusted {adj['param']}: {adj['old']:.3f} → {adj['new']:.3f}")
                
        # Check if retrain needed
        should_retrain, new_samples = self.check_retrain_needed()
        results['retrain_status'] = {
            'needed': should_retrain,
            'new_samples': new_samples,
            'threshold': self.config['min_predictions_to_retrain']
        }
        
        if should_retrain:
            logger.info(f"  🔄 RETRAIN RECOMMENDED: {new_samples} new samples available")
            
        # Collect evening predictions
        results['predictions'] = self.collect_predictions(session='evening')
        
        return results
        
    def get_performance_report(self) -> Dict:
        """Generate comprehensive performance report."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Overall stats
        cursor.execute('''
            SELECT COUNT(*), SUM(correct_1d), AVG(return_1d)
            FROM predictions WHERE eval_1d = 1
        ''')
        total, correct, avg_ret = cursor.fetchone()
        
        # By signal
        cursor.execute('''
            SELECT signal, COUNT(*), SUM(correct_1d), AVG(return_1d)
            FROM predictions WHERE eval_1d = 1
            GROUP BY signal
        ''')
        by_signal = {row[0]: {'count': row[1], 'correct': row[2], 'avg_return': row[3]} 
                     for row in cursor.fetchall()}
        
        # Trend over time
        cursor.execute('''
            SELECT DATE(timestamp), COUNT(*), SUM(correct_1d)
            FROM predictions WHERE eval_1d = 1
            GROUP BY DATE(timestamp)
            ORDER BY DATE(timestamp) DESC
            LIMIT 14
        ''')
        daily_trend = [{'date': row[0], 'total': row[1], 'correct': row[2]} 
                      for row in cursor.fetchall()]
        
        # Training data stats
        cursor.execute('SELECT COUNT(*), AVG(return_achieved) FROM training_data')
        train_count, train_avg_ret = cursor.fetchone()
        
        conn.close()
        
        return {
            'total_predictions': total or 0,
            'total_correct': correct or 0,
            'accuracy_1d': (correct / total * 100) if total else 0,
            'avg_return': avg_ret or 0,
            'by_signal': by_signal,
            'daily_trend': daily_trend,
            'training_samples': train_count or 0,
            'current_config': self.config.copy()
        }


def run_twice_daily():
    """Run the collector on a twice-daily schedule."""
    collector = RealTimeDataCollector()
    
    # Schedule morning session at 10:30 AM
    schedule.every().day.at("10:30").do(collector.run_morning_session)
    
    # Schedule evening session at 16:30 (4:30 PM)
    schedule.every().day.at("16:30").do(collector.run_evening_session)
    
    logger.info("="*60)
    logger.info("🚀 REAL-TIME DATA COLLECTOR STARTED")
    logger.info("="*60)
    logger.info("Schedule:")
    logger.info("  📅 Morning predictions: 10:30 AM")
    logger.info("  📅 Evening eval + predictions: 4:30 PM")
    logger.info("\nPress Ctrl+C to stop")
    
    # Run immediately for testing
    logger.info("\n🔄 Running initial collection...")
    collector.run_morning_session()
    
    while True:
        schedule.run_pending()
        time.sleep(60)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Real-Time Data Collector')
    parser.add_argument('--continuous', action='store_true', help='Run continuously twice daily')
    parser.add_argument('--morning', action='store_true', help='Run morning session now')
    parser.add_argument('--evening', action='store_true', help='Run evening session now')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate pending predictions')
    parser.add_argument('--report', action='store_true', help='Show performance report')
    parser.add_argument('--learn', action='store_true', help='Run learning/adjustment cycle')
    
    args = parser.parse_args()
    
    collector = RealTimeDataCollector()
    
    if args.continuous:
        run_twice_daily()
    elif args.morning:
        results = collector.run_morning_session()
        print(json.dumps(results, indent=2, default=str))
    elif args.evening:
        results = collector.run_evening_session()
        print(json.dumps(results, indent=2, default=str))
    elif args.evaluate:
        for horizon in [1, 3, 5]:
            results = collector.evaluate_predictions(horizon_days=horizon)
            print(f"\n{horizon}-day evaluation:")
            print(json.dumps(results, indent=2, default=str))
    elif args.learn:
        results = collector.learn_and_adjust()
        print(json.dumps(results, indent=2, default=str))
    elif args.report:
        report = collector.get_performance_report()
        print("\n" + "="*60)
        print("📊 PERFORMANCE REPORT")
        print("="*60)
        print(f"Total Predictions: {report['total_predictions']}")
        print(f"Accuracy (1-day): {report['accuracy_1d']:.1f}%")
        print(f"Avg Return: {report['avg_return']*100:.2f}%")
        print(f"Training Samples: {report['training_samples']}")
        print(f"\nBy Signal:")
        for sig, data in report['by_signal'].items():
            acc = (data['correct']/data['count']*100) if data['count'] else 0
            print(f"  {sig}: {data['count']} predictions, {acc:.1f}% accuracy")
    else:
        # Default: run both sessions
        collector.run_morning_session()
        

if __name__ == '__main__':
    main()
