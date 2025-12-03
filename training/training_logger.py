"""
COMPREHENSIVE TRAINING LOGGER
Logs all model performance, pattern statistics, regime transitions for continuous improvement.

Features:
- Model performance tracking (accuracy, Sharpe, calibration)
- Pattern edge monitoring (win rate, IC, edge decay)
- Regime transition logging
- Trade performance attribution
- Self-improvement recommendations
- JSON export for analysis
- SQLite storage for historical tracking

Usage:
    logger = TrainingLogger()
    
    # Log model training
    logger.log_model_training('ai_recommender_v2', metrics={'accuracy': 0.67, 'sharpe': 1.8})
    
    # Log pattern performance
    logger.log_pattern_performance('CDLHAMMER', win_rate=0.58, edge=0.015)
    
    # Generate self-improvement report
    recommendations = logger.generate_improvement_recommendations()
"""

import sqlite3
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModelTrainingLog:
    """Model training session log"""
    timestamp: str
    model_name: str
    version: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    sharpe_ratio: Optional[float]
    brier_score: Optional[float]
    calibration_error: Optional[float]
    train_samples: int
    test_samples: int
    feature_count: int
    training_time_seconds: float
    hyperparameters: Dict
    notes: str


@dataclass
class PatternPerformanceLog:
    """Pattern performance tracking"""
    timestamp: str
    pattern_name: str
    timeframe: str
    regime: str
    sample_count: int
    win_rate: float
    avg_return: float
    sharpe_ratio: float
    rank_ic: float
    edge_per_trade: float
    edge_status: str  # 'STRONG', 'MODERATE', 'WEAK', 'DEAD', 'DECAYING'
    notes: str


@dataclass
class RegimeTransitionLog:
    """Market regime transition tracking"""
    timestamp: str
    ticker: str
    previous_regime: str
    new_regime: str
    duration_days: int
    performance_in_regime: Optional[float]
    pattern_performance: Dict
    notes: str


@dataclass
class TradePerformanceLog:
    """Individual trade performance for attribution"""
    timestamp: str
    ticker: str
    entry_price: float
    exit_price: float
    return_pct: float
    holding_period_days: int
    signal_sources: List[str]  # ['CDLHAMMER', 'MACD_CROSSOVER']
    confidence: float
    regime: str
    outcome: str  # 'WIN', 'LOSS', 'BREAKEVEN'
    notes: str


class TrainingLogger:
    """
    Comprehensive logging system for model and pattern performance.
    
    Enables continuous self-improvement by tracking:
    - Which models/patterns work best in which regimes
    - When pattern edges are decaying
    - Which feature sets are most predictive
    - Trade attribution (what signals led to best/worst trades)
    """
    
    def __init__(self, db_path: str = 'data/training_logs.db'):
        """
        Initialize training logger.
        
        Args:
            db_path: Path to SQLite database for persistent storage
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._init_database()
        logger.info(f"✅ TrainingLogger initialized: {self.db_path}")
    
    def _init_database(self):
        """Create database schema"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Model training logs
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_training_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP NOT NULL,
                model_name TEXT NOT NULL,
                version TEXT,
                accuracy REAL,
                precision_score REAL,
                recall_score REAL,
                f1_score REAL,
                sharpe_ratio REAL,
                brier_score REAL,
                calibration_error REAL,
                train_samples INTEGER,
                test_samples INTEGER,
                feature_count INTEGER,
                training_time_seconds REAL,
                hyperparameters TEXT,
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Pattern performance logs
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS pattern_performance_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP NOT NULL,
                pattern_name TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                regime TEXT NOT NULL,
                sample_count INTEGER,
                win_rate REAL,
                avg_return REAL,
                sharpe_ratio REAL,
                rank_ic REAL,
                edge_per_trade REAL,
                edge_status TEXT,
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Regime transition logs
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS regime_transition_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP NOT NULL,
                ticker TEXT NOT NULL,
                previous_regime TEXT NOT NULL,
                new_regime TEXT NOT NULL,
                duration_days INTEGER,
                performance_in_regime REAL,
                pattern_performance TEXT,
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Trade performance logs
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trade_performance_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP NOT NULL,
                ticker TEXT NOT NULL,
                entry_price REAL,
                exit_price REAL,
                return_pct REAL,
                holding_period_days INTEGER,
                signal_sources TEXT,
                confidence REAL,
                regime TEXT,
                outcome TEXT,
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Indexes
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_model_name ON model_training_logs(model_name)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_pattern_name ON pattern_performance_logs(pattern_name)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_regime ON regime_transition_logs(ticker, new_regime)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_trade_outcome ON trade_performance_logs(outcome)')
        
        conn.commit()
        conn.close()
    
    def log_model_training(
        self,
        model_name: str,
        metrics: Dict,
        hyperparameters: Dict = None,
        version: str = '1.0',
        notes: str = ''
    ):
        """
        Log model training session.
        
        Args:
            model_name: Model identifier (e.g., 'ai_recommender_v2')
            metrics: Dict with accuracy, precision, recall, f1, sharpe, brier, etc.
            hyperparameters: Model hyperparameters used
            version: Model version
            notes: Additional notes
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO model_training_logs (
                timestamp, model_name, version, accuracy, precision_score, recall_score,
                f1_score, sharpe_ratio, brier_score, calibration_error,
                train_samples, test_samples, feature_count, training_time_seconds,
                hyperparameters, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            model_name,
            version,
            metrics.get('accuracy'),
            metrics.get('precision'),
            metrics.get('recall'),
            metrics.get('f1'),
            metrics.get('sharpe_ratio'),
            metrics.get('brier_score'),
            metrics.get('calibration_error'),
            metrics.get('train_samples'),
            metrics.get('test_samples'),
            metrics.get('feature_count'),
            metrics.get('training_time', 0),
            json.dumps(hyperparameters) if hyperparameters else None,
            notes
        ))
        
        conn.commit()
        conn.close()
        
        logger.info(f"📝 Logged training for {model_name} (acc: {metrics.get('accuracy', 0):.3f})")
    
    def log_pattern_performance(
        self,
        pattern_name: str,
        timeframe: str,
        regime: str,
        metrics: Dict,
        notes: str = ''
    ):
        """
        Log pattern performance.
        
        Args:
            pattern_name: Pattern identifier
            timeframe: Timeframe ('1d', '1h', etc.)
            regime: Market regime ('BULL', 'BEAR', 'RANGE')
            metrics: Dict with win_rate, avg_return, sharpe, rank_ic, edge, status
            notes: Additional notes
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO pattern_performance_logs (
                timestamp, pattern_name, timeframe, regime, sample_count,
                win_rate, avg_return, sharpe_ratio, rank_ic, edge_per_trade,
                edge_status, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            pattern_name,
            timeframe,
            regime,
            metrics.get('sample_count'),
            metrics.get('win_rate'),
            metrics.get('avg_return'),
            metrics.get('sharpe_ratio'),
            metrics.get('rank_ic'),
            metrics.get('edge_per_trade'),
            metrics.get('edge_status'),
            notes
        ))
        
        conn.commit()
        conn.close()
        
        logger.info(f"📝 Logged pattern {pattern_name} ({regime}: WR={metrics.get('win_rate', 0):.1%})")
    
    def log_regime_transition(
        self,
        ticker: str,
        previous_regime: str,
        new_regime: str,
        duration_days: int,
        performance_in_regime: Optional[float] = None,
        pattern_performance: Dict = None,
        notes: str = ''
    ):
        """Log market regime transition"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO regime_transition_logs (
                timestamp, ticker, previous_regime, new_regime, duration_days,
                performance_in_regime, pattern_performance, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            ticker,
            previous_regime,
            new_regime,
            duration_days,
            performance_in_regime,
            json.dumps(pattern_performance) if pattern_performance else None,
            notes
        ))
        
        conn.commit()
        conn.close()
        
        logger.info(f"📝 Logged regime transition: {ticker} {previous_regime}→{new_regime}")
    
    def log_trade_performance(
        self,
        ticker: str,
        entry_price: float,
        exit_price: float,
        holding_period_days: int,
        signal_sources: List[str],
        confidence: float,
        regime: str,
        notes: str = ''
    ):
        """Log individual trade performance"""
        return_pct = (exit_price - entry_price) / entry_price
        outcome = 'WIN' if return_pct > 0.01 else ('LOSS' if return_pct < -0.01 else 'BREAKEVEN')
        
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO trade_performance_logs (
                timestamp, ticker, entry_price, exit_price, return_pct,
                holding_period_days, signal_sources, confidence, regime, outcome, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            ticker,
            entry_price,
            exit_price,
            return_pct,
            holding_period_days,
            json.dumps(signal_sources),
            confidence,
            regime,
            outcome,
            notes
        ))
        
        conn.commit()
        conn.close()
        
        logger.info(f"📝 Logged trade: {ticker} {outcome} ({return_pct:+.2%})")
    
    def generate_improvement_recommendations(self) -> Dict:
        """
        Analyze logs and generate self-improvement recommendations.
        
        Returns:
            Dict with recommendations for model retraining, pattern filtering, etc.
        """
        recommendations = {
            'timestamp': datetime.now().isoformat(),
            'model_recommendations': [],
            'pattern_recommendations': [],
            'regime_recommendations': [],
            'feature_recommendations': []
        }
        
        conn = sqlite3.connect(str(self.db_path))
        
        # 1. Model performance degradation
        model_df = pd.read_sql_query('''
            SELECT model_name, timestamp, accuracy, sharpe_ratio
            FROM model_training_logs
            ORDER BY timestamp DESC
            LIMIT 100
        ''', conn)
        
        if len(model_df) > 10:
            for model_name in model_df['model_name'].unique():
                model_data = model_df[model_df['model_name'] == model_name].sort_values('timestamp')
                if len(model_data) >= 3:
                    recent_acc = model_data['accuracy'].iloc[-3:].mean()
                    historical_acc = model_data['accuracy'].iloc[:-3].mean() if len(model_data) > 3 else recent_acc
                    
                    if recent_acc < historical_acc - 0.05:
                        recommendations['model_recommendations'].append({
                            'model': model_name,
                            'action': 'RETRAIN',
                            'reason': f'Accuracy dropped {(historical_acc - recent_acc):.1%}',
                            'priority': 'HIGH'
                        })
        
        # 2. Pattern edge decay
        pattern_df = pd.read_sql_query('''
            SELECT pattern_name, timeframe, regime, timestamp, win_rate, edge_status
            FROM pattern_performance_logs
            ORDER BY timestamp DESC
            LIMIT 500
        ''', conn)
        
        for pattern_name in pattern_df['pattern_name'].unique():
            pattern_data = pattern_df[pattern_df['pattern_name'] == pattern_name]
            if (pattern_data['edge_status'] == 'DECAYING').any():
                recommendations['pattern_recommendations'].append({
                    'pattern': pattern_name,
                    'action': 'REDUCE_WEIGHT',
                    'reason': 'Edge is decaying',
                    'priority': 'MEDIUM'
                })
            elif (pattern_data['edge_status'] == 'DEAD').any():
                recommendations['pattern_recommendations'].append({
                    'pattern': pattern_name,
                    'action': 'DISABLE',
                    'reason': 'Pattern edge has died',
                    'priority': 'HIGH'
                })
        
        # 3. Regime-specific performance
        trade_df = pd.read_sql_query('''
            SELECT regime, outcome, return_pct
            FROM trade_performance_logs
            WHERE timestamp > datetime('now', '-30 days')
        ''', conn)
        
        if len(trade_df) > 20:
            regime_performance = trade_df.groupby('regime').agg({
                'return_pct': 'mean',
                'outcome': lambda x: (x == 'WIN').mean()
            }).to_dict('index')
            
            for regime, perf in regime_performance.items():
                if perf['outcome'] < 0.45:
                    recommendations['regime_recommendations'].append({
                        'regime': regime,
                        'action': 'REDUCE_POSITION_SIZE',
                        'reason': f'Win rate only {perf["outcome"]:.1%} in {regime}',
                        'priority': 'HIGH'
                    })
        
        conn.close()
        
        # Save recommendations
        rec_path = Path('data/improvement_recommendations.json')
        rec_path.parent.mkdir(exist_ok=True)
        with open(rec_path, 'w') as f:
            json.dump(recommendations, f, indent=2)
        
        logger.info(f"💡 Generated {sum(len(v) for v in recommendations.values() if isinstance(v, list))} recommendations")
        
        return recommendations
    
    def get_recent_logs(self, log_type: str = 'model', limit: int = 10) -> pd.DataFrame:
        """
        Retrieve recent logs.
        
        Args:
            log_type: 'model', 'pattern', 'regime', or 'trade'
            limit: Number of records to return
        
        Returns:
            DataFrame with logs
        """
        conn = sqlite3.connect(str(self.db_path))
        
        table_map = {
            'model': 'model_training_logs',
            'pattern': 'pattern_performance_logs',
            'regime': 'regime_transition_logs',
            'trade': 'trade_performance_logs'
        }
        
        table = table_map.get(log_type)
        if not table:
            raise ValueError(f"Invalid log_type: {log_type}")
        
        df = pd.read_sql_query(f'''
            SELECT * FROM {table}
            ORDER BY timestamp DESC
            LIMIT {limit}
        ''', conn)
        
        conn.close()
        return df


if __name__ == '__main__':
    # Example usage
    print("🔧 Testing Training Logger...")
    
    logger_sys = TrainingLogger()
    
    # Log model training
    logger_sys.log_model_training(
        model_name='ai_recommender_v2',
        version='2.1',
        metrics={
            'accuracy': 0.672,
            'precision': 0.68,
            'recall': 0.65,
            'f1': 0.665,
            'sharpe_ratio': 1.85,
            'brier_score': 0.21,
            'train_samples': 5000,
            'test_samples': 1250,
            'feature_count': 52,
            'training_time': 45.2
        },
        hyperparameters={'learning_rate': 0.05, 'max_iter': 200},
        notes='Trained with new institutional features'
    )
    
    # Log pattern performance
    logger_sys.log_pattern_performance(
        pattern_name='CDLHAMMER',
        timeframe='1d',
        regime='BULL',
        metrics={
            'sample_count': 87,
            'win_rate': 0.58,
            'avg_return': 0.015,
            'sharpe_ratio': 1.4,
            'rank_ic': 0.12,
            'edge_per_trade': 0.008,
            'edge_status': 'MODERATE'
        }
    )
    
    # Log regime transition
    logger_sys.log_regime_transition(
        ticker='SPY',
        previous_regime='RANGE',
        new_regime='BULL',
        duration_days=18,
        performance_in_regime=0.023,
        pattern_performance={'CDLHAMMER': 0.62, 'CDLENGULFING': 0.58}
    )
    
    # Log trade
    logger_sys.log_trade_performance(
        ticker='AAPL',
        entry_price=150.25,
        exit_price=153.80,
        holding_period_days=3,
        signal_sources=['CDLHAMMER', 'MACD_BULLISH', 'RSI_OVERSOLD'],
        confidence=0.75,
        regime='BULL'
    )
    
    # Generate recommendations
    print("\n💡 Generating improvement recommendations...")
    recommendations = logger_sys.generate_improvement_recommendations()
    
    print(f"\n📋 Recommendations:")
    for category, recs in recommendations.items():
        if isinstance(recs, list) and recs:
            print(f"\n{category}:")
            for rec in recs:
                print(f"  • {rec.get('action')} - {rec.get('reason')} [{rec.get('priority')}]")
    
    # Show recent logs
    print("\n📊 Recent Model Training Logs:")
    recent_models = logger_sys.get_recent_logs('model', limit=5)
    if len(recent_models) > 0:
        print(recent_models[['model_name', 'version', 'accuracy', 'sharpe_ratio']].to_string(index=False))
    
    print("\n✅ Training Logger Ready for Production!")
