"""
Production-Grade Logging System for Live Trading
Captures every decision, pattern detection, model prediction, and trade execution
Real money on the line - no data loss, full audit trail, microsecond timestamps

Run: python production_logger.py
"""
import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import sys

# Create logs directory
LOGS_DIR = Path(__file__).parent / 'logs'
LOGS_DIR.mkdir(exist_ok=True)

DB_PATH = LOGS_DIR / 'trading_log.db'

class SignalType(Enum):
    BUY = 'BUY'
    SELL = 'SELL'
    HOLD = 'HOLD'

class LogLevel(Enum):
    DEBUG = 'DEBUG'
    INFO = 'INFO'
    WARNING = 'WARNING'
    ERROR = 'ERROR'
    CRITICAL = 'CRITICAL'

@dataclass
class TradeSignal:
    """Complete record of a trading signal."""
    timestamp: datetime
    ticker: str
    signal_type: SignalType
    
    # Pattern detection
    patterns_detected: List[Dict]  # All patterns found
    primary_pattern: str  # Main pattern triggering signal
    pattern_confidence: float  # 0-1
    
    # ML model
    model_prediction: str  # BUY/SELL/HOLD
    model_confidence: float  # 0-1
    model_version: str  # e.g., 'v1.2.3'
    
    # Confluence
    confluence_score: float  # Combined confidence from all sources
    confluence_factors: List[str]  # ['EMA_BULLISH', 'VWAP_SUPPORT', etc.]
    
    # Price & position
    current_price: float
    entry_price: Optional[float]  # If executed
    stop_loss: Optional[float]
    take_profit: Optional[float]
    position_size: Optional[int]  # Shares
    position_value: Optional[float]  # USD
    
    # Market conditions
    volatility_atr: float
    trend_adx: float
    rsi_9: float
    rsi_14: float
    macd_signal: str  # 'BULLISH', 'BEARISH', 'NEUTRAL'
    volume_ratio: float  # Current volume / 20-day avg
    
    # Execution
    executed: bool
    execution_timestamp: Optional[datetime]
    execution_price: Optional[float]
    slippage: Optional[float]  # Execution price - expected price
    
    # Outcome (filled after trade closes)
    exit_timestamp: Optional[datetime] = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    pnl_percent: Optional[float] = None
    hold_duration_minutes: Optional[int] = None
    outcome: Optional[str] = None  # 'WIN', 'LOSS', 'BREAKEVEN'
    
    def to_dict(self) -> Dict:
        """Convert to dict for JSON serialization."""
        d = asdict(self)
        # Convert enums and datetime
        d['timestamp'] = self.timestamp.isoformat()
        d['signal_type'] = self.signal_type.value
        if self.execution_timestamp:
            d['execution_timestamp'] = self.execution_timestamp.isoformat()
        if self.exit_timestamp:
            d['exit_timestamp'] = self.exit_timestamp.isoformat()
        return d


class ProductionLogger:
    """
    Dual logging: JSON files + SQLite database
    JSON: Human-readable, easy to analyze with scripts
    SQLite: Fast queries, analytics, dashboards
    """
    
    def __init__(self):
        self._setup_file_logging()
        self._setup_database()
        self.logger = logging.getLogger('TradingSystem')
    
    def _setup_file_logging(self):
        """Configure rotating JSON file logs."""
        # Create formatter for structured JSON
        class JSONFormatter(logging.Formatter):
            def format(self, record):
                log_obj = {
                    'timestamp': datetime.utcnow().isoformat(),
                    'level': record.levelname,
                    'logger': record.name,
                    'message': record.getMessage(),
                    'module': record.module,
                    'function': record.funcName,
                    'line': record.lineno
                }
                if hasattr(record, 'extra_data'):
                    log_obj.update(record.extra_data)
                return json.dumps(log_obj)
        
        # Daily rotating file handler
        from logging.handlers import TimedRotatingFileHandler
        
        file_handler = TimedRotatingFileHandler(
            filename=LOGS_DIR / 'trading_system.log',
            when='midnight',
            interval=1,
            backupCount=90,  # Keep 90 days
            encoding='utf-8'
        )
        file_handler.setFormatter(JSONFormatter())
        
        # Console handler (plain text for readability)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(
            logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
        )
        
        # Configure root logger
        logger = logging.getLogger('TradingSystem')
        logger.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    def _setup_database(self):
        """Create SQLite tables for trade logs and analytics."""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Trade signals table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trade_signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                ticker TEXT NOT NULL,
                signal_type TEXT NOT NULL,
                
                -- Pattern detection
                patterns_detected TEXT,  -- JSON array
                primary_pattern TEXT,
                pattern_confidence REAL,
                
                -- ML model
                model_prediction TEXT,
                model_confidence REAL,
                model_version TEXT,
                
                -- Confluence
                confluence_score REAL,
                confluence_factors TEXT,  -- JSON array
                
                -- Price & position
                current_price REAL,
                entry_price REAL,
                stop_loss REAL,
                take_profit REAL,
                position_size INTEGER,
                position_value REAL,
                
                -- Market conditions
                volatility_atr REAL,
                trend_adx REAL,
                rsi_9 REAL,
                rsi_14 REAL,
                macd_signal TEXT,
                volume_ratio REAL,
                
                -- Execution
                executed INTEGER,
                execution_timestamp TEXT,
                execution_price REAL,
                slippage REAL,
                
                -- Outcome
                exit_timestamp TEXT,
                exit_price REAL,
                pnl REAL,
                pnl_percent REAL,
                hold_duration_minutes INTEGER,
                outcome TEXT
            )
        ''')
        
        # Pattern performance table (for auto-weighting)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS pattern_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_name TEXT NOT NULL,
                ticker TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                success INTEGER,  -- 1 = win, 0 = loss
                pnl REAL,
                confidence REAL
            )
        ''')
        
        # Model performance table (for continuous learning)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_version TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                ticker TEXT NOT NULL,
                predicted_class TEXT,
                actual_class TEXT,
                confidence REAL,
                correct INTEGER,  -- 1 if prediction correct, 0 if wrong
                brier_score REAL
            )
        ''')
        
        # System health table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_health (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                cpu_percent REAL,
                memory_percent REAL,
                disk_usage_gb REAL,
                api_response_time_ms REAL,
                pattern_detection_time_ms REAL,
                model_inference_time_ms REAL,
                error_count INTEGER
            )
        ''')
        
        # Create indices for fast queries
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_ticker_timestamp ON trade_signals(ticker, timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_outcome ON trade_signals(outcome)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_pattern_perf ON pattern_performance(pattern_name, timestamp)')
        
        conn.commit()
        conn.close()
    
    def log_signal(self, signal: TradeSignal):
        """Log a trade signal to both JSON and database."""
        # JSON file log
        self.logger.info(
            f"SIGNAL: {signal.ticker} {signal.signal_type.value} @ ${signal.current_price:.2f} "
            f"(conf: {signal.confluence_score:.2%})",
            extra={'extra_data': signal.to_dict()}
        )
        
        # Database insert
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO trade_signals (
                timestamp, ticker, signal_type,
                patterns_detected, primary_pattern, pattern_confidence,
                model_prediction, model_confidence, model_version,
                confluence_score, confluence_factors,
                current_price, entry_price, stop_loss, take_profit,
                position_size, position_value,
                volatility_atr, trend_adx, rsi_9, rsi_14, macd_signal, volume_ratio,
                executed, execution_timestamp, execution_price, slippage,
                exit_timestamp, exit_price, pnl, pnl_percent, hold_duration_minutes, outcome
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            signal.timestamp.isoformat(),
            signal.ticker,
            signal.signal_type.value,
            json.dumps(signal.patterns_detected),
            signal.primary_pattern,
            signal.pattern_confidence,
            signal.model_prediction,
            signal.model_confidence,
            signal.model_version,
            signal.confluence_score,
            json.dumps(signal.confluence_factors),
            signal.current_price,
            signal.entry_price,
            signal.stop_loss,
            signal.take_profit,
            signal.position_size,
            signal.position_value,
            signal.volatility_atr,
            signal.trend_adx,
            signal.rsi_9,
            signal.rsi_14,
            signal.macd_signal,
            signal.volume_ratio,
            1 if signal.executed else 0,
            signal.execution_timestamp.isoformat() if signal.execution_timestamp else None,
            signal.execution_price,
            signal.slippage,
            signal.exit_timestamp.isoformat() if signal.exit_timestamp else None,
            signal.exit_price,
            signal.pnl,
            signal.pnl_percent,
            signal.hold_duration_minutes,
            signal.outcome
        ))
        
        conn.commit()
        conn.close()
    
    def update_signal_outcome(self, signal_id: int, exit_price: float, exit_timestamp: datetime):
        """Update trade outcome when position closes."""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Get original signal
        cursor.execute('SELECT entry_price, signal_type FROM trade_signals WHERE id = ?', (signal_id,))
        entry_price, signal_type = cursor.fetchone()
        
        # Calculate P&L
        if signal_type == 'BUY':
            pnl_percent = (exit_price - entry_price) / entry_price
        else:  # SELL
            pnl_percent = (entry_price - exit_price) / entry_price
        
        outcome = 'WIN' if pnl_percent > 0.001 else ('LOSS' if pnl_percent < -0.001 else 'BREAKEVEN')
        
        # Calculate hold duration
        cursor.execute('SELECT execution_timestamp FROM trade_signals WHERE id = ?', (signal_id,))
        exec_time_str = cursor.fetchone()[0]
        exec_time = datetime.fromisoformat(exec_time_str)
        hold_minutes = int((exit_timestamp - exec_time).total_seconds() / 60)
        
        # Update database
        cursor.execute('''
            UPDATE trade_signals
            SET exit_timestamp = ?,
                exit_price = ?,
                pnl_percent = ?,
                outcome = ?,
                hold_duration_minutes = ?
            WHERE id = ?
        ''', (exit_timestamp.isoformat(), exit_price, pnl_percent, outcome, hold_minutes, signal_id))
        
        conn.commit()
        conn.close()
        
        self.logger.info(
            f"TRADE CLOSED: ID={signal_id} Outcome={outcome} P&L={pnl_percent:.2%} Duration={hold_minutes}min"
        )
    
    def log_pattern_performance(self, pattern_name: str, ticker: str, success: bool, pnl: float, confidence: float):
        """Track individual pattern success rate."""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO pattern_performance (pattern_name, ticker, timestamp, success, pnl, confidence)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (pattern_name, ticker, datetime.utcnow().isoformat(), 1 if success else 0, pnl, confidence))
        
        conn.commit()
        conn.close()
    
    def get_pattern_win_rate(self, pattern_name: str, lookback_days: int = 30) -> float:
        """Calculate win rate for a specific pattern over lookback period."""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cutoff = (datetime.utcnow() - timedelta(days=lookback_days)).isoformat()
        
        cursor.execute('''
            SELECT AVG(success) FROM pattern_performance
            WHERE pattern_name = ? AND timestamp > ?
        ''', (pattern_name, cutoff))
        
        result = cursor.fetchone()[0]
        conn.close()
        
        return result if result is not None else 0.5  # Default to 50% if no data
    
    def get_model_accuracy(self, model_version: str, lookback_days: int = 30) -> Dict:
        """Calculate model performance metrics."""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cutoff = (datetime.utcnow() - timedelta(days=lookback_days)).isoformat()
        
        cursor.execute('''
            SELECT 
                AVG(correct) as accuracy,
                AVG(brier_score) as avg_brier,
                COUNT(*) as total_predictions
            FROM model_performance
            WHERE model_version = ? AND timestamp > ?
        ''', (model_version, cutoff))
        
        row = cursor.fetchone()
        conn.close()
        
        return {
            'accuracy': row[0] if row[0] else 0.0,
            'avg_brier_score': row[1] if row[1] else 1.0,
            'total_predictions': row[2] if row[2] else 0
        }


# Example usage
if __name__ == '__main__':
    from datetime import timedelta
    
    logger = ProductionLogger()
    
    # Example: Log a signal
    signal = TradeSignal(
        timestamp=datetime.utcnow(),
        ticker='MU',
        signal_type=SignalType.BUY,
        patterns_detected=[
            {'pattern': 'Hammer', 'confidence': 0.85},
            {'pattern': 'EMA Ribbon Bullish', 'confidence': 0.72}
        ],
        primary_pattern='Hammer',
        pattern_confidence=0.85,
        model_prediction='BUY',
        model_confidence=0.78,
        model_version='v1.0.0',
        confluence_score=0.81,
        confluence_factors=['HAMMER', 'EMA_BULLISH', 'VOLUME_CONFIRM'],
        current_price=236.50,
        entry_price=None,
        stop_loss=232.00,
        take_profit=245.00,
        position_size=None,
        position_value=None,
        volatility_atr=4.2,
        trend_adx=28.5,
        rsi_9=58.3,
        rsi_14=57.1,
        macd_signal='BULLISH',
        volume_ratio=1.4,
        executed=False,
        execution_timestamp=None,
        execution_price=None,
        slippage=None
    )
    
    logger.log_signal(signal)
    
    print("\n✅ Production logging system initialized!")
    print(f"📂 Logs directory: {LOGS_DIR}")
    print(f"💾 Database: {DB_PATH}")
    print("\nNext: Run this 24/7 alongside trading system to capture all decisions")
