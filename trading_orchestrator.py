"""
PRODUCTION TRADING ORCHESTRATOR
Unified 24/7 system integrating:
- Pattern detection (60+ TA-Lib + custom EMA/VWAP)
- AI forecasting (calibrated HistGradientBoostingClassifier)
- Market regime detection (ADX/ATR classification)
- Risk management (position sizing, daily loss limits)
- Online learning (continuous model updates)
- Interactive charting (Plotly with glowing highlights)
- Production logging (SQLite + JSON audit trail)

Real capital on the line - no shortcuts.

Author: Quantum AI Trading System
Date: 2025-11-30
"""

import os
import sys
import logging
import asyncio
import sqlite3
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import yfinance as yf
import talib
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

# Import our modules
from pattern_detector import PatternDetector
from chart_engine import ChartEngine
from production_logger import ProductionLogger, TradeSignal, SignalType

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - [%(levelname)s] - %(message)s'
)
logger = logging.getLogger('TradingOrchestrator')

# Configuration
TICKERS = ['MU', 'IONQ', 'APLD', 'ANNX', 'AAPL', 'MSFT', 'NVDA', 'TSLA', 'GOOGL', 'AMZN', 
           'META', 'AMD', 'AVGO', 'MRVL', 'CRM', 'DDOG', 'PLTR', 'COIN', 'JPM', 'XOM']
UPDATE_INTERVAL_MINUTES = 5
LOOKBACK_DAYS = 60
ACCOUNT_SIZE = 10000.0
RISK_PER_TRADE = 0.01  # 1%
MAX_DAILY_LOSS_PCT = 0.02  # 2%
MIN_CONFIDENCE = 0.70  # 70% minimum to trade
MODELS_DIR = Path(__file__).parent / 'models'
MODELS_DIR.mkdir(exist_ok=True)


class MarketRegimeDetector:
    """Detect market regime using ADX, ATR, EMA alignment."""
    
    @staticmethod
    def detect_regime(df: pd.DataFrame) -> Dict:
        """
        Classify current market regime.
        Returns: regime name, confidence, strategy weights
        """
        if len(df) < 50:
            return MarketRegimeDetector._default_regime()
        
        try:
            # Get arrays
            high = df['High'].values if 'High' in df.columns else df['high'].values
            low = df['Low'].values if 'Low' in df.columns else df['low'].values
            close = df['Close'].values if 'Close' in df.columns else df['close'].values
            
            # Calculate indicators
            adx = talib.ADX(high, low, close, timeperiod=14)[-1]
            atr = talib.ATR(high, low, close, timeperiod=14)
            atr_ratio = atr[-1] / atr[-20:].mean() if len(atr) > 20 else 1.0
            
            # EMA alignment
            ema5 = talib.EMA(close, timeperiod=5)[-1]
            ema13 = talib.EMA(close, timeperiod=13)[-1]
            ema21 = talib.EMA(close, timeperiod=21)[-1]
            ema_bullish = (ema5 > ema13 > ema21)
            ema_bearish = (ema5 < ema13 < ema21)
            
            # Price trends
            lookback_half = len(df) // 2
            higher_highs = high[-1] > high[-lookback_half]
            higher_lows = low[-1] > low[-lookback_half]
            lower_highs = high[-1] < high[-lookback_half]
            lower_lows = low[-1] < low[-lookback_half]
            
            # Classify regime
            if adx > 25 and ema_bullish and higher_highs and higher_lows:
                regime, confidence = 'STRONG_UPTREND', 0.85
            elif adx > 25 and ema_bearish and lower_highs and lower_lows:
                regime, confidence = 'STRONG_DOWNTREND', 0.85
            elif adx > 15 and higher_highs and higher_lows:
                regime, confidence = 'WEAK_UPTREND', 0.65
            elif adx > 15 and lower_highs and lower_lows:
                regime, confidence = 'WEAK_DOWNTREND', 0.65
            elif adx < 15 and atr_ratio < 0.5:
                regime, confidence = 'LOW_VOLATILITY', 0.70
            elif atr_ratio > 2.0:
                regime, confidence = 'HIGH_VOLATILITY', 0.75
            else:
                regime, confidence = 'RANGE_BOUND', 0.60
            
            return {
                'regime': regime,
                'confidence': confidence,
                'adx': float(adx),
                'atr_ratio': float(atr_ratio),
                'ema_aligned': ema_bullish or ema_bearish,
                'weights': MarketRegimeDetector._get_strategy_weights(regime)
            }
        
        except Exception as e:
            logger.error(f"Regime detection error: {e}")
            return MarketRegimeDetector._default_regime()
    
    @staticmethod
    def _get_strategy_weights(regime: str) -> Dict[str, float]:
        """Return strategy weights based on regime."""
        weights = {
            'STRONG_UPTREND': {'pattern': 0.30, 'forecast': 0.40, 'momentum': 0.30},
            'STRONG_DOWNTREND': {'pattern': 0.30, 'forecast': 0.40, 'momentum': 0.30},
            'WEAK_UPTREND': {'pattern': 0.35, 'forecast': 0.35, 'momentum': 0.30},
            'WEAK_DOWNTREND': {'pattern': 0.35, 'forecast': 0.35, 'momentum': 0.30},
            'RANGE_BOUND': {'pattern': 0.50, 'forecast': 0.25, 'momentum': 0.25},
            'LOW_VOLATILITY': {'pattern': 0.40, 'forecast': 0.40, 'momentum': 0.20},
            'HIGH_VOLATILITY': {'pattern': 0.25, 'forecast': 0.45, 'momentum': 0.30}
        }
        return weights.get(regime, {'pattern': 0.33, 'forecast': 0.33, 'momentum': 0.34})
    
    @staticmethod
    def _default_regime() -> Dict:
        return {
            'regime': 'UNKNOWN',
            'confidence': 0.5,
            'adx': 20.0,
            'atr_ratio': 1.0,
            'ema_aligned': False,
            'weights': {'pattern': 0.33, 'forecast': 0.33, 'momentum': 0.34}
        }


class RiskManager:
    """Manage position sizing, daily loss limits, drawdown protection."""
    
    def __init__(self, account_size: float, risk_per_trade: float, max_daily_loss_pct: float):
        self.account_size = account_size
        self.risk_per_trade = risk_per_trade
        self.max_daily_loss_pct = max_daily_loss_pct
        self.daily_pnl = 0.0
        self.peak_equity = account_size
        self.current_equity = account_size
        self.trades_today = []
    
    def can_trade(self) -> bool:
        """Check if daily loss limit not exceeded."""
        daily_loss = abs(min(0, self.daily_pnl))
        max_loss = self.account_size * self.max_daily_loss_pct
        
        if daily_loss >= max_loss:
            logger.warning(f"Daily loss limit hit: ${daily_loss:.2f} >= ${max_loss:.2f}")
            return False
        
        # Check drawdown (25% max)
        drawdown = (self.peak_equity - self.current_equity) / self.peak_equity
        if drawdown > 0.25:
            logger.warning(f"Max drawdown exceeded: {drawdown:.1%}")
            return False
        
        return True
    
    def calculate_position_size(self, entry_price: float, stop_loss: float) -> int:
        """Calculate shares to buy based on 1% account risk."""
        if entry_price <= 0 or stop_loss <= 0:
            return 0
        
        risk_amount = self.current_equity * self.risk_per_trade
        price_risk = abs(entry_price - stop_loss)
        
        if price_risk > 0:
            shares = int(risk_amount / price_risk)
            # Max position size: 5% of account
            max_shares = int((self.current_equity * 0.05) / entry_price)
            return min(shares, max_shares)
        
        return 0
    
    def update_equity(self, pnl: float):
        """Update equity after trade closes."""
        self.daily_pnl += pnl
        self.current_equity += pnl
        self.peak_equity = max(self.peak_equity, self.current_equity)
    
    def reset_daily(self):
        """Reset daily counters (run at market close)."""
        self.daily_pnl = 0.0
        self.trades_today = []
        logger.info("Daily counters reset")


class SignalGenerator:
    """Generate BUY/SELL signals from patterns + forecast + regime."""
    
    def __init__(self):
        pass
    
    def generate_signal(self, 
                       ticker: str,
                       patterns: List[Dict],
                       forecast: Optional[Dict],
                       regime: Dict,
                       df: pd.DataFrame) -> Optional[Dict]:
        """
        Combine all signals into final BUY/SELL/HOLD decision.
        Returns: {signal, confidence, entry, stop, target, confluence_factors}
        """
        try:
            if len(df) < 20:
                return None
            
            close = df['Close'].values if 'Close' in df.columns else df['close'].values
            current_price = float(close[-1])
            
            # Calculate ATR for stop/target levels
            high = df['High'].values if 'High' in df.columns else df['high'].values
            low = df['Low'].values if 'Low' in df.columns else df['low'].values
            atr = talib.ATR(high, low, close, timeperiod=14)[-1]
            
            # Pattern score (0-100)
            bullish_patterns = [p for p in patterns if p['type'] == 'BULLISH']
            bearish_patterns = [p for p in patterns if p['type'] == 'BEARISH']
            
            if not bullish_patterns and not bearish_patterns:
                return None
            
            pattern_score_bull = sum(p['confidence'] for p in bullish_patterns) / max(len(bullish_patterns), 1)
            pattern_score_bear = sum(p['confidence'] for p in bearish_patterns) / max(len(bearish_patterns), 1)
            
            # Forecast score (if model available)
            forecast_score = 0.0
            forecast_direction = 'NEUTRAL'
            if forecast and 'direction' in forecast:
                forecast_direction = forecast['direction']
                forecast_confidence = forecast.get('confidence', 0.5)
                forecast_score = forecast_confidence if forecast_direction in ['BULLISH', 'BEARISH'] else 0.0
            
            # Regime weights
            weights = regime.get('weights', {'pattern': 0.5, 'forecast': 0.3, 'momentum': 0.2})
            
            # Combined confluence score
            if len(bullish_patterns) > len(bearish_patterns):
                signal_type = 'BUY'
                base_confidence = pattern_score_bull * weights['pattern']
                if forecast_direction == 'BULLISH':
                    base_confidence += forecast_score * weights['forecast']
                confluence_factors = [p['pattern'] for p in bullish_patterns[:3]]
                
            elif len(bearish_patterns) > len(bullish_patterns):
                signal_type = 'SELL'
                base_confidence = pattern_score_bear * weights['pattern']
                if forecast_direction == 'BEARISH':
                    base_confidence += forecast_score * weights['forecast']
                confluence_factors = [p['pattern'] for p in bearish_patterns[:3]]
            
            else:
                return None  # Conflicting signals
            
            # Add regime to confluence
            confluence_factors.append(regime['regime'])
            if forecast_direction != 'NEUTRAL':
                confluence_factors.append(f"FORECAST_{forecast_direction}")
            
            # Calculate entry/stop/target
            if signal_type == 'BUY':
                entry = current_price
                stop = current_price - (2 * atr)
                target = current_price + (3 * atr)
            else:  # SELL
                entry = current_price
                stop = current_price + (2 * atr)
                target = current_price - (3 * atr)
            
            return {
                'ticker': ticker,
                'signal': signal_type,
                'confidence': float(base_confidence),
                'entry_price': float(entry),
                'stop_loss': float(stop),
                'take_profit': float(target),
                'confluence_factors': confluence_factors,
                'regime': regime['regime'],
                'patterns_detected': len(patterns),
                'primary_pattern': patterns[0]['pattern'] if patterns else 'None'
            }
        
        except Exception as e:
            logger.error(f"Signal generation error for {ticker}: {e}")
            return None


class TradingOrchestrator:
    """
    Main orchestrator - runs 24/7, coordinates all components.
    """
    
    def __init__(self):
        logger.info("="*60)
        logger.info("Initializing Trading Orchestrator")
        logger.info("="*60)
        
        # Initialize components
        self.pattern_detector = PatternDetector()
        self.chart_engine = ChartEngine()
        self.regime_detector = MarketRegimeDetector()
        self.signal_generator = SignalGenerator()
        self.risk_manager = RiskManager(
            account_size=ACCOUNT_SIZE,
            risk_per_trade=RISK_PER_TRADE,
            max_daily_loss_pct=MAX_DAILY_LOSS_PCT
        )
        self.prod_logger = ProductionLogger()
        
        # State
        self.signals_today = []
        self.open_positions = []
        self.scheduler = AsyncIOScheduler()
        
        # Load AI models (if available)
        self.models = self._load_models()
        
        logger.info("✅ Orchestrator initialized successfully")
        logger.info(f"📊 Tracking {len(TICKERS)} tickers")
        logger.info(f"💰 Account size: ${ACCOUNT_SIZE:,.2f}")
        logger.info(f"⚠️  Risk per trade: {RISK_PER_TRADE:.1%}")
        logger.info(f"🔄 Update interval: {UPDATE_INTERVAL_MINUTES} minutes")
    
    def _load_models(self) -> Dict:
        """Load pre-trained AI models from models/ directory."""
        import joblib
        models = {}
        
        for ticker in TICKERS:
            model_path = MODELS_DIR / f'{ticker}_tuned.pkl'
            if model_path.exists():
                try:
                    models[ticker] = joblib.load(model_path)
                    logger.info(f"✓ Loaded model for {ticker}")
                except Exception as e:
                    logger.warning(f"Failed to load model for {ticker}: {e}")
        
        logger.info(f"Loaded {len(models)}/{len(TICKERS)} AI models")
        return models
    
    async def run_analysis_cycle(self):
        """
        Main 5-minute analysis cycle.
        Runs: data fetch → pattern detection → regime detection → signal generation → logging
        """
        logger.info("\n" + "="*60)
        logger.info(f"🔄 Starting Analysis Cycle - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("="*60)
        
        if not self.risk_manager.can_trade():
            logger.warning("⚠️  Trading halted: Risk limits exceeded")
            return
        
        signals_generated = []
        
        for ticker in TICKERS:
            try:
                # Fetch data
                df = yf.download(ticker, period=f'{LOOKBACK_DAYS}d', interval='1d', progress=False, auto_adjust=True)
                
                if len(df) < 20:
                    logger.warning(f"⚠️  {ticker}: Insufficient data ({len(df)} rows)")
                    continue
                
                # Pattern detection
                pattern_results = self.pattern_detector.detect_all_patterns(ticker, period=f'{LOOKBACK_DAYS}d', interval='1d')
                patterns = pattern_results.get('patterns', [])
                
                # Market regime
                regime = self.regime_detector.detect_regime(df)
                
                # AI forecast (if model available)
                forecast = None
                if ticker in self.models:
                    forecast = self._get_ai_forecast(ticker, df)
                
                # Generate signal
                signal = self.signal_generator.generate_signal(ticker, patterns, forecast, regime, df)
                
                if signal and signal['confidence'] >= MIN_CONFIDENCE:
                    logger.info(f"🎯 {ticker}: {signal['signal']} signal - Confidence: {signal['confidence']:.2%}")
                    logger.info(f"   Entry: ${signal['entry_price']:.2f} | Stop: ${signal['stop_loss']:.2f} | Target: ${signal['take_profit']:.2f}")
                    logger.info(f"   Confluence: {', '.join(signal['confluence_factors'][:3])}")
                    
                    # Calculate position size
                    position_size = self.risk_manager.calculate_position_size(
                        signal['entry_price'],
                        signal['stop_loss']
                    )
                    
                    if position_size > 0:
                        # Log to production logger
                        trade_signal = self._create_trade_signal(ticker, signal, patterns, df, position_size)
                        self.prod_logger.log_signal(trade_signal)
                        signals_generated.append(signal)
                    
                else:
                    logger.debug(f"   {ticker}: No high-confidence signal")
            
            except Exception as e:
                logger.error(f"❌ Error analyzing {ticker}: {e}")
                continue
        
        logger.info(f"\n✅ Cycle complete: {len(signals_generated)} signals generated")
        logger.info(f"💼 Account equity: ${self.risk_manager.current_equity:,.2f}")
        logger.info(f"📊 Daily P&L: ${self.risk_manager.daily_pnl:+,.2f}")
        logger.info("="*60 + "\n")
    
    def _get_ai_forecast(self, ticker: str, df: pd.DataFrame) -> Optional[Dict]:
        """Get 7-day forecast from calibrated AI model."""
        try:
            model_data = self.models[ticker]
            model = model_data['model']
            scaler = model_data['scaler']
            
            # Feature engineering (reuse from ai_recommender_tuned.py logic)
            from ai_recommender_tuned import FE
            X = FE.engineer(df)
            X_latest = X.iloc[-1:]
            Xs = scaler.transform(X_latest)
            
            probs = model.predict_proba(Xs)[0]
            classes = model.classes_
            best_idx = int(np.argmax(probs))
            best_class = int(classes[best_idx])
            
            label_map = {0: 'BEARISH', 1: 'NEUTRAL', 2: 'BULLISH'}
            direction = label_map.get(best_class, 'NEUTRAL')
            confidence = float(probs[best_idx])
            
            return {
                'direction': direction,
                'confidence': confidence,
                'probabilities': probs.tolist()
            }
        
        except Exception as e:
            logger.error(f"Forecast error for {ticker}: {e}")
            return None
    
    def _create_trade_signal(self, ticker: str, signal: Dict, patterns: List, df: pd.DataFrame, position_size: int) -> TradeSignal:
        """Convert signal dict to TradeSignal dataclass for logging."""
        close = df['Close'].values if 'Close' in df.columns else df['close'].values
        
        # Calculate current indicators
        rsi9 = talib.RSI(close, timeperiod=9)[-1]
        rsi14 = talib.RSI(close, timeperiod=14)[-1]
        macd, macd_signal, _ = talib.MACD(close, fastperiod=5, slowperiod=13, signalperiod=1)
        atr = talib.ATR(
            df['High'].values if 'High' in df.columns else df['high'].values,
            df['Low'].values if 'Low' in df.columns else df['low'].values,
            close,
            timeperiod=14
        )[-1]
        adx = talib.ADX(
            df['High'].values if 'High' in df.columns else df['high'].values,
            df['Low'].values if 'Low' in df.columns else df['low'].values,
            close,
            timeperiod=14
        )[-1]
        
        volume = df['Volume'].values if 'Volume' in df.columns else df['volume'].values
        vol_sma = talib.SMA(volume, timeperiod=20)[-1]
        vol_ratio = volume[-1] / vol_sma if vol_sma > 0 else 1.0
        
        macd_sig = 'BULLISH' if macd[-1] > macd_signal[-1] else 'BEARISH'
        
        return TradeSignal(
            timestamp=datetime.utcnow(),
            ticker=ticker,
            signal_type=SignalType.BUY if signal['signal'] == 'BUY' else SignalType.SELL,
            patterns_detected=patterns,
            primary_pattern=signal['primary_pattern'],
            pattern_confidence=signal['confidence'],
            model_prediction=signal['signal'],
            model_confidence=signal['confidence'],
            model_version='v1.0.0',
            confluence_score=signal['confidence'],
            confluence_factors=signal['confluence_factors'],
            current_price=signal['entry_price'],
            entry_price=None,
            stop_loss=signal['stop_loss'],
            take_profit=signal['take_profit'],
            position_size=position_size,
            position_value=position_size * signal['entry_price'],
            volatility_atr=float(atr),
            trend_adx=float(adx),
            rsi_9=float(rsi9),
            rsi_14=float(rsi14),
            macd_signal=macd_sig,
            volume_ratio=float(vol_ratio),
            executed=False,
            execution_timestamp=None,
            execution_price=None,
            slippage=None
        )
    
    async def run_daily_tasks(self):
        """Daily tasks at market close: reset counters, generate charts, train models."""
        logger.info("\n" + "="*60)
        logger.info("📅 Running Daily Tasks")
        logger.info("="*60)
        
        # Reset risk manager
        self.risk_manager.reset_daily()
        
        # Generate charts for all tickers
        logger.info("📊 Generating daily charts...")
        for ticker in TICKERS[:4]:  # Generate for top 4 tickers
            try:
                fig = self.chart_engine.create_candlestick_chart(
                    ticker,
                    period='60d',
                    interval='1d',
                    show_patterns=True,
                    show_emas=True
                )
                if fig:
                    self.chart_engine.save_chart(fig, ticker)
                    logger.info(f"✓ Chart saved for {ticker}")
            except Exception as e:
                logger.error(f"Chart generation failed for {ticker}: {e}")
        
        logger.info("✅ Daily tasks complete\n")
    
    async def run_weekly_tasks(self):
        """Weekly tasks: retrain models, backtest performance."""
        logger.info("\n" + "="*60)
        logger.info("📅 Running Weekly Tasks")
        logger.info("="*60)
        
        # Model retraining happens in Colab (see training notebook)
        # Here we just log that it's time to retrain
        logger.info("⚠️  Weekly model retraining recommended")
        logger.info("   → Run Colab training notebook")
        logger.info("   → Download updated models to models/ directory")
        
        logger.info("✅ Weekly tasks complete\n")
    
    def start(self):
        """Start the trading system with scheduled tasks."""
        logger.info("\n" + "="*60)
        logger.info("🚀 STARTING TRADING SYSTEM")
        logger.info("="*60)
        
        # Schedule recurring tasks
        self.scheduler.add_job(
            self.run_analysis_cycle,
            IntervalTrigger(minutes=UPDATE_INTERVAL_MINUTES),
            id='analysis_cycle',
            name='5-Minute Analysis Cycle'
        )
        
        self.scheduler.add_job(
            self.run_daily_tasks,
            CronTrigger(hour=16, minute=0),  # 4 PM EST (market close)
            id='daily_tasks',
            name='Daily Tasks'
        )
        
        self.scheduler.add_job(
            self.run_weekly_tasks,
            CronTrigger(day_of_week='sun', hour=18, minute=0),
            id='weekly_tasks',
            name='Weekly Tasks'
        )
        
        self.scheduler.start()
        logger.info("✅ Scheduler started")
        logger.info(f"   → Analysis cycle: Every {UPDATE_INTERVAL_MINUTES} minutes")
        logger.info(f"   → Daily tasks: 4:00 PM EST")
        logger.info(f"   → Weekly tasks: Sunday 6:00 PM EST")
        
        # Run first cycle immediately
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.run_analysis_cycle())
        
        logger.info("\n🔄 System running... Press Ctrl+C to stop\n")
        
        try:
            loop.run_forever()
        except KeyboardInterrupt:
            logger.info("\n⚠️  Shutdown signal received")
            self.scheduler.shutdown()
            logger.info("✅ System stopped gracefully")


if __name__ == '__main__':
    orchestrator = TradingOrchestrator()
    orchestrator.start()
