#!/usr/bin/env python3
"""
SIGNAL FACTORY - The Bridge from Research to Operation
=======================================================
Phase 5: Automated daily signal generation system.

This is NOT a trading bot. This is an intelligence augmentation system that:
1. Maps the market terrain (regime detection)
2. Generates systematic signals from validated multi-factor models
3. Analyzes discretionary picks against quantitative factors
4. Outputs actionable trade sheets and journals

Run daily at market close: python3 signal_factory.py --date 2024-12-19

Author: Quantum Trading Research Team
Date: December 20, 2025
"""

import sqlite3
import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime, timedelta
import os
import json
import warnings
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================

DB_PATH = 'data/market_data.db'
MODELS_CSV = 'data/factor_lab/recommended_models.csv'
TWO_FACTOR_CSV = 'data/factor_lab/two_factor_robust.csv'
OUTPUT_DIR = 'signals'

# Portfolio settings
PORTFOLIO_CAPITAL = 100000
SYSTEMATIC_ALLOCATION = 0.70  # 70% to systematic
DISCRETIONARY_ALLOCATION = 0.30  # 30% to discretionary
TARGET_RISK_PER_POSITION = 0.01  # 1% portfolio risk per position

# Filters
MIN_PRICE = 5.0
MIN_AVG_VOLUME = 100000
MAX_POSITIONS = 10

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class StockSignal:
    """Represents a trading signal for a single stock"""
    ticker: str
    strategy: str
    action: str  # BUY, SELL, HOLD
    score: float
    confidence: str  # HIGH, MEDIUM, LOW
    suggested_units: int
    rationale: str
    factors: Dict[str, float]


@dataclass 
class MarketRegime:
    """Represents current market regime"""
    state: str  # BULL, BEAR, RANGE, HIGH_VOL, LOW_VOL
    confidence: float
    vix_level: float
    breadth: float
    description: str


# ============================================================
# REGIME ENGINE - The Market's State of Mind
# ============================================================

class RegimeEngine:
    """
    Detects market regime using multiple signals.
    
    This is the "brainstem" of the companion - it tells us
    what kind of market we're in before making any decisions.
    
    Regimes:
    - BULL_LOWVOL: Trending up, low volatility (best for momentum)
    - BULL_HIGHVOL: Trending up, high volatility (momentum but watch stops)
    - BEAR_LOWVOL: Trending down, low volatility (short or cash)
    - BEAR_HIGHVOL: Panic/crash mode (mean reversion opportunities)
    - RANGE: Sideways, no clear trend (mean reversion)
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.spy_data = None
        self.current_regime = None
        
    def load_spy_data(self):
        """Load SPY data for regime detection"""
        self.spy_data = self.df[self.df['ticker'] == 'SPY'].copy()
        if len(self.spy_data) == 0:
            # Use market-wide metrics if no SPY
            logger.warning("No SPY data found, using market-wide metrics")
            self.spy_data = self.df.groupby('date').agg({
                'close': 'mean',
                'volume': 'sum',
                'high': 'max',
                'low': 'min'
            }).reset_index()
        
        self.spy_data = self.spy_data.sort_values('date').reset_index(drop=True)
        
    def calculate_regime_indicators(self) -> Dict:
        """Calculate regime indicators"""
        if self.spy_data is None:
            self.load_spy_data()
        
        df = self.spy_data.copy()
        
        # Trend indicators
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        df['sma_200'] = df['close'].rolling(200).mean()
        
        # Volatility
        df['returns'] = df['close'].pct_change()
        df['volatility_20'] = df['returns'].rolling(20).std() * np.sqrt(252)
        df['volatility_60'] = df['returns'].rolling(60).std() * np.sqrt(252)
        
        # Get latest values
        latest = df.iloc[-1]
        
        # Trend score: -1 (bear) to +1 (bull)
        trend_score = 0
        if latest['close'] > latest['sma_20']:
            trend_score += 0.33
        if latest['close'] > latest['sma_50']:
            trend_score += 0.33
        if latest['close'] > latest['sma_200']:
            trend_score += 0.34
        
        # Adjust for MA alignment
        if latest['sma_20'] > latest['sma_50'] > latest['sma_200']:
            trend_score = min(1.0, trend_score + 0.2)  # Bullish alignment
        elif latest['sma_20'] < latest['sma_50'] < latest['sma_200']:
            trend_score = max(-1.0, trend_score - 0.5)  # Bearish alignment
        
        # Volatility regime
        vol_percentile = (df['volatility_20'] < latest['volatility_20']).mean()
        high_vol = vol_percentile > 0.7
        
        # Breadth (using available data)
        recent_data = self.df[self.df['date'] == df['date'].max()]
        if len(recent_data) > 0:
            recent_data['up'] = recent_data['close'] > recent_data['close'].shift(1)
            breadth = recent_data['close'].pct_change().mean()
        else:
            breadth = 0
        
        return {
            'trend_score': trend_score,
            'volatility_20': latest['volatility_20'],
            'vol_percentile': vol_percentile,
            'high_vol': high_vol,
            'breadth': breadth,
            'above_sma200': latest['close'] > latest['sma_200'],
            'price': latest['close']
        }
    
    def detect_regime(self) -> MarketRegime:
        """Detect current market regime"""
        indicators = self.calculate_regime_indicators()
        
        trend = indicators['trend_score']
        high_vol = indicators['high_vol']
        
        # Determine regime
        if trend > 0.5:
            if high_vol:
                state = "BULL_HIGHVOL"
                description = "Bullish trend with elevated volatility. Momentum works but use wider stops."
            else:
                state = "BULL_LOWVOL"
                description = "Ideal conditions. Bullish trend, low volatility. Momentum and trend-following shine."
        elif trend < -0.3:
            if high_vol:
                state = "BEAR_HIGHVOL"
                description = "Panic/distress mode. Mean reversion opportunities but high risk. Reduce size."
            else:
                state = "BEAR_LOWVOL"
                description = "Grinding bear market. Avoid longs, consider shorts or cash."
        else:
            state = "RANGE"
            description = "Sideways/choppy market. Mean reversion strategies preferred. Reduce position sizes."
        
        # Confidence based on indicator agreement
        confidence = abs(trend) * 0.5 + (0.3 if indicators['above_sma200'] else 0) + 0.2
        confidence = min(1.0, confidence)
        
        self.current_regime = MarketRegime(
            state=state,
            confidence=confidence,
            vix_level=indicators['volatility_20'] * 100,  # Approximate VIX
            breadth=indicators['breadth'],
            description=description
        )
        
        return self.current_regime


# ============================================================
# FACTOR CALCULATOR - Point-in-Time Factor Scores
# ============================================================

class PointInTimeFactorCalculator:
    """
    Calculates factor scores using ONLY data up to the signal date.
    
    CRITICAL: No look-ahead bias. All calculations use only past data.
    """
    
    def __init__(self, df: pd.DataFrame, signal_date: str):
        self.signal_date = pd.Timestamp(signal_date)
        # Filter to only data up to and including signal date
        self.df = df[df['date'] <= self.signal_date].copy()
        self.df = self.df.sort_values(['ticker', 'date']).reset_index(drop=True)
        
    def calculate_all_factors(self) -> pd.DataFrame:
        """Calculate all factors for all stocks as of signal date"""
        logger.info(f"Calculating factors as of {self.signal_date.date()}")
        
        df = self.df.copy()
        
        # Get the latest date's data
        latest_date = df['date'].max()
        
        # ====== RSI (14-day) ======
        df['_delta'] = df.groupby('ticker')['close'].diff()
        df['_gain'] = df['_delta'].where(df['_delta'] > 0, 0)
        df['_loss'] = (-df['_delta']).where(df['_delta'] < 0, 0)
        df['_avg_gain'] = df.groupby('ticker')['_gain'].transform(lambda x: x.rolling(14).mean())
        df['_avg_loss'] = df.groupby('ticker')['_loss'].transform(lambda x: x.rolling(14).mean())
        df['_rs'] = df['_avg_gain'] / df['_avg_loss'].replace(0, np.nan)
        df['rsi'] = 100 - (100 / (1 + df['_rs']))
        
        # ====== Volatility (ATR-based) ======
        df['_tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df.groupby('ticker')['close'].shift(1)),
                abs(df['low'] - df.groupby('ticker')['close'].shift(1))
            )
        )
        df['atr_14'] = df.groupby('ticker')['_tr'].transform(lambda x: x.rolling(14).mean())
        df['atr_pct'] = df['atr_14'] / df['close'] * 100
        
        # ====== Momentum ======
        df['momentum_5'] = df.groupby('ticker')['close'].transform(lambda x: x.pct_change(5))
        df['momentum_20'] = df.groupby('ticker')['close'].transform(lambda x: x.pct_change(20))
        df['momentum_60'] = df.groupby('ticker')['close'].transform(lambda x: x.pct_change(60))
        
        # ====== Trend (EMA 200) ======
        df['ema_200'] = df.groupby('ticker')['close'].transform(lambda x: x.ewm(span=200).mean())
        df['above_ema200'] = (df['close'] > df['ema_200']).astype(int)
        df['dist_from_ema200'] = (df['close'] - df['ema_200']) / df['ema_200'] * 100
        
        # ====== Mean Reversion ======
        df['sma_20'] = df.groupby('ticker')['close'].transform(lambda x: x.rolling(20).mean())
        df['std_20'] = df.groupby('ticker')['close'].transform(lambda x: x.rolling(20).std())
        df['zscore'] = (df['close'] - df['sma_20']) / df['std_20'].replace(0, np.nan)
        
        # ====== Volume ======
        df['vol_sma_20'] = df.groupby('ticker')['volume'].transform(lambda x: x.rolling(20).mean())
        df['vol_ratio'] = df['volume'] / df['vol_sma_20'].replace(0, np.nan)
        
        # ====== 52-Week Position ======
        df['high_52w'] = df.groupby('ticker')['high'].transform(lambda x: x.rolling(252).max())
        df['low_52w'] = df.groupby('ticker')['low'].transform(lambda x: x.rolling(252).min())
        df['pct_from_52w_high'] = (df['high_52w'] - df['close']) / df['high_52w'] * 100
        df['pct_from_52w_low'] = (df['close'] - df['low_52w']) / df['low_52w'] * 100
        
        # ====== Consecutive Days ======
        df['daily_return'] = df.groupby('ticker')['close'].pct_change()
        df['_prev_ret1'] = df.groupby('ticker')['daily_return'].shift(1)
        df['_prev_ret2'] = df.groupby('ticker')['daily_return'].shift(2)
        df['after_2down'] = ((df['daily_return'] < 0) & (df['_prev_ret1'] < 0)).astype(int)
        df['after_2up'] = ((df['daily_return'] > 0) & (df['_prev_ret1'] > 0)).astype(int)
        
        # ====== Bollinger Bands ======
        df['bb_upper'] = df['sma_20'] + 2 * df['std_20']
        df['bb_lower'] = df['sma_20'] - 2 * df['std_20']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower']).replace(0, np.nan)
        
        # Filter to latest date and clean up
        latest = df[df['date'] == latest_date].copy()
        
        # Drop temporary columns
        temp_cols = [c for c in latest.columns if c.startswith('_')]
        latest = latest.drop(columns=temp_cols)
        
        logger.info(f"Calculated factors for {len(latest)} stocks")
        
        return latest
    
    def rank_factors_cross_sectional(self, factor_df: pd.DataFrame) -> pd.DataFrame:
        """Add cross-sectional percentile ranks for each factor"""
        df = factor_df.copy()
        
        rank_cols = ['rsi', 'atr_pct', 'momentum_5', 'momentum_20', 'momentum_60',
                     'zscore', 'vol_ratio', 'pct_from_52w_high']
        
        for col in rank_cols:
            if col in df.columns:
                df[f'{col}_rank'] = df[col].rank(pct=True) * 100
        
        # Special handling: for volatility, lower is better
        if 'atr_pct_rank' in df.columns:
            df['low_vol_rank'] = 100 - df['atr_pct_rank']
        
        return df


# ============================================================
# SIGNAL GENERATOR - Multi-Factor Model Signals
# ============================================================

class SystematicSignalGenerator:
    """
    Generates signals from validated multi-factor models.
    
    Uses the models discovered in FACTOR_COMBINATION_LAB.py:
    - LowVol + After2Down (Sharpe 0.56)
    - Oversold_zscore + LowVol + VolSpike (Sharpe 2.04)
    - RSI_Oversold + VolSpike (Sharpe 0.27)
    """
    
    def __init__(self, factor_df: pd.DataFrame, regime: MarketRegime):
        self.factor_df = factor_df
        self.regime = regime
        self.models = self._load_models()
        
    def _load_models(self) -> List[Dict]:
        """Load validated models"""
        # Hardcoded best models from Factor Lab results
        models = [
            {
                'name': 'LowVol_After2Down',
                'description': 'Low volatility stocks after 2 consecutive down days',
                'factors': ['low_vol', 'after_2down'],
                'conditions': lambda df: (df['atr_pct_rank'] < 30) & (df['after_2down'] == 1),
                'regimes': ['BULL_LOWVOL', 'BULL_HIGHVOL', 'RANGE'],
                'sharpe': 0.56,
                'hold_days': 10
            },
            {
                'name': 'RSI_Oversold_VolSpike',
                'description': 'RSI oversold with volume spike (accumulation)',
                'factors': ['rsi_oversold', 'volume_spike'],
                'conditions': lambda df: (df['rsi'] < 30) & (df['vol_ratio'] > 2.0),
                'regimes': ['BULL_LOWVOL', 'BULL_HIGHVOL', 'RANGE', 'BEAR_HIGHVOL'],
                'sharpe': 0.27,
                'hold_days': 10
            },
            {
                'name': 'Momentum_Quality',
                'description': 'Strong momentum + low volatility + above EMA200',
                'factors': ['momentum', 'low_vol', 'trend'],
                'conditions': lambda df: (df['momentum_20'] > 0.05) & (df['atr_pct_rank'] < 40) & (df['above_ema200'] == 1),
                'regimes': ['BULL_LOWVOL', 'BULL_HIGHVOL'],
                'sharpe': 0.45,
                'hold_days': 20
            },
            {
                'name': 'Mean_Reversion_Oversold',
                'description': 'Z-score below -2 with low volatility',
                'factors': ['zscore', 'low_vol'],
                'conditions': lambda df: (df['zscore'] < -2) & (df['atr_pct_rank'] < 40),
                'regimes': ['RANGE', 'BEAR_HIGHVOL'],
                'sharpe': 0.35,
                'hold_days': 5
            },
            {
                'name': 'Breakout_52wHigh',
                'description': 'Near 52-week high with positive momentum',
                'factors': ['52w_high', 'momentum'],
                'conditions': lambda df: (df['pct_from_52w_high'] < 5) & (df['momentum_20'] > 0),
                'regimes': ['BULL_LOWVOL', 'BULL_HIGHVOL'],
                'sharpe': 0.40,
                'hold_days': 20
            }
        ]
        
        return models
    
    def generate_signals(self) -> pd.DataFrame:
        """Generate signals for all stocks using all models"""
        df = self.factor_df.copy()
        
        # Apply filters
        df = df[df['close'] >= MIN_PRICE]
        df = df[df['vol_sma_20'] >= MIN_AVG_VOLUME]
        
        all_signals = []
        
        for model in self.models:
            # Check if current regime is suitable for this model
            if self.regime.state not in model['regimes']:
                logger.info(f"Skipping {model['name']} - not suited for {self.regime.state} regime")
                continue
            
            # Apply model conditions
            try:
                mask = model['conditions'](df)
                candidates = df[mask].copy()
                
                if len(candidates) > 0:
                    candidates['model'] = model['name']
                    candidates['model_sharpe'] = model['sharpe']
                    candidates['hold_days'] = model['hold_days']
                    candidates['model_description'] = model['description']
                    
                    # Score by composite of relevant factors
                    candidates['score'] = self._calculate_composite_score(candidates, model)
                    
                    all_signals.append(candidates)
                    logger.info(f"{model['name']}: {len(candidates)} candidates")
                    
            except Exception as e:
                logger.warning(f"Error in {model['name']}: {e}")
                continue
        
        if len(all_signals) > 0:
            signals_df = pd.concat(all_signals, ignore_index=True)
            signals_df = signals_df.sort_values('score', ascending=False)
            return signals_df
        
        return pd.DataFrame()
    
    def _calculate_composite_score(self, df: pd.DataFrame, model: Dict) -> pd.Series:
        """Calculate composite score for ranking"""
        score = pd.Series(0.0, index=df.index)
        
        # Factor weights
        if 'low_vol_rank' in df.columns:
            score += df['low_vol_rank'] * 0.3
        if 'momentum_20_rank' in df.columns:
            score += df['momentum_20_rank'] * 0.2
        if 'rsi' in df.columns:
            # Lower RSI = higher score for oversold strategies
            if 'oversold' in model['name'].lower() or 'mean_reversion' in model['name'].lower():
                score += (100 - df['rsi']) * 0.3
        
        # Bonus for regime alignment
        score += model['sharpe'] * 20
        
        return score


# ============================================================
# DISCRETIONARY ANALYZER - Your Picks + Factor Scores
# ============================================================

class DiscretionaryAnalyzer:
    """
    Analyzes your discretionary picks against quantitative factors.
    Provides "Factor Health Report" for each ticker.
    """
    
    def __init__(self, factor_df: pd.DataFrame, regime: MarketRegime):
        self.factor_df = factor_df
        self.regime = regime
        
    def analyze_watchlist(self, watchlist: List[str]) -> List[Dict]:
        """Analyze each ticker in your watchlist"""
        results = []
        
        for ticker in watchlist:
            analysis = self.analyze_ticker(ticker)
            if analysis:
                results.append(analysis)
        
        return results
    
    def analyze_ticker(self, ticker: str) -> Optional[Dict]:
        """Generate Factor Health Report for a single ticker"""
        df = self.factor_df[self.factor_df['ticker'] == ticker]
        
        if len(df) == 0:
            return {
                'ticker': ticker,
                'status': 'NOT_FOUND',
                'message': f'{ticker} not in database',
                'confidence': 'N/A'
            }
        
        row = df.iloc[0]
        
        # Extract factor percentiles
        factors = {
            'RSI': row.get('rsi', np.nan),
            'RSI_pct': row.get('rsi_rank', 50),
            'LowVol_pct': row.get('low_vol_rank', 50),
            'Momentum20_pct': row.get('momentum_20_rank', 50),
            'Momentum60_pct': row.get('momentum_60_rank', 50) if 'momentum_60_rank' in row else 50,
            'ZScore': row.get('zscore', 0),
            'AboveEMA200': bool(row.get('above_ema200', 0)),
            'PctFrom52wHigh': row.get('pct_from_52w_high', 50),
            'VolRatio': row.get('vol_ratio', 1.0),
            'After2Down': bool(row.get('after_2down', 0))
        }
        
        # Calculate bullish/bearish signal count
        bullish = 0
        bearish = 0
        signals = []
        
        # RSI
        if factors['RSI'] < 30:
            bullish += 2
            signals.append("✅ RSI oversold - strong buy signal")
        elif factors['RSI'] > 70:
            bearish += 2
            signals.append("⚠️ RSI overbought - caution")
        
        # Low Volatility
        if factors['LowVol_pct'] > 70:
            bullish += 1
            signals.append("✅ Low volatility - clean signal")
        elif factors['LowVol_pct'] < 30:
            signals.append("⚠️ High volatility - expect large swings")
        
        # Momentum
        if factors['Momentum20_pct'] > 70:
            bullish += 1
            signals.append("✅ Strong momentum")
        elif factors['Momentum20_pct'] < 30:
            bearish += 1
            signals.append("⚠️ Weak momentum")
        
        # Trend
        if factors['AboveEMA200']:
            bullish += 1
            signals.append("✅ Above 200 EMA - uptrend")
        else:
            bearish += 1
            signals.append("⚠️ Below 200 EMA - downtrend")
        
        # 52-week position
        if factors['PctFrom52wHigh'] < 10:
            bullish += 1
            signals.append("✅ Near 52-week high - momentum")
        elif factors['PctFrom52wHigh'] > 40:
            bearish += 1
            signals.append("⚠️ Far from 52-week high")
        
        # After 2 down days
        if factors['After2Down']:
            bullish += 1
            signals.append("✅ After 2 down days - bounce candidate")
        
        # Volume spike
        if factors['VolRatio'] > 2.0:
            signals.append("📊 Volume spike detected")
        
        # Determine confidence
        total = bullish + bearish
        if total > 0:
            score = (bullish - bearish) / total
        else:
            score = 0
        
        if score > 0.5:
            confidence = "HIGH"
        elif score > 0:
            confidence = "MEDIUM"
        elif score > -0.5:
            confidence = "LOW"
        else:
            confidence = "BEARISH"
        
        # Regime alignment check
        regime_aligned = self._check_regime_alignment(factors)
        
        return {
            'ticker': ticker,
            'status': 'OK',
            'confidence': confidence,
            'score': score,
            'bullish_signals': bullish,
            'bearish_signals': bearish,
            'factors': factors,
            'signals': signals,
            'regime_aligned': regime_aligned,
            'price': row.get('close', 0),
            'one_liner': f"{ticker}: RSI({factors['RSI']:.0f}), LowVol({factors['LowVol_pct']:.0f}), Mom({factors['Momentum20_pct']:.0f}) - CONFIDENCE: {confidence}"
        }
    
    def _check_regime_alignment(self, factors: Dict) -> str:
        """Check if stock aligns with current regime"""
        regime = self.regime.state
        
        if regime in ['BULL_LOWVOL', 'BULL_HIGHVOL']:
            # Bullish regimes favor momentum and trend
            if factors['AboveEMA200'] and factors['Momentum20_pct'] > 50:
                return "ALIGNED"
            else:
                return "CONTRARIAN"
        
        elif regime in ['BEAR_LOWVOL', 'BEAR_HIGHVOL']:
            # Bearish regimes favor oversold and mean reversion
            if factors['RSI'] < 30 or factors['ZScore'] < -2:
                return "ALIGNED"
            else:
                return "CONTRARIAN"
        
        else:  # RANGE
            # Range-bound favors mean reversion
            if abs(factors['ZScore']) > 1.5:
                return "ALIGNED"
            else:
                return "NEUTRAL"


# ============================================================
# PORTFOLIO CONSTRUCTOR - Position Sizing & Risk
# ============================================================

class PortfolioConstructor:
    """
    Constructs portfolio with volatility-based position sizing.
    """
    
    def __init__(self, capital: float = PORTFOLIO_CAPITAL):
        self.capital = capital
        self.systematic_budget = capital * SYSTEMATIC_ALLOCATION
        self.discretionary_budget = capital * DISCRETIONARY_ALLOCATION
        
    def size_positions(self, signals_df: pd.DataFrame, track: str = 'systematic') -> pd.DataFrame:
        """Calculate position sizes based on volatility targeting"""
        if len(signals_df) == 0:
            return signals_df
        
        df = signals_df.copy()
        
        budget = self.systematic_budget if track == 'systematic' else self.discretionary_budget
        max_positions = min(MAX_POSITIONS, len(df))
        
        # Equal weight base
        base_allocation = budget / max_positions
        
        # Volatility adjustment: smaller positions for high vol stocks
        if 'atr_pct' in df.columns:
            # Target risk per position
            target_risk = self.capital * TARGET_RISK_PER_POSITION
            
            df['position_value'] = df.apply(
                lambda row: min(
                    base_allocation,
                    target_risk / (row['atr_pct'] / 100) if row['atr_pct'] > 0 else base_allocation
                ),
                axis=1
            )
        else:
            df['position_value'] = base_allocation
        
        # Calculate shares
        df['suggested_shares'] = (df['position_value'] / df['close']).astype(int)
        
        # Filter to positions with at least 1 share
        df = df[df['suggested_shares'] >= 1]
        
        return df.head(max_positions)


# ============================================================
# SIGNAL FACTORY - Main Orchestrator
# ============================================================

class SignalFactory:
    """
    The main orchestrator that runs the daily signal generation.
    
    Usage:
        factory = SignalFactory()
        factory.run('2024-12-19')
    """
    
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self.df = None
        self.signal_date = None
        
        # Components
        self.regime_engine = None
        self.factor_calc = None
        self.signal_gen = None
        self.discretionary = None
        self.portfolio = None
        
        # Results
        self.regime = None
        self.factor_df = None
        self.systematic_signals = None
        self.discretionary_analysis = None
        
        # Your manual watchlist (edit this!)
        self.manual_watchlist = ['ASTS', 'MU', 'PALI', 'SAVA', 'LUNR', 'HUT', 'ANNX']
        
    def load_data(self):
        """Load market data"""
        logger.info(f"Loading market data from {self.db_path}")
        conn = sqlite3.connect(self.db_path)
        self.df = pd.read_sql("SELECT * FROM ohlcv", conn)
        conn.close()
        
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df = self.df.sort_values(['ticker', 'date']).reset_index(drop=True)
        
        logger.info(f"Loaded {len(self.df):,} records, {self.df['ticker'].nunique():,} tickers")
        
    def run(self, signal_date: str = None):
        """
        Run the signal factory for a given date.
        
        Args:
            signal_date: Date string (YYYY-MM-DD). Defaults to latest data date.
        """
        logger.info("="*60)
        logger.info("SIGNAL FACTORY - Daily Signal Generation")
        logger.info("="*60)
        
        # Load data
        if self.df is None:
            self.load_data()
        
        # Determine signal date
        if signal_date is None:
            self.signal_date = self.df['date'].max()
        else:
            self.signal_date = pd.Timestamp(signal_date)
        
        logger.info(f"\n📅 Signal Date: {self.signal_date.date()}")
        
        # Step 1: Detect Market Regime
        logger.info("\n🌍 STEP 1: Detecting Market Regime...")
        self.regime_engine = RegimeEngine(self.df[self.df['date'] <= self.signal_date])
        self.regime = self.regime_engine.detect_regime()
        
        logger.info(f"   Regime: {self.regime.state}")
        logger.info(f"   Confidence: {self.regime.confidence:.0%}")
        logger.info(f"   Description: {self.regime.description}")
        
        # Step 2: Calculate Point-in-Time Factors
        logger.info("\n📊 STEP 2: Calculating Point-in-Time Factors...")
        self.factor_calc = PointInTimeFactorCalculator(
            self.df[self.df['date'] <= self.signal_date],
            str(self.signal_date.date())
        )
        self.factor_df = self.factor_calc.calculate_all_factors()
        self.factor_df = self.factor_calc.rank_factors_cross_sectional(self.factor_df)
        
        # Step 3: Generate Systematic Signals
        logger.info("\n🤖 STEP 3: Generating Systematic Signals...")
        self.signal_gen = SystematicSignalGenerator(self.factor_df, self.regime)
        self.systematic_signals = self.signal_gen.generate_signals()
        
        if len(self.systematic_signals) > 0:
            logger.info(f"   Generated {len(self.systematic_signals)} raw signals")
        
        # Step 4: Analyze Discretionary Watchlist
        logger.info("\n👤 STEP 4: Analyzing Discretionary Watchlist...")
        self.discretionary = DiscretionaryAnalyzer(self.factor_df, self.regime)
        self.discretionary_analysis = self.discretionary.analyze_watchlist(self.manual_watchlist)
        
        for analysis in self.discretionary_analysis:
            if analysis['status'] == 'OK':
                logger.info(f"   {analysis['one_liner']}")
        
        # Step 5: Construct Portfolio
        logger.info("\n💰 STEP 5: Constructing Portfolio...")
        self.portfolio = PortfolioConstructor()
        
        if len(self.systematic_signals) > 0:
            self.systematic_signals = self.portfolio.size_positions(
                self.systematic_signals, track='systematic'
            )
        
        # Step 6: Generate Outputs
        logger.info("\n📝 STEP 6: Generating Outputs...")
        self._generate_trade_sheet()
        self._generate_trading_journal()
        
        logger.info("\n" + "="*60)
        logger.info("🎯 SIGNAL FACTORY COMPLETE")
        logger.info("="*60)
        
        return {
            'regime': self.regime,
            'systematic_signals': self.systematic_signals,
            'discretionary_analysis': self.discretionary_analysis
        }
    
    def _generate_trade_sheet(self):
        """Generate signals_YYYYMMDD.csv"""
        date_str = self.signal_date.strftime('%Y%m%d')
        filepath = f'{OUTPUT_DIR}/signals_{date_str}.csv'
        
        rows = []
        
        # Systematic signals
        if len(self.systematic_signals) > 0:
            for _, row in self.systematic_signals.head(5).iterrows():
                rows.append({
                    'Ticker': row['ticker'],
                    'Strategy': row['model'],
                    'Action': 'BUY',
                    'Price': f"${row['close']:.2f}",
                    'Suggested_Units': row['suggested_shares'],
                    'Position_Value': f"${row['position_value']:.0f}",
                    'Confidence': 'HIGH' if row['model_sharpe'] > 0.4 else 'MEDIUM',
                    'Rationale': row['model_description'],
                    'Track': 'SYSTEMATIC'
                })
        
        # Discretionary signals
        for analysis in self.discretionary_analysis:
            if analysis['status'] == 'OK':
                rows.append({
                    'Ticker': analysis['ticker'],
                    'Strategy': 'DISCRETIONARY',
                    'Action': 'ANALYZE',
                    'Price': f"${analysis['price']:.2f}",
                    'Suggested_Units': '-',
                    'Position_Value': '-',
                    'Confidence': analysis['confidence'],
                    'Rationale': f"Regime: {analysis['regime_aligned']}",
                    'Track': 'DISCRETIONARY'
                })
        
        if rows:
            pd.DataFrame(rows).to_csv(filepath, index=False)
            logger.info(f"   Saved: {filepath}")
    
    def _generate_trading_journal(self):
        """Generate trading_journal_YYYYMMDD.md"""
        date_str = self.signal_date.strftime('%Y%m%d')
        filepath = f'{OUTPUT_DIR}/trading_journal_{date_str}.md'
        
        lines = [
            f"# Signal Factory Output for {self.signal_date.date()}",
            "",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "---",
            "",
            "## 🌍 Market Regime",
            "",
            f"**State:** `{self.regime.state}`",
            f"**Confidence:** {self.regime.confidence:.0%}",
            f"**Implied VIX:** {self.regime.vix_level:.1f}",
            "",
            f"> {self.regime.description}",
            "",
            "---",
            "",
            "## 🤖 Systematic Picks (70% Allocation)",
            ""
        ]
        
        if len(self.systematic_signals) > 0:
            for i, (_, row) in enumerate(self.systematic_signals.head(5).iterrows(), 1):
                lines.extend([
                    f"### {i}. **{row['ticker']}** - {row['model']}",
                    "",
                    f"- **Price:** ${row['close']:.2f}",
                    f"- **Suggested Size:** {row['suggested_shares']} shares (${row['position_value']:.0f})",
                    f"- **Hold Period:** {row['hold_days']} days",
                    f"- **Model Sharpe:** {row['model_sharpe']:.2f}",
                    f"- **Rationale:** {row['model_description']}",
                    ""
                ])
        else:
            lines.append("*No systematic signals for current regime.*\n")
        
        lines.extend([
            "---",
            "",
            "## 👤 Discretionary Analysis (30% Allocation)",
            ""
        ])
        
        for analysis in self.discretionary_analysis:
            if analysis['status'] == 'OK':
                emoji = "🟢" if analysis['confidence'] == 'HIGH' else ("🟡" if analysis['confidence'] == 'MEDIUM' else "🟠")
                lines.extend([
                    f"### {emoji} **{analysis['ticker']}** - {analysis['confidence']}",
                    "",
                    f"- **Price:** ${analysis['price']:.2f}",
                    f"- **Regime Alignment:** {analysis['regime_aligned']}",
                    f"- **Factor Summary:** RSI({analysis['factors']['RSI']:.0f}), LowVol({analysis['factors']['LowVol_pct']:.0f}), Mom({analysis['factors']['Momentum20_pct']:.0f})",
                    "",
                    "**Signals:**"
                ])
                for sig in analysis['signals']:
                    lines.append(f"  - {sig}")
                lines.append("")
            else:
                lines.append(f"- **{analysis['ticker']}:** {analysis.get('message', 'Not found')}\n")
        
        lines.extend([
            "---",
            "",
            "## 🎯 Today's Action Plan",
            "",
        ])
        
        # Generate action items
        if len(self.systematic_signals) > 0:
            top = self.systematic_signals.head(3)
            lines.append(f"1. **SYSTEMATIC:** Consider opening positions in: {', '.join(top['ticker'].tolist())}")
        
        high_conf = [a for a in self.discretionary_analysis if a.get('confidence') == 'HIGH']
        if high_conf:
            lines.append(f"2. **DISCRETIONARY:** High confidence on: {', '.join([a['ticker'] for a in high_conf])}")
        
        low_conf = [a for a in self.discretionary_analysis if a.get('confidence') in ['LOW', 'BEARISH']]
        if low_conf:
            lines.append(f"3. **CAUTION:** Review these positions: {', '.join([a['ticker'] for a in low_conf])}")
        
        lines.extend([
            "",
            "---",
            "",
            "*This is research output, not financial advice. Always do your own analysis.*"
        ])
        
        with open(filepath, 'w') as f:
            f.write('\n'.join(lines))
        
        logger.info(f"   Saved: {filepath}")


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Signal Factory - Daily Signal Generation')
    parser.add_argument('--date', type=str, default=None,
                       help='Signal date (YYYY-MM-DD). Defaults to latest data.')
    parser.add_argument('--watchlist', type=str, default=None,
                       help='Comma-separated list of tickers to analyze')
    args = parser.parse_args()
    
    factory = SignalFactory()
    
    if args.watchlist:
        factory.manual_watchlist = [t.strip() for t in args.watchlist.split(',')]
    
    factory.run(signal_date=args.date)
    
    # Print summary
    print("\n" + "="*60)
    print("📊 QUICK SUMMARY")
    print("="*60)
    
    print(f"\n🌍 Market Regime: {factory.regime.state}")
    print(f"   {factory.regime.description}")
    
    if len(factory.systematic_signals) > 0:
        print(f"\n🤖 Top Systematic Picks:")
        for _, row in factory.systematic_signals.head(3).iterrows():
            print(f"   {row['ticker']}: {row['model']} (${row['close']:.2f})")
    
    print(f"\n👤 Your Watchlist:")
    for a in factory.discretionary_analysis:
        if a['status'] == 'OK':
            print(f"   {a['one_liner']}")
    
    print(f"\n📁 Output files in: {OUTPUT_DIR}/")


if __name__ == '__main__':
    main()
