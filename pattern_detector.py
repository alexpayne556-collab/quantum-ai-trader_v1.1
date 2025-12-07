"""
Pattern Detection Engine
Combines TA-Lib (60+ candlestick patterns) + custom patterns (EMA ribbon, ORB, VWAP)
+ OPTIMIZED ENTRY SIGNALS (from DEEP_PATTERN_EVOLUTION_TRAINER analysis)

Outputs structured data with coordinates for chart visualization

Run: python pattern_detector.py
"""
import numpy as np
import pandas as pd
import yfinance as yf
import talib
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import json

# Import optimized signal configuration
try:
    from optimized_signal_config import (
        OPTIMAL_SIGNAL_WEIGHTS,
        REGIME_SIGNAL_WEIGHTS,
        ENABLED_SIGNALS,
        DISABLED_SIGNALS,
        classify_regime,
        get_signal_weight,
        SIGNAL_PARAMS
    )
    OPTIMIZED_SIGNALS_AVAILABLE = True
except ImportError:
    OPTIMIZED_SIGNALS_AVAILABLE = False
    ENABLED_SIGNALS = ['trend', 'rsi_divergence', 'dip_buy', 'bounce', 'momentum']
    DISABLED_SIGNALS = ['nuclear_dip', 'vol_squeeze', 'consolidation', 'uptrend_pullback']

class PatternDetector:
    """Unified pattern detection engine for candlestick + custom patterns."""
    
    # TA-Lib candlestick patterns (most important 30 for swing trading)
    TALIB_PATTERNS = [
        'CDL2CROWS', 'CDL3BLACKCROWS', 'CDL3INSIDE', 'CDL3LINESTRIKE',
        'CDL3OUTSIDE', 'CDL3STARSINSOUTH', 'CDL3WHITESOLDIERS', 'CDLABANDENBABY',
        'CDLADVANCEBLOCK', 'CDLBELTHOLD', 'CDLBREAKAWAY', 'CDLCLOSINGMARUBOZU',
        'CDLCONCEALBABYSWALL', 'CDLCOUNTERATTACK', 'CDLDARKCLOUDCOVER', 'CDLDOJI',
        'CDLDOJISTAR', 'CDLDRAGONFLYDOJI', 'CDLENGULFING', 'CDLEVENINGDOJISTAR',
        'CDLEVENINGSTAR', 'CDLGAPSIDESIDEWHITE', 'CDLGRAVESTONEDOJI', 'CDLHAMMER',
        'CDLHANGINGMAN', 'CDLHARAMI', 'CDLHARAMICROSS', 'CDLHIGHWAVE',
        'CDLHIKKAKE', 'CDLHIKKAKEMOD', 'CDLHOMINGPIGEON', 'CDLIDENTICAL3CROWS',
        'CDLINNECK', 'CDLINVERTEDHAMMER', 'CDLKICKING', 'CDLKICKINGBYLENGTH',
        'CDLLADDERBOTTOM', 'CDLLONGLEGGEDDOJI', 'CDLLONGLINE', 'CDLMARUBOZU',
        'CDLMATCHINGLOW', 'CDLMATHOLD', 'CDLMORNINGDOJISTAR', 'CDLMORNINGSTAR',
        'CDLONNECK', 'CDLPIERCING', 'CDLRICKSHAWMAN', 'CDLRISEFALL3METHODS',
        'CDLSEPARATINGLINES', 'CDLSHOOTINGSTAR', 'CDLSHORTLINE', 'CDLSPINNINGTOP',
        'CDLSTALLEDPATTERN', 'CDLSTICKSANDWICH', 'CDLTAKURI', 'CDLTASUKIGAP',
        'CDLTHRUSTING', 'CDLTRISTAR', 'CDLUNIQUE3RIVER', 'CDLUPSIDEGAP2CROWS',
        'CDLXSIDEGAP3METHODS'
    ]
    
    # Pattern name mapping (readable names)
    PATTERN_NAMES = {
        'CDLENGULFING': 'Engulfing',
        'CDLHAMMER': 'Hammer',
        'CDLSHOOTINGSTAR': 'Shooting Star',
        'CDLMORNINGSTAR': 'Morning Star',
        'CDLEVENINGSTAR': 'Evening Star',
        'CDLDOJI': 'Doji',
        'CDLHARAMI': 'Harami',
        'CDLPIERCING': 'Piercing Line',
        'CDLDARKCLOUDCOVER': 'Dark Cloud Cover',
        'CDL3WHITESOLDIERS': 'Three White Soldiers',
        'CDL3BLACKCROWS': 'Three Black Crows',
        'CDLINVERTEDHAMMER': 'Inverted Hammer',
        'CDLHANGINGMAN': 'Hanging Man',
    }
    
    def __init__(self):
        self.patterns_detected = []
    
    @staticmethod
    def get_array(df, col):
        """Extract numpy array from DataFrame column (handles multi-index)."""
        if isinstance(df[col], pd.DataFrame):
            return df[col].iloc[:, 0].values
        return df[col].values
    
    def detect_talib_patterns(self, df: pd.DataFrame) -> List[Dict]:
        """
        Detect all TA-Lib candlestick patterns.
        Returns list of detected patterns with metadata.
        """
        open_arr = np.asarray(self.get_array(df, 'Open'), dtype='float64')
        high_arr = np.asarray(self.get_array(df, 'High'), dtype='float64')
        low_arr = np.asarray(self.get_array(df, 'Low'), dtype='float64')
        close_arr = np.asarray(self.get_array(df, 'Close'), dtype='float64')
        
        detected = []
        
        for pattern_func in self.TALIB_PATTERNS:
            try:
                result = getattr(talib, pattern_func)(open_arr, high_arr, low_arr, close_arr)
                
                # Find where pattern is detected (non-zero values)
                pattern_indices = np.where(result != 0)[0]
                
                for idx in pattern_indices:
                    if idx >= len(df) - 1:  # Skip if at edge
                        continue
                    
                    pattern_type = 'BULLISH' if result[idx] > 0 else 'BEARISH'
                    confidence = min(abs(result[idx]) / 100.0, 1.0)  # Normalize to 0-1
                    
                    pattern_name = self.PATTERN_NAMES.get(pattern_func, pattern_func.replace('CDL', ''))
                    
                    detected.append({
                        'pattern': pattern_name,
                        'type': pattern_type,
                        'start_idx': int(idx),
                        'end_idx': int(idx + 1),
                        'price_level': float(close_arr[idx]),
                        'confidence': float(confidence),
                        'timestamp': df.index[idx],
                        'source': 'talib'
                    })
            except Exception as e:
                # Skip patterns that fail
                continue
        
        return detected
    
    def detect_ema_ribbon_alignment(self, df: pd.DataFrame) -> List[Dict]:
        """
        Detect EMA ribbon bullish/bearish alignment with ML-grade scoring.
        Uses weighted alignment score (not binary) - realistic for live trading.
        
        Bullish trend: Fast EMAs trending above slow EMAs (80%+ alignment)
        Bearish trend: Fast EMAs trending below slow EMAs (80%+ alignment)
        """
        close_arr = np.asarray(self.get_array(df, 'Close'), dtype='float64')
        
        # Calculate all EMAs
        emas = {}
        for period in [5, 8, 13, 21, 34, 55, 89]:
            emas[period] = talib.EMA(close_arr, timeperiod=period)
        
        detected = []
        
        # Start after longest EMA has enough data
        for i in range(89, len(df)):
            # Skip if any EMA is NaN
            if any(np.isnan(emas[p][i]) for p in [5, 8, 13, 21, 34, 55, 89]):
                continue
            
            # Calculate alignment score (0-1) based on how many pairs are correctly ordered
            bullish_checks = [
                emas[5][i] > emas[8][i],
                emas[8][i] > emas[13][i],
                emas[13][i] > emas[21][i],
                emas[21][i] > emas[34][i],
                emas[34][i] > emas[55][i],
                emas[55][i] > emas[89][i]
            ]
            
            bearish_checks = [
                emas[5][i] < emas[8][i],
                emas[8][i] < emas[13][i],
                emas[13][i] < emas[21][i],
                emas[21][i] < emas[34][i],
                emas[34][i] < emas[55][i],
                emas[55][i] < emas[89][i]
            ]
            
            bullish_score = sum(bullish_checks) / len(bullish_checks)
            bearish_score = sum(bearish_checks) / len(bearish_checks)
            
            # Require 80%+ alignment (5 out of 6 checks)
            threshold = 0.83
            
            if bullish_score >= threshold:
                # Calculate trend strength based on EMA spread and slope
                spread = (emas[5][i] - emas[89][i]) / emas[89][i]
                
                # EMA slope (momentum): compare current vs 5 bars ago
                if i >= 94:
                    slope_fast = (emas[5][i] - emas[5][i-5]) / emas[5][i-5]
                    slope_slow = (emas[89][i] - emas[89][i-5]) / emas[89][i-5]
                    momentum = (slope_fast + slope_slow) / 2
                else:
                    momentum = 0.0
                
                # Confidence combines alignment score, spread, and momentum
                confidence = min((bullish_score * 0.5 + 
                                 min(abs(spread) * 20, 0.3) +
                                 min(abs(momentum) * 10, 0.2)), 1.0)
                
                # Only add if we just entered alignment (not already detected)
                if i == 89 or i > 0 and detected and detected[-1]['end_idx'] < i - 3:
                    detected.append({
                        'pattern': 'EMA Ribbon Bullish',
                        'type': 'BULLISH',
                        'start_idx': int(i - 3),
                        'end_idx': int(i),
                        'price_level': float(close_arr[i]),
                        'confidence': float(confidence),
                        'timestamp': df.index[i],
                        'source': 'custom',
                        'confluence': ['EMA_BULLISH', f'ALIGNMENT_{bullish_score:.0%}'],
                        'metadata': {
                            'alignment_score': float(bullish_score),
                            'spread': float(spread),
                            'momentum': float(momentum)
                        }
                    })
            
            elif bearish_score >= threshold:
                # Calculate trend strength
                spread = (emas[89][i] - emas[5][i]) / emas[89][i]
                
                # EMA slope
                if i >= 94:
                    slope_fast = (emas[5][i] - emas[5][i-5]) / emas[5][i-5]
                    slope_slow = (emas[89][i] - emas[89][i-5]) / emas[89][i-5]
                    momentum = (slope_fast + slope_slow) / 2
                else:
                    momentum = 0.0
                
                confidence = min((bearish_score * 0.5 + 
                                 min(abs(spread) * 20, 0.3) +
                                 min(abs(momentum) * 10, 0.2)), 1.0)
                
                # Only add if we just entered alignment
                if i == 89 or i > 0 and detected and detected[-1]['end_idx'] < i - 3:
                    detected.append({
                        'pattern': 'EMA Ribbon Bearish',
                        'type': 'BEARISH',
                        'start_idx': int(i - 3),
                        'end_idx': int(i),
                        'price_level': float(close_arr[i]),
                        'confidence': float(confidence),
                        'timestamp': df.index[i],
                        'source': 'custom',
                        'confluence': ['EMA_BEARISH', f'ALIGNMENT_{bearish_score:.0%}'],
                        'metadata': {
                            'alignment_score': float(bearish_score),
                            'spread': float(spread),
                            'momentum': float(momentum)
                        }
                    })
        
        return detected
    
    def detect_vwap_pullback(self, df: pd.DataFrame) -> List[Dict]:
        """
        Detect VWAP pullback setups - institutional-grade.
        Bullish: Price pulls back to VWAP in strong uptrend with volume confirmation
        Bearish: Price rallies to VWAP in downtrend (rejection setup)
        """
        close_arr = np.asarray(self.get_array(df, 'Close'), dtype='float64')
        high_arr = np.asarray(self.get_array(df, 'High'), dtype='float64')
        low_arr = np.asarray(self.get_array(df, 'Low'), dtype='float64')
        volume_arr = np.asarray(self.get_array(df, 'Volume'), dtype='float64')
        
        # Calculate VWAP (daily reset for intraday, rolling for daily)
        typical_price = (high_arr + low_arr + close_arr) / 3
        
        # Detect if intraday data (reset VWAP daily)
        if isinstance(df.index, pd.DatetimeIndex) and len(df) > 100:
            # Rolling VWAP with 20-period window for daily data
            vwap = pd.Series(typical_price).rolling(window=20).apply(
                lambda x: np.sum(x * volume_arr[x.index]) / np.sum(volume_arr[x.index]) if np.sum(volume_arr[x.index]) > 0 else np.nan
            ).values
        else:
            # Cumulative VWAP
            vwap = np.cumsum(typical_price * volume_arr) / (np.cumsum(volume_arr) + 1e-9)
        
        # Trend indicators
        ema20 = talib.EMA(close_arr, timeperiod=20)
        ema50 = talib.EMA(close_arr, timeperiod=50)
        
        # Volume analysis
        vol_sma = talib.SMA(volume_arr, timeperiod=20)
        
        # ATR for volatility
        atr = talib.ATR(high_arr, low_arr, close_arr, timeperiod=14)
        
        detected = []
        
        for i in range(50, len(df)):
            if np.isnan(vwap[i]) or np.isnan(ema20[i]) or np.isnan(ema50[i]):
                continue
            
            # Strong uptrend: price above both EMAs, EMAs rising
            in_uptrend = (close_arr[i] > ema20[i] > ema50[i] and 
                         ema20[i] > ema20[i-5])
            
            # Strong downtrend
            in_downtrend = (close_arr[i] < ema20[i] < ema50[i] and
                           ema20[i] < ema20[i-5])
            
            # BULLISH: Pullback to VWAP in uptrend
            if in_uptrend:
                # Price touched/crossed VWAP from above
                vwap_touch = (low_arr[i] <= vwap[i] * 1.005 and 
                             close_arr[i] >= vwap[i] * 0.995)
                
                # Volume confirmation (above average)
                volume_confirm = volume_arr[i] > vol_sma[i] * 0.8
                
                if vwap_touch and volume_confirm:
                    # Calculate confidence
                    price_distance = abs(close_arr[i] - vwap[i]) / vwap[i]
                    trend_strength = (ema20[i] - ema50[i]) / ema50[i]
                    volume_ratio = volume_arr[i] / (vol_sma[i] + 1e-9)
                    
                    confidence = min(
                        0.5 +  # Base
                        (1.0 - price_distance * 100) * 0.2 +  # Closeness to VWAP
                        min(trend_strength * 20, 0.2) +  # Trend strength
                        min((volume_ratio - 1) * 0.2, 0.1),  # Volume surge
                        1.0
                    )
                    
                    detected.append({
                        'pattern': 'VWAP Pullback Buy',
                        'type': 'BULLISH',
                        'start_idx': int(i - 2),
                        'end_idx': int(i),
                        'price_level': float(vwap[i]),
                        'confidence': float(confidence),
                        'timestamp': df.index[i],
                        'source': 'custom',
                        'confluence': ['VWAP_SUPPORT', 'UPTREND', 'VOLUME_CONFIRM'],
                        'metadata': {
                            'vwap_level': float(vwap[i]),
                            'distance_from_vwap': float(price_distance),
                            'volume_ratio': float(volume_ratio)
                        }
                    })
            
            # BEARISH: Rally to VWAP in downtrend (rejection)
            elif in_downtrend:
                vwap_touch = (high_arr[i] >= vwap[i] * 0.995 and 
                             close_arr[i] <= vwap[i] * 1.005)
                
                volume_confirm = volume_arr[i] > vol_sma[i] * 0.8
                
                if vwap_touch and volume_confirm:
                    price_distance = abs(close_arr[i] - vwap[i]) / vwap[i]
                    trend_strength = abs((ema20[i] - ema50[i]) / ema50[i])
                    volume_ratio = volume_arr[i] / (vol_sma[i] + 1e-9)
                    
                    confidence = min(
                        0.5 +
                        (1.0 - price_distance * 100) * 0.2 +
                        min(trend_strength * 20, 0.2) +
                        min((volume_ratio - 1) * 0.2, 0.1),
                        1.0
                    )
                    
                    detected.append({
                        'pattern': 'VWAP Rejection Sell',
                        'type': 'BEARISH',
                        'start_idx': int(i - 2),
                        'end_idx': int(i),
                        'price_level': float(vwap[i]),
                        'confidence': float(confidence),
                        'timestamp': df.index[i],
                        'source': 'custom',
                        'confluence': ['VWAP_RESISTANCE', 'DOWNTREND', 'VOLUME_CONFIRM'],
                        'metadata': {
                            'vwap_level': float(vwap[i]),
                            'distance_from_vwap': float(price_distance),
                            'volume_ratio': float(volume_ratio)
                        }
                    })
        
        return detected
    
    def detect_optimized_entry_signals(self, df: pd.DataFrame) -> List[Dict]:
        """
        Detect OPTIMIZED ENTRY SIGNALS from DEEP_PATTERN_EVOLUTION_TRAINER.
        
        TIER S (weight 1.8): trend - 65% WR, +13.7% avg PnL
        TIER A (weight 1.0): rsi_divergence - 57.9% WR, +8.0% avg PnL
        TIER B (weight 0.5): dip_buy, bounce, momentum - conditional use
        TIER F (disabled): nuclear_dip, vol_squeeze, consolidation, uptrend_pullback
        
        CRITICAL: 'trend' is DISABLED in sideways markets (negative returns there!)
        """
        if len(df) < 60:
            return []
        
        detected = []
        close_arr = np.asarray(self.get_array(df, 'Close'), dtype='float64')
        high_arr = np.asarray(self.get_array(df, 'High'), dtype='float64')
        low_arr = np.asarray(self.get_array(df, 'Low'), dtype='float64')
        volume_arr = np.asarray(self.get_array(df, 'Volume'), dtype='float64')
        
        # Calculate all indicators
        rsi = talib.RSI(close_arr, timeperiod=14)
        macd, macd_signal, _ = talib.MACD(close_arr)
        
        # EMAs for ribbon
        ema_8 = talib.EMA(close_arr, timeperiod=8)
        ema_13 = talib.EMA(close_arr, timeperiod=13)
        ema_21 = talib.EMA(close_arr, timeperiod=21)
        
        for i in range(60, len(df)):
            if np.isnan(rsi[i]) or np.isnan(macd[i]):
                continue
            
            # Calculate features for this bar
            mom_5d = (close_arr[i] / close_arr[i-5] - 1) * 100 if i >= 5 else 0
            ret_21d = (close_arr[i] / close_arr[i-21] - 1) * 100 if i >= 21 else 0
            
            # Ribbon bullish check
            ribbon_bullish = ema_8[i] > ema_13[i] > ema_21[i]
            
            # MACD rising
            macd_rising = macd[i] > macd_signal[i]
            
            # Bounce calculation
            low_5d = min(low_arr[max(0, i-4):i+1])
            bounce = (close_arr[i] / low_5d - 1) * 100
            ema_8_rising = ema_8[i] > ema_8[i-3] if i >= 3 else False
            bounce_signal = bounce > 3 and ema_8_rising
            
            # Trend alignment
            ret_5d = (close_arr[i] / close_arr[i-5] - 1) if i >= 5 else 0
            ret_10d = (close_arr[i] / close_arr[i-10] - 1) if i >= 10 else 0
            trend_align = (np.sign(ret_5d) + np.sign(ret_10d) + np.sign(ret_21d/100)) / 3
            
            # RSI Divergence (simplified)
            price_low_5d = min(close_arr[max(0, i-4):i+1])
            rsi_min_5d = min(rsi[max(0, i-4):i+1])
            rsi_divergence = (close_arr[i] <= price_low_5d * 1.02) and (rsi[i] > rsi_min_5d + 5)
            
            # === CLASSIFY REGIME FOR SIGNAL SWITCHING ===
            if OPTIMIZED_SIGNALS_AVAILABLE:
                regime = classify_regime(ret_21d, ribbon_bullish)
                weights = REGIME_SIGNAL_WEIGHTS.get(regime, OPTIMAL_SIGNAL_WEIGHTS)
            else:
                regime = 'bull' if ret_21d > 5 and ribbon_bullish else ('bear' if ret_21d < -5 else 'sideways')
                weights = {'trend': 1.8 if regime != 'sideways' else 0.0,
                          'rsi_divergence': 1.0, 'dip_buy': 0.5, 'bounce': 0.5, 'momentum': 0.5}
            
            # === TIER S: TREND SIGNAL (65% WR, +13.7% avg) ===
            if weights.get('trend', 0) > 0 and 'trend' not in DISABLED_SIGNALS:
                if trend_align > 0.5 and ribbon_bullish:
                    confidence = min(0.65 + (trend_align - 0.5) * 0.3, 0.95)
                    detected.append({
                        'pattern': '🏆 TREND (Tier S)',
                        'type': 'BULLISH',
                        'start_idx': int(i - 1),
                        'end_idx': int(i),
                        'price_level': float(close_arr[i]),
                        'confidence': float(confidence),
                        'timestamp': df.index[i],
                        'source': 'optimized',
                        'signal_weight': weights['trend'],
                        'regime': regime,
                        'confluence': ['TREND_ALIGN', 'RIBBON_BULLISH'],
                        'metadata': {
                            'trend_align': float(trend_align),
                            'win_rate': 0.65,
                            'avg_pnl': 13.7
                        }
                    })
            
            # === TIER A: RSI DIVERGENCE (57.9% WR, +8.0% avg) ===
            if weights.get('rsi_divergence', 0) > 0 and 'rsi_divergence' not in DISABLED_SIGNALS:
                if rsi_divergence and macd_rising:
                    confidence = 0.58
                    detected.append({
                        'pattern': '🥇 RSI Divergence (Tier A)',
                        'type': 'BULLISH',
                        'start_idx': int(i - 1),
                        'end_idx': int(i),
                        'price_level': float(close_arr[i]),
                        'confidence': float(confidence),
                        'timestamp': df.index[i],
                        'source': 'optimized',
                        'signal_weight': weights['rsi_divergence'],
                        'regime': regime,
                        'confluence': ['RSI_DIVERGENCE', 'MACD_RISING'],
                        'metadata': {
                            'rsi': float(rsi[i]),
                            'win_rate': 0.579,
                            'avg_pnl': 8.0
                        }
                    })
            
            # === TIER B: DIP BUY (best in sideways: 62.5% WR, +12.2% avg) ===
            if weights.get('dip_buy', 0) > 0 and 'dip_buy' not in DISABLED_SIGNALS:
                if rsi[i] < 30 and mom_5d < -3:
                    confidence = 0.54 + (0.08 if regime == 'sideways' else 0)
                    detected.append({
                        'pattern': '📉 Dip Buy (Tier B)',
                        'type': 'BULLISH',
                        'start_idx': int(i - 1),
                        'end_idx': int(i),
                        'price_level': float(close_arr[i]),
                        'confidence': float(confidence),
                        'timestamp': df.index[i],
                        'source': 'optimized',
                        'signal_weight': weights['dip_buy'],
                        'regime': regime,
                        'confluence': ['RSI_OVERSOLD', 'MOMENTUM_NEGATIVE'],
                        'metadata': {
                            'rsi': float(rsi[i]),
                            'momentum': float(mom_5d),
                            'best_regime': 'sideways'
                        }
                    })
            
            # === TIER B: BOUNCE (works in bear/sideways: 60% WR) ===
            if weights.get('bounce', 0) > 0 and 'bounce' not in DISABLED_SIGNALS:
                if bounce > 8 and macd_rising:
                    confidence = 0.54 + (0.06 if regime in ['bear', 'sideways'] else 0)
                    detected.append({
                        'pattern': '📈 Bounce (Tier B)',
                        'type': 'BULLISH',
                        'start_idx': int(i - 1),
                        'end_idx': int(i),
                        'price_level': float(close_arr[i]),
                        'confidence': float(confidence),
                        'timestamp': df.index[i],
                        'source': 'optimized',
                        'signal_weight': weights['bounce'],
                        'regime': regime,
                        'confluence': ['BOUNCE_FROM_LOW', 'MACD_RISING'],
                        'metadata': {
                            'bounce_pct': float(bounce),
                            'best_regime': 'bear/sideways'
                        }
                    })
            
            # === TIER B: MOMENTUM (51% WR, needs confirmation) ===
            if weights.get('momentum', 0) > 0 and 'momentum' not in DISABLED_SIGNALS:
                if mom_5d > 5 and macd_rising and bounce_signal:
                    confidence = 0.51
                    detected.append({
                        'pattern': '🚀 Momentum (Tier B)',
                        'type': 'BULLISH',
                        'start_idx': int(i - 1),
                        'end_idx': int(i),
                        'price_level': float(close_arr[i]),
                        'confidence': float(confidence),
                        'timestamp': df.index[i],
                        'source': 'optimized',
                        'signal_weight': weights['momentum'],
                        'regime': regime,
                        'confluence': ['STRONG_MOMENTUM', 'MACD_RISING', 'BOUNCE_CONFIRM'],
                        'metadata': {
                            'momentum': float(mom_5d)
                        }
                    })
        
        return detected
    
    def detect_opening_range_breakout(self, df: pd.DataFrame) -> List[Dict]:
        """
        Detect opening range breakouts (first 30 minutes).
        Works on intraday data only.
        """
        # Only works on minute-level data
        if len(df) < 30:
            return []
        
        detected = []
        close_arr = np.asarray(self.get_array(df, 'Close'), dtype='float64')
        high_arr = np.asarray(self.get_array(df, 'High'), dtype='float64')
        low_arr = np.asarray(self.get_array(df, 'Low'), dtype='float64')
        
        # Group by date to find opening ranges
        if isinstance(df.index, pd.DatetimeIndex):
            for date in df.index.date:
                day_data = df[df.index.date == date]
                if len(day_data) < 30:
                    continue
                
                # First 30 bars = opening range
                or_high = day_data['High'].iloc[:30].max()
                or_low = day_data['Low'].iloc[:30].min()
                
                # Look for breakouts after bar 30
                for i in range(30, len(day_data)):
                    idx_global = df.index.get_loc(day_data.index[i])
                    
                    # Bullish breakout
                    if close_arr[idx_global] > or_high * 1.001:
                        confidence = min((close_arr[idx_global] - or_high) / or_high * 50, 1.0)
                        detected.append({
                            'pattern': 'Opening Range Breakout',
                            'type': 'BULLISH',
                            'start_idx': int(idx_global - 2),
                            'end_idx': int(idx_global),
                            'price_level': float(or_high),
                            'confidence': float(confidence),
                            'timestamp': df.index[idx_global],
                            'source': 'custom',
                            'confluence': ['ORB_BULLISH']
                        })
                    
                    # Bearish breakdown
                    elif close_arr[idx_global] < or_low * 0.999:
                        confidence = min((or_low - close_arr[idx_global]) / or_low * 50, 1.0)
                        detected.append({
                            'pattern': 'Opening Range Breakdown',
                            'type': 'BEARISH',
                            'start_idx': int(idx_global - 2),
                            'end_idx': int(idx_global),
                            'price_level': float(or_low),
                            'confidence': float(confidence),
                            'timestamp': df.index[idx_global],
                            'source': 'custom',
                            'confluence': ['ORB_BEARISH']
                        })
        
        return detected
    
    def detect_all_patterns(self, ticker: str, period='60d', interval='1d') -> Dict:
        """
        Main detection pipeline: fetch data and run all pattern detectors.
        Now includes OPTIMIZED ENTRY SIGNALS from DEEP_PATTERN_EVOLUTION_TRAINER.
        """
        print(f"\n{'='*60}")
        print(f"Pattern Detection: {ticker}")
        if OPTIMIZED_SIGNALS_AVAILABLE:
            print(f"🎯 OPTIMIZED SIGNALS ENABLED (Tier S/A/B ranking)")
        print(f"{'='*60}")
        
        # Fetch data
        df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
        
        if len(df) < 20:
            print(f"Insufficient data for {ticker}")
            return {'ticker': ticker, 'patterns': [], 'error': 'Insufficient data'}
        
        # Run all detectors
        import time
        start_time = time.time()
        
        talib_patterns = self.detect_talib_patterns(df)
        ema_patterns = self.detect_ema_ribbon_alignment(df)
        vwap_patterns = self.detect_vwap_pullback(df)
        orb_patterns = self.detect_opening_range_breakout(df)
        
        # 🎯 NEW: Run OPTIMIZED ENTRY SIGNALS
        optimized_signals = self.detect_optimized_entry_signals(df)
        
        all_patterns = talib_patterns + ema_patterns + vwap_patterns + orb_patterns + optimized_signals
        
        detection_time = time.time() - start_time
        
        # Sort by timestamp (most recent first)
        all_patterns.sort(key=lambda x: x['timestamp'], reverse=True)
        
        # Calculate confluence (patterns happening at same time)
        confluence_patterns = self._calculate_confluence(all_patterns)
        
        print(f"\n📊 Detection Summary:")
        print(f"  Total patterns detected: {len(all_patterns)}")
        print(f"  TA-Lib patterns: {len(talib_patterns)}")
        print(f"  EMA ribbon alignments: {len(ema_patterns)}")
        print(f"  VWAP pullbacks: {len(vwap_patterns)}")
        print(f"  Opening range breakouts: {len(orb_patterns)}")
        print(f"  🎯 OPTIMIZED SIGNALS: {len(optimized_signals)}")
        print(f"  High-confidence patterns (>0.7): {len([p for p in all_patterns if p['confidence'] > 0.7])}")
        print(f"  Detection time: {detection_time*1000:.1f}ms")
        
        # Show optimized signals first (they're the most important)
        if optimized_signals:
            print(f"\n🎯 OPTIMIZED ENTRY SIGNALS (from training):")
            for i, sig in enumerate(sorted(optimized_signals, key=lambda x: x.get('signal_weight', 0), reverse=True)[:5], 1):
                weight = sig.get('signal_weight', 0)
                regime = sig.get('regime', 'unknown')
                print(f"  {i}. {sig['pattern']} - Weight: {weight:.1f} - Regime: {regime} - "
                      f"Confidence: {sig['confidence']:.2%}")
        
        # Show latest 5 patterns
        print(f"\n🔍 Latest Patterns Detected:")
        for i, pattern in enumerate(all_patterns[:5], 1):
            print(f"  {i}. {pattern['pattern']} ({pattern['type']}) - "
                  f"Confidence: {pattern['confidence']:.2%} - "
                  f"Price: ${pattern['price_level']:.2f} - "
                  f"{pattern['timestamp'].strftime('%Y-%m-%d')}")
        
        return {
            'ticker': ticker,
            'patterns': all_patterns,
            'optimized_signals': optimized_signals,
            'confluence_patterns': confluence_patterns,
            'stats': {
                'total': len(all_patterns),
                'talib': len(talib_patterns),
                'custom': len(ema_patterns) + len(vwap_patterns) + len(orb_patterns),
                'optimized': len(optimized_signals),
                'high_confidence': len([p for p in all_patterns if p['confidence'] > 0.7]),
                'detection_time_ms': detection_time * 1000
            }
        }
    
    def _calculate_confluence(self, patterns: List[Dict]) -> List[Dict]:
        """
        Find patterns that happen at same time (within 3 bars).
        High confluence = multiple patterns agreeing = stronger signal.
        """
        confluence_groups = []
        
        for i, pattern in enumerate(patterns):
            # Find other patterns within ±3 indices
            nearby = []
            for other in patterns:
                if other != pattern:
                    idx_diff = abs(pattern['start_idx'] - other['start_idx'])
                    if idx_diff <= 3 and pattern['type'] == other['type']:
                        nearby.append(other)
            
            if len(nearby) >= 2:  # At least 3 patterns total (including this one)
                confluence_groups.append({
                    'main_pattern': pattern,
                    'confluent_patterns': nearby,
                    'confluence_count': len(nearby) + 1,
                    'combined_confidence': min(pattern['confidence'] * (1 + len(nearby) * 0.1), 1.0)
                })
        
        return confluence_groups


if __name__ == '__main__':
    # Test on 4 tickers
    detector = PatternDetector()
    
    tickers = ['MU', 'IONQ', 'APLD', 'ANNX']
    
    all_results = {}
    
    for ticker in tickers:
        result = detector.detect_all_patterns(ticker, period='60d', interval='1d')
        all_results[ticker] = result
    
    # Save to JSON
    output_file = 'data/pattern_detection_results.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n\n✅ Pattern detection complete!")
    print(f"Results saved to: {output_file}")
    print(f"\n{'='*60}")
    print("Next: Build chart_engine.py to visualize these patterns")
    print(f"{'='*60}")
