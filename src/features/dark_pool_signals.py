"""
Dark Pool & Microstructure Signals Module
==========================================

Institutional flow detection using FREE data sources (yfinance, FINRA).
Based on Perplexity research + dark pool formulas from December 8, 2025.

This module implements 5 core institutional activity signals:
1. IFI - Institutional Flow Index (buy/sell imbalance on large trades)
2. A/D - Accumulation/Distribution Line (money flow tracking)
3. OBV - On-Balance Volume (volume-weighted directional flow)
4. VROC - Volume Rate of Change (acceleration/deceleration)
5. SMI - Smart Money Index (composite 0-100 score)

Author: Quantum AI Trader Team
Date: December 8, 2025
Version: 1.0.0
"""

import yfinance as yf
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta
import logging
from functools import lru_cache

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DarkPoolSignals:
    """
    Institutional flow detection using free data sources.
    
    All signals are computed from yfinance data (free, unlimited).
    Returns 0-100 scores that can be fed into the meta-learner.
    
    Usage:
        signals = DarkPoolSignals("NVDA")
        smi_result = signals.smart_money_index(lookback=20)
        print(f"SMI Score: {smi_result['SMI']:.1f}/100")
        print(f"Signal: {smi_result['signal']}")  # BUY/SELL/HOLD
    """
    
    def __init__(self, ticker: str, cache_enabled: bool = True):
        """
        Initialize Dark Pool Signals for a ticker.
        
        Args:
            ticker: Stock symbol (e.g., "NVDA", "TSLA")
            cache_enabled: Use in-memory cache for repeated queries
        """
        self.ticker = ticker.upper()
        self.cache_enabled = cache_enabled
        self._data_cache: Dict[str, pd.DataFrame] = {}
        
        logger.info(f"Initialized DarkPoolSignals for {self.ticker}")
    
    def _get_data(self, interval: str = "1d", period: str = "60d") -> pd.DataFrame:
        """
        Fetch OHLCV data with caching.
        
        Args:
            interval: Data granularity ("1m", "5m", "1h", "1d")
            period: Lookback period ("30d", "60d", "1y")
            
        Returns:
            DataFrame with OHLCV columns
        """
        cache_key = f"{self.ticker}_{interval}_{period}"
        
        # Check cache (valid for 5 minutes)
        if self.cache_enabled and cache_key in self._data_cache:
            cached_data, cache_time = self._data_cache[cache_key]
            if (datetime.now() - cache_time).seconds < 300:  # 5 min TTL
                logger.debug(f"Cache hit for {cache_key}")
                return cached_data
        
        # Fetch fresh data
        try:
            logger.info(f"Fetching {interval} data for {self.ticker} (period={period})")
            data = yf.download(self.ticker, interval=interval, period=period, progress=False)
            
            if data.empty:
                raise ValueError(f"No data returned for {self.ticker}")
            
            # Flatten MultiIndex columns if present (yfinance returns MultiIndex for single tickers)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            
            # Cache it
            if self.cache_enabled:
                self._data_cache[cache_key] = (data.copy(), datetime.now())
            
            return data
        
        except Exception as e:
            logger.error(f"Failed to fetch data for {self.ticker}: {e}")
            raise
    
    def institutional_flow_index(self, days: int = 7) -> Dict[str, float]:
        """
        IFI - Institutional Flow Index
        
        Formula: IFI = (Large Buy Volume - Large Sell Volume) / Total Volume
        
        "Large" = volume bars > 90th percentile (proxy for block trades).
        Positive IFI = net institutional buying.
        Negative IFI = net institutional selling.
        
        Args:
            days: Lookback period in trading days (default 7, yfinance 8-day limit for 1m data)
            
        Returns:
            dict:
                IFI: Raw score (-0.3 to +0.3 typical range)
                IFI_score: Normalized 0-100 score
                buy_volume: Total large buy volume
                sell_volume: Total large sell volume
                interpretation: BULLISH/BEARISH/NEUTRAL
                
        Example:
            >>> signals = DarkPoolSignals("NVDA")
            >>> ifi = signals.institutional_flow_index(days=20)
            >>> print(f"IFI Score: {ifi['IFI_score']:.1f}/100")
            >>> print(f"Interpretation: {ifi['interpretation']}")
        """
        try:
            # Get minute data (yfinance limit: 7-8 days for 1m interval)
            period = min(days + 1, 7)  # Cap at 7 days due to API limit
            data = self._get_data(interval="1m", period=f"{period}d")
            
            # Take only requested days (390 minutes per trading day)
            minutes_needed = days * 390
            recent = data.tail(minutes_needed)
            
            if len(recent) < 100:  # Minimum data requirement
                logger.warning(f"Insufficient data for IFI calculation (got {len(recent)} bars)")
                return self._ifi_fallback()
            
            # Define "large" trades as volume > 90th percentile
            vol_threshold = recent['Volume'].quantile(0.9)
            
            # Calculate directional volume
            recent = recent.copy()
            recent['price_direction'] = np.sign(recent['Close'].diff())
            recent['large_vol'] = np.where(recent['Volume'] > vol_threshold, recent['Volume'], 0)
            recent['buy_vol'] = np.where(recent['price_direction'] > 0, recent['large_vol'], 0)
            recent['sell_vol'] = np.where(recent['price_direction'] < 0, recent['large_vol'], 0)
            
            # Calculate IFI (convert sums to scalars)
            buy_total = float(recent['buy_vol'].sum())
            sell_total = float(recent['sell_vol'].sum())
            total_vol = float(recent['Volume'].sum())
            
            ifi = (buy_total - sell_total) / total_vol if total_vol > 0 else 0.0
            
            # Convert to 0-100 scale (assume IFI ranges -0.3 to +0.3)
            # IFI = -0.3 → score = 0
            # IFI = 0 → score = 50
            # IFI = +0.3 → score = 100
            ifi_score = max(0, min(100, (ifi / 0.3 + 1) * 50))
            
            # Interpretation
            if ifi > 0.15:
                interpretation = 'BULLISH'
            elif ifi < -0.15:
                interpretation = 'BEARISH'
            else:
                interpretation = 'NEUTRAL'
            
            logger.info(f"IFI for {self.ticker}: {ifi:.4f} (score={ifi_score:.1f}, {interpretation})")
            
            return {
                'IFI': float(ifi),
                'IFI_score': float(ifi_score),
                'buy_volume': float(buy_total),
                'sell_volume': float(sell_total),
                'interpretation': interpretation,
                'lookback_days': days
            }
        
        except Exception as e:
            logger.error(f"IFI calculation failed for {self.ticker}: {e}", exc_info=True)
            return self._ifi_fallback()
    
    def _ifi_fallback(self) -> Dict[str, float]:
        """Fallback values when IFI calculation fails."""
        return {
            'IFI': 0.0,
            'IFI_score': 50.0,
            'buy_volume': 0.0,
            'sell_volume': 0.0,
            'interpretation': 'NEUTRAL',
            'error': 'Insufficient data or calculation failed'
        }
    
    def accumulation_distribution(self, lookback: int = 20) -> Dict[str, float]:
        """
        A/D Line - Accumulation/Distribution
        
        Formula: 
            CLV = ((Close - Low) - (High - Close)) / (High - Low)
            Money Flow = CLV × Volume
            A/D Line = Cumulative sum of Money Flow
        
        Positive slope = accumulation (institutional buying).
        Negative slope = distribution (institutional selling).
        Divergence with price = hidden strength/weakness.
        
        Args:
            lookback: Number of days to analyze (default 20)
            
        Returns:
            dict:
                AD_score: Normalized 0-100 score
                AD_trend: 5-day change in A/D line
                signal: ACCUMULATION/DISTRIBUTION/NEUTRAL
                momentum: Trend strength (-1 to +1)
                
        Example:
            >>> ad = signals.accumulation_distribution(lookback=20)
            >>> print(f"A/D Score: {ad['AD_score']:.1f}/100")
            >>> print(f"Signal: {ad['signal']}")
        """
        try:
            data = self._get_data(interval="1d", period=f"{lookback+10}d")
            recent = data.tail(lookback)
            
            if len(recent) < 10:
                logger.warning(f"Insufficient data for A/D calculation")
                return self._ad_fallback()
            
            # Calculate Close Location Value (CLV)
            high_low_range = recent['High'] - recent['Low']
            # Replace zeros to avoid division errors (vectorized)
            high_low_range = high_low_range.replace(0, 1e-10)
            
            clv = ((recent['Close'] - recent['Low']) - (recent['High'] - recent['Close'])) / high_low_range
            
            # Money Flow = CLV × Volume
            money_flow = clv * recent['Volume']
            
            # Cumulative A/D Line
            ad_line = money_flow.cumsum()
            
            # Normalize to 0-100 scale
            ad_min, ad_max = ad_line.min(), ad_line.max()
            if ad_max > ad_min:
                ad_normalized = (ad_line - ad_min) / (ad_max - ad_min) * 100
            else:
                ad_normalized = pd.Series([50] * len(ad_line), index=ad_line.index)
            
            # Trend: Is A/D rising or falling? (convert to scalars)
            if len(ad_line) >= 5:
                ad_trend = float(ad_line.iloc[-1] - ad_line.iloc[-5])
            else:
                ad_trend = 0.0
            
            # Signal determination
            if ad_trend > 0:
                signal = 'ACCUMULATION'
                momentum = min(1.0, ad_trend / (abs(ad_trend) + 1e-10))
            elif ad_trend < 0:
                signal = 'DISTRIBUTION'
                momentum = max(-1.0, ad_trend / (abs(ad_trend) + 1e-10))
            else:
                signal = 'NEUTRAL'
                momentum = 0.0
            
            ad_score = float(ad_normalized.iloc[-1])
            
            logger.info(f"A/D for {self.ticker}: score={ad_score:.1f}, trend={ad_trend:.2e}, {signal}")
            
            return {
                'AD_score': ad_score,
                'AD_trend': float(ad_trend),
                'signal': signal,
                'momentum': float(momentum),
                'lookback_days': lookback
            }
        
        except Exception as e:
            logger.error(f"A/D calculation failed for {self.ticker}: {e}")
            return self._ad_fallback()
    
    def _ad_fallback(self) -> Dict[str, float]:
        """Fallback values when A/D calculation fails."""
        return {
            'AD_score': 50.0,
            'AD_trend': 0.0,
            'signal': 'NEUTRAL',
            'momentum': 0.0,
            'error': 'Insufficient data or calculation failed'
        }
    
    def obv_institutional(self, lookback: int = 20) -> Dict[str, float]:
        """
        OBV - On-Balance Volume (Institutional-Weighted)
        
        Standard OBV: Add volume on up days, subtract on down days.
        Enhancement: Weight large volume days more heavily (>85th percentile).
        
        Divergence detection:
        - Price up, OBV down = bearish (distribution on rallies)
        - Price down, OBV up = bullish (accumulation on dips)
        
        Args:
            lookback: Number of days (default 20)
            
        Returns:
            dict:
                OBV_score: Normalized 0-100
                divergence: BULLISH/BEARISH/NONE
                price_direction: UP/DOWN
                obv_direction: UP/DOWN
                
        Example:
            >>> obv = signals.obv_institutional(lookback=20)
            >>> if obv['divergence'] != 'NONE':
            >>>     print(f"⚠️ DIVERGENCE: {obv['divergence']}")
        """
        try:
            data = self._get_data(interval="1d", period=f"{lookback+10}d")
            recent = data.tail(lookback).copy()
            
            if len(recent) < 10:
                return self._obv_fallback()
            
            # Standard OBV calculation (flatten arrays if MultiIndex present)
            close_vals = recent['Close'].values
            volume_vals = recent['Volume'].values
            
            # Handle MultiIndex DataFrames from yf.download
            if close_vals.ndim > 1:
                close_vals = close_vals.flatten()
            if volume_vals.ndim > 1:
                volume_vals = volume_vals.flatten()
            
            price_change = recent['Close'].diff()
            
            obv_standard = np.where(price_change > 0, volume_vals, 
                                   np.where(price_change < 0, -volume_vals, 0))
            obv_standard = pd.Series(obv_standard, index=recent.index).cumsum()
            
            # Institutional-weighted OBV (weight large volume days more)
            vol_threshold = recent['Volume'].quantile(0.85)
            institutional_vol = np.where(volume_vals > vol_threshold,
                                         volume_vals,
                                         volume_vals * 0.5)  # Reduce weight of retail
            
            obv_inst = np.where(price_change > 0, institutional_vol,
                               np.where(price_change < 0, -institutional_vol, 0))
            obv_inst = pd.Series(obv_inst, index=recent.index).cumsum()
            
            # Normalize to 0-100
            obv_min, obv_max = obv_inst.min(), obv_inst.max()
            if obv_max > obv_min:
                obv_score = (obv_inst.iloc[-1] - obv_min) / (obv_max - obv_min) * 100
            else:
                obv_score = 50.0
            
            # Detect divergence (last 5 days)
            if len(recent) >= 5:
                price_trend = recent['Close'].iloc[-1] - recent['Close'].iloc[-5]
                obv_trend = obv_inst.iloc[-1] - obv_inst.iloc[-5]
                
                # Divergence conditions
                if price_trend < 0 and obv_trend > 0:
                    divergence = 'BULLISH'  # Accumulation on dips
                elif price_trend > 0 and obv_trend < 0:
                    divergence = 'BEARISH'  # Distribution on rallies
                else:
                    divergence = 'NONE'
                
                price_direction = 'UP' if price_trend > 0 else 'DOWN'
                obv_direction = 'UP' if obv_trend > 0 else 'DOWN'
            else:
                divergence = 'NONE'
                price_direction = 'NEUTRAL'
                obv_direction = 'NEUTRAL'
            
            logger.info(f"OBV for {self.ticker}: score={obv_score:.1f}, divergence={divergence}")
            
            return {
                'OBV_score': float(obv_score),
                'divergence': divergence,
                'price_direction': price_direction,
                'obv_direction': obv_direction,
                'lookback_days': lookback
            }
        
        except Exception as e:
            logger.error(f"OBV calculation failed for {self.ticker}: {e}")
            return self._obv_fallback()
    
    def _obv_fallback(self) -> Dict[str, float]:
        """Fallback values when OBV calculation fails."""
        return {
            'OBV_score': 50.0,
            'divergence': 'NONE',
            'price_direction': 'NEUTRAL',
            'obv_direction': 'NEUTRAL',
            'error': 'Insufficient data or calculation failed'
        }
    
    def volume_acceleration_index(self, lookback: int = 20) -> Dict[str, float]:
        """
        VROC - Volume Rate of Change (Acceleration Index)
        
        Formula: VROC = (Volume_current - Volume_MA) / Volume_MA × 100
        
        High VROC (>50%) = Institutional surge.
        Combined with price direction:
        - VROC > 50% + Price UP = Strong accumulation
        - VROC > 50% + Price DOWN = Strong distribution
        
        Args:
            lookback: Number of days (default 20)
            
        Returns:
            dict:
                VROC: Raw rate of change (%)
                VROC_score: 0-100 score
                direction: BULLISH/BEARISH
                vol_trend: ACCELERATING/DECELERATING/NORMAL
                
        Example:
            >>> vroc = signals.volume_acceleration_index(lookback=20)
            >>> if vroc['vol_trend'] == 'ACCELERATING':
            >>>     print(f"Volume surge detected: {vroc['VROC']:.1f}%")
        """
        try:
            data = self._get_data(interval="1d", period=f"{lookback+10}d")
            recent = data.tail(lookback).copy()
            
            if len(recent) < 10:
                return self._vroc_fallback()
            
            # Calculate moving averages
            vol_ma_short = recent['Volume'].rolling(5).mean()
            vol_ma_long = recent['Volume'].rolling(20).mean()
            
            # VROC: short-term volume vs long-term average (extract scalars)
            vroc = float(((vol_ma_short.iloc[-1] - vol_ma_long.iloc[-1]) / vol_ma_long.iloc[-1] * 100))
            
            # Directional volume (does surge happen on up or down days?)
            price_change = recent['Close'].diff()
            up_days = recent[price_change > 0]
            down_days = recent[price_change < 0]
            
            avg_up_vol = float(up_days['Volume'].mean()) if len(up_days) > 0 else 0.0
            avg_down_vol = float(down_days['Volume'].mean()) if len(down_days) > 0 else 0.0
            
            # Determine direction
            if avg_up_vol > avg_down_vol:
                direction = 'BULLISH'
                vroc_score = min(100.0, 50.0 + vroc)  # UP volume surge
            else:
                direction = 'BEARISH'
                vroc_score = max(0.0, 50.0 + vroc)  # DOWN volume surge
            
            # Volume trend classification
            if vroc > 30:
                vol_trend = 'ACCELERATING'
            elif vroc < -30:
                vol_trend = 'DECELERATING'
            else:
                vol_trend = 'NORMAL'
            
            logger.info(f"VROC for {self.ticker}: {vroc:.1f}%, trend={vol_trend}, {direction}")
            
            return {
                'VROC': float(vroc),
                'VROC_score': max(0, min(100, float(vroc_score))),
                'direction': direction,
                'vol_trend': vol_trend,
                'lookback_days': lookback
            }
        
        except Exception as e:
            logger.error(f"VROC calculation failed for {self.ticker}: {e}")
            return self._vroc_fallback()
    
    def _vroc_fallback(self) -> Dict[str, float]:
        """Fallback values when VROC calculation fails."""
        return {
            'VROC': 0.0,
            'VROC_score': 50.0,
            'direction': 'NEUTRAL',
            'vol_trend': 'NORMAL',
            'error': 'Insufficient data or calculation failed'
        }
    
    def smart_money_index(self, lookback: int = 20) -> Dict[str, float]:
        """
        SMI - Smart Money Index (Composite Institutional Activity Score)
        
        Weighted average of all 4 indicators:
        - IFI: 30% (most direct institutional signal)
        - A/D: 25% (classic accumulation metric)
        - OBV: 25% (divergence detection)
        - VROC: 20% (acceleration/urgency)
        
        Adjustments:
        - Boost for bullish divergence (+10 points)
        - Reduce for bearish divergence (-10 points)
        - Consistency score (0-1): How aligned are all indicators?
        
        Args:
            lookback: Number of days (default 20)
            
        Returns:
            dict:
                SMI: Composite score (0-100)
                signal: STRONG_BUY/BUY/NEUTRAL/SELL/STRONG_SELL
                consistency: Alignment of indicators (0-1)
                confidence: Confidence in signal (0-100)
                components: Individual scores (IFI, AD, OBV, VROC)
                
        Example:
            >>> smi = signals.smart_money_index(lookback=20)
            >>> print(f"SMI: {smi['SMI']:.1f}/100 - {smi['signal']}")
            >>> print(f"Confidence: {smi['confidence']:.0f}%")
        """
        try:
            # Calculate all components
            ifi_result = self.institutional_flow_index(days=lookback)
            ad_result = self.accumulation_distribution(lookback=lookback)
            obv_result = self.obv_institutional(lookback=lookback)
            vroc_result = self.volume_acceleration_index(lookback=lookback)
            
            # Extract scores
            ifi_score = ifi_result['IFI_score']
            ad_score = ad_result['AD_score']
            obv_score = obv_result['OBV_score']
            vroc_score = vroc_result['VROC_score']
            
            # Weighted composite (research-backed weights)
            weights = {
                'IFI': 0.30,
                'AD': 0.25,
                'OBV': 0.25,
                'VROC': 0.20
            }
            
            smi = (
                ifi_score * weights['IFI'] +
                ad_score * weights['AD'] +
                obv_score * weights['OBV'] +
                vroc_score * weights['VROC']
            )
            
            # Adjust for divergences
            if obv_result['divergence'] == 'BULLISH':
                smi = min(100, smi + 10)
            elif obv_result['divergence'] == 'BEARISH':
                smi = max(0, smi - 10)
            
            # Calculate consistency (how aligned are signals?)
            bullish_signals = sum([
                ifi_result['interpretation'] == 'BULLISH',
                ad_result['signal'] == 'ACCUMULATION',
                obv_result['obv_direction'] == 'UP',
                vroc_result['direction'] == 'BULLISH'
            ])
            consistency = bullish_signals / 4.0  # 0-1 score
            
            # Signal classification
            if smi > 75:
                signal = 'STRONG_BUY'
            elif smi > 60:
                signal = 'BUY'
            elif smi > 40:
                signal = 'NEUTRAL'
            elif smi > 25:
                signal = 'SELL'
            else:
                signal = 'STRONG_SELL'
            
            confidence = consistency * 100  # Convert to percentage
            
            logger.info(f"SMI for {self.ticker}: {smi:.1f}/100, {signal}, confidence={confidence:.0f}%")
            
            return {
                'SMI': float(smi),
                'signal': signal,
                'consistency': float(consistency),
                'confidence': float(confidence),
                'components': {
                    'IFI': float(ifi_score),
                    'AD': float(ad_score),
                    'OBV': float(obv_score),
                    'VROC': float(vroc_score)
                },
                'divergence': obv_result['divergence'],
                'lookback_days': lookback
            }
        
        except Exception as e:
            logger.error(f"SMI calculation failed for {self.ticker}: {e}")
            return self._smi_fallback()
    
    def _smi_fallback(self) -> Dict[str, float]:
        """Fallback values when SMI calculation fails."""
        return {
            'SMI': 50.0,
            'signal': 'NEUTRAL',
            'consistency': 0.5,
            'confidence': 50.0,
            'components': {
                'IFI': 50.0,
                'AD': 50.0,
                'OBV': 50.0,
                'VROC': 50.0
            },
            'divergence': 'NONE',
            'error': 'Calculation failed'
        }
    
    def get_all_signals(self, lookback: int = 20) -> Dict[str, Dict]:
        """
        Convenience method: Get all 5 signals at once.
        
        Args:
            lookback: Number of days (default 20)
            
        Returns:
            dict with keys: IFI, AD, OBV, VROC, SMI (each containing full result dict)
            
        Example:
            >>> all_signals = signals.get_all_signals(lookback=20)
            >>> print(f"SMI: {all_signals['SMI']['SMI']:.1f}/100")
            >>> print(f"IFI: {all_signals['IFI']['interpretation']}")
        """
        return {
            'IFI': self.institutional_flow_index(days=lookback),
            'AD': self.accumulation_distribution(lookback=lookback),
            'OBV': self.obv_institutional(lookback=lookback),
            'VROC': self.volume_acceleration_index(lookback=lookback),
            'SMI': self.smart_money_index(lookback=lookback)
        }


# ========================================
# EXAMPLE USAGE & VALIDATION
# ========================================

if __name__ == "__main__":
    # Test on known institutional accumulation (NVDA Q1 2023)
    print("=" * 80)
    print("DARK POOL SIGNALS - MODULE 1 VALIDATION")
    print("=" * 80)
    
    ticker = "NVDA"
    print(f"\nTesting {ticker} (known AI leader with institutional interest)...\n")
    
    signals = DarkPoolSignals(ticker)
    
    # Get all signals
    all_signals = signals.get_all_signals(lookback=20)
    
    # Display results
    print(f"1. INSTITUTIONAL FLOW INDEX (IFI):")
    ifi = all_signals['IFI']
    print(f"   Score: {ifi['IFI_score']:.1f}/100")
    print(f"   Interpretation: {ifi['interpretation']}")
    print(f"   Raw IFI: {ifi['IFI']:.4f}")
    
    print(f"\n2. ACCUMULATION/DISTRIBUTION (A/D):")
    ad = all_signals['AD']
    print(f"   Score: {ad['AD_score']:.1f}/100")
    print(f"   Signal: {ad['signal']}")
    print(f"   5-day Trend: {ad['AD_trend']:.2e}")
    
    print(f"\n3. ON-BALANCE VOLUME (OBV):")
    obv = all_signals['OBV']
    print(f"   Score: {obv['OBV_score']:.1f}/100")
    print(f"   Price Direction: {obv['price_direction']}")
    print(f"   OBV Direction: {obv['obv_direction']}")
    if obv['divergence'] != 'NONE':
        print(f"   ⚠️ DIVERGENCE: {obv['divergence']}")
    
    print(f"\n4. VOLUME RATE OF CHANGE (VROC):")
    vroc = all_signals['VROC']
    print(f"   Score: {vroc['VROC_score']:.1f}/100")
    print(f"   Direction: {vroc['direction']}")
    print(f"   Volume Trend: {vroc['vol_trend']}")
    print(f"   Raw VROC: {vroc['VROC']:.1f}%")
    
    print(f"\n5. SMART MONEY INDEX (SMI) - COMPOSITE:")
    smi = all_signals['SMI']
    print(f"   SMI Score: {smi['SMI']:.1f}/100")
    print(f"   Signal: {smi['signal']}")
    print(f"   Confidence: {smi['confidence']:.0f}%")
    print(f"   Consistency: {smi['consistency']:.2f}")
    print(f"   Components:")
    for component, score in smi['components'].items():
        print(f"      {component}: {score:.1f}/100")
    
    print("\n" + "=" * 80)
    print(f"FINAL RECOMMENDATION: {smi['signal']}")
    print(f"CONFIDENCE LEVEL: {smi['confidence']:.0f}%")
    print("=" * 80)
    
    # Integration example
    print("\n\nINTEGRATION EXAMPLE:")
    print("```python")
    print("# In your meta-learner:")
    print("from src.features.dark_pool_signals import DarkPoolSignals")
    print("")
    print("signals = DarkPoolSignals('NVDA')")
    print("smi_result = signals.smart_money_index(lookback=20)")
    print("")
    print("# Use SMI score (0-100) as a feature in your ensemble")
    print("institutional_signal = smi_result['SMI']  # Feed to meta-learner")
    print("confidence_boost = smi_result['consistency']  # Use to adjust position size")
    print("```")
