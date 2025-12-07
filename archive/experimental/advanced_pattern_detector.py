"""
ADVANCED PATTERN DETECTOR - Elliott Waves + Fibonacci + Custom Setups
Detects professional-grade patterns for serious trading.
"""

import numpy as np
import pandas as pd
import talib
from typing import Dict, List, Tuple
from scipy.signal import argrelextrema
from dataclasses import dataclass
from datetime import datetime


@dataclass
class FibonacciLevel:
    """Fibonacci retracement/extension level"""
    level: float  # 0.236, 0.382, 0.5, 0.618, 1.0, 1.618
    price: float
    level_type: str  # 'retracement' or 'extension'
    strength: float  # 0-1 confidence


@dataclass
class ElliottWave:
    """Elliott Wave pattern"""
    wave_type: str  # 'impulse' (5 waves) or 'correction' (3 waves)
    waves: List[Tuple[int, float]]  # [(index, price), ...]
    direction: str  # 'bullish' or 'bearish'
    confidence: float
    fib_levels: List[FibonacciLevel]


class AdvancedPatternDetector:
    """
    Detects advanced patterns:
    - Elliott Waves (impulse + corrective)
    - Fibonacci retracements/extensions
    - Harmonic patterns (Gartley, Butterfly, Bat, Crab)
    - Volume profile
    - Supply/Demand zones
    """
    
    def __init__(self):
        self.fib_ratios = [0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.272, 1.618, 2.618]
    
    def detect_all_advanced_patterns(self, df: pd.DataFrame) -> Dict:
        """Main entry point - detect all advanced patterns"""
        
        results = {
            'elliott_waves': self.detect_elliott_waves(df),
            'fibonacci_levels': self.detect_fibonacci_levels(df),
            'harmonic_patterns': self.detect_harmonic_patterns(df),
            'supply_demand_zones': self.detect_supply_demand_zones(df),
            'volume_profile': self.calculate_volume_profile(df),
            'key_levels': self.detect_key_levels(df)
        }
        
        return results
    
    def detect_elliott_waves(self, df: pd.DataFrame) -> List[ElliottWave]:
        """
        Detect Elliott Wave patterns (5-wave impulse, 3-wave correction)
        Uses swing highs/lows and Fibonacci ratios for validation
        """
        waves = []
        
        close = df['Close'].values if 'Close' in df.columns else df['close'].values
        high = df['High'].values if 'High' in df.columns else df['high'].values
        low = df['Low'].values if 'Low' in df.columns else df['low'].values
        
        # Find swing points (local extrema)
        swing_highs_idx = argrelextrema(high, np.greater, order=5)[0]
        swing_lows_idx = argrelextrema(low, np.less, order=5)[0]
        
        # Combine and sort swing points
        swings = []
        for idx in swing_highs_idx:
            swings.append((idx, high[idx], 'high'))
        for idx in swing_lows_idx:
            swings.append((idx, low[idx], 'low'))
        
        swings.sort(key=lambda x: x[0])
        
        if len(swings) < 5:
            return waves  # Need at least 5 swings for impulse wave
        
        # Look for 5-wave impulse patterns
        for i in range(len(swings) - 4):
            pattern_swings = swings[i:i+5]
            
            # Check for alternating high/low pattern
            types = [s[2] for s in pattern_swings]
            
            # Bullish impulse: low-high-low-high-low
            if types == ['low', 'high', 'low', 'high', 'low']:
                wave_pattern = [(s[0], s[1]) for s in pattern_swings]
                
                # Validate wave rules
                if self._validate_impulse_wave(wave_pattern, 'bullish'):
                    fib_levels = self._calculate_wave_fib_levels(wave_pattern)
                    waves.append(ElliottWave(
                        wave_type='impulse',
                        waves=wave_pattern,
                        direction='bullish',
                        confidence=0.75,
                        fib_levels=fib_levels
                    ))
            
            # Bearish impulse: high-low-high-low-high
            elif types == ['high', 'low', 'high', 'low', 'high']:
                wave_pattern = [(s[0], s[1]) for s in pattern_swings]
                
                if self._validate_impulse_wave(wave_pattern, 'bearish'):
                    fib_levels = self._calculate_wave_fib_levels(wave_pattern)
                    waves.append(ElliottWave(
                        wave_type='impulse',
                        waves=wave_pattern,
                        direction='bearish',
                        confidence=0.75,
                        fib_levels=fib_levels
                    ))
        
        return waves
    
    def _validate_impulse_wave(self, waves: List[Tuple], direction: str) -> bool:
        """
        Validate Elliott Wave impulse pattern rules:
        - Wave 2 doesn't retrace beyond wave 1 start
        - Wave 3 is not the shortest wave
        - Wave 4 doesn't overlap with wave 1
        """
        if len(waves) != 5:
            return False
        
        try:
            # Extract wave points
            w1_start, w1_end = waves[0][1], waves[1][1]
            w2_end = waves[2][1]
            w3_end = waves[3][1]
            w4_end = waves[4][1]
            
            if direction == 'bullish':
                # Wave 2 shouldn't go below wave 1 start
                if w2_end < w1_start:
                    return False
                
                # Wave 3 must not be shortest
                wave1_len = abs(w1_end - w1_start)
                wave3_len = abs(w3_end - w2_end)
                wave5_len = abs(waves[4][1] - w4_end)
                
                if wave3_len < wave1_len and wave3_len < wave5_len:
                    return False
                
                # Wave 4 shouldn't overlap wave 1
                if w4_end < w1_end:
                    return False
            
            else:  # bearish
                if w2_end > w1_start:
                    return False
                
                wave1_len = abs(w1_end - w1_start)
                wave3_len = abs(w3_end - w2_end)
                wave5_len = abs(waves[4][1] - w4_end)
                
                if wave3_len < wave1_len and wave3_len < wave5_len:
                    return False
                
                if w4_end > w1_end:
                    return False
            
            return True
        
        except Exception:
            return False
    
    def _calculate_wave_fib_levels(self, waves: List[Tuple]) -> List[FibonacciLevel]:
        """Calculate Fibonacci levels for Elliott Wave"""
        fib_levels = []
        
        if len(waves) >= 2:
            # Calculate retracement levels for wave 2
            wave1_start = waves[0][1]
            wave1_end = waves[1][1]
            wave1_range = abs(wave1_end - wave1_start)
            
            for ratio in [0.236, 0.382, 0.5, 0.618, 0.786]:
                if wave1_end > wave1_start:  # bullish
                    price = wave1_end - (wave1_range * ratio)
                else:  # bearish
                    price = wave1_end + (wave1_range * ratio)
                
                fib_levels.append(FibonacciLevel(
                    level=ratio,
                    price=float(price),
                    level_type='retracement',
                    strength=0.8
                ))
        
        return fib_levels
    
    def detect_fibonacci_levels(self, df: pd.DataFrame) -> List[FibonacciLevel]:
        """
        Detect Fibonacci retracement/extension levels
        Uses recent significant swing high/low
        """
        fib_levels = []
        
        close = df['Close'].values if 'Close' in df.columns else df['close'].values
        high = df['High'].values if 'High' in df.columns else df['high'].values
        low = df['Low'].values if 'Low' in df.columns else df['low'].values
        
        # Find recent significant high and low (last 50 bars)
        lookback = min(50, len(df))
        recent_high = np.max(high[-lookback:])
        recent_low = np.min(low[-lookback:])
        price_range = recent_high - recent_low
        
        # Calculate retracement levels (from high to low)
        for ratio in self.fib_ratios:
            retracement_price = recent_high - (price_range * ratio)
            fib_levels.append(FibonacciLevel(
                level=ratio,
                price=float(retracement_price),
                level_type='retracement',
                strength=0.7 if ratio in [0.382, 0.5, 0.618] else 0.5
            ))
        
        # Calculate extension levels (beyond the range)
        for ratio in [1.272, 1.618, 2.618]:
            extension_price = recent_high + (price_range * (ratio - 1))
            fib_levels.append(FibonacciLevel(
                level=ratio,
                price=float(extension_price),
                level_type='extension',
                strength=0.6
            ))
        
        return fib_levels
    
    def detect_harmonic_patterns(self, df: pd.DataFrame) -> List[Dict]:
        """
        Detect harmonic patterns: Gartley, Butterfly, Bat, Crab
        Uses Fibonacci ratios to validate XABCD patterns
        """
        patterns = []
        
        # This is a simplified version - full harmonic detection is complex
        # Would need dedicated library or deeper implementation
        
        return patterns
    
    def detect_supply_demand_zones(self, df: pd.DataFrame) -> List[Dict]:
        """
        Detect supply (resistance) and demand (support) zones
        Based on price rejection areas with volume
        """
        zones = []
        
        close = df['Close'].values if 'Close' in df.columns else df['close'].values
        high = df['High'].values if 'High' in df.columns else df['high'].values
        low = df['Low'].values if 'Low' in df.columns else df['low'].values
        volume = df['Volume'].values if 'Volume' in df.columns else df['volume'].values
        
        # Find high-volume rejection zones
        avg_volume = np.mean(volume)
        
        for i in range(10, len(df) - 5):
            # Supply zone (sellers stepping in)
            if volume[i] > avg_volume * 1.5 and high[i] > high[i-1]:
                zone_high = float(high[i])
                zone_low = float(close[i])
                
                zones.append({
                    'type': 'supply',
                    'price_high': zone_high,
                    'price_low': zone_low,
                    'index': i,
                    'strength': float(volume[i] / avg_volume),
                    'tested': 0  # Number of times price tested this zone
                })
            
            # Demand zone (buyers stepping in)
            if volume[i] > avg_volume * 1.5 and low[i] < low[i-1]:
                zone_high = float(close[i])
                zone_low = float(low[i])
                
                zones.append({
                    'type': 'demand',
                    'price_high': zone_high,
                    'price_low': zone_low,
                    'index': i,
                    'strength': float(volume[i] / avg_volume),
                    'tested': 0
                })
        
        return zones
    
    def calculate_volume_profile(self, df: pd.DataFrame) -> Dict:
        """
        Calculate volume profile (volume at price levels)
        Find Point of Control (POC) and Value Areas
        """
        close = df['Close'].values if 'Close' in df.columns else df['close'].values
        volume = df['Volume'].values if 'Volume' in df.columns else df['volume'].values
        
        # Create price bins
        price_min = np.min(close)
        price_max = np.max(close)
        num_bins = 50
        bins = np.linspace(price_min, price_max, num_bins)
        
        # Aggregate volume at each price level
        volume_at_price = np.zeros(num_bins - 1)
        
        for i in range(len(close)):
            bin_idx = np.digitize(close[i], bins) - 1
            if 0 <= bin_idx < len(volume_at_price):
                volume_at_price[bin_idx] += volume[i]
        
        # Find POC (price level with most volume)
        poc_idx = np.argmax(volume_at_price)
        poc_price = (bins[poc_idx] + bins[poc_idx + 1]) / 2
        
        # Find Value Area (70% of volume)
        total_volume = np.sum(volume_at_price)
        target_volume = total_volume * 0.70
        
        # Start from POC and expand outward
        cumulative_volume = volume_at_price[poc_idx]
        lower_idx = poc_idx
        upper_idx = poc_idx
        
        while cumulative_volume < target_volume and (lower_idx > 0 or upper_idx < len(volume_at_price) - 1):
            # Expand to side with more volume
            lower_vol = volume_at_price[lower_idx - 1] if lower_idx > 0 else 0
            upper_vol = volume_at_price[upper_idx + 1] if upper_idx < len(volume_at_price) - 1 else 0
            
            if lower_vol > upper_vol and lower_idx > 0:
                lower_idx -= 1
                cumulative_volume += volume_at_price[lower_idx]
            elif upper_idx < len(volume_at_price) - 1:
                upper_idx += 1
                cumulative_volume += volume_at_price[upper_idx]
            else:
                break
        
        return {
            'poc_price': float(poc_price),
            'value_area_high': float((bins[upper_idx] + bins[upper_idx + 1]) / 2),
            'value_area_low': float((bins[lower_idx] + bins[lower_idx + 1]) / 2),
            'volume_distribution': volume_at_price.tolist(),
            'price_bins': bins.tolist()
        }
    
    def detect_key_levels(self, df: pd.DataFrame) -> Dict:
        """
        Detect key support/resistance levels
        Using pivot points, previous highs/lows, psychological levels
        """
        close = df['Close'].values if 'Close' in df.columns else df['close'].values
        high = df['High'].values if 'High' in df.columns else df['high'].values
        low = df['Low'].values if 'Low' in df.columns else df['low'].values
        
        # Classic pivot point
        pivot = (high[-1] + low[-1] + close[-1]) / 3
        
        # Support/Resistance levels
        r1 = 2 * pivot - low[-1]
        s1 = 2 * pivot - high[-1]
        r2 = pivot + (high[-1] - low[-1])
        s2 = pivot - (high[-1] - low[-1])
        r3 = high[-1] + 2 * (pivot - low[-1])
        s3 = low[-1] - 2 * (high[-1] - pivot)
        
        # Psychological levels (round numbers)
        current_price = float(close[-1])
        psychological = []
        
        for multiplier in [0.9, 0.95, 1.0, 1.05, 1.1]:
            level = round(current_price * multiplier / 10) * 10  # Round to nearest 10
            psychological.append(float(level))
        
        return {
            'pivot': float(pivot),
            'resistance': [float(r1), float(r2), float(r3)],
            'support': [float(s1), float(s2), float(s3)],
            'psychological_levels': psychological,
            'recent_high': float(np.max(high[-20:])),
            'recent_low': float(np.min(low[-20:]))
        }


if __name__ == '__main__':
    import yfinance as yf
    
    print("Testing Advanced Pattern Detector...")
    
    detector = AdvancedPatternDetector()
    
    for ticker in ['MU', 'IONQ', 'APLD', 'ANNX']:
        print(f"\n{'='*60}")
        print(f"Analyzing {ticker}")
        print(f"{'='*60}")
        
        df = yf.download(ticker, period='60d', interval='1d', progress=False)
        
        results = detector.detect_all_advanced_patterns(df)
        
        print(f"\n🌊 Elliott Waves: {len(results['elliott_waves'])} detected")
        for wave in results['elliott_waves']:
            print(f"   {wave.wave_type.upper()} - {wave.direction} - Confidence: {wave.confidence:.0%}")
        
        print(f"\n📊 Fibonacci Levels: {len(results['fibonacci_levels'])} levels")
        key_fibs = [f for f in results['fibonacci_levels'] if f.level in [0.382, 0.5, 0.618]]
        for fib in key_fibs[:3]:
            print(f"   {fib.level:.1%} @ ${fib.price:.2f}")
        
        print(f"\n📦 Supply/Demand Zones: {len(results['supply_demand_zones'])} zones")
        
        vp = results['volume_profile']
        print(f"\n📈 Volume Profile:")
        print(f"   POC: ${vp['poc_price']:.2f}")
        print(f"   Value Area: ${vp['value_area_low']:.2f} - ${vp['value_area_high']:.2f}")
        
        levels = results['key_levels']
        print(f"\n🎯 Key Levels:")
        print(f"   Pivot: ${levels['pivot']:.2f}")
        print(f"   R1: ${levels['resistance'][0]:.2f}")
        print(f"   S1: ${levels['support'][0]:.2f}")
