"""
🎯 INTEGRATED PATTERN BASELINE SCORER
Combines existing pattern_detector.py with real win rates from pattern_battle_results.json

Real Win Rates (From Battle Tests):
- Test Set Overall: 64.58% WR
- nuclear_dip: 82.35% WR
- ribbon_mom: 71.43% WR  
- dip_buy: 71.43% WR
- bounce: 66.10% WR
- quantum_mom: 65.63% WR
- trend_cont: 62.96% WR
- squeeze: 50% WR
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime, timedelta
import json

# Import existing pattern detector
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

try:
    from pattern_detector import PatternDetector
    PATTERN_DETECTOR_AVAILABLE = True
except ImportError:
    PATTERN_DETECTOR_AVAILABLE = False
    print("⚠️  pattern_detector.py not found - pattern detection will be limited")


class PatternBaselineScorer:
    """
    Pattern confidence scorer using REAL win rates from battle tests.
    
    Uses actual tested performance, not theoretical assumptions.
    """
    
    # Real win rates from pattern_battle_results.json (TEST SET)
    PATTERN_WIN_RATES = {
        # AI Patterns (from battle results)
        'nuclear_dip': 0.8235,        # 82.35% WR - BEST PATTERN
        'ribbon_mom': 0.7143,         # 71.43% WR - EMA ribbon momentum
        'dip_buy': 0.7143,            # 71.43% WR - RSI oversold bounce
        'bounce': 0.6610,             # 66.10% WR - general bounce
        'quantum_mom': 0.6563,        # 65.63% WR - quantum momentum
        'trend_cont': 0.6296,         # 62.96% WR - trend continuation
        'squeeze': 0.5000,            # 50% WR - AVOID (coin flip)
        
        # TA-Lib Patterns (conservative estimates based on research)
        'CDL_MORNINGSTAR': 0.68,
        'CDL_EVENINGSTAR': 0.65,
        'CDL_ENGULFING': 0.62,
        'CDL_HAMMER': 0.60,
        'CDL_SHOOTINGSTAR': 0.60,
        'CDL_PIERCING': 0.61,
        'CDL_DARKCLOUDCOVER': 0.61,
        'CDL_3WHITESOLDIERS': 0.64,
        'CDL_3BLACKCROWS': 0.64,
        'CDL_DOJI': 0.55,
        'CDL_HARAMI': 0.58,
        'CDL_INVERTEDHAMMER': 0.59,
        'CDL_HANGINGMAN': 0.59,
        
        # Custom patterns (from existing pattern_detector.py)
        'EMA_RIBBON_BULLISH': 0.7143,  # Same as ribbon_mom
        'EMA_RIBBON_BEARISH': 0.7143,  # Same as ribbon_mom
        'VWAP_PULLBACK_BULLISH': 0.66,
        'VWAP_PULLBACK_BEARISH': 0.66,
        'ORB_BULLISH': 0.62,
        'ORB_BEARISH': 0.62,
        
        # Default for unknown patterns
        'DEFAULT': 0.55
    }
    
    # Pattern name mapping (TA-Lib -> readable)
    PATTERN_NAME_MAP = {
        'CDLMORNINGSTAR': 'CDL_MORNINGSTAR',
        'CDLEVENINGSTAR': 'CDL_EVENINGSTAR',
        'CDLENGULFING': 'CDL_ENGULFING',
        'CDLHAMMER': 'CDL_HAMMER',
        'CDLSHOOTINGSTAR': 'CDL_SHOOTINGSTAR',
        'CDLPIERCING': 'CDL_PIERCING',
        'CDLDARKCLOUDCOVER': 'CDL_DARKCLOUDCOVER',
        'CDL3WHITESOLDIERS': 'CDL_3WHITESOLDIERS',
        'CDL3BLACKCROWS': 'CDL_3BLACKCROWS',
        'CDLDOJI': 'CDL_DOJI',
        'CDLHARAMI': 'CDL_HARAMI',
        'CDLINVERTEDHAMMER': 'CDL_INVERTEDHAMMER',
        'CDLHANGINGMAN': 'CDL_HANGINGMAN',
    }
    
    def __init__(self):
        """Initialize with pattern detector if available."""
        self.pattern_detector = PatternDetector() if PATTERN_DETECTOR_AVAILABLE else None
    
    def get_pattern_confidence(self, pattern_name: str, pattern_type: str = 'BULLISH') -> float:
        """
        Get confidence score (win rate) for a pattern.
        
        Args:
            pattern_name: Pattern name (e.g., 'nuclear_dip', 'CDL_MORNINGSTAR')
            pattern_type: 'BULLISH' or 'BEARISH'
            
        Returns:
            Confidence score (0.0 to 1.0)
        """
        # Normalize pattern name
        pattern_key = pattern_name.upper().replace(' ', '_')
        
        # Try direct lookup
        if pattern_key in self.PATTERN_WIN_RATES:
            return self.PATTERN_WIN_RATES[pattern_key]
        
        # Try with mapping
        if pattern_key in self.PATTERN_NAME_MAP:
            mapped_key = self.PATTERN_NAME_MAP[pattern_key]
            if mapped_key in self.PATTERN_WIN_RATES:
                return self.PATTERN_WIN_RATES[mapped_key]
        
        # Try converting from readable format (e.g., "Morning Star" -> "CDL_MORNINGSTAR")
        readable_to_key = pattern_name.upper().replace(' ', '')
        if readable_to_key in self.PATTERN_NAME_MAP:
            mapped_key = self.PATTERN_NAME_MAP[readable_to_key]
            if mapped_key in self.PATTERN_WIN_RATES:
                return self.PATTERN_WIN_RATES[mapped_key]
        
        # Default confidence
        return self.PATTERN_WIN_RATES['DEFAULT']
    
    def score_pattern(self, pattern: Dict) -> Dict:
        """
        Score a pattern detected by pattern_detector.py.
        
        Args:
            pattern: Pattern dict from PatternDetector.detect_all_patterns()
            
        Returns:
            Pattern dict with updated confidence score
        """
        pattern_name = pattern.get('pattern', 'UNKNOWN')
        pattern_type = pattern.get('type', 'BULLISH')
        
        # Get baseline confidence from win rate
        baseline_confidence = self.get_pattern_confidence(pattern_name, pattern_type)
        
        # Combine with pattern's internal confidence (if available)
        internal_confidence = pattern.get('confidence', 0.5)
        
        # Weighted average: 70% baseline (real win rate), 30% internal
        final_confidence = (baseline_confidence * 0.7) + (internal_confidence * 0.3)
        
        # Update pattern
        pattern['confidence'] = final_confidence
        pattern['baseline_wr'] = baseline_confidence  # Add baseline for transparency
        pattern['internal_confidence'] = internal_confidence
        
        return pattern
    
    def detect_and_score_patterns(self, ticker: str, period: str = '60d', 
                                   interval: str = '1d') -> Dict:
        """
        Detect patterns using existing pattern_detector.py and score them.
        
        Args:
            ticker: Stock ticker (e.g., 'AAPL')
            period: Data period (e.g., '60d', '1y')
            interval: Data interval (e.g., '1d', '1h')
            
        Returns:
            Dict with patterns and scores
        """
        if not PATTERN_DETECTOR_AVAILABLE:
            return {
                'ticker': ticker,
                'error': 'PatternDetector not available',
                'patterns': []
            }
        
        # Detect patterns using existing system
        result = self.pattern_detector.detect_all_patterns(ticker, period, interval)
        
        # Score each pattern with real win rates
        scored_patterns = []
        for pattern in result.get('patterns', []):
            scored_pattern = self.score_pattern(pattern)
            scored_patterns.append(scored_pattern)
        
        # Sort by confidence (highest first)
        scored_patterns.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Update result
        result['patterns'] = scored_patterns
        
        # Add scoring metadata
        result['scoring'] = {
            'method': 'real_win_rates',
            'source': 'pattern_battle_results.json',
            'test_wr': 0.6458,  # Overall test win rate
            'train_wr': 0.51,   # Overall train win rate
            'patterns_scored': len(scored_patterns),
            'high_confidence_count': len([p for p in scored_patterns if p['confidence'] > 0.7])
        }
        
        return result
    
    def get_top_patterns(self, ticker: str, top_n: int = 5, 
                        min_confidence: float = 0.65) -> List[Dict]:
        """
        Get top N highest-confidence patterns for a ticker.
        
        Args:
            ticker: Stock ticker
            top_n: Number of top patterns to return
            min_confidence: Minimum confidence threshold (e.g., 0.65 for 65% WR)
            
        Returns:
            List of top patterns
        """
        result = self.detect_and_score_patterns(ticker)
        
        # Filter by minimum confidence
        high_confidence = [
            p for p in result.get('patterns', [])
            if p['confidence'] >= min_confidence
        ]
        
        # Return top N
        return high_confidence[:top_n]


# Example usage
if __name__ == '__main__':
    scorer = PatternBaselineScorer()
    
    # Test on a ticker
    ticker = 'AAPL'
    print(f"\n🎯 Detecting and scoring patterns for {ticker}...")
    
    result = scorer.detect_and_score_patterns(ticker, period='60d')
    
    print(f"\n📊 Results:")
    print(f"  Total patterns: {len(result.get('patterns', []))}")
    scoring = result.get('scoring', {})
    if scoring:
        print(f"  Scoring method: {scoring.get('method')}")
        print(f"  Test WR: {scoring.get('test_wr', 0):.2%}")
    else:
        print(f"  Scoring: Not available (pattern detector not found)")
    
    # Show top 5 patterns
    top_patterns = scorer.get_top_patterns(ticker, top_n=5, min_confidence=0.65)
    
    print(f"\n🔥 Top Patterns (confidence ≥ 65%):")
    for i, pattern in enumerate(top_patterns, 1):
        print(f"\n  {i}. {pattern['pattern']} ({pattern['type']})")
        print(f"     Final Confidence: {pattern['confidence']:.2%}")
        print(f"     Baseline WR: {pattern['baseline_wr']:.2%}")
        print(f"     Internal Confidence: {pattern['internal_confidence']:.2%}")
        print(f"     Price Level: ${pattern['price_level']:.2f}")
        print(f"     Date: {pattern['timestamp']}")
    
    # Test individual pattern lookups
    print(f"\n\n🧪 Testing Pattern Win Rate Lookups:")
    test_patterns = ['nuclear_dip', 'ribbon_mom', 'dip_buy', 'bounce', 'CDL_MORNINGSTAR']
    for pattern in test_patterns:
        wr = scorer.get_pattern_confidence(pattern)
        print(f"  {pattern}: {wr:.2%}")
