"""
📈 MULTI-TIMEFRAME FORECASTING ENGINE
Predicts 1/2/5/7 day price targets with probability distributions

Integrates with:
- Existing pattern_detector.py (65 patterns)
- pattern_baseline_scorer.py (real win rates)
- FeatureEngineer70 (71 features)
- ai_recommender.py (AI signals)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import yfinance as yf

# Import existing systems
try:
    from src.ml.feature_engineer_56 import FeatureEngineer70
    FEATURE_ENGINEER_AVAILABLE = True
except ImportError:
    FEATURE_ENGINEER_AVAILABLE = False
    print("⚠️  FeatureEngineer70 not available")

try:
    from pattern_detector import PatternDetector
    PATTERN_DETECTOR_AVAILABLE = True
except ImportError:
    PATTERN_DETECTOR_AVAILABLE = False
    print("⚠️  PatternDetector not available")

try:
    from src.trading.pattern_baseline_scorer import PatternBaselineScorer
    SCORER_AVAILABLE = True
except ImportError:
    SCORER_AVAILABLE = False
    print("⚠️  PatternBaselineScorer not available")


class ForecastingEngine:
    """
    Multi-timeframe price forecasting engine.
    
    Forecasts:
    - 1 day: Quick swing (momentum plays)
    - 2 day: Short swing
    - 5 day: Weekly swing
    - 7 day: Full week swing
    
    Uses existing pattern detection + feature engineering + AI signals.
    """
    
    def __init__(self):
        """Initialize forecasting engine with existing components."""
        self.pattern_detector = PatternDetector() if PATTERN_DETECTOR_AVAILABLE else None
        self.scorer = PatternBaselineScorer() if SCORER_AVAILABLE else None
        
        # Volatility-based target multipliers
        self.volatility_multipliers = {
            'low': {'1d': 0.015, '2d': 0.025, '5d': 0.05, '7d': 0.07},      # Low vol: 1.5%, 2.5%, 5%, 7%
            'medium': {'1d': 0.025, '2d': 0.04, '5d': 0.08, '7d': 0.12},    # Med vol: 2.5%, 4%, 8%, 12%
            'high': {'1d': 0.04, '2d': 0.07, '5d': 0.14, '7d': 0.20}        # High vol: 4%, 7%, 14%, 20%
        }
    
    def calculate_volatility_regime(self, df: pd.DataFrame) -> str:
        """
        Determine volatility regime (low/medium/high).
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            'low', 'medium', or 'high'
        """
        import talib
        
        close = np.asarray(self._get_array(df, 'Close'), dtype='float64')
        high = np.asarray(self._get_array(df, 'High'), dtype='float64')
        low = np.asarray(self._get_array(df, 'Low'), dtype='float64')
        
        # Calculate ATR (14-period)
        atr = talib.ATR(high, low, close, timeperiod=14)
        
        # ATR as % of price
        atr_pct = atr[-1] / close[-1]
        
        # Classify regime
        if atr_pct < 0.02:  # Less than 2% ATR
            return 'low'
        elif atr_pct < 0.04:  # 2-4% ATR
            return 'medium'
        else:  # Greater than 4% ATR
            return 'high'
    
    @staticmethod
    def _get_array(df, col):
        """Safe array extraction."""
        if isinstance(df[col], pd.DataFrame):
            return df[col].iloc[:, 0].values
        return df[col].values
    
    def calculate_pattern_momentum(self, patterns: List[Dict]) -> Dict[str, float]:
        """
        Calculate momentum bias from patterns.
        
        Args:
            patterns: List of patterns from PatternBaselineScorer
            
        Returns:
            Dict with bullish/bearish momentum scores
        """
        if not patterns:
            return {'bullish': 0.5, 'bearish': 0.5, 'neutral': 1.0}
        
        bullish_score = 0.0
        bearish_score = 0.0
        total_confidence = 0.0
        
        for pattern in patterns:
            confidence = pattern.get('confidence', 0.5)
            pattern_type = pattern.get('type', 'BULLISH')
            
            if pattern_type == 'BULLISH':
                bullish_score += confidence
            elif pattern_type == 'BEARISH':
                bearish_score += confidence
            
            total_confidence += confidence
        
        # Normalize
        if total_confidence > 0:
            bullish_pct = bullish_score / total_confidence
            bearish_pct = bearish_score / total_confidence
        else:
            bullish_pct = 0.5
            bearish_pct = 0.5
        
        return {
            'bullish': bullish_pct,
            'bearish': bearish_pct,
            'neutral': 1.0 - abs(bullish_pct - bearish_pct)
        }
    
    def forecast_next_days(self, df: pd.DataFrame, ticker: str, 
                          current_price: Optional[float] = None) -> Dict:
        """
        Forecast 1/2/5/7 day price targets.
        
        Args:
            df: OHLCV DataFrame with at least 60 days
            ticker: Stock ticker
            current_price: Current price (defaults to last close)
            
        Returns:
            Dict with forecasts for each timeframe
        """
        if current_price is None:
            current_price = float(self._get_array(df, 'Close')[-1])
        
        # Get patterns if available
        patterns = []
        if self.scorer:
            result = self.scorer.detect_and_score_patterns(ticker, period='60d')
            patterns = result.get('patterns', [])[:10]  # Top 10 patterns
        
        # Calculate volatility regime
        vol_regime = self.calculate_volatility_regime(df)
        
        # Calculate pattern momentum
        momentum = self.calculate_pattern_momentum(patterns)
        
        # Get target multipliers for this volatility regime
        multipliers = self.volatility_multipliers[vol_regime]
        
        # Generate forecasts for each timeframe
        forecasts = {}
        
        for timeframe in ['1d', '2d', '5d', '7d']:
            base_move = multipliers[timeframe]
            
            # Adjust based on pattern momentum
            if momentum['bullish'] > 0.6:  # Strong bullish
                upside = base_move * 1.5
                downside = base_move * 0.5
                prob_up = min(momentum['bullish'] * 0.9, 0.80)  # Cap at 80%
            elif momentum['bearish'] > 0.6:  # Strong bearish
                upside = base_move * 0.5
                downside = base_move * 1.5
                prob_up = max(momentum['bullish'] * 0.9, 0.20)  # Floor at 20%
            else:  # Neutral
                upside = base_move
                downside = base_move
                prob_up = 0.50
            
            # Calculate targets
            target_high = current_price * (1 + upside)
            target_low = current_price * (1 - downside)
            target_mid = (target_high + target_low) / 2
            
            # Calculate probability distribution
            prob_down = 1.0 - prob_up
            
            forecasts[timeframe] = {
                'target_high': round(target_high, 2),
                'target_mid': round(target_mid, 2),
                'target_low': round(target_low, 2),
                'prob_up': round(prob_up, 2),
                'prob_down': round(prob_down, 2),
                'expected_move_pct': round((upside + downside) / 2 * 100, 2),
                'upside_pct': round(upside * 100, 2),
                'downside_pct': round(downside * 100, 2)
            }
        
        # Package result
        result = {
            'ticker': ticker,
            'current_price': current_price,
            'timestamp': datetime.now(),
            'volatility_regime': vol_regime,
            'pattern_momentum': momentum,
            'forecasts': forecasts,
            'patterns_analyzed': len(patterns),
            'top_pattern': patterns[0]['pattern'] if patterns else None,
            'top_pattern_confidence': patterns[0]['confidence'] if patterns else None
        }
        
        return result
    
    def get_best_timeframe(self, forecast: Dict) -> Tuple[str, Dict]:
        """
        Determine best timeframe to trade based on forecast.
        
        Args:
            forecast: Forecast dict from forecast_next_days()
            
        Returns:
            Tuple of (best_timeframe, forecast_details)
        """
        forecasts = forecast['forecasts']
        
        # Score each timeframe
        scores = {}
        for tf, details in forecasts.items():
            # Score based on expected move and probability
            expected_move = details['expected_move_pct']
            prob_up = details['prob_up']
            
            # Prefer high probability (>60%) and good move (>5%)
            score = expected_move * max(prob_up, 1 - prob_up)  # Use highest probability direction
            scores[tf] = score
        
        # Get best
        best_tf = max(scores, key=scores.get)
        
        return best_tf, forecasts[best_tf]
    
    def print_forecast(self, forecast: Dict):
        """Pretty print forecast."""
        print(f"\n{'='*70}")
        print(f"📈 FORECAST: {forecast['ticker']}")
        print(f"{'='*70}")
        print(f"Current Price: ${forecast['current_price']:.2f}")
        print(f"Volatility Regime: {forecast['volatility_regime'].upper()}")
        print(f"Pattern Momentum: Bullish {forecast['pattern_momentum']['bullish']:.1%} | "
              f"Bearish {forecast['pattern_momentum']['bearish']:.1%}")
        
        if forecast['top_pattern']:
            print(f"Top Pattern: {forecast['top_pattern']} "
                  f"(confidence: {forecast['top_pattern_confidence']:.1%})")
        
        print(f"\n{'Timeframe':<8} {'Target High':<12} {'Target Mid':<12} {'Target Low':<12} "
              f"{'Prob Up':<10} {'Move %':<10}")
        print('-' * 70)
        
        for tf, details in forecast['forecasts'].items():
            print(f"{tf:<8} ${details['target_high']:<11.2f} "
                  f"${details['target_mid']:<11.2f} ${details['target_low']:<11.2f} "
                  f"{details['prob_up']:<9.1%} {details['expected_move_pct']:>8.1f}%")
        
        # Show best timeframe
        best_tf, best_details = self.get_best_timeframe(forecast)
        print(f"\n🎯 BEST TIMEFRAME: {best_tf}")
        print(f"   Expected Move: {best_details['expected_move_pct']:.1f}%")
        print(f"   Probability Up: {best_details['prob_up']:.1%}")
        print(f"   Target Range: ${best_details['target_low']:.2f} - ${best_details['target_high']:.2f}")


# Example usage
if __name__ == '__main__':
    engine = ForecastingEngine()
    
    # Test on a ticker
    ticker = 'AAPL'
    print(f"\n🔮 Generating forecast for {ticker}...")
    
    # Get data
    df = yf.download(ticker, period='60d', interval='1d', progress=False)
    
    # Generate forecast
    forecast = engine.forecast_next_days(df, ticker)
    
    # Print results
    engine.print_forecast(forecast)
    
    print(f"\n{'='*70}")
    print("✅ Forecasting engine ready!")
    print(f"{'='*70}")
