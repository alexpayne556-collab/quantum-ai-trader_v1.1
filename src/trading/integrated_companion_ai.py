"""
🤖 INTEGRATED COMPANION AI
Daily action plan generator combining ALL systems:
- pattern_detector.py (65 patterns)
- pattern_baseline_scorer.py (real win rates)  
- forecasting_engine.py (1/2/5/7 day predictions)
- existing companion_ai.py (signal decay monitoring)

Generates complete daily action plans:
- Entry signals (which pattern, when to enter)
- Exit signals (profit targets, stop loss)
- Position sizing (based on confidence)
- Hold duration (based on forecast)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import yfinance as yf

# Import existing systems
try:
    from src.trading.pattern_baseline_scorer import PatternBaselineScorer
    SCORER_AVAILABLE = True
except ImportError:
    SCORER_AVAILABLE = False

try:
    from src.trading.forecasting_engine import ForecastingEngine
    FORECASTER_AVAILABLE = True
except ImportError:
    FORECASTER_AVAILABLE = False

try:
    from src.trading.companion_ai import CompanionAI as SignalMonitor
    SIGNAL_MONITOR_AVAILABLE = True
except ImportError:
    SIGNAL_MONITOR_AVAILABLE = False


class IntegratedCompanionAI:
    """
    Complete companion AI that generates daily action plans.
    
    Combines:
    - Pattern detection + scoring (real win rates)
    - Multi-timeframe forecasting (1/2/5/7 days)
    - Signal decay monitoring (30-min half-life)
    - Regime shift detection
    """
    
    def __init__(self):
        """Initialize with all components."""
        self.scorer = PatternBaselineScorer() if SCORER_AVAILABLE else None
        self.forecaster = ForecastingEngine() if FORECASTER_AVAILABLE else None
        self.signal_monitor = SignalMonitor() if SIGNAL_MONITOR_AVAILABLE else None
        
        # Risk management defaults
        self.max_position_size = 0.20  # 20% max per position
        self.min_confidence = 0.65     # 65% min win rate
        self.profit_target_multiplier = 1.5  # Target 1.5x expected move
        self.stop_loss_multiplier = 0.5      # Stop at 0.5x expected move
    
    def generate_daily_action_plan(self, ticker: str, df: Optional[pd.DataFrame] = None) -> Dict:
        """
        Generate complete daily action plan for a ticker.
        
        Args:
            ticker: Stock ticker (e.g., 'AAPL')
            df: Optional pre-loaded DataFrame (will fetch if not provided)
            
        Returns:
            Dict with complete action plan
        """
        # Fetch data if not provided
        if df is None:
            df = yf.download(ticker, period='60d', interval='1d', progress=False)
        
        current_price = float(df['Close'].iloc[-1])
        
        # Step 1: Detect and score patterns
        patterns_result = self.scorer.detect_and_score_patterns(ticker) if self.scorer else {'patterns': []}
        patterns = patterns_result.get('patterns', [])
        
        # Filter high-confidence patterns (≥65% WR)
        high_conf_patterns = [p for p in patterns if p['confidence'] >= self.min_confidence]
        
        # Step 2: Generate forecast
        forecast = self.forecaster.forecast_next_days(df, ticker, current_price) if self.forecaster else None
        
        # Step 3: Determine best timeframe
        if forecast:
            best_timeframe, best_forecast = self.forecaster.get_best_timeframe(forecast)
        else:
            best_timeframe = '5d'
            best_forecast = {'expected_move_pct': 5.0, 'prob_up': 0.5}
        
        # Step 4: Generate signals
        signals = self._generate_signals(patterns, high_conf_patterns, forecast, current_price)
        
        # Step 5: Calculate position sizing
        position_size = self._calculate_position_size(signals, high_conf_patterns)
        
        # Step 6: Set profit targets and stop loss
        targets = self._calculate_targets(current_price, best_forecast, signals)
        
        # Package action plan
        action_plan = {
            'ticker': ticker,
            'generated_at': datetime.now(),
            'current_price': current_price,
            'market_regime': forecast.get('volatility_regime', 'unknown') if forecast else 'unknown',
            
            # Signal summary
            'signal': signals['overall_signal'],
            'confidence': signals['overall_confidence'],
            'pattern_count': len(patterns),
            'high_conf_pattern_count': len(high_conf_patterns),
            
            # Top patterns
            'top_patterns': high_conf_patterns[:3],
            
            # Forecast
            'best_timeframe': best_timeframe,
            'expected_move_pct': best_forecast.get('expected_move_pct', 0),
            'prob_up': best_forecast.get('prob_up', 0.5),
            
            # Action items
            'entry_price': targets['entry_price'],
            'profit_target': targets['profit_target'],
            'stop_loss': targets['stop_loss'],
            'position_size_pct': position_size,
            'hold_duration_days': self._extract_days(best_timeframe),
            
            # Risk metrics
            'risk_reward_ratio': targets['risk_reward_ratio'],
            'max_loss_pct': targets['max_loss_pct'],
            'expected_gain_pct': targets['expected_gain_pct'],
            
            # Recommendation
            'recommendation': self._generate_recommendation(signals, targets, high_conf_patterns)
        }
        
        return action_plan
    
    def _generate_signals(self, all_patterns: List[Dict], high_conf_patterns: List[Dict],
                         forecast: Optional[Dict], current_price: float) -> Dict:
        """Generate trading signals from patterns and forecast."""
        if not high_conf_patterns:
            return {
                'overall_signal': 'HOLD',
                'overall_confidence': 0.0,
                'bullish_score': 0.0,
                'bearish_score': 0.0
            }
        
        # Calculate bullish/bearish scores
        bullish_score = sum(p['confidence'] for p in high_conf_patterns if p['type'] == 'BULLISH')
        bearish_score = sum(p['confidence'] for p in high_conf_patterns if p['type'] == 'BEARISH')
        total_score = bullish_score + bearish_score
        
        # Normalize
        if total_score > 0:
            bullish_pct = bullish_score / total_score
            bearish_pct = bearish_score / total_score
        else:
            bullish_pct = 0.5
            bearish_pct = 0.5
        
        # Determine signal
        if bullish_pct > 0.65:
            signal = 'BUY'
            confidence = bullish_pct
        elif bearish_pct > 0.65:
            signal = 'SELL'
            confidence = bearish_pct
        else:
            signal = 'HOLD'
            confidence = max(bullish_pct, bearish_pct)
        
        return {
            'overall_signal': signal,
            'overall_confidence': confidence,
            'bullish_score': bullish_pct,
            'bearish_score': bearish_pct
        }
    
    def _calculate_position_size(self, signals: Dict, high_conf_patterns: List[Dict]) -> float:
        """
        Calculate position size based on confidence and pattern count.
        
        Higher confidence + more patterns = larger position.
        """
        confidence = signals['overall_confidence']
        pattern_count = len(high_conf_patterns)
        
        # Base size from confidence
        base_size = confidence * self.max_position_size
        
        # Boost for multiple confirming patterns (up to 1.5x)
        pattern_boost = min(1.0 + (pattern_count - 1) * 0.1, 1.5)
        
        # Final size
        position_size = min(base_size * pattern_boost, self.max_position_size)
        
        return round(position_size, 2)
    
    def _calculate_targets(self, current_price: float, forecast: Dict, signals: Dict) -> Dict:
        """Calculate entry, profit target, and stop loss."""
        expected_move = forecast.get('expected_move_pct', 5.0) / 100.0
        
        # Entry: current price (market order assumed)
        entry_price = current_price
        
        # Profit target: 1.5x expected move in signal direction
        if signals['overall_signal'] == 'BUY':
            profit_target = entry_price * (1 + expected_move * self.profit_target_multiplier)
            stop_loss = entry_price * (1 - expected_move * self.stop_loss_multiplier)
        elif signals['overall_signal'] == 'SELL':
            profit_target = entry_price * (1 - expected_move * self.profit_target_multiplier)
            stop_loss = entry_price * (1 + expected_move * self.stop_loss_multiplier)
        else:  # HOLD
            profit_target = entry_price
            stop_loss = entry_price
        
        # Calculate metrics
        expected_gain_pct = ((profit_target - entry_price) / entry_price) * 100
        max_loss_pct = ((stop_loss - entry_price) / entry_price) * 100
        risk_reward_ratio = abs(expected_gain_pct / max_loss_pct) if max_loss_pct != 0 else 0
        
        return {
            'entry_price': round(entry_price, 2),
            'profit_target': round(profit_target, 2),
            'stop_loss': round(stop_loss, 2),
            'expected_gain_pct': round(expected_gain_pct, 2),
            'max_loss_pct': round(max_loss_pct, 2),
            'risk_reward_ratio': round(risk_reward_ratio, 2)
        }
    
    @staticmethod
    def _extract_days(timeframe: str) -> int:
        """Extract days from timeframe string (e.g., '5d' -> 5)."""
        return int(timeframe.replace('d', ''))
    
    def _generate_recommendation(self, signals: Dict, targets: Dict, 
                                high_conf_patterns: List[Dict]) -> str:
        """Generate human-readable recommendation."""
        signal = signals['overall_signal']
        confidence = signals['overall_confidence']
        pattern_count = len(high_conf_patterns)
        
        if signal == 'HOLD':
            return "⏸️  HOLD - No high-confidence patterns detected. Wait for better setup."
        
        # Get top pattern
        top_pattern = high_conf_patterns[0] if high_conf_patterns else None
        pattern_name = top_pattern['pattern'] if top_pattern else 'UNKNOWN'
        
        # Generate recommendation
        if signal == 'BUY':
            rec = f"🟢 BUY SIGNAL\n"
            rec += f"   Pattern: {pattern_name} ({confidence:.1%} confidence)\n"
            rec += f"   Entry: ${targets['entry_price']:.2f}\n"
            rec += f"   Target: ${targets['profit_target']:.2f} (+{targets['expected_gain_pct']:.1f}%)\n"
            rec += f"   Stop Loss: ${targets['stop_loss']:.2f} ({targets['max_loss_pct']:.1f}%)\n"
            rec += f"   Risk/Reward: {targets['risk_reward_ratio']:.2f}\n"
            if pattern_count > 1:
                rec += f"   Confluence: {pattern_count} patterns confirming"
        else:  # SELL
            rec = f"🔴 SELL SIGNAL\n"
            rec += f"   Pattern: {pattern_name} ({confidence:.1%} confidence)\n"
            rec += f"   Entry: ${targets['entry_price']:.2f}\n"
            rec += f"   Target: ${targets['profit_target']:.2f} ({targets['expected_gain_pct']:.1f}%)\n"
            rec += f"   Stop Loss: ${targets['stop_loss']:.2f} (+{targets['max_loss_pct']:.1f}%)\n"
            rec += f"   Risk/Reward: {targets['risk_reward_ratio']:.2f}\n"
            if pattern_count > 1:
                rec += f"   Confluence: {pattern_count} patterns confirming"
        
        return rec
    
    def print_action_plan(self, plan: Dict):
        """Pretty print action plan."""
        print(f"\n{'='*70}")
        print(f"🤖 DAILY ACTION PLAN: {plan['ticker']}")
        print(f"{'='*70}")
        print(f"Generated: {plan['generated_at'].strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Current Price: ${plan['current_price']:.2f}")
        print(f"Market Regime: {plan['market_regime'].upper()}")
        print(f"\n{plan['recommendation']}")
        
        if plan['signal'] != 'HOLD':
            print(f"\n📊 Position Details:")
            print(f"   Position Size: {plan['position_size_pct']:.1%} of portfolio")
            print(f"   Hold Duration: {plan['hold_duration_days']} days")
            print(f"   Expected Move: {plan['expected_move_pct']:.1f}%")
            print(f"   Probability Up: {plan['prob_up']:.1%}")
            
            if plan['top_patterns']:
                print(f"\n🎯 Top Patterns:")
                for i, p in enumerate(plan['top_patterns'], 1):
                    print(f"   {i}. {p['pattern']} - {p['confidence']:.1%} confidence")


# Example usage
if __name__ == '__main__':
    ai = IntegratedCompanionAI()
    
    # Test on multiple tickers
    tickers = ['AAPL', 'NVDA', 'TSLA']
    
    for ticker in tickers:
        print(f"\n{'#'*70}")
        print(f"Generating action plan for {ticker}...")
        
        plan = ai.generate_daily_action_plan(ticker)
        ai.print_action_plan(plan)
    
    print(f"\n{'='*70}")
    print("✅ Integrated Companion AI ready!")
    print(f"{'='*70}")
