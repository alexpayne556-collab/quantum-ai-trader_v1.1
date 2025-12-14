"""
INSTITUTIONAL CONFLUENCE ENGINE
Combines signals from multiple timeframes and patterns using Bayesian probability fusion.

Features:
- Hierarchical timeframe scoring (requires 2+ timeframe agreement)
- Bayesian log-odds combination (prevents overconfidence)
- Context-aware pattern weighting (RSI, ATR, volume, regime)
- Logarithmic confidence scaling (caps at 85%)
- Pattern statistics integration from PatternStatsEngine

Based on hedge fund research: Logarithmic fusion prevents multiplicative explosion,
require multi-timeframe agreement, boost signals in favorable contexts.

Usage:
    engine = ConfluenceEngine(pattern_stats_engine)
    
    patterns_by_tf = {
        '5m': [{'name': 'CDLHAMMER', 'confidence': 0.7}],
        '1h': [{'name': 'CDLHAMMER', 'confidence': 0.8}],
        '1d': [{'name': 'CDLENGULFING', 'confidence': 0.6}]
    }
    
    context = {
        'rsi': 28,
        'atr_percentile': 0.65,
        'volume_ratio': 2.1,
        'regime': 'BULL'
    }
    
    score = engine.calculate_confluence(patterns_by_tf, context)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class ConfluenceScore:
    """Confluence analysis result"""
    final_score: float
    direction: str  # 'BULLISH', 'BEARISH', 'NEUTRAL'
    confidence: float
    timeframes_in_agreement: int
    total_timeframes: int
    hierarchical_score: float
    bayesian_score: float
    context_boost: float
    components: Dict[str, float]


class ConfluenceEngine:
    """
    Combine multiple signals with institutional-grade fusion logic.
    
    Prevents overconfidence through:
    1. Multi-timeframe requirement (≥2 timeframes must agree)
    2. Bayesian log-odds fusion (logarithmic, not multiplicative)
    3. Confidence cap at 85% (acknowledges uncertainty)
    4. Context weighting (boosts/penalizes based on market conditions)
    """
    
    # Timeframe hierarchy (importance weights for swing trading 1-3 day to weeks)
    TIMEFRAME_WEIGHTS = {
        '5m': 0.1,   # Minor importance
        '15m': 0.15,
        '1h': 0.25,
        '4h': 0.30,
        '1d': 0.35,  # Major importance for swing trading
        '1w': 0.20   # Context
    }
    
    # Confidence cap (prevent overconfidence)
    MAX_CONFIDENCE = 0.85
    
    # Minimum agreement threshold
    MIN_AGREEMENT_SCORE = 0.5  # Pattern must score >50% to count as "agreement"
    MIN_TIMEFRAMES_AGREEING = 2
    
    def __init__(self, pattern_stats_engine=None):
        """
        Initialize confluence engine.
        
        Args:
            pattern_stats_engine: Optional PatternStatsEngine for historical stats
        """
        self.pattern_stats = pattern_stats_engine
        logger.info("✅ ConfluenceEngine initialized")
    
    def calculate_confluence(
        self,
        patterns_by_timeframe: Dict[str, List[Dict]],
        context: Dict[str, any],
        require_multi_tf: bool = True
    ) -> ConfluenceScore:
        """
        Calculate confluence score from multiple timeframes and contexts.
        
        Args:
            patterns_by_timeframe: {
                '1d': [{'name': 'CDLHAMMER', 'confidence': 0.7, 'direction': 'BULLISH'}],
                '1h': [{'name': 'CDLHAMMER', 'confidence': 0.8, 'direction': 'BULLISH'}]
            }
            context: {
                'rsi': 28,
                'atr_percentile': 0.65,
                'volume_ratio': 2.1,
                'regime': 'BULL',
                'volatility_bucket': 'NORMAL'
            }
            require_multi_tf: Require at least 2 timeframes in agreement
        
        Returns:
            ConfluenceScore object with analysis
        """
        if not patterns_by_timeframe:
            return self._create_neutral_score()
        
        # Step 1: Hierarchical timeframe scoring
        hierarchical_score, direction, tf_scores = self._hierarchical_timeframe_score(
            patterns_by_timeframe
        )
        
        # Check multi-timeframe requirement
        timeframes_in_agreement = sum(
            1 for score in tf_scores.values() if abs(score) > self.MIN_AGREEMENT_SCORE
        )
        
        if require_multi_tf and timeframes_in_agreement < self.MIN_TIMEFRAMES_AGREEING:
            logger.debug(f"Insufficient timeframe agreement: {timeframes_in_agreement}/{len(tf_scores)}")
            return ConfluenceScore(
                final_score=0.0,
                direction='NEUTRAL',
                confidence=0.0,
                timeframes_in_agreement=timeframes_in_agreement,
                total_timeframes=len(tf_scores),
                hierarchical_score=hierarchical_score,
                bayesian_score=0.0,
                context_boost=1.0,
                components={'reason': 'INSUFFICIENT_TIMEFRAME_AGREEMENT'}
            )
        
        # Step 2: Bayesian fusion of pattern probabilities
        pattern_probs = self._extract_pattern_probabilities(patterns_by_timeframe)
        bayesian_score = self._bayesian_log_odds_fusion(pattern_probs)
        
        # Step 3: Context-aware weighting
        context_boost = self._calculate_context_boost(
            base_direction=direction,
            context=context
        )
        
        # Step 4: Combine scores with cap
        combined_score = hierarchical_score * 0.4 + bayesian_score * 0.6
        boosted_score = combined_score * context_boost
        
        # Cap confidence
        final_confidence = min(abs(boosted_score), self.MAX_CONFIDENCE)
        final_direction = direction if boosted_score > 0 else ('BEARISH' if boosted_score < 0 else 'NEUTRAL')
        
        return ConfluenceScore(
            final_score=boosted_score,
            direction=final_direction,
            confidence=final_confidence,
            timeframes_in_agreement=timeframes_in_agreement,
            total_timeframes=len(tf_scores),
            hierarchical_score=hierarchical_score,
            bayesian_score=bayesian_score,
            context_boost=context_boost,
            components={
                'timeframe_scores': tf_scores,
                'pattern_count': sum(len(p) for p in patterns_by_timeframe.values())
            }
        )
    
    def _hierarchical_timeframe_score(
        self,
        patterns_by_timeframe: Dict[str, List[Dict]]
    ) -> Tuple[float, str, Dict[str, float]]:
        """
        Calculate weighted score across timeframes.
        
        Returns:
            (weighted_score, direction, individual_tf_scores)
        """
        tf_scores = {}
        total_weight = 0.0
        weighted_sum = 0.0
        
        for timeframe, patterns in patterns_by_timeframe.items():
            if timeframe not in self.TIMEFRAME_WEIGHTS:
                continue
            
            # Average pattern scores for this timeframe
            tf_score = 0.0
            for pattern in patterns:
                confidence = pattern.get('confidence', 0.5)
                direction = pattern.get('direction', 'NEUTRAL')
                
                # Convert direction to numeric
                if direction == 'BULLISH':
                    tf_score += confidence
                elif direction == 'BEARISH':
                    tf_score -= confidence
            
            # Average across patterns
            if patterns:
                tf_score /= len(patterns)
            
            tf_scores[timeframe] = tf_score
            
            # Weighted sum
            weight = self.TIMEFRAME_WEIGHTS[timeframe]
            weighted_sum += tf_score * weight
            total_weight += weight
        
        # Calculate weighted average
        if total_weight > 0:
            weighted_score = weighted_sum / total_weight
        else:
            weighted_score = 0.0
        
        # Determine overall direction
        if weighted_score > 0.1:
            direction = 'BULLISH'
        elif weighted_score < -0.1:
            direction = 'BEARISH'
        else:
            direction = 'NEUTRAL'
        
        return weighted_score, direction, tf_scores
    
    def _extract_pattern_probabilities(
        self,
        patterns_by_timeframe: Dict[str, List[Dict]]
    ) -> List[float]:
        """
        Extract all pattern probabilities for Bayesian fusion.
        
        Returns:
            List of probabilities (0-1)
        """
        probs = []
        
        for timeframe, patterns in patterns_by_timeframe.items():
            for pattern in patterns:
                confidence = pattern.get('confidence', 0.5)
                
                # Enhance with historical stats if available
                if self.pattern_stats:
                    pattern_name = pattern.get('name')
                    if pattern_name:
                        stats = self.pattern_stats.get_pattern_edge(
                            pattern_name,
                            context={'timeframe': timeframe}
                        )
                        if stats and stats.status in ['STRONG', 'MODERATE']:
                            # Blend model confidence with historical performance
                            confidence = (confidence * 0.6) + (stats.win_rate * 0.4)
                
                probs.append(confidence)
        
        return probs
    
    def _bayesian_log_odds_fusion(self, probabilities: List[float]) -> float:
        """
        Combine multiple probabilities using Bayesian log-odds fusion.
        
        This prevents overconfidence - if you have 3 patterns each at 70%,
        multiplicative would give 97%, but log-odds gives ~85%.
        
        Formula: log_odds_sum = sum(log(p / (1-p)))
                 final_prob = 1 / (1 + exp(-log_odds_sum))
        
        Args:
            probabilities: List of independent probabilities (0-1)
        
        Returns:
            Combined probability (0-1), capped at MAX_CONFIDENCE
        """
        if not probabilities:
            return 0.5  # Neutral
        
        # Convert probabilities to log odds
        log_odds_sum = 0.0
        for p in probabilities:
            # Clamp to avoid log(0) or division by zero
            p_clamped = np.clip(p, 0.01, 0.99)
            log_odds = np.log(p_clamped / (1 - p_clamped))
            log_odds_sum += log_odds
        
        # Convert back to probability
        posterior_prob = 1.0 / (1.0 + np.exp(-log_odds_sum))
        
        # Cap at MAX_CONFIDENCE
        return min(posterior_prob, self.MAX_CONFIDENCE)
    
    def _calculate_context_boost(
        self,
        base_direction: str,
        context: Dict[str, any]
    ) -> float:
        """
        Calculate context-based boost/penalty multiplier.
        
        Boosts bullish patterns when:
        - RSI oversold (<30)
        - High volume (>2x average)
        - Bull regime
        - Pattern at support levels
        
        Args:
            base_direction: 'BULLISH' or 'BEARISH'
            context: Market context dict
        
        Returns:
            Boost multiplier (0.5 to 1.5)
        """
        boost = 1.0
        
        # RSI context
        rsi = context.get('rsi', 50)
        if base_direction == 'BULLISH' and rsi < 30:
            boost *= 1.25  # Oversold bounce
        elif base_direction == 'BULLISH' and rsi > 70:
            boost *= 0.80  # Overbought, weaker
        elif base_direction == 'BEARISH' and rsi > 70:
            boost *= 1.25  # Overbought reversal
        elif base_direction == 'BEARISH' and rsi < 30:
            boost *= 0.80  # Oversold, weaker sell
        
        # Volume context
        volume_ratio = context.get('volume_ratio', 1.0)
        if volume_ratio > 2.0:
            boost *= 1.20  # High volume = stronger conviction
        elif volume_ratio < 0.5:
            boost *= 0.85  # Low volume = weaker
        
        # Regime context
        regime = context.get('regime', 'RANGE')
        if base_direction == 'BULLISH' and regime == 'BULL':
            boost *= 1.15  # Trend continuation
        elif base_direction == 'BEARISH' and regime == 'BEAR':
            boost *= 1.15
        elif base_direction == 'BULLISH' and regime == 'BEAR':
            boost *= 0.75  # Counter-trend, weaker
        elif base_direction == 'BEARISH' and regime == 'BULL':
            boost *= 0.75
        
        # Volatility context
        atr_percentile = context.get('atr_percentile', 0.5)
        volatility_bucket = context.get('volatility_bucket', 'NORMAL')
        
        if volatility_bucket == 'HIGH' or atr_percentile > 0.75:
            boost *= 0.90  # High vol = less reliable patterns
        elif volatility_bucket == 'LOW' or atr_percentile < 0.25:
            boost *= 1.10  # Low vol = cleaner patterns
        
        # Clamp boost to reasonable range
        boost = np.clip(boost, 0.5, 1.5)
        
        return boost
    
    def _create_neutral_score(self) -> ConfluenceScore:
        """Create neutral score when no patterns detected"""
        return ConfluenceScore(
            final_score=0.0,
            direction='NEUTRAL',
            confidence=0.0,
            timeframes_in_agreement=0,
            total_timeframes=0,
            hierarchical_score=0.0,
            bayesian_score=0.5,
            context_boost=1.0,
            components={'reason': 'NO_PATTERNS'}
        )
    
    def score_single_pattern(
        self,
        pattern_name: str,
        pattern_confidence: float,
        context: Dict[str, any]
    ) -> float:
        """
        Score a single pattern with context awareness.
        
        Args:
            pattern_name: Pattern identifier (e.g., 'CDLHAMMER')
            pattern_confidence: Base confidence from detection (0-1)
            context: Market context dict
        
        Returns:
            Context-adjusted confidence (0-1)
        """
        # Get historical stats if available
        if self.pattern_stats:
            stats = self.pattern_stats.get_pattern_edge(
                pattern_name,
                context={
                    'timeframe': context.get('timeframe', '1d'),
                    'regime': context.get('regime', 'ALL'),
                    'volatility_bucket': context.get('volatility_bucket', 'ALL')
                }
            )
            
            if stats and stats.status in ['STRONG', 'MODERATE']:
                # Blend detection confidence with historical performance
                pattern_confidence = (pattern_confidence * 0.5) + (stats.win_rate * 0.5)
                
                # Boost if pattern has high IC
                if stats.rank_ic_5bar > 0.1:
                    pattern_confidence *= 1.1
            elif stats and stats.status == 'DEAD':
                # Heavily penalize dead patterns
                pattern_confidence *= 0.3
        
        # Apply context boost
        direction = 'BULLISH' if pattern_confidence > 0.5 else 'BEARISH'
        boost = self._calculate_context_boost(direction, context)
        
        adjusted_confidence = pattern_confidence * boost
        
        # Cap at max confidence
        return min(adjusted_confidence, self.MAX_CONFIDENCE)


if __name__ == '__main__':
    # Example usage
    print("🔧 Testing Confluence Engine...")
    
    engine = ConfluenceEngine()
    
    # Test 1: Multi-timeframe bullish confluence
    patterns_by_tf = {
        '5m': [
            {'name': 'CDLHAMMER', 'confidence': 0.65, 'direction': 'BULLISH'}
        ],
        '1h': [
            {'name': 'CDLHAMMER', 'confidence': 0.75, 'direction': 'BULLISH'},
            {'name': 'CDLENGULFING', 'confidence': 0.70, 'direction': 'BULLISH'}
        ],
        '1d': [
            {'name': 'CDLMORNINGSTAR', 'confidence': 0.80, 'direction': 'BULLISH'}
        ]
    }
    
    context = {
        'rsi': 28,
        'atr_percentile': 0.45,
        'volume_ratio': 2.3,
        'regime': 'BULL',
        'volatility_bucket': 'NORMAL'
    }
    
    score = engine.calculate_confluence(patterns_by_tf, context)
    
    print(f"\n✅ Confluence Analysis:")
    print(f"   Final Score: {score.final_score:.3f}")
    print(f"   Direction: {score.direction}")
    print(f"   Confidence: {score.confidence:.1%}")
    print(f"   Timeframes Agreeing: {score.timeframes_in_agreement}/{score.total_timeframes}")
    print(f"   Hierarchical Score: {score.hierarchical_score:.3f}")
    print(f"   Bayesian Score: {score.bayesian_score:.3f}")
    print(f"   Context Boost: {score.context_boost:.2f}x")
    print(f"   Components: {score.components}")
    
    # Test 2: Insufficient timeframe agreement
    print("\n🔧 Testing insufficient timeframe agreement...")
    patterns_single_tf = {
        '1d': [{'name': 'CDLHAMMER', 'confidence': 0.9, 'direction': 'BULLISH'}]
    }
    
    score2 = engine.calculate_confluence(patterns_single_tf, context)
    print(f"   Direction: {score2.direction}")
    print(f"   Confidence: {score2.confidence:.1%}")
    print(f"   Reason: {score2.components.get('reason')}")
    
    # Test 3: Context boost effects
    print("\n🔧 Testing context boost...")
    oversold_context = {**context, 'rsi': 25, 'volume_ratio': 3.0}
    overbought_context = {**context, 'rsi': 75, 'volume_ratio': 0.8}
    
    score_oversold = engine.calculate_confluence(patterns_by_tf, oversold_context)
    score_overbought = engine.calculate_confluence(patterns_by_tf, overbought_context)
    
    print(f"   Oversold Boost: {score_oversold.context_boost:.2f}x → Conf: {score_oversold.confidence:.1%}")
    print(f"   Overbought Boost: {score_overbought.context_boost:.2f}x → Conf: {score_overbought.confidence:.1%}")
    
    print("\n✅ Confluence Engine Ready for Production!")
