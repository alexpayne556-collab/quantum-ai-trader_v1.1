"""
Integrated Meta-Learner
Combines pattern detection, regime signals, and research features into unified forecast.

Design:
- Regime-aware weight matrices
- Sector-specific adjustments
- Catalyst proximity weighting
- Hierarchical ensemble structure
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum


class SignalType(Enum):
    """Types of input signals"""
    PATTERN = "pattern"
    REGIME = "regime"
    RESEARCH = "research"
    CATALYST = "catalyst"
    CROSS_ASSET = "cross_asset"


@dataclass
class MetaLearnerConfig:
    """Configuration for meta-learner"""
    # Weight assignment strategy
    weight_strategy: str = "adaptive"  # "fixed", "adaptive", "learned"
    
    # Regime-aware weighting
    regime_aware: bool = True
    
    # Sector-specific tuning
    sector_aware: bool = True
    
    # Catalyst proximity adjustment
    catalyst_proximity_days: int = 3  # Boost catalyst signals within N days
    
    # Minimum confidence threshold
    min_confidence: float = 0.55  # Don't trade below this


class IntegratedMetaLearner:
    """
    Combines multiple signal sources into single forecast using intelligent weighting.
    
    Architecture:
        Input: pattern_signal, regime_state, research_features, catalyst_signals
        Processing: Regime-aware weight matrix → weighted average → calibration
        Output: direction, confidence, reasoning
    
    Usage:
        meta = IntegratedMetaLearner(config=MetaLearnerConfig())
        forecast = meta.combine(
            pattern_sig=pattern_detector.analyze(ticker, data),
            regime_sig=regime_detector.analyze(ticker, data),
            research_features=research_features.calculate_all(ticker, data),
            current_regime='BULL_LOW_VOL_STABLE',
            current_sector='AI_INFRA'
        )
    """
    
    def __init__(self, config: Optional[MetaLearnerConfig] = None):
        self.config = config or MetaLearnerConfig()
        
        # Weight matrices by regime (12 regimes × 5 signal types)
        self.regime_weights = self._init_regime_weights()
        
        # Sector adjustments
        self.sector_multipliers = self._init_sector_multipliers()
        
        # Signal history for adaptive weighting
        self.signal_history = []
    
    def combine(
        self,
        pattern_sig: Dict,
        regime_sig: Dict,
        research_features: Dict[str, float],
        current_regime: str,
        current_sector: str,
        catalyst_proximity: Optional[int] = None
    ) -> Dict:
        """
        Combine all signals into unified forecast.
        
        Args:
            pattern_sig: {signal: 'BUY/SELL/HOLD', confidence: 0.68, type: 'ELLIOTT_5_WAVE'}
            regime_sig: {regime_state: 'STRONG_TREND_UP', adc_value: 42, confidence: 0.72}
            research_features: {dark_pool_ratio: 0.42, pre_breakout_score: 4, ...}
            current_regime: One of 12 regime labels
            current_sector: 'AI_INFRA', 'QUANTUM', 'ROBOTAXI', etc.
            catalyst_proximity: Days until next catalyst (if known)
        
        Returns:
            {
                'direction': 'UP/CHOP/DOWN',
                'confidence': 0.72,  # Calibrated confidence
                'raw_confidence': 0.68,  # Before calibration
                'signal_contributions': {...},  # Breakdown by signal type
                'reasoning': [...],  # Top signals with weights
            }
        """
        
        # 1. Get base weights for this regime
        weights = self._get_regime_weights(current_regime)
        
        # 2. Adjust weights by sector
        if self.config.sector_aware:
            weights = self._adjust_for_sector(weights, current_sector)
        
        # 3. Adjust for catalyst proximity
        if catalyst_proximity is not None and catalyst_proximity <= self.config.catalyst_proximity_days:
            weights = self._adjust_for_catalyst(weights, catalyst_proximity)
        
        # 4. Calculate weighted average of signals
        signals = {
            SignalType.PATTERN: self._parse_pattern_signal(pattern_sig),
            SignalType.REGIME: self._parse_regime_signal(regime_sig),
            SignalType.RESEARCH: self._parse_research_features(research_features),
            SignalType.CATALYST: self._parse_catalyst_signal(research_features),
            SignalType.CROSS_ASSET: self._parse_cross_asset(research_features),
        }
        
        # 5. Weighted ensemble
        raw_score = sum(signals[sig_type] * weights[sig_type] for sig_type in SignalType)
        raw_confidence = abs(raw_score)
        
        # 6. Direction classification
        if raw_score > 0.15:
            direction = 'UP'
        elif raw_score < -0.15:
            direction = 'DOWN'
        else:
            direction = 'CHOP'
        
        # 7. Build reasoning (explainability)
        reasoning = self._build_reasoning(signals, weights, pattern_sig, regime_sig, research_features)
        
        return {
            'direction': direction,
            'raw_confidence': raw_confidence,
            'confidence': raw_confidence,  # Will be calibrated externally
            'signal_contributions': {sig_type.value: signals[sig_type] * weights[sig_type] for sig_type in SignalType},
            'weights_used': {sig_type.value: weights[sig_type] for sig_type in SignalType},
            'reasoning': reasoning,
            'regime': current_regime,
            'sector': current_sector,
        }
    
    # ========== WEIGHT INITIALIZATION ==========
    
    def _init_regime_weights(self) -> Dict[str, Dict[SignalType, float]]:
        """
        Initialize regime-specific weight matrices.
        
        Research guidance (from Perplexity questions):
        - BULL_LOW_VOL: favor patterns (40%), regime (30%), research (20%)
        - EXTREME_VOL: favor catalysts (40%), research (30%), patterns (20%)
        - CHOP: favor microstructure (35%), regime (25%), cross-asset (20%)
        """
        
        # Default weights (will be refined with research answers)
        default = {
            SignalType.PATTERN: 0.35,
            SignalType.REGIME: 0.25,
            SignalType.RESEARCH: 0.20,
            SignalType.CATALYST: 0.10,
            SignalType.CROSS_ASSET: 0.10,
        }
        
        # Regime-specific overrides
        regime_weights = {
            # Bull regimes (favor patterns + trends)
            'BULL_LOW_VOL_STABLE': {**default, SignalType.PATTERN: 0.40, SignalType.REGIME: 0.30},
            'BULL_NORMAL_VOL_STABLE': {**default, SignalType.PATTERN: 0.38, SignalType.REGIME: 0.28},
            'BULL_ELEVATED_VOL_MODERATE': {**default, SignalType.PATTERN: 0.30, SignalType.CATALYST: 0.20},
            'BULL_EXTREME_VOL_VOLATILE': {**default, SignalType.CATALYST: 0.35, SignalType.PATTERN: 0.25},
            
            # Bear regimes (favor catalysts + cross-asset)
            'BEAR_LOW_VOL_STABLE': {**default, SignalType.CATALYST: 0.30, SignalType.CROSS_ASSET: 0.25},
            'BEAR_NORMAL_VOL_STABLE': {**default, SignalType.CATALYST: 0.32, SignalType.CROSS_ASSET: 0.23},
            'BEAR_ELEVATED_VOL_MODERATE': {**default, SignalType.CATALYST: 0.35, SignalType.RESEARCH: 0.25},
            'BEAR_EXTREME_VOL_VOLATILE': {**default, SignalType.CATALYST: 0.40, SignalType.RESEARCH: 0.30},
            
            # Neutral/Chop regimes (favor microstructure + regime)
            'NEUTRAL_LOW_VOL_STABLE': {**default, SignalType.RESEARCH: 0.35, SignalType.REGIME: 0.30},
            'NEUTRAL_NORMAL_VOL_STABLE': {**default, SignalType.RESEARCH: 0.33, SignalType.REGIME: 0.27},
            'NEUTRAL_ELEVATED_VOL_MODERATE': {**default, SignalType.RESEARCH: 0.30, SignalType.CROSS_ASSET: 0.25},
            'NEUTRAL_EXTREME_VOL_VOLATILE': {**default, SignalType.CATALYST: 0.35, SignalType.RESEARCH: 0.30},
        }
        
        return regime_weights
    
    def _init_sector_multipliers(self) -> Dict[str, Dict[SignalType, float]]:
        """
        Sector-specific signal multipliers.
        
        Research guidance:
        - AI_INFRA: patterns work great (0.82 Sharpe) → boost patterns
        - QUANTUM: catalysts matter more (0.68 Sharpe) → boost catalysts
        - ROBOTAXI: microstructure matters (0.61 Sharpe) → boost research
        """
        return {
            'AI_INFRA': {
                SignalType.PATTERN: 1.2,
                SignalType.REGIME: 1.1,
                SignalType.RESEARCH: 1.0,
                SignalType.CATALYST: 0.9,
                SignalType.CROSS_ASSET: 1.15,  # BTC correlation strong
            },
            'QUANTUM': {
                SignalType.PATTERN: 0.9,
                SignalType.REGIME: 1.0,
                SignalType.RESEARCH: 1.1,
                SignalType.CATALYST: 1.3,  # Earnings/news driven
                SignalType.CROSS_ASSET: 0.95,
            },
            'ROBOTAXI': {
                SignalType.PATTERN: 0.95,
                SignalType.REGIME: 1.0,
                SignalType.RESEARCH: 1.25,  # Microstructure matters
                SignalType.CATALYST: 1.2,
                SignalType.CROSS_ASSET: 0.9,
            },
            'HEALTHCARE': {
                SignalType.PATTERN: 1.0,
                SignalType.REGIME: 0.95,
                SignalType.RESEARCH: 1.05,
                SignalType.CATALYST: 1.25,  # FDA approvals matter
                SignalType.CROSS_ASSET: 0.85,
            },
            'ENERGY': {
                SignalType.PATTERN: 0.85,
                SignalType.REGIME: 1.1,
                SignalType.RESEARCH: 1.0,
                SignalType.CATALYST: 1.15,
                SignalType.CROSS_ASSET: 1.3,  # Oil/macro driven
            },
            'DEFAULT': {sig: 1.0 for sig in SignalType},  # No adjustment
        }
    
    # ========== WEIGHT ADJUSTMENT ==========
    
    def _get_regime_weights(self, regime: str) -> Dict[SignalType, float]:
        """Get weights for current regime"""
        if regime in self.regime_weights:
            return self.regime_weights[regime].copy()
        # Fallback to neutral default
        return self.regime_weights.get('NEUTRAL_NORMAL_VOL_STABLE', {sig: 0.2 for sig in SignalType})
    
    def _adjust_for_sector(self, weights: Dict[SignalType, float], sector: str) -> Dict[SignalType, float]:
        """Apply sector-specific multipliers"""
        multipliers = self.sector_multipliers.get(sector, self.sector_multipliers['DEFAULT'])
        adjusted = {sig_type: weights[sig_type] * multipliers[sig_type] for sig_type in SignalType}
        
        # Renormalize to sum to 1
        total = sum(adjusted.values())
        return {sig_type: val / total for sig_type, val in adjusted.items()}
    
    def _adjust_for_catalyst(self, weights: Dict[SignalType, float], days_until: int) -> Dict[SignalType, float]:
        """Boost catalyst signals when event is near"""
        boost_factor = 1.0 + (1.0 - days_until / self.config.catalyst_proximity_days)
        weights[SignalType.CATALYST] *= boost_factor
        
        # Renormalize
        total = sum(weights.values())
        return {sig_type: val / total for sig_type, val in weights.items()}
    
    # ========== SIGNAL PARSING ==========
    
    @staticmethod
    def _parse_pattern_signal(pattern_sig: Dict) -> float:
        """Convert pattern signal to [-1, +1] score"""
        signal_map = {'STRONG_BUY': 1.0, 'BUY': 0.6, 'HOLD': 0.0, 'SELL': -0.6, 'STRONG_SELL': -1.0}
        signal = pattern_sig.get('signal', 'HOLD')
        confidence = pattern_sig.get('confidence', 0.5)
        return signal_map.get(signal, 0.0) * confidence
    
    @staticmethod
    def _parse_regime_signal(regime_sig: Dict) -> float:
        """Convert regime signal to [-1, +1] score"""
        regime_map = {'STRONG_TREND_UP': 1.0, 'WEAK_TREND_UP': 0.5, 'CHOPPY': 0.0, 
                      'WEAK_TREND_DOWN': -0.5, 'STRONG_TREND_DOWN': -1.0}
        regime = regime_sig.get('regime_state', 'CHOPPY')
        confidence = regime_sig.get('confidence', 0.5)
        return regime_map.get(regime, 0.0) * confidence
    
    @staticmethod
    def _parse_research_features(features: Dict[str, float]) -> float:
        """Aggregate research features into single score"""
        # TODO: Replace with learned aggregation (linear model or neural net)
        # For now: simple heuristic based on pre-breakout score
        breakout_score = features.get('pre_breakout_score', 0)
        return (breakout_score - 2.5) / 2.5  # Normalize to [-1, +1], neutral at 2.5/5
    
    @staticmethod
    def _parse_catalyst_signal(features: Dict[str, float]) -> float:
        """Extract catalyst signal from features"""
        pre_catalyst = features.get('pre_catalyst_signal', 0.0)
        return (pre_catalyst - 0.5) * 2  # Normalize [0, 1] → [-1, +1]
    
    @staticmethod
    def _parse_cross_asset(features: Dict[str, float]) -> float:
        """Extract cross-asset sentiment"""
        # TODO: Aggregate BTC/yields/VIX signals
        btc_momentum = features.get('btc_momentum_lag_1', 0.0)
        return btc_momentum  # Already normalized
    
    def _build_reasoning(self, signals: Dict[SignalType, float], weights: Dict[SignalType, float],
                         pattern_sig: Dict, regime_sig: Dict, features: Dict) -> List[Tuple[str, float, str]]:
        """
        Build explainability: which signals contributed most?
        
        Returns: [(signal_name, contribution, explanation), ...]
        """
        contributions = [(sig_type.value, signals[sig_type] * weights[sig_type]) for sig_type in SignalType]
        contributions.sort(key=lambda x: abs(x[1]), reverse=True)
        
        reasoning = []
        for sig_name, contrib in contributions[:5]:  # Top 5 contributors
            if sig_name == 'pattern':
                explanation = f"Pattern: {pattern_sig.get('type', 'unknown')} ({pattern_sig.get('confidence', 0):.0%} confidence)"
            elif sig_name == 'regime':
                explanation = f"Regime: {regime_sig.get('regime_state', 'unknown')}"
            elif sig_name == 'research':
                breakout = features.get('pre_breakout_score', 0)
                explanation = f"Microstructure: {breakout}/5 pre-breakout features"
            elif sig_name == 'catalyst':
                catalyst_val = features.get('pre_catalyst_signal', 0)
                explanation = f"Catalyst signal: {catalyst_val:.0%} probability"
            elif sig_name == 'cross_asset':
                explanation = "Cross-asset momentum"
            else:
                explanation = sig_name
            
            reasoning.append((sig_name, abs(contrib), explanation))
        
        return reasoning


# ========== ADAPTIVE WEIGHT LEARNING (OPTIONAL) ==========

class AdaptiveWeightLearner:
    """
    Learn optimal weights from historical performance.
    
    TODO: Implement meta-learning layer that:
    1. Tracks which signal combinations perform best per regime
    2. Updates weights based on recent win-rate
    3. Uses Bayesian updating or gradient descent
    """
    
    def __init__(self):
        raise NotImplementedError("Adaptive learning coming soon")
