"""
OPTIMIZED SIGNAL CONFIGURATION
Generated from DEEP_PATTERN_EVOLUTION_TRAINER analysis (2025-12-05)

Results Summary:
- Best Solo Signal: trend (65% WR, +13.7% avg PnL)
- Best Combination: trend alone (combinations don't improve)
- Robust signals: trend, rsi_divergence, dip_buy (all passed adversarial tests)

Key Insights:
- 'trend' works in bull (77.8% WR) and bear (71.4% WR), but FAILS in sideways (-1.1%)
- 'dip_buy' works best in sideways (62.5% WR, +12.2% avg)
- 'bounce' works in bear/sideways (60% WR each)

Implementation: Use regime-aware signal switching
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional


# =============================================================================
# SIGNAL TIER RANKINGS (from Phase 1 Independent Testing)
# =============================================================================

SIGNAL_TIERS = {
    # TIER S - Excellent (Use with weight 1.5-2.0)
    'tier_s': ['trend'],
    
    # TIER A - Good (Use with weight 1.0)  
    'tier_a': ['rsi_divergence'],
    
    # TIER B - OK (Use with weight 0.5 or conditional)
    'tier_b': ['dip_buy', 'bounce', 'momentum'],
    
    # TIER F - Fail (Disable these - low trades or poor WR)
    'tier_f': ['nuclear_dip', 'vol_squeeze', 'consolidation', 'uptrend_pullback']
}


# =============================================================================
# OPTIMAL SIGNAL WEIGHTS (from multi-phase analysis)
# =============================================================================

OPTIMAL_SIGNAL_WEIGHTS = {
    # Tier S - Excellent
    'trend': 1.8,
    
    # Tier A - Good
    'rsi_divergence': 1.0,
    
    # Tier B - Use cautiously (regime-dependent)
    'dip_buy': 0.5,
    'bounce': 0.5,
    'momentum': 0.5,
    
    # Tier F - Disabled
    'nuclear_dip': 0.0,
    'vol_squeeze': 0.0,
    'consolidation': 0.0,
    'uptrend_pullback': 0.0
}


# =============================================================================
# REGIME-SPECIFIC WEIGHTS (from Phase 5 Analysis)
# =============================================================================

REGIME_SIGNAL_WEIGHTS = {
    'bull': {
        'trend': 2.0,           # 77.8% WR, +21.7% avg - BEST in bull
        'rsi_divergence': 0.8,  # 60% WR, decent
        'momentum': 0.7,        # Works in bull trends
        'dip_buy': 0.3,         # Less effective in bull
        'bounce': 0.4,          # 43.8% WR - weak in bull
        'nuclear_dip': 0.0,
        'vol_squeeze': 0.0,
        'consolidation': 0.0,
        'uptrend_pullback': 0.0
    },
    'bear': {
        'trend': 1.5,           # 71.4% WR, +11.9% avg - still good
        'rsi_divergence': 0.7,  # 50% WR
        'dip_buy': 0.8,         # Mean reversion works in bear
        'bounce': 1.0,          # 60% WR - bounces work in bear
        'momentum': 0.5,        # Less reliable
        'nuclear_dip': 0.3,     # Can catch real bottoms
        'vol_squeeze': 0.0,
        'consolidation': 0.0,
        'uptrend_pullback': 0.0
    },
    'sideways': {
        'trend': 0.0,           # -1.1% avg PnL - DISABLE in sideways!
        'rsi_divergence': 1.2,  # 62.5% WR, +9.1% avg - good
        'dip_buy': 1.5,         # 62.5% WR, +12.2% avg - BEST in sideways
        'bounce': 1.2,          # 60% WR - works in chop
        'momentum': 0.3,        # Momentum fails in chop
        'nuclear_dip': 0.0,
        'vol_squeeze': 0.0,
        'consolidation': 0.0,
        'uptrend_pullback': 0.0
    }
}


# =============================================================================
# ENABLED/DISABLED SIGNALS (simple list for quick filtering)
# =============================================================================

ENABLED_SIGNALS = ['trend', 'rsi_divergence']
DISABLED_SIGNALS = ['nuclear_dip', 'vol_squeeze', 'consolidation', 'uptrend_pullback']


# =============================================================================
# ADVERSARIAL ROBUSTNESS (from Phase 4)
# =============================================================================

ROBUST_SIGNALS = {
    'trend': {
        'robust': True,
        'avg_degradation': 0.0,
        'tests_passed': 3,
        'note': 'Completely robust to gaussian, outliers, and jitter noise'
    },
    'rsi_divergence': {
        'robust': True,
        'avg_degradation': 0.0,
        'tests_passed': 3,
        'note': 'Completely robust to all noise types'
    },
    'dip_buy': {
        'robust': True,
        'avg_degradation': 1.4,
        'tests_passed': 3,
        'note': 'Minor degradation with gaussian noise, otherwise robust'
    }
}


# =============================================================================
# SIGNAL PARAMETERS (optimized thresholds)
# =============================================================================

@dataclass
class SignalParams:
    """Optimized signal parameters from evolution"""
    # Trend Following (Tier S)
    trend_threshold: float = 0.5
    ribbon_required: bool = True
    
    # RSI Divergence (Tier A)
    rsi_divergence_macd_confirm: bool = True
    
    # Dip Buy (Tier B - use in sideways)
    dip_buy_rsi: float = 30
    dip_buy_momentum: float = -3
    
    # Bounce (Tier B - use in bear/sideways)
    bounce_threshold: float = 8
    bounce_macd_confirm: bool = True
    
    # Momentum (Tier B)
    momentum_threshold: float = 5
    momentum_macd_confirm: bool = True
    momentum_bounce_confirm: bool = True
    
    # Exit rules
    profit_target: float = 20.0
    stop_loss: float = -10.0
    time_stop_days: int = 30


SIGNAL_PARAMS = SignalParams()


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_signal_weight(signal_name: str, regime: str = 'bull') -> float:
    """Get weight for a signal based on current regime"""
    if regime.lower() in REGIME_SIGNAL_WEIGHTS:
        return REGIME_SIGNAL_WEIGHTS[regime.lower()].get(signal_name, 0.0)
    return OPTIMAL_SIGNAL_WEIGHTS.get(signal_name, 0.0)


def is_signal_enabled(signal_name: str, regime: str = None) -> bool:
    """Check if a signal should be used"""
    # First check if globally disabled
    if signal_name in DISABLED_SIGNALS:
        return False
    
    # If regime specified, check regime-specific weight
    if regime:
        weight = get_signal_weight(signal_name, regime)
        return weight > 0.0
    
    # Otherwise check global weight
    return OPTIMAL_SIGNAL_WEIGHTS.get(signal_name, 0.0) > 0.0


def classify_regime(ret_21d: float, ribbon_bullish: bool) -> str:
    """Classify market regime for signal switching"""
    if ret_21d > 5 and ribbon_bullish:
        return 'bull'
    elif ret_21d < -5 and not ribbon_bullish:
        return 'bear'
    else:
        return 'sideways'


def get_best_signals_for_regime(regime: str, top_n: int = 3) -> List[str]:
    """Get the best signals for a given regime"""
    weights = REGIME_SIGNAL_WEIGHTS.get(regime.lower(), OPTIMAL_SIGNAL_WEIGHTS)
    sorted_signals = sorted(weights.items(), key=lambda x: x[1], reverse=True)
    return [s[0] for s in sorted_signals[:top_n] if s[1] > 0]


# =============================================================================
# SIGNAL EVALUATION FUNCTION
# =============================================================================

def evaluate_entry_signals(features: Dict, regime: str = 'bull') -> Dict[str, float]:
    """
    Evaluate all entry signals and return weighted scores.
    
    Args:
        features: Dict with keys like 'rsi', 'mom_5d', 'trend_align', 'ribbon_bullish', etc.
        regime: Current market regime ('bull', 'bear', 'sideways')
    
    Returns:
        Dict mapping signal names to weighted scores
    """
    scores = {}
    weights = REGIME_SIGNAL_WEIGHTS.get(regime, OPTIMAL_SIGNAL_WEIGHTS)
    
    # Get features (with defaults)
    rsi = features.get('rsi', 50)
    mom_5d = features.get('mom_5d', 0)
    trend_align = features.get('trend_align', 0)
    ribbon_bullish = features.get('ribbon_bullish', 0)
    macd_rising = features.get('macd_rising', 0)
    bounce = features.get('bounce', 0)
    bounce_signal = features.get('bounce_signal', 0)
    rsi_divergence = features.get('rsi_divergence', 0)
    
    # TREND signal (Tier S)
    if weights.get('trend', 0) > 0:
        if trend_align > SIGNAL_PARAMS.trend_threshold and ribbon_bullish > 0:
            scores['trend'] = 1.0 * weights['trend']
    
    # RSI DIVERGENCE signal (Tier A)
    if weights.get('rsi_divergence', 0) > 0:
        if rsi_divergence > 0 and macd_rising > 0:
            scores['rsi_divergence'] = 1.0 * weights['rsi_divergence']
    
    # DIP BUY signal (Tier B - best in sideways)
    if weights.get('dip_buy', 0) > 0:
        if rsi < SIGNAL_PARAMS.dip_buy_rsi and mom_5d < SIGNAL_PARAMS.dip_buy_momentum:
            scores['dip_buy'] = 1.0 * weights['dip_buy']
    
    # BOUNCE signal (Tier B - works in bear/sideways)
    if weights.get('bounce', 0) > 0:
        if bounce > SIGNAL_PARAMS.bounce_threshold and macd_rising > 0:
            scores['bounce'] = 1.0 * weights['bounce']
    
    # MOMENTUM signal (Tier B)
    if weights.get('momentum', 0) > 0:
        if (mom_5d > SIGNAL_PARAMS.momentum_threshold and 
            macd_rising > 0 and bounce_signal > 0):
            scores['momentum'] = 1.0 * weights['momentum']
    
    return scores


def should_enter_trade(features: Dict, regime: str = 'bull', min_score: float = 0.5) -> tuple:
    """
    Determine if we should enter a trade based on weighted signals.
    
    Returns:
        (should_buy: bool, total_score: float, triggered_signals: List[str])
    """
    scores = evaluate_entry_signals(features, regime)
    
    if not scores:
        return False, 0.0, []
    
    total_score = sum(scores.values())
    triggered_signals = list(scores.keys())
    
    return total_score >= min_score, total_score, triggered_signals


# =============================================================================
# PRINT SUMMARY ON IMPORT
# =============================================================================

if __name__ == '__main__':
    print("="*60)
    print("🎯 OPTIMIZED SIGNAL CONFIGURATION")
    print("="*60)
    print(f"\nEnabled Signals: {ENABLED_SIGNALS}")
    print(f"Disabled Signals: {DISABLED_SIGNALS}")
    print(f"\nOptimal Weights:")
    for signal, weight in OPTIMAL_SIGNAL_WEIGHTS.items():
        status = "✅" if weight > 0 else "❌"
        print(f"  {status} {signal}: {weight}")
    
    print(f"\nRegime-Specific Best Signals:")
    for regime in ['bull', 'bear', 'sideways']:
        best = get_best_signals_for_regime(regime)
        print(f"  {regime.upper()}: {', '.join(best)}")
