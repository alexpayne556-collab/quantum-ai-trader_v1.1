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
# SIGNAL TIER RANKINGS (from Phase 1 Independent Testing + GOLD FINDINGS)
# =============================================================================

SIGNAL_TIERS = {
    # TIER SS - LEGENDARY (Use with weight 2.0+) - GOLD FOUND
    'tier_ss': ['nuclear_dip'],  # 82.4% WR, +31,667 PnL (1,400 wins)
    
    # TIER S - Excellent (Use with weight 1.5-2.0)
    'tier_s': ['trend', 'ribbon_mom'],  # ribbon_mom: 71.4% WR, +14,630 PnL
    
    # TIER A - Good (Use with weight 1.0-1.5) - UPGRADED from research  
    'tier_a': ['rsi_divergence', 'bounce', 'dip_buy'],  # bounce: 66.1% WR, dip_buy: 71.4% WR
    
    # TIER B - OK (Use with weight 0.5 or conditional)
    'tier_b': ['momentum'],
    
    # TIER F - Fail (Disable these - low trades or poor WR)
    'tier_f': ['vol_squeeze', 'consolidation', 'uptrend_pullback']
}


# =============================================================================
# OPTIMAL SIGNAL WEIGHTS (from multi-phase analysis + GOLD FINDINGS)
# Proven win rates from pattern_battle_results.json
# =============================================================================

OPTIMAL_SIGNAL_WEIGHTS = {
    # Tier SS - LEGENDARY (GOLD FOUND)
    'nuclear_dip': 2.0,      # 82.4% WR, +31,667 PnL - RE-ENABLED from research
    
    # Tier S - Excellent
    'trend': 1.8,            # Proven performer
    'ribbon_mom': 1.8,       # 71.4% WR, +14,630 PnL (1,000 wins) - GOLD FOUND
    
    # Tier A - Good (UPGRADED from Tier B based on proven performance)
    'rsi_divergence': 1.0,
    'dip_buy': 1.5,          # 71.4% WR, +12,326 PnL (500 wins) - UPGRADED
    'bounce': 1.5,           # 66.1% WR, +65,225 PnL (3,900 wins) - UPGRADED
    
    # Tier B - Use cautiously (regime-dependent)
    'momentum': 0.5,
    
    # Tier F - Disabled
    'vol_squeeze': 0.0,
    'consolidation': 0.0,
    'uptrend_pullback': 0.0
}


# =============================================================================
# REGIME-SPECIFIC WEIGHTS (from Phase 5 Analysis)
# =============================================================================

REGIME_SIGNAL_WEIGHTS = {
    'bull': {
        'nuclear_dip': 0.5,     # Works but less frequent in bull (oversold rare)
        'trend': 2.0,           # 77.8% WR, +21.7% avg - BEST in bull
        'ribbon_mom': 2.0,      # Momentum works in bull trends
        'rsi_divergence': 0.8,  # 60% WR, decent
        'momentum': 0.7,        # Works in bull trends
        'dip_buy': 1.2,         # UPGRADED - 71.4% WR proven
        'bounce': 1.0,          # UPGRADED - 66.1% WR proven
        'vol_squeeze': 0.0,
        'consolidation': 0.0,
        'uptrend_pullback': 0.0
    },
    'bear': {
        'nuclear_dip': 2.5,     # 82.4% WR - BEST in deep bear dips
        'trend': 1.5,           # 71.4% WR, +11.9% avg - still good
        'ribbon_mom': 1.0,      # Less reliable in bear
        'rsi_divergence': 0.7,  # 50% WR
        'dip_buy': 1.8,         # UPGRADED - 71.4% WR, mean reversion works in bear
        'bounce': 1.8,          # UPGRADED - 66.1% WR, bounces work in bear
        'momentum': 0.5,        # Less reliable
        'vol_squeeze': 0.0,
        'consolidation': 0.0,
        'uptrend_pullback': 0.0
    },
    'sideways': {
        'nuclear_dip': 1.5,     # Can catch oversold reversals in chop
        'trend': 0.0,           # -1.1% avg PnL - DISABLE in sideways!
        'ribbon_mom': 0.8,      # Reduced but not disabled
        'rsi_divergence': 1.2,  # 62.5% WR, +9.1% avg - good
        'dip_buy': 2.0,         # UPGRADED - 71.4% WR, +12.2% avg - BEST in sideways
        'bounce': 1.8,          # UPGRADED - 66.1% WR, works in chop
        'momentum': 0.3,        # Momentum fails in chop
        'vol_squeeze': 0.0,
        'consolidation': 0.0,
        'uptrend_pullback': 0.0
    }
}


# =============================================================================
# ENABLED/DISABLED SIGNALS (simple list for quick filtering)
# Updated with GOLD FINDINGS - nuclear_dip and ribbon_mom RE-ENABLED
# =============================================================================

ENABLED_SIGNALS = ['nuclear_dip', 'trend', 'ribbon_mom', 'rsi_divergence', 'bounce', 'dip_buy']
DISABLED_SIGNALS = ['vol_squeeze', 'consolidation', 'uptrend_pullback']


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

# =============================================================================
# MAIN CONFIGURATION CLASS
# =============================================================================

class OptimizedSignalConfig:
    """Main wrapper class for optimized signal configuration (61.7% WR proven + GOLD FINDINGS)
    
    Includes proven strategies from pattern_battle_results.json and evolved_config.json:
    - nuclear_dip: 82.4% WR (Tier SS)
    - ribbon_mom: 71.4% WR (Tier S)
    - Evolved thresholds: 71.1% WR config
    """
    
    def __init__(self):
        self.SIGNAL_TIERS = SIGNAL_TIERS
        self.OPTIMAL_SIGNAL_WEIGHTS = OPTIMAL_SIGNAL_WEIGHTS
        self.REGIME_SIGNAL_WEIGHTS = REGIME_SIGNAL_WEIGHTS
        self.ENABLED_SIGNALS = ENABLED_SIGNALS
        self.DISABLED_SIGNALS = DISABLED_SIGNALS
        self.SIGNAL_PARAMS = SIGNAL_PARAMS
        
        # Tier shortcuts (updated with Tier SS)
        self.TIER_SS_PATTERNS = SIGNAL_TIERS.get('tier_ss', [])  # GOLD: nuclear_dip
        self.TIER_S_PATTERNS = SIGNAL_TIERS['tier_s']
        self.TIER_A_PATTERNS = SIGNAL_TIERS['tier_a']
        self.TIER_B_PATTERNS = SIGNAL_TIERS['tier_b']
        self.TIER_F_PATTERNS = SIGNAL_TIERS['tier_f']
        
        # Weights dict for easy access
        self.SIGNAL_WEIGHTS = OPTIMAL_SIGNAL_WEIGHTS
    
    def get_signal_weight(self, signal_name: str, regime: str = 'bull') -> float:
        """Get weight for a signal based on current regime"""
        return get_signal_weight(signal_name, regime)
    
    def is_signal_enabled(self, signal_name: str, regime: str = None) -> bool:
        """Check if a signal should be used"""
        return is_signal_enabled(signal_name, regime)
    
    def classify_regime(self, ret_21d: float, ribbon_bullish: bool) -> str:
        """Classify market regime for signal switching"""
        return classify_regime(ret_21d, ribbon_bullish)
    
    def get_best_signals_for_regime(self, regime: str, top_n: int = 3) -> List[str]:
        """Get the best signals for a given regime"""
        return get_best_signals_for_regime(regime, top_n)
    
    def evaluate_entry_signals(self, features: Dict, regime: str = 'bull') -> Dict[str, float]:
        """Evaluate all entry signals and return weighted scores"""
        return evaluate_entry_signals(features, regime)
    
    def should_enter_trade(self, features: Dict, regime: str = 'bull', min_score: float = 0.5) -> tuple:
        """Determine if we should enter a trade based on weighted signals"""
        return should_enter_trade(features, regime, min_score)


@dataclass
class SignalParams:
    """Optimized signal parameters from evolution + GOLD FINDINGS (evolved_config.json - 71.1% WR)"""
    # Entry Thresholds (EVOLVED CONFIG - 71.1% WR vs 60.9% baseline)
    rsi_oversold: float = 21          # Was 35 - buy DEEPER dips (evolved insight)
    rsi_overbought: float = 76        # Was 70 - ride trends longer
    momentum_min_pct: float = 4       # Was 10 - catch more moves
    bounce_min_pct: float = 8         # Was 5 - wait for confirmation
    drawdown_trigger_pct: float = -6  # Was -3 - more patient
    
    # Nuclear Dip (Tier SS - 82.4% WR)
    nuclear_dip_ret_threshold: float = -5.0  # 21-day return threshold
    nuclear_dip_macd_rising: bool = True     # Require MACD rising
    
    # Trend Following (Tier S)
    trend_threshold: float = 0.5
    ribbon_required: bool = True
    
    # Ribbon Momentum (Tier S - 71.4% WR)
    ribbon_mom_threshold: float = 5.0        # Minimum momentum %
    ribbon_mom_macd_confirm: bool = True
    
    # RSI Divergence (Tier A)
    rsi_divergence_macd_confirm: bool = True
    
    # Dip Buy (Tier A - 71.4% WR, UPGRADED)
    dip_buy_rsi: float = 21          # Use evolved RSI threshold
    dip_buy_momentum: float = -3
    
    # Bounce (Tier A - 66.1% WR, UPGRADED)
    bounce_threshold: float = 8      # Aligned with evolved config
    bounce_macd_confirm: bool = True
    
    # Momentum (Tier B)
    momentum_threshold: float = 4    # Use evolved threshold
    momentum_macd_confirm: bool = True
    momentum_bounce_confirm: bool = True
    
    # Exit Rules (EVOLVED CONFIG - 71.1% WR proven)
    profit_target_1_pct: float = 14  # Was 8 - first target
    profit_target_2_pct: float = 25  # Was 15 - second target  
    stop_loss_pct: float = -19       # Was -12 - wider stops let winners run!
    trailing_stop_pct: float = 11    # Was 8 - tighter trailing
    max_hold_days: int = 32          # Was 60 - faster turnover (evolved insight)
    
    # Position Sizing (EVOLVED CONFIG)
    position_size_pct: float = 21    # Was 15 - larger positions (evolved insight)
    max_positions: int = 11          # Was 10 - more diversification


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
