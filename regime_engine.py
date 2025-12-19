#!/usr/bin/env python3
"""
REGIME ENGINE - Hidden Markov Model for Market State Detection
================================================================
Phase 1 Foundation: The "Brainstem" of the Trading Companion

Market regimes determine WHICH strategies work. This engine:
1. Detects current regime using HMM on returns + volatility
2. Validates with VIX-based classifier (robustness check)
3. Maps strategies to favorable regimes
4. Provides regime transition probabilities

Key Insight: RSI oversold edge is REGIME-DEPENDENT
- Works great in BULL regimes (bounces)
- Fails in BEAR regimes (falling knives)

Author: Quantum Trading Research Team
Date: December 20, 2025
"""

import sqlite3
import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime, timedelta
import os
import json
import warnings
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import logging

warnings.filterwarnings('ignore')

# Try to import hmmlearn, fall back to manual implementation
try:
    from hmmlearn import hmm
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    print("Note: hmmlearn not installed. Using simplified regime detection.")

# ============================================================
# CONFIGURATION
# ============================================================

DB_PATH = 'data/market_data.db'
OUTPUT_DIR = 'data/regime_analysis'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# HMM Configuration
N_REGIMES = 3  # Bull, Bear, Range
LOOKBACK_DAYS = 252 * 2  # 2 years for regime estimation

# Regime definitions
REGIME_NAMES = {
    0: 'BULL',
    1: 'BEAR', 
    2: 'RANGE'
}

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class RegimeState:
    """Current regime state with full context"""
    regime: str  # BULL, BEAR, RANGE
    regime_id: int
    probability: float  # Confidence in current regime
    volatility_state: str  # HIGH_VOL, LOW_VOL, NORMAL
    trend_strength: float  # -1 to +1
    days_in_regime: int
    transition_probs: Dict[str, float]  # P(next regime | current)
    description: str
    
@dataclass
class RegimeAnalysis:
    """Full regime analysis result"""
    current_state: RegimeState
    regime_history: pd.DataFrame
    strategy_recommendations: Dict[str, str]
    edge_by_regime: Dict[str, Dict]


# ============================================================
# HIDDEN MARKOV MODEL REGIME DETECTOR
# ============================================================

class HMMRegimeDetector:
    """
    3-State Hidden Markov Model for regime detection.
    
    States:
    - BULL: Positive drift, moderate volatility
    - BEAR: Negative drift, high volatility  
    - RANGE: Zero drift, low volatility
    
    Observable features:
    - Daily returns
    - Rolling volatility
    - Trend strength
    """
    
    def __init__(self, n_regimes: int = N_REGIMES):
        self.n_regimes = n_regimes
        self.model = None
        self.fitted = False
        self.regime_stats = {}
        
    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare HMM observable features from price data"""
        # Daily returns
        returns = df['close'].pct_change().fillna(0)
        
        # Rolling volatility (20-day)
        volatility = returns.rolling(20).std().fillna(returns.std())
        
        # Trend strength (normalized momentum)
        momentum = df['close'].pct_change(20).fillna(0)
        
        # Volume trend
        vol_ma = df['volume'].rolling(20).mean()
        vol_ratio = (df['volume'] / vol_ma).fillna(1).clip(0.1, 10)
        
        # Stack features
        features = np.column_stack([
            returns.values,
            volatility.values,
            momentum.values,
            np.log(vol_ratio.values)
        ])
        
        # Handle any NaN/Inf
        features = np.nan_to_num(features, nan=0, posinf=0, neginf=0)
        
        return features
    
    def fit(self, df: pd.DataFrame):
        """Fit HMM to historical data"""
        features = self.prepare_features(df)
        
        if HMM_AVAILABLE:
            # Use hmmlearn
            self.model = hmm.GaussianHMM(
                n_components=self.n_regimes,
                covariance_type='full',
                n_iter=100,
                random_state=42
            )
            self.model.fit(features)
        else:
            # Simplified: cluster-based regime detection
            self._fit_simplified(features, df)
        
        self.fitted = True
        self._compute_regime_stats(df, features)
        
    def _fit_simplified(self, features: np.ndarray, df: pd.DataFrame):
        """Simplified regime detection without HMM"""
        # Use returns and volatility percentiles
        returns = df['close'].pct_change().fillna(0)
        volatility = returns.rolling(20).std().fillna(returns.std())
        
        # Simple rule-based classification
        self.return_thresholds = {
            'bull': returns.quantile(0.6),
            'bear': returns.quantile(0.4)
        }
        self.vol_threshold = volatility.median()
        
    def _compute_regime_stats(self, df: pd.DataFrame, features: np.ndarray):
        """Compute statistics for each regime"""
        if HMM_AVAILABLE and self.model is not None:
            regimes = self.model.predict(features)
        else:
            regimes = self._predict_simplified(df)
        
        df_with_regime = df.copy()
        df_with_regime['regime'] = regimes
        df_with_regime['returns'] = df_with_regime['close'].pct_change()
        
        for regime_id in range(self.n_regimes):
            regime_data = df_with_regime[df_with_regime['regime'] == regime_id]
            
            self.regime_stats[regime_id] = {
                'name': REGIME_NAMES.get(regime_id, f'REGIME_{regime_id}'),
                'avg_return': regime_data['returns'].mean() * 252,  # Annualized
                'volatility': regime_data['returns'].std() * np.sqrt(252),
                'sharpe': (regime_data['returns'].mean() / regime_data['returns'].std() * np.sqrt(252)) if regime_data['returns'].std() > 0 else 0,
                'frequency': len(regime_data) / len(df_with_regime),
                'avg_duration': self._calc_avg_duration(regimes, regime_id)
            }
            
    def _calc_avg_duration(self, regimes: np.ndarray, target_regime: int) -> float:
        """Calculate average duration of regime stays"""
        durations = []
        current_duration = 0
        
        for r in regimes:
            if r == target_regime:
                current_duration += 1
            else:
                if current_duration > 0:
                    durations.append(current_duration)
                current_duration = 0
        
        if current_duration > 0:
            durations.append(current_duration)
            
        return np.mean(durations) if durations else 0
    
    def predict(self, df: pd.DataFrame) -> Tuple[int, np.ndarray]:
        """
        Predict current regime and get probability distribution.
        
        Returns:
            (regime_id, probabilities array)
        """
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if HMM_AVAILABLE and self.model is not None:
            features = self.prepare_features(df)
            regime = self.model.predict(features)[-1]
            
            # Get probability distribution
            log_probs = self.model.predict_proba(features)
            probs = log_probs[-1]
        else:
            regime = self._predict_simplified(df)[-1]
            # Simplified probability (just confidence in current)
            probs = np.zeros(self.n_regimes)
            probs[regime] = 0.8
            probs[(regime + 1) % self.n_regimes] = 0.1
            probs[(regime + 2) % self.n_regimes] = 0.1
        
        return regime, probs
    
    def _predict_simplified(self, df: pd.DataFrame) -> np.ndarray:
        """Simplified regime prediction without HMM"""
        returns = df['close'].pct_change().fillna(0)
        volatility = returns.rolling(20).std().fillna(returns.std())
        momentum = df['close'].pct_change(20).fillna(0)
        
        regimes = np.zeros(len(df), dtype=int)
        
        for i in range(len(df)):
            ret = returns.iloc[i] if i < len(returns) else 0
            vol = volatility.iloc[i] if i < len(volatility) else volatility.median()
            mom = momentum.iloc[i] if i < len(momentum) else 0
            
            # Rule-based classification
            if mom > 0.02 and ret > -0.02:
                regimes[i] = 0  # BULL
            elif mom < -0.02 or (ret < -0.02 and vol > self.vol_threshold):
                regimes[i] = 1  # BEAR
            else:
                regimes[i] = 2  # RANGE
        
        return regimes
    
    def get_transition_matrix(self) -> np.ndarray:
        """Get regime transition probability matrix"""
        if HMM_AVAILABLE and self.model is not None:
            return self.model.transmat_
        else:
            # Empirical estimate (placeholder)
            return np.array([
                [0.92, 0.05, 0.03],  # BULL -> BULL/BEAR/RANGE
                [0.08, 0.85, 0.07],  # BEAR -> ...
                [0.10, 0.10, 0.80]   # RANGE -> ...
            ])


# ============================================================
# VIX-BASED REGIME CLASSIFIER (Robustness Check)
# ============================================================

class VIXRegimeClassifier:
    """
    Alternative regime classifier using volatility levels.
    Used as robustness check against HMM.
    
    Regimes based on VIX/volatility percentiles:
    - LOW_VOL: VIX < 15 (calm markets)
    - NORMAL: 15 <= VIX < 25
    - HIGH_VOL: VIX >= 25 (stressed markets)
    - EXTREME: VIX >= 35 (panic)
    """
    
    def __init__(self):
        self.vol_percentiles = {}
        
    def fit(self, df: pd.DataFrame):
        """Compute volatility percentiles from historical data"""
        returns = df['close'].pct_change()
        volatility = returns.rolling(20).std() * np.sqrt(252) * 100  # Annualized %
        
        self.vol_percentiles = {
            'p25': volatility.quantile(0.25),
            'p50': volatility.quantile(0.50),
            'p75': volatility.quantile(0.75),
            'p90': volatility.quantile(0.90)
        }
        
    def classify(self, current_vol: float) -> str:
        """Classify current volatility state"""
        if current_vol < self.vol_percentiles.get('p25', 12):
            return 'LOW_VOL'
        elif current_vol < self.vol_percentiles.get('p75', 20):
            return 'NORMAL'
        elif current_vol < self.vol_percentiles.get('p90', 30):
            return 'HIGH_VOL'
        else:
            return 'EXTREME'


# ============================================================
# STRATEGY-REGIME MAPPER
# ============================================================

class StrategyRegimeMapper:
    """
    Maps trading strategies to favorable/unfavorable regimes.
    
    Key insight: Not all strategies work in all regimes!
    """
    
    STRATEGY_REGIME_MAP = {
        # Momentum strategies
        'Near52WkHigh': {
            'favorable': ['BULL'],
            'unfavorable': ['BEAR'],
            'rationale': 'Momentum works in trends, fails in reversals'
        },
        'Momentum_Quality': {
            'favorable': ['BULL'],
            'unfavorable': ['BEAR', 'RANGE'],
            'rationale': 'Requires trending market'
        },
        
        # Mean reversion strategies
        'RSI_Oversold': {
            'favorable': ['BULL', 'RANGE'],
            'unfavorable': ['BEAR'],
            'rationale': 'Oversold bounces in bull, catching knives in bear'
        },
        'RSI_Oversold_VolSpike': {
            'favorable': ['BULL', 'RANGE', 'BEAR'],  # Works even in bear with vol spike
            'unfavorable': [],
            'rationale': 'Volume spike = capitulation, works in all regimes'
        },
        'Mean_Reversion_Oversold': {
            'favorable': ['RANGE'],
            'unfavorable': ['BULL', 'BEAR'],
            'rationale': 'Mean reversion needs range-bound market'
        },
        
        # Low volatility strategies  
        'LowVol_After2Down': {
            'favorable': ['BULL', 'RANGE'],
            'unfavorable': ['BEAR'],
            'rationale': 'Low vol stocks bounce in calm markets'
        },
        
        # Breakout strategies
        'Breakout_52wHigh': {
            'favorable': ['BULL'],
            'unfavorable': ['BEAR', 'RANGE'],
            'rationale': 'Breakouts need momentum environment'
        },
        
        # Calendar strategies
        'FOMC_Week': {
            'favorable': ['BULL', 'RANGE'],
            'unfavorable': ['BEAR'],  # Fed can't save a bear market
            'rationale': 'Fed usually reassuring in normal times'
        },
        'SantaClaus': {
            'favorable': [],  # Bearish signal!
            'unfavorable': ['BULL', 'RANGE', 'BEAR'],
            'rationale': 'Data shows Santa rally is actually bearish'
        },
        'TaxLossSelling': {
            'favorable': ['BULL', 'RANGE'],
            'unfavorable': ['BEAR'],
            'rationale': 'Tax-loss bounce needs buying pressure'
        }
    }
    
    @classmethod
    def get_recommendations(cls, regime: str) -> Dict[str, str]:
        """Get strategy recommendations for current regime"""
        recommendations = {}
        
        for strategy, mapping in cls.STRATEGY_REGIME_MAP.items():
            if regime in mapping['favorable']:
                recommendations[strategy] = f"✅ FAVORABLE - {mapping['rationale']}"
            elif regime in mapping['unfavorable']:
                recommendations[strategy] = f"❌ AVOID - {mapping['rationale']}"
            else:
                recommendations[strategy] = f"⚠️ NEUTRAL"
        
        return recommendations


# ============================================================
# REGIME BACKTESTER - Test Edge by Regime
# ============================================================

class RegimeBacktester:
    """
    Backtest strategies conditioned on regime.
    
    Key question: Does RSI oversold ONLY work in BULL regime?
    """
    
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self.df = None
        
    def load_data(self):
        """Load market data"""
        conn = sqlite3.connect(self.db_path)
        self.df = pd.read_sql("SELECT * FROM ohlcv WHERE ticker = 'SPY'", conn)
        conn.close()
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df = self.df.sort_values('date').reset_index(drop=True)
        
    def test_strategy_by_regime(self, strategy_func, regimes: np.ndarray, 
                                 hold_period: int = 5) -> Dict:
        """
        Test a strategy's performance in each regime.
        
        Args:
            strategy_func: Function that returns signal series (1=long, 0=neutral)
            regimes: Array of regime labels
            hold_period: Days to hold
        """
        if self.df is None:
            self.load_data()
            
        df = self.df.copy()
        df['regime'] = regimes[:len(df)]
        df['signal'] = strategy_func(df)
        df['fwd_return'] = df['close'].shift(-hold_period) / df['close'] - 1
        
        results = {}
        
        for regime_id in range(N_REGIMES):
            regime_name = REGIME_NAMES.get(regime_id, f'REGIME_{regime_id}')
            regime_data = df[(df['regime'] == regime_id) & (df['signal'] == 1)]
            
            if len(regime_data) < 30:
                results[regime_name] = {'n': len(regime_data), 'insufficient_data': True}
                continue
            
            returns = regime_data['fwd_return'].dropna()
            
            if len(returns) > 0:
                mean_ret = returns.mean()
                std_ret = returns.std()
                t_stat = mean_ret / (std_ret / np.sqrt(len(returns))) if std_ret > 0 else 0
                
                results[regime_name] = {
                    'n': len(returns),
                    'avg_return': mean_ret * 100,
                    'std': std_ret * 100,
                    't_stat': t_stat,
                    'win_rate': (returns > 0).mean() * 100,
                    'significant': abs(t_stat) > 3.0
                }
        
        return results


# ============================================================
# MAIN REGIME ENGINE
# ============================================================

class RegimeEngine:
    """
    Main orchestrator for regime detection and analysis.
    
    Usage:
        engine = RegimeEngine()
        state = engine.analyze()
        print(state.current_state.regime)
    """
    
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self.df = None
        self.spy_data = None
        
        # Components
        self.hmm_detector = HMMRegimeDetector()
        self.vix_classifier = VIXRegimeClassifier()
        self.backtester = RegimeBacktester(db_path)
        
        # Results
        self.current_state = None
        self.regime_history = None
        
    def load_data(self):
        """Load SPY data for regime detection"""
        logger.info(f"Loading market data from {self.db_path}")
        conn = sqlite3.connect(self.db_path)
        
        # Load SPY for regime detection
        self.spy_data = pd.read_sql(
            "SELECT * FROM ohlcv WHERE ticker = 'SPY' ORDER BY date",
            conn
        )
        conn.close()
        
        self.spy_data['date'] = pd.to_datetime(self.spy_data['date'])
        logger.info(f"Loaded {len(self.spy_data)} SPY records")
        
    def fit(self):
        """Fit regime detection models"""
        if self.spy_data is None:
            self.load_data()
        
        logger.info("Fitting HMM regime detector...")
        self.hmm_detector.fit(self.spy_data)
        
        logger.info("Fitting VIX classifier...")
        self.vix_classifier.fit(self.spy_data)
        
        logger.info("Models fitted successfully")
        
    def detect_current_regime(self) -> RegimeState:
        """Detect current market regime"""
        if not self.hmm_detector.fitted:
            self.fit()
        
        # HMM prediction
        regime_id, probs = self.hmm_detector.predict(self.spy_data)
        regime_name = REGIME_NAMES.get(regime_id, 'UNKNOWN')
        
        # VIX/volatility classification
        returns = self.spy_data['close'].pct_change()
        current_vol = returns.rolling(20).std().iloc[-1] * np.sqrt(252) * 100
        vol_state = self.vix_classifier.classify(current_vol)
        
        # Trend strength
        momentum_20 = self.spy_data['close'].pct_change(20).iloc[-1]
        trend_strength = np.clip(momentum_20 * 10, -1, 1)  # Scale to -1, +1
        
        # Days in current regime
        features = self.hmm_detector.prepare_features(self.spy_data)
        if HMM_AVAILABLE and self.hmm_detector.model is not None:
            all_regimes = self.hmm_detector.model.predict(features)
        else:
            all_regimes = self.hmm_detector._predict_simplified(self.spy_data)
        
        days_in_regime = 1
        for i in range(len(all_regimes) - 2, -1, -1):
            if all_regimes[i] == regime_id:
                days_in_regime += 1
            else:
                break
        
        # Transition probabilities
        trans_matrix = self.hmm_detector.get_transition_matrix()
        transition_probs = {
            REGIME_NAMES[i]: trans_matrix[regime_id, i]
            for i in range(self.hmm_detector.n_regimes)
        }
        
        # Generate description
        descriptions = {
            'BULL': f"Bullish regime with {vol_state.lower().replace('_', ' ')} volatility. Momentum and trend strategies favored.",
            'BEAR': f"Bearish regime with {vol_state.lower().replace('_', ' ')} volatility. Defensive positioning recommended.",
            'RANGE': f"Range-bound market with {vol_state.lower().replace('_', ' ')} volatility. Mean reversion strategies favored."
        }
        
        self.current_state = RegimeState(
            regime=regime_name,
            regime_id=regime_id,
            probability=probs[regime_id],
            volatility_state=vol_state,
            trend_strength=trend_strength,
            days_in_regime=days_in_regime,
            transition_probs=transition_probs,
            description=descriptions.get(regime_name, "Unknown regime")
        )
        
        return self.current_state
    
    def get_regime_history(self) -> pd.DataFrame:
        """Get historical regime classifications"""
        if not self.hmm_detector.fitted:
            self.fit()
        
        features = self.hmm_detector.prepare_features(self.spy_data)
        
        if HMM_AVAILABLE and self.hmm_detector.model is not None:
            regimes = self.hmm_detector.model.predict(features)
        else:
            regimes = self.hmm_detector._predict_simplified(self.spy_data)
        
        history = self.spy_data[['date', 'close']].copy()
        history['regime_id'] = regimes
        history['regime'] = history['regime_id'].map(REGIME_NAMES)
        history['returns'] = history['close'].pct_change()
        
        self.regime_history = history
        return history
    
    def analyze(self) -> RegimeAnalysis:
        """Run full regime analysis"""
        logger.info("="*60)
        logger.info("REGIME ENGINE - Full Analysis")
        logger.info("="*60)
        
        # Load and fit
        if self.spy_data is None:
            self.load_data()
        if not self.hmm_detector.fitted:
            self.fit()
        
        # Current state
        logger.info("\n📊 Detecting current regime...")
        current = self.detect_current_regime()
        logger.info(f"   Regime: {current.regime} (probability: {current.probability:.0%})")
        logger.info(f"   Volatility: {current.volatility_state}")
        logger.info(f"   Days in regime: {current.days_in_regime}")
        
        # Regime history
        logger.info("\n📈 Generating regime history...")
        history = self.get_regime_history()
        
        # Regime statistics
        logger.info("\n📉 Regime Statistics:")
        for regime_id, stats in self.hmm_detector.regime_stats.items():
            logger.info(f"   {stats['name']}: "
                       f"Return={stats['avg_return']:.1%}/yr, "
                       f"Vol={stats['volatility']:.1%}, "
                       f"Sharpe={stats['sharpe']:.2f}, "
                       f"Freq={stats['frequency']:.0%}")
        
        # Strategy recommendations
        logger.info("\n🎯 Strategy Recommendations:")
        recommendations = StrategyRegimeMapper.get_recommendations(current.regime)
        for strategy, rec in recommendations.items():
            logger.info(f"   {strategy}: {rec}")
        
        # Save outputs
        history.to_csv(f'{OUTPUT_DIR}/regime_history.csv', index=False)
        
        with open(f'{OUTPUT_DIR}/current_regime.json', 'w') as f:
            json.dump(asdict(current), f, indent=2, default=str)
        
        logger.info(f"\n📁 Saved to {OUTPUT_DIR}/")
        
        return RegimeAnalysis(
            current_state=current,
            regime_history=history,
            strategy_recommendations=recommendations,
            edge_by_regime=self.hmm_detector.regime_stats
        )
    
    def validate_strategy_by_regime(self, strategy_name: str = 'RSI_Oversold'):
        """
        Validate if a strategy's edge is regime-dependent.
        
        Key test from DeepSeek: Does RSI oversold only work in BULL?
        """
        logger.info(f"\n🔬 Validating {strategy_name} by regime...")
        
        if self.spy_data is None:
            self.load_data()
        if not self.hmm_detector.fitted:
            self.fit()
        
        # Get regimes
        features = self.hmm_detector.prepare_features(self.spy_data)
        if HMM_AVAILABLE and self.hmm_detector.model is not None:
            regimes = self.hmm_detector.model.predict(features)
        else:
            regimes = self.hmm_detector._predict_simplified(self.spy_data)
        
        df = self.spy_data.copy()
        df['regime'] = regimes
        df['returns'] = df['close'].pct_change()
        
        # Calculate RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta).where(delta < 0, 0).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Signal: RSI < 30
        df['signal'] = (df['rsi'] < 30).astype(int)
        df['fwd_5'] = df['close'].shift(-5) / df['close'] - 1
        
        results = {}
        
        for regime_id in range(N_REGIMES):
            regime_name = REGIME_NAMES.get(regime_id)
            regime_data = df[(df['regime'] == regime_id) & (df['signal'] == 1)]
            returns = regime_data['fwd_5'].dropna()
            
            if len(returns) >= 30:
                mean_ret = returns.mean()
                std_ret = returns.std()
                t_stat = mean_ret / (std_ret / np.sqrt(len(returns))) if std_ret > 0 else 0
                
                results[regime_name] = {
                    'n': len(returns),
                    'avg_return': mean_ret * 100,
                    't_stat': t_stat,
                    'significant': abs(t_stat) > 2.0
                }
                
                status = "✅ WORKS" if t_stat > 2 else ("❌ FAILS" if t_stat < -2 else "⚠️ NEUTRAL")
                logger.info(f"   {regime_name}: n={len(returns)}, "
                           f"return={mean_ret*100:.2f}%, t={t_stat:.2f} {status}")
            else:
                results[regime_name] = {'n': len(returns), 'insufficient_data': True}
                logger.info(f"   {regime_name}: Insufficient data (n={len(returns)})")
        
        return results


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Regime Engine - Market State Detection')
    parser.add_argument('--validate', type=str, default=None,
                       help='Validate strategy by regime (e.g., RSI_Oversold)')
    args = parser.parse_args()
    
    engine = RegimeEngine()
    analysis = engine.analyze()
    
    # Print summary
    print("\n" + "="*60)
    print("📊 REGIME ENGINE SUMMARY")
    print("="*60)
    
    state = analysis.current_state
    print(f"\n🌍 Current Regime: {state.regime}")
    print(f"   Confidence: {state.probability:.0%}")
    print(f"   Volatility State: {state.volatility_state}")
    print(f"   Trend Strength: {state.trend_strength:+.2f}")
    print(f"   Days in Regime: {state.days_in_regime}")
    print(f"\n   {state.description}")
    
    print(f"\n🔄 Transition Probabilities:")
    for regime, prob in state.transition_probs.items():
        print(f"   → {regime}: {prob:.0%}")
    
    print(f"\n📈 Regime Statistics:")
    for regime_id, stats in analysis.edge_by_regime.items():
        print(f"   {stats['name']}: {stats['avg_return']:.1%}/yr, "
              f"Sharpe={stats['sharpe']:.2f}, Freq={stats['frequency']:.0%}")
    
    # Validate RSI by regime (key DeepSeek insight)
    if args.validate:
        engine.validate_strategy_by_regime(args.validate)
    else:
        print("\n🔬 Validating RSI_Oversold by regime (DeepSeek insight):")
        engine.validate_strategy_by_regime('RSI_Oversold')
    
    print(f"\n📁 Output saved to: {OUTPUT_DIR}/")


if __name__ == '__main__':
    main()
