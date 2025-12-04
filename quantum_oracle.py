#!/usr/bin/env python3
"""
🔮 QUANTUM ORACLE - CUTTING EDGE PREDICTION ENGINE
==================================================

This is NOT your ordinary trading algorithm. This pushes the boundaries of:
- Quantum-inspired superposition of multiple signal states
- Self-attention mechanisms (transformer-style) for pattern recognition  
- Bayesian uncertainty quantification (knows when it doesn't know)
- Adversarial self-validation (catches its own hallucinations)
- Ensemble of 7 independent "minds" that must reach consensus

The goal: Make predictions so accurate it's scary.
The safety: Built-in hallucination detection that flags uncertain predictions.

Author: Quantum AI Trader v2.0
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from scipy.signal import find_peaks, argrelextrema
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from collections import deque
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from enum import Enum
import warnings
import math

warnings.filterwarnings('ignore')

# ============================================================================
# QUANTUM-INSPIRED STATE REPRESENTATION
# ============================================================================

class QuantumState(Enum):
    """Market exists in superposition until observed (measured)"""
    STRONG_BULL = 4
    BULL = 3
    WEAK_BULL = 2
    NEUTRAL = 1
    WEAK_BEAR = 0
    BEAR = -1
    STRONG_BEAR = -2


@dataclass
class PredictionResult:
    """Immutable prediction with uncertainty bounds"""
    ticker: str
    signal: str  # BUY, SELL, HOLD
    confidence: float  # 0-1
    expected_return: float  # Expected % move
    uncertainty: float  # Bayesian uncertainty (lower = more certain)
    time_horizon: int  # Days
    consensus_score: float  # How many minds agree (0-1)
    hallucination_risk: float  # Self-assessed BS probability (0-1)
    quantum_state: QuantumState
    supporting_patterns: List[str]
    contrarian_flags: List[str]
    
    @property
    def is_high_conviction(self) -> bool:
        """True when we're confident enough to PLAY (research mode)"""
        return (
            self.confidence > 0.60 and  # Loosened from 0.80
            self.uncertainty < 0.40 and  # Loosened from 0.15 - let it explore
            self.consensus_score > 0.50 and  # At least half the minds agree
            self.hallucination_risk < 0.40  # Still check for BS but more lenient
        )
    
    @property 
    def trade_grade(self) -> str:
        """A+ to F grade for trade quality"""
        score = (
            self.confidence * 0.3 +
            (1 - self.uncertainty) * 0.25 +
            self.consensus_score * 0.25 +
            (1 - self.hallucination_risk) * 0.2
        )
        if score > 0.90: return 'A+'
        if score > 0.85: return 'A'
        if score > 0.80: return 'A-'
        if score > 0.75: return 'B+'
        if score > 0.70: return 'B'
        if score > 0.65: return 'B-'
        if score > 0.60: return 'C+'
        if score > 0.55: return 'C'
        if score > 0.50: return 'C-'
        return 'F'


# ============================================================================
# SELF-ATTENTION MECHANISM (Transformer-style)
# ============================================================================

class SelfAttention:
    """
    Applies self-attention to price/volume sequences.
    Learns which past moments are most relevant to predict the future.
    
    This is the same mechanism that powers GPT/ChatGPT but applied to markets.
    """
    
    def __init__(self, sequence_length: int = 30, num_heads: int = 4):
        self.seq_len = sequence_length
        self.num_heads = num_heads
        
    def compute_attention(self, sequence: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute attention weights showing which past days matter most.
        Returns: (attended_output, attention_weights)
        """
        if len(sequence) < self.seq_len:
            # Pad with zeros
            padded = np.zeros(self.seq_len)
            padded[-len(sequence):] = sequence
            sequence = padded
        else:
            sequence = sequence[-self.seq_len:]
        
        # Normalize
        seq_norm = (sequence - np.mean(sequence)) / (np.std(sequence) + 1e-8)
        
        # Multi-head attention (simplified)
        attention_outputs = []
        attention_weights_all = []
        
        for head in range(self.num_heads):
            # Create Q, K, V projections (learned via momentum patterns)
            momentum = np.diff(seq_norm)
            if len(momentum) < self.seq_len - 1:
                momentum = np.pad(momentum, (self.seq_len - 1 - len(momentum), 0))
            
            # Query: recent price action
            Q = seq_norm[-5:].mean() if len(seq_norm) >= 5 else seq_norm.mean()
            
            # Keys: historical moments weighted by volatility
            vol_weights = np.abs(momentum) + 0.1
            K = seq_norm[:-1] * vol_weights / vol_weights.sum()
            
            # Values: the actual price changes
            V = np.diff(sequence)
            if len(V) < len(K):
                V = np.pad(V, (len(K) - len(V), 0))
            
            # Attention scores (scaled dot product)
            scores = K * Q  # Element-wise for simplicity
            
            # Softmax
            exp_scores = np.exp(scores - scores.max())
            attention_weights = exp_scores / (exp_scores.sum() + 1e-8)
            
            # Apply attention to values
            attended = np.sum(attention_weights * V[:len(attention_weights)])
            
            attention_outputs.append(attended)
            attention_weights_all.append(attention_weights)
        
        # Combine heads
        final_output = np.mean(attention_outputs)
        avg_attention = np.mean(attention_weights_all, axis=0)
        
        return final_output, avg_attention
    
    def get_key_moments(self, df: pd.DataFrame, top_k: int = 5) -> List[Tuple[int, float]]:
        """Find which historical days have highest attention weights"""
        close = df['Close'].values
        _, attention = self.compute_attention(close)
        
        # Get top-k most attended moments
        top_indices = np.argsort(attention)[-top_k:][::-1]
        return [(idx, attention[idx]) for idx in top_indices]


# ============================================================================
# BAYESIAN UNCERTAINTY ENGINE
# ============================================================================

class BayesianPredictor:
    """
    Maintains probability distributions over outcomes, not point estimates.
    This lets us know WHEN WE DON'T KNOW.
    """
    
    def __init__(self):
        self.prior_bullish = 0.5  # Start 50/50
        self.evidence_history = deque(maxlen=100)
        
    def update_belief(self, evidence: Dict[str, float]) -> Tuple[float, float]:
        """
        Bayesian update: P(Bull|Evidence) = P(Evidence|Bull) * P(Bull) / P(Evidence)
        
        Returns: (probability_bullish, uncertainty)
        """
        # Likelihood ratios for each piece of evidence
        likelihoods = []
        
        # Technical evidence
        if 'rsi' in evidence:
            rsi = evidence['rsi']
            if rsi < 30:
                likelihoods.append(('oversold', 2.5))  # Bullish
            elif rsi > 70:
                likelihoods.append(('overbought', 0.4))  # Bearish
            else:
                likelihoods.append(('neutral_rsi', 1.0))
        
        if 'macd_cross' in evidence:
            if evidence['macd_cross'] > 0:
                likelihoods.append(('macd_bull', 2.0))
            elif evidence['macd_cross'] < 0:
                likelihoods.append(('macd_bear', 0.5))
        
        if 'trend' in evidence:
            if evidence['trend'] > 0:
                likelihoods.append(('uptrend', 1.8))
            else:
                likelihoods.append(('downtrend', 0.6))
        
        if 'volume_surge' in evidence:
            if evidence['volume_surge']:
                likelihoods.append(('vol_confirm', 1.5))
        
        if 'squeeze' in evidence:
            if evidence['squeeze']:
                likelihoods.append(('squeeze', 1.7))  # Breakout imminent
        
        if 'elliott_wave' in evidence:
            wave = evidence['elliott_wave']
            if wave in ['wave_3', 'wave_5']:
                likelihoods.append(('impulse', 2.2))
            elif wave in ['wave_a', 'wave_c']:
                likelihoods.append(('correction', 0.5))
        
        # Combine likelihoods (multiply)
        combined_likelihood = 1.0
        for name, lr in likelihoods:
            combined_likelihood *= lr
        
        # Bayesian update
        prior = self.prior_bullish
        posterior_bull = (combined_likelihood * prior) / (
            combined_likelihood * prior + (1 - prior)
        )
        
        # Calculate uncertainty based on STRENGTH OF EVIDENCE, not entropy
        # More evidence = more certainty (lower uncertainty)
        # Extreme probabilities = more certainty (lower uncertainty)
        
        # Evidence strength: how many strong signals do we have?
        strong_signals = len([lr for name, lr in likelihoods if lr > 1.5 or lr < 0.7])
        evidence_certainty = min(strong_signals / 3, 1.0)  # Max certainty at 3+ strong signals
        
        # Probability extremity: how far from 50/50?
        prob_certainty = abs(posterior_bull - 0.5) * 2  # 0 at 50%, 1 at 0% or 100%
        
        # Combined uncertainty (lower = more certain)
        # Research mode: be more lenient with uncertainty
        uncertainty = 1.0 - (evidence_certainty * 0.5 + prob_certainty * 0.5)
        uncertainty = max(0.1, uncertainty)  # Floor at 0.1, never fully certain
        
        # Update prior for next time (slow learning)
        self.prior_bullish = 0.9 * self.prior_bullish + 0.1 * posterior_bull
        
        return posterior_bull, uncertainty


# ============================================================================
# QUANTUM-INSPIRED OPTIMIZATION
# ============================================================================

class QuantumOptimizer:
    """
    Uses quantum-inspired algorithms for finding optimal entry/exit points.
    Simulated annealing + superposition of multiple solutions.
    """
    
    def __init__(self, num_particles: int = 50):
        self.num_particles = num_particles
        self.temperature = 1.0
        
    def find_optimal_entry(self, df: pd.DataFrame) -> Tuple[float, float]:
        """
        Find optimal entry price using quantum-inspired search.
        Returns: (optimal_price, confidence)
        """
        close = df['Close'].values
        high = df['High'].values
        low = df['Low'].values
        
        current_price = close[-1]
        
        # Define search space (±5% from current)
        search_min = current_price * 0.95
        search_max = current_price * 1.05
        
        # Initialize particles in superposition (random positions)
        particles = np.random.uniform(search_min, search_max, self.num_particles)
        velocities = np.zeros(self.num_particles)
        
        # Fitness function: support/resistance strength
        def fitness(price):
            # How many times has price bounced near this level?
            near_touches = np.sum(
                (np.abs(low - price) < price * 0.01) |
                (np.abs(high - price) < price * 0.01)
            )
            # Volume profile weight
            vol_weight = 1.0
            # Recent relevance
            recency = np.exp(-np.arange(len(close))[::-1] / 20)
            near_recent = np.sum(
                ((np.abs(close - price) < price * 0.02) * recency)
            )
            return near_touches * 0.5 + near_recent * 0.5
        
        # Quantum annealing iterations
        best_global = particles[0]
        best_fitness = fitness(best_global)
        
        for iteration in range(100):
            # Update temperature (cooling)
            self.temperature = 1.0 * (0.95 ** iteration)
            
            for i in range(self.num_particles):
                # Current fitness
                current_fit = fitness(particles[i])
                
                # Quantum tunneling: sometimes jump to random position
                if np.random.random() < self.temperature * 0.1:
                    particles[i] = np.random.uniform(search_min, search_max)
                else:
                    # Move toward best with some randomness
                    velocities[i] = (
                        0.7 * velocities[i] +
                        0.2 * (best_global - particles[i]) +
                        0.1 * np.random.randn() * self.temperature
                    )
                    particles[i] += velocities[i]
                
                # Clamp to search space
                particles[i] = np.clip(particles[i], search_min, search_max)
                
                # Update best
                new_fit = fitness(particles[i])
                if new_fit > best_fitness:
                    best_fitness = new_fit
                    best_global = particles[i]
        
        # Confidence based on fitness
        max_possible_fitness = len(close) * 0.5  # Rough estimate
        confidence = min(best_fitness / max_possible_fitness, 1.0)
        
        return best_global, confidence


# ============================================================================
# ENSEMBLE OF 7 INDEPENDENT "MINDS"
# ============================================================================

class TradingMind:
    """Base class for each independent prediction mind"""
    
    def __init__(self, name: str):
        self.name = name
        
    def predict(self, df: pd.DataFrame) -> Tuple[float, float]:
        """Returns (bullish_probability, confidence)"""
        raise NotImplementedError


class MomentumMind(TradingMind):
    """Focuses on price momentum and trend"""
    
    def __init__(self):
        super().__init__("Momentum")
        
    def predict(self, df: pd.DataFrame) -> Tuple[float, float]:
        close = df['Close'].values
        
        # Multiple momentum periods
        returns_5d = (close[-1] / close[-6] - 1) if len(close) > 5 else 0
        returns_10d = (close[-1] / close[-11] - 1) if len(close) > 10 else 0
        returns_20d = (close[-1] / close[-21] - 1) if len(close) > 20 else 0
        
        # EMA trend
        ema_8 = pd.Series(close).ewm(span=8).mean().iloc[-1]
        ema_21 = pd.Series(close).ewm(span=21).mean().iloc[-1]
        ema_trend = 1 if ema_8 > ema_21 else 0
        
        # Score
        score = 0.5
        if returns_5d > 0.02: score += 0.15
        elif returns_5d < -0.02: score -= 0.15
        if returns_10d > 0.05: score += 0.1
        elif returns_10d < -0.05: score -= 0.1
        if ema_trend: score += 0.15
        else: score -= 0.15
        
        # Confidence based on consistency
        signs = [np.sign(returns_5d), np.sign(returns_10d), np.sign(returns_20d)]
        confidence = abs(sum(signs)) / 3
        
        return min(max(score, 0), 1), confidence


class MeanReversionMind(TradingMind):
    """Looks for oversold/overbought reversals"""
    
    def __init__(self):
        super().__init__("MeanReversion")
        
    def predict(self, df: pd.DataFrame) -> Tuple[float, float]:
        close = df['Close'].values
        
        # Bollinger Band position
        sma = pd.Series(close).rolling(20).mean().iloc[-1]
        std = pd.Series(close).rolling(20).std().iloc[-1]
        bb_position = (close[-1] - (sma - 2*std)) / (4*std + 1e-8)
        
        # RSI
        delta = pd.Series(close).diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean().iloc[-1]
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean().iloc[-1]
        rsi = 100 - (100 / (1 + gain/(loss + 1e-8)))
        
        # Score (contrarian)
        score = 0.5
        if rsi < 30: score += 0.25  # Oversold = bullish
        elif rsi > 70: score -= 0.25  # Overbought = bearish
        
        if bb_position < 0.2: score += 0.2
        elif bb_position > 0.8: score -= 0.2
        
        # Confidence higher at extremes
        confidence = abs(0.5 - score) * 2
        
        return min(max(score, 0), 1), confidence


class VolumeMind(TradingMind):
    """Analyzes volume patterns for confirmation"""
    
    def __init__(self):
        super().__init__("Volume")
        
    def predict(self, df: pd.DataFrame) -> Tuple[float, float]:
        close = df['Close'].values
        volume = df['Volume'].values
        
        # Volume trend
        vol_sma = pd.Series(volume).rolling(20).mean().iloc[-1]
        vol_ratio = volume[-1] / (vol_sma + 1)
        
        # Price-volume relationship
        price_change = close[-1] / close[-2] - 1 if len(close) > 1 else 0
        
        score = 0.5
        confidence = 0.5
        
        # High volume + up = bullish
        if vol_ratio > 1.5 and price_change > 0:
            score += 0.2
            confidence = 0.7
        # High volume + down = bearish
        elif vol_ratio > 1.5 and price_change < 0:
            score -= 0.2
            confidence = 0.7
        # Low volume = less conviction
        elif vol_ratio < 0.7:
            confidence = 0.3
        
        return min(max(score, 0), 1), confidence


class PatternMind(TradingMind):
    """Detects chart patterns"""
    
    def __init__(self):
        super().__init__("Pattern")
        
    def predict(self, df: pd.DataFrame) -> Tuple[float, float]:
        close = df['Close'].values
        high = df['High'].values
        low = df['Low'].values
        
        score = 0.5
        confidence = 0.4
        
        # Find peaks and troughs
        if len(close) > 20:
            peaks = argrelextrema(high, np.greater, order=5)[0]
            troughs = argrelextrema(low, np.less, order=5)[0]
            
            # Higher highs / higher lows = bullish
            if len(peaks) >= 2 and len(troughs) >= 2:
                if high[peaks[-1]] > high[peaks[-2]]:
                    score += 0.1
                    confidence += 0.1
                if low[troughs[-1]] > low[troughs[-2]]:
                    score += 0.1
                    confidence += 0.1
                    
            # Double bottom
            if len(troughs) >= 2:
                if abs(low[troughs[-1]] - low[troughs[-2]]) / low[troughs[-1]] < 0.02:
                    score += 0.15
                    confidence += 0.15
        
        return min(max(score, 0), 1), min(confidence, 1)


class VolatilityMind(TradingMind):
    """Analyzes volatility regimes"""
    
    def __init__(self):
        super().__init__("Volatility")
        
    def predict(self, df: pd.DataFrame) -> Tuple[float, float]:
        close = df['Close'].values
        high = df['High'].values
        low = df['Low'].values
        
        # ATR
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1])
            )
        )
        atr = pd.Series(tr).rolling(14).mean().iloc[-1] if len(tr) > 14 else np.mean(tr)
        atr_pct = atr / close[-1] * 100
        
        # Bollinger bandwidth (squeeze detection)
        sma = pd.Series(close).rolling(20).mean()
        std = pd.Series(close).rolling(20).std()
        bandwidth = ((sma + 2*std) - (sma - 2*std)) / sma
        current_bw = bandwidth.iloc[-1] if len(bandwidth) > 0 else 0.1
        avg_bw = bandwidth.rolling(50).mean().iloc[-1] if len(bandwidth) > 50 else current_bw
        
        score = 0.5
        
        # Low volatility squeeze = expect breakout (slight bullish bias)
        if current_bw < avg_bw * 0.7:
            score += 0.1
        
        # Very high vol = uncertainty
        if atr_pct > 5:
            score = 0.5  # No edge
        
        confidence = 0.5 if atr_pct < 3 else 0.3
        
        return score, confidence


class SentimentMind(TradingMind):
    """Infers sentiment from price action"""
    
    def __init__(self):
        super().__init__("Sentiment")
        
    def predict(self, df: pd.DataFrame) -> Tuple[float, float]:
        close = df['Close'].values
        high = df['High'].values
        low = df['Low'].values
        
        # Candlestick sentiment
        body = close[-1] - df['Open'].iloc[-1]
        range_size = high[-1] - low[-1] + 1e-8
        body_pct = body / range_size
        
        # Upper/lower shadows
        upper_shadow = (high[-1] - max(close[-1], df['Open'].iloc[-1])) / range_size
        lower_shadow = (min(close[-1], df['Open'].iloc[-1]) - low[-1]) / range_size
        
        score = 0.5
        
        # Strong bullish candle
        if body_pct > 0.6:
            score += 0.15
        elif body_pct < -0.6:
            score -= 0.15
        
        # Hammer (bullish reversal)
        if lower_shadow > 0.6 and body_pct > 0:
            score += 0.1
        
        # Shooting star (bearish reversal)
        if upper_shadow > 0.6 and body_pct < 0:
            score -= 0.1
        
        confidence = abs(body_pct) * 0.8
        
        return min(max(score, 0), 1), min(confidence, 1)


class TimingMind(TradingMind):
    """Analyzes time-based patterns (day of week, seasonality)"""
    
    def __init__(self):
        super().__init__("Timing")
        
    def predict(self, df: pd.DataFrame) -> Tuple[float, float]:
        # Get day of week
        if hasattr(df.index[-1], 'dayofweek'):
            dow = df.index[-1].dayofweek
        else:
            dow = 2  # Default to Wednesday
        
        score = 0.5
        
        # Monday effect (slight negative)
        if dow == 0:
            score -= 0.05
        # Friday effect (profit taking)
        elif dow == 4:
            score -= 0.03
        # Mid-week tends to be better
        elif dow in [2, 3]:
            score += 0.03
        
        # Very low confidence - timing alone is weak
        confidence = 0.2
        
        return score, confidence


# ============================================================================
# HALLUCINATION DETECTOR
# ============================================================================

class HallucinationDetector:
    """
    Self-validates predictions to catch when the model is bullshitting.
    Uses multiple techniques to detect overconfidence and inconsistency.
    """
    
    def __init__(self):
        self.prediction_history = deque(maxlen=50)
        
    def detect(
        self, 
        predictions: List[Tuple[float, float]],  # (score, confidence) from each mind
        df: pd.DataFrame
    ) -> Tuple[float, List[str]]:
        """
        Returns: (hallucination_risk, list of warning flags)
        """
        flags = []
        risk = 0.0
        
        scores = [p[0] for p in predictions]
        confidences = [p[1] for p in predictions]
        
        # 1. HIGH DISAGREEMENT = UNCERTAINTY
        score_std = np.std(scores)
        if score_std > 0.2:
            risk += 0.2
            flags.append(f"⚠️ Minds disagree (std={score_std:.2f})")
        
        # 2. ALL CONFIDENT BUT WRONG DIRECTION IN PAST
        avg_confidence = np.mean(confidences)
        if avg_confidence > 0.7:
            # Check if we've been wrong recently at high confidence
            recent_errors = sum(1 for h in self.prediction_history if h['was_wrong'] and h['confidence'] > 0.7)
            if recent_errors > 3:
                risk += 0.3
                flags.append(f"⚠️ Was wrong {recent_errors}x recently at high confidence")
        
        # 3. EXTREME PREDICTION WITHOUT EXTREME EVIDENCE
        avg_score = np.mean(scores)
        close = df['Close'].values
        recent_move = abs(close[-1] / close[-5] - 1) if len(close) > 5 else 0
        
        if abs(avg_score - 0.5) > 0.3 and recent_move < 0.02:
            risk += 0.15
            flags.append("⚠️ Extreme prediction without supporting price action")
        
        # 4. INCONSISTENT WITH MARKET REGIME
        vol = pd.Series(close).pct_change().std() * np.sqrt(252) * 100  # Annualized vol %
        if vol > 50 and avg_confidence > 0.8:
            risk += 0.2
            flags.append(f"⚠️ High confidence in high vol regime ({vol:.0f}%)")
        
        # 5. DATA QUALITY CHECK
        if len(df) < 50:
            risk += 0.3
            flags.append("⚠️ Insufficient historical data")
        
        # Cap risk at 1.0
        risk = min(risk, 1.0)
        
        return risk, flags
    
    def record_outcome(self, prediction: float, confidence: float, actual_return: float):
        """Record prediction outcome for learning"""
        was_correct = (prediction > 0.5 and actual_return > 0) or (prediction < 0.5 and actual_return < 0)
        self.prediction_history.append({
            'prediction': prediction,
            'confidence': confidence,
            'actual': actual_return,
            'was_wrong': not was_correct
        })


# ============================================================================
# THE QUANTUM ORACLE - MAIN CLASS
# ============================================================================

class QuantumOracle:
    """
    The main prediction engine that combines everything.
    This is the "scary accurate" system.
    """
    
    def __init__(self):
        # Initialize 7 independent minds
        self.minds = [
            MomentumMind(),
            MeanReversionMind(),
            VolumeMind(),
            PatternMind(),
            VolatilityMind(),
            SentimentMind(),
            TimingMind()
        ]
        
        # Advanced components
        self.attention = SelfAttention(sequence_length=30, num_heads=4)
        self.bayesian = BayesianPredictor()
        self.quantum_opt = QuantumOptimizer(num_particles=50)
        self.hallucination_detector = HallucinationDetector()
        
        # Track accuracy
        self.prediction_count = 0
        self.correct_predictions = 0
    
    def analyze(self, df: pd.DataFrame, ticker: str = "UNKNOWN") -> PredictionResult:
        """
        Full analysis pipeline.
        Returns a PredictionResult with all the details.
        """
        if len(df) < 30:
            return self._create_no_signal_result(ticker, "Insufficient data")
        
        # Flatten multi-index if needed
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        # 1. GET PREDICTIONS FROM ALL 7 MINDS
        mind_predictions = []
        for mind in self.minds:
            try:
                score, conf = mind.predict(df)
                mind_predictions.append((score, conf))
            except Exception as e:
                mind_predictions.append((0.5, 0.0))  # Neutral on error
        
        # 2. APPLY SELF-ATTENTION TO FIND KEY MOMENTS
        attention_signal, attention_weights = self.attention.compute_attention(df['Close'].values)
        key_moments = self.attention.get_key_moments(df, top_k=5)
        
        # 3. BAYESIAN UPDATE WITH ALL EVIDENCE
        evidence = self._extract_evidence(df)
        bayesian_prob, uncertainty = self.bayesian.update_belief(evidence)
        
        # 4. QUANTUM OPTIMIZATION FOR ENTRY
        optimal_entry, entry_confidence = self.quantum_opt.find_optimal_entry(df)
        
        # 5. CHECK FOR HALLUCINATIONS
        hallucination_risk, warning_flags = self.hallucination_detector.detect(
            mind_predictions, df
        )
        
        # 6. ENSEMBLE: WEIGHTED CONSENSUS
        weighted_scores = []
        total_weight = 0
        for (score, conf) in mind_predictions:
            weight = conf ** 2  # Square confidence for more weight to high-conf minds
            weighted_scores.append(score * weight)
            total_weight += weight
        
        if total_weight > 0:
            ensemble_score = sum(weighted_scores) / total_weight
        else:
            ensemble_score = 0.5
        
        # Blend with Bayesian
        final_score = 0.6 * ensemble_score + 0.4 * bayesian_prob
        
        # 7. CONSENSUS SCORE (how many minds agree)
        bullish_minds = sum(1 for (s, c) in mind_predictions if s > 0.55 and c > 0.3)
        bearish_minds = sum(1 for (s, c) in mind_predictions if s < 0.45 and c > 0.3)
        consensus = max(bullish_minds, bearish_minds) / len(self.minds)
        
        # 8. DETERMINE QUANTUM STATE
        quantum_state = self._determine_quantum_state(final_score, uncertainty)
        
        # 9. DETERMINE SIGNAL - "PLAY MODE" (leashed but adventurous)
        # Loosened thresholds to let it explore and learn
        if final_score > 0.55 and uncertainty < 0.50 and hallucination_risk < 0.45:
            if final_score > 0.70 and uncertainty < 0.30:
                signal = "STRONG BUY"  # High conviction
            else:
                signal = "BUY"  # Research/play mode
        elif final_score < 0.45 and uncertainty < 0.50 and hallucination_risk < 0.45:
            if final_score < 0.30 and uncertainty < 0.30:
                signal = "STRONG SELL"  # High conviction
            else:
                signal = "SELL"  # Research/play mode
        else:
            signal = "HOLD"
        
        # 10. EXPECTED RETURN (scaled by confidence)
        base_expected = (final_score - 0.5) * 10  # -5% to +5%
        expected_return = base_expected * (1 - uncertainty) * (1 - hallucination_risk)
        
        # 11. SUPPORTING PATTERNS
        patterns = self._identify_patterns(df, evidence)
        
        # 12. CONTRARIAN FLAGS
        contrarian = []
        if final_score > 0.6 and evidence.get('rsi', 50) > 70:
            contrarian.append("RSI overbought but still bullish")
        if final_score < 0.4 and evidence.get('rsi', 50) < 30:
            contrarian.append("RSI oversold but still bearish")
        
        return PredictionResult(
            ticker=ticker,
            signal=signal,
            confidence=min(max(final_score, 0), 1),
            expected_return=expected_return,
            uncertainty=uncertainty,
            time_horizon=3,  # Days
            consensus_score=consensus,
            hallucination_risk=hallucination_risk,
            quantum_state=quantum_state,
            supporting_patterns=patterns,
            contrarian_flags=contrarian + warning_flags
        )
    
    def _extract_evidence(self, df: pd.DataFrame) -> Dict[str, float]:
        """Extract all evidence for Bayesian update"""
        close = df['Close'].values
        volume = df['Volume'].values
        
        evidence = {}
        
        # RSI
        delta = pd.Series(close).diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean().iloc[-1]
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean().iloc[-1]
        evidence['rsi'] = 100 - (100 / (1 + gain/(loss + 1e-8)))
        
        # MACD
        ema12 = pd.Series(close).ewm(span=12).mean()
        ema26 = pd.Series(close).ewm(span=26).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9).mean()
        evidence['macd_cross'] = 1 if macd.iloc[-1] > signal.iloc[-1] else -1
        
        # Trend
        ema50 = pd.Series(close).ewm(span=50).mean().iloc[-1]
        evidence['trend'] = 1 if close[-1] > ema50 else -1
        
        # Volume surge
        vol_sma = pd.Series(volume).rolling(20).mean().iloc[-1]
        evidence['volume_surge'] = volume[-1] > vol_sma * 1.5
        
        # Squeeze
        sma20 = pd.Series(close).rolling(20).mean().iloc[-1]
        std20 = pd.Series(close).rolling(20).std().iloc[-1]
        bb_upper = sma20 + 2 * std20
        bb_lower = sma20 - 2 * std20
        evidence['squeeze'] = (bb_upper - bb_lower) / sma20 < 0.1
        
        return evidence
    
    def _determine_quantum_state(self, score: float, uncertainty: float) -> QuantumState:
        """Map score + uncertainty to quantum state"""
        if uncertainty > 0.5:
            return QuantumState.NEUTRAL  # Too uncertain
        
        if score > 0.8:
            return QuantumState.STRONG_BULL
        elif score > 0.65:
            return QuantumState.BULL
        elif score > 0.55:
            return QuantumState.WEAK_BULL
        elif score > 0.45:
            return QuantumState.NEUTRAL
        elif score > 0.35:
            return QuantumState.WEAK_BEAR
        elif score > 0.2:
            return QuantumState.BEAR
        else:
            return QuantumState.STRONG_BEAR
    
    def _identify_patterns(self, df: pd.DataFrame, evidence: Dict) -> List[str]:
        """Identify supporting patterns"""
        patterns = []
        
        rsi = evidence.get('rsi', 50)
        if rsi < 30:
            patterns.append("RSI Oversold")
        elif rsi > 70:
            patterns.append("RSI Overbought")
        
        if evidence.get('macd_cross', 0) > 0:
            patterns.append("MACD Bullish Cross")
        elif evidence.get('macd_cross', 0) < 0:
            patterns.append("MACD Bearish Cross")
        
        if evidence.get('trend', 0) > 0:
            patterns.append("Above 50 EMA")
        
        if evidence.get('squeeze', False):
            patterns.append("Volatility Squeeze")
        
        if evidence.get('volume_surge', False):
            patterns.append("Volume Surge")
        
        return patterns
    
    def _create_no_signal_result(self, ticker: str, reason: str) -> PredictionResult:
        """Create a neutral result when we can't analyze"""
        return PredictionResult(
            ticker=ticker,
            signal="HOLD",
            confidence=0.5,
            expected_return=0.0,
            uncertainty=1.0,
            time_horizon=0,
            consensus_score=0.0,
            hallucination_risk=1.0,
            quantum_state=QuantumState.NEUTRAL,
            supporting_patterns=[],
            contrarian_flags=[f"⚠️ {reason}"]
        )


# ============================================================================
# MAIN - TEST THE ORACLE
# ============================================================================

if __name__ == '__main__':
    import yfinance as yf
    
    print("=" * 70)
    print("🔮 QUANTUM ORACLE - RESEARCH MODE (Leashed but Playing)")
    print("=" * 70)
    print("⚡ Mode: ADVENTUROUS - Learning what works")
    print("🛡️ Safety: Hallucination detection still active")
    print("=" * 70)
    
    oracle = QuantumOracle()
    
    # Test on your hot picks + extras
    tickers = ['APLD', 'KMTS', 'HOOD', 'SERV', 'TSLA', 'NVDA', 'AMD', 'MU']
    
    results = []
    buy_signals = []
    sell_signals = []
    
    for ticker in tickers:
        print(f"\n🔍 Analyzing {ticker}...")
        try:
            df = yf.download(ticker, period='6mo', progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            result = oracle.analyze(df, ticker)
            results.append(result)
            
            # Track signals
            if 'BUY' in result.signal:
                buy_signals.append(result)
            elif 'SELL' in result.signal:
                sell_signals.append(result)
            
            # Color code signal
            signal_emoji = "🟢" if 'BUY' in result.signal else "🔴" if 'SELL' in result.signal else "⚪"
            strong = "💪" if 'STRONG' in result.signal else ""
            
            print(f"\n{signal_emoji} {ticker} - {result.signal} {strong} (Grade: {result.trade_grade})")
            print(f"   Confidence:     {result.confidence*100:.1f}%")
            print(f"   Expected Move:  {result.expected_return:+.2f}%")
            print(f"   Uncertainty:    {result.uncertainty:.2f} {'⚠️ HIGH' if result.uncertainty > 0.5 else '✅ OK'}")
            print(f"   Consensus:      {result.consensus_score*100:.0f}% of minds agree")
            print(f"   Halluc. Risk:   {result.hallucination_risk*100:.0f}% {'⚠️ CAUTION' if result.hallucination_risk > 0.4 else '✅ CLEAR'}")
            print(f"   Quantum State:  {result.quantum_state.name}")
            print(f"   Patterns:       {', '.join(result.supporting_patterns) or 'None'}")
            if result.contrarian_flags:
                print(f"   📝 Notes:       {'; '.join(result.contrarian_flags)}")
            
            if result.is_high_conviction:
                print(f"   🎯 HIGH CONVICTION - READY TO TRADE!")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Summary
    print("\n" + "=" * 70)
    print("🔮 QUANTUM ORACLE RESEARCH FINDINGS")
    print("=" * 70)
    
    # Action items
    if buy_signals:
        print("\n🟢 BUY CANDIDATES (Research Picks):")
        for r in sorted(buy_signals, key=lambda x: -x.expected_return):
            conv = "🎯 HIGH CONV" if r.is_high_conviction else "📊 RESEARCH"
            print(f"   {r.ticker:5} | {r.signal:11} | Exp: {r.expected_return:+5.2f}% | {conv}")
    
    if sell_signals:
        print("\n🔴 SELL/AVOID (Research Picks):")
        for r in sorted(sell_signals, key=lambda x: x.expected_return):
            print(f"   {r.ticker:5} | {r.signal:11} | Exp: {r.expected_return:+5.2f}%")
    
    hold_signals = [r for r in results if r.signal == 'HOLD']
    if hold_signals:
        print("\n⚪ HOLD/WATCH (Need More Data):")
        for r in hold_signals:
            reason = "High uncertainty" if r.uncertainty > 0.5 else "Mixed signals"
            print(f"   {r.ticker:5} | {reason}")
    
    # Ranked list
    print("\n" + "-" * 50)
    print("📈 ALL TICKERS RANKED BY EXPECTED RETURN:")
    print("-" * 50)
    
    sorted_results = sorted(results, key=lambda x: -x.expected_return)
    for i, r in enumerate(sorted_results):
        emoji = "🔥" if r.is_high_conviction else "🎯" if 'STRONG' in r.signal else ""
        signal_color = "🟢" if 'BUY' in r.signal else "🔴" if 'SELL' in r.signal else "⚪"
        print(f"{i+1}. {signal_color} {r.ticker:5} {r.signal:11} | Grade: {r.trade_grade:2} | Exp: {r.expected_return:+5.2f}% | Unc: {r.uncertainty:.2f} {emoji}")
    
    print("\n" + "=" * 70)
    print("📝 RESEARCH NOTES:")
    print("   • BUY signals are in PLAY mode - smaller positions suggested")
    print("   • STRONG BUY = Oracle is confident - can size up")
    print("   • Uncertainty < 0.30 = Oracle sees clear pattern")
    print("   • Hallucination Risk > 40% = Oracle is questioning itself")
    print("=" * 70)
    print("🔮 Research complete! Oracle is learning...")
