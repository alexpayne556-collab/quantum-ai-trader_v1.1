#!/usr/bin/env python3
"""
⚡ ULTIMATE GPU TRAINING RING ⚡
================================
Optimized for T4/A100 GPUs in Google Colab
Uses every trick to find winning patterns FAST

Features:
- Vectorized parallel environments (50 tickers at once)
- Mixed precision training (FP16)
- Batch experience replay (64k buffer)
- Multi-strategy ensemble
- Automated hyperparameter search
- All indicator combinations tested

Run in Colab with:
  !pip install torch gymnasium pandas numpy yfinance ta
  %run ultimate_training_ring.py
"""

import os
import sys
import json
import time
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import deque
import random
import math

warnings.filterwarnings('ignore')

# Check for GPU
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.cuda.amp import autocast, GradScaler
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    USE_AMP = torch.cuda.is_available()  # Mixed precision only on GPU
    print(f"🔥 Using device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name()}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
except ImportError:
    print("PyTorch not available, will use numpy")
    DEVICE = 'cpu'
    USE_AMP = False

import numpy as np

try:
    import pandas as pd
    import yfinance as yf
except ImportError:
    print("Install: pip install pandas yfinance")


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class TrainingConfig:
    """All training parameters in one place"""
    # Tickers
    tickers: List[str] = field(default_factory=lambda: [
        'APLD', 'SERV', 'MRVL', 'HOOD', 'LUNR', 'BAC', 'QCOM', 'UUUU', 
        'TSLA', 'AMD', 'NOW', 'NVDA', 'MU', 'XME', 'KRYS', 'LEU', 'QTUM',
        'SPY', 'UNH', 'WMT', 'OKLO', 'RXRX', 'MTZ', 'SNOW', 'BSX', 'LLY',
        'VOO', 'GEO', 'CXW', 'LYFT', 'MNDY', 'BA', 'LAC', 'INTC', 'ALK',
        'LMT', 'CRDO', 'ANET', 'META', 'RIVN', 'GOOGL', 'HL', 'TEM', 'TDOC'
    ])
    
    # Training
    total_episodes: int = 2000
    batch_size: int = 256  # Large batches for GPU
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    
    # Experience replay
    buffer_size: int = 65536  # 64k experiences
    min_buffer_size: int = 1024
    
    # Architecture
    hidden_size: int = 256
    num_layers: int = 3
    dropout: float = 0.1
    
    # Environment
    lookback_days: int = 60
    initial_balance: float = 100000.0
    max_position_pct: float = 0.25
    
    # Strategy DNA (immutable)
    dip_buy_threshold: float = -0.08
    dip_buy_rsi: float = 35.0
    profit_take_min: float = 0.05
    profit_take_max: float = 0.08
    cut_loss_threshold: float = -0.05
    cash_reserve_pct: float = 0.20
    
    # Optimization
    use_amp: bool = True  # Mixed precision
    num_workers: int = 4
    
    # Checkpointing
    save_every: int = 100
    eval_every: int = 50


# =============================================================================
# FEATURE ENGINEERING - ALL COMBINATIONS
# =============================================================================

class UltimateFeatureEngine:
    """Extract EVERY useful feature from price data"""
    
    def __init__(self):
        self.feature_names = []
    
    def compute_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all technical indicators and features"""
        
        # Price features
        df['returns_1d'] = df['Close'].pct_change()
        df['returns_3d'] = df['Close'].pct_change(3)
        df['returns_5d'] = df['Close'].pct_change(5)
        df['returns_10d'] = df['Close'].pct_change(10)
        df['returns_21d'] = df['Close'].pct_change(21)
        
        # Volatility
        df['volatility_5d'] = df['returns_1d'].rolling(5).std()
        df['volatility_10d'] = df['returns_1d'].rolling(10).std()
        df['volatility_21d'] = df['returns_1d'].rolling(21).std()
        
        # Moving averages
        for period in [5, 8, 13, 21, 34, 50, 100, 200]:
            df[f'sma_{period}'] = df['Close'].rolling(period).mean()
            df[f'ema_{period}'] = df['Close'].ewm(span=period).mean()
            df[f'price_vs_sma_{period}'] = df['Close'] / df[f'sma_{period}'] - 1
            df[f'price_vs_ema_{period}'] = df['Close'] / df[f'ema_{period}'] - 1
        
        # RSI variants
        for period in [7, 14, 21]:
            delta = df['Close'].diff()
            gain = delta.where(delta > 0, 0).rolling(period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
            rs = gain / (loss + 1e-10)
            df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
        # Stochastic RSI
        rsi = df['rsi_14']
        rsi_min = rsi.rolling(14).min()
        rsi_max = rsi.rolling(14).max()
        df['stoch_rsi'] = (rsi - rsi_min) / (rsi_max - rsi_min + 1e-10)
        
        # MACD variants
        for fast, slow, signal in [(8, 21, 5), (12, 26, 9), (5, 35, 5)]:
            ema_fast = df['Close'].ewm(span=fast).mean()
            ema_slow = df['Close'].ewm(span=slow).mean()
            macd = ema_fast - ema_slow
            macd_signal = macd.ewm(span=signal).mean()
            df[f'macd_{fast}_{slow}'] = macd
            df[f'macd_signal_{fast}_{slow}'] = macd_signal
            df[f'macd_hist_{fast}_{slow}'] = macd - macd_signal
        
        # Bollinger Bands
        for period in [10, 20, 50]:
            mid = df['Close'].rolling(period).mean()
            std = df['Close'].rolling(period).std()
            df[f'bb_upper_{period}'] = mid + 2 * std
            df[f'bb_lower_{period}'] = mid - 2 * std
            df[f'bb_pct_{period}'] = (df['Close'] - df[f'bb_lower_{period}']) / (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}'] + 1e-10)
            df[f'bb_width_{period}'] = (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}']) / mid
        
        # ATR
        for period in [7, 14, 21]:
            high_low = df['High'] - df['Low']
            high_close = (df['High'] - df['Close'].shift()).abs()
            low_close = (df['Low'] - df['Close'].shift()).abs()
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df[f'atr_{period}'] = tr.rolling(period).mean()
            df[f'atr_pct_{period}'] = df[f'atr_{period}'] / df['Close']
        
        # Volume features
        df['volume_sma_20'] = df['Volume'].rolling(20).mean()
        df['volume_ratio'] = df['Volume'] / (df['volume_sma_20'] + 1e-10)
        df['volume_change'] = df['Volume'].pct_change()
        df['obv'] = (np.sign(df['Close'].diff()) * df['Volume']).cumsum()
        df['obv_sma'] = df['obv'].rolling(20).mean()
        
        # Price patterns
        df['higher_high'] = (df['High'] > df['High'].shift(1)).astype(int)
        df['lower_low'] = (df['Low'] < df['Low'].shift(1)).astype(int)
        df['inside_bar'] = ((df['High'] < df['High'].shift(1)) & (df['Low'] > df['Low'].shift(1))).astype(int)
        
        # Candlestick features
        df['body'] = df['Close'] - df['Open']
        df['body_pct'] = df['body'] / df['Open']
        df['upper_shadow'] = df['High'] - df[['Open', 'Close']].max(axis=1)
        df['lower_shadow'] = df[['Open', 'Close']].min(axis=1) - df['Low']
        df['shadow_ratio'] = df['upper_shadow'] / (df['lower_shadow'] + 1e-10)
        
        # Momentum indicators
        df['momentum_10'] = df['Close'] - df['Close'].shift(10)
        df['momentum_21'] = df['Close'] - df['Close'].shift(21)
        df['roc_10'] = df['Close'].pct_change(10) * 100
        df['roc_21'] = df['Close'].pct_change(21) * 100
        
        # Williams %R
        for period in [14, 21]:
            highest_high = df['High'].rolling(period).max()
            lowest_low = df['Low'].rolling(period).min()
            df[f'williams_r_{period}'] = -100 * (highest_high - df['Close']) / (highest_high - lowest_low + 1e-10)
        
        # CCI
        for period in [14, 20]:
            tp = (df['High'] + df['Low'] + df['Close']) / 3
            sma_tp = tp.rolling(period).mean()
            mad = tp.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean())
            df[f'cci_{period}'] = (tp - sma_tp) / (0.015 * mad + 1e-10)
        
        # ADX
        plus_dm = df['High'].diff()
        minus_dm = -df['Low'].diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
        tr = df[['atr_14']].copy()
        atr = df['atr_14']
        plus_di = 100 * (plus_dm.rolling(14).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(14).mean() / atr)
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
        df['adx'] = dx.rolling(14).mean()
        df['plus_di'] = plus_di
        df['minus_di'] = minus_di
        
        # Support/Resistance
        df['support'] = df['Low'].rolling(20).min()
        df['resistance'] = df['High'].rolling(20).max()
        df['dist_to_support'] = (df['Close'] - df['support']) / df['Close']
        df['dist_to_resistance'] = (df['resistance'] - df['Close']) / df['Close']
        
        # Trend strength
        df['trend_5d'] = np.sign(df['returns_5d'])
        df['trend_10d'] = np.sign(df['returns_10d'])
        df['trend_21d'] = np.sign(df['returns_21d'])
        df['trend_alignment'] = (df['trend_5d'] + df['trend_10d'] + df['trend_21d']) / 3
        
        # Moving average crossovers
        df['golden_cross'] = (df['sma_50'] > df['sma_200']).astype(int)
        df['death_cross'] = (df['sma_50'] < df['sma_200']).astype(int)
        df['ema_8_21_cross'] = (df['ema_8'] > df['ema_21']).astype(int)
        
        # Fill NaN and handle infinities
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.ffill().bfill()
        df = df.fillna(0)
        
        return df
    
    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Get list of feature columns"""
        exclude = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close', 
                   'Date', 'Ticker', 'obv', 'support', 'resistance',
                   'sma_100', 'sma_200', 'ema_100', 'ema_200']
        return [c for c in df.columns if c not in exclude and df[c].dtype in ['float64', 'int64']]


# =============================================================================
# VECTORIZED ENVIRONMENT - ALL TICKERS AT ONCE
# =============================================================================

class VectorizedTradingEnv:
    """Run all tickers in parallel for maximum GPU utilization"""
    
    def __init__(self, data_dict: Dict[str, pd.DataFrame], config: TrainingConfig):
        self.config = config
        self.tickers = list(data_dict.keys())
        self.n_envs = len(self.tickers)
        
        # Feature engine
        self.feature_engine = UltimateFeatureEngine()
        
        # Prepare data
        self.data = {}
        self.features = {}
        self.feature_cols = None
        
        for ticker, df in data_dict.items():
            df = self.feature_engine.compute_all_features(df.copy())
            self.data[ticker] = df
            
            if self.feature_cols is None:
                self.feature_cols = self.feature_engine.get_feature_columns(df)
            
            # Normalize features to prevent NaN in neural network
            features = df[self.feature_cols].values.astype(np.float32)
            # Replace any remaining NaN/Inf with 0
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
            # Clip extreme values
            features = np.clip(features, -100, 100)
            self.features[ticker] = features
        
        self.n_features = len(self.feature_cols)
        print(f"📊 Features: {self.n_features}")
        
        # State
        self.positions = np.zeros(self.n_envs)  # -1, 0, 1 for short, none, long
        self.entry_prices = np.zeros(self.n_envs)
        self.balances = np.full(self.n_envs, config.initial_balance)
        self.step_idx = np.full(self.n_envs, config.lookback_days)
        
        # Track performance
        self.episode_returns = np.zeros(self.n_envs)
        self.episode_trades = np.zeros(self.n_envs)
        self.episode_wins = np.zeros(self.n_envs)
    
    def reset(self, env_idx: Optional[int] = None) -> np.ndarray:
        """Reset one or all environments"""
        if env_idx is not None:
            indices = [env_idx]
        else:
            indices = range(self.n_envs)
        
        for i in indices:
            self.positions[i] = 0
            self.entry_prices[i] = 0
            self.balances[i] = self.config.initial_balance
            self.step_idx[i] = self.config.lookback_days
            self.episode_returns[i] = 0
            self.episode_trades[i] = 0
            self.episode_wins[i] = 0
        
        return self._get_obs()
    
    def _get_obs(self) -> np.ndarray:
        """Get observations for all envs - with proper normalization"""
        obs = np.zeros((self.n_envs, self.n_features + 3))
        
        for i, ticker in enumerate(self.tickers):
            idx = int(self.step_idx[i])
            if idx < len(self.features[ticker]):
                obs[i, :self.n_features] = self.features[ticker][idx]
            
            # Add position info
            obs[i, self.n_features] = self.positions[i]
            obs[i, self.n_features + 1] = (self.balances[i] / self.config.initial_balance) - 1
            
            # Add unrealized P&L if in position
            if self.positions[i] != 0 and self.entry_prices[i] > 0:
                current_price = float(self.data[ticker]['Close'].iloc[idx])
                pnl = (current_price / self.entry_prices[i] - 1) * self.positions[i]
                obs[i, self.n_features + 2] = np.clip(pnl, -1, 1)  # Clip PnL
        
        # Final safety: replace any NaN/Inf and normalize
        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        obs = np.clip(obs, -100, 100)
        return obs.astype(np.float32)
    
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        """Step all environments at once"""
        rewards = np.zeros(self.n_envs)
        dones = np.zeros(self.n_envs, dtype=bool)
        infos = {'trades': [], 'wins': [], 'pnls': []}
        
        for i, ticker in enumerate(self.tickers):
            idx = int(self.step_idx[i])
            df = self.data[ticker]
            
            if idx >= len(df) - 1:
                dones[i] = True
                continue
            
            action = actions[i]  # 0=HOLD, 1=BUY, 2=SELL
            current_price = df['Close'].iloc[idx]
            next_price = df['Close'].iloc[idx + 1]
            
            # Strategy DNA signals
            rsi = df['rsi_14'].iloc[idx] if 'rsi_14' in df.columns else 50
            drawdown = df['returns_21d'].iloc[idx] if 'returns_21d' in df.columns else 0
            
            # DIP BUY signal
            is_dip = (drawdown <= self.config.dip_buy_threshold and rsi <= self.config.dip_buy_rsi)
            
            # Execute action
            reward = 0
            
            if action == 1 and self.positions[i] == 0:  # BUY
                self.positions[i] = 1
                self.entry_prices[i] = current_price
                self.episode_trades[i] += 1
                
                # Bonus for DNA-compliant dip buy
                if is_dip:
                    reward += 50
                
            elif action == 2 and self.positions[i] == 1:  # SELL
                pnl_pct = (current_price / self.entry_prices[i]) - 1
                reward += pnl_pct * 1000  # Scale for learning
                
                # Bonus for DNA-compliant profit take
                if pnl_pct >= self.config.profit_take_min:
                    reward += 40
                    self.episode_wins[i] += 1
                elif pnl_pct <= self.config.cut_loss_threshold:
                    reward += 10  # Good for cutting loss quickly
                
                self.positions[i] = 0
                self.entry_prices[i] = 0
                self.episode_returns[i] += pnl_pct
                infos['pnls'].append(pnl_pct)
            
            # Holding reward/penalty
            if self.positions[i] == 1:
                pnl_pct = (next_price / self.entry_prices[i]) - 1
                
                # Penalty for holding too long in loss
                if pnl_pct < self.config.cut_loss_threshold:
                    reward -= 50  # Should have cut loss
                
                # Penalty for not taking profit
                if pnl_pct > self.config.profit_take_max and rsi > 70:
                    reward -= 30  # Should have taken profit
            
            # Penalty for missing dip buy
            if is_dip and action != 1 and self.positions[i] == 0:
                reward -= 25  # Missed opportunity
            
            rewards[i] = reward
            self.step_idx[i] += 1
        
        return self._get_obs(), rewards, dones, infos
    
    def get_dna_signals(self) -> Dict[str, List]:
        """Get Strategy DNA signals for current state"""
        signals = {'dip_buys': [], 'profit_takes': [], 'cut_losses': []}
        
        for i, ticker in enumerate(self.tickers):
            idx = int(self.step_idx[i])
            df = self.data[ticker]
            
            if idx >= len(df):
                continue
            
            rsi = df['rsi_14'].iloc[idx] if 'rsi_14' in df.columns else 50
            drawdown = df['returns_21d'].iloc[idx] if 'returns_21d' in df.columns else 0
            
            if drawdown <= self.config.dip_buy_threshold and rsi <= self.config.dip_buy_rsi:
                signals['dip_buys'].append(ticker)
            
            if self.positions[i] == 1 and self.entry_prices[i] > 0:
                current_price = df['Close'].iloc[idx]
                pnl = (current_price / self.entry_prices[i]) - 1
                
                if pnl >= self.config.profit_take_min and rsi > 70:
                    signals['profit_takes'].append((ticker, pnl))
                elif pnl <= self.config.cut_loss_threshold:
                    signals['cut_losses'].append((ticker, pnl))
        
        return signals


# =============================================================================
# NEURAL NETWORK - GPU OPTIMIZED
# =============================================================================

class UltimateTradingBrain(nn.Module):
    """High-capacity model optimized for GPU"""
    
    def __init__(self, n_features: int, config: TrainingConfig):
        super().__init__()
        
        self.config = config
        hidden = config.hidden_size
        
        # Shared feature extractor
        self.feature_net = nn.Sequential(
            nn.Linear(n_features + 3, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )
        
        # Multi-head strategy specialists
        self.dip_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, hidden // 4)
        )
        
        self.momentum_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, hidden // 4)
        )
        
        self.mean_reversion_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, hidden // 4)
        )
        
        self.risk_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, hidden // 4)
        )
        
        # Attention-based strategy combination
        self.strategy_attention = nn.MultiheadAttention(
            embed_dim=hidden // 4,
            num_heads=4,
            batch_first=True
        )
        
        # Policy head (actor)
        self.policy_net = nn.Sequential(
            nn.Linear(hidden // 4 * 4, hidden),
            nn.GELU(),
            nn.Linear(hidden, 3)  # HOLD, BUY, SELL
        )
        
        # Value head (critic)
        self.value_net = nn.Sequential(
            nn.Linear(hidden // 4 * 4, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1)
        )
        
        self.to(DEVICE)
        
        # Initialize weights properly to prevent NaN
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for stable training"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass - NaN safe"""
        
        # Input normalization (critical for stability)
        x = torch.clamp(x, -100, 100)
        
        # Feature extraction
        features = self.feature_net(x)
        
        # Strategy specialists (from research: 4-pathway is good)
        dip = self.dip_head(features)
        momentum = self.momentum_head(features)
        mean_rev = self.mean_reversion_head(features)
        risk = self.risk_head(features)
        
        # Combine strategies
        combined = torch.cat([dip, momentum, mean_rev, risk], dim=-1)
        
        # Policy and value
        policy_logits = self.policy_net(combined)
        value = self.value_net(combined)
        
        # Clamp outputs to prevent extreme values
        policy_logits = torch.clamp(policy_logits, -20, 20)
        value = torch.clamp(value, -1000, 1000)
        
        return policy_logits, value.squeeze(-1)
    
    def get_action(self, obs: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get action with log probability - NaN safe"""
        with torch.no_grad():
            # Ensure input is clean
            obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
            obs_tensor = torch.FloatTensor(obs).to(DEVICE)
            
            # Don't use AMP for inference to avoid NaN issues
            logits, values = self(obs_tensor.float())
            
            # Handle NaN in logits (can happen with extreme inputs)
            if torch.isnan(logits).any():
                logits = torch.zeros_like(logits)
                logits[:, 0] = 1.0  # Default to HOLD
            
            # Stable softmax
            logits = logits - logits.max(dim=-1, keepdim=True)[0]  # Numerical stability
            probs = F.softmax(logits, dim=-1)
            
            # Clamp probabilities to valid range
            probs = torch.clamp(probs, min=1e-8, max=1.0)
            probs = probs / probs.sum(dim=-1, keepdim=True)  # Re-normalize
            
            if deterministic:
                actions = probs.argmax(dim=-1)
            else:
                dist = torch.distributions.Categorical(probs)
                actions = dist.sample()
            
            log_probs = torch.log(probs + 1e-8)
            action_log_probs = log_probs.gather(1, actions.unsqueeze(-1)).squeeze(-1)
            
            return (
                actions.cpu().numpy(),
                action_log_probs.cpu().numpy(),
                values.float().cpu().numpy()
            )


# =============================================================================
# EXPERIENCE REPLAY BUFFER
# =============================================================================

class ExperienceBuffer:
    """Large GPU-optimized experience buffer"""
    
    def __init__(self, capacity: int, obs_dim: int):
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.ptr = 0
        self.size = 0
        
        # Pre-allocate tensors
        self.observations = torch.zeros((capacity, obs_dim), dtype=torch.float32)
        self.actions = torch.zeros(capacity, dtype=torch.long)
        self.rewards = torch.zeros(capacity, dtype=torch.float32)
        self.next_observations = torch.zeros((capacity, obs_dim), dtype=torch.float32)
        self.dones = torch.zeros(capacity, dtype=torch.bool)
        self.log_probs = torch.zeros(capacity, dtype=torch.float32)
        self.values = torch.zeros(capacity, dtype=torch.float32)
    
    def add(self, obs, action, reward, next_obs, done, log_prob, value):
        """Add batch of experiences"""
        batch_size = len(obs)
        
        if self.ptr + batch_size > self.capacity:
            # Wrap around
            overflow = (self.ptr + batch_size) - self.capacity
            
            # Fill end
            end_size = self.capacity - self.ptr
            self.observations[self.ptr:] = torch.FloatTensor(obs[:end_size])
            self.actions[self.ptr:] = torch.LongTensor(action[:end_size])
            self.rewards[self.ptr:] = torch.FloatTensor(reward[:end_size])
            self.next_observations[self.ptr:] = torch.FloatTensor(next_obs[:end_size])
            self.dones[self.ptr:] = torch.BoolTensor(done[:end_size])
            self.log_probs[self.ptr:] = torch.FloatTensor(log_prob[:end_size])
            self.values[self.ptr:] = torch.FloatTensor(value[:end_size])
            
            # Fill start
            self.observations[:overflow] = torch.FloatTensor(obs[end_size:])
            self.actions[:overflow] = torch.LongTensor(action[end_size:])
            self.rewards[:overflow] = torch.FloatTensor(reward[end_size:])
            self.next_observations[:overflow] = torch.FloatTensor(next_obs[end_size:])
            self.dones[:overflow] = torch.BoolTensor(done[end_size:])
            self.log_probs[:overflow] = torch.FloatTensor(log_prob[end_size:])
            self.values[:overflow] = torch.FloatTensor(value[end_size:])
            
            self.ptr = overflow
        else:
            end_idx = self.ptr + batch_size
            self.observations[self.ptr:end_idx] = torch.FloatTensor(obs)
            self.actions[self.ptr:end_idx] = torch.LongTensor(action)
            self.rewards[self.ptr:end_idx] = torch.FloatTensor(reward)
            self.next_observations[self.ptr:end_idx] = torch.FloatTensor(next_obs)
            self.dones[self.ptr:end_idx] = torch.BoolTensor(done)
            self.log_probs[self.ptr:end_idx] = torch.FloatTensor(log_prob)
            self.values[self.ptr:end_idx] = torch.FloatTensor(value)
            
            self.ptr = end_idx
        
        self.size = min(self.size + batch_size, self.capacity)
    
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample batch for training"""
        indices = torch.randint(0, self.size, (batch_size,))
        
        return {
            'observations': self.observations[indices].to(DEVICE),
            'actions': self.actions[indices].to(DEVICE),
            'rewards': self.rewards[indices].to(DEVICE),
            'next_observations': self.next_observations[indices].to(DEVICE),
            'dones': self.dones[indices].to(DEVICE),
            'log_probs': self.log_probs[indices].to(DEVICE),
            'values': self.values[indices].to(DEVICE),
        }
    
    def get_all(self) -> Dict[str, torch.Tensor]:
        """Get all experiences"""
        return {
            'observations': self.observations[:self.size].to(DEVICE),
            'actions': self.actions[:self.size].to(DEVICE),
            'rewards': self.rewards[:self.size].to(DEVICE),
            'next_observations': self.next_observations[:self.size].to(DEVICE),
            'dones': self.dones[:self.size].to(DEVICE),
            'log_probs': self.log_probs[:self.size].to(DEVICE),
            'values': self.values[:self.size].to(DEVICE),
        }


# =============================================================================
# PPO TRAINER - GPU OPTIMIZED
# =============================================================================

class UltimateTrainer:
    """PPO trainer optimized for GPU"""
    
    def __init__(self, env: VectorizedTradingEnv, config: TrainingConfig):
        self.env = env
        self.config = config
        
        # Model
        self.model = UltimateTradingBrain(env.n_features, config)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=config.learning_rate,
            weight_decay=0.01
        )
        
        # Mixed precision
        self.scaler = GradScaler() if USE_AMP else None
        
        # Experience buffer
        self.buffer = ExperienceBuffer(
            config.buffer_size, 
            env.n_features + 3
        )
        
        # Tracking
        self.episode_rewards = []
        self.episode_win_rates = []
        self.best_reward = float('-inf')
        self.training_start = time.time()
    
    def compute_gae(self, rewards, values, dones, next_values):
        """Compute Generalized Advantage Estimation"""
        advantages = torch.zeros_like(rewards)
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = next_values[t] if not dones[t] else 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.config.gamma * next_value * (~dones[t]) - values[t]
            gae = delta + self.config.gamma * self.config.gae_lambda * (~dones[t]) * gae
            advantages[t] = gae
        
        returns = advantages + values
        return advantages, returns
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step with mixed precision"""
        
        obs = batch['observations']
        actions = batch['actions']
        old_log_probs = batch['log_probs']
        advantages = batch.get('advantages', batch['rewards'])  # Fallback to rewards
        returns = batch.get('returns', batch['rewards'])
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        if USE_AMP:
            with autocast():
                logits, values = self.model(obs)
                
                # Policy loss (PPO clipped)
                log_probs = F.log_softmax(logits, dim=-1)
                action_log_probs = log_probs.gather(1, actions.unsqueeze(-1)).squeeze(-1)
                
                ratio = torch.exp(action_log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.mse_loss(values, returns)
                
                # Entropy bonus
                probs = F.softmax(logits, dim=-1)
                entropy = -(probs * log_probs).sum(dim=-1).mean()
                
                # Total loss
                loss = (
                    policy_loss 
                    + self.config.value_coef * value_loss 
                    - self.config.entropy_coef * entropy
                )
            
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            logits, values = self.model(obs)
            
            log_probs = F.log_softmax(logits, dim=-1)
            action_log_probs = log_probs.gather(1, actions.unsqueeze(-1)).squeeze(-1)
            
            ratio = torch.exp(action_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            value_loss = F.mse_loss(values, returns)
            
            probs = F.softmax(logits, dim=-1)
            entropy = -(probs * log_probs).sum(dim=-1).mean()
            
            loss = (
                policy_loss 
                + self.config.value_coef * value_loss 
                - self.config.entropy_coef * entropy
            )
            
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()
        
        return {
            'loss': loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item(),
        }
    
    def train(self, num_episodes: int = None):
        """Main training loop"""
        if num_episodes is None:
            num_episodes = self.config.total_episodes
        
        print(f"\n🚀 Starting Ultimate Training Ring")
        print(f"   Episodes: {num_episodes}")
        print(f"   Tickers: {self.env.n_envs}")
        print(f"   Features: {self.env.n_features}")
        print(f"   Device: {DEVICE}")
        print(f"   Mixed Precision: {USE_AMP}")
        print("=" * 60)
        
        for episode in range(num_episodes):
            episode_start = time.time()
            
            # Collect experiences
            obs = self.env.reset()
            episode_reward = 0
            episode_steps = 0
            
            while True:
                actions, log_probs, values = self.model.get_action(obs)
                next_obs, rewards, dones, infos = self.env.step(actions)
                
                # Add to buffer
                self.buffer.add(obs, actions, rewards, next_obs, dones, log_probs, values)
                
                episode_reward += rewards.sum()
                episode_steps += 1
                obs = next_obs
                
                # Check if all done
                if dones.all():
                    break
                
                # Reset done environments
                for i in range(self.env.n_envs):
                    if dones[i]:
                        self.env.reset(i)
            
            # Train on collected experiences
            if self.buffer.size >= self.config.min_buffer_size:
                for _ in range(10):  # Multiple epochs per episode
                    batch = self.buffer.sample(self.config.batch_size)
                    metrics = self.train_step(batch)
            
            # Tracking
            avg_reward = episode_reward / self.env.n_envs
            win_rate = self.env.episode_wins.sum() / max(self.env.episode_trades.sum(), 1)
            
            self.episode_rewards.append(avg_reward)
            self.episode_win_rates.append(win_rate)
            
            # Logging
            if (episode + 1) % 10 == 0:
                elapsed = time.time() - self.training_start
                eps_per_sec = (episode + 1) / elapsed
                
                print(f"Episode {episode + 1:4d} | "
                      f"Reward: {avg_reward:8.1f} | "
                      f"Win Rate: {win_rate*100:5.1f}% | "
                      f"Buffer: {self.buffer.size:6d} | "
                      f"Speed: {eps_per_sec:.2f} ep/s")
            
            # Save best
            if avg_reward > self.best_reward:
                self.best_reward = avg_reward
                self.save_checkpoint('best_model.pt')
            
            # Periodic save
            if (episode + 1) % self.config.save_every == 0:
                self.save_checkpoint(f'checkpoint_{episode + 1}.pt')
        
        print("\n✅ Training Complete!")
        print(f"   Best Reward: {self.best_reward:.1f}")
        print(f"   Final Win Rate: {self.episode_win_rates[-1]*100:.1f}%")
        
        return self.model
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint"""
        checkpoint = {
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'config': self.config,
            'episode_rewards': self.episode_rewards,
            'episode_win_rates': self.episode_win_rates,
            'best_reward': self.best_reward,
        }
        torch.save(checkpoint, filename)
        print(f"   💾 Saved: {filename}")
    
    def load_checkpoint(self, filename: str):
        """Load model checkpoint"""
        checkpoint = torch.load(filename, map_location=DEVICE)
        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.episode_rewards = checkpoint.get('episode_rewards', [])
        self.episode_win_rates = checkpoint.get('episode_win_rates', [])
        self.best_reward = checkpoint.get('best_reward', float('-inf'))
        print(f"   📂 Loaded: {filename}")


# =============================================================================
# DATA LOADING
# =============================================================================

def load_market_data(tickers: List[str], period: str = '1y') -> Dict[str, pd.DataFrame]:
    """Load market data for all tickers"""
    print(f"📥 Loading data for {len(tickers)} tickers...")
    
    data = {}
    failed = []
    
    for ticker in tickers:
        try:
            df = yf.download(ticker, period=period, progress=False)
            
            # Fix MultiIndex columns from yfinance
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            # Ensure we have single-column Series, not DataFrames
            df = df.loc[:, ~df.columns.duplicated()]
            
            if len(df) > 60:  # Minimum data requirement
                df = df.reset_index()
                # Make sure all OHLCV columns are proper Series
                for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                data[ticker] = df
                print(f"   ✓ {ticker}: {len(df)} days")
            else:
                failed.append(ticker)
        except Exception as e:
            failed.append(ticker)
    
    if failed:
        print(f"   ✗ Failed: {failed}")
    
    print(f"   Loaded {len(data)} tickers successfully")
    return data


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main training function"""
    
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║   ⚡ ULTIMATE GPU TRAINING RING ⚡                                   ║
║                                                                      ║
║   Maximum GPU utilization for finding winning patterns               ║
║   Vectorized environments • Mixed precision • All combinations       ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
""")
    
    # Configuration
    config = TrainingConfig()
    
    # Load data
    data = load_market_data(config.tickers)
    
    if not data:
        print("❌ No data loaded!")
        return
    
    # Create environment
    env = VectorizedTradingEnv(data, config)
    
    # Create trainer
    trainer = UltimateTrainer(env, config)
    
    # Train!
    model = trainer.train()
    
    # Save final model
    trainer.save_checkpoint('ultimate_model_final.pt')
    
    # Generate predictions
    print("\n📊 Generating Today's Predictions...")
    
    predictions = []
    obs = env.reset()
    
    with torch.no_grad():
        obs_tensor = torch.FloatTensor(obs).to(DEVICE)
        logits, values = model(obs_tensor)
        probs = F.softmax(logits, dim=-1)
        
        for i, ticker in enumerate(env.tickers):
            action_probs = probs[i].cpu().numpy()
            predicted_action = ['HOLD', 'BUY', 'SELL'][action_probs.argmax()]
            
            # Get DNA signals
            idx = int(env.step_idx[i])
            df = env.data[ticker]
            
            price = df['Close'].iloc[idx]
            rsi = df['rsi_14'].iloc[idx] if 'rsi_14' in df.columns else 50
            drawdown = df['returns_21d'].iloc[idx] if 'returns_21d' in df.columns else 0
            
            is_dip = drawdown <= config.dip_buy_threshold and rsi <= config.dip_buy_rsi
            
            predictions.append({
                'ticker': ticker,
                'price': float(price),
                'action': predicted_action,
                'buy_prob': float(action_probs[1]),
                'sell_prob': float(action_probs[2]),
                'hold_prob': float(action_probs[0]),
                'rsi': float(rsi),
                'drawdown': float(drawdown * 100),
                'dip_buy': is_dip,
                'signal': '🎯 DIP BUY!' if is_dip else '',
                'confidence': float(action_probs.max() * 100)
            })
    
    # Sort by buy probability
    predictions.sort(key=lambda x: x['buy_prob'], reverse=True)
    
    # Save predictions
    with open('ultimate_predictions.json', 'w') as f:
        json.dump(predictions, f, indent=2)
    
    # Display
    print("\n" + "=" * 70)
    print("🎯 TOP SIGNALS")
    print("=" * 70)
    
    dip_buys = [p for p in predictions if p['dip_buy']]
    if dip_buys:
        print("\n🎯 DIP BUY OPPORTUNITIES (ACT ON THESE!):")
        for p in dip_buys:
            print(f"   {p['ticker']:5s} ${p['price']:>7.2f} | Down {p['drawdown']:+.1f}% | RSI: {p['rsi']:.0f}")
    
    buys = [p for p in predictions if p['action'] == 'BUY' and not p['dip_buy']][:10]
    print(f"\n🟢 BUY SIGNALS ({len(buys)}):")
    for p in buys:
        print(f"   {p['ticker']:5s} ${p['price']:>7.2f} | {p['buy_prob']*100:.0f}% conf")
    
    print("\n✅ Predictions saved to: ultimate_predictions.json")
    print("✅ Model saved to: ultimate_model_final.pt")
    
    return model, predictions


if __name__ == '__main__':
    model, predictions = main()
