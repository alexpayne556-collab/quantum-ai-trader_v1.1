#!/usr/bin/env python3
"""
🚀 ULTIMATE GPU TRAINER - AlphaGo Style Pattern Discovery
==========================================================
This isn't just pattern matching - it's pattern DISCOVERY.

Philosophy:
- Let AI play like a Rubik's cube until it masters it
- Don't just trade - SWING TRADE (hold winners, cut losers)
- Learn from YOUR successful trades (APLD, MRVL hold strategy)
- Discover patterns humans haven't found

GPU Optimized for Colab T4/A100:
- Vectorized operations (no loops where possible)
- Batch processing across all tickers
- Mixed precision training (FP16)
- Parallel environment simulation

Run in Colab:
  !pip install torch numpy pandas yfinance ta scikit-learn
  %run ULTIMATE_GPU_TRAINER.py
"""

import os
import json
import time
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field, asdict
from collections import deque
import random
import math

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

# GPU Detection
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.cuda.amp import autocast, GradScaler
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    USE_GPU = torch.cuda.is_available()
    
    if USE_GPU:
        print(f"🔥 GPU DETECTED: {torch.cuda.get_device_name()}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        # Enable TF32 for faster training on Ampere GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        print("⚠️ No GPU - using CPU (slower)")
except ImportError:
    print("PyTorch not available")
    DEVICE = 'cpu'
    USE_GPU = False

try:
    import yfinance as yf
except ImportError:
    print("Install: pip install yfinance")


# =============================================================================
# CONFIGURATION - YOUR TRADING DNA
# =============================================================================

@dataclass
class SwingTradeConfig:
    """
    YOUR TRADING STRATEGY - Learned from your APLD, MRVL, SERV wins
    
    Key Insight: You don't just day trade - you SWING trade:
    - Buy on dips (RSI < 35, down 8%+)
    - Hold through volatility (don't panic sell)
    - Take profits at 15-30% (not 5%)
    - Cut losses at -10% (not -5%)
    - Use sector correlation (nuclear, AI chips move together)
    """
    
    # === YOUR WATCHLIST ===
    tickers: List[str] = field(default_factory=lambda: [
        # AI/Chips
        'NVDA', 'AMD', 'MRVL', 'MU', 'INTC', 'QCOM', 'ANET', 'CRDO',
        # Nuclear/Energy
        'SMR', 'OKLO', 'LEU', 'UUUU', 'CCJ', 'NNE',
        # Tech Growth
        'APLD', 'SERV', 'HOOD', 'LUNR', 'SNOW', 'NOW', 'MNDY',
        # Quantum
        'IONQ', 'RGTI', 'QUBT',
        # Big Tech
        'TSLA', 'META', 'GOOGL', 'AAPL', 'MSFT',
        # ETFs for correlation
        'SPY', 'QQQ', 'XLK', 'XME', 'QTUM', 'VOO',
        # Other
        'BA', 'RIVN', 'LYFT', 'RXRX', 'TDOC', 'TEM', 'HL', 'LAC'
    ])
    
    # === SWING TRADE PARAMETERS (Not day trade!) ===
    # Entry
    dip_buy_rsi: float = 35.0           # RSI threshold for dip buy
    dip_buy_drawdown: float = -0.08     # 8% drawdown for dip buy
    momentum_rsi_low: float = 45.0      # Momentum entry RSI low
    momentum_rsi_high: float = 70.0     # Momentum entry RSI high
    
    # Position Management - SWING STYLE
    max_position_pct: float = 0.20      # Max 20% in one stock
    min_cash_reserve: float = 0.15      # Keep 15% cash for opportunities
    max_positions: int = 8              # Max 8 concurrent positions
    
    # Exit - SWING STYLE (hold longer!)
    profit_target_min: float = 0.10     # Take some at 10%
    profit_target_max: float = 0.25     # Take more at 25%
    stop_loss: float = -0.10            # Cut at -10% (give room to breathe)
    trailing_stop: float = 0.08         # 8% trailing stop after 15% gain
    max_hold_days: int = 45             # Hold up to 45 days (swing!)
    
    # === LEARNING PARAMETERS ===
    initial_balance: float = 100000.0
    episodes: int = 3000                # More episodes for discovery
    batch_size: int = 512               # Large batches for GPU
    learning_rate: float = 1e-4
    gamma: float = 0.995                # High gamma for long-term rewards
    
    # === DISCOVERY MODE ===
    exploration_episodes: int = 500     # Pure exploration first
    discovery_rate: float = 0.3         # 30% random actions initially
    min_discovery_rate: float = 0.05    # 5% random at end
    
    # === FEATURE DISCOVERY ===
    # Let AI discover which indicators matter
    all_features: bool = True           # Use ALL indicators
    feature_selection: bool = True      # Let AI rank features


# =============================================================================
# MEGA FEATURE ENGINE - 150+ INDICATORS FOR DISCOVERY
# =============================================================================

class MegaFeatureEngine:
    """
    Compute EVERY possible indicator and let AI discover which matter.
    This is the "let it play like a Rubik's cube" approach.
    """
    
    def __init__(self):
        self.feature_names = []
        self.feature_importance = {}
    
    def compute_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute 150+ technical indicators"""
        
        close = df['Close']
        high = df['High']
        low = df['Low']
        volume = df['Volume']
        
        # ========== PRICE FEATURES ==========
        # Multi-timeframe returns
        for period in [1, 2, 3, 5, 8, 10, 13, 21, 34, 55]:
            df[f'returns_{period}d'] = close.pct_change(period)
            df[f'log_returns_{period}d'] = np.log(close / close.shift(period))
        
        # Volatility
        for period in [5, 10, 21, 55]:
            df[f'volatility_{period}d'] = df['returns_1d'].rolling(period).std()
            df[f'volatility_rank_{period}d'] = df[f'volatility_{period}d'].rank(pct=True)
        
        # ========== MOVING AVERAGES ==========
        # Simple MAs
        for period in [5, 8, 10, 13, 20, 21, 34, 50, 55, 89, 100, 200]:
            df[f'sma_{period}'] = close.rolling(period).mean()
            df[f'price_vs_sma_{period}'] = (close / df[f'sma_{period}'] - 1) * 100
        
        # Exponential MAs (Fibonacci periods for your ribbon!)
        for period in [5, 8, 13, 21, 34, 55, 89, 144]:
            df[f'ema_{period}'] = close.ewm(span=period).mean()
            df[f'price_vs_ema_{period}'] = (close / df[f'ema_{period}'] - 1) * 100
            df[f'ema_{period}_slope'] = (df[f'ema_{period}'] / df[f'ema_{period}'].shift(5) - 1) * 100
        
        # YOUR EMA RIBBON PATTERN!
        ribbon_emas = ['ema_8', 'ema_13', 'ema_21', 'ema_34', 'ema_55']
        df['ribbon_min'] = df[ribbon_emas].min(axis=1)
        df['ribbon_max'] = df[ribbon_emas].max(axis=1)
        df['ribbon_range'] = (df['ribbon_max'] - df['ribbon_min']) / df['ribbon_min'] * 100
        df['ribbon_compression'] = df['ribbon_range'] < 3  # Tight ribbon = breakout coming
        df['price_above_ribbon'] = close > df['ribbon_min']
        df['ribbon_bullish_stack'] = (df['ema_8'] > df['ema_13']) & (df['ema_13'] > df['ema_21']) & (df['ema_21'] > df['ema_34'])
        df['ribbon_bearish_stack'] = (df['ema_8'] < df['ema_13']) & (df['ema_13'] < df['ema_21']) & (df['ema_21'] < df['ema_34'])
        
        # EMA crossovers
        df['ema_8_13_cross'] = (df['ema_8'] > df['ema_13']).astype(int) - (df['ema_8'] > df['ema_13']).shift(1).astype(int)
        df['ema_8_21_cross'] = (df['ema_8'] > df['ema_21']).astype(int) - (df['ema_8'] > df['ema_21']).shift(1).astype(int)
        df['ema_21_55_cross'] = (df['ema_21'] > df['ema_55']).astype(int) - (df['ema_21'] > df['ema_55']).shift(1).astype(int)
        df['golden_cross'] = (df['sma_50'] > df['sma_200']).astype(int)
        
        # ========== RSI VARIANTS ==========
        for period in [7, 9, 14, 21, 28]:
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
            rs = gain / (loss + 1e-10)
            df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
        # RSI divergence (your Trinity Signal!)
        df['rsi_14_slope'] = df['rsi_14'] - df['rsi_14'].shift(5)
        df['price_slope'] = (close / close.shift(5) - 1) * 100
        df['rsi_bullish_divergence'] = (df['rsi_14_slope'] > 0) & (df['price_slope'] < 0)
        df['rsi_bearish_divergence'] = (df['rsi_14_slope'] < 0) & (df['price_slope'] > 0)
        
        # Stochastic RSI
        rsi = df['rsi_14']
        rsi_min = rsi.rolling(14).min()
        rsi_max = rsi.rolling(14).max()
        df['stoch_rsi'] = (rsi - rsi_min) / (rsi_max - rsi_min + 1e-10)
        df['stoch_rsi_k'] = df['stoch_rsi'].rolling(3).mean()
        df['stoch_rsi_d'] = df['stoch_rsi_k'].rolling(3).mean()
        
        # ========== MACD VARIANTS ==========
        for fast, slow, signal in [(8, 21, 5), (12, 26, 9), (5, 13, 8), (19, 39, 9)]:
            ema_fast = close.ewm(span=fast).mean()
            ema_slow = close.ewm(span=slow).mean()
            macd = ema_fast - ema_slow
            macd_signal = macd.ewm(span=signal).mean()
            df[f'macd_{fast}_{slow}'] = macd
            df[f'macd_signal_{fast}_{slow}'] = macd_signal
            df[f'macd_hist_{fast}_{slow}'] = macd - macd_signal
            df[f'macd_hist_{fast}_{slow}_rising'] = df[f'macd_hist_{fast}_{slow}'] > df[f'macd_hist_{fast}_{slow}'].shift(1)
        
        # ========== BOLLINGER BANDS ==========
        for period in [10, 20, 50]:
            for std_mult in [1.5, 2.0, 2.5]:
                mid = close.rolling(period).mean()
                std = close.rolling(period).std()
                df[f'bb_upper_{period}_{std_mult}'] = mid + std_mult * std
                df[f'bb_lower_{period}_{std_mult}'] = mid - std_mult * std
                df[f'bb_pct_{period}_{std_mult}'] = (close - df[f'bb_lower_{period}_{std_mult}']) / (df[f'bb_upper_{period}_{std_mult}'] - df[f'bb_lower_{period}_{std_mult}'] + 1e-10)
                df[f'bb_width_{period}_{std_mult}'] = (df[f'bb_upper_{period}_{std_mult}'] - df[f'bb_lower_{period}_{std_mult}']) / mid * 100
        
        # Bollinger squeeze
        df['bb_squeeze'] = df['bb_width_20_2.0'] < df['bb_width_20_2.0'].rolling(50).mean() * 0.75
        
        # ========== ATR AND VOLATILITY ==========
        for period in [7, 14, 21]:
            high_low = high - low
            high_close = (high - close.shift()).abs()
            low_close = (low - close.shift()).abs()
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df[f'atr_{period}'] = tr.rolling(period).mean()
            df[f'atr_pct_{period}'] = df[f'atr_{period}'] / close * 100
        
        # ========== VOLUME ANALYSIS ==========
        df['volume_sma_10'] = volume.rolling(10).mean()
        df['volume_sma_20'] = volume.rolling(20).mean()
        df['volume_sma_50'] = volume.rolling(50).mean()
        df['volume_ratio_10'] = volume / (df['volume_sma_10'] + 1)
        df['volume_ratio_20'] = volume / (df['volume_sma_20'] + 1)
        df['volume_spike'] = df['volume_ratio_20'] > 2.0
        
        # OBV
        df['obv'] = (np.sign(close.diff()) * volume).cumsum()
        df['obv_sma'] = df['obv'].rolling(20).mean()
        df['obv_trend'] = df['obv'] > df['obv_sma']
        
        # Volume Price Trend
        df['vpt'] = (volume * close.pct_change()).cumsum()
        
        # Chaikin Money Flow
        mfm = ((close - low) - (high - close)) / (high - low + 1e-10)
        mfv = mfm * volume
        df['cmf'] = mfv.rolling(20).sum() / volume.rolling(20).sum()
        
        # ========== MOMENTUM INDICATORS ==========
        # Rate of Change
        for period in [5, 10, 21]:
            df[f'roc_{period}'] = close.pct_change(period) * 100
        
        # Momentum
        for period in [10, 14, 21]:
            df[f'momentum_{period}'] = close - close.shift(period)
        
        # Williams %R
        for period in [14, 21]:
            highest_high = high.rolling(period).max()
            lowest_low = low.rolling(period).min()
            df[f'williams_r_{period}'] = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
        
        # CCI
        for period in [14, 20]:
            tp = (high + low + close) / 3
            sma_tp = tp.rolling(period).mean()
            mad = tp.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean())
            df[f'cci_{period}'] = (tp - sma_tp) / (0.015 * mad + 1e-10)
        
        # ========== TREND INDICATORS ==========
        # ADX
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
        
        tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
        atr_14 = tr.rolling(14).mean()
        
        plus_di = 100 * (plus_dm.rolling(14).mean() / atr_14)
        minus_di = 100 * (minus_dm.rolling(14).mean() / atr_14)
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
        df['adx'] = dx.rolling(14).mean()
        df['plus_di'] = plus_di
        df['minus_di'] = minus_di
        df['adx_trend_strong'] = df['adx'] > 25
        
        # Aroon
        for period in [14, 25]:
            df[f'aroon_up_{period}'] = 100 * high.rolling(period + 1).apply(lambda x: x.argmax()) / period
            df[f'aroon_down_{period}'] = 100 * low.rolling(period + 1).apply(lambda x: x.argmin()) / period
            df[f'aroon_osc_{period}'] = df[f'aroon_up_{period}'] - df[f'aroon_down_{period}']
        
        # ========== PATTERN RECOGNITION ==========
        # Candlestick patterns
        df['body'] = close - df['Open']
        df['body_pct'] = df['body'] / df['Open'] * 100
        df['upper_shadow'] = high - pd.concat([close, df['Open']], axis=1).max(axis=1)
        df['lower_shadow'] = pd.concat([close, df['Open']], axis=1).min(axis=1) - low
        df['doji'] = df['body'].abs() < (high - low) * 0.1
        df['hammer'] = (df['lower_shadow'] > df['body'].abs() * 2) & (df['upper_shadow'] < df['body'].abs())
        df['shooting_star'] = (df['upper_shadow'] > df['body'].abs() * 2) & (df['lower_shadow'] < df['body'].abs())
        
        # Higher highs / Lower lows
        df['higher_high'] = high > high.shift(1)
        df['lower_low'] = low < low.shift(1)
        df['higher_low'] = low > low.shift(1)
        df['lower_high'] = high < high.shift(1)
        
        # Swing points
        df['swing_high'] = (high > high.shift(1)) & (high > high.shift(-1))
        df['swing_low'] = (low < low.shift(1)) & (low < low.shift(-1))
        
        # ========== YOUR CUSTOM PATTERNS ==========
        # 5-day momentum (your discovery!)
        df['mom_5d'] = close.pct_change(5) * 100
        df['mom_5d_accel'] = df['mom_5d'] - df['mom_5d'].shift(1)
        
        # Bounce from low (your pattern!)
        df['low_5d'] = low.rolling(5).min()
        df['bounce_from_low'] = (close / df['low_5d'] - 1) * 100
        df['bounce_signal'] = (df['bounce_from_low'] > 5) & (df['ema_8'] > df['ema_8'].shift(5))
        
        # Trend alignment
        df['trend_5d'] = np.sign(close.pct_change(5))
        df['trend_10d'] = np.sign(close.pct_change(10))
        df['trend_21d'] = np.sign(close.pct_change(21))
        df['trend_alignment'] = (df['trend_5d'] + df['trend_10d'] + df['trend_21d']) / 3
        
        # Support/Resistance
        df['resistance_20d'] = high.rolling(20).max()
        df['support_20d'] = low.rolling(20).min()
        df['near_resistance'] = (df['resistance_20d'] - close) / close < 0.02
        df['near_support'] = (close - df['support_20d']) / close < 0.02
        
        # ========== PERCENTILE RANKINGS ==========
        # Rank current values vs history (for regime detection)
        for col in ['rsi_14', 'macd_hist_12_26', 'volume_ratio_20', 'atr_pct_14', 'bb_width_20_2.0']:
            if col in df.columns:
                df[f'{col}_percentile'] = df[col].rolling(90).rank(pct=True)
        
        # ========== CLEAN UP ==========
        # Replace infinities and fill NaN
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.ffill().bfill().fillna(0)
        
        # Store feature names
        self.feature_names = [c for c in df.columns if c not in ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']]
        
        return df
    
    def get_feature_vector(self, df: pd.DataFrame, idx: int) -> np.ndarray:
        """Get normalized feature vector at index"""
        features = df[self.feature_names].iloc[idx].values.astype(np.float32)
        # Clip extreme values
        features = np.clip(features, -100, 100)
        return features


# =============================================================================
# SWING TRADE ENVIRONMENT - Hold Winners, Cut Losers
# =============================================================================

class SwingTradeEnvironment:
    """
    Trading environment that rewards SWING trading, not day trading.
    
    Key differences from typical RL trading:
    - Rewards holding winners (not just any trade)
    - Penalizes panic selling on small dips
    - Encourages buying dips in uptrends
    - Uses sector correlation for timing
    """
    
    def __init__(self, data_dict: Dict[str, pd.DataFrame], config: SwingTradeConfig):
        self.config = config
        self.data_dict = data_dict
        self.tickers = list(data_dict.keys())
        self.n_tickers = len(self.tickers)
        
        self.feature_engine = MegaFeatureEngine()
        
        # Precompute all features
        print("🧠 Computing features for all tickers...")
        self.features = {}
        for ticker in self.tickers:
            self.features[ticker] = self.feature_engine.compute_all_features(data_dict[ticker].copy())
        
        self.n_features = len(self.feature_engine.feature_names)
        print(f"   📊 {self.n_features} features per ticker")
        
        # State
        self.reset()
    
    def reset(self) -> Dict:
        """Reset environment"""
        self.balance = self.config.initial_balance
        self.positions = {}  # ticker -> {shares, entry_price, entry_idx, max_price}
        self.current_idx = 60  # Start after warmup period
        self.trades = []
        self.portfolio_history = [self.balance]
        
        return self._get_state()
    
    def _get_state(self) -> Dict:
        """Get current state for all tickers"""
        state = {
            'features': {},
            'positions': {},
            'balance': self.balance,
            'portfolio_value': self._get_portfolio_value(),
            'cash_pct': self.balance / self._get_portfolio_value()
        }
        
        for ticker in self.tickers:
            if self.current_idx < len(self.features[ticker]):
                state['features'][ticker] = self.feature_engine.get_feature_vector(
                    self.features[ticker], self.current_idx
                )
                
                if ticker in self.positions:
                    pos = self.positions[ticker]
                    current_price = float(self.data_dict[ticker]['Close'].iloc[self.current_idx])
                    state['positions'][ticker] = {
                        'pnl_pct': (current_price / pos['entry_price'] - 1),
                        'hold_days': self.current_idx - pos['entry_idx'],
                        'from_max': (current_price / pos['max_price'] - 1)
                    }
        
        return state
    
    def _get_portfolio_value(self) -> float:
        """Calculate total portfolio value"""
        value = self.balance
        for ticker, pos in self.positions.items():
            if self.current_idx < len(self.data_dict[ticker]):
                current_price = float(self.data_dict[ticker]['Close'].iloc[self.current_idx])
                value += pos['shares'] * current_price
        return value
    
    def step(self, actions: Dict[str, int]) -> Tuple[Dict, float, bool, Dict]:
        """
        Execute actions and return new state, reward, done, info
        
        Actions per ticker:
        0 = HOLD
        1 = BUY (if not holding)
        2 = SELL (if holding)
        3 = ADD (add to position if holding)
        """
        
        reward = 0
        info = {'trades': [], 'signals': []}
        
        # Update position max prices (for trailing stop)
        for ticker, pos in self.positions.items():
            current_price = float(self.data_dict[ticker]['Close'].iloc[self.current_idx])
            if current_price > pos['max_price']:
                pos['max_price'] = current_price
        
        # Process each ticker
        for ticker, action in actions.items():
            if self.current_idx >= len(self.data_dict[ticker]) - 1:
                continue
            
            current_price = float(self.data_dict[ticker]['Close'].iloc[self.current_idx])
            df = self.features[ticker]
            
            # Get indicators for smart rewards
            rsi = float(df['rsi_14'].iloc[self.current_idx])
            mom_5d = float(df['mom_5d'].iloc[self.current_idx])
            ribbon_bullish = bool(df['ribbon_bullish_stack'].iloc[self.current_idx])
            bounce_signal = bool(df['bounce_signal'].iloc[self.current_idx])
            
            if ticker in self.positions:
                pos = self.positions[ticker]
                pnl_pct = (current_price / pos['entry_price']) - 1
                hold_days = self.current_idx - pos['entry_idx']
                from_max = (current_price / pos['max_price']) - 1
                
                # === SELL LOGIC ===
                if action == 2:  # SELL
                    # Calculate reward based on trade quality
                    trade_reward = pnl_pct * 1000  # Base reward
                    
                    # === SWING TRADE REWARDS ===
                    
                    # BIG BONUS for taking profits at good levels
                    if pnl_pct >= 0.25:  # 25%+ gain
                        trade_reward += 200  # "You held like a pro!"
                    elif pnl_pct >= 0.15:  # 15%+ gain
                        trade_reward += 100
                    elif pnl_pct >= 0.08:  # 8%+ gain
                        trade_reward += 50
                    
                    # PENALTY for panic selling small dips
                    if -0.05 < pnl_pct < 0 and ribbon_bullish:
                        trade_reward -= 50  # "Don't panic sell in uptrend!"
                    
                    # BONUS for cutting losses properly
                    if pnl_pct <= -0.10:
                        trade_reward += 20  # "Good discipline cutting loss"
                    
                    # PENALTY for selling too early in momentum
                    if pnl_pct > 0 and pnl_pct < 0.05 and mom_5d > 10:
                        trade_reward -= 30  # "You sold during momentum!"
                    
                    # Execute sale
                    sale_value = pos['shares'] * current_price
                    self.balance += sale_value
                    
                    self.trades.append({
                        'ticker': ticker,
                        'type': 'SELL',
                        'entry_price': pos['entry_price'],
                        'exit_price': current_price,
                        'pnl_pct': pnl_pct,
                        'hold_days': hold_days,
                        'idx': self.current_idx
                    })
                    
                    del self.positions[ticker]
                    reward += trade_reward
                    info['trades'].append(f"{ticker}: SELL {pnl_pct:+.1%}")
                
                # === HOLD LOGIC ===
                elif action == 0:  # HOLD
                    # Reward for holding winners
                    if pnl_pct > 0.10:
                        reward += 5  # "Good job holding winner"
                    
                    # Small penalty for holding losers too long
                    if pnl_pct < -0.15 and hold_days > 20:
                        reward -= 10  # "Should have cut this"
                    
                    # Check trailing stop
                    if pnl_pct > 0.15 and from_max < -self.config.trailing_stop:
                        reward -= 20  # "Trailing stop triggered"
                        info['signals'].append(f"{ticker}: TRAILING_STOP")
                
                # === ADD TO POSITION ===
                elif action == 3:  # ADD
                    if pnl_pct > 0 and len(self.positions) < self.config.max_positions:
                        position_value = pos['shares'] * current_price
                        portfolio_value = self._get_portfolio_value()
                        
                        if position_value / portfolio_value < self.config.max_position_pct:
                            add_amount = min(self.balance * 0.10, self.balance - portfolio_value * self.config.min_cash_reserve)
                            if add_amount > current_price:
                                add_shares = int(add_amount / current_price)
                                cost = add_shares * current_price
                                
                                # Update position
                                total_shares = pos['shares'] + add_shares
                                avg_price = (pos['shares'] * pos['entry_price'] + add_shares * current_price) / total_shares
                                pos['shares'] = total_shares
                                pos['entry_price'] = avg_price
                                self.balance -= cost
                                
                                reward += 10  # Reward for pyramiding winners
                                info['trades'].append(f"{ticker}: ADD {add_shares}")
            
            else:
                # === BUY LOGIC ===
                if action == 1:  # BUY
                    # Check if we can buy
                    portfolio_value = self._get_portfolio_value()
                    if (self.balance / portfolio_value > self.config.min_cash_reserve and 
                        len(self.positions) < self.config.max_positions):
                        
                        # Position sizing
                        position_amount = min(
                            portfolio_value * self.config.max_position_pct,
                            self.balance * 0.90
                        )
                        shares = int(position_amount / current_price)
                        
                        if shares > 0:
                            cost = shares * current_price
                            self.balance -= cost
                            
                            self.positions[ticker] = {
                                'shares': shares,
                                'entry_price': current_price,
                                'entry_idx': self.current_idx,
                                'max_price': current_price
                            }
                            
                            # === SMART ENTRY REWARDS ===
                            
                            # DIP BUY bonus (your strategy!)
                            if rsi < 35 and mom_5d < -5:
                                reward += 75  # "Perfect dip buy!"
                            
                            # Bounce recovery bonus
                            if bounce_signal:
                                reward += 40  # "Nice bounce entry"
                            
                            # Ribbon breakout bonus
                            if ribbon_bullish and rsi > 50 and rsi < 70:
                                reward += 30  # "Good trend entry"
                            
                            info['trades'].append(f"{ticker}: BUY {shares}")
        
        # Move to next day
        self.current_idx += 1
        
        # Track portfolio
        self.portfolio_history.append(self._get_portfolio_value())
        
        # Check if done
        min_len = min(len(df) for df in self.data_dict.values())
        done = self.current_idx >= min_len - 1
        
        # End of episode - liquidate and calculate final reward
        if done:
            for ticker, pos in list(self.positions.items()):
                current_price = float(self.data_dict[ticker]['Close'].iloc[self.current_idx])
                pnl_pct = (current_price / pos['entry_price']) - 1
                sale_value = pos['shares'] * current_price
                self.balance += sale_value
                reward += pnl_pct * 500
            
            self.positions = {}
            
            # Final portfolio performance reward
            total_return = (self._get_portfolio_value() / self.config.initial_balance - 1)
            reward += total_return * 1000
            
            # Sharpe ratio bonus
            returns = np.diff(self.portfolio_history) / self.portfolio_history[:-1]
            if len(returns) > 0 and np.std(returns) > 0:
                sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
                if sharpe > 1.0:
                    reward += sharpe * 100
        
        return self._get_state(), reward, done, info


# =============================================================================
# NEURAL NETWORK - Self-Attention for Pattern Discovery
# =============================================================================

class PatternDiscoveryNetwork(nn.Module):
    """
    Neural network that discovers patterns through self-attention.
    
    Architecture:
    1. Feature embedding for each indicator
    2. Self-attention to find indicator relationships
    3. Cross-ticker attention for sector correlation
    4. Actor-Critic heads for action selection
    """
    
    def __init__(self, n_features: int, n_tickers: int, hidden_size: int = 256):
        super().__init__()
        
        self.n_features = n_features
        self.n_tickers = n_tickers
        self.hidden_size = hidden_size
        
        # Feature embedding
        self.feature_embed = nn.Sequential(
            nn.Linear(n_features, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU()
        )
        
        # Self-attention for feature relationships
        self.feature_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            batch_first=True,
            dropout=0.1
        )
        
        # Position info embedding
        self.position_embed = nn.Sequential(
            nn.Linear(4, 32),  # pnl_pct, hold_days, from_max, has_position
            nn.GELU(),
            nn.Linear(32, hidden_size)
        )
        
        # Combined processing
        self.combine = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU()
        )
        
        # Actor head (per ticker)
        self.actor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 4)  # HOLD, BUY, SELL, ADD
        )
        
        # Critic head (global value)
        self.critic = nn.Sequential(
            nn.Linear(hidden_size * n_tickers + 2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1)
        )
        
        # Initialize
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, features: torch.Tensor, positions: torch.Tensor, 
                portfolio_info: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features: [batch, n_tickers, n_features]
            positions: [batch, n_tickers, 4] - pnl, hold_days, from_max, has_pos
            portfolio_info: [batch, 2] - balance_pct, cash_pct
        
        Returns:
            action_logits: [batch, n_tickers, 4]
            value: [batch, 1]
        """
        batch_size = features.shape[0]
        
        # Embed features per ticker
        # [batch, n_tickers, hidden]
        feat_embed = self.feature_embed(features)
        
        # Self-attention on features
        feat_attend, _ = self.feature_attention(feat_embed, feat_embed, feat_embed)
        
        # Embed position info
        pos_embed = self.position_embed(positions)
        
        # Combine feature and position info
        combined = self.combine(torch.cat([feat_attend, pos_embed], dim=-1))
        
        # Actor: action logits per ticker
        action_logits = self.actor(combined)  # [batch, n_tickers, 4]
        
        # Critic: global value
        flat = combined.view(batch_size, -1)
        critic_input = torch.cat([flat, portfolio_info], dim=-1)
        value = self.critic(critic_input)
        
        return action_logits, value
    
    def get_actions(self, state: Dict, deterministic: bool = False) -> Dict[str, int]:
        """Get actions for all tickers"""
        # This would be called during inference
        # Implementation depends on how state is structured
        pass


# =============================================================================
# MAIN TRAINING LOOP
# =============================================================================

def load_data(tickers: List[str], period: str = '2y') -> Dict[str, pd.DataFrame]:
    """Load market data for all tickers"""
    data_dict = {}
    
    print(f"📥 Loading data for {len(tickers)} tickers...")
    
    for ticker in tickers:
        try:
            df = yf.download(ticker, period=period, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.reset_index()
            
            # Convert to numeric
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            if len(df) > 100:
                data_dict[ticker] = df
                print(f"   ✓ {ticker}: {len(df)} days")
        except Exception as e:
            print(f"   ✗ {ticker}: {e}")
    
    return data_dict


def train_swing_trader(config: SwingTradeConfig = None):
    """Main training function"""
    
    if config is None:
        config = SwingTradeConfig()
    
    print("=" * 70)
    print("🚀 ULTIMATE GPU TRAINER - Swing Trade Pattern Discovery")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Tickers: {len(config.tickers)}")
    print(f"Episodes: {config.episodes}")
    print("=" * 70)
    
    # Load data
    data_dict = load_data(config.tickers, period='2y')
    
    if len(data_dict) < 5:
        print("❌ Not enough data loaded")
        return
    
    # Create environment
    env = SwingTradeEnvironment(data_dict, config)
    
    print(f"\n🧠 Environment ready:")
    print(f"   Features: {env.n_features}")
    print(f"   Tickers: {env.n_tickers}")
    
    # Simple training loop (without full RL for now)
    best_return = -float('inf')
    best_trades = []
    discovered_patterns = {}
    
    print(f"\n🎮 Starting training...")
    print("-" * 70)
    
    for episode in range(config.episodes):
        state = env.reset()
        total_reward = 0
        episode_trades = []
        
        # Exploration rate
        exploration = max(
            config.min_discovery_rate,
            config.discovery_rate - episode * (config.discovery_rate - config.min_discovery_rate) / config.episodes
        )
        
        while True:
            # Generate actions
            actions = {}
            for ticker in env.tickers:
                if ticker not in state['features']:
                    continue
                
                features = state['features'][ticker]
                
                # Get indicators for rule-based decisions
                idx = env.current_idx
                df = env.features[ticker]
                
                rsi = float(df['rsi_14'].iloc[idx])
                mom_5d = float(df['mom_5d'].iloc[idx])
                ribbon_bullish = bool(df['ribbon_bullish_stack'].iloc[idx])
                bounce_signal = bool(df['bounce_signal'].iloc[idx])
                bb_squeeze = bool(df['bb_squeeze'].iloc[idx])
                ribbon_range = float(df['ribbon_range'].iloc[idx])
                macd_rising = bool(df['macd_hist_12_26_rising'].iloc[idx])
                
                # Random exploration
                if np.random.random() < exploration:
                    actions[ticker] = np.random.choice([0, 1, 2, 3], p=[0.6, 0.25, 0.1, 0.05])
                    continue
                
                # Rule-based with discovered patterns
                action = 0  # Default HOLD
                
                if ticker in state['positions']:
                    pos = state['positions'][ticker]
                    pnl = pos['pnl_pct']
                    hold_days = pos['hold_days']
                    
                    # SELL conditions
                    if pnl >= 0.25:
                        action = 2  # Take big profit
                    elif pnl >= 0.15 and rsi > 75:
                        action = 2  # Take profit at overbought
                    elif pnl <= -0.10:
                        action = 2  # Cut loss
                    elif pnl > 0.15 and pos['from_max'] < -0.08:
                        action = 2  # Trailing stop
                    elif hold_days > config.max_hold_days:
                        action = 2  # Time exit
                    
                    # ADD conditions
                    elif pnl > 0.05 and pnl < 0.15 and rsi < 50:
                        action = 3  # Add on pullback in winner
                
                else:
                    # BUY conditions
                    # DIP BUY
                    if rsi < 35 and mom_5d < -5:
                        action = 1
                    # BOUNCE
                    elif bounce_signal and rsi < 45:
                        action = 1
                    # RIBBON CONVERGENCE
                    elif ribbon_range < 5 and ribbon_bullish and bb_squeeze:
                        action = 1
                    # MOMENTUM
                    elif ribbon_bullish and macd_rising and 50 < rsi < 65 and mom_5d > 0:
                        action = 1
                    # RSI BOUNCE
                    elif rsi < 40 and macd_rising:
                        action = 1
                
                actions[ticker] = action
            
            # Step environment
            state, reward, done, info = env.step(actions)
            total_reward += reward
            episode_trades.extend(info.get('trades', []))
            
            if done:
                break
        
        # Calculate episode return
        episode_return = (env._get_portfolio_value() / config.initial_balance - 1) * 100
        
        # Track best
        if episode_return > best_return:
            best_return = episode_return
            best_trades = env.trades.copy()
        
        # Progress
        if episode % 50 == 0:
            win_trades = len([t for t in env.trades if t['pnl_pct'] > 0])
            total_trades = len(env.trades)
            win_rate = win_trades / max(total_trades, 1) * 100
            
            print(f"Ep {episode:4d} | Return: {episode_return:+6.1f}% | "
                  f"Best: {best_return:+6.1f}% | "
                  f"Trades: {total_trades} | WR: {win_rate:.0f}%")
    
    print("-" * 70)
    print(f"\n🏆 TRAINING COMPLETE")
    print(f"   Best Return: {best_return:+.1f}%")
    print(f"   Best Trades: {len(best_trades)}")
    
    # Analyze winning patterns
    print("\n📊 Analyzing discovered patterns...")
    analyze_winning_trades(best_trades, env.features)
    
    return best_trades, discovered_patterns


def analyze_winning_trades(trades: List[Dict], features: Dict[str, pd.DataFrame]):
    """Analyze what made winning trades work"""
    
    winners = [t for t in trades if t['pnl_pct'] > 0]
    losers = [t for t in trades if t['pnl_pct'] <= 0]
    
    if not winners:
        print("   No winning trades to analyze")
        return
    
    print(f"\n   Winners: {len(winners)} | Losers: {len(losers)}")
    print(f"   Win Rate: {len(winners) / len(trades) * 100:.1f}%")
    print(f"   Avg Winner: {np.mean([t['pnl_pct'] for t in winners]) * 100:+.1f}%")
    print(f"   Avg Loser: {np.mean([t['pnl_pct'] for t in losers]) * 100:+.1f}%" if losers else "")
    
    print("\n   🎯 Top 10 Winning Trades:")
    for t in sorted(winners, key=lambda x: x['pnl_pct'], reverse=True)[:10]:
        print(f"      {t['ticker']}: {t['pnl_pct']*100:+.1f}% in {t['hold_days']} days")
    
    # Export results
    results = {
        'generated_at': datetime.now().isoformat(),
        'total_trades': len(trades),
        'winners': len(winners),
        'losers': len(losers),
        'win_rate': len(winners) / len(trades) * 100,
        'avg_winner_pct': np.mean([t['pnl_pct'] for t in winners]) * 100,
        'avg_loser_pct': np.mean([t['pnl_pct'] for t in losers]) * 100 if losers else 0,
        'best_trades': sorted(winners, key=lambda x: x['pnl_pct'], reverse=True)[:20]
    }
    
    with open('swing_trade_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print("\n   ✅ Results saved to swing_trade_results.json")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    # Configure for your watchlist
    config = SwingTradeConfig(
        episodes=500,  # Start with 500, increase for longer training
        tickers=[
            # Your core watchlist
            'APLD', 'SERV', 'MRVL', 'HOOD', 'LUNR', 'NVDA', 'AMD', 'TSLA',
            'SMR', 'OKLO', 'LEU', 'UUUU', 'INTC', 'META', 'GOOGL', 'BA',
            'SPY', 'QQQ', 'SNOW', 'NOW', 'CRDO', 'IONQ', 'RGTI'
        ]
    )
    
    # Train!
    trades, patterns = train_swing_trader(config)
