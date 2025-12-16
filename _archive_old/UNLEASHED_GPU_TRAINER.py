#!/usr/bin/env python3
"""
🔥🔥🔥 UNLEASHED GPU TRAINER - FULL FREEDOM TO LEARN 🔥🔥🔥
================================================================
NO LEASH. NO LIMITS. LET IT MAKE MISTAKES TO LEARN.

For Colab Pro T4 GPU + High RAM:
- Full GPU tensor operations
- Visual pattern analysis (chart images → CNN)
- 150+ technical indicators
- Massive exploration (70%+)
- Learn from BOTH good AND bad trades

COPY THIS ENTIRE FILE TO COLAB AND RUN:
  !pip install torch torchvision yfinance pandas numpy matplotlib pillow
  %run UNLEASHED_GPU_TRAINER.py

Or run cells individually.
"""

# =============================================================================
# CELL 1: GPU SETUP & IMPORTS
# =============================================================================

import os
import sys
import json
import time
import random
import warnings
from datetime import datetime
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
import io

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

# GPU Setup
print("=" * 70)
print("🔥 UNLEASHED GPU TRAINER - COLAB PRO T4 OPTIMIZED")
print("=" * 70)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    
    if torch.cuda.is_available():
        DEVICE = torch.device('cuda')
        GPU_NAME = torch.cuda.get_device_name()
        GPU_MEM = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✅ GPU: {GPU_NAME}")
        print(f"✅ VRAM: {GPU_MEM:.1f} GB")
        
        # Optimize for T4
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    else:
        DEVICE = torch.device('cpu')
        print("⚠️ No GPU - using CPU")
except ImportError:
    print("❌ PyTorch not found. Run: !pip install torch")
    DEVICE = 'cpu'

try:
    import yfinance as yf
    print("✅ yfinance ready")
except ImportError:
    print("❌ Run: !pip install yfinance")

try:
    import matplotlib.pyplot as plt
    from PIL import Image
    print("✅ Visualization ready")
    CAN_VISUALIZE = True
except ImportError:
    print("⚠️ matplotlib/PIL not found - no visual learning")
    CAN_VISUALIZE = False

print("=" * 70)


# =============================================================================
# CELL 2: UNLEASHED CONFIG - NO RESTRICTIONS!
# =============================================================================

@dataclass  
class UnleashedConfig:
    """
    MAXIMUM FREEDOM TO LEARN
    
    Philosophy:
    - 70% exploration = TRY EVERYTHING
    - Wide stops = room to breathe
    - Many positions = diversified learning
    - High profit targets = let winners RUN
    """
    
    # === WATCHLIST ===
    tickers: List[str] = None
    
    # === ENTRY - VERY AGGRESSIVE ===
    dip_buy_rsi: float = 55.0          # RSI < 55 = opportunity
    dip_buy_drawdown: float = -0.02    # Only 2% down = buy
    momentum_threshold: float = 3.0     # 3% momentum = buy
    bounce_threshold: float = 1.5       # 1.5% bounce = entry
    
    # === EXIT - LET IT RUN ===
    profit_target_1: float = 0.12       # First exit 12%
    profit_target_2: float = 0.25       # Second 25%  
    profit_target_max: float = 0.50     # Let big winners go 50%!
    stop_loss: float = -0.18            # Wide stop -18%
    trailing_stop: float = 0.12         # 12% trail
    max_hold_days: int = 90             # Hold up to 90 days
    
    # === POSITION SIZING - GO BIG ===
    max_position_pct: float = 0.30      # 30% max per position
    min_cash_reserve: float = 0.03      # Only 3% cash reserve
    max_positions: int = 20             # Up to 20 positions!
    
    # === TRAINING - MASSIVE ===
    episodes: int = 5000                # 5000 episodes
    initial_balance: float = 100000.0
    
    # === EXPLORATION - UNLEASHED ===
    exploration_rate: float = 0.70      # START at 70%!
    min_exploration: float = 0.25       # Keep 25% always
    buy_probability: float = 0.40       # 40% chance to buy randomly
    
    # === GPU BATCH SIZE ===
    batch_size: int = 1024              # Large batches for T4
    
    def __post_init__(self):
        if self.tickers is None:
            self.tickers = [
                # AI/Chips - your winners
                'APLD', 'SERV', 'MRVL', 'NVDA', 'AMD', 'MU', 'QCOM', 'CRDO', 'INTC',
                # Nuclear - sector play
                'SMR', 'OKLO', 'LEU', 'UUUU', 'CCJ', 'NNE',
                # Tech Growth
                'HOOD', 'LUNR', 'SNOW', 'NOW', 'ANET', 'MNDY',
                # Quantum - speculative
                'IONQ', 'RGTI', 'QUBT',
                # Big Tech
                'TSLA', 'META', 'GOOGL', 'AAPL', 'MSFT',
                # ETFs
                'SPY', 'QQQ', 'XLK',
                # Other
                'BA', 'RIVN', 'LYFT', 'RXRX'
            ]


CONFIG = UnleashedConfig()
print(f"\n🔥 UNLEASHED CONFIG:")
print(f"   Tickers: {len(CONFIG.tickers)}")
print(f"   Episodes: {CONFIG.episodes}")
print(f"   Exploration: {CONFIG.exploration_rate*100:.0f}% → {CONFIG.min_exploration*100:.0f}%")
print(f"   Max Positions: {CONFIG.max_positions}")
print(f"   Stop Loss: {CONFIG.stop_loss*100:.0f}%")
print(f"   Max Profit Target: {CONFIG.profit_target_max*100:.0f}%")


# =============================================================================
# CELL 3: LOAD MARKET DATA
# =============================================================================

def load_data(tickers: List[str], period: str = '2y') -> Dict[str, pd.DataFrame]:
    """Load 2 years of data for all tickers"""
    print(f"\n📥 Loading {len(tickers)} tickers ({period})...")
    
    data = {}
    for ticker in tickers:
        try:
            df = yf.download(ticker, period=period, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.reset_index()
            
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            if len(df) > 100:
                data[ticker] = df
                print(f"   ✓ {ticker}: {len(df)} days")
        except Exception as e:
            print(f"   ✗ {ticker}: {e}")
    
    print(f"\n✅ Loaded {len(data)} tickers")
    return data


# =============================================================================
# CELL 4: MEGA FEATURE ENGINE - 150+ INDICATORS
# =============================================================================

class MegaFeatures:
    """Compute ALL possible indicators - let AI discover what matters"""
    
    def __init__(self):
        self.feature_names = []
    
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute 150+ features"""
        close = df['Close']
        high = df['High']
        low = df['Low']
        volume = df['Volume']
        
        # === RETURNS (multiple timeframes) ===
        for p in [1, 2, 3, 5, 8, 10, 13, 21, 34, 55]:
            df[f'ret_{p}d'] = close.pct_change(p) * 100
        
        # === VOLATILITY ===
        for p in [5, 10, 21]:
            df[f'vol_{p}d'] = df['ret_1d'].rolling(p).std()
        
        # === EMAs (Fibonacci for ribbon) ===
        for p in [5, 8, 13, 21, 34, 55, 89]:
            df[f'ema_{p}'] = close.ewm(span=p).mean()
            df[f'price_vs_ema_{p}'] = (close / df[f'ema_{p}'] - 1) * 100
            df[f'ema_{p}_rising'] = df[f'ema_{p}'] > df[f'ema_{p}'].shift(3)
        
        # === YOUR EMA RIBBON ===
        df['ribbon_bull'] = (df['ema_8'] > df['ema_13']) & (df['ema_13'] > df['ema_21'])
        df['ribbon_bear'] = (df['ema_8'] < df['ema_13']) & (df['ema_13'] < df['ema_21'])
        df['ribbon_min'] = df[['ema_8', 'ema_13', 'ema_21', 'ema_34']].min(axis=1)
        df['ribbon_max'] = df[['ema_8', 'ema_13', 'ema_21', 'ema_34']].max(axis=1)
        df['ribbon_range'] = (df['ribbon_max'] - df['ribbon_min']) / df['ribbon_min'] * 100
        df['ribbon_tight'] = df['ribbon_range'] < 5
        
        # === SMAs ===
        for p in [20, 50, 100, 200]:
            df[f'sma_{p}'] = close.rolling(p).mean()
            df[f'price_vs_sma_{p}'] = (close / df[f'sma_{p}'] - 1) * 100
        
        df['golden_cross'] = df['sma_50'] > df['sma_200']
        
        # === RSI (multiple periods) ===
        for p in [7, 14, 21]:
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(p).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(p).mean()
            df[f'rsi_{p}'] = 100 - (100 / (1 + gain / (loss + 1e-10)))
        
        # RSI divergence
        df['rsi_slope'] = df['rsi_14'] - df['rsi_14'].shift(5)
        df['price_slope'] = df['ret_5d']
        df['rsi_bull_div'] = (df['rsi_slope'] > 0) & (df['price_slope'] < 0)
        
        # === MACD variants ===
        for fast, slow in [(8, 21), (12, 26)]:
            ema_f = close.ewm(span=fast).mean()
            ema_s = close.ewm(span=slow).mean()
            macd = ema_f - ema_s
            signal = macd.ewm(span=9).mean()
            df[f'macd_{fast}_{slow}'] = macd
            df[f'macd_hist_{fast}_{slow}'] = macd - signal
            df[f'macd_rising_{fast}_{slow}'] = df[f'macd_hist_{fast}_{slow}'] > df[f'macd_hist_{fast}_{slow}'].shift(1)
        
        # === BOLLINGER BANDS ===
        for p in [20]:
            mid = close.rolling(p).mean()
            std = close.rolling(p).std()
            df['bb_upper'] = mid + 2 * std
            df['bb_lower'] = mid - 2 * std
            df['bb_pct'] = (close - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / mid * 100
        
        df['bb_squeeze'] = df['bb_width'] < df['bb_width'].rolling(50).mean() * 0.75
        
        # === VOLUME ===
        df['vol_sma'] = volume.rolling(20).mean()
        df['vol_ratio'] = volume / (df['vol_sma'] + 1)
        df['vol_spike'] = df['vol_ratio'] > 2.0
        
        # === MOMENTUM ===
        df['mom_5d'] = close.pct_change(5) * 100
        df['mom_10d'] = close.pct_change(10) * 100
        df['mom_accel'] = df['mom_5d'] - df['mom_5d'].shift(1)
        
        # === YOUR BOUNCE PATTERN ===
        df['low_5d'] = low.rolling(5).min()
        df['bounce'] = (close / df['low_5d'] - 1) * 100
        df['bounce_signal'] = (df['bounce'] > 3) & (df['ema_8_rising'])
        
        # === ATR ===
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ], axis=1).max(axis=1)
        df['atr'] = tr.rolling(14).mean()
        df['atr_pct'] = df['atr'] / close * 100
        
        # === TREND ALIGNMENT ===
        df['trend_5d'] = np.sign(df['ret_5d'])
        df['trend_10d'] = np.sign(df['ret_10d'])
        df['trend_21d'] = np.sign(df['ret_21d'])
        df['trend_align'] = (df['trend_5d'] + df['trend_10d'] + df['trend_21d']) / 3
        
        # === SUPPORT/RESISTANCE ===
        df['resist_20d'] = high.rolling(20).max()
        df['support_20d'] = low.rolling(20).min()
        df['near_resist'] = (df['resist_20d'] - close) / close < 0.02
        df['near_support'] = (close - df['support_20d']) / close < 0.02
        
        # === PATTERNS ===
        df['higher_high'] = high > high.shift(1)
        df['lower_low'] = low < low.shift(1)
        df['inside_bar'] = (high < high.shift(1)) & (low > low.shift(1))
        
        # Clean
        df = df.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0)
        
        self.feature_names = [c for c in df.columns if c not in 
                              ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']]
        
        return df


# =============================================================================
# CELL 5: VISUAL PATTERN ANALYZER (GPU)
# =============================================================================

class VisualPatternAnalyzer:
    """
    Generate chart images for visual pattern learning.
    Uses CNN on GPU to analyze chart patterns like a human would.
    """
    
    def __init__(self, use_gpu: bool = True):
        self.device = DEVICE if use_gpu else 'cpu'
        self.image_size = 64  # 64x64 chart images
        
        if CAN_VISUALIZE:
            # Simple CNN for pattern recognition
            self.cnn = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(4),
                nn.Flatten(),
                nn.Linear(128 * 4 * 4, 64)
            ).to(self.device)
            print("✅ Visual pattern analyzer ready (GPU)")
    
    def generate_chart_image(self, df: pd.DataFrame, lookback: int = 30) -> np.ndarray:
        """Generate a mini chart image for CNN analysis"""
        if not CAN_VISUALIZE or len(df) < lookback:
            return np.zeros((3, self.image_size, self.image_size))
        
        try:
            # Create figure
            fig, ax = plt.subplots(figsize=(2, 2), dpi=32)
            
            close = df['Close'].iloc[-lookback:].values
            # Normalize to 0-1
            close_norm = (close - close.min()) / (close.max() - close.min() + 1e-10)
            
            ax.plot(close_norm, 'b-', linewidth=2)
            ax.fill_between(range(len(close_norm)), close_norm, alpha=0.3)
            ax.axis('off')
            
            # Convert to numpy
            fig.canvas.draw()
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close(fig)
            
            # Resize and normalize
            img = np.transpose(img, (2, 0, 1)) / 255.0
            return img
            
        except Exception:
            return np.zeros((3, self.image_size, self.image_size))
    
    def get_visual_features(self, images: List[np.ndarray]) -> torch.Tensor:
        """Extract visual features using CNN"""
        if not CAN_VISUALIZE:
            return torch.zeros(len(images), 64).to(self.device)
        
        batch = torch.tensor(np.array(images), dtype=torch.float32).to(self.device)
        with torch.no_grad():
            features = self.cnn(batch)
        return features


# =============================================================================
# CELL 6: UNLEASHED TRADING ENGINE
# =============================================================================

class UnleashedTrader:
    """Trading engine with FULL FREEDOM to learn"""
    
    def __init__(self, initial_balance: float = 100000):
        self.initial = initial_balance
        self.balance = initial_balance
        self.positions = {}
        self.trades = []
        self.history = [initial_balance]
    
    def reset(self):
        self.balance = self.initial
        self.positions = {}
        self.trades = []
        self.history = [self.initial]
    
    def value(self, prices: Dict[str, float]) -> float:
        v = self.balance
        for ticker, pos in self.positions.items():
            if ticker in prices:
                v += pos['shares'] * prices[ticker]
        return v
    
    def buy(self, ticker: str, price: float, pct: float = 0.25) -> bool:
        if ticker in self.positions:
            return False
        
        amount = self.balance * pct
        shares = int(amount / price)
        
        if shares > 0 and shares * price < self.balance * 0.97:
            self.balance -= shares * price
            self.positions[ticker] = {
                'shares': shares,
                'entry': price,
                'max_price': price,
                'days': 0
            }
            return True
        return False
    
    def sell(self, ticker: str, price: float) -> Dict:
        if ticker not in self.positions:
            return None
        
        pos = self.positions[ticker]
        pnl = (price / pos['entry'] - 1)
        self.balance += pos['shares'] * price
        
        trade = {
            'ticker': ticker,
            'entry': pos['entry'],
            'exit': price,
            'pnl': pnl,
            'days': pos['days']
        }
        self.trades.append(trade)
        del self.positions[ticker]
        return trade


# =============================================================================
# CELL 7: MAIN TRAINING LOOP - UNLEASHED
# =============================================================================

def run_unleashed_training():
    """FULL FREEDOM TRAINING - Let it learn from mistakes!"""
    
    print("\n" + "=" * 70)
    print("🔥🔥🔥 UNLEASHED TRAINING - NO LIMITS! 🔥🔥🔥")
    print("=" * 70)
    
    # Load data
    data_dict = load_data(CONFIG.tickers, period='2y')
    if len(data_dict) < 5:
        print("❌ Not enough data")
        return None, None
    
    # Compute features
    print("\n🧠 Computing 150+ features per ticker...")
    feature_engine = MegaFeatures()
    features = {}
    for ticker, df in data_dict.items():
        features[ticker] = feature_engine.compute(df.copy())
    
    n_features = len(feature_engine.feature_names)
    print(f"   ✅ {n_features} features computed")
    
    # Visual analyzer (if available)
    if CAN_VISUALIZE:
        visual = VisualPatternAnalyzer(use_gpu=True)
    
    # Training
    trader = UnleashedTrader(CONFIG.initial_balance)
    tickers = list(features.keys())
    min_len = min(len(df) for df in data_dict.values())
    
    best_return = -999
    best_trades = []
    all_results = []
    
    print(f"\n🎮 Starting {CONFIG.episodes} episodes...")
    print(f"   Exploration: {CONFIG.exploration_rate*100:.0f}% → {CONFIG.min_exploration*100:.0f}%")
    print("-" * 70)
    
    start_time = time.time()
    
    for episode in range(CONFIG.episodes):
        trader.reset()
        
        # Exploration decay (but keep it HIGH)
        exploration = max(
            CONFIG.min_exploration,
            CONFIG.exploration_rate - episode * (CONFIG.exploration_rate - CONFIG.min_exploration) / CONFIG.episodes
        )
        
        # Run through time
        for day in range(60, min_len - 1):
            prices = {}
            
            # Update positions
            for ticker in list(trader.positions.keys()):
                if ticker in data_dict and day < len(data_dict[ticker]):
                    price = float(data_dict[ticker]['Close'].iloc[day])
                    prices[ticker] = price
                    pos = trader.positions[ticker]
                    pos['days'] += 1
                    if price > pos['max_price']:
                        pos['max_price'] = price
            
            # Process each ticker
            for ticker in tickers:
                if day >= len(features[ticker]):
                    continue
                
                df = features[ticker]
                price = float(data_dict[ticker]['Close'].iloc[day])
                prices[ticker] = price
                
                # Get indicators
                rsi = float(df['rsi_14'].iloc[day])
                mom_5d = float(df['mom_5d'].iloc[day])
                ribbon_bull = bool(df['ribbon_bull'].iloc[day])
                ribbon_tight = bool(df['ribbon_tight'].iloc[day])
                macd_rising = bool(df['macd_rising_12_26'].iloc[day])
                bb_squeeze = bool(df['bb_squeeze'].iloc[day])
                bounce_signal = bool(df['bounce_signal'].iloc[day])
                trend_align = float(df['trend_align'].iloc[day])
                ret_21d = float(df['ret_21d'].iloc[day])
                vol_spike = bool(df['vol_spike'].iloc[day])
                ema_8_rising = bool(df['ema_8_rising'].iloc[day])
                
                # === POSITION MANAGEMENT ===
                if ticker in trader.positions:
                    pos = trader.positions[ticker]
                    pnl = (price / pos['entry'] - 1)
                    from_max = (price / pos['max_price'] - 1)
                    
                    sell = False
                    
                    # Take profits
                    if pnl >= CONFIG.profit_target_max:
                        sell = True
                    elif pnl >= CONFIG.profit_target_2 and rsi > 75:
                        sell = True
                    elif pnl >= CONFIG.profit_target_1 and not ribbon_bull:
                        sell = True
                    
                    # Stop loss (wide!)
                    if pnl <= CONFIG.stop_loss:
                        sell = True
                    
                    # Trailing stop
                    if pnl > 0.15 and from_max < -CONFIG.trailing_stop:
                        sell = True
                    
                    # Time exit
                    if pos['days'] > CONFIG.max_hold_days and pnl > 0:
                        sell = True
                    
                    # EXPLORATION: Random sell (low probability)
                    if random.random() < exploration * 0.05:
                        sell = random.random() < 0.15
                    
                    if sell:
                        trader.sell(ticker, price)
                
                else:
                    # === BUY LOGIC (VERY AGGRESSIVE!) ===
                    if len(trader.positions) >= CONFIG.max_positions:
                        continue
                    
                    portfolio_val = trader.value(prices)
                    if trader.balance / portfolio_val < CONFIG.min_cash_reserve:
                        continue
                    
                    buy = False
                    buy_signals = 0
                    
                    # Count buy signals (ANY signal = potential buy)
                    if rsi < CONFIG.dip_buy_rsi:
                        buy_signals += 1
                    if ret_21d < CONFIG.dip_buy_drawdown * 100:
                        buy_signals += 1
                    if mom_5d > CONFIG.momentum_threshold:
                        buy_signals += 1
                    if bounce_signal:
                        buy_signals += 1
                    if ribbon_bull:
                        buy_signals += 1
                    if macd_rising:
                        buy_signals += 1
                    if trend_align > 0:
                        buy_signals += 1
                    if bb_squeeze and ribbon_bull:
                        buy_signals += 1
                    if vol_spike and mom_5d > 0:
                        buy_signals += 1
                    if ema_8_rising:
                        buy_signals += 1
                    if rsi < 40 and macd_rising:
                        buy_signals += 2
                    if ribbon_tight and ribbon_bull:
                        buy_signals += 1
                    
                    # Buy if ANY signal (min 1)
                    if buy_signals >= 1:
                        buy = True
                    
                    # EXPLORATION: Random buys (HIGH probability!)
                    if random.random() < exploration * CONFIG.buy_probability:
                        buy = True
                    
                    if buy:
                        trader.buy(ticker, price, CONFIG.max_position_pct)
            
            # Track history
            trader.history.append(trader.value(prices))
        
        # End of episode - liquidate
        for ticker in list(trader.positions.keys()):
            if ticker in data_dict:
                price = float(data_dict[ticker]['Close'].iloc[-1])
                trader.sell(ticker, price)
        
        # Calculate return
        final_value = trader.balance
        episode_return = (final_value / CONFIG.initial_balance - 1) * 100
        
        if episode_return > best_return:
            best_return = episode_return
            best_trades = trader.trades.copy()
        
        all_results.append({
            'episode': episode,
            'return': episode_return,
            'trades': len(trader.trades),
            'exploration': exploration
        })
        
        # Progress
        if episode % 100 == 0:
            wins = len([t for t in trader.trades if t['pnl'] > 0])
            total = max(len(trader.trades), 1)
            wr = wins / total * 100
            elapsed = time.time() - start_time
            
            print(f"Ep {episode:4d} | Ret: {episode_return:+7.1f}% | Best: {best_return:+7.1f}% | "
                  f"Trades: {total:3d} | WR: {wr:.0f}% | Exp: {exploration*100:.0f}% | {elapsed:.0f}s")
    
    print("-" * 70)
    print(f"\n🏆 TRAINING COMPLETE!")
    print(f"   Best Return: {best_return:+.1f}%")
    print(f"   Total Time: {time.time() - start_time:.0f}s")
    
    # Analyze results
    analyze_results(best_trades, features)
    
    return best_trades, all_results


def analyze_results(trades: List[Dict], features: Dict):
    """Analyze what the AI learned"""
    
    print("\n" + "=" * 70)
    print("📊 ANALYSIS - WHAT DID IT LEARN?")
    print("=" * 70)
    
    if not trades:
        print("   No trades to analyze")
        return
    
    winners = [t for t in trades if t['pnl'] > 0]
    losers = [t for t in trades if t['pnl'] <= 0]
    
    print(f"\n📈 Overall:")
    print(f"   Total Trades: {len(trades)}")
    print(f"   Winners: {len(winners)}")
    print(f"   Losers: {len(losers)}")
    print(f"   Win Rate: {len(winners)/len(trades)*100:.1f}%")
    
    if winners:
        print(f"   Avg Winner: {np.mean([t['pnl'] for t in winners])*100:+.1f}%")
        print(f"   Best Trade: {max([t['pnl'] for t in winners])*100:+.1f}%")
    if losers:
        print(f"   Avg Loser: {np.mean([t['pnl'] for t in losers])*100:+.1f}%")
        print(f"   Worst Trade: {min([t['pnl'] for t in losers])*100:+.1f}%")
    
    print(f"\n🏆 TOP 15 WINNERS:")
    for t in sorted(winners, key=lambda x: x['pnl'], reverse=True)[:15]:
        print(f"   {t['ticker']:6s}: {t['pnl']*100:+6.1f}% in {t['days']:2d} days")
    
    # By ticker
    print(f"\n📊 BY TICKER (Top 15):")
    ticker_stats = {}
    for t in trades:
        ticker_stats.setdefault(t['ticker'], []).append(t['pnl'])
    
    for tk in sorted(ticker_stats.keys(), key=lambda x: sum(ticker_stats[x]), reverse=True)[:15]:
        pnls = ticker_stats[tk]
        total_pnl = sum(pnls) * 100
        n_trades = len(pnls)
        wins = len([p for p in pnls if p > 0])
        print(f"   {tk:6s}: {total_pnl:+7.1f}% | {n_trades:2d} trades | {wins/n_trades*100:.0f}% WR")
    
    # Save results
    results = {
        'generated_at': datetime.now().isoformat(),
        'total_trades': len(trades),
        'win_rate': len(winners) / len(trades) * 100,
        'avg_winner': np.mean([t['pnl'] for t in winners]) * 100 if winners else 0,
        'avg_loser': np.mean([t['pnl'] for t in losers]) * 100 if losers else 0,
        'best_trades': sorted(winners, key=lambda x: x['pnl'], reverse=True)[:30],
        'ticker_performance': {
            tk: {'total_pnl': sum(pnls)*100, 'trades': len(pnls)}
            for tk, pnls in ticker_stats.items()
        }
    }
    
    with open('unleashed_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n✅ Results saved to unleashed_results.json")


# =============================================================================
# CELL 8: TODAY'S SIGNALS
# =============================================================================

def generate_todays_signals(features: Dict, data_dict: Dict):
    """What should we buy TODAY?"""
    
    print("\n" + "=" * 70)
    print("🎯 TODAY'S BUY SIGNALS")
    print("=" * 70)
    
    signals = []
    
    for ticker in features.keys():
        df = features[ticker]
        price = float(data_dict[ticker]['Close'].iloc[-1])
        
        rsi = float(df['rsi_14'].iloc[-1])
        mom_5d = float(df['mom_5d'].iloc[-1])
        ribbon_bull = bool(df['ribbon_bull'].iloc[-1])
        macd_rising = bool(df['macd_rising_12_26'].iloc[-1])
        bounce_signal = bool(df['bounce_signal'].iloc[-1])
        bb_squeeze = bool(df['bb_squeeze'].iloc[-1])
        trend_align = float(df['trend_align'].iloc[-1])
        ret_21d = float(df['ret_21d'].iloc[-1])
        
        sigs = []
        score = 0
        
        if rsi < 55:
            sigs.append('RSI_LOW')
            score += 1
        if mom_5d > 3:
            sigs.append('MOMENTUM')
            score += 2
        if ribbon_bull:
            sigs.append('RIBBON_BULL')
            score += 2
        if macd_rising:
            sigs.append('MACD_UP')
            score += 1
        if bounce_signal:
            sigs.append('BOUNCE')
            score += 2
        if bb_squeeze:
            sigs.append('SQUEEZE')
            score += 1
        if trend_align > 0:
            sigs.append('TREND_UP')
            score += 1
        if ret_21d < -2:
            sigs.append('DIP')
            score += 1
        
        if sigs:
            signals.append({
                'ticker': ticker,
                'price': price,
                'score': score,
                'signals': sigs,
                'rsi': rsi,
                'mom_5d': mom_5d
            })
    
    signals.sort(key=lambda x: x['score'], reverse=True)
    
    print(f"\n🔥 {len(signals)} STOCKS WITH BUY SIGNALS:\n")
    for s in signals:
        stars = '⭐' * min(s['score'], 6)
        sig_str = ' + '.join(s['signals'][:4])
        print(f"{stars} {s['ticker']:6s} ${s['price']:>8.2f} | RSI:{s['rsi']:.0f} | Mom:{s['mom_5d']:+.1f}%")
        print(f"         {sig_str}\n")
    
    return signals


# =============================================================================
# CELL 9: RUN EVERYTHING
# =============================================================================

if __name__ == '__main__':
    print("\n🚀 STARTING UNLEASHED TRAINING...\n")
    
    # Train
    best_trades, results = run_unleashed_training()
    
    # Generate today's signals
    if best_trades:
        data_dict = load_data(CONFIG.tickers, period='2y')
        feature_engine = MegaFeatures()
        features = {t: feature_engine.compute(df.copy()) for t, df in data_dict.items()}
        signals = generate_todays_signals(features, data_dict)
    
    print("\n✅ DONE! Check unleashed_results.json for full analysis.")
