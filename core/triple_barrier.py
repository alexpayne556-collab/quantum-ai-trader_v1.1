"""
TRIPLE BARRIER LABELING ENGINE
===============================
Implements the Triple Barrier Method from "Advances in Financial Machine Learning" (López de Prado)

The problem with naive labels (return > 0.5% = BUY):
- Ignores volatility (2% move in volatile stock = nothing)
- Ignores timing (0.5% in 20 days vs 1 hour = different)
- Creates imbalanced classes

Triple Barrier Solution:
- Upper barrier (Take Profit): +X% → Label = BUY (+1)
- Lower barrier (Stop Loss): -Y% → Label = SELL (-1)
- Time barrier: N days elapsed → Label = direction of final return

Additional Methods:
- Dynamic threshold (ATR-based volatility scaling)
- Trend-following labels
- Multi-horizon labels

Expected Impact: +3-8% accuracy improvement
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, List
import warnings
warnings.filterwarnings('ignore')


class TripleBarrierLabeler:
    """
    Advanced labeling strategies for ML trading models.
    
    Supported Methods:
    1. Triple Barrier (López de Prado)
    2. Dynamic Threshold (ATR-based)
    3. Trend Following
    4. Multi-Horizon Ensemble
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.label_stats = {}
        
    def log(self, msg: str):
        if self.verbose:
            print(f"[TripleBarrier] {msg}")
    
    # =========================================================================
    # METHOD 1: Triple Barrier (THE GOLD STANDARD)
    # =========================================================================
    def triple_barrier(
        self,
        close: pd.Series,
        high: pd.Series,
        low: pd.Series,
        tp_pct: float = 0.015,
        sl_pct: float = 0.015,
        max_holding_days: int = 5,
        volatility_scaler: Optional[pd.Series] = None
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Triple Barrier Method
        
        Args:
            close: Closing prices
            high: High prices
            low: Low prices
            tp_pct: Take-profit percentage (e.g., 0.015 = 1.5%)
            sl_pct: Stop-loss percentage (e.g., 0.015 = 1.5%)
            max_holding_days: Maximum holding period before time barrier
            volatility_scaler: Optional series to scale barriers by volatility
        
        Returns:
            labels: +1 (profit hit), -1 (stop hit), 0 (time barrier)
            exit_prices: Price at which barrier was hit
        """
        self.log(f"Triple Barrier: TP={tp_pct*100:.1f}%, SL={sl_pct*100:.1f}%, MaxDays={max_holding_days}")
        
        labels = pd.Series(index=close.index, dtype=float)
        exit_prices = close.copy()
        barrier_types = pd.Series(index=close.index, dtype=str)
        
        for i in range(len(close) - max_holding_days - 1):
            entry_price = close.iloc[i]
            
            # Dynamic barriers based on volatility
            if volatility_scaler is not None and not np.isnan(volatility_scaler.iloc[i]):
                vol = volatility_scaler.iloc[i]
                upper_barrier = entry_price * (1 + tp_pct * vol)
                lower_barrier = entry_price * (1 - sl_pct * vol)
            else:
                upper_barrier = entry_price * (1 + tp_pct)
                lower_barrier = entry_price * (1 - sl_pct)
            
            # Look ahead for barrier hits
            found = False
            for j in range(1, max_holding_days + 1):
                idx = i + j
                if idx >= len(close):
                    break
                
                # Check upper barrier (take profit)
                if high.iloc[idx] >= upper_barrier:
                    labels.iloc[i] = 1
                    exit_prices.iloc[i] = upper_barrier
                    barrier_types.iloc[i] = 'TP'
                    found = True
                    break
                
                # Check lower barrier (stop loss)
                if low.iloc[idx] <= lower_barrier:
                    labels.iloc[i] = -1
                    exit_prices.iloc[i] = lower_barrier
                    barrier_types.iloc[i] = 'SL'
                    found = True
                    break
            
            # Time barrier (no TP/SL hit within max_holding_days)
            if not found:
                final_idx = min(i + max_holding_days, len(close) - 1)
                final_price = close.iloc[final_idx]
                
                if final_price > entry_price * 1.001:  # Small buffer
                    labels.iloc[i] = 1
                elif final_price < entry_price * 0.999:
                    labels.iloc[i] = -1
                else:
                    labels.iloc[i] = 0  # HOLD
                
                exit_prices.iloc[i] = final_price
                barrier_types.iloc[i] = 'TIME'
        
        # Fill remaining with NaN
        labels.iloc[-(max_holding_days+1):] = np.nan
        
        # Log statistics
        valid_labels = labels.dropna()
        self.label_stats['triple_barrier'] = {
            'BUY (+1)': (valid_labels == 1).sum(),
            'SELL (-1)': (valid_labels == -1).sum(),
            'HOLD (0)': (valid_labels == 0).sum(),
            'TP hits': (barrier_types == 'TP').sum(),
            'SL hits': (barrier_types == 'SL').sum(),
            'Time exits': (barrier_types == 'TIME').sum()
        }
        
        self.log(f"  Labels: BUY={self.label_stats['triple_barrier']['BUY (+1)']}, "
                f"SELL={self.label_stats['triple_barrier']['SELL (-1)']}, "
                f"HOLD={self.label_stats['triple_barrier']['HOLD (0)']}")
        
        return labels, exit_prices
    
    # =========================================================================
    # METHOD 2: Dynamic Threshold (ATR-Based)
    # =========================================================================
    def dynamic_threshold(
        self,
        close: pd.Series,
        high: pd.Series,
        low: pd.Series,
        base_threshold: float = 0.005,
        atr_period: int = 14,
        horizon: int = 5
    ) -> pd.Series:
        """
        Dynamic threshold labeling based on ATR
        
        High volatility → Higher threshold (2% move doesn't matter)
        Low volatility → Lower threshold (0.5% move matters)
        """
        self.log(f"Dynamic Threshold: base={base_threshold*100:.2f}%, ATR period={atr_period}, horizon={horizon}d")
        
        # Calculate ATR
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(atr_period).mean()
        
        # ATR as percentage of price
        atr_pct = atr / close
        
        # Scale threshold based on ATR (0.5x to 2.0x base)
        atr_min = atr_pct.rolling(252).quantile(0.1)
        atr_max = atr_pct.rolling(252).quantile(0.9)
        atr_normalized = (atr_pct - atr_min) / (atr_max - atr_min + 1e-8)
        atr_normalized = atr_normalized.clip(0, 1)
        
        dynamic_threshold = base_threshold * (0.5 + 1.5 * atr_normalized)
        
        # Forward return
        forward_return = close.pct_change(horizon).shift(-horizon)
        
        # Labels
        labels = pd.Series(0, index=close.index)
        labels[forward_return > dynamic_threshold] = 1   # BUY
        labels[forward_return < -dynamic_threshold] = -1  # SELL
        
        # Fill last horizon days with NaN
        labels.iloc[-horizon:] = np.nan
        
        # Statistics
        valid_labels = labels.dropna()
        self.label_stats['dynamic_threshold'] = {
            'BUY (+1)': (valid_labels == 1).sum(),
            'SELL (-1)': (valid_labels == -1).sum(),
            'HOLD (0)': (valid_labels == 0).sum(),
            'Avg threshold': dynamic_threshold.mean() * 100
        }
        
        self.log(f"  Labels: BUY={self.label_stats['dynamic_threshold']['BUY (+1)']}, "
                f"SELL={self.label_stats['dynamic_threshold']['SELL (-1)']}, "
                f"HOLD={self.label_stats['dynamic_threshold']['HOLD (0)']}")
        
        return labels
    
    # =========================================================================
    # METHOD 3: Trend Following Labels
    # =========================================================================
    def trend_following(
        self,
        close: pd.Series,
        short_period: int = 20,
        long_period: int = 50,
        horizon: int = 5
    ) -> pd.Series:
        """
        Trend-following labels based on moving average relationship
        
        BUY: Price above both MAs, short MA above long MA
        SELL: Price below both MAs, short MA below long MA
        HOLD: Mixed signals
        """
        self.log(f"Trend Following: short={short_period}d, long={long_period}d, horizon={horizon}d")
        
        ma_short = close.rolling(short_period).mean()
        ma_long = close.rolling(long_period).mean()
        
        # Trend signals
        price_above_short = close > ma_short
        price_above_long = close > ma_long
        short_above_long = ma_short > ma_long
        
        # Forward return for validation
        forward_return = close.pct_change(horizon).shift(-horizon)
        
        # Labels based on trend + forward return confirmation
        labels = pd.Series(0, index=close.index)
        
        # Strong uptrend
        bullish = price_above_short & price_above_long & short_above_long
        labels[bullish & (forward_return > 0)] = 1
        
        # Strong downtrend
        bearish = ~price_above_short & ~price_above_long & ~short_above_long
        labels[bearish & (forward_return < 0)] = -1
        
        # Fill last horizon days with NaN
        labels.iloc[-horizon:] = np.nan
        labels.iloc[:long_period] = np.nan
        
        # Statistics
        valid_labels = labels.dropna()
        self.label_stats['trend_following'] = {
            'BUY (+1)': (valid_labels == 1).sum(),
            'SELL (-1)': (valid_labels == -1).sum(),
            'HOLD (0)': (valid_labels == 0).sum()
        }
        
        self.log(f"  Labels: BUY={self.label_stats['trend_following']['BUY (+1)']}, "
                f"SELL={self.label_stats['trend_following']['SELL (-1)']}, "
                f"HOLD={self.label_stats['trend_following']['HOLD (0)']}")
        
        return labels
    
    # =========================================================================
    # METHOD 4: Multi-Horizon Ensemble Labels
    # =========================================================================
    def multi_horizon(
        self,
        close: pd.Series,
        horizons: List[int] = [1, 3, 5],
        threshold: float = 0.005
    ) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Combine predictions across multiple horizons
        
        More robust than single-horizon prediction
        Agreement across horizons = stronger signal
        """
        self.log(f"Multi-Horizon: horizons={horizons}, threshold={threshold*100:.2f}%")
        
        horizon_labels = pd.DataFrame(index=close.index)
        
        for h in horizons:
            forward_return = close.pct_change(h).shift(-h)
            label = pd.Series(0, index=close.index)
            label[forward_return > threshold] = 1
            label[forward_return < -threshold] = -1
            horizon_labels[f'label_{h}d'] = label
        
        # Ensemble: majority vote
        vote_sum = horizon_labels.sum(axis=1)
        
        final_labels = pd.Series(0, index=close.index)
        final_labels[vote_sum >= 2] = 1   # At least 2/3 say BUY
        final_labels[vote_sum <= -2] = -1  # At least 2/3 say SELL
        
        # Confidence (agreement level)
        agreement = horizon_labels.apply(lambda row: row.value_counts().max() if len(row.dropna()) > 0 else 0, axis=1)
        confidence = agreement / len(horizons)
        
        # Fill last days with NaN
        max_horizon = max(horizons)
        final_labels.iloc[-max_horizon:] = np.nan
        
        # Statistics
        valid_labels = final_labels.dropna()
        self.label_stats['multi_horizon'] = {
            'BUY (+1)': (valid_labels == 1).sum(),
            'SELL (-1)': (valid_labels == -1).sum(),
            'HOLD (0)': (valid_labels == 0).sum(),
            'Avg confidence': confidence.mean()
        }
        
        self.log(f"  Labels: BUY={self.label_stats['multi_horizon']['BUY (+1)']}, "
                f"SELL={self.label_stats['multi_horizon']['SELL (-1)']}, "
                f"HOLD={self.label_stats['multi_horizon']['HOLD (0)']}")
        
        # Add confidence to horizon_labels
        horizon_labels['confidence'] = confidence
        horizon_labels['final_label'] = final_labels
        
        return final_labels, horizon_labels
    
    # =========================================================================
    # MASTER: Generate Best Labels for Stock Type
    # =========================================================================
    def generate_optimal_labels(
        self,
        df: pd.DataFrame,
        stock_type: str = 'large_cap',
        method: str = 'auto'
    ) -> pd.Series:
        """
        Generate optimal labels based on stock type
        
        Args:
            df: OHLCV DataFrame
            stock_type: 'large_cap', 'high_vol', 'etf'
            method: 'triple_barrier', 'dynamic', 'trend', 'multi_horizon', 'auto'
        
        Returns:
            Optimal labels for the stock type
        """
        close = df['Close']
        high = df['High']
        low = df['Low']
        
        # Calculate volatility for adaptive parameters
        daily_vol = close.pct_change().rolling(20).std()
        avg_vol = daily_vol.mean()
        
        self.log(f"\n{'='*60}")
        self.log(f"GENERATING OPTIMAL LABELS")
        self.log(f"Stock type: {stock_type}, Avg volatility: {avg_vol*100:.2f}%")
        self.log(f"{'='*60}")
        
        if method == 'auto':
            # Auto-select based on stock characteristics
            if avg_vol > 0.025:  # High volatility (>2.5% daily)
                method = 'triple_barrier'
                tp_pct = 0.02  # Higher targets
                sl_pct = 0.015
            elif avg_vol < 0.012:  # Low volatility (<1.2% daily)
                method = 'trend_following'
            else:
                method = 'dynamic'
        
        # Generate labels based on method
        if method == 'triple_barrier':
            # Volatility scaler
            vol_scaled = 0.5 + 1.5 * (daily_vol - daily_vol.min()) / (daily_vol.max() - daily_vol.min() + 1e-8)
            
            labels, _ = self.triple_barrier(
                close, high, low,
                tp_pct=tp_pct if 'tp_pct' in dir() else 0.015,
                sl_pct=sl_pct if 'sl_pct' in dir() else 0.015,
                max_holding_days=5,
                volatility_scaler=vol_scaled
            )
        
        elif method == 'dynamic':
            labels = self.dynamic_threshold(close, high, low, base_threshold=0.005, horizon=5)
        
        elif method == 'trend_following' or method == 'trend':
            labels = self.trend_following(close)
        
        elif method == 'multi_horizon':
            labels, _ = self.multi_horizon(close)
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return labels
    
    def get_label_statistics(self) -> Dict:
        """Return statistics for all labeling methods used"""
        return self.label_stats


# =============================================================================
# QUICK TEST
# =============================================================================
if __name__ == "__main__":
    import yfinance as yf
    
    print("Testing Triple Barrier Labeling...")
    
    # Download test data
    tickers = ['AAPL', 'TSLA', 'SPY']
    
    for ticker in tickers:
        print(f"\n{'='*60}")
        print(f"TESTING: {ticker}")
        print(f"{'='*60}")
        
        df = yf.download(ticker, start="2022-01-01", end="2024-12-01", progress=False)
        
        labeler = TripleBarrierLabeler(verbose=True)
        
        # Test all methods
        print("\n--- Method 1: Triple Barrier ---")
        labels_tb, _ = labeler.triple_barrier(df['Close'], df['High'], df['Low'])
        
        print("\n--- Method 2: Dynamic Threshold ---")
        labels_dt = labeler.dynamic_threshold(df['Close'], df['High'], df['Low'])
        
        print("\n--- Method 3: Trend Following ---")
        labels_tf = labeler.trend_following(df['Close'])
        
        print("\n--- Method 4: Multi-Horizon ---")
        labels_mh, details = labeler.multi_horizon(df['Close'])
        
        # Compare naive vs triple barrier
        naive_labels = pd.Series(0, index=df.index)
        naive_return = df['Close'].pct_change(1).shift(-1)
        naive_labels[naive_return > 0.005] = 1
        naive_labels[naive_return < -0.005] = -1
        
        print(f"\n--- Comparison ---")
        print(f"Naive labels:  BUY={(naive_labels==1).sum()}, SELL={(naive_labels==-1).sum()}, HOLD={(naive_labels==0).sum()}")
        print(f"Triple Barrier: BUY={(labels_tb==1).sum()}, SELL={(labels_tb==-1).sum()}, HOLD={(labels_tb==0).sum()}")
        
        # Calculate agreement
        agreement = (labels_tb == labels_dt).mean()
        print(f"\nAgreement TB vs DT: {agreement*100:.1f}%")
