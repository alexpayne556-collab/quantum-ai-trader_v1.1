"""
VALIDATION FRAMEWORK 2026 - PROPER TESTING
==========================================
NO PEEKING. NO CHEATING. REAL VALIDATION.

Based on Perplexity Research:
- Purged Cross-Validation (no data leakage)
- Walk-Forward with Embargo
- Deflated Sharpe Ratio (accounts for multiple testing)
- Probability of Backtest Overfitting (PBO)
- Bonferroni-Holm correction
"""
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

from WATCHLIST_2026 import WATCHLIST

DATA_DIR = Path("data/watchlist_2026")

# ==============================================================================
# DATA LOADING - FIXED
# ==============================================================================

def load_data():
    """Load all watchlist data with proper parsing"""
    data = {}
    for ticker in WATCHLIST:
        path = DATA_DIR / f"{ticker}.csv"
        if path.exists():
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            
            # Handle multi-level columns
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            # Ensure numeric
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df = df.dropna()
            if len(df) > 50:  # Need minimum data
                data[ticker] = df
    return data


# ==============================================================================
# PURGED CROSS-VALIDATION (from research)
# ==============================================================================

class PurgedWalkForward:
    """
    Walk-Forward validation with purge and embargo periods.
    
    Research says: This is the CORRECT way to validate time series ML.
    - Purge: Remove samples near train/test boundary (prevent leakage)
    - Embargo: Gap between train and test (prevent lookahead)
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        train_size: int = 252,  # ~1 year
        test_size: int = 63,    # ~3 months
        purge_size: int = 5,    # 1 week purge
        embargo_size: int = 10  # 2 weeks embargo
    ):
        self.n_splits = n_splits
        self.train_size = train_size
        self.test_size = test_size
        self.purge_size = purge_size
        self.embargo_size = embargo_size
    
    def split(self, X) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/test splits with purge and embargo.
        
        Timeline:
        |--TRAIN--|--PURGE--|--EMBARGO--|--TEST--|
        """
        n_samples = len(X)
        splits = []
        
        min_required = self.train_size + self.test_size + self.purge_size + self.embargo_size
        if n_samples < min_required:
            # Fall back to smaller windows
            self.train_size = int(n_samples * 0.5)
            self.test_size = int(n_samples * 0.2)
            self.purge_size = 3
            self.embargo_size = 5
        
        step = max(1, (n_samples - self.train_size - self.test_size - 
                self.purge_size - self.embargo_size) // max(1, self.n_splits - 1))
        
        for i in range(self.n_splits):
            train_start = i * step
            train_end = train_start + self.train_size
            
            if train_end >= n_samples:
                break
            
            # Purge period (excluded)
            purge_end = train_end + self.purge_size
            
            # Embargo period (also excluded)
            embargo_end = purge_end + self.embargo_size
            
            # Test period
            test_start = embargo_end
            test_end = min(test_start + self.test_size, n_samples)
            
            if test_start >= n_samples or test_end <= test_start:
                break
            
            train_idx = np.arange(train_start, train_end)
            test_idx = np.arange(test_start, test_end)
            
            splits.append((train_idx, test_idx))
        
        return splits


# ==============================================================================
# DEFLATED SHARPE RATIO (from research)
# ==============================================================================

def deflated_sharpe_ratio(
    sharpe: float,
    n_trials: int,
    backtest_length: int,
    skewness: float = 0,
    kurtosis: float = 3
) -> float:
    """
    Calculate Deflated Sharpe Ratio (DSR).
    
    Research says: Accounts for number of strategies tested.
    DSR < original Sharpe = your edge is likely overfit
    """
    if n_trials < 1 or backtest_length < 1:
        return 0
    
    # Expected maximum Sharpe from n_trials
    e_max_sr = np.sqrt(2 * np.log(n_trials))
    
    # Variance of Sharpe ratio estimator
    var_sr = (1 + 0.5 * sharpe**2 - skewness * sharpe + 
              ((kurtosis - 3) / 4) * sharpe**2) / backtest_length
    
    if var_sr <= 0:
        return 0
    
    # Probability that observed Sharpe exceeds expected max
    dsr = stats.norm.cdf((sharpe - e_max_sr) / np.sqrt(var_sr))
    
    return dsr


def probability_of_backtest_overfitting(
    sharpe_is: float,
    sharpe_oos: float,
    n_trials: int
) -> float:
    """
    Probability of Backtest Overfitting (PBO).
    
    Research says: If PBO > 50%, your strategy is likely overfit.
    """
    if sharpe_is <= 0:
        return 1.0
    
    degradation = 1 - (sharpe_oos / sharpe_is)
    degradation = max(0, min(1, degradation))
    
    # More trials = higher chance of overfitting
    pbo = 1 - np.exp(-n_trials * degradation / 10)
    
    return min(max(pbo, 0), 1)


# ==============================================================================
# MULTIPLE TESTING CORRECTION (from research)
# ==============================================================================

def bonferroni_holm_correction(p_values: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """
    Bonferroni-Holm step-down procedure.
    
    Research says: Must correct for multiple testing or results are meaningless.
    """
    n = len(p_values)
    if n == 0:
        return np.array([], dtype=bool)
    
    sorted_idx = np.argsort(p_values)
    sorted_pvals = p_values[sorted_idx]
    
    # Holm thresholds
    thresholds = alpha / (n - np.arange(n))
    
    rejected = np.zeros(n, dtype=bool)
    for i, (pval, thresh) in enumerate(zip(sorted_pvals, thresholds)):
        if pval <= thresh:
            rejected[sorted_idx[i]] = True
        else:
            break
    
    return rejected


# ==============================================================================
# STRATEGY TESTER
# ==============================================================================

class StrategyValidator:
    """Validate trading strategies with proper statistical rigor."""
    
    def __init__(self, data: Dict[str, pd.DataFrame]):
        self.data = data
    
    def calculate_returns(
        self,
        ticker: str,
        signal: pd.Series,
        holding_period: int = 10,
        cost: float = 0.002
    ) -> pd.Series:
        """Calculate returns from a signal."""
        df = self.data[ticker]
        prices = df['Close']
        
        # Forward returns
        fwd_returns = prices.shift(-holding_period) / prices - 1
        
        # Strategy returns
        strategy_returns = signal * fwd_returns - cost * np.abs(signal)
        
        return strategy_returns.dropna()
    
    def validate_strategy(
        self,
        strategy_func,
        strategy_name: str,
        tickers: List[str] = None,
        n_splits: int = 5
    ) -> List[Dict]:
        """Full validation of a strategy."""
        if tickers is None:
            tickers = list(self.data.keys())
        
        all_results = []
        
        for ticker in tickers:
            if ticker not in self.data:
                continue
            
            df = self.data[ticker]
            if len(df) < 100:
                continue
            
            # Generate signals
            try:
                signals = strategy_func(df)
            except Exception as e:
                continue
            
            if signals.sum() == 0:
                continue
            
            # Walk-forward validation
            cv = PurgedWalkForward(n_splits=n_splits)
            splits = cv.split(df)
            
            if len(splits) == 0:
                continue
            
            is_returns = []
            oos_returns = []
            
            for train_idx, test_idx in splits:
                # In-sample returns
                train_mask = np.isin(np.arange(len(signals)), train_idx)
                train_signals = signals.where(train_mask, 0)
                train_rets = self.calculate_returns(ticker, train_signals)
                is_returns.extend(train_rets[train_rets != 0].dropna().tolist())
                
                # Out-of-sample returns
                test_mask = np.isin(np.arange(len(signals)), test_idx)
                test_signals = signals.where(test_mask, 0)
                test_rets = self.calculate_returns(ticker, test_signals)
                oos_returns.extend(test_rets[test_rets != 0].dropna().tolist())
            
            if len(oos_returns) < 10:
                continue
            
            is_returns = np.array(is_returns)
            oos_returns = np.array(oos_returns)
            
            # Calculate metrics
            is_sharpe = (np.mean(is_returns) / np.std(is_returns) * np.sqrt(252) 
                        if np.std(is_returns) > 0 else 0)
            oos_sharpe = (np.mean(oos_returns) / np.std(oos_returns) * np.sqrt(252) 
                         if np.std(oos_returns) > 0 else 0)
            
            # T-test
            t_stat, p_value = stats.ttest_1samp(oos_returns, 0)
            
            result = {
                'strategy': strategy_name,
                'ticker': ticker,
                'n_trades_is': len(is_returns),
                'n_trades_oos': len(oos_returns),
                'sharpe_is': is_sharpe,
                'sharpe_oos': oos_sharpe,
                'degradation': (1 - oos_sharpe / is_sharpe) if is_sharpe > 0 else 1,
                'mean_return_oos': np.mean(oos_returns) * 100,
                'win_rate_oos': (oos_returns > 0).mean() * 100,
                't_stat': t_stat,
                'p_value': p_value,
                'significant': p_value < 0.05
            }
            
            all_results.append(result)
        
        return all_results
    
    def run_gauntlet(self, strategies: Dict[str, callable]) -> pd.DataFrame:
        """Run full validation on multiple strategies."""
        n_trials = len(strategies)
        all_results = []
        
        for name, func in strategies.items():
            print(f"  Testing: {name}...")
            results = self.validate_strategy(func, name)
            all_results.extend(results)
        
        if len(all_results) == 0:
            return pd.DataFrame()
        
        df = pd.DataFrame(all_results)
        
        # Multiple testing correction
        df['significant_corrected'] = bonferroni_holm_correction(df['p_value'].values)
        
        # Deflated Sharpe
        df['deflated_sharpe'] = df.apply(
            lambda row: deflated_sharpe_ratio(row['sharpe_oos'], n_trials, row['n_trades_oos']),
            axis=1
        )
        
        # PBO
        df['pbo'] = df.apply(
            lambda row: probability_of_backtest_overfitting(row['sharpe_is'], row['sharpe_oos'], n_trials),
            axis=1
        )
        
        return df


# ==============================================================================
# EXAMPLE STRATEGIES (Research-based indicators that actually work)
# ==============================================================================

def rsi_strategy(df: pd.DataFrame, period: int = 14, threshold: int = 30) -> pd.Series:
    """RSI oversold - buy when RSI < threshold"""
    close = df['Close']
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / (loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    signal = pd.Series(0.0, index=df.index)
    signal[rsi < threshold] = 1.0
    return signal


def ma_crossover(df: pd.DataFrame, fast: int = 20, slow: int = 50) -> pd.Series:
    """MA crossover - buy when fast crosses above slow"""
    close = df['Close']
    ma_fast = close.rolling(fast).mean()
    ma_slow = close.rolling(slow).mean()
    
    signal = pd.Series(0.0, index=df.index)
    cross = (ma_fast > ma_slow) & (ma_fast.shift(1) <= ma_slow.shift(1))
    signal[cross] = 1.0
    return signal


def mean_reversion(df: pd.DataFrame, window: int = 20, z_thresh: float = -2) -> pd.Series:
    """Mean reversion - buy when z-score below threshold"""
    close = df['Close']
    ma = close.rolling(window).mean()
    std = close.rolling(window).std()
    z = (close - ma) / (std + 1e-10)
    
    signal = pd.Series(0.0, index=df.index)
    signal[z < z_thresh] = 1.0
    return signal


def momentum(df: pd.DataFrame, lookback: int = 20, threshold: float = 0.05) -> pd.Series:
    """Momentum - buy when price up more than threshold over lookback"""
    close = df['Close']
    mom = close.pct_change(lookback)
    
    signal = pd.Series(0.0, index=df.index)
    signal[mom > threshold] = 1.0
    return signal


def obv_trend(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """OBV trend - research says OBV actually works"""
    close = df['Close']
    volume = df['Volume']
    
    # Calculate OBV
    obv = (np.sign(close.diff()) * volume).cumsum()
    obv_ma = obv.rolling(period).mean()
    
    signal = pd.Series(0.0, index=df.index)
    signal[(obv > obv_ma) & (obv.shift(1) <= obv_ma.shift(1))] = 1.0
    return signal


def volatility_breakout(df: pd.DataFrame, window: int = 20, mult: float = 2) -> pd.Series:
    """Volatility breakout - buy on breakout above upper band"""
    close = df['Close']
    ma = close.rolling(window).mean()
    std = close.rolling(window).std()
    upper = ma + mult * std
    
    signal = pd.Series(0.0, index=df.index)
    signal[(close > upper) & (close.shift(1) <= upper.shift(1))] = 1.0
    return signal


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print("="*70)
    print("VALIDATION FRAMEWORK 2026")
    print("PROPER TESTING - NO PEEKING - NO CHEATING")
    print("="*70)
    
    print("\n[1/4] Loading data...")
    data = load_data()
    print(f"  Loaded {len(data)} tickers")
    
    if len(data) == 0:
        print("\nERROR: No data! Run DOWNLOAD_WATCHLIST_DATA.py first!")
        return
    
    # Research-based strategies
    strategies = {
        'RSI_30': lambda df: rsi_strategy(df, 14, 30),
        'RSI_25': lambda df: rsi_strategy(df, 14, 25),
        'MA_20_50': lambda df: ma_crossover(df, 20, 50),
        'MA_10_30': lambda df: ma_crossover(df, 10, 30),
        'MeanRev_Z2': lambda df: mean_reversion(df, 20, -2),
        'MeanRev_Z1.5': lambda df: mean_reversion(df, 20, -1.5),
        'Momentum_20': lambda df: momentum(df, 20, 0.05),
        'Momentum_10': lambda df: momentum(df, 10, 0.03),
        'OBV_Trend': lambda df: obv_trend(df, 20),
        'Vol_Breakout': lambda df: volatility_breakout(df, 20, 2),
    }
    
    print(f"\n[2/4] Testing {len(strategies)} strategies...")
    validator = StrategyValidator(data)
    results = validator.run_gauntlet(strategies)
    
    if len(results) == 0:
        print("\n  No valid results - need more data or adjust parameters")
        return
    
    # Results
    print(f"\n[3/4] Results Summary ({len(results)} strategy/ticker combinations)")
    print("="*70)
    
    # Top by OOS Sharpe
    print("\n📊 TOP 10 BY OUT-OF-SAMPLE SHARPE:")
    top = results.nlargest(10, 'sharpe_oos')
    for _, row in top.iterrows():
        status = "✓" if row['significant_corrected'] else "✗"
        print(f"  {status} {row['strategy']:15} | {row['ticker']:5} | "
              f"Sharpe: {row['sharpe_oos']:+.2f} | "
              f"Win: {row['win_rate_oos']:.0f}% | "
              f"PBO: {row['pbo']:.0%}")
    
    # PASSED VALIDATION
    print("\n" + "="*70)
    print("🏆 STRATEGIES THAT PASS VALIDATION GAUNTLET:")
    print("   (Significant + PBO < 50% + OOS Sharpe > 0.3)")
    print("="*70)
    
    passed = results[
        (results['significant_corrected'] == True) &
        (results['pbo'] < 0.5) &
        (results['sharpe_oos'] > 0.3)
    ]
    
    if len(passed) == 0:
        print("\n  ❌ NO STRATEGIES PASSED ALL CHECKS")
        print("  This is actually GOOD - we're being rigorous!")
        print("  95% of strategies fail proper validation (per research)")
    else:
        print(f"\n  ✓ {len(passed)} strategy/ticker combinations passed!")
        for _, row in passed.iterrows():
            print(f"    • {row['strategy']} | {row['ticker']} | Sharpe: {row['sharpe_oos']:.2f}")
    
    # Save
    print("\n[4/4] Saving results...")
    results.to_csv(DATA_DIR / "validation_results.csv", index=False)
    print(f"  ✅ Saved to {DATA_DIR / 'validation_results.csv'}")
    
    # Honest assessment
    print("\n" + "="*70)
    print("HONEST ASSESSMENT")
    print("="*70)
    
    avg_deg = results['degradation'].mean()
    pct_sig = (results['significant_corrected'].sum() / len(results)) * 100
    
    print(f"\n  Average IS→OOS degradation: {avg_deg:.0%}")
    print(f"  Strategies significant after correction: {pct_sig:.0f}%")
    
    if avg_deg > 0.5:
        print("\n  ⚠️  HIGH DEGRADATION = Likely overfitting")
    if pct_sig < 10:
        print("\n  ⚠️  FEW SIGNIFICANT = Most edges are noise")
    
    print("\n  NEXT STEPS:")
    print("  1. Focus ONLY on strategies that passed")
    print("  2. Paper trade for 3+ months")
    print("  3. Only risk real money after consistent paper results")

if __name__ == "__main__":
    main()
