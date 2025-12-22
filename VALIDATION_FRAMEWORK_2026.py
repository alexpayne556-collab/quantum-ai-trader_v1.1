"""
VALIDATION FRAMEWORK 2026 - PROPER TESTING
==========================================
NO PEEKING. NO CHEATING. REAL VALIDATION.

Based on research:
- Purged Cross-Validation (no data leakage)
- Walk-Forward with Embargo
- Deflated Sharpe Ratio
- Multiple Testing Correction
"""
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from sklearn.model_selection import TimeSeriesSplit
from typing import List, Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

from WATCHLIST_2026 import WATCHLIST

DATA_DIR = Path("data/watchlist_2026")

# ==============================================================================
# PURGED CROSS-VALIDATION
# ==============================================================================

class PurgedWalkForward:
    """
    Walk-Forward validation with purge and embargo periods.
    
    - Purge: Remove samples near train/test boundary (prevent leakage)
    - Embargo: Gap between train and test (prevent lookahead)
    
    This is the CORRECT way to validate time series ML.
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
        
        # Calculate step size
        step = (n_samples - self.train_size - self.test_size - 
                self.purge_size - self.embargo_size) // (self.n_splits - 1)
        
        if step < 1:
            step = 1
        
        for i in range(self.n_splits):
            train_start = i * step
            train_end = train_start + self.train_size
            
            # Purge period (excluded from both train and test)
            purge_end = train_end + self.purge_size
            
            # Embargo period (also excluded)
            embargo_end = purge_end + self.embargo_size
            
            # Test period
            test_start = embargo_end
            test_end = min(test_start + self.test_size, n_samples)
            
            if test_end > n_samples:
                break
            
            train_idx = np.arange(train_start, train_end)
            test_idx = np.arange(test_start, test_end)
            
            splits.append((train_idx, test_idx))
        
        return splits


# ==============================================================================
# DEFLATED SHARPE RATIO
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
    
    Accounts for:
    - Number of strategies tested (multiple testing)
    - Length of backtest
    - Non-normality of returns
    
    DSR < original Sharpe = your edge is likely overfit
    """
    # Expected maximum Sharpe from n_trials
    e_max_sr = np.sqrt(2 * np.log(n_trials))
    
    # Variance of Sharpe ratio estimator
    var_sr = (1 + 0.5 * sharpe**2 - skewness * sharpe + 
              ((kurtosis - 3) / 4) * sharpe**2) / backtest_length
    
    # Probability that observed Sharpe exceeds expected max
    dsr = stats.norm.cdf((sharpe - e_max_sr) / np.sqrt(var_sr))
    
    return dsr


def probability_of_backtest_overfitting(
    sharpe_is: float,  # In-sample Sharpe
    sharpe_oos: float,  # Out-of-sample Sharpe
    n_trials: int
) -> float:
    """
    Probability of Backtest Overfitting (PBO).
    
    If PBO > 50%, your strategy is likely overfit.
    """
    # Simple approximation
    degradation = 1 - (sharpe_oos / sharpe_is) if sharpe_is > 0 else 1
    
    # More trials = higher chance of overfitting
    pbo = 1 - np.exp(-n_trials * degradation / 10)
    
    return min(max(pbo, 0), 1)


# ==============================================================================
# MULTIPLE TESTING CORRECTION
# ==============================================================================

def bonferroni_holm_correction(p_values: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """
    Bonferroni-Holm step-down procedure.
    
    Returns boolean array: True = significant after correction
    """
    n = len(p_values)
    sorted_idx = np.argsort(p_values)
    sorted_pvals = p_values[sorted_idx]
    
    # Holm thresholds
    thresholds = alpha / (n - np.arange(n))
    
    # Find first non-rejection
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
    """
    Validate trading strategies with proper statistical rigor.
    """
    
    def __init__(self, data: Dict[str, pd.DataFrame]):
        self.data = data
        self.results = []
    
    def calculate_signal_returns(
        self,
        ticker: str,
        signal: pd.Series,
        holding_period: int = 10,
        transaction_cost: float = 0.002  # 0.2% round trip
    ) -> pd.Series:
        """
        Calculate returns from a signal.
        
        Signal: 1 = buy, -1 = sell, 0 = neutral
        """
        df = self.data[ticker]
        close_col = 'Close' if 'Close' in df.columns else 'close'
        prices = df[close_col]
        
        # Forward returns
        fwd_returns = prices.shift(-holding_period) / prices - 1
        
        # Strategy returns (signal * forward return - costs)
        strategy_returns = signal * fwd_returns - transaction_cost * np.abs(signal)
        
        return strategy_returns.dropna()
    
    def validate_strategy(
        self,
        strategy_func,
        strategy_name: str,
        tickers: List[str] = None,
        n_splits: int = 5,
        min_samples: int = 100
    ) -> Dict:
        """
        Full validation of a strategy.
        """
        if tickers is None:
            tickers = list(self.data.keys())
        
        all_results = []
        
        for ticker in tickers:
            if ticker not in self.data:
                continue
            
            df = self.data[ticker]
            if len(df) < min_samples:
                continue
            
            # Generate signals
            signals = strategy_func(df)
            
            if signals.sum() == 0:
                continue
            
            # Walk-forward validation
            cv = PurgedWalkForward(n_splits=n_splits)
            
            is_returns = []  # In-sample
            oos_returns = []  # Out-of-sample
            
            for train_idx, test_idx in cv.split(df):
                # In-sample
                train_signals = signals.iloc[train_idx]
                train_rets = self.calculate_signal_returns(
                    ticker, train_signals
                ).iloc[train_idx[train_idx < len(signals)]]
                is_returns.extend(train_rets.dropna().tolist())
                
                # Out-of-sample
                test_signals = signals.iloc[test_idx]
                test_rets = self.calculate_signal_returns(
                    ticker, test_signals
                ).iloc[test_idx[test_idx < len(signals)]]
                oos_returns.extend(test_rets.dropna().tolist())
            
            if len(oos_returns) < 20:
                continue
            
            # Calculate metrics
            is_returns = np.array(is_returns)
            oos_returns = np.array(oos_returns)
            
            is_sharpe = np.mean(is_returns) / np.std(is_returns) * np.sqrt(252) if np.std(is_returns) > 0 else 0
            oos_sharpe = np.mean(oos_returns) / np.std(oos_returns) * np.sqrt(252) if np.std(oos_returns) > 0 else 0
            
            # T-statistic for OOS
            t_stat, p_value = stats.ttest_1samp(oos_returns, 0)
            
            result = {
                'strategy': strategy_name,
                'ticker': ticker,
                'n_trades_is': len(is_returns),
                'n_trades_oos': len(oos_returns),
                'sharpe_is': is_sharpe,
                'sharpe_oos': oos_sharpe,
                'degradation': 1 - (oos_sharpe / is_sharpe) if is_sharpe > 0 else 1,
                'mean_return_oos': np.mean(oos_returns) * 100,
                'win_rate_oos': (oos_returns > 0).mean() * 100,
                't_stat': t_stat,
                'p_value': p_value,
                'significant': p_value < 0.05
            }
            
            all_results.append(result)
        
        return all_results
    
    def run_validation_gauntlet(
        self,
        strategies: Dict[str, callable],
        n_trials: int = None
    ) -> pd.DataFrame:
        """
        Run full validation on multiple strategies.
        """
        if n_trials is None:
            n_trials = len(strategies)
        
        all_results = []
        
        for name, func in strategies.items():
            print(f"  Testing: {name}...")
            results = self.validate_strategy(func, name)
            all_results.extend(results)
        
        df = pd.DataFrame(all_results)
        
        if len(df) == 0:
            return df
        
        # Apply multiple testing correction
        if 'p_value' in df.columns:
            df['significant_corrected'] = bonferroni_holm_correction(
                df['p_value'].values
            )
        
        # Calculate deflated Sharpe
        df['deflated_sharpe'] = df.apply(
            lambda row: deflated_sharpe_ratio(
                row['sharpe_oos'],
                n_trials,
                row['n_trades_oos']
            ),
            axis=1
        )
        
        # Calculate PBO
        df['pbo'] = df.apply(
            lambda row: probability_of_backtest_overfitting(
                row['sharpe_is'],
                row['sharpe_oos'],
                n_trials
            ),
            axis=1
        )
        
        return df


# ==============================================================================
# EXAMPLE STRATEGIES TO TEST
# ==============================================================================

def rsi_oversold_strategy(df: pd.DataFrame, period: int = 14, threshold: int = 30) -> pd.Series:
    """Buy when RSI < threshold"""
    close = df['Close'] if 'Close' in df.columns else df['close']
    
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    signal = pd.Series(0, index=df.index)
    signal[rsi < threshold] = 1  # Buy when oversold
    
    return signal


def ma_crossover_strategy(df: pd.DataFrame, fast: int = 20, slow: int = 50) -> pd.Series:
    """Buy when fast MA crosses above slow MA"""
    close = df['Close'] if 'Close' in df.columns else df['close']
    
    ma_fast = close.rolling(fast).mean()
    ma_slow = close.rolling(slow).mean()
    
    signal = pd.Series(0, index=df.index)
    signal[(ma_fast > ma_slow) & (ma_fast.shift(1) <= ma_slow.shift(1))] = 1
    
    return signal


def mean_reversion_strategy(df: pd.DataFrame, window: int = 20, z_threshold: float = -2) -> pd.Series:
    """Buy when price is z_threshold std devs below mean"""
    close = df['Close'] if 'Close' in df.columns else df['close']
    
    ma = close.rolling(window).mean()
    std = close.rolling(window).std()
    z_score = (close - ma) / std
    
    signal = pd.Series(0, index=df.index)
    signal[z_score < z_threshold] = 1  # Buy when oversold
    
    return signal


def momentum_strategy(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
    """Buy when momentum is positive"""
    close = df['Close'] if 'Close' in df.columns else df['close']
    
    momentum = close.pct_change(lookback)
    
    signal = pd.Series(0, index=df.index)
    signal[momentum > 0.05] = 1  # Buy when +5% over lookback
    
    return signal


# ==============================================================================
# MAIN
# ==============================================================================

def load_data():
    """Load all watchlist data"""
    data = {}
    for ticker in WATCHLIST:
        path = DATA_DIR / f"{ticker}.csv"
        if path.exists():
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            data[ticker] = df
    return data


def main():
    print("="*70)
    print("VALIDATION FRAMEWORK 2026")
    print("PROPER TESTING - NO PEEKING - NO CHEATING")
    print("="*70)
    
    # Load data
    print("\n[1/4] Loading data...")
    data = load_data()
    print(f"  Loaded {len(data)} tickers")
    
    # Define strategies to test
    strategies = {
        'RSI_Oversold_30': lambda df: rsi_oversold_strategy(df, 14, 30),
        'RSI_Oversold_25': lambda df: rsi_oversold_strategy(df, 14, 25),
        'MA_Cross_20_50': lambda df: ma_crossover_strategy(df, 20, 50),
        'MA_Cross_10_30': lambda df: ma_crossover_strategy(df, 10, 30),
        'Mean_Rev_Z2': lambda df: mean_reversion_strategy(df, 20, -2),
        'Mean_Rev_Z1.5': lambda df: mean_reversion_strategy(df, 20, -1.5),
        'Momentum_20': lambda df: momentum_strategy(df, 20),
        'Momentum_10': lambda df: momentum_strategy(df, 10),
    }
    
    # Run validation
    print(f"\n[2/4] Testing {len(strategies)} strategies...")
    validator = StrategyValidator(data)
    results = validator.run_validation_gauntlet(strategies)
    
    if len(results) == 0:
        print("  No valid results!")
        return
    
    # Summary
    print("\n[3/4] Results Summary")
    print("="*70)
    
    # Best strategies (by OOS Sharpe)
    print("\nTOP 10 BY OUT-OF-SAMPLE SHARPE:")
    top = results.nlargest(10, 'sharpe_oos')
    for _, row in top.iterrows():
        status = "✓" if row['significant_corrected'] else "✗"
        print(f"  {status} {row['strategy']} | {row['ticker']} | "
              f"Sharpe: {row['sharpe_oos']:.2f} | "
              f"Win: {row['win_rate_oos']:.0f}% | "
              f"PBO: {row['pbo']:.0%}")
    
    # Strategies that PASS all checks
    print("\n" + "="*70)
    print("STRATEGIES THAT PASS VALIDATION GAUNTLET:")
    print("(Significant after correction + PBO < 50% + OOS Sharpe > 0.5)")
    print("="*70)
    
    passed = results[
        (results['significant_corrected'] == True) &
        (results['pbo'] < 0.5) &
        (results['sharpe_oos'] > 0.5)
    ]
    
    if len(passed) == 0:
        print("\n  ❌ NO STRATEGIES PASSED ALL CHECKS")
        print("  This is actually good - it means we're being rigorous!")
    else:
        print(f"\n  ✓ {len(passed)} strategy/ticker combinations passed!")
        for _, row in passed.iterrows():
            print(f"    • {row['strategy']} | {row['ticker']} | "
                  f"Sharpe: {row['sharpe_oos']:.2f}")
    
    # Save results
    print("\n[4/4] Saving results...")
    results.to_csv(DATA_DIR / "validation_results.csv", index=False)
    print(f"  Saved to {DATA_DIR / 'validation_results.csv'}")
    
    # Final assessment
    print("\n" + "="*70)
    print("HONEST ASSESSMENT")
    print("="*70)
    
    avg_degradation = results['degradation'].mean()
    pct_significant = (results['significant_corrected'].sum() / len(results)) * 100
    
    print(f"\n  Average IS→OOS degradation: {avg_degradation:.0%}")
    print(f"  Strategies significant after correction: {pct_significant:.0f}%")
    
    if avg_degradation > 0.5:
        print("\n  ⚠️  HIGH DEGRADATION = Likely overfitting in backtests")
    if pct_significant < 10:
        print("\n  ⚠️  FEW SIGNIFICANT = Most edges are likely noise")
    
    print("\n  NEXT STEPS:")
    print("  1. Focus only on strategies that PASSED all checks")
    print("  2. Test on completely new data (paper trading)")
    print("  3. Only risk real money after 3+ months of paper trading")

if __name__ == "__main__":
    main()
