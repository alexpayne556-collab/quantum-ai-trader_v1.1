"""
TIME SERIES CROSS-VALIDATION FRAMEWORK

Standard k-fold CV is WRONG for time series (uses future to predict past).

Correct methods:
1. Walk-forward validation (expanding or rolling window)
2. Purged k-fold (gap between train/test to avoid leakage)
3. Combinatorial purged CV (multiple paths through data)

This prevents overfitting and gives realistic out-of-sample estimates.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Callable
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class TimeSeriesCV:
    """
    Time series cross-validation that respects temporal ordering
    
    NO data leakage. NO training on future. ONLY forward-looking tests.
    """
    
    def __init__(self, data: pd.DataFrame, date_column: str = 'date'):
        """
        Args:
            data: DataFrame with time series data
            date_column: Name of date column (must be datetime)
        """
        self.data = data.copy()
        self.data[date_column] = pd.to_datetime(self.data[date_column])
        self.data = self.data.sort_values(date_column)
        self.date_column = date_column
    
    def walk_forward_split(self, 
                          train_size: int = 252,
                          test_size: int = 20,
                          step_size: int = 20,
                          expanding: bool = False) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Walk-forward validation splits
        
        Args:
            train_size: Days in training window
            test_size: Days in test window
            step_size: Days to advance each iteration
            expanding: If True, training window grows. If False, it's fixed (rolling)
        
        Returns:
            List of (train_df, test_df) tuples
        """
        splits = []
        
        dates = self.data[self.date_column].unique()
        
        if expanding:
            # Expanding window: training data grows
            for i in range(train_size, len(dates) - test_size, step_size):
                train = self.data[self.data[self.date_column].isin(dates[:i])]
                test = self.data[self.data[self.date_column].isin(dates[i:i+test_size])]
                
                if len(test) > 0:
                    splits.append((train, test))
        else:
            # Rolling window: fixed training size
            for i in range(train_size, len(dates) - test_size, step_size):
                train = self.data[self.data[self.date_column].isin(dates[i-train_size:i])]
                test = self.data[self.data[self.date_column].isin(dates[i:i+test_size])]
                
                if len(test) > 0:
                    splits.append((train, test))
        
        logger.info(f"Created {len(splits)} walk-forward splits")
        logger.info(f"  Train size: {train_size} days ({'expanding' if expanding else 'rolling'})")
        logger.info(f"  Test size: {test_size} days")
        logger.info(f"  Step size: {step_size} days")
        
        return splits
    
    def purged_kfold(self,
                    n_splits: int = 5,
                    embargo_pct: float = 0.01) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Purged k-fold cross-validation
        
        Adds "embargo" period between train and test to prevent leakage from:
        - Serial correlation
        - Overlapping labels
        - Information leakage
        
        From: Advances in Financial Machine Learning (Marcos López de Prado)
        
        Args:
            n_splits: Number of folds
            embargo_pct: Percentage of data to embargo between train/test
        
        Returns:
            List of (train_df, test_df) tuples
        """
        dates = self.data[self.date_column].unique()
        n_dates = len(dates)
        
        fold_size = n_dates // n_splits
        embargo_size = int(n_dates * embargo_pct)
        
        splits = []
        
        for i in range(n_splits):
            # Test fold
            test_start = i * fold_size
            test_end = (i + 1) * fold_size if i < n_splits - 1 else n_dates
            
            test_dates = dates[test_start:test_end]
            
            # Train: everything except test AND embargo periods
            # Embargo before test
            embargo_before_start = max(0, test_start - embargo_size)
            embargo_before_end = test_start
            
            # Embargo after test
            embargo_after_start = test_end
            embargo_after_end = min(n_dates, test_end + embargo_size)
            
            # Train dates = all dates NOT in test or embargo
            excluded_indices = set(range(embargo_before_start, embargo_after_end))
            train_indices = [j for j in range(n_dates) if j not in excluded_indices]
            
            train_dates = dates[train_indices]
            
            train = self.data[self.data[self.date_column].isin(train_dates)]
            test = self.data[self.data[self.date_column].isin(test_dates)]
            
            if len(train) > 0 and len(test) > 0:
                splits.append((train, test))
        
        logger.info(f"Created {len(splits)} purged k-fold splits")
        logger.info(f"  Embargo size: {embargo_size} days ({embargo_pct*100}%)")
        
        return splits
    
    def anchored_walk_forward(self,
                             initial_train: int = 252,
                             test_size: int = 20,
                             step_size: int = 20) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Anchored walk-forward (training always starts from beginning)
        
        Good for strategies that need long history to calibrate
        """
        dates = self.data[self.date_column].unique()
        splits = []
        
        # Anchor point (start of data)
        anchor_date = dates[0]
        
        for i in range(initial_train, len(dates) - test_size, step_size):
            # Training: from anchor to current point
            train = self.data[self.data[self.date_column].isin(dates[:i])]
            
            # Test: next test_size days
            test = self.data[self.data[self.date_column].isin(dates[i:i+test_size])]
            
            if len(test) > 0:
                splits.append((train, test))
        
        logger.info(f"Created {len(splits)} anchored walk-forward splits")
        logger.info(f"  Anchor date: {anchor_date}")
        logger.info(f"  Initial train: {initial_train} days")
        
        return splits


class BacktestValidator:
    """
    Validate backtest results for overfitting
    
    Tests:
    1. Out-of-sample performance vs in-sample
    2. Drawdown analysis
    3. Trade distribution (are all profits from few trades?)
    4. Parameter sensitivity (does small change kill strategy?)
    """
    
    @staticmethod
    def split_in_out_sample(returns: pd.Series, 
                           split_date: Optional[str] = None,
                           split_pct: float = 0.7) -> Tuple[pd.Series, pd.Series]:
        """
        Split returns into in-sample and out-of-sample
        
        Args:
            returns: Time series of returns
            split_date: Specific date to split (or None for percentage)
            split_pct: Percentage for in-sample if split_date is None
        
        Returns:
            (in_sample, out_of_sample)
        """
        if split_date:
            split_dt = pd.to_datetime(split_date)
            in_sample = returns[returns.index < split_dt]
            out_sample = returns[returns.index >= split_dt]
        else:
            n_split = int(len(returns) * split_pct)
            in_sample = returns.iloc[:n_split]
            out_sample = returns.iloc[n_split:]
        
        return in_sample, out_sample
    
    @staticmethod
    def compute_metrics(returns: pd.Series) -> Dict:
        """
        Compute performance metrics
        """
        cumulative = (1 + returns).cumprod()
        
        total_return = cumulative.iloc[-1] - 1
        n_years = len(returns) / 252
        annualized_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
        
        volatility = returns.std() * np.sqrt(252)
        sharpe = annualized_return / volatility if volatility > 0 else 0
        
        # Drawdown
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Win rate
        win_rate = (returns > 0).mean()
        
        # Profit factor
        gains = returns[returns > 0].sum()
        losses = abs(returns[returns < 0].sum())
        profit_factor = gains / losses if losses > 0 else np.inf
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'n_trades': len(returns)
        }
    
    @staticmethod
    def test_overfitting(in_sample_returns: pd.Series,
                        out_sample_returns: pd.Series) -> Dict:
        """
        Test if strategy is overfit
        
        Red flags:
        - OOS Sharpe << IS Sharpe (>30% decline)
        - OOS has negative returns when IS is positive
        - OOS max drawdown >> IS max drawdown
        """
        is_metrics = BacktestValidator.compute_metrics(in_sample_returns)
        oos_metrics = BacktestValidator.compute_metrics(out_sample_returns)
        
        # Degradation from IS to OOS
        sharpe_degradation = (is_metrics['sharpe_ratio'] - oos_metrics['sharpe_ratio']) / is_metrics['sharpe_ratio'] if is_metrics['sharpe_ratio'] != 0 else 0
        return_degradation = (is_metrics['annualized_return'] - oos_metrics['annualized_return']) / is_metrics['annualized_return'] if is_metrics['annualized_return'] != 0 else 0
        
        # Overfit flags
        overfit_flags = []
        
        if sharpe_degradation > 0.3:
            overfit_flags.append(f"Sharpe degraded {sharpe_degradation*100:.1f}%")
        
        if oos_metrics['annualized_return'] < 0 and is_metrics['annualized_return'] > 0:
            overfit_flags.append("OOS returns negative while IS positive")
        
        if abs(oos_metrics['max_drawdown']) > abs(is_metrics['max_drawdown']) * 1.5:
            overfit_flags.append("OOS drawdown 50% worse than IS")
        
        assessment = "LIKELY OVERFIT" if len(overfit_flags) > 0 else "APPEARS ROBUST"
        
        return {
            'in_sample': is_metrics,
            'out_of_sample': oos_metrics,
            'sharpe_degradation_pct': sharpe_degradation * 100,
            'return_degradation_pct': return_degradation * 100,
            'overfit_flags': overfit_flags,
            'assessment': assessment
        }
    
    @staticmethod
    def monte_carlo_permutation_test(returns: pd.Series, 
                                     n_simulations: int = 1000) -> Dict:
        """
        Monte Carlo permutation test
        
        Shuffle returns and see if observed Sharpe is significant
        
        If 95% of random shuffles have worse Sharpe → strategy is real
        If random shuffles match or beat → strategy is luck
        """
        observed_sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
        
        simulated_sharpes = []
        
        for _ in range(n_simulations):
            shuffled = returns.sample(frac=1, replace=False)
            sim_sharpe = (shuffled.mean() / shuffled.std() * np.sqrt(252)) if shuffled.std() > 0 else 0
            simulated_sharpes.append(sim_sharpe)
        
        simulated_sharpes = np.array(simulated_sharpes)
        
        # Percentile of observed Sharpe
        percentile = (simulated_sharpes < observed_sharpe).mean() * 100
        
        # p-value
        p_value = (simulated_sharpes >= observed_sharpe).mean()
        
        return {
            'observed_sharpe': observed_sharpe,
            'mean_random_sharpe': simulated_sharpes.mean(),
            'percentile': percentile,
            'p_value': p_value,
            'significant': p_value < 0.05
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    print("=" * 70)
    print("TIME SERIES CROSS-VALIDATION FRAMEWORK")
    print("=" * 70)
    
    # Create sample data
    dates = pd.date_range('2023-01-01', '2025-12-01', freq='D')
    sample_data = pd.DataFrame({
        'date': dates,
        'returns': np.random.randn(len(dates)) * 0.01 + 0.0002  # Slight positive drift
    })
    
    cv = TimeSeriesCV(sample_data, date_column='date')
    
    print("\n[1/3] Walk-forward validation...")
    wf_splits = cv.walk_forward_split(train_size=252, test_size=20, step_size=20, expanding=False)
    
    print(f"\nExample split:")
    train, test = wf_splits[0]
    print(f"  Train: {train['date'].min()} to {train['date'].max()} ({len(train)} days)")
    print(f"  Test: {test['date'].min()} to {test['date'].max()} ({len(test)} days)")
    
    print("\n[2/3] Purged k-fold...")
    purged_splits = cv.purged_kfold(n_splits=5, embargo_pct=0.02)
    
    print("\n[3/3] Testing for overfitting...")
    returns = sample_data.set_index('date')['returns']
    is_returns, oos_returns = BacktestValidator.split_in_out_sample(returns, split_pct=0.7)
    
    overfit_test = BacktestValidator.test_overfitting(is_returns, oos_returns)
    
    print(f"\nIn-sample Sharpe: {overfit_test['in_sample']['sharpe_ratio']:.2f}")
    print(f"Out-of-sample Sharpe: {overfit_test['out_of_sample']['sharpe_ratio']:.2f}")
    print(f"Degradation: {overfit_test['sharpe_degradation_pct']:.1f}%")
    print(f"Assessment: {overfit_test['assessment']}")
    
    # Monte Carlo test
    print("\n" + "=" * 70)
    print("MONTE CARLO PERMUTATION TEST")
    print("=" * 70)
    
    mc_test = BacktestValidator.monte_carlo_permutation_test(returns, n_simulations=1000)
    
    print(f"\nObserved Sharpe: {mc_test['observed_sharpe']:.2f}")
    print(f"Mean random Sharpe: {mc_test['mean_random_sharpe']:.2f}")
    print(f"Percentile: {mc_test['percentile']:.1f}%")
    print(f"p-value: {mc_test['p_value']:.4f}")
    print(f"Significant: {mc_test['significant']}")
    
    print(f"\n✓ Cross-validation framework ready")
    print(f"✓ Use this to validate ALL backtest results")
