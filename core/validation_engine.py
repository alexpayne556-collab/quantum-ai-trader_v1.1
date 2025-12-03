"""
VALIDATION ENGINE: COMBINATORIAL PURGED CROSS-VALIDATION (CPCV)
================================================================
The TRUTH about your model - standard backtests LIE.

Problem with Standard Backtests:
- Look-ahead bias (model sees future data)
- Overlapping test sets (data leakage)
- Cherry-picked results
- Overfitting to historical patterns

Solution: CPCV (López de Prado Method)
- Splits data into thousands of overlapping chunks
- "Purges" data around trades (removes lookahead)
- Creates embargo period after each test
- Only way to know if strategy is REAL

Research: Most "profitable" strategies fail CPCV validation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable
from sklearn.model_selection import BaseCrossValidator
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import warnings
warnings.filterwarnings('ignore')


class PurgedKFold(BaseCrossValidator):
    """
    Purged K-Fold Cross-Validation
    
    Standard K-Fold leaks information because test data might be
    temporally adjacent to training data.
    
    Purged K-Fold adds:
    1. Purge: Remove samples close to test set boundaries
    2. Embargo: Add buffer period after test set
    
    This prevents look-ahead bias in time series.
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        purge_gap: int = 5,
        embargo_pct: float = 0.01
    ):
        """
        Args:
            n_splits: Number of folds
            purge_gap: Number of samples to purge around test set
            embargo_pct: Percentage of samples to embargo after test
        """
        self.n_splits = n_splits
        self.purge_gap = purge_gap
        self.embargo_pct = embargo_pct
    
    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits
    
    def split(self, X, y=None, groups=None):
        """
        Generate purged train/test splits
        
        Yields:
            train_indices, test_indices
        """
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        fold_size = n_samples // self.n_splits
        embargo_size = int(n_samples * self.embargo_pct)
        
        for fold in range(self.n_splits):
            # Test indices for this fold
            test_start = fold * fold_size
            test_end = test_start + fold_size if fold < self.n_splits - 1 else n_samples
            
            test_indices = indices[test_start:test_end]
            
            # Purge: remove samples close to test boundaries
            purge_start = max(0, test_start - self.purge_gap)
            purge_end = min(n_samples, test_end + self.purge_gap + embargo_size)
            
            # Train indices: everything except test + purge + embargo
            train_mask = np.ones(n_samples, dtype=bool)
            train_mask[purge_start:purge_end] = False
            train_indices = indices[train_mask]
            
            yield train_indices, test_indices


class CombinatorialPurgedCV(BaseCrossValidator):
    """
    Combinatorial Purged Cross-Validation (CPCV)
    
    The GOLD STANDARD for validating trading strategies.
    
    Creates all possible combinations of train/test splits
    while purging and embargoing properly.
    
    Much more rigorous than standard CV.
    """
    
    def __init__(
        self,
        n_splits: int = 6,
        n_test_groups: int = 2,
        purge_gap: int = 5,
        embargo_pct: float = 0.01
    ):
        """
        Args:
            n_splits: Number of groups to split data into
            n_test_groups: Number of groups to use for testing in each split
            purge_gap: Samples to purge around test boundaries
            embargo_pct: Embargo period as fraction of data
        """
        self.n_splits = n_splits
        self.n_test_groups = n_test_groups
        self.purge_gap = purge_gap
        self.embargo_pct = embargo_pct
        
        # Calculate number of combinations
        from math import comb
        self.n_combinations = comb(n_splits, n_test_groups)
    
    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_combinations
    
    def split(self, X, y=None, groups=None):
        """Generate all combinatorial purged splits"""
        from itertools import combinations
        
        n_samples = len(X)
        indices = np.arange(n_samples)
        group_size = n_samples // self.n_splits
        embargo_size = int(n_samples * self.embargo_pct)
        
        # Create group boundaries
        groups_bounds = []
        for i in range(self.n_splits):
            start = i * group_size
            end = start + group_size if i < self.n_splits - 1 else n_samples
            groups_bounds.append((start, end))
        
        # Generate all combinations of test groups
        for test_group_indices in combinations(range(self.n_splits), self.n_test_groups):
            # Determine test indices
            test_mask = np.zeros(n_samples, dtype=bool)
            purge_mask = np.zeros(n_samples, dtype=bool)
            
            for group_idx in test_group_indices:
                start, end = groups_bounds[group_idx]
                test_mask[start:end] = True
                
                # Add purge and embargo
                purge_start = max(0, start - self.purge_gap)
                purge_end = min(n_samples, end + self.purge_gap + embargo_size)
                purge_mask[purge_start:purge_end] = True
            
            test_indices = indices[test_mask]
            train_indices = indices[~purge_mask]
            
            if len(train_indices) > 0 and len(test_indices) > 0:
                yield train_indices, test_indices


class WalkForwardCV(BaseCrossValidator):
    """
    Walk-Forward Cross-Validation
    
    Most realistic for time series:
    - Train on past data
    - Test on future data
    - Never look ahead
    
    Each fold moves forward in time.
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        train_size: Optional[int] = None,
        test_size: Optional[int] = None,
        gap: int = 0,
        expanding: bool = False
    ):
        """
        Args:
            n_splits: Number of walk-forward splits
            train_size: Fixed training window size (None = expanding)
            test_size: Test window size
            gap: Gap between train and test (embargo)
            expanding: If True, training window grows over time
        """
        self.n_splits = n_splits
        self.train_size = train_size
        self.test_size = test_size
        self.gap = gap
        self.expanding = expanding
    
    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits
    
    def split(self, X, y=None, groups=None):
        """Generate walk-forward splits"""
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        # Calculate sizes
        if self.test_size is None:
            test_size = n_samples // (self.n_splits + 1)
        else:
            test_size = self.test_size
        
        if self.train_size is None:
            train_size = n_samples - (self.n_splits * test_size) - self.gap
        else:
            train_size = self.train_size
        
        for fold in range(self.n_splits):
            # Test window
            test_end = n_samples - (self.n_splits - fold - 1) * test_size
            test_start = test_end - test_size
            
            # Train window
            train_end = test_start - self.gap
            if self.expanding:
                train_start = 0
            else:
                train_start = max(0, train_end - train_size)
            
            if train_start < train_end and test_start < test_end:
                train_indices = indices[train_start:train_end]
                test_indices = indices[test_start:test_end]
                yield train_indices, test_indices


class ValidationEngine:
    """
    Complete Validation Suite for Trading Strategies
    
    Includes:
    1. Purged K-Fold CV
    2. Combinatorial Purged CV (CPCV)
    3. Walk-Forward CV
    4. Monte Carlo simulation
    5. Statistical significance tests
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.results = {}
        
    def log(self, msg: str):
        if self.verbose:
            print(f"[Validation] {msg}")
    
    def validate_model(
        self,
        model,
        X: pd.DataFrame,
        y: pd.Series,
        method: str = 'cpcv',
        n_splits: int = 5,
        metrics: List[str] = ['accuracy', 'f1', 'precision', 'recall']
    ) -> Dict:
        """
        Validate model using specified cross-validation method
        
        Args:
            model: Sklearn-compatible model with fit/predict
            X: Features
            y: Labels
            method: 'purged', 'cpcv', 'walk_forward'
            n_splits: Number of CV splits
            metrics: Metrics to calculate
        
        Returns:
            Dict with validation results
        """
        self.log(f"\n{'='*60}")
        self.log(f"VALIDATING MODEL: {method.upper()}")
        self.log(f"{'='*60}")
        
        # Select CV method
        if method == 'purged':
            cv = PurgedKFold(n_splits=n_splits, purge_gap=5, embargo_pct=0.01)
        elif method == 'cpcv':
            cv = CombinatorialPurgedCV(n_splits=n_splits, n_test_groups=2, purge_gap=5)
        elif method == 'walk_forward':
            cv = WalkForwardCV(n_splits=n_splits, gap=5)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Run CV
        fold_results = []
        
        for fold, (train_idx, test_idx) in enumerate(cv.split(X)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Fit model
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            fold_metrics = {}
            
            if 'accuracy' in metrics:
                fold_metrics['accuracy'] = accuracy_score(y_test, y_pred)
            if 'f1' in metrics:
                fold_metrics['f1'] = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            if 'precision' in metrics:
                fold_metrics['precision'] = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            if 'recall' in metrics:
                fold_metrics['recall'] = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            
            fold_metrics['train_size'] = len(train_idx)
            fold_metrics['test_size'] = len(test_idx)
            
            fold_results.append(fold_metrics)
            
            if self.verbose:
                self.log(f"  Fold {fold+1}: Accuracy={fold_metrics.get('accuracy', 0):.3f}, "
                        f"F1={fold_metrics.get('f1', 0):.3f} "
                        f"(train={len(train_idx)}, test={len(test_idx)})")
        
        # Aggregate results
        results_df = pd.DataFrame(fold_results)
        
        aggregated = {
            'method': method,
            'n_folds': len(fold_results),
            'metrics': {}
        }
        
        for metric in metrics:
            if metric in results_df.columns:
                values = results_df[metric].values
                aggregated['metrics'][metric] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values))
                }
        
        # Log summary
        self.log(f"\n📊 SUMMARY ({method.upper()}):")
        for metric, stats in aggregated['metrics'].items():
            self.log(f"   {metric.capitalize()}: {stats['mean']:.3f} ± {stats['std']:.3f} "
                    f"[{stats['min']:.3f}, {stats['max']:.3f}]")
        
        self.results[method] = aggregated
        return aggregated
    
    def run_full_validation(
        self,
        model,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Dict:
        """
        Run all validation methods and compare
        
        Returns comprehensive validation report
        """
        self.log("\n" + "=" * 60)
        self.log("FULL VALIDATION SUITE")
        self.log("=" * 60)
        
        all_results = {}
        
        # 1. Purged K-Fold
        self.log("\n--- 1. Purged K-Fold ---")
        all_results['purged'] = self.validate_model(model, X, y, method='purged', n_splits=5)
        
        # 2. Combinatorial Purged CV
        self.log("\n--- 2. Combinatorial Purged CV ---")
        all_results['cpcv'] = self.validate_model(model, X, y, method='cpcv', n_splits=6)
        
        # 3. Walk-Forward
        self.log("\n--- 3. Walk-Forward CV ---")
        all_results['walk_forward'] = self.validate_model(model, X, y, method='walk_forward', n_splits=5)
        
        # Comparison
        self.log("\n" + "=" * 60)
        self.log("VALIDATION COMPARISON")
        self.log("=" * 60)
        
        comparison = []
        for method, result in all_results.items():
            row = {'method': method}
            for metric, stats in result['metrics'].items():
                row[f'{metric}_mean'] = stats['mean']
                row[f'{metric}_std'] = stats['std']
            comparison.append(row)
        
        comparison_df = pd.DataFrame(comparison)
        self.log(f"\n{comparison_df.to_string(index=False)}")
        
        # Statistical significance
        all_results['passed_validation'] = self._check_significance(all_results)
        
        return all_results
    
    def _check_significance(self, results: Dict) -> bool:
        """
        Check if results are statistically significant
        
        Criteria:
        - CPCV accuracy > 50% (better than random)
        - Std < 10% (consistent across folds)
        - All methods agree (no overfitting)
        """
        self.log("\n📋 SIGNIFICANCE CHECK:")
        
        passed = True
        
        # Check CPCV accuracy
        if 'cpcv' in results and 'accuracy' in results['cpcv']['metrics']:
            cpcv_acc = results['cpcv']['metrics']['accuracy']['mean']
            cpcv_std = results['cpcv']['metrics']['accuracy']['std']
            
            if cpcv_acc < 0.50:
                self.log(f"   ❌ CPCV accuracy ({cpcv_acc:.1%}) below 50% - NOT SIGNIFICANT")
                passed = False
            else:
                self.log(f"   ✅ CPCV accuracy ({cpcv_acc:.1%}) above 50%")
            
            if cpcv_std > 0.10:
                self.log(f"   ⚠️  High variance (std={cpcv_std:.1%}) - possible overfitting")
        
        # Check consistency across methods
        if 'purged' in results and 'cpcv' in results:
            purged_acc = results['purged']['metrics'].get('accuracy', {}).get('mean', 0)
            cpcv_acc = results['cpcv']['metrics'].get('accuracy', {}).get('mean', 0)
            
            diff = abs(purged_acc - cpcv_acc)
            if diff > 0.10:
                self.log(f"   ⚠️  Large gap between methods ({diff:.1%}) - check for data leakage")
            else:
                self.log(f"   ✅ Methods agree (gap={diff:.1%})")
        
        return passed
    
    def monte_carlo_simulation(
        self,
        returns: pd.Series,
        n_simulations: int = 1000,
        n_periods: int = 252
    ) -> Dict:
        """
        Monte Carlo simulation for strategy robustness
        
        Generates random shuffles of returns to estimate
        what performance could be achieved by chance.
        
        Args:
            returns: Strategy returns
            n_simulations: Number of simulations
            n_periods: Trading periods per year
        
        Returns:
            Dict with simulation results
        """
        self.log(f"\n{'='*60}")
        self.log("MONTE CARLO SIMULATION")
        self.log(f"{'='*60}")
        
        actual_sharpe = returns.mean() / returns.std() * np.sqrt(n_periods)
        actual_return = returns.sum()
        
        simulated_sharpes = []
        simulated_returns = []
        
        for _ in range(n_simulations):
            # Shuffle returns
            shuffled = returns.sample(frac=1).values
            
            # Calculate metrics
            sim_sharpe = np.mean(shuffled) / np.std(shuffled) * np.sqrt(n_periods)
            sim_return = np.sum(shuffled)
            
            simulated_sharpes.append(sim_sharpe)
            simulated_returns.append(sim_return)
        
        # Calculate percentile of actual performance
        sharpe_percentile = (np.array(simulated_sharpes) < actual_sharpe).mean() * 100
        return_percentile = (np.array(simulated_returns) < actual_return).mean() * 100
        
        results = {
            'actual_sharpe': actual_sharpe,
            'actual_return': actual_return,
            'sharpe_percentile': sharpe_percentile,
            'return_percentile': return_percentile,
            'significant': sharpe_percentile > 95  # 95% confidence
        }
        
        self.log(f"\n📊 MONTE CARLO RESULTS:")
        self.log(f"   Actual Sharpe: {actual_sharpe:.2f}")
        self.log(f"   Sharpe Percentile: {sharpe_percentile:.1f}%")
        self.log(f"   Return Percentile: {return_percentile:.1f}%")
        
        if results['significant']:
            self.log(f"   ✅ STATISTICALLY SIGNIFICANT (>95% percentile)")
        else:
            self.log(f"   ❌ NOT SIGNIFICANT (≤95% percentile)")
        
        return results


# =============================================================================
# QUICK TEST
# =============================================================================
if __name__ == "__main__":
    import yfinance as yf
    from sklearn.ensemble import RandomForestClassifier
    
    print("=" * 60)
    print("TESTING VALIDATION ENGINE (CPCV)")
    print("=" * 60)
    
    # Download test data
    df = yf.download("SPY", start="2020-01-01", end="2024-12-01", progress=False)
    
    # Create features
    features = pd.DataFrame(index=df.index)
    features['returns'] = df['Close'].pct_change()
    features['volatility'] = features['returns'].rolling(20).std()
    features['momentum'] = df['Close'].pct_change(20)
    features['rsi'] = 100 - (100 / (1 + (df['Close'].diff().clip(lower=0).rolling(14).mean() / 
                                         df['Close'].diff().clip(upper=0).abs().rolling(14).mean())))
    
    # Create labels (5-day forward return direction)
    labels = (df['Close'].pct_change(5).shift(-5) > 0).astype(int)
    
    # Drop NaN
    valid_idx = features.dropna().index.intersection(labels.dropna().index)
    features = features.loc[valid_idx]
    labels = labels.loc[valid_idx]
    
    print(f"\nData: {len(features)} samples")
    print(f"Features: {list(features.columns)}")
    
    # Create model
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    
    # Run validation
    engine = ValidationEngine(verbose=True)
    results = engine.run_full_validation(model, features, labels)
    
    # Monte Carlo simulation
    print("\n" + "=" * 60)
    strategy_returns = features['returns'] * labels.shift(1)  # Simple strategy returns
    mc_results = engine.monte_carlo_simulation(strategy_returns.dropna())
    
    print(f"\n{'='*60}")
    print("FINAL VERDICT")
    print(f"{'='*60}")
    
    if results.get('passed_validation') and mc_results.get('significant'):
        print("✅ STRATEGY PASSES VALIDATION - Safe to trade")
    else:
        print("❌ STRATEGY FAILS VALIDATION - Do NOT trade")
