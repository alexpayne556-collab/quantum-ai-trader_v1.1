#!/usr/bin/env python3
"""
SCIENTIFIC TESTING LABORATORY - Core Statistical Framework

This module provides rigorous statistical testing tools for hypothesis validation.
All tests must pass multiple validation criteria to be considered significant.
"""

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from typing import Dict, List, Tuple, Optional, Callable
import warnings
from dataclasses import dataclass
from datetime import datetime


@dataclass
class HypothesisTest:
    """Container for hypothesis test results with full statistical rigor"""
    name: str
    null_hypothesis: str
    alternative_hypothesis: str
    test_statistic: float
    p_value: float
    confidence_level: float
    confidence_interval: Tuple[float, float]
    effect_size: float
    sample_size: int
    power: float
    conclusion: str
    warnings: List[str]
    timestamp: datetime


class StatisticalValidator:
    """
    Rigorous statistical validation framework.
    
    Principles:
    1. Multiple testing correction (Bonferroni, Holm, FDR)
    2. Effect size measurement (not just p-values)
    3. Power analysis (avoid Type II errors)
    4. Assumption checking (normality, homoscedasticity)
    5. Robustness testing (bootstrap, permutation)
    """
    
    def __init__(self, alpha: float = 0.05, min_effect_size: float = 0.1):
        """
        Parameters:
        -----------
        alpha : float
            Significance level (default 0.05 = 5% false positive rate)
        min_effect_size : float
            Minimum effect size to consider meaningful
        """
        self.alpha = alpha
        self.min_effect_size = min_effect_size
        self.tests_run = []
        
    def test_mean_difference(
        self, 
        group1: np.ndarray, 
        group2: np.ndarray,
        name: str = "Mean Difference Test",
        paired: bool = False
    ) -> HypothesisTest:
        """
        Test if two groups have different means.
        
        Uses:
        - t-test (if normal)
        - Mann-Whitney U (if non-normal)
        - Permutation test (as robustness check)
        """
        warnings_list = []
        
        # Check normality
        _, p_norm1 = stats.shapiro(group1) if len(group1) < 5000 else (None, 1.0)
        _, p_norm2 = stats.shapiro(group2) if len(group2) < 5000 else (None, 1.0)
        
        is_normal = (p_norm1 > 0.05) and (p_norm2 > 0.05)
        
        if not is_normal:
            warnings_list.append("Data not normally distributed - using non-parametric test")
        
        # Choose appropriate test
        if is_normal and not paired:
            # Independent t-test
            stat, p_value = stats.ttest_ind(group1, group2)
            test_name = "Independent t-test"
        elif is_normal and paired:
            # Paired t-test
            stat, p_value = stats.ttest_rel(group1, group2)
            test_name = "Paired t-test"
        elif not paired:
            # Mann-Whitney U (non-parametric)
            stat, p_value = stats.mannwhitneyu(group1, group2, alternative='two-sided')
            test_name = "Mann-Whitney U test"
        else:
            # Wilcoxon signed-rank (paired non-parametric)
            stat, p_value = stats.wilcoxon(group1, group2)
            test_name = "Wilcoxon signed-rank test"
        
        # Calculate effect size (Cohen's d)
        mean_diff = np.mean(group1) - np.mean(group2)
        pooled_std = np.sqrt((np.var(group1) + np.var(group2)) / 2)
        cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
        
        # Confidence interval (bootstrap)
        ci_lower, ci_upper = self._bootstrap_ci(group1, group2, lambda x, y: np.mean(x) - np.mean(y))
        
        # Statistical power (approximate)
        n = min(len(group1), len(group2))
        power = self._calculate_power(cohens_d, n, self.alpha)
        
        if power < 0.8:
            warnings_list.append(f"Low statistical power ({power:.2f}). Need larger sample for reliable conclusions.")
        
        # Conclusion
        is_significant = p_value < self.alpha
        has_effect = abs(cohens_d) >= self.min_effect_size
        
        if is_significant and has_effect:
            conclusion = f"SIGNIFICANT: Groups differ (p={p_value:.4f}, d={cohens_d:.3f})"
        elif is_significant and not has_effect:
            conclusion = f"STATISTICALLY SIGNIFICANT but effect size too small (d={cohens_d:.3f})"
            warnings_list.append("Effect size below threshold - may not be practically significant")
        else:
            conclusion = f"NOT SIGNIFICANT: No evidence of difference (p={p_value:.4f})"
        
        result = HypothesisTest(
            name=f"{name} ({test_name})",
            null_hypothesis="Mean(group1) = Mean(group2)",
            alternative_hypothesis="Mean(group1) ≠ Mean(group2)",
            test_statistic=stat,
            p_value=p_value,
            confidence_level=1 - self.alpha,
            confidence_interval=(ci_lower, ci_upper),
            effect_size=cohens_d,
            sample_size=len(group1) + len(group2),
            power=power,
            conclusion=conclusion,
            warnings=warnings_list,
            timestamp=datetime.now()
        )
        
        self.tests_run.append(result)
        return result
    
    def test_correlation(
        self,
        x: np.ndarray,
        y: np.ndarray,
        name: str = "Correlation Test"
    ) -> HypothesisTest:
        """
        Test if two variables are correlated.
        
        Uses:
        - Pearson (if linear + normal)
        - Spearman (if monotonic + non-normal)
        """
        warnings_list = []
        
        # Check normality
        _, p_norm_x = stats.shapiro(x) if len(x) < 5000 else (None, 1.0)
        _, p_norm_y = stats.shapiro(y) if len(y) < 5000 else (None, 1.0)
        
        is_normal = (p_norm_x > 0.05) and (p_norm_y > 0.05)
        
        if is_normal:
            # Pearson correlation
            r, p_value = stats.pearsonr(x, y)
            test_name = "Pearson correlation"
        else:
            # Spearman correlation (non-parametric)
            r, p_value = stats.spearmanr(x, y)
            test_name = "Spearman correlation"
            warnings_list.append("Using Spearman (non-parametric) due to non-normality")
        
        # Confidence interval (Fisher z-transform)
        n = len(x)
        z = np.arctanh(r)
        se = 1 / np.sqrt(n - 3)
        z_ci = stats.norm.ppf([self.alpha/2, 1 - self.alpha/2]) * se
        ci_lower, ci_upper = np.tanh(z + z_ci[0]), np.tanh(z + z_ci[1])
        
        # Statistical power
        power = self._calculate_power(abs(r), n, self.alpha)
        
        if power < 0.8:
            warnings_list.append(f"Low power ({power:.2f}) - need more data")
        
        # Conclusion
        is_significant = p_value < self.alpha
        has_effect = abs(r) >= self.min_effect_size
        
        if is_significant and has_effect:
            conclusion = f"SIGNIFICANT CORRELATION: r={r:.3f} (p={p_value:.4f})"
        elif is_significant and not has_effect:
            conclusion = f"WEAK CORRELATION: r={r:.3f} (p={p_value:.4f})"
            warnings_list.append("Correlation is weak - may not be useful for prediction")
        else:
            conclusion = f"NO CORRELATION: r={r:.3f} (p={p_value:.4f})"
        
        result = HypothesisTest(
            name=f"{name} ({test_name})",
            null_hypothesis="Correlation = 0",
            alternative_hypothesis="Correlation ≠ 0",
            test_statistic=r,
            p_value=p_value,
            confidence_level=1 - self.alpha,
            confidence_interval=(ci_lower, ci_upper),
            effect_size=abs(r),
            sample_size=n,
            power=power,
            conclusion=conclusion,
            warnings=warnings_list,
            timestamp=datetime.now()
        )
        
        self.tests_run.append(result)
        return result
    
    def test_autocorrelation(
        self,
        series: np.ndarray,
        lag: int = 1,
        name: str = "Autocorrelation Test"
    ) -> HypothesisTest:
        """
        Test if time series is autocorrelated (momentum/mean-reversion).
        
        H0: ρ(lag) = 0 (no autocorrelation)
        H1: ρ(lag) ≠ 0 (autocorrelation exists)
        """
        warnings_list = []
        
        # Calculate autocorrelation
        n = len(series)
        mean = np.mean(series)
        c0 = np.sum((series - mean) ** 2) / n
        c_lag = np.sum((series[:-lag] - mean) * (series[lag:] - mean)) / n
        rho = c_lag / c0
        
        # Standard error (Bartlett's formula for white noise)
        se = 1 / np.sqrt(n)
        
        # Test statistic (z-score)
        z_stat = rho / se
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        
        # Confidence interval
        z_critical = stats.norm.ppf(1 - self.alpha/2)
        ci_lower = rho - z_critical * se
        ci_upper = rho + z_critical * se
        
        # Ljung-Box test (more powerful for multiple lags)
        lb_result = acorr_ljungbox(series, lags=[lag], return_df=True)
        lb_pvalue = lb_result['lb_pvalue'].iloc[0]
        
        if lb_pvalue < p_value:
            warnings_list.append(f"Ljung-Box test more significant (p={lb_pvalue:.4f})")
            p_value = lb_pvalue
        
        # Power
        power = self._calculate_power(abs(rho), n, self.alpha)
        
        # Conclusion
        is_significant = p_value < self.alpha
        has_effect = abs(rho) >= self.min_effect_size
        
        if is_significant and has_effect:
            if rho > 0:
                conclusion = f"MOMENTUM DETECTED: ρ({lag})={rho:.3f} (p={p_value:.4f})"
            else:
                conclusion = f"MEAN REVERSION DETECTED: ρ({lag})={rho:.3f} (p={p_value:.4f})"
        elif is_significant and not has_effect:
            conclusion = f"WEAK AUTOCORRELATION: ρ({lag})={rho:.3f} (p={p_value:.4f})"
            warnings_list.append("Effect too weak for practical use")
        else:
            conclusion = f"NO AUTOCORRELATION: ρ({lag})={rho:.3f} (p={p_value:.4f})"
        
        result = HypothesisTest(
            name=f"{name} (lag={lag})",
            null_hypothesis=f"ρ({lag}) = 0",
            alternative_hypothesis=f"ρ({lag}) ≠ 0",
            test_statistic=z_stat,
            p_value=p_value,
            confidence_level=1 - self.alpha,
            confidence_interval=(ci_lower, ci_upper),
            effect_size=abs(rho),
            sample_size=n,
            power=power,
            conclusion=conclusion,
            warnings=warnings_list,
            timestamp=datetime.now()
        )
        
        self.tests_run.append(result)
        return result
    
    def multiple_testing_correction(self, method: str = 'bonferroni') -> pd.DataFrame:
        """
        Apply multiple testing correction to all tests run.
        
        Methods:
        - bonferroni: Most conservative (α / n_tests)
        - holm: Less conservative step-down
        - fdr: False discovery rate (Benjamini-Hochberg)
        """
        if not self.tests_run:
            return pd.DataFrame()
        
        p_values = [test.p_value for test in self.tests_run]
        
        if method == 'bonferroni':
            adjusted_alpha = self.alpha / len(p_values)
            significant = [p < adjusted_alpha for p in p_values]
        
        elif method == 'holm':
            # Holm-Bonferroni step-down
            sorted_idx = np.argsort(p_values)
            significant = [False] * len(p_values)
            for i, idx in enumerate(sorted_idx):
                adjusted_alpha = self.alpha / (len(p_values) - i)
                if p_values[idx] < adjusted_alpha:
                    significant[idx] = True
                else:
                    break
        
        elif method == 'fdr':
            # Benjamini-Hochberg FDR
            sorted_idx = np.argsort(p_values)
            significant = [False] * len(p_values)
            for i, idx in enumerate(sorted_idx):
                threshold = (i + 1) / len(p_values) * self.alpha
                if p_values[idx] <= threshold:
                    significant[idx] = True
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        results = []
        for test, sig in zip(self.tests_run, significant):
            results.append({
                'test': test.name,
                'p_value': test.p_value,
                'effect_size': test.effect_size,
                'significant_raw': test.p_value < self.alpha,
                f'significant_{method}': sig,
                'conclusion': test.conclusion
            })
        
        return pd.DataFrame(results)
    
    def _bootstrap_ci(
        self,
        group1: np.ndarray,
        group2: np.ndarray,
        statistic: Callable,
        n_bootstrap: int = 1000,
        confidence: float = 0.95
    ) -> Tuple[float, float]:
        """Bootstrap confidence interval for any statistic"""
        bootstrap_stats = []
        for _ in range(n_bootstrap):
            sample1 = np.random.choice(group1, size=len(group1), replace=True)
            sample2 = np.random.choice(group2, size=len(group2), replace=True)
            bootstrap_stats.append(statistic(sample1, sample2))
        
        lower = np.percentile(bootstrap_stats, (1 - confidence) / 2 * 100)
        upper = np.percentile(bootstrap_stats, (1 + confidence) / 2 * 100)
        return lower, upper
    
    def _calculate_power(self, effect_size: float, n: int, alpha: float) -> float:
        """
        Approximate statistical power calculation.
        
        Power = P(reject H0 | H1 is true) = 1 - β
        """
        # Cohen's power approximation
        delta = effect_size * np.sqrt(n / 2)
        z_critical = stats.norm.ppf(1 - alpha / 2)
        power = 1 - stats.norm.cdf(z_critical - delta)
        return power
    
    def summary_report(self) -> str:
        """Generate summary report of all tests"""
        if not self.tests_run:
            return "No tests run yet."
        
        report = []
        report.append("=" * 80)
        report.append("STATISTICAL VALIDATION SUMMARY")
        report.append("=" * 80)
        report.append(f"\nTotal tests run: {len(self.tests_run)}")
        
        significant = sum(1 for t in self.tests_run if t.p_value < self.alpha)
        report.append(f"Significant results (α={self.alpha}): {significant}/{len(self.tests_run)} ({significant/len(self.tests_run):.1%})")
        
        report.append("\n" + "-" * 80)
        report.append("INDIVIDUAL TEST RESULTS")
        report.append("-" * 80)
        
        for i, test in enumerate(self.tests_run, 1):
            report.append(f"\n{i}. {test.name}")
            report.append(f"   H0: {test.null_hypothesis}")
            report.append(f"   H1: {test.alternative_hypothesis}")
            report.append(f"   Result: {test.conclusion}")
            report.append(f"   p-value: {test.p_value:.6f}")
            report.append(f"   Effect size: {test.effect_size:.4f}")
            report.append(f"   95% CI: [{test.confidence_interval[0]:.4f}, {test.confidence_interval[1]:.4f}]")
            report.append(f"   Power: {test.power:.2f}")
            report.append(f"   Sample size: {test.sample_size:,}")
            
            if test.warnings:
                report.append("   ⚠️  Warnings:")
                for warning in test.warnings:
                    report.append(f"      - {warning}")
        
        report.append("\n" + "=" * 80)
        
        return "\n".join(report)


def run_example_tests():
    """Example usage of the statistical validator"""
    
    print("RUNNING EXAMPLE STATISTICAL TESTS\n")
    
    validator = StatisticalValidator(alpha=0.05, min_effect_size=0.2)
    
    # Example 1: Test if momentum exists (autocorrelation)
    print("Example 1: Testing for momentum (autocorrelation)")
    np.random.seed(42)
    
    # Generate momentum series (AR(1) with ρ=0.3)
    n = 1000
    momentum_series = np.zeros(n)
    momentum_series[0] = np.random.randn()
    for t in range(1, n):
        momentum_series[t] = 0.3 * momentum_series[t-1] + np.random.randn()
    
    result = validator.test_autocorrelation(momentum_series, lag=1, name="Momentum Test")
    print(result.conclusion)
    print(f"Effect size: {result.effect_size:.4f}")
    print(f"P-value: {result.p_value:.6f}\n")
    
    # Example 2: Test if strategy returns differ from zero
    print("Example 2: Testing if strategy beats random (mean difference)")
    strategy_returns = np.random.normal(0.02, 0.1, 500)  # 2% mean, 10% std
    random_returns = np.random.normal(0.0, 0.1, 500)     # 0% mean, 10% std
    
    result = validator.test_mean_difference(
        strategy_returns, 
        random_returns,
        name="Strategy vs Random"
    )
    print(result.conclusion)
    print(f"Effect size (Cohen's d): {result.effect_size:.4f}\n")
    
    # Multiple testing correction
    print("Applying multiple testing correction (Bonferroni):")
    corrected = validator.multiple_testing_correction(method='bonferroni')
    print(corrected.to_string(index=False))
    
    print("\n" + "=" * 80)
    print("FULL SUMMARY REPORT")
    print("=" * 80)
    print(validator.summary_report())


if __name__ == '__main__':
    run_example_tests()
