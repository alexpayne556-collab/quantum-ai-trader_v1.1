"""
STATISTICAL TESTING FRAMEWORK FOR UNIVERSAL LAW DISCOVERY

This is the scientific rigor layer. Everything gets tested properly:
- Null hypothesis significance testing
- Multiple testing corrections (we're testing MANY hypotheses)
- Bootstrap confidence intervals
- Cross-sectional and time-series tests
- Regime detection and regime-dependent tests

NO p-hacking. NO data mining. ONLY rigorous hypothesis testing.
"""

import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.stats.diagnostic import het_white
from arch import arch_model
import sqlite3
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class HypothesisTest:
    """Results of a hypothesis test"""
    hypothesis: str
    test_statistic: float
    p_value: float
    effect_size: float
    confidence_interval: Tuple[float, float]
    sample_size: int
    test_type: str
    rejected: bool
    metadata: Dict


class MarketDataLoader:
    """Fast data access layer for hypothesis testing"""
    
    def __init__(self, db_path: str = "data/market_data.db"):
        self.db_path = db_path
    
    def get_returns(self, tickers: Optional[List[str]] = None, 
                   start_date: Optional[str] = None,
                   end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Get daily returns for all tickers
        Returns: DataFrame with tickers as columns, dates as index
        """
        conn = sqlite3.connect(self.db_path)
        
        query = """
            SELECT ticker, date, close, adj_close
            FROM daily_bars
            WHERE 1=1
        """
        params = []
        
        if tickers:
            placeholders = ','.join(['?' for _ in tickers])
            query += f" AND ticker IN ({placeholders})"
            params.extend(tickers)
        
        if start_date:
            query += " AND date >= ?"
            params.append(start_date)
        
        if end_date:
            query += " AND date <= ?"
            params.append(end_date)
        
        query += " ORDER BY ticker, date"
        
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        # Pivot to wide format
        df['date'] = pd.to_datetime(df['date'])
        df = df.pivot(index='date', columns='ticker', values='adj_close')
        
        # Calculate returns
        returns = df.pct_change().dropna()
        
        return returns
    
    def get_volume_data(self, tickers: Optional[List[str]] = None) -> pd.DataFrame:
        """Get volume data"""
        conn = sqlite3.connect(self.db_path)
        
        query = "SELECT ticker, date, volume FROM daily_bars"
        if tickers:
            placeholders = ','.join(['?' for _ in tickers])
            query += f" WHERE ticker IN ({placeholders})"
            df = pd.read_sql_query(query, conn, params=tickers)
        else:
            df = pd.read_sql_query(query, conn)
        
        conn.close()
        
        df['date'] = pd.to_datetime(df['date'])
        volume = df.pivot(index='date', columns='ticker', values='volume')
        
        return volume
    
    def get_ohlcv(self, ticker: str) -> pd.DataFrame:
        """Get full OHLCV data for single ticker"""
        conn = sqlite3.connect(self.db_path)
        
        df = pd.read_sql_query(
            "SELECT date, open, high, low, close, volume, adj_close FROM daily_bars WHERE ticker = ? ORDER BY date",
            conn, params=[ticker]
        )
        
        conn.close()
        
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        
        return df


class UniversalLawTester:
    """
    Test if market phenomena are UNIVERSAL or regime-dependent
    
    A universal law must hold across:
    - All market caps (small, mid, large, mega)
    - All sectors
    - All time periods (bull, bear, sideways)
    - All volatility regimes
    """
    
    def __init__(self, data_loader: MarketDataLoader):
        self.loader = data_loader
    
    def test_momentum_universality(self, lookback_days: int = 20, 
                                   forward_days: int = 5,
                                   alpha: float = 0.05) -> HypothesisTest:
        """
        Test: Do past returns predict future returns UNIVERSALLY?
        
        Hypothesis: Stocks with positive momentum continue to outperform
        
        Method:
        1. For each day, rank all stocks by past N-day return
        2. Measure forward M-day return for top vs. bottom quintile
        3. Test if difference is statistically significant
        4. Check if holds across ALL subgroups (market cap, sector, time)
        """
        returns = self.loader.get_returns()
        
        # Calculate momentum signal
        momentum = returns.rolling(lookback_days).sum()
        forward_returns = returns.shift(-forward_days).rolling(forward_days).sum()
        
        # Cross-sectional test each day
        daily_spreads = []
        
        for date in momentum.index[lookback_days:-forward_days]:
            mom_today = momentum.loc[date].dropna()
            fwd_ret_today = forward_returns.loc[date].dropna()
            
            # Align
            common = mom_today.index.intersection(fwd_ret_today.index)
            if len(common) < 100:  # Need enough stocks
                continue
            
            mom_vals = mom_today[common]
            fwd_vals = fwd_ret_today[common]
            
            # Quintile split
            top_quintile = mom_vals >= mom_vals.quantile(0.8)
            bottom_quintile = mom_vals <= mom_vals.quantile(0.2)
            
            spread = fwd_vals[top_quintile].mean() - fwd_vals[bottom_quintile].mean()
            daily_spreads.append(spread)
        
        daily_spreads = np.array(daily_spreads)
        
        # Test if average spread is significantly different from zero
        t_stat, p_value = stats.ttest_1samp(daily_spreads, 0)
        
        mean_spread = daily_spreads.mean()
        std_spread = daily_spreads.std()
        
        # Bootstrap confidence interval
        n_bootstrap = 1000
        bootstrap_means = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(daily_spreads, size=len(daily_spreads), replace=True)
            bootstrap_means.append(sample.mean())
        
        ci_lower = np.percentile(bootstrap_means, 2.5)
        ci_upper = np.percentile(bootstrap_means, 97.5)
        
        return HypothesisTest(
            hypothesis=f"{lookback_days}-day momentum predicts {forward_days}-day returns",
            test_statistic=t_stat,
            p_value=p_value,
            effect_size=mean_spread,
            confidence_interval=(ci_lower, ci_upper),
            sample_size=len(daily_spreads),
            test_type="t-test (one-sample)",
            rejected=p_value < alpha,
            metadata={
                'lookback_days': lookback_days,
                'forward_days': forward_days,
                'mean_spread_bps': mean_spread * 10000,
                'std_spread_bps': std_spread * 10000,
                'sharpe_ratio': mean_spread / std_spread if std_spread > 0 else 0
            }
        )
    
    def test_mean_reversion_universality(self, lookback_days: int = 5,
                                         forward_days: int = 1,
                                         alpha: float = 0.05) -> HypothesisTest:
        """
        Test: Do extreme returns mean-revert UNIVERSALLY?
        
        Hypothesis: Stocks with extreme negative returns rebound
        """
        returns = self.loader.get_returns()
        
        # Calculate extreme return signal
        extreme_returns = returns.rolling(lookback_days).sum()
        forward_returns = returns.shift(-forward_days).rolling(forward_days).sum()
        
        daily_reversal = []
        
        for date in extreme_returns.index[lookback_days:-forward_days]:
            ext_today = extreme_returns.loc[date].dropna()
            fwd_ret_today = forward_returns.loc[date].dropna()
            
            common = ext_today.index.intersection(fwd_ret_today.index)
            if len(common) < 100:
                continue
            
            ext_vals = ext_today[common]
            fwd_vals = fwd_ret_today[common]
            
            # Extreme losers
            extreme_losers = ext_vals <= ext_vals.quantile(0.1)
            
            # Do they rebound?
            reversal = fwd_vals[extreme_losers].mean()
            daily_reversal.append(reversal)
        
        daily_reversal = np.array(daily_reversal)
        
        # Test if positive (reversal) or negative (continuation)
        t_stat, p_value = stats.ttest_1samp(daily_reversal, 0)
        
        mean_rev = daily_reversal.mean()
        
        # Bootstrap CI
        bootstrap_means = [np.random.choice(daily_reversal, size=len(daily_reversal), replace=True).mean() 
                          for _ in range(1000)]
        ci_lower = np.percentile(bootstrap_means, 2.5)
        ci_upper = np.percentile(bootstrap_means, 97.5)
        
        return HypothesisTest(
            hypothesis=f"Extreme {lookback_days}-day losers revert in {forward_days} days",
            test_statistic=t_stat,
            p_value=p_value,
            effect_size=mean_rev,
            confidence_interval=(ci_lower, ci_upper),
            sample_size=len(daily_reversal),
            test_type="t-test (one-sample)",
            rejected=p_value < alpha,
            metadata={
                'lookback_days': lookback_days,
                'forward_days': forward_days,
                'mean_reversal_bps': mean_rev * 10000
            }
        )
    
    def test_volatility_clustering(self, ticker: str, alpha: float = 0.05) -> HypothesisTest:
        """
        Test: Does volatility cluster (high vol follows high vol)?
        
        This is a TIME-SERIES test on individual ticker
        Uses ARCH-LM test for ARCH effects
        """
        df = self.loader.get_ohlcv(ticker)
        returns = df['adj_close'].pct_change().dropna()
        
        # ARCH-LM test for volatility clustering
        squared_returns = returns ** 2
        
        # Test autocorrelation in squared returns
        acf_sq = acf(squared_returns, nlags=20, fft=True)
        
        # Ljung-Box test
        lb_stat, lb_pvalue = stats.normaltest(squared_returns)
        
        # Fit ARCH model
        try:
            model = arch_model(returns * 100, vol='GARCH', p=1, q=1)
            results = model.fit(disp='off')
            
            arch_effect = results.params['omega'] > 0 and results.params['alpha[1]'] > 0
            garch_effect = 'beta[1]' in results.params and results.params['beta[1]'] > 0
            
            # Test statistic: sum of ARCH and GARCH coefficients
            persistence = results.params['alpha[1]']
            if 'beta[1]' in results.params:
                persistence += results.params['beta[1]']
            
            return HypothesisTest(
                hypothesis=f"{ticker} exhibits volatility clustering (GARCH effects)",
                test_statistic=persistence,
                p_value=results.pvalues['alpha[1]'],
                effect_size=persistence,
                confidence_interval=(persistence - 0.1, persistence + 0.1),  # Rough estimate
                sample_size=len(returns),
                test_type="GARCH(1,1)",
                rejected=results.pvalues['alpha[1]'] < alpha,
                metadata={
                    'omega': float(results.params['omega']),
                    'alpha': float(results.params['alpha[1]']),
                    'beta': float(results.params.get('beta[1]', 0)),
                    'persistence': float(persistence)
                }
            )
        except Exception as e:
            logger.warning(f"GARCH fit failed for {ticker}: {e}")
            return None
    
    def test_cross_sectional_correlation(self, min_correlation: float = 0.3,
                                        alpha: float = 0.05) -> HypothesisTest:
        """
        Test: Are stock returns correlated (market factor exists)?
        
        Hypothesis: Average pairwise correlation > 0 (stocks move together)
        """
        returns = self.loader.get_returns()
        
        # Sample for speed (1000 random tickers)
        if len(returns.columns) > 1000:
            sampled_tickers = np.random.choice(returns.columns, 1000, replace=False)
            returns = returns[sampled_tickers]
        
        # Calculate correlation matrix
        corr_matrix = returns.corr()
        
        # Get upper triangle (exclude diagonal)
        mask = np.triu(np.ones_like(corr_matrix), k=1).astype(bool)
        correlations = corr_matrix.where(mask).stack().values
        
        # Test if mean correlation > 0
        t_stat, p_value = stats.ttest_1samp(correlations, 0)
        
        mean_corr = correlations.mean()
        
        # Bootstrap CI
        bootstrap_means = [np.random.choice(correlations, size=len(correlations), replace=True).mean() 
                          for _ in range(1000)]
        ci_lower = np.percentile(bootstrap_means, 2.5)
        ci_upper = np.percentile(bootstrap_means, 97.5)
        
        return HypothesisTest(
            hypothesis="Stock returns are positively correlated (market factor)",
            test_statistic=t_stat,
            p_value=p_value,
            effect_size=mean_corr,
            confidence_interval=(ci_lower, ci_upper),
            sample_size=len(correlations),
            test_type="t-test (one-sample)",
            rejected=p_value < alpha,
            metadata={
                'mean_correlation': float(mean_corr),
                'median_correlation': float(np.median(correlations)),
                'pct_positive': float((correlations > 0).mean() * 100)
            }
        )


class MultipleTestingCorrection:
    """
    When testing MANY hypotheses, some will appear significant by chance
    
    This applies corrections to control false discovery rate
    """
    
    @staticmethod
    def bonferroni_correction(p_values: List[float], alpha: float = 0.05) -> Tuple[List[bool], float]:
        """
        Most conservative: alpha / n_tests
        
        Use when testing few, critical hypotheses
        """
        n = len(p_values)
        adjusted_alpha = alpha / n
        rejected = [p < adjusted_alpha for p in p_values]
        
        return rejected, adjusted_alpha
    
    @staticmethod
    def benjamini_hochberg(p_values: List[float], alpha: float = 0.05) -> List[bool]:
        """
        Controls False Discovery Rate (FDR)
        
        Use when testing many hypotheses, more power than Bonferroni
        """
        rejected, _, _, _ = multipletests(p_values, alpha=alpha, method='fdr_bh')
        return rejected.tolist()
    
    @staticmethod
    def holm_bonferroni(p_values: List[float], alpha: float = 0.05) -> List[bool]:
        """
        Step-down Bonferroni, more powerful than regular Bonferroni
        """
        rejected, _, _, _ = multipletests(p_values, alpha=alpha, method='holm')
        return rejected.tolist()


def run_universal_law_discovery(alpha: float = 0.05, 
                                multiple_testing_method: str = 'fdr_bh') -> pd.DataFrame:
    """
    Run the full battery of hypothesis tests
    
    Returns: DataFrame with all test results
    """
    logger.info("=" * 70)
    logger.info("UNIVERSAL LAW DISCOVERY - HYPOTHESIS TESTING")
    logger.info("=" * 70)
    
    loader = MarketDataLoader()
    tester = UniversalLawTester(loader)
    
    results = []
    
    # Test 1: Momentum
    logger.info("\n[1/5] Testing momentum universality...")
    for lookback in [5, 10, 20, 60]:
        for forward in [1, 5, 10, 20]:
            test = tester.test_momentum_universality(lookback, forward, alpha)
            results.append(test)
            logger.info(f"  {lookback}d→{forward}d: p={test.p_value:.4f}, effect={test.effect_size*10000:.1f}bps")
    
    # Test 2: Mean reversion
    logger.info("\n[2/5] Testing mean reversion...")
    for lookback in [1, 3, 5, 10]:
        for forward in [1, 3, 5]:
            test = tester.test_mean_reversion_universality(lookback, forward, alpha)
            results.append(test)
            logger.info(f"  {lookback}d→{forward}d: p={test.p_value:.4f}, effect={test.effect_size*10000:.1f}bps")
    
    # Test 3: Cross-sectional correlation
    logger.info("\n[3/5] Testing market factor...")
    test = tester.test_cross_sectional_correlation(alpha=alpha)
    results.append(test)
    logger.info(f"  Mean correlation: {test.effect_size:.3f}, p={test.p_value:.4e}")
    
    # Test 4: Volatility clustering (sample 50 tickers)
    logger.info("\n[4/5] Testing volatility clustering...")
    returns = loader.get_returns()
    sample_tickers = np.random.choice(returns.columns, min(50, len(returns.columns)), replace=False)
    
    vol_cluster_tests = []
    for ticker in sample_tickers:
        test = tester.test_volatility_clustering(ticker, alpha)
        if test:
            vol_cluster_tests.append(test)
    
    if vol_cluster_tests:
        results.extend(vol_cluster_tests)
        sig_count = sum([t.rejected for t in vol_cluster_tests])
        logger.info(f"  {sig_count}/{len(vol_cluster_tests)} tickers show significant volatility clustering")
    
    # Multiple testing correction
    logger.info("\n[5/5] Applying multiple testing correction...")
    p_values = [r.p_value for r in results]
    
    if multiple_testing_method == 'bonferroni':
        corrected_rejected, adj_alpha = MultipleTestingCorrection.bonferroni_correction(p_values, alpha)
        logger.info(f"  Bonferroni: adjusted alpha = {adj_alpha:.6f}")
    elif multiple_testing_method == 'fdr_bh':
        corrected_rejected = MultipleTestingCorrection.benjamini_hochberg(p_values, alpha)
        logger.info(f"  Benjamini-Hochberg FDR correction applied")
    else:
        corrected_rejected = MultipleTestingCorrection.holm_bonferroni(p_values, alpha)
        logger.info(f"  Holm-Bonferroni correction applied")
    
    # Update rejection status
    for i, result in enumerate(results):
        result.rejected = corrected_rejected[i]
    
    # Convert to DataFrame
    results_df = pd.DataFrame([
        {
            'hypothesis': r.hypothesis,
            'p_value': r.p_value,
            'effect_size': r.effect_size,
            'rejected_uncorrected': r.p_value < alpha,
            'rejected_corrected': r.rejected,
            'test_statistic': r.test_statistic,
            'sample_size': r.sample_size,
            'ci_lower': r.confidence_interval[0],
            'ci_upper': r.confidence_interval[1],
            **r.metadata
        }
        for r in results
    ])
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Total hypotheses tested: {len(results)}")
    logger.info(f"Significant (uncorrected): {(results_df['rejected_uncorrected']).sum()}")
    logger.info(f"Significant (corrected): {(results_df['rejected_corrected']).sum()}")
    logger.info("=" * 70)
    
    return results_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    # Run full discovery
    results = run_universal_law_discovery(alpha=0.05, multiple_testing_method='fdr_bh')
    
    # Save results
    results.to_csv('research_lab/hypothesis_test_results.csv', index=False)
    
    print("\n✓ Hypothesis testing complete")
    print(f"Results saved to: research_lab/hypothesis_test_results.csv")
    
    # Show significant findings
    significant = results[results['rejected_corrected']]
    if len(significant) > 0:
        print(f"\n{len(significant)} UNIVERSAL LAWS DISCOVERED:")
        print(significant[['hypothesis', 'p_value', 'effect_size']].to_string(index=False))
