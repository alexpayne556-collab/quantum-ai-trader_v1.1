"""
FACTOR ANALYSIS FRAMEWORK

Discover latent factors that drive returns across the universe.

Traditional factors (Fama-French):
- Market (beta)
- Size (SMB - small minus big)
- Value (HML - high minus low book/market)
- Momentum (UMD - up minus down)
- Quality (profitability, investment)

But we DON'T assume these exist. We DISCOVER them using:
1. PCA (Principal Component Analysis) - find orthogonal return drivers
2. Factor extraction - which stocks load on which factors?
3. Time-varying factor importance - do factors change in different regimes?
4. Cross-sectional tests - do factors predict returns out-of-sample?

Real factor research. Not assuming Fama-French, discovering what actually exists.
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.stats import spearmanr
import sqlite3
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class FactorDiscovery:
    """
    Discover latent factors using PCA and factor analysis
    
    NO assumptions. Let the data reveal the structure.
    """
    
    def __init__(self, db_path: str = "data/market_data.db"):
        self.db_path = db_path
        self.factors = None
        self.factor_returns = None
        self.factor_loadings = None
    
    def get_cross_section_returns(self, start_date: Optional[str] = None,
                                  end_date: Optional[str] = None,
                                  min_stocks: int = 100) -> pd.DataFrame:
        """
        Get returns matrix: dates × tickers
        
        Filter to dates with sufficient cross-section
        """
        conn = sqlite3.connect(self.db_path)
        
        query = "SELECT date, ticker, adj_close FROM daily_bars"
        params = []
        
        if start_date:
            query += " WHERE date >= ?"
            params.append(start_date)
        
        if end_date:
            if start_date:
                query += " AND date <= ?"
            else:
                query += " WHERE date <= ?"
            params.append(end_date)
        
        query += " ORDER BY date, ticker"
        
        df = pd.read_sql_query(query, conn, params=params if params else None)
        conn.close()
        
        df['date'] = pd.to_datetime(df['date'])
        prices = df.pivot(index='date', columns='ticker', values='adj_close')
        returns = prices.pct_change().dropna()
        
        # Filter to dates with enough stocks
        valid_dates = returns.count(axis=1) >= min_stocks
        returns = returns[valid_dates]
        
        # Drop tickers with too many missing values
        valid_tickers = returns.count() >= len(returns) * 0.5
        returns = returns.loc[:, valid_tickers]
        
        return returns
    
    def extract_pca_factors(self, returns: pd.DataFrame, 
                           n_components: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract principal components from return matrix
        
        PCA finds orthogonal factors that explain variance
        
        Returns:
            (factor_returns, factor_loadings)
            factor_returns: T × n_components (time series of factor returns)
            factor_loadings: n_stocks × n_components (how each stock loads on factors)
        """
        # Fill missing values with 0 (stock didn't trade that day)
        returns_filled = returns.fillna(0)
        
        # Standardize returns (mean 0, std 1) for each date
        scaler = StandardScaler()
        returns_standardized = scaler.fit_transform(returns_filled)
        
        # Fit PCA
        pca = PCA(n_components=n_components)
        factor_returns = pca.fit_transform(returns_standardized)
        
        # Loadings = how each stock loads on each factor
        factor_loadings = pca.components_.T
        
        # Explained variance
        explained_var = pca.explained_variance_ratio_
        
        logger.info(f"PCA Factors extracted:")
        for i, var in enumerate(explained_var):
            logger.info(f"  PC{i+1}: {var*100:.2f}% of variance")
        logger.info(f"  Total: {explained_var.sum()*100:.2f}% explained")
        
        self.factor_returns = pd.DataFrame(
            factor_returns,
            index=returns.index,
            columns=[f'PC{i+1}' for i in range(n_components)]
        )
        
        self.factor_loadings = pd.DataFrame(
            factor_loadings,
            index=returns.columns,
            columns=[f'PC{i+1}' for i in range(n_components)]
        )
        
        return factor_returns, factor_loadings
    
    def interpret_factors(self, top_n: int = 10) -> Dict:
        """
        Interpret factors by looking at top loading stocks
        
        e.g., if PC1 loads heavily on tech stocks → tech factor
             if PC2 loads heavily on small caps → size factor
        
        Returns:
            Dict mapping factor names to interpretation
        """
        if self.factor_loadings is None:
            raise ValueError("Must run extract_pca_factors first")
        
        interpretations = {}
        
        for factor in self.factor_loadings.columns:
            # Get top positive and negative loadings
            top_pos = self.factor_loadings[factor].nlargest(top_n)
            top_neg = self.factor_loadings[factor].nsmallest(top_n)
            
            interpretations[factor] = {
                'top_positive_tickers': top_pos.index.tolist(),
                'top_positive_loadings': top_pos.values.tolist(),
                'top_negative_tickers': top_neg.index.tolist(),
                'top_negative_loadings': top_neg.values.tolist()
            }
        
        return interpretations
    
    def test_factor_predictive_power(self, factor_returns: pd.DataFrame,
                                    future_returns: pd.DataFrame,
                                    horizon: int = 20) -> pd.DataFrame:
        """
        Test if factor returns predict future stock returns
        
        Method: Cross-sectional regression
        For each date t:
            Run regression: returns[t+horizon] ~ factor_loadings[t]
            Test if loadings are significant
        
        Returns:
            DataFrame with t-stats and p-values for each factor
        """
        results = []
        
        for date in factor_returns.index[:-horizon]:
            # Future returns for this date
            future_date = factor_returns.index[factor_returns.index.get_loc(date) + horizon]
            
            if future_date not in future_returns.index:
                continue
            
            # Cross-sectional regression: future returns ~ current factor exposures
            y = future_returns.loc[future_date].dropna()
            
            # Get factor exposures for stocks in y
            common_stocks = y.index.intersection(self.factor_loadings.index)
            
            if len(common_stocks) < 50:
                continue
            
            y_aligned = y[common_stocks].values
            X = self.factor_loadings.loc[common_stocks].values
            
            # OLS regression
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
            model.fit(X, y_aligned)
            
            # Store coefficients
            results.append({
                'date': date,
                **{f'{col}_coef': coef for col, coef in zip(self.factor_loadings.columns, model.coef_)}
            })
        
        results_df = pd.DataFrame(results)
        
        # Test if coefficients are significantly different from zero
        summary = {}
        for col in self.factor_loadings.columns:
            coef_col = f'{col}_coef'
            if coef_col in results_df.columns:
                coefs = results_df[coef_col].dropna()
                t_stat, p_value = stats.ttest_1samp(coefs, 0)
                
                summary[col] = {
                    'mean_coef': coefs.mean(),
                    't_stat': t_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }
        
        return pd.DataFrame(summary).T
    
    def build_market_factor(self, returns: pd.DataFrame) -> pd.Series:
        """
        Build market factor (equal-weight or value-weight average)
        
        This is the traditional "beta" - exposure to overall market
        """
        # Equal-weight for now (we don't have market caps in DB yet)
        market_returns = returns.mean(axis=1)
        market_returns.name = 'Market'
        
        return market_returns
    
    def build_size_factor(self, returns: pd.DataFrame) -> pd.Series:
        """
        Build size factor (SMB - Small Minus Big)
        
        Problem: We need market caps to sort stocks
        TODO: Add market cap data from API
        
        For now: proxy with volatility (small caps more volatile)
        """
        # Calculate volatility for each stock
        vols = returns.std()
        
        # Sort into quintiles
        high_vol = vols >= vols.quantile(0.8)
        low_vol = vols <= vols.quantile(0.2)
        
        # SMB = high vol (small) - low vol (large)
        smb = returns.loc[:, high_vol].mean(axis=1) - returns.loc[:, low_vol].mean(axis=1)
        smb.name = 'SMB_proxy'
        
        return smb
    
    def build_momentum_factor(self, returns: pd.DataFrame, 
                             lookback: int = 120,
                             skip_days: int = 20) -> pd.Series:
        """
        Build momentum factor (UMD - Up Minus Down)
        
        Past winners vs. past losers
        """
        # Calculate past returns (skip recent to avoid microstructure effects)
        past_returns = returns.shift(skip_days).rolling(lookback).sum()
        
        momentum_scores = []
        dates = []
        
        for date in past_returns.index[lookback + skip_days:]:
            scores = past_returns.loc[date].dropna()
            
            if len(scores) < 50:
                continue
            
            # Winners vs losers
            winners = scores >= scores.quantile(0.8)
            losers = scores <= scores.quantile(0.2)
            
            # UMD = winner returns - loser returns
            current_returns = returns.loc[date]
            winner_ret = current_returns[scores.index[winners]].mean()
            loser_ret = current_returns[scores.index[losers]].mean()
            
            momentum_scores.append(winner_ret - loser_ret)
            dates.append(date)
        
        umd = pd.Series(momentum_scores, index=dates, name='UMD')
        
        return umd
    
    def build_reversal_factor(self, returns: pd.DataFrame,
                             lookback: int = 5) -> pd.Series:
        """
        Build short-term reversal factor
        
        Recent losers bounce back
        """
        past_returns = returns.rolling(lookback).sum()
        
        reversal_scores = []
        dates = []
        
        for date in past_returns.index[lookback:]:
            scores = past_returns.loc[date].dropna()
            
            if len(scores) < 50:
                continue
            
            # Recent losers vs recent winners
            losers = scores <= scores.quantile(0.2)
            winners = scores >= scores.quantile(0.8)
            
            # Reversal = loser returns - winner returns (opposite of momentum)
            current_returns = returns.loc[date]
            loser_ret = current_returns[scores.index[losers]].mean()
            winner_ret = current_returns[scores.index[winners]].mean()
            
            reversal_scores.append(loser_ret - winner_ret)
            dates.append(date)
        
        rev = pd.Series(reversal_scores, index=dates, name='REV')
        
        return rev


class FactorTimingAnalysis:
    """
    Test if factor returns are predictable (time-varying)
    
    Do factors work better in certain regimes?
    """
    
    @staticmethod
    def test_factor_autocorrelation(factor_returns: pd.Series) -> Dict:
        """
        Test if factor returns are autocorrelated (momentum in factors)
        """
        from statsmodels.tsa.stattools import acf, pacf
        
        acf_vals = acf(factor_returns.dropna(), nlags=20, fft=True)
        pacf_vals = pacf(factor_returns.dropna(), nlags=20)
        
        # Test first lag
        from statsmodels.stats.diagnostic import acorr_ljungbox
        lb_test = acorr_ljungbox(factor_returns.dropna(), lags=[1, 5, 20], return_df=True)
        
        return {
            'acf_1': float(acf_vals[1]),
            'acf_5': float(acf_vals[5]),
            'pacf_1': float(pacf_vals[1]),
            'ljung_box_p': float(lb_test['lb_pvalue'].iloc[0])
        }
    
    @staticmethod
    def test_regime_dependence(factor_returns: pd.Series,
                               regime_labels: pd.Series) -> pd.DataFrame:
        """
        Test if factor performs differently in different regimes
        """
        # Align
        common_dates = factor_returns.index.intersection(regime_labels.index)
        factor_aligned = factor_returns[common_dates]
        regimes_aligned = regime_labels[common_dates]
        
        # Stats per regime
        stats_list = []
        
        for regime in regimes_aligned.unique():
            mask = regimes_aligned == regime
            subset = factor_aligned[mask]
            
            stats_list.append({
                'regime': regime,
                'mean_return': subset.mean(),
                'volatility': subset.std(),
                'sharpe': subset.mean() / subset.std() if subset.std() > 0 else 0,
                'n_days': len(subset)
            })
        
        return pd.DataFrame(stats_list)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    print("=" * 70)
    print("FACTOR DISCOVERY")
    print("=" * 70)
    
    discoverer = FactorDiscovery()
    
    # Load returns
    print("\n[1/5] Loading cross-sectional returns...")
    returns = discoverer.get_cross_section_returns()
    
    print(f"Returns matrix: {returns.shape[0]} days × {returns.shape[1]} stocks")
    
    # Extract PCA factors
    print("\n[2/5] Extracting PCA factors...")
    factor_returns, factor_loadings = discoverer.extract_pca_factors(returns, n_components=10)
    
    # Interpret factors
    print("\n[3/5] Interpreting factors...")
    interpretations = discoverer.interpret_factors(top_n=10)
    
    for factor, interp in list(interpretations.items())[:3]:
        print(f"\n{factor}:")
        print(f"  Top positive: {', '.join(interp['top_positive_tickers'][:5])}")
        print(f"  Top negative: {', '.join(interp['top_negative_tickers'][:5])}")
    
    # Build traditional factors
    print("\n[4/5] Building traditional factors...")
    market = discoverer.build_market_factor(returns)
    smb = discoverer.build_size_factor(returns)
    umd = discoverer.build_momentum_factor(returns, lookback=120)
    rev = discoverer.build_reversal_factor(returns, lookback=5)
    
    # Test autocorrelation
    print("\n[5/5] Testing factor properties...")
    timing = FactorTimingAnalysis()
    
    print("\nMarket factor autocorrelation:")
    print(timing.test_factor_autocorrelation(market))
    
    print("\nMomentum factor autocorrelation:")
    print(timing.test_factor_autocorrelation(umd))
    
    # Save
    factor_returns.to_csv('research_lab/pca_factor_returns.csv')
    factor_loadings.to_csv('research_lab/pca_factor_loadings.csv')
    
    traditional_factors = pd.DataFrame({
        'Market': market,
        'SMB_proxy': smb,
        'UMD': umd,
        'REV': rev
    })
    traditional_factors.to_csv('research_lab/traditional_factors.csv')
    
    print(f"\n✓ Factor analysis complete")
    print(f"✓ Files saved to research_lab/")
