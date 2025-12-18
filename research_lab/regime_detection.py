"""
REGIME DETECTION AND CLASSIFICATION

A "universal law" must hold across ALL market regimes.
But first we need to IDENTIFY the regimes objectively.

Regimes to detect:
1. Bull/Bear/Sideways markets (trend)
2. High/Low volatility (VIX, realized vol)
3. Risk-on/Risk-off (sector rotation)
4. Liquidity regimes (bid-ask spreads, volume)
5. Correlation regimes (dispersion)

Method: Hidden Markov Models + manual breakpoint detection

This takes time to build properly. No rushing.
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.signal import find_peaks
from sklearn.mixture import GaussianMixture
from hmmlearn import hmm
import sqlite3
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class MarketRegimeDetector:
    """
    Detect market regimes using multiple methods:
    - HMM (Hidden Markov Model) for state detection
    - Gaussian Mixture for clustering
    - Manual breakpoints (known crises)
    - Rolling statistics (vol, correlation, trend)
    """
    
    def __init__(self, db_path: str = "data/market_data.db"):
        self.db_path = db_path
        self.regimes = None  # Will store detected regimes
    
    def get_market_index_returns(self, proxy_ticker: str = 'SPY') -> pd.Series:
        """
        Get market index returns as proxy for overall market regime
        
        If SPY not in DB, use equal-weight average of all stocks
        """
        conn = sqlite3.connect(self.db_path)
        
        # Try to get SPY first
        df = pd.read_sql_query(
            "SELECT date, adj_close FROM daily_bars WHERE ticker = ? ORDER BY date",
            conn, params=[proxy_ticker]
        )
        
        if len(df) > 0:
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
            returns = df['adj_close'].pct_change().dropna()
            conn.close()
            return returns
        
        # Fallback: equal-weight all stocks
        logger.info(f"{proxy_ticker} not found, using equal-weight market returns")
        
        # Get all returns, average across stocks each day
        all_data = pd.read_sql_query(
            "SELECT date, ticker, adj_close FROM daily_bars ORDER BY date, ticker",
            conn
        )
        conn.close()
        
        all_data['date'] = pd.to_datetime(all_data['date'])
        prices = all_data.pivot(index='date', columns='ticker', values='adj_close')
        returns = prices.pct_change()
        
        # Equal-weight average
        market_returns = returns.mean(axis=1).dropna()
        
        return market_returns
    
    def detect_volatility_regimes(self, returns: pd.Series, 
                                  n_regimes: int = 3,
                                  window: int = 20) -> pd.Series:
        """
        Detect high/normal/low volatility regimes
        
        Method: HMM on rolling volatility
        
        Returns: Series with regime labels (0=low, 1=normal, 2=high)
        """
        # Calculate rolling volatility
        vol = returns.rolling(window).std() * np.sqrt(252)  # Annualized
        vol = vol.dropna()
        
        # Fit HMM
        X = vol.values.reshape(-1, 1)
        
        model = hmm.GaussianHMM(
            n_components=n_regimes,
            covariance_type="full",
            n_iter=1000,
            random_state=42
        )
        
        model.fit(X)
        states = model.predict(X)
        
        # Map states to low/normal/high based on mean vol
        state_means = []
        for i in range(n_regimes):
            state_means.append(vol[states == i].mean())
        
        # Sort states by mean vol
        state_order = np.argsort(state_means)
        state_mapping = {old: new for new, old in enumerate(state_order)}
        
        states_mapped = np.array([state_mapping[s] for s in states])
        
        regime_series = pd.Series(states_mapped, index=vol.index)
        regime_series.name = 'vol_regime'
        
        return regime_series
    
    def detect_trend_regimes(self, returns: pd.Series,
                            sma_short: int = 50,
                            sma_long: int = 200) -> pd.Series:
        """
        Detect bull/bear/sideways using moving averages
        
        Bull: price > SMA200 and SMA50 > SMA200
        Bear: price < SMA200 and SMA50 < SMA200
        Sideways: otherwise
        
        Returns: Series with regime labels (0=bear, 1=sideways, 2=bull)
        """
        # Convert returns to price index
        price = (1 + returns).cumprod()
        
        sma_s = price.rolling(sma_short).mean()
        sma_l = price.rolling(sma_long).mean()
        
        # Define regimes
        bull = (price > sma_l) & (sma_s > sma_l)
        bear = (price < sma_l) & (sma_s < sma_l)
        
        regimes = pd.Series(1, index=price.index)  # Default sideways
        regimes[bull] = 2
        regimes[bear] = 0
        
        regimes.name = 'trend_regime'
        
        return regimes
    
    def detect_correlation_regimes(self, window: int = 60, 
                                   n_regimes: int = 2) -> pd.Series:
        """
        Detect high/low correlation regimes
        
        High correlation = crisis, low correlation = normal markets
        
        Method: Average pairwise correlation of stocks, HMM clustering
        """
        conn = sqlite3.connect(self.db_path)
        
        # Get returns for subset of stocks (sample for speed)
        df = pd.read_sql_query(
            """
            SELECT date, ticker, adj_close 
            FROM daily_bars 
            WHERE ticker IN (
                SELECT DISTINCT ticker FROM daily_bars 
                ORDER BY RANDOM() LIMIT 500
            )
            ORDER BY date, ticker
            """,
            conn
        )
        conn.close()
        
        df['date'] = pd.to_datetime(df['date'])
        prices = df.pivot(index='date', columns='ticker', values='adj_close')
        returns = prices.pct_change().dropna()
        
        # Rolling average correlation
        avg_corr = []
        dates = []
        
        for i in range(window, len(returns)):
            window_returns = returns.iloc[i-window:i]
            corr_matrix = window_returns.corr()
            
            # Average correlation (exclude diagonal)
            mask = np.triu(np.ones_like(corr_matrix), k=1).astype(bool)
            avg = corr_matrix.where(mask).stack().mean()
            
            avg_corr.append(avg)
            dates.append(returns.index[i])
        
        avg_corr_series = pd.Series(avg_corr, index=dates)
        
        # Fit HMM
        X = avg_corr_series.values.reshape(-1, 1)
        
        model = hmm.GaussianHMM(
            n_components=n_regimes,
            covariance_type="full",
            n_iter=1000,
            random_state=42
        )
        
        model.fit(X)
        states = model.predict(X)
        
        # Map: 0=low corr, 1=high corr
        state_means = [avg_corr_series[states == i].mean() for i in range(n_regimes)]
        state_order = np.argsort(state_means)
        state_mapping = {old: new for new, old in enumerate(state_order)}
        
        states_mapped = np.array([state_mapping[s] for s in states])
        
        regime_series = pd.Series(states_mapped, index=avg_corr_series.index)
        regime_series.name = 'corr_regime'
        
        return regime_series
    
    def detect_manual_breakpoints(self, returns: pd.Series) -> pd.DataFrame:
        """
        Known regime changes (crises, policy shifts)
        
        Major events in 2023-2025:
        - 2023 Q1: Banking crisis (SVB, Credit Suisse)
        - 2023 Q2-Q4: AI boom starts
        - 2024 Q1: Rate cut expectations
        - 2024 Q2-Q3: Soft landing narrative
        - 2024 Q4: Election uncertainty
        - 2025: Current (AI acceleration)
        """
        breakpoints = [
            ('2023-03-10', '2023-03-31', 'Banking Crisis'),
            ('2023-04-01', '2023-12-31', 'AI Boom Start'),
            ('2024-01-01', '2024-03-31', 'Rate Cut Hopes'),
            ('2024-04-01', '2024-09-30', 'Soft Landing'),
            ('2024-10-01', '2024-11-30', 'Election Uncertainty'),
            ('2024-12-01', '2025-12-31', 'AI Acceleration')
        ]
        
        regime_labels = pd.Series('Unknown', index=returns.index)
        
        for start, end, label in breakpoints:
            mask = (returns.index >= start) & (returns.index <= end)
            regime_labels[mask] = label
        
        return regime_labels
    
    def create_regime_calendar(self, returns: pd.Series) -> pd.DataFrame:
        """
        Combine all regime detections into unified calendar
        
        Returns: DataFrame with date index and regime columns
        """
        logger.info("Detecting market regimes...")
        
        # Volatility regimes
        logger.info("  [1/4] Volatility regimes (HMM)...")
        vol_regimes = self.detect_volatility_regimes(returns)
        
        # Trend regimes
        logger.info("  [2/4] Trend regimes (SMA crossover)...")
        trend_regimes = self.detect_trend_regimes(returns)
        
        # Correlation regimes
        logger.info("  [3/4] Correlation regimes (HMM)...")
        corr_regimes = self.detect_correlation_regimes()
        
        # Manual breakpoints
        logger.info("  [4/4] Manual breakpoints (crisis events)...")
        manual_regimes = self.detect_manual_breakpoints(returns)
        
        # Combine all
        calendar = pd.DataFrame({
            'returns': returns,
            'vol_regime': vol_regimes,
            'trend_regime': trend_regimes,
            'corr_regime': corr_regimes,
            'manual_regime': manual_regimes
        })
        
        # Add regime labels
        calendar['vol_label'] = calendar['vol_regime'].map({0: 'Low Vol', 1: 'Normal Vol', 2: 'High Vol'})
        calendar['trend_label'] = calendar['trend_regime'].map({0: 'Bear', 1: 'Sideways', 2: 'Bull'})
        calendar['corr_label'] = calendar['corr_regime'].map({0: 'Low Corr', 1: 'High Corr'})
        
        self.regimes = calendar
        
        logger.info("✓ Regime calendar created")
        
        return calendar
    
    def get_regime_statistics(self) -> pd.DataFrame:
        """
        Summary statistics for each regime
        
        For each regime type, calculate:
        - Average return
        - Volatility
        - Sharpe ratio
        - Duration
        - Frequency
        """
        if self.regimes is None:
            raise ValueError("Must run create_regime_calendar first")
        
        stats = []
        
        # Volatility regime stats
        for regime in [0, 1, 2]:
            mask = self.regimes['vol_regime'] == regime
            if mask.sum() > 0:
                subset = self.regimes[mask]
                stats.append({
                    'regime_type': 'Volatility',
                    'regime': regime,
                    'label': {0: 'Low', 1: 'Normal', 2: 'High'}[regime],
                    'avg_return': subset['returns'].mean() * 252,
                    'volatility': subset['returns'].std() * np.sqrt(252),
                    'sharpe': (subset['returns'].mean() / subset['returns'].std() * np.sqrt(252)) if subset['returns'].std() > 0 else 0,
                    'days': len(subset),
                    'pct_time': len(subset) / len(self.regimes) * 100
                })
        
        # Trend regime stats
        for regime in [0, 1, 2]:
            mask = self.regimes['trend_regime'] == regime
            if mask.sum() > 0:
                subset = self.regimes[mask]
                stats.append({
                    'regime_type': 'Trend',
                    'regime': regime,
                    'label': {0: 'Bear', 1: 'Sideways', 2: 'Bull'}[regime],
                    'avg_return': subset['returns'].mean() * 252,
                    'volatility': subset['returns'].std() * np.sqrt(252),
                    'sharpe': (subset['returns'].mean() / subset['returns'].std() * np.sqrt(252)) if subset['returns'].std() > 0 else 0,
                    'days': len(subset),
                    'pct_time': len(subset) / len(self.regimes) * 100
                })
        
        # Correlation regime stats
        for regime in [0, 1]:
            mask = self.regimes['corr_regime'] == regime
            if mask.sum() > 0:
                subset = self.regimes[mask]
                stats.append({
                    'regime_type': 'Correlation',
                    'regime': regime,
                    'label': {0: 'Low', 1: 'High'}[regime],
                    'avg_return': subset['returns'].mean() * 252,
                    'volatility': subset['returns'].std() * np.sqrt(252),
                    'sharpe': (subset['returns'].mean() / subset['returns'].std() * np.sqrt(252)) if subset['returns'].std() > 0 else 0,
                    'days': len(subset),
                    'pct_time': len(subset) / len(self.regimes) * 100
                })
        
        return pd.DataFrame(stats)
    
    def test_regime_dependence(self, hypothesis_test_func: callable,
                               regime_type: str = 'vol_regime') -> Dict:
        """
        Test if a hypothesis holds WITHIN each regime
        
        This is the KEY test for universality:
        - If law holds in ALL regimes → Universal
        - If law holds in SOME regimes → Regime-dependent
        - If law holds in NO regimes → Not a law
        
        Args:
            hypothesis_test_func: Function that takes returns DataFrame and returns HypothesisTest
            regime_type: 'vol_regime', 'trend_regime', or 'corr_regime'
        
        Returns: Dict with test results per regime
        """
        if self.regimes is None:
            raise ValueError("Must run create_regime_calendar first")
        
        results = {}
        
        unique_regimes = self.regimes[regime_type].dropna().unique()
        
        for regime in sorted(unique_regimes):
            mask = self.regimes[regime_type] == regime
            regime_dates = self.regimes[mask].index
            
            # Run test on this regime's data
            test_result = hypothesis_test_func(regime_dates)
            
            results[int(regime)] = test_result
        
        return results


def analyze_regime_transitions(regimes: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze how markets transition between regimes
    
    Key questions:
    - Do we go from low vol → high vol suddenly (crisis) or gradually?
    - Does high correlation predict regime change?
    - Are there absorbing states?
    """
    transitions = []
    
    for regime_type in ['vol_regime', 'trend_regime', 'corr_regime']:
        regime_col = regimes[regime_type].dropna()
        
        # Count transitions
        for i in range(1, len(regime_col)):
            prev_state = regime_col.iloc[i-1]
            curr_state = regime_col.iloc[i]
            
            if prev_state != curr_state:
                transitions.append({
                    'date': regime_col.index[i],
                    'regime_type': regime_type,
                    'from_state': int(prev_state),
                    'to_state': int(curr_state),
                    'transition': f"{int(prev_state)}→{int(curr_state)}"
                })
    
    transition_df = pd.DataFrame(transitions)
    
    # Transition probability matrix for each regime type
    for regime_type in ['vol_regime', 'trend_regime', 'corr_regime']:
        subset = transition_df[transition_df['regime_type'] == regime_type]
        
        if len(subset) > 0:
            print(f"\n{regime_type} transition counts:")
            print(subset['transition'].value_counts())
    
    return transition_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    print("=" * 70)
    print("MARKET REGIME DETECTION")
    print("=" * 70)
    
    # Initialize detector
    detector = MarketRegimeDetector()
    
    # Get market returns
    print("\nLoading market data...")
    market_returns = detector.get_market_index_returns()
    
    print(f"Data range: {market_returns.index[0]} to {market_returns.index[-1]}")
    print(f"Trading days: {len(market_returns)}")
    
    # Detect all regimes
    print("\nDetecting regimes...")
    calendar = detector.create_regime_calendar(market_returns)
    
    # Save calendar
    calendar.to_csv('research_lab/regime_calendar.csv')
    print(f"\n✓ Regime calendar saved to: research_lab/regime_calendar.csv")
    
    # Get statistics
    print("\nRegime Statistics:")
    print("=" * 70)
    stats = detector.get_regime_statistics()
    print(stats.to_string(index=False))
    
    # Analyze transitions
    print("\n" + "=" * 70)
    print("REGIME TRANSITIONS")
    print("=" * 70)
    transitions = analyze_regime_transitions(calendar)
    
    print(f"\n✓ Regime detection complete")
    print(f"✓ Use this to test if 'laws' are universal or regime-dependent")
