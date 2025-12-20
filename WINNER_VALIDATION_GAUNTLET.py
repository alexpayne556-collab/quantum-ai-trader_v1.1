#!/usr/bin/env python3
"""
WINNER VALIDATION GAUNTLET
===========================
Full statistical validation of the 20 winning hypotheses.

Tests:
1. Walk-Forward Validation (5 folds, 252d train / 63d test)
2. Monte Carlo Simulation (5000 shuffled permutations)
3. Multiple Timeframes (2010-2015, 2015-2020, 2020-2025)
4. Market Regime Analysis (Bull/Bear/Sideways)
5. Transaction Cost Sensitivity (0-50bps)
6. Out-of-Sample Hold-Out (last 2 years)

Run:
    python WINNER_VALIDATION_GAUNTLET.py
"""

import yfinance as yf
import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime, timedelta
import warnings
import time
from pathlib import Path

warnings.filterwarnings('ignore')

# ============================================================================
# TOP 20 WINNERS TO VALIDATE
# ============================================================================

WINNERS = [
    {'id': 'H18', 'name': 'RSI Extreme (20/80)', 'spread': 1.095, 'category': 'Mean Reversion'},
    {'id': 'H19', 'name': 'Bollinger Band MR', 'spread': 0.413, 'category': 'Mean Reversion'},
    {'id': 'H17', 'name': 'RSI Mean Reversion', 'spread': 0.407, 'category': 'Mean Reversion'},
    {'id': 'H27E', 'name': 'Multi-Indicator Oversold', 'spread': 0.395, 'category': 'Mean Reversion'},
    {'id': 'H27', 'name': 'Post-Large-Move Reversal', 'spread': 0.397, 'category': 'Mean Reversion'},
    {'id': 'H27C', 'name': 'MA Distance', 'spread': 0.365, 'category': 'Mean Reversion'},
    {'id': 'H73', 'name': 'Flight to Quality', 'spread': 0.296, 'category': 'Cross-Asset'},
    {'id': 'H27D', 'name': 'Consecutive Down Reversal', 'spread': 0.264, 'category': 'Mean Reversion'},
    {'id': 'H20', 'name': 'VIX Mean Reversion', 'spread': 0.231, 'category': 'Volatility'},
    {'id': 'H128', 'name': 'VIX Turbulence', 'spread': 0.211, 'category': 'Creative'},
    {'id': 'H21', 'name': 'VIX Percentile', 'spread': 0.182, 'category': 'Volatility'},
    {'id': 'H16', 'name': 'Weekly Reversal', 'spread': 0.165, 'category': 'Mean Reversion'},
    {'id': 'H27B', 'name': 'Z-Score Mean Reversion', 'spread': 0.162, 'category': 'Mean Reversion'},
    {'id': 'H57', 'name': 'Bond Leading Indicator', 'spread': 0.140, 'category': 'Cross-Asset'},
    {'id': 'H53', 'name': 'Quarter-End Effect', 'spread': 0.133, 'category': 'Seasonality'},
    {'id': 'H62', 'name': 'Oil-Equity Relationship', 'spread': 0.121, 'category': 'Cross-Asset'},
    {'id': 'H39', 'name': 'Cross-Asset Vol Signal', 'spread': 0.116, 'category': 'Volatility'},
    {'id': 'H126', 'name': 'Small vs Large Cap', 'spread': 0.068, 'category': 'Sentiment'},
    {'id': 'H49', 'name': 'September Effect', 'spread': 0.067, 'category': 'Seasonality'},
    {'id': 'H142', 'name': 'Dollar Valuation', 'spread': 0.054, 'category': 'Creative'},
]


# ============================================================================
# SIGNAL FUNCTIONS
# ============================================================================

def calc_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def signal_rsi_extreme(data: pd.DataFrame, oversold: int = 20, overbought: int = 80) -> pd.Series:
    """H18: RSI Extreme (20/80)."""
    rsi = calc_rsi(data['close'], 14)
    return (rsi < oversold).astype(int)


def signal_rsi_mean_reversion(data: pd.DataFrame, oversold: int = 30, overbought: int = 70) -> pd.Series:
    """H17: RSI Mean Reversion (30/70)."""
    rsi = calc_rsi(data['close'], 14)
    return (rsi < oversold).astype(int)


def signal_bollinger_mr(data: pd.DataFrame, period: int = 20, num_std: float = 2.0) -> pd.Series:
    """H19: Bollinger Band Mean Reversion."""
    ma = data['close'].rolling(period).mean()
    std = data['close'].rolling(period).std()
    lower = ma - num_std * std
    return (data['close'] < lower).astype(int)


def signal_multi_indicator_oversold(data: pd.DataFrame) -> pd.Series:
    """H27E: Multi-Indicator Oversold."""
    rsi = calc_rsi(data['close'], 14)
    ma20 = data['close'].rolling(20).mean()
    ma50 = data['close'].rolling(50).mean()
    
    rsi_oversold = rsi < 30
    below_ma20 = data['close'] < ma20 * 0.95
    below_ma50 = data['close'] < ma50 * 0.95
    
    score = rsi_oversold.astype(int) + below_ma20.astype(int) + below_ma50.astype(int)
    return (score >= 2).astype(int)


def signal_post_large_move(data: pd.DataFrame, threshold: float = 0.03) -> pd.Series:
    """H27: Post-Large-Move Reversal."""
    daily_ret = data['close'].pct_change()
    large_down = daily_ret < -threshold
    return large_down.shift(1).fillna(0).astype(int)


def signal_ma_distance(data: pd.DataFrame, period: int = 50, threshold: float = 0.05) -> pd.Series:
    """H27C: MA Distance."""
    ma = data['close'].rolling(period).mean()
    distance = (data['close'] - ma) / ma
    return (distance < -threshold).astype(int)


def signal_flight_to_quality(data: pd.DataFrame, tlt_data: pd.DataFrame = None, 
                              gld_data: pd.DataFrame = None, lookback: int = 5) -> pd.Series:
    """H73: Flight to Quality."""
    if tlt_data is None or gld_data is None:
        return pd.Series(0, index=data.index)
    
    spy_ret = data['close'].pct_change(lookback)
    tlt_ret = tlt_data['close'].pct_change(lookback).reindex(data.index).ffill()
    gld_ret = gld_data['close'].pct_change(lookback).reindex(data.index).ffill()
    
    ftq = (tlt_ret > 0.01) & (gld_ret > 0.01) & (spy_ret < -0.01)
    return ftq.shift(1).fillna(0).astype(int)


def signal_consecutive_down(data: pd.DataFrame, days: int = 3) -> pd.Series:
    """H27D: Consecutive Down Reversal."""
    daily_ret = data['close'].pct_change()
    down_day = (daily_ret < 0).astype(int)
    consecutive = down_day.rolling(days).sum()
    return (consecutive >= days).astype(int)


def signal_vix_mean_reversion(data: pd.DataFrame, vix_data: pd.Series = None, threshold: int = 25) -> pd.Series:
    """H20: VIX Mean Reversion."""
    if vix_data is None:
        return pd.Series(0, index=data.index)
    vix = vix_data.reindex(data.index).ffill()
    return (vix > threshold).astype(int)


def signal_vix_turbulence(data: pd.DataFrame, vix_data: pd.Series = None, 
                          lookback: int = 14, threshold: float = 2.0) -> pd.Series:
    """H128: VIX Turbulence (Vol of VIX)."""
    if vix_data is None:
        return pd.Series(0, index=data.index)
    
    vix = vix_data.reindex(data.index).ffill()
    vix_changes = vix.diff()
    vix_vol = vix_changes.rolling(lookback).std()
    vix_vol_mean = vix_vol.rolling(252).mean()
    vix_vol_std = vix_vol.rolling(252).std()
    z_score = (vix_vol - vix_vol_mean) / vix_vol_std
    
    return (z_score > threshold).astype(int)


def signal_vix_percentile(data: pd.DataFrame, vix_data: pd.Series = None, threshold: float = 0.8) -> pd.Series:
    """H21: VIX Percentile."""
    if vix_data is None:
        return pd.Series(0, index=data.index)
    vix = vix_data.reindex(data.index).ffill()
    percentile = vix.rolling(252).rank(pct=True)
    return (percentile > threshold).astype(int)


def signal_weekly_reversal(data: pd.DataFrame, threshold: float = -0.03) -> pd.Series:
    """H16: Weekly Reversal."""
    weekly_ret = data['close'].pct_change(5)
    return (weekly_ret < threshold).astype(int)


def signal_zscore_mr(data: pd.DataFrame, period: int = 20, threshold: float = -2.0) -> pd.Series:
    """H27B: Z-Score Mean Reversion."""
    ma = data['close'].rolling(period).mean()
    std = data['close'].rolling(period).std()
    zscore = (data['close'] - ma) / std
    return (zscore < threshold).astype(int)


def signal_bond_leading(data: pd.DataFrame, tlt_data: pd.DataFrame = None, lookback: int = 5) -> pd.Series:
    """H57: Bond Leading Indicator."""
    if tlt_data is None:
        return pd.Series(1, index=data.index)
    tlt_ret = tlt_data['close'].pct_change(lookback).reindex(data.index).ffill()
    return (tlt_ret > 0).astype(int)


def signal_quarter_end(data: pd.DataFrame) -> pd.Series:
    """H53: Quarter-End Effect."""
    month = pd.Series(data.index.month, index=data.index)
    day = pd.Series(data.index.day, index=data.index)
    days_in_month = pd.Series(data.index.days_in_month, index=data.index)
    
    quarter_end_months = [3, 6, 9, 12]
    is_quarter_end = month.isin(quarter_end_months) & (day >= days_in_month - 5)
    return is_quarter_end.astype(int)


def signal_oil_equity(data: pd.DataFrame, uso_data: pd.DataFrame = None, lookback: int = 21) -> pd.Series:
    """H62: Oil-Equity Relationship."""
    if uso_data is None:
        return pd.Series(1, index=data.index)
    
    spy_mom = data['close'].pct_change(lookback)
    uso_mom = uso_data['close'].pct_change(lookback).reindex(data.index).ffill()
    
    both_up = (spy_mom > 0) & (uso_mom > 0)
    divergence = (spy_mom > 0) & (uso_mom < -0.05)
    
    signal = pd.Series(1, index=data.index)
    signal[divergence] = 0
    return signal


def signal_cross_asset_vol(data: pd.DataFrame, vix_data: pd.Series = None,
                           iwm_data: pd.DataFrame = None, qqq_data: pd.DataFrame = None) -> pd.Series:
    """H39: Cross-Asset Vol Signal."""
    if vix_data is None:
        return pd.Series(1, index=data.index)
    
    vix = vix_data.reindex(data.index).ffill()
    vix_high = vix > vix.rolling(63).quantile(0.8)
    
    return vix_high.astype(int)


def signal_small_large(data: pd.DataFrame, iwm_data: pd.DataFrame = None, lookback: int = 21) -> pd.Series:
    """H126: Small vs Large Cap."""
    if iwm_data is None:
        return pd.Series(1, index=data.index)
    
    iwm = iwm_data['close'].reindex(data.index).ffill()
    spy = data['close']
    rel_strength = iwm.pct_change(lookback) - spy.pct_change(lookback)
    return (rel_strength > 0).astype(int)


def signal_september_effect(data: pd.DataFrame) -> pd.Series:
    """H49: September Effect (avoid September)."""
    month = pd.Series(data.index.month, index=data.index)
    return (month != 9).astype(int)


def signal_dollar_valuation(data: pd.DataFrame, uup_data: pd.DataFrame = None, lookback: int = 252) -> pd.Series:
    """H142: Dollar Valuation."""
    if uup_data is None:
        return pd.Series(1, index=data.index)
    
    uup = uup_data['close'].reindex(data.index).ffill()
    uup_pct = uup.rolling(lookback).rank(pct=True)
    return (uup_pct < 0.5).astype(int)


# Signal registry
SIGNAL_REGISTRY = {
    'H18': signal_rsi_extreme,
    'H17': signal_rsi_mean_reversion,
    'H19': signal_bollinger_mr,
    'H27E': signal_multi_indicator_oversold,
    'H27': signal_post_large_move,
    'H27C': signal_ma_distance,
    'H73': signal_flight_to_quality,
    'H27D': signal_consecutive_down,
    'H20': signal_vix_mean_reversion,
    'H128': signal_vix_turbulence,
    'H21': signal_vix_percentile,
    'H16': signal_weekly_reversal,
    'H27B': signal_zscore_mr,
    'H57': signal_bond_leading,
    'H53': signal_quarter_end,
    'H62': signal_oil_equity,
    'H39': signal_cross_asset_vol,
    'H126': signal_small_large,
    'H49': signal_september_effect,
    'H142': signal_dollar_valuation,
}


# ============================================================================
# VALIDATION CLASS
# ============================================================================

class WinnerValidationGauntlet:
    """Full validation suite for winning hypotheses."""
    
    def __init__(self, start_date='2010-01-01'):
        self.start_date = start_date
        self.data = {}
        self.results = []
        self.cache_path = Path('./hypothesis_data/')
        self.cache_path.mkdir(exist_ok=True)
        
    def download_data(self, force_refresh: bool = False):
        """Download all required data."""
        print("\n📥 Downloading validation data...")
        
        tickers = ['SPY', 'QQQ', 'IWM', 'TLT', 'GLD', 'USO', 'UUP', '^VIX']
        
        cache_file = self.cache_path / 'validation_data_v2.pkl'
        if cache_file.exists() and not force_refresh:
            print("   ⚡ Loading from cache...")
            self.data = pd.read_pickle(cache_file)
            # Verify data is valid
            if 'SPY' in self.data and len(self.data['SPY']) > 100:
                return
            print("   ⚠ Cache invalid, re-downloading...")
        
        # Download one at a time for reliability
        for ticker in tickers:
            try:
                raw = yf.download(ticker, start=self.start_date, progress=False)
                
                # Handle column names
                if isinstance(raw.columns, pd.MultiIndex):
                    raw.columns = raw.columns.get_level_values(0)
                
                # Standardize columns to lowercase
                col_map = {c: c.lower().replace(' ', '_') for c in raw.columns}
                raw = raw.rename(columns=col_map)
                
                # Create standardized dataframe
                df = pd.DataFrame(index=raw.index)
                df['close'] = raw['close'] if 'close' in raw.columns else raw['adj_close'] if 'adj_close' in raw.columns else None
                df['open'] = raw['open'] if 'open' in raw.columns else None
                df['high'] = raw['high'] if 'high' in raw.columns else None
                df['low'] = raw['low'] if 'low' in raw.columns else None
                df['volume'] = raw['volume'] if 'volume' in raw.columns else 0
                
                self.data[ticker] = df.dropna(subset=['close'])
                print(f"   ✓ {ticker}: {len(df)} rows")
            except Exception as e:
                print(f"   ⚠ {ticker}: {e}")
        
        pd.to_pickle(self.data, cache_file)
        print(f"   ✓ Downloaded {len(self.data)} tickers")
    
    def generate_signal(self, hyp_id: str, data: pd.DataFrame, debug: bool = False) -> pd.Series:
        """Generate signal for a hypothesis."""
        signal_func = SIGNAL_REGISTRY.get(hyp_id)
        if signal_func is None:
            if debug:
                print(f"      [DEBUG] No signal function for {hyp_id}")
            return pd.Series(0, index=data.index)
        
        # Prepare kwargs with cross-asset data
        kwargs = {}
        
        if 'TLT' in self.data:
            kwargs['tlt_data'] = self.data['TLT']
        if 'GLD' in self.data:
            kwargs['gld_data'] = self.data['GLD']
        if 'USO' in self.data:
            kwargs['uso_data'] = self.data['USO']
        if 'UUP' in self.data:
            kwargs['uup_data'] = self.data['UUP']
        if 'IWM' in self.data:
            kwargs['iwm_data'] = self.data['IWM']
        if 'QQQ' in self.data:
            kwargs['qqq_data'] = self.data['QQQ']
        if '^VIX' in self.data:
            kwargs['vix_data'] = self.data['^VIX']['close']
        
        try:
            # Filter kwargs to only those accepted by the function
            import inspect
            sig = inspect.signature(signal_func)
            valid_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
            
            result = signal_func(data, **valid_kwargs)
            
            if debug:
                n_signals = result.sum()
                print(f"      [DEBUG] {hyp_id}: {n_signals} signals out of {len(result)} days")
            
            return result
        except Exception as e:
            if debug:
                print(f"      [DEBUG] Error in {hyp_id}: {e}")
            return pd.Series(0, index=data.index)
    
    def walk_forward_validation(self, hyp_id: str, train_days: int = 252, 
                                 test_days: int = 63, n_folds: int = 5) -> dict:
        """Walk-forward validation with expanding window."""
        data = self.data['SPY'].copy()
        signal = self.generate_signal(hyp_id, data)
        
        hold_period = 21
        forward_ret = data['close'].pct_change(hold_period).shift(-hold_period)
        
        results = []
        total_days = len(data)
        fold_size = (total_days - train_days) // n_folds
        
        for fold in range(n_folds):
            train_end = train_days + fold * fold_size
            test_end = train_end + test_days
            
            if test_end > total_days:
                break
            
            # In-sample
            is_signal = signal.iloc[:train_end]
            is_returns = forward_ret.iloc[:train_end]
            is_long = is_returns[is_signal == 1].dropna()
            is_other = is_returns[is_signal == 0].dropna()
            
            # Out-of-sample
            oos_signal = signal.iloc[train_end:test_end]
            oos_returns = forward_ret.iloc[train_end:test_end]
            oos_long = oos_returns[oos_signal == 1].dropna()
            oos_other = oos_returns[oos_signal == 0].dropna()
            
            if len(is_long) > 10 and len(oos_long) >= 5:
                is_spread = (is_long.mean() - is_other.mean()) * (252 / hold_period)
                oos_spread = (oos_long.mean() - oos_other.mean()) * (252 / hold_period) if len(oos_other) > 0 else oos_long.mean() * (252 / hold_period)
                
                results.append({
                    'fold': fold + 1,
                    'is_spread': is_spread,
                    'oos_spread': oos_spread,
                    'is_n': len(is_long),
                    'oos_n': len(oos_long),
                    'oos_positive': oos_spread > 0,
                })
        
        if not results:
            return {'pass': False, 'reason': 'Insufficient data'}
        
        df = pd.DataFrame(results)
        avg_oos = df['oos_spread'].mean()
        oos_positive_rate = df['oos_positive'].mean()
        consistency = df['oos_spread'].std() / abs(df['oos_spread'].mean()) if df['oos_spread'].mean() != 0 else float('inf')
        
        return {
            'pass': oos_positive_rate >= 0.6 and avg_oos > 0,
            'avg_oos_spread': avg_oos,
            'oos_positive_rate': oos_positive_rate,
            'consistency': consistency,
            'n_folds': len(results),
            'folds': results,
        }
    
    def monte_carlo_test(self, hyp_id: str, n_simulations: int = 5000) -> dict:
        """Monte Carlo permutation test."""
        data = self.data['SPY'].copy()
        signal = self.generate_signal(hyp_id, data)
        
        hold_period = 21
        forward_ret = data['close'].pct_change(hold_period).shift(-hold_period)
        
        valid_idx = signal.notna() & forward_ret.notna()
        signal_valid = signal[valid_idx]
        returns_valid = forward_ret[valid_idx]
        
        # Calculate actual spread
        long_returns = returns_valid[signal_valid == 1]
        other_returns = returns_valid[signal_valid == 0]
        
        if len(long_returns) < 20 or len(other_returns) < 20:
            return {'pass': False, 'reason': 'Insufficient signals'}
        
        actual_spread = long_returns.mean() - other_returns.mean()
        
        # Monte Carlo simulation
        returns_arr = returns_valid.values
        signal_arr = signal_valid.values
        n_long = int(signal_arr.sum())
        
        simulated_spreads = []
        for _ in range(n_simulations):
            shuffled_signal = np.random.permutation(signal_arr)
            sim_long = returns_arr[shuffled_signal == 1]
            sim_other = returns_arr[shuffled_signal == 0]
            if len(sim_long) > 0 and len(sim_other) > 0:
                sim_spread = sim_long.mean() - sim_other.mean()
                simulated_spreads.append(sim_spread)
        
        simulated_spreads = np.array(simulated_spreads)
        
        # P-value: proportion of simulated spreads >= actual
        p_value = (simulated_spreads >= actual_spread).mean()
        
        # 95% confidence interval
        ci_lower = np.percentile(simulated_spreads, 2.5)
        ci_upper = np.percentile(simulated_spreads, 97.5)
        
        return {
            'pass': p_value < 0.05,
            'actual_spread': actual_spread * (252 / hold_period),
            'p_value': p_value,
            'ci_lower': ci_lower * (252 / hold_period),
            'ci_upper': ci_upper * (252 / hold_period),
            'n_simulations': n_simulations,
            'percentile': (simulated_spreads < actual_spread).mean() * 100,
        }
    
    def timeframe_analysis(self, hyp_id: str) -> dict:
        """Test across different time periods."""
        periods = [
            ('2010-2015', '2010-01-01', '2015-01-01'),
            ('2015-2020', '2015-01-01', '2020-01-01'),
            ('2020-2025', '2020-01-01', '2025-01-01'),
        ]
        
        data = self.data['SPY'].copy()
        signal = self.generate_signal(hyp_id, data)
        
        hold_period = 21
        forward_ret = data['close'].pct_change(hold_period).shift(-hold_period)
        
        results = []
        for name, start, end in periods:
            mask = (data.index >= start) & (data.index < end)
            period_signal = signal[mask]
            period_returns = forward_ret[mask]
            
            long_ret = period_returns[period_signal == 1].dropna()
            other_ret = period_returns[period_signal == 0].dropna()
            
            if len(long_ret) >= 10:
                spread = (long_ret.mean() - other_ret.mean()) * (252 / hold_period) if len(other_ret) > 0 else long_ret.mean() * (252 / hold_period)
                results.append({
                    'period': name,
                    'spread': spread,
                    'n_signals': len(long_ret),
                    'positive': spread > 0,
                })
        
        if not results:
            return {'pass': False, 'reason': 'Insufficient data'}
        
        df = pd.DataFrame(results)
        positive_rate = df['positive'].mean()
        
        return {
            'pass': positive_rate >= 0.67,  # Works in at least 2/3 periods
            'periods': results,
            'positive_rate': positive_rate,
            'avg_spread': df['spread'].mean(),
        }
    
    def regime_analysis(self, hyp_id: str) -> dict:
        """Test in different market regimes."""
        data = self.data['SPY'].copy()
        signal = self.generate_signal(hyp_id, data)
        
        # Define regimes
        returns_200d = data['close'].pct_change(200)
        vol_21d = data['close'].pct_change().rolling(21).std() * np.sqrt(252)
        
        # Bull: positive 200d return, normal vol
        bull = (returns_200d > 0.1) & (vol_21d < 0.2)
        # Bear: negative 200d return
        bear = returns_200d < -0.1
        # High Vol: vol > 25%
        high_vol = vol_21d > 0.25
        # Sideways: everything else
        sideways = ~bull & ~bear & ~high_vol
        
        hold_period = 21
        forward_ret = data['close'].pct_change(hold_period).shift(-hold_period)
        
        results = []
        for regime_name, regime_mask in [('Bull', bull), ('Bear', bear), ('High Vol', high_vol), ('Sideways', sideways)]:
            regime_signal = signal[regime_mask]
            regime_returns = forward_ret[regime_mask]
            
            long_ret = regime_returns[regime_signal == 1].dropna()
            other_ret = regime_returns[regime_signal == 0].dropna()
            
            if len(long_ret) >= 10:
                spread = (long_ret.mean() - other_ret.mean()) * (252 / hold_period) if len(other_ret) > 0 else long_ret.mean() * (252 / hold_period)
                results.append({
                    'regime': regime_name,
                    'spread': spread,
                    'n_signals': len(long_ret),
                    'positive': spread > 0,
                })
        
        if not results:
            return {'pass': False, 'reason': 'Insufficient data'}
        
        df = pd.DataFrame(results)
        
        return {
            'pass': df['positive'].mean() >= 0.5,  # Works in at least half of regimes
            'regimes': results,
            'positive_rate': df['positive'].mean(),
            'best_regime': df.loc[df['spread'].idxmax(), 'regime'] if len(df) > 0 else 'N/A',
        }
    
    def transaction_cost_sensitivity(self, hyp_id: str) -> dict:
        """Test sensitivity to transaction costs."""
        data = self.data['SPY'].copy()
        signal = self.generate_signal(hyp_id, data)
        
        hold_period = 21
        forward_ret = data['close'].pct_change(hold_period).shift(-hold_period)
        
        # Calculate base spread
        long_ret = forward_ret[signal == 1].dropna()
        other_ret = forward_ret[signal == 0].dropna()
        
        if len(long_ret) < 20:
            return {'pass': False, 'reason': 'Insufficient signals'}
        
        base_spread = (long_ret.mean() - other_ret.mean()) * (252 / hold_period) if len(other_ret) > 0 else long_ret.mean() * (252 / hold_period)
        
        # Test different cost levels (round-trip costs)
        costs = [0, 10, 20, 30, 40, 50]  # bps
        results = []
        
        # Estimate trades per year
        signal_changes = signal.diff().abs().sum()
        trades_per_year = signal_changes / (len(data) / 252) * 2  # Entry + exit
        
        for cost_bps in costs:
            cost_drag = trades_per_year * cost_bps / 10000
            net_spread = base_spread - cost_drag
            results.append({
                'cost_bps': cost_bps,
                'net_spread': net_spread,
                'profitable': net_spread > 0,
            })
        
        df = pd.DataFrame(results)
        max_profitable_cost = df[df['profitable']]['cost_bps'].max() if df['profitable'].any() else 0
        
        return {
            'pass': max_profitable_cost >= 20,  # Still profitable at 20bps round-trip
            'base_spread': base_spread,
            'trades_per_year': trades_per_year,
            'max_profitable_cost_bps': max_profitable_cost,
            'cost_analysis': results,
        }
    
    def holdout_test(self, hyp_id: str, holdout_years: int = 2) -> dict:
        """Pure out-of-sample test on last N years."""
        data = self.data['SPY'].copy()
        signal = self.generate_signal(hyp_id, data)
        
        cutoff_date = data.index.max() - pd.DateOffset(years=holdout_years)
        
        # Train period
        train_data = data[data.index < cutoff_date]
        train_signal = signal[signal.index < cutoff_date]
        
        # Test period (holdout)
        test_data = data[data.index >= cutoff_date]
        test_signal = signal[signal.index >= cutoff_date]
        
        hold_period = 21
        
        # Train metrics
        train_ret = train_data['close'].pct_change(hold_period).shift(-hold_period)
        train_long = train_ret[train_signal == 1].dropna()
        train_other = train_ret[train_signal == 0].dropna()
        
        # Test metrics
        test_ret = test_data['close'].pct_change(hold_period).shift(-hold_period)
        test_long = test_ret[test_signal == 1].dropna()
        test_other = test_ret[test_signal == 0].dropna()
        
        if len(train_long) < 20 or len(test_long) < 5:
            return {'pass': False, 'reason': 'Insufficient data'}
        
        train_spread = (train_long.mean() - train_other.mean()) * (252 / hold_period) if len(train_other) > 0 else train_long.mean() * (252 / hold_period)
        test_spread = (test_long.mean() - test_other.mean()) * (252 / hold_period) if len(test_other) > 0 else test_long.mean() * (252 / hold_period)
        
        # Decay ratio
        decay = 1 - (test_spread / train_spread) if train_spread != 0 else 1
        
        return {
            'pass': test_spread > 0 and decay < 0.7,  # Positive OOS and less than 70% decay
            'train_spread': train_spread,
            'test_spread': test_spread,
            'decay_ratio': decay,
            'train_n': len(train_long),
            'test_n': len(test_long),
            'holdout_years': holdout_years,
        }
    
    def run_full_gauntlet(self, hyp_id: str, hyp_name: str) -> dict:
        """Run all validation tests for a hypothesis."""
        print(f"\n{'='*60}")
        print(f"[{hyp_id}] {hyp_name}")
        print(f"{'='*60}")
        
        results = {
            'id': hyp_id,
            'name': hyp_name,
            'tests': {},
            'passed': 0,
            'total': 6,
        }
        
        # 1. Walk-Forward
        print("  1. Walk-Forward Validation...", end=" ")
        wf = self.walk_forward_validation(hyp_id)
        results['tests']['walk_forward'] = wf
        status = "✓" if wf.get('pass', False) else "✗"
        print(f"{status} (OOS: {wf.get('avg_oos_spread', 0):.1%})")
        if wf.get('pass'): results['passed'] += 1
        
        # 2. Monte Carlo
        print("  2. Monte Carlo (5000 sims)...", end=" ")
        mc = self.monte_carlo_test(hyp_id)
        results['tests']['monte_carlo'] = mc
        status = "✓" if mc.get('pass', False) else "✗"
        print(f"{status} (p={mc.get('p_value', 1):.4f})")
        if mc.get('pass'): results['passed'] += 1
        
        # 3. Timeframe Analysis
        print("  3. Timeframe Analysis...", end=" ")
        tf = self.timeframe_analysis(hyp_id)
        results['tests']['timeframe'] = tf
        status = "✓" if tf.get('pass', False) else "✗"
        print(f"{status} ({tf.get('positive_rate', 0)*100:.0f}% periods)")
        if tf.get('pass'): results['passed'] += 1
        
        # 4. Regime Analysis
        print("  4. Regime Analysis...", end=" ")
        reg = self.regime_analysis(hyp_id)
        results['tests']['regime'] = reg
        status = "✓" if reg.get('pass', False) else "✗"
        print(f"{status} (Best: {reg.get('best_regime', 'N/A')})")
        if reg.get('pass'): results['passed'] += 1
        
        # 5. Transaction Costs
        print("  5. Transaction Cost Test...", end=" ")
        tc = self.transaction_cost_sensitivity(hyp_id)
        results['tests']['transaction_costs'] = tc
        status = "✓" if tc.get('pass', False) else "✗"
        print(f"{status} (Max: {tc.get('max_profitable_cost_bps', 0)}bps)")
        if tc.get('pass'): results['passed'] += 1
        
        # 6. Holdout Test
        print("  6. Holdout Test (2yr OOS)...", end=" ")
        ho = self.holdout_test(hyp_id)
        results['tests']['holdout'] = ho
        status = "✓" if ho.get('pass', False) else "✗"
        print(f"{status} (OOS: {ho.get('test_spread', 0):.1%})")
        if ho.get('pass'): results['passed'] += 1
        
        # Summary
        results['pass_rate'] = results['passed'] / results['total']
        results['grade'] = self._calculate_grade(results['passed'])
        
        print(f"\n  📊 RESULT: {results['passed']}/{results['total']} tests passed → Grade: {results['grade']}")
        
        return results
    
    def _calculate_grade(self, passed: int) -> str:
        """Calculate grade based on tests passed."""
        if passed == 6:
            return "A+ (Production Ready)"
        elif passed == 5:
            return "A (Strong Edge)"
        elif passed == 4:
            return "B (Good Edge)"
        elif passed == 3:
            return "C (Marginal Edge)"
        elif passed == 2:
            return "D (Weak Edge)"
        else:
            return "F (No Edge)"
    
    def run_all_winners(self):
        """Run gauntlet on all winners."""
        print("\n" + "="*70)
        print("WINNER VALIDATION GAUNTLET - FULL TEST SUITE")
        print("="*70)
        print(f"Testing {len(WINNERS)} winning hypotheses")
        print("6 tests each: Walk-Forward, Monte Carlo, Timeframe, Regime, Costs, Holdout")
        
        self.download_data()
        
        all_results = []
        
        for winner in WINNERS:
            result = self.run_full_gauntlet(winner['id'], winner['name'])
            result['original_spread'] = winner['spread']
            result['category'] = winner['category']
            all_results.append(result)
        
        # Save detailed results
        pd.to_pickle(all_results, self.cache_path / 'gauntlet_detailed_results.pkl')
        
        # Create summary
        summary = pd.DataFrame([{
            'id': r['id'],
            'name': r['name'],
            'category': r['category'],
            'original_spread': r['original_spread'],
            'tests_passed': r['passed'],
            'pass_rate': r['pass_rate'],
            'grade': r['grade'],
            'wf_pass': r['tests']['walk_forward'].get('pass', False),
            'mc_pass': r['tests']['monte_carlo'].get('pass', False),
            'tf_pass': r['tests']['timeframe'].get('pass', False),
            'regime_pass': r['tests']['regime'].get('pass', False),
            'cost_pass': r['tests']['transaction_costs'].get('pass', False),
            'holdout_pass': r['tests']['holdout'].get('pass', False),
            'mc_pvalue': r['tests']['monte_carlo'].get('p_value', 1),
            'holdout_spread': r['tests']['holdout'].get('test_spread', 0),
        } for r in all_results])
        
        summary = summary.sort_values('tests_passed', ascending=False)
        summary.to_csv(self.cache_path / 'GAUNTLET_RESULTS.csv', index=False)
        
        # Print final summary
        print("\n" + "="*70)
        print("FINAL GAUNTLET RESULTS")
        print("="*70)
        
        grade_counts = summary['grade'].value_counts()
        for grade, count in grade_counts.items():
            print(f"  {grade}: {count}")
        
        print(f"\n🏆 PRODUCTION READY (A+ Grade):")
        a_plus = summary[summary['tests_passed'] == 6]
        if len(a_plus) > 0:
            for _, row in a_plus.iterrows():
                print(f"  {row['id']}: {row['name']} (OOS: {row['holdout_spread']:.1%})")
        else:
            print("  None achieved A+ (all 6 tests)")
        
        print(f"\n⭐ STRONG EDGES (A Grade - 5/6 tests):")
        a_grade = summary[summary['tests_passed'] == 5]
        for _, row in a_grade.iterrows():
            print(f"  {row['id']}: {row['name']} (OOS: {row['holdout_spread']:.1%})")
        
        print(f"\n✓ GOOD EDGES (B Grade - 4/6 tests):")
        b_grade = summary[summary['tests_passed'] == 4]
        for _, row in b_grade.iterrows():
            print(f"  {row['id']}: {row['name']} (OOS: {row['holdout_spread']:.1%})")
        
        print(f"\n📊 Results saved to: {self.cache_path / 'GAUNTLET_RESULTS.csv'}")
        
        return summary


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    gauntlet = WinnerValidationGauntlet()
    results = gauntlet.run_all_winners()
