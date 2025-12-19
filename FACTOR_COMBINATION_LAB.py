#!/usr/bin/env python3
"""
FACTOR COMBINATION LAB
======================
Phase 3 of the Battle Plan: Build multi-factor combinations from vetted strategies.

Single edges get arbitraged. Combinations of uncorrelated edges create a moat.

This lab:
1. Creates signal vectors from all vetted strategies
2. Runs cross-sectional analysis (decile portfolios)
3. Performs PCA to find orthogonal meta-factors
4. Tests 2- and 3-factor combinations
5. Discovers models with higher risk-adjusted returns than single factors

Author: Quantum Trading Research Team
Date: December 20, 2025
"""

import sqlite3
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from tqdm import tqdm
import warnings
import pickle
import os

warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================

DB_PATH = 'data/market_data.db'
VETTED_CSV = 'data/VETTED_STRATEGIES.csv'
FULL_VALIDATION_CSV = 'data/FULL_VALIDATION_RESULTS.csv'
OUTPUT_DIR = 'data/factor_lab'

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Time splits (same as validation pipeline)
TRAIN_END = '2024-09-30'
TEST_START = '2024-10-01'


# ============================================================
# CORE FACTOR DEFINITIONS
# ============================================================

class CoreFactors:
    """
    Core factor definitions that are proven robust.
    These are the building blocks for multi-factor models.
    """
    
    @staticmethod
    def calculate_all_factors(df):
        """
        Calculate all core factors on the dataframe.
        Returns df with factor columns added.
        """
        df = df.copy()
        
        # ====================================================
        # FACTOR 1: RSI (Relative Strength Index)
        # ====================================================
        df['_delta'] = df.groupby('ticker')['close'].diff()
        df['_gain'] = df['_delta'].where(df['_delta'] > 0, 0)
        df['_loss'] = (-df['_delta']).where(df['_delta'] < 0, 0)
        df['_avg_gain'] = df.groupby('ticker')['_gain'].transform(lambda x: x.rolling(14).mean())
        df['_avg_loss'] = df.groupby('ticker')['_loss'].transform(lambda x: x.rolling(14).mean())
        df['_rs'] = df['_avg_gain'] / df['_avg_loss'].replace(0, np.nan)
        df['rsi'] = 100 - (100 / (1 + df['_rs']))
        df['rsi_oversold'] = (df['rsi'] < 30).astype(float)
        
        # ====================================================
        # FACTOR 2: Volatility (ATR-based)
        # ====================================================
        df['_tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df.groupby('ticker')['close'].shift(1)),
                abs(df['low'] - df.groupby('ticker')['close'].shift(1))
            )
        )
        df['atr'] = df.groupby('ticker')['_tr'].transform(lambda x: x.rolling(14).mean())
        df['atr_pct'] = df['atr'] / df['close']
        df['low_volatility'] = df.groupby('date')['atr_pct'].transform(
            lambda x: (x.rank(pct=True) < 0.3).astype(float)
        )
        
        # ====================================================
        # FACTOR 3: Momentum (20-day)
        # ====================================================
        df['momentum_20'] = df.groupby('ticker')['close'].transform(lambda x: x.pct_change(20))
        df['momentum_positive'] = (df['momentum_20'] > 0).astype(float)
        df['momentum_rank'] = df.groupby('date')['momentum_20'].transform(
            lambda x: x.rank(pct=True)
        )
        
        # ====================================================
        # FACTOR 4: Trend (EMA 200)
        # ====================================================
        df['ema_200'] = df.groupby('ticker')['close'].transform(lambda x: x.ewm(span=200).mean())
        df['above_ema200'] = (df['close'] > df['ema_200']).astype(float)
        
        # ====================================================
        # FACTOR 5: Mean Reversion (Z-score)
        # ====================================================
        df['sma_20'] = df.groupby('ticker')['close'].transform(lambda x: x.rolling(20).mean())
        df['std_20'] = df.groupby('ticker')['close'].transform(lambda x: x.rolling(20).std())
        df['zscore'] = (df['close'] - df['sma_20']) / df['std_20'].replace(0, np.nan)
        df['oversold_zscore'] = (df['zscore'] < -2).astype(float)
        
        # ====================================================
        # FACTOR 6: Volume (relative)
        # ====================================================
        df['vol_ma'] = df.groupby('ticker')['volume'].transform(lambda x: x.rolling(20).mean())
        df['vol_ratio'] = df['volume'] / df['vol_ma'].replace(0, np.nan)
        df['volume_spike'] = (df['vol_ratio'] > 2.0).astype(float)
        df['low_volume'] = (df['vol_ratio'] < 0.5).astype(float)
        
        # ====================================================
        # FACTOR 7: Consecutive Days
        # ====================================================
        df['daily_return'] = df.groupby('ticker')['close'].pct_change()
        df['_prev_ret1'] = df.groupby('ticker')['daily_return'].shift(1)
        df['_prev_ret2'] = df.groupby('ticker')['daily_return'].shift(2)
        df['after_2down'] = ((df['daily_return'] < 0) & (df['_prev_ret1'] < 0)).astype(float)
        df['after_2up'] = ((df['daily_return'] > 0) & (df['_prev_ret1'] > 0)).astype(float)
        
        # ====================================================
        # FACTOR 8: 52-Week High Proximity
        # ====================================================
        df['high_52w'] = df.groupby('ticker')['high'].transform(lambda x: x.rolling(252).max())
        df['pct_from_high'] = (df['high_52w'] - df['close']) / df['high_52w']
        df['near_52w_high'] = (df['pct_from_high'] < 0.05).astype(float)
        
        # ====================================================
        # FACTOR 9: Bollinger Band Position
        # ====================================================
        df['bb_upper'] = df['sma_20'] + 2 * df['std_20']
        df['bb_lower'] = df['sma_20'] - 2 * df['std_20']
        df['below_bb_lower'] = (df['close'] < df['bb_lower']).astype(float)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['sma_20']
        df['bb_squeeze'] = df.groupby('date')['bb_width'].transform(
            lambda x: (x.rank(pct=True) < 0.1).astype(float)
        )
        
        # Clean up temporary columns
        temp_cols = [c for c in df.columns if c.startswith('_')]
        df = df.drop(columns=temp_cols)
        
        return df
    
    @staticmethod
    def get_factor_columns():
        """Return list of factor column names"""
        return [
            'rsi', 'rsi_oversold',
            'atr_pct', 'low_volatility',
            'momentum_20', 'momentum_positive', 'momentum_rank',
            'above_ema200',
            'zscore', 'oversold_zscore',
            'vol_ratio', 'volume_spike', 'low_volume',
            'daily_return', 'after_2down', 'after_2up',
            'pct_from_high', 'near_52w_high',
            'below_bb_lower', 'bb_width', 'bb_squeeze'
        ]
    
    @staticmethod
    def get_signal_factors():
        """Return list of binary signal factors (used for combinations)"""
        return [
            'rsi_oversold',
            'low_volatility',
            'momentum_positive',
            'above_ema200',
            'oversold_zscore',
            'volume_spike',
            'low_volume',
            'after_2down',
            'after_2up',
            'near_52w_high',
            'below_bb_lower',
            'bb_squeeze'
        ]


# ============================================================
# FACTOR COMBINATION LAB
# ============================================================

class FactorCombinationLab:
    """
    Laboratory for testing factor combinations.
    
    Tests 2-factor and 3-factor combinations to find
    models with higher risk-adjusted returns than single factors.
    """
    
    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path
        self.df = None
        self.train_df = None
        self.test_df = None
        self.factor_correlations = None
        self.pca_results = None
        self.combination_results = []
        
    def load_and_prepare_data(self):
        """Load market data and calculate all factors"""
        print("="*60)
        print("FACTOR COMBINATION LAB")
        print("="*60)
        
        print("\n📊 Loading market data...")
        conn = sqlite3.connect(self.db_path)
        self.df = pd.read_sql("SELECT * FROM ohlcv", conn)
        conn.close()
        
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df = self.df.sort_values(['ticker', 'date']).reset_index(drop=True)
        
        print(f"  Records: {len(self.df):,}")
        print(f"  Tickers: {self.df['ticker'].nunique():,}")
        
        print("\n🔧 Calculating all factors...")
        self.df = CoreFactors.calculate_all_factors(self.df)
        
        # Calculate forward returns for multiple horizons
        for h in [1, 5, 10, 20]:
            self.df[f'fwd_{h}'] = self.df.groupby('ticker')['close'].transform(
                lambda x: x.shift(-h) / x - 1
            )
        
        # Split into train/test
        train_end = pd.Timestamp(TRAIN_END)
        test_start = pd.Timestamp(TEST_START)
        
        self.train_df = self.df[self.df['date'] <= train_end].copy()
        self.test_df = self.df[self.df['date'] >= test_start].copy()
        
        print(f"\n📈 Train period: {self.train_df['date'].min()} to {self.train_df['date'].max()}")
        print(f"   Records: {len(self.train_df):,}")
        print(f"📉 Test period: {self.test_df['date'].min()} to {self.test_df['date'].max()}")
        print(f"   Records: {len(self.test_df):,}")
        
    def analyze_factor_correlations(self):
        """Analyze correlations between factors to find uncorrelated pairs"""
        print("\n" + "="*60)
        print("FACTOR CORRELATION ANALYSIS")
        print("="*60)
        
        signal_factors = CoreFactors.get_signal_factors()
        
        # Calculate correlations on training data
        factor_data = self.train_df[signal_factors].dropna()
        self.factor_correlations = factor_data.corr()
        
        print("\n📊 Factor Correlation Matrix (subset):")
        print(self.factor_correlations.round(2).to_string())
        
        # Find most uncorrelated pairs
        print("\n🔍 Most Uncorrelated Factor Pairs:")
        pairs = []
        for i, f1 in enumerate(signal_factors):
            for f2 in signal_factors[i+1:]:
                corr = abs(self.factor_correlations.loc[f1, f2])
                pairs.append((f1, f2, corr))
        
        pairs.sort(key=lambda x: x[2])
        
        for f1, f2, corr in pairs[:10]:
            print(f"  {f1} + {f2}: corr = {corr:.3f}")
        
        self.uncorrelated_pairs = pairs[:10]
        
        # Save correlation matrix
        self.factor_correlations.to_csv(f'{OUTPUT_DIR}/factor_correlations.csv')
        
        return pairs
    
    def run_pca_analysis(self, n_components=5):
        """Run PCA to find orthogonal meta-factors"""
        print("\n" + "="*60)
        print("PCA: ORTHOGONAL META-FACTORS")
        print("="*60)
        
        signal_factors = CoreFactors.get_signal_factors()
        
        # Prepare data
        factor_data = self.train_df[signal_factors].dropna()
        
        # Standardize
        scaler = StandardScaler()
        factor_scaled = scaler.fit_transform(factor_data)
        
        # PCA
        pca = PCA(n_components=n_components)
        pca_transformed = pca.fit_transform(factor_scaled)
        
        # Explained variance
        print("\n📊 Explained Variance by Component:")
        for i, var in enumerate(pca.explained_variance_ratio_):
            cumvar = pca.explained_variance_ratio_[:i+1].sum()
            print(f"  PC{i+1}: {var*100:.1f}% (cumulative: {cumvar*100:.1f}%)")
        
        # Component loadings
        print("\n📊 Component Loadings (top factors per PC):")
        loadings = pd.DataFrame(
            pca.components_.T,
            columns=[f'PC{i+1}' for i in range(n_components)],
            index=signal_factors
        )
        
        for pc in loadings.columns[:3]:  # Show first 3 PCs
            print(f"\n  {pc}:")
            sorted_loadings = loadings[pc].abs().sort_values(ascending=False)
            for factor in sorted_loadings.head(5).index:
                val = loadings.loc[factor, pc]
                print(f"    {factor}: {val:+.3f}")
        
        self.pca_results = {
            'pca': pca,
            'scaler': scaler,
            'loadings': loadings,
            'transformed': pca_transformed
        }
        
        # Save PCA results
        loadings.to_csv(f'{OUTPUT_DIR}/pca_loadings.csv')
        
        return loadings
    
    def calc_factor_returns(self, df, factor_col, fwd_col='fwd_10'):
        """Calculate returns when factor is active"""
        active = df[df[factor_col] == 1][fwd_col].dropna()
        inactive = df[df[factor_col] == 0][fwd_col].dropna()
        
        if len(active) < 100 or len(inactive) < 100:
            return None
        
        active_mean = active.mean()
        active_std = active.std()
        active_n = len(active)
        active_t = active_mean / (active_std / np.sqrt(active_n)) if active_std > 0 else 0
        active_sharpe = active_mean / active_std * np.sqrt(252/10) if active_std > 0 else 0
        
        return {
            'mean': active_mean,
            'std': active_std,
            'n': active_n,
            't_stat': active_t,
            'sharpe': active_sharpe,
            'spread': active_mean - inactive.mean()
        }
    
    def run_decile_analysis(self, factor_col, fwd_col='fwd_10'):
        """Run cross-sectional decile analysis for a continuous factor"""
        print(f"\n  Decile analysis for {factor_col}...")
        
        df = self.test_df.copy()
        
        # Rank stocks by factor each day
        df['factor_decile'] = df.groupby('date')[factor_col].transform(
            lambda x: pd.qcut(x, 10, labels=False, duplicates='drop') + 1 if len(x.dropna()) >= 10 else np.nan
        )
        
        # Calculate returns by decile
        decile_returns = df.groupby('factor_decile')[fwd_col].agg(['mean', 'std', 'count'])
        decile_returns['sharpe'] = decile_returns['mean'] / decile_returns['std'] * np.sqrt(252/10)
        
        # Long-short spread
        if 1 in decile_returns.index and 10 in decile_returns.index:
            spread = decile_returns.loc[10, 'mean'] - decile_returns.loc[1, 'mean']
            print(f"    Spread (D10 - D1): {spread*100:.3f}%")
        
        return decile_returns
    
    def test_single_factors(self):
        """Test all single factors individually"""
        print("\n" + "="*60)
        print("SINGLE FACTOR PERFORMANCE")
        print("="*60)
        
        signal_factors = CoreFactors.get_signal_factors()
        
        results = []
        for factor in signal_factors:
            # Train performance
            train_stats = self.calc_factor_returns(self.train_df, factor, 'fwd_10')
            test_stats = self.calc_factor_returns(self.test_df, factor, 'fwd_10')
            
            if train_stats and test_stats:
                results.append({
                    'factor': factor,
                    'train_mean': train_stats['mean'],
                    'train_t': train_stats['t_stat'],
                    'train_sharpe': train_stats['sharpe'],
                    'train_n': train_stats['n'],
                    'test_mean': test_stats['mean'],
                    'test_t': test_stats['t_stat'],
                    'test_sharpe': test_stats['sharpe'],
                    'test_n': test_stats['n'],
                    'degradation': 1 - (test_stats['t_stat'] / train_stats['t_stat']) if train_stats['t_stat'] != 0 else np.nan
                })
        
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('test_sharpe', ascending=False)
        
        print("\n📊 Single Factor Results (sorted by OOS Sharpe):")
        print(results_df[['factor', 'train_t', 'test_t', 'test_sharpe', 'degradation']].to_string(index=False))
        
        self.single_factor_results = results_df
        results_df.to_csv(f'{OUTPUT_DIR}/single_factor_results.csv', index=False)
        
        return results_df
    
    def test_two_factor_combinations(self, n_hold=10):
        """Test all 2-factor combinations"""
        print("\n" + "="*60)
        print("TWO-FACTOR COMBINATION TESTING")
        print("="*60)
        
        signal_factors = CoreFactors.get_signal_factors()
        fwd_col = f'fwd_{n_hold}'
        
        results = []
        total = len(signal_factors) * (len(signal_factors) - 1) // 2
        
        with tqdm(total=total, desc="Testing 2F combos") as pbar:
            for i, f1 in enumerate(signal_factors):
                for f2 in signal_factors[i+1:]:
                    # Create combined signal (both factors active)
                    train_mask = (self.train_df[f1] == 1) & (self.train_df[f2] == 1)
                    test_mask = (self.test_df[f1] == 1) & (self.test_df[f2] == 1)
                    
                    train_returns = self.train_df.loc[train_mask, fwd_col].dropna()
                    test_returns = self.test_df.loc[test_mask, fwd_col].dropna()
                    
                    if len(train_returns) >= 100 and len(test_returns) >= 100:
                        # Train stats
                        train_mean = train_returns.mean()
                        train_std = train_returns.std()
                        train_t = train_mean / (train_std / np.sqrt(len(train_returns))) if train_std > 0 else 0
                        train_sharpe = train_mean / train_std * np.sqrt(252/n_hold) if train_std > 0 else 0
                        
                        # Test stats
                        test_mean = test_returns.mean()
                        test_std = test_returns.std()
                        test_t = test_mean / (test_std / np.sqrt(len(test_returns))) if test_std > 0 else 0
                        test_sharpe = test_mean / test_std * np.sqrt(252/n_hold) if test_std > 0 else 0
                        
                        # Correlation between factors
                        corr = abs(self.factor_correlations.loc[f1, f2])
                        
                        results.append({
                            'factor_1': f1,
                            'factor_2': f2,
                            'correlation': corr,
                            'train_mean': train_mean,
                            'train_n': len(train_returns),
                            'train_t': train_t,
                            'train_sharpe': train_sharpe,
                            'test_mean': test_mean,
                            'test_n': len(test_returns),
                            'test_t': test_t,
                            'test_sharpe': test_sharpe,
                            'degradation': 1 - (test_t / train_t) if train_t != 0 else np.nan
                        })
                    
                    pbar.update(1)
        
        results_df = pd.DataFrame(results)
        
        # Filter for robust combinations
        robust = results_df[
            (results_df['test_t'] >= 3.0) &
            (results_df['degradation'] < 0.5)
        ].sort_values('test_sharpe', ascending=False)
        
        print(f"\n✅ Total 2F combinations tested: {len(results_df)}")
        print(f"✅ Robust combinations (OOS t>3, deg<50%): {len(robust)}")
        
        print("\n📊 TOP 20 TWO-FACTOR COMBINATIONS:")
        print(robust.head(20)[['factor_1', 'factor_2', 'correlation', 'train_t', 'test_t', 'test_sharpe', 'degradation']].to_string(index=False))
        
        self.two_factor_results = results_df
        results_df.to_csv(f'{OUTPUT_DIR}/two_factor_results.csv', index=False)
        robust.to_csv(f'{OUTPUT_DIR}/two_factor_robust.csv', index=False)
        
        return robust
    
    def test_three_factor_combinations(self, n_hold=10, top_factors=8):
        """Test top 3-factor combinations"""
        print("\n" + "="*60)
        print("THREE-FACTOR COMBINATION TESTING")
        print("="*60)
        
        # Use only top factors to limit combinations
        if hasattr(self, 'single_factor_results'):
            top = self.single_factor_results.head(top_factors)['factor'].tolist()
        else:
            top = CoreFactors.get_signal_factors()[:top_factors]
        
        print(f"  Using top {len(top)} factors: {top}")
        
        fwd_col = f'fwd_{n_hold}'
        results = []
        
        from itertools import combinations
        combos = list(combinations(top, 3))
        
        with tqdm(total=len(combos), desc="Testing 3F combos") as pbar:
            for (f1, f2, f3) in combos:
                # Create combined signal (all three factors active)
                train_mask = (self.train_df[f1] == 1) & (self.train_df[f2] == 1) & (self.train_df[f3] == 1)
                test_mask = (self.test_df[f1] == 1) & (self.test_df[f2] == 1) & (self.test_df[f3] == 1)
                
                train_returns = self.train_df.loc[train_mask, fwd_col].dropna()
                test_returns = self.test_df.loc[test_mask, fwd_col].dropna()
                
                if len(train_returns) >= 50 and len(test_returns) >= 50:  # Lower threshold for 3F
                    # Train stats
                    train_mean = train_returns.mean()
                    train_std = train_returns.std()
                    train_t = train_mean / (train_std / np.sqrt(len(train_returns))) if train_std > 0 else 0
                    train_sharpe = train_mean / train_std * np.sqrt(252/n_hold) if train_std > 0 else 0
                    
                    # Test stats
                    test_mean = test_returns.mean()
                    test_std = test_returns.std()
                    test_t = test_mean / (test_std / np.sqrt(len(test_returns))) if test_std > 0 else 0
                    test_sharpe = test_mean / test_std * np.sqrt(252/n_hold) if test_std > 0 else 0
                    
                    results.append({
                        'factor_1': f1,
                        'factor_2': f2,
                        'factor_3': f3,
                        'train_mean': train_mean,
                        'train_n': len(train_returns),
                        'train_t': train_t,
                        'train_sharpe': train_sharpe,
                        'test_mean': test_mean,
                        'test_n': len(test_returns),
                        'test_t': test_t,
                        'test_sharpe': test_sharpe,
                        'degradation': 1 - (test_t / train_t) if train_t != 0 else np.nan
                    })
                
                pbar.update(1)
        
        results_df = pd.DataFrame(results)
        
        if len(results_df) > 0:
            # Filter for robust combinations
            robust = results_df[
                (results_df['test_t'] >= 2.5) &  # Slightly lower threshold for 3F
                (results_df['degradation'] < 0.5)
            ].sort_values('test_sharpe', ascending=False)
            
            print(f"\n✅ Total 3F combinations tested: {len(results_df)}")
            print(f"✅ Robust combinations: {len(robust)}")
            
            if len(robust) > 0:
                print("\n📊 TOP 10 THREE-FACTOR COMBINATIONS:")
                print(robust.head(10)[['factor_1', 'factor_2', 'factor_3', 'train_t', 'test_t', 'test_sharpe']].to_string(index=False))
            
            self.three_factor_results = results_df
            results_df.to_csv(f'{OUTPUT_DIR}/three_factor_results.csv', index=False)
            if len(robust) > 0:
                robust.to_csv(f'{OUTPUT_DIR}/three_factor_robust.csv', index=False)
            
            return robust
        
        return pd.DataFrame()
    
    def generate_final_models(self):
        """Generate final recommended multi-factor models"""
        print("\n" + "="*60)
        print("FINAL RECOMMENDED MULTI-FACTOR MODELS")
        print("="*60)
        
        models = []
        
        # Best single factors
        if hasattr(self, 'single_factor_results'):
            best_single = self.single_factor_results[
                (self.single_factor_results['test_t'] >= 3.0) &
                (self.single_factor_results['degradation'] < 0.5)
            ].head(3)
            
            for _, row in best_single.iterrows():
                models.append({
                    'model_name': f"SF_{row['factor']}",
                    'type': 'single_factor',
                    'factors': [row['factor']],
                    'test_sharpe': row['test_sharpe'],
                    'test_t': row['test_t'],
                    'degradation': row['degradation']
                })
        
        # Best 2-factor combos
        if hasattr(self, 'two_factor_results'):
            best_2f = self.two_factor_results[
                (self.two_factor_results['test_t'] >= 3.0) &
                (self.two_factor_results['degradation'] < 0.5)
            ].head(5)
            
            for _, row in best_2f.iterrows():
                models.append({
                    'model_name': f"2F_{row['factor_1']}_{row['factor_2']}",
                    'type': 'two_factor',
                    'factors': [row['factor_1'], row['factor_2']],
                    'test_sharpe': row['test_sharpe'],
                    'test_t': row['test_t'],
                    'degradation': row['degradation']
                })
        
        # Best 3-factor combos
        if hasattr(self, 'three_factor_results') and len(self.three_factor_results) > 0:
            best_3f = self.three_factor_results[
                (self.three_factor_results['test_t'] >= 2.5) &
                (self.three_factor_results['degradation'] < 0.5)
            ].head(3)
            
            for _, row in best_3f.iterrows():
                models.append({
                    'model_name': f"3F_{row['factor_1']}_{row['factor_2']}_{row['factor_3']}",
                    'type': 'three_factor',
                    'factors': [row['factor_1'], row['factor_2'], row['factor_3']],
                    'test_sharpe': row['test_sharpe'],
                    'test_t': row['test_t'],
                    'degradation': row['degradation']
                })
        
        models_df = pd.DataFrame(models)
        models_df = models_df.sort_values('test_sharpe', ascending=False)
        
        print("\n🏆 RECOMMENDED MODELS FOR PRODUCTION:")
        print(models_df.to_string(index=False))
        
        models_df.to_csv(f'{OUTPUT_DIR}/recommended_models.csv', index=False)
        
        self.recommended_models = models_df
        return models_df
    
    def run_full_analysis(self):
        """Run complete factor combination analysis"""
        self.load_and_prepare_data()
        self.analyze_factor_correlations()
        self.run_pca_analysis()
        self.test_single_factors()
        self.test_two_factor_combinations()
        self.test_three_factor_combinations()
        self.generate_final_models()
        
        print("\n" + "="*60)
        print("🎯 FACTOR COMBINATION LAB COMPLETE")
        print("="*60)
        print(f"\nResults saved to: {OUTPUT_DIR}/")


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == '__main__':
    lab = FactorCombinationLab()
    lab.run_full_analysis()
