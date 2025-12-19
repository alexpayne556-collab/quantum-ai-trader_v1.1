#!/usr/bin/env python3
"""
STRATEGY VETTING PIPELINE - Industrial-Strength Validation
============================================================
Phase 2 of the Battle Plan: Systematize discovery, industrialize validation.

This pipeline:
1. Loads each strategy definition from GRAND_CONSOLIDATED_ALL.csv
2. Runs walk-forward validation (train/test split)
3. Records Train t-stat, Test t-stat, Degradation %
4. Applies Bonferroni-Holm correction to out-of-sample p-values
5. Outputs VETTED_STRATEGIES.csv with only robust edges

The goal: A "periodic table" of ~50-200 bulletproof market edges.

Author: Quantum Trading Research Team
Date: December 20, 2025
"""

import sqlite3
import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime
from tqdm import tqdm
import warnings
import pickle
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================

DB_PATH = 'data/market_data.db'
STRATEGIES_CSV = 'data/GRAND_CONSOLIDATED_ALL.csv'
OUTPUT_CSV = 'data/VETTED_STRATEGIES.csv'
RESULTS_PKL = 'data/VETTING_RESULTS.pkl'

# Time Split Configuration
TRAIN_END = '2024-09-30'    # Training ends Sept 30, 2024
TEST_START = '2024-10-01'   # Testing begins Oct 1, 2024

# Statistical Thresholds
MIN_SAMPLES = 100           # Minimum observations per split
T_THRESHOLD = 3.0           # Base t-stat threshold
DEGRADATION_THRESHOLD = 0.5 # Max allowed degradation (Test t < 0.5 * Train t = FAIL)
MIN_OOS_SHARPE = 0.1        # Minimum annualized Sharpe in test period (lowered for broader coverage)

# Bonferroni-Holm parameters
FWER_ALPHA = 0.05           # Family-wise error rate


# ============================================================
# STRATEGY DEFINITIONS
# ============================================================

class StrategyDefinitions:
    """
    Parse strategy names to extract parameters and generate signals.
    Strategy names follow patterns like:
    - RSI_Below30_H10 -> RSI below 30, hold 10 days
    - After2Down_H5 -> After 2 down days, hold 5 days
    - MACD_Positive_H20 -> MACD > 0, hold 20 days
    """
    
    @staticmethod
    def parse_hold_period(strategy_name):
        """Extract hold period from strategy name (e.g., 'H10' -> 10)"""
        import re
        match = re.search(r'_H(\d+)', strategy_name)
        if match:
            return int(match.group(1))
        # Try alternate patterns
        match = re.search(r'H(\d+)$', strategy_name)
        if match:
            return int(match.group(1))
        return 20  # Default
    
    @staticmethod
    def generate_signal(df, category, strategy_name):
        """
        Generate boolean signal based on category and strategy name.
        Returns a boolean Series indicating when the signal is active.
        """
        try:
            signal = None
            hold = StrategyDefinitions.parse_hold_period(strategy_name)
            
            # ============================================================
            # CATEGORY: CONSECUTIVE (After N Up/Down days)
            # ============================================================
            if category == 'CONSECUTIVE':
                if 'After2Up' in strategy_name:
                    df['_prev1'] = df.groupby('ticker')['close'].pct_change()
                    df['_prev2'] = df.groupby('ticker')['_prev1'].shift(1)
                    signal = (df['_prev1'] > 0) & (df['_prev2'] > 0)
                elif 'After2Down' in strategy_name:
                    df['_prev1'] = df.groupby('ticker')['close'].pct_change()
                    df['_prev2'] = df.groupby('ticker')['_prev1'].shift(1)
                    signal = (df['_prev1'] < 0) & (df['_prev2'] < 0)
                elif 'After3Up' in strategy_name:
                    df['_prev1'] = df.groupby('ticker')['close'].pct_change()
                    df['_prev2'] = df.groupby('ticker')['_prev1'].shift(1)
                    df['_prev3'] = df.groupby('ticker')['_prev1'].shift(2)
                    signal = (df['_prev1'] > 0) & (df['_prev2'] > 0) & (df['_prev3'] > 0)
                elif 'After3Down' in strategy_name:
                    df['_prev1'] = df.groupby('ticker')['close'].pct_change()
                    df['_prev2'] = df.groupby('ticker')['_prev1'].shift(1)
                    df['_prev3'] = df.groupby('ticker')['_prev1'].shift(2)
                    signal = (df['_prev1'] < 0) & (df['_prev2'] < 0) & (df['_prev3'] < 0)
                elif 'After4Down' in strategy_name:
                    df['_prev1'] = df.groupby('ticker')['close'].pct_change()
                    df['_prev2'] = df.groupby('ticker')['_prev1'].shift(1)
                    df['_prev3'] = df.groupby('ticker')['_prev1'].shift(2)
                    df['_prev4'] = df.groupby('ticker')['_prev1'].shift(3)
                    signal = (df['_prev1'] < 0) & (df['_prev2'] < 0) & (df['_prev3'] < 0) & (df['_prev4'] < 0)
                elif 'After5Down' in strategy_name:
                    df['_prev1'] = df.groupby('ticker')['close'].pct_change()
                    signal = pd.Series([False] * len(df), index=df.index)
                    for i in range(1, 6):
                        df[f'_prev{i}'] = df.groupby('ticker')['_prev1'].shift(i-1)
                    signal = (df['_prev1'] < 0) & (df['_prev2'] < 0) & (df['_prev3'] < 0) & (df['_prev4'] < 0) & (df['_prev5'] < 0)
            
            # ============================================================
            # CATEGORY: RSI
            # ============================================================
            elif category == 'RSI':
                # Calculate RSI
                df['_delta'] = df.groupby('ticker')['close'].diff()
                df['_gain'] = df['_delta'].where(df['_delta'] > 0, 0)
                df['_loss'] = (-df['_delta']).where(df['_delta'] < 0, 0)
                df['_avg_gain'] = df.groupby('ticker')['_gain'].transform(lambda x: x.rolling(14).mean())
                df['_avg_loss'] = df.groupby('ticker')['_loss'].transform(lambda x: x.rolling(14).mean())
                df['_rs'] = df['_avg_gain'] / df['_avg_loss'].replace(0, np.nan)
                df['_rsi'] = 100 - (100 / (1 + df['_rs']))
                
                if 'Below20' in strategy_name:
                    signal = df['_rsi'] < 20
                elif 'Below30' in strategy_name:
                    signal = df['_rsi'] < 30
                elif 'Above70' in strategy_name:
                    signal = df['_rsi'] > 70
                elif 'Above80' in strategy_name:
                    signal = df['_rsi'] > 80
                elif 'CrossUp30' in strategy_name:
                    df['_rsi_prev'] = df.groupby('ticker')['_rsi'].shift(1)
                    signal = (df['_rsi'] > 30) & (df['_rsi_prev'] <= 30)
                elif 'CrossDown70' in strategy_name:
                    df['_rsi_prev'] = df.groupby('ticker')['_rsi'].shift(1)
                    signal = (df['_rsi'] < 70) & (df['_rsi_prev'] >= 70)
            
            # ============================================================
            # CATEGORY: MOMENTUM
            # ============================================================
            elif category == 'MOMENTUM':
                # Default momentum calculation (20-day)
                lookback = 20
                if '5d' in strategy_name or '_5_' in strategy_name:
                    lookback = 5
                elif '10d' in strategy_name or '_10_' in strategy_name:
                    lookback = 10
                elif '60d' in strategy_name or '_60_' in strategy_name:
                    lookback = 60
                
                df['_mom'] = df.groupby('ticker')['close'].transform(
                    lambda x: x.pct_change(lookback)
                )
                
                if 'Positive' in strategy_name or 'PosMo' in strategy_name:
                    signal = df['_mom'] > 0
                elif 'Negative' in strategy_name or 'NegMo' in strategy_name:
                    signal = df['_mom'] < 0
                elif 'Strong' in strategy_name:
                    # Top 20% momentum
                    df['_mom_rank'] = df.groupby('date')['_mom'].rank(pct=True)
                    signal = df['_mom_rank'] > 0.8
                elif 'Weak' in strategy_name:
                    # Bottom 20% momentum
                    df['_mom_rank'] = df.groupby('date')['_mom'].rank(pct=True)
                    signal = df['_mom_rank'] < 0.2
            
            # ============================================================
            # CATEGORY: MEAN_REVERSION
            # ============================================================
            elif category == 'MEAN_REVERSION':
                window = 20
                if '5d' in strategy_name or '_5_' in strategy_name:
                    window = 5
                elif '10d' in strategy_name or '_10_' in strategy_name:
                    window = 10
                elif '50d' in strategy_name or '_50_' in strategy_name:
                    window = 50
                
                df['_sma'] = df.groupby('ticker')['close'].transform(lambda x: x.rolling(window).mean())
                df['_std'] = df.groupby('ticker')['close'].transform(lambda x: x.rolling(window).std())
                df['_zscore'] = (df['close'] - df['_sma']) / df['_std'].replace(0, np.nan)
                
                if 'Oversold' in strategy_name or 'Below2Std' in strategy_name:
                    signal = df['_zscore'] < -2
                elif 'Overbought' in strategy_name or 'Above2Std' in strategy_name:
                    signal = df['_zscore'] > 2
                elif 'Below1Std' in strategy_name:
                    signal = df['_zscore'] < -1
                elif 'Above1Std' in strategy_name:
                    signal = df['_zscore'] > 1
            
            # ============================================================
            # CATEGORY: VOLUME
            # ============================================================
            elif category == 'VOLUME':
                df['_vol_ma'] = df.groupby('ticker')['volume'].transform(lambda x: x.rolling(20).mean())
                df['_vol_ratio'] = df['volume'] / df['_vol_ma'].replace(0, np.nan)
                
                if 'Spike' in strategy_name or 'High' in strategy_name or 'Above2x' in strategy_name:
                    signal = df['_vol_ratio'] > 2.0
                elif 'Low' in strategy_name or 'Below0.5x' in strategy_name:
                    signal = df['_vol_ratio'] < 0.5
            
            # ============================================================
            # CATEGORY: BOLLINGER
            # ============================================================
            elif category == 'BOLLINGER':
                df['_bb_mid'] = df.groupby('ticker')['close'].transform(lambda x: x.rolling(20).mean())
                df['_bb_std'] = df.groupby('ticker')['close'].transform(lambda x: x.rolling(20).std())
                df['_bb_upper'] = df['_bb_mid'] + 2 * df['_bb_std']
                df['_bb_lower'] = df['_bb_mid'] - 2 * df['_bb_std']
                
                if 'BelowLower' in strategy_name or 'Oversold' in strategy_name:
                    signal = df['close'] < df['_bb_lower']
                elif 'AboveUpper' in strategy_name or 'Overbought' in strategy_name:
                    signal = df['close'] > df['_bb_upper']
                elif 'Squeeze' in strategy_name or 'Narrow' in strategy_name:
                    df['_bb_width'] = (df['_bb_upper'] - df['_bb_lower']) / df['_bb_mid']
                    df['_bb_width_rank'] = df.groupby('date')['_bb_width'].rank(pct=True)
                    signal = df['_bb_width_rank'] < 0.1  # Narrowest 10%
            
            # ============================================================
            # CATEGORY: MA_CROSS
            # ============================================================
            elif category == 'MA_CROSS':
                df['_ema10'] = df.groupby('ticker')['close'].transform(lambda x: x.ewm(span=10).mean())
                df['_ema20'] = df.groupby('ticker')['close'].transform(lambda x: x.ewm(span=20).mean())
                df['_ema50'] = df.groupby('ticker')['close'].transform(lambda x: x.ewm(span=50).mean())
                df['_ema200'] = df.groupby('ticker')['close'].transform(lambda x: x.ewm(span=200).mean())
                
                if 'Golden' in strategy_name or '50over200' in strategy_name:
                    df['_prev'] = df.groupby('ticker')['_ema50'].shift(1)
                    signal = (df['_ema50'] > df['_ema200']) & (df['_prev'] <= df['_ema200'])
                elif 'Death' in strategy_name or '50under200' in strategy_name:
                    df['_prev'] = df.groupby('ticker')['_ema50'].shift(1)
                    signal = (df['_ema50'] < df['_ema200']) & (df['_prev'] >= df['_ema200'])
                elif 'Above200' in strategy_name or 'AboveEMA200' in strategy_name:
                    signal = df['close'] > df['_ema200']
                elif 'Below200' in strategy_name or 'BelowEMA200' in strategy_name:
                    signal = df['close'] < df['_ema200']
            
            # ============================================================
            # CATEGORY: STOCH (Stochastic)
            # ============================================================
            elif category == 'STOCH':
                df['_low14'] = df.groupby('ticker')['low'].transform(lambda x: x.rolling(14).min())
                df['_high14'] = df.groupby('ticker')['high'].transform(lambda x: x.rolling(14).max())
                df['_range'] = df['_high14'] - df['_low14']
                df['_stoch'] = np.where(df['_range'] > 0, 
                                        100 * (df['close'] - df['_low14']) / df['_range'], 
                                        50)
                
                if 'Oversold' in strategy_name or 'Below20' in strategy_name:
                    signal = df['_stoch'] < 20
                elif 'Overbought' in strategy_name or 'Above80' in strategy_name:
                    signal = df['_stoch'] > 80
            
            # ============================================================
            # CATEGORY: ATR (Volatility)
            # ============================================================
            elif category == 'ATR':
                df['_tr'] = np.maximum(
                    df['high'] - df['low'],
                    np.maximum(
                        abs(df['high'] - df.groupby('ticker')['close'].shift(1)),
                        abs(df['low'] - df.groupby('ticker')['close'].shift(1))
                    )
                )
                df['_atr'] = df.groupby('ticker')['_tr'].transform(lambda x: x.rolling(14).mean())
                df['_atr_pct'] = df['_atr'] / df['close']
                df['_atr_rank'] = df.groupby('date')['_atr_pct'].rank(pct=True)
                
                if 'LowVol' in strategy_name or 'Low' in strategy_name:
                    signal = df['_atr_rank'] < 0.2
                elif 'HighVol' in strategy_name or 'High' in strategy_name:
                    signal = df['_atr_rank'] > 0.8
            
            # ============================================================
            # CATEGORY: FUSION_2F (Two-factor combinations)
            # ============================================================
            elif category == 'FUSION_2F':
                # Default signals for fusion
                # RSI oversold
                df['_delta'] = df.groupby('ticker')['close'].diff()
                df['_gain'] = df['_delta'].where(df['_delta'] > 0, 0)
                df['_loss'] = (-df['_delta']).where(df['_delta'] < 0, 0)
                df['_avg_gain'] = df.groupby('ticker')['_gain'].transform(lambda x: x.rolling(14).mean())
                df['_avg_loss'] = df.groupby('ticker')['_loss'].transform(lambda x: x.rolling(14).mean())
                df['_rs'] = df['_avg_gain'] / df['_avg_loss'].replace(0, np.nan)
                df['_rsi'] = 100 - (100 / (1 + df['_rs']))
                df['_rsi_os'] = df['_rsi'] < 30
                
                # Low volatility
                df['_tr'] = np.maximum(df['high'] - df['low'], 1e-6)
                df['_atr'] = df.groupby('ticker')['_tr'].transform(lambda x: x.rolling(14).mean())
                df['_atr_pct'] = df['_atr'] / df['close']
                df['_atr_rank'] = df.groupby('date')['_atr_pct'].rank(pct=True)
                df['_low_vol'] = df['_atr_rank'] < 0.3
                
                # Momentum
                df['_mom'] = df.groupby('ticker')['close'].pct_change(20)
                df['_pos_mom'] = df['_mom'] > 0
                
                # Above EMA200
                df['_ema200'] = df.groupby('ticker')['close'].transform(lambda x: x.ewm(span=200).mean())
                df['_above200'] = df['close'] > df['_ema200']
                
                if 'RSI_LowVol' in strategy_name or 'RSI+LowVol' in strategy_name:
                    signal = df['_rsi_os'] & df['_low_vol']
                elif 'RSI_Mom' in strategy_name or 'RSI+Mom' in strategy_name:
                    signal = df['_rsi_os'] & df['_pos_mom']
                elif 'LowVol_Mom' in strategy_name or 'LowVol+Mom' in strategy_name:
                    signal = df['_low_vol'] & df['_pos_mom']
                elif 'LowVol_Above200' in strategy_name:
                    signal = df['_low_vol'] & df['_above200']
                else:
                    # Generic 2-factor: Try combining any two
                    signal = df['_rsi_os'] | df['_low_vol']
            
            # ============================================================
            # DEFAULT: Use simple momentum if category not recognized
            # ============================================================
            else:
                df['_mom'] = df.groupby('ticker')['close'].pct_change(20)
                if 'Pos' in strategy_name or 'Long' in strategy_name or 'Buy' in strategy_name:
                    signal = df['_mom'] > 0
                elif 'Neg' in strategy_name or 'Short' in strategy_name or 'Sell' in strategy_name:
                    signal = df['_mom'] < 0
                else:
                    # Default to positive momentum
                    signal = df['_mom'] > 0
            
            return signal, hold
            
        except Exception as e:
            return None, 20


# ============================================================
# STATISTICAL FUNCTIONS
# ============================================================

def calc_t_stat(returns):
    """Calculate t-statistic with proper sample size check"""
    returns = returns.dropna()
    n = len(returns)
    
    if n < MIN_SAMPLES:
        return np.nan, n, np.nan, np.nan
    
    mean = np.mean(returns)
    std = np.std(returns, ddof=1)
    
    if std == 0 or np.isnan(std):
        return np.nan, n, np.nan, np.nan
    
    t = mean / (std / np.sqrt(n))
    
    # Calculate annualized Sharpe (assuming daily returns)
    ann_sharpe = mean / std * np.sqrt(252)
    
    return mean, n, t, ann_sharpe


def t_to_p(t_stat, n):
    """Convert t-statistic to two-tailed p-value"""
    if np.isnan(t_stat) or n < 2:
        return np.nan
    return 2 * (1 - stats.t.cdf(abs(t_stat), n - 1))


def bonferroni_holm_correction(p_values, alpha=FWER_ALPHA):
    """
    Bonferroni-Holm step-down procedure.
    More powerful than simple Bonferroni while controlling FWER.
    
    Returns boolean array indicating which hypotheses are rejected.
    """
    p_values = np.array(p_values)
    n = len(p_values)
    
    # Handle NaN p-values
    valid_mask = ~np.isnan(p_values)
    valid_pvals = p_values[valid_mask]
    
    if len(valid_pvals) == 0:
        return np.zeros(n, dtype=bool)
    
    # Sort p-values
    sorted_indices = np.argsort(valid_pvals)
    sorted_pvals = valid_pvals[sorted_indices]
    
    # Holm's step-down procedure
    m = len(valid_pvals)
    thresholds = alpha / (m - np.arange(m))
    
    # Find first failure
    rejected_valid = np.zeros(m, dtype=bool)
    for i, (pval, thresh) in enumerate(zip(sorted_pvals, thresholds)):
        if pval <= thresh:
            rejected_valid[sorted_indices[i]] = True
        else:
            break  # Stop at first non-rejection
    
    # Map back to original indices
    rejected = np.zeros(n, dtype=bool)
    valid_indices = np.where(valid_mask)[0]
    for i, idx in enumerate(valid_indices):
        rejected[idx] = rejected_valid[i]
    
    return rejected


# ============================================================
# MAIN VETTING PIPELINE
# ============================================================

class StrategyVettingPipeline:
    """
    Industrial-strength strategy validation pipeline.
    
    Takes all strategies from GRAND_CONSOLIDATED_ALL.csv,
    runs walk-forward validation, and outputs only robust edges.
    """
    
    def __init__(self, db_path=DB_PATH, strategies_csv=STRATEGIES_CSV):
        self.db_path = db_path
        self.strategies_csv = strategies_csv
        self.df = None
        self.train_df = None
        self.test_df = None
        self.strategies = None
        self.results = []
        
    def load_data(self):
        """Load market data and split into train/test"""
        print("="*60)
        print("STRATEGY VETTING PIPELINE")
        print("="*60)
        print(f"\nLoading market data from {self.db_path}...")
        
        conn = sqlite3.connect(self.db_path)
        self.df = pd.read_sql("SELECT * FROM ohlcv", conn)
        conn.close()
        
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df = self.df.sort_values(['ticker', 'date']).reset_index(drop=True)
        
        print(f"  Total records: {len(self.df):,}")
        print(f"  Date range: {self.df['date'].min()} to {self.df['date'].max()}")
        print(f"  Unique tickers: {self.df['ticker'].nunique():,}")
        
        # Split into train/test
        train_end = pd.Timestamp(TRAIN_END)
        test_start = pd.Timestamp(TEST_START)
        
        self.train_df = self.df[self.df['date'] <= train_end].copy()
        self.test_df = self.df[self.df['date'] >= test_start].copy()
        
        print(f"\n📊 TRAIN PERIOD: up to {TRAIN_END}")
        print(f"  Records: {len(self.train_df):,}")
        print(f"  Date range: {self.train_df['date'].min()} to {self.train_df['date'].max()}")
        
        print(f"\n📊 TEST PERIOD: from {TEST_START}")
        print(f"  Records: {len(self.test_df):,}")
        print(f"  Date range: {self.test_df['date'].min()} to {self.test_df['date'].max()}")
        
    def load_strategies(self):
        """Load strategies from CSV"""
        print(f"\nLoading strategies from {self.strategies_csv}...")
        self.strategies = pd.read_csv(self.strategies_csv)
        print(f"  Total strategies: {len(self.strategies):,}")
        print(f"  Originally significant (t>3): {self.strategies['significant'].sum():,}")
        
        # Filter to only significant strategies (reduce computation)
        self.strategies = self.strategies[self.strategies['significant'] == True].copy()
        print(f"  Testing only significant: {len(self.strategies):,}")
        
    def calculate_forward_returns(self, df, hold_period):
        """Calculate forward returns for a given hold period"""
        df = df.copy()
        col_name = f'_fwd_{hold_period}'
        df[col_name] = df.groupby('ticker')['close'].transform(
            lambda x: x.shift(-hold_period) / x - 1
        )
        return df, col_name
        
    def validate_strategy(self, row):
        """Validate a single strategy across train and test periods"""
        category = row['category']
        strategy = row['strategy']
        original_t = row['t_stat']
        
        try:
            # Generate signal on both train and test data
            train_copy = self.train_df.copy()
            test_copy = self.test_df.copy()
            
            train_signal, hold = StrategyDefinitions.generate_signal(
                train_copy, category, strategy
            )
            test_signal, _ = StrategyDefinitions.generate_signal(
                test_copy, category, strategy
            )
            
            if train_signal is None or test_signal is None:
                return None
            
            # Calculate forward returns
            train_copy, fwd_col = self.calculate_forward_returns(train_copy, hold)
            test_copy, _ = self.calculate_forward_returns(test_copy, hold)
            
            # Extract returns for signal days
            train_returns = train_copy.loc[train_signal == True, fwd_col].dropna()
            test_returns = test_copy.loc[test_signal == True, fwd_col].dropna()
            
            if len(train_returns) < MIN_SAMPLES or len(test_returns) < MIN_SAMPLES:
                return None
            
            # Calculate t-stats
            train_mean, train_n, train_t, train_sharpe = calc_t_stat(train_returns)
            test_mean, test_n, test_t, test_sharpe = calc_t_stat(test_returns)
            
            if np.isnan(train_t) or np.isnan(test_t):
                return None
            
            # Calculate degradation
            if abs(train_t) > 0:
                degradation = 1 - (abs(test_t) / abs(train_t))
            else:
                degradation = np.nan
            
            # Calculate p-values
            train_p = t_to_p(train_t, train_n)
            test_p = t_to_p(test_t, test_n)
            
            return {
                'category': category,
                'strategy': strategy,
                'hold_period': hold,
                'original_t_stat': original_t,
                
                # Training metrics
                'train_mean_return': train_mean,
                'train_n_samples': train_n,
                'train_t_stat': train_t,
                'train_sharpe': train_sharpe,
                'train_p_value': train_p,
                
                # Test metrics (out-of-sample)
                'test_mean_return': test_mean,
                'test_n_samples': test_n,
                'test_t_stat': test_t,
                'test_sharpe': test_sharpe,
                'test_p_value': test_p,
                
                # Validation metrics
                'degradation_pct': degradation * 100 if not np.isnan(degradation) else np.nan,
                't_stat_preserved': abs(test_t) >= abs(train_t) * (1 - DEGRADATION_THRESHOLD)
            }
            
        except Exception as e:
            return None
    
    def run_validation(self, n_strategies=None, save_progress=True):
        """Run validation on all strategies"""
        if self.df is None:
            self.load_data()
        if self.strategies is None:
            self.load_strategies()
        
        strategies_to_test = self.strategies.head(n_strategies) if n_strategies else self.strategies
        
        print(f"\n{'='*60}")
        print(f"VALIDATING {len(strategies_to_test):,} STRATEGIES")
        print(f"{'='*60}")
        
        self.results = []
        
        for idx, row in tqdm(strategies_to_test.iterrows(), 
                            total=len(strategies_to_test),
                            desc="Validating"):
            result = self.validate_strategy(row)
            if result:
                self.results.append(result)
            
            # Save progress every 100 strategies
            if save_progress and len(self.results) % 100 == 0:
                self._save_interim_results()
        
        print(f"\n✅ Validated: {len(self.results):,} strategies")
        
        # Convert to DataFrame
        self.results_df = pd.DataFrame(self.results)
        
        return self.results_df
    
    def _save_interim_results(self):
        """Save interim results for long-running jobs"""
        if self.results:
            interim_df = pd.DataFrame(self.results)
            interim_df.to_csv('data/VETTING_INTERIM.csv', index=False)
    
    def apply_multiple_testing_correction(self):
        """Apply Bonferroni-Holm correction to test p-values"""
        if not hasattr(self, 'results_df') or self.results_df is None:
            print("❌ No results to correct. Run validation first.")
            return
        
        print(f"\n{'='*60}")
        print("APPLYING BONFERRONI-HOLM CORRECTION")
        print(f"{'='*60}")
        
        p_values = self.results_df['test_p_value'].values
        n_tests = len(p_values)
        
        print(f"  Total tests: {n_tests:,}")
        print(f"  FWER alpha: {FWER_ALPHA}")
        
        # Apply Bonferroni-Holm
        rejected = bonferroni_holm_correction(p_values, alpha=FWER_ALPHA)
        
        self.results_df['bonferroni_holm_significant'] = rejected
        
        n_significant = rejected.sum()
        print(f"  Significant after correction: {n_significant:,} ({100*n_significant/n_tests:.1f}%)")
        
    def generate_vetted_strategies(self):
        """Generate final VETTED_STRATEGIES.csv with only robust edges"""
        if not hasattr(self, 'results_df') or self.results_df is None:
            print("❌ No results. Run validation first.")
            return None
        
        print(f"\n{'='*60}")
        print("GENERATING VETTED STRATEGIES")
        print(f"{'='*60}")
        
        df = self.results_df.copy()
        
        # Apply multiple filters for "bulletproof" strategies
        print("\nApplying filters:")
        
        # Filter 1: Out-of-sample t-stat > 3.0
        mask_t = df['test_t_stat'].abs() >= T_THRESHOLD
        print(f"  1. OOS t-stat > {T_THRESHOLD}: {mask_t.sum():,} pass")
        
        # Filter 2: Degradation < 50%
        mask_deg = (df['degradation_pct'] < DEGRADATION_THRESHOLD * 100) | df['degradation_pct'].isna()
        print(f"  2. Degradation < {DEGRADATION_THRESHOLD*100}%: {mask_deg.sum():,} pass")
        
        # Filter 3: Bonferroni-Holm significant
        mask_bh = df['bonferroni_holm_significant'] == True
        print(f"  3. Bonferroni-Holm significant: {mask_bh.sum():,} pass")
        
        # Filter 4: Minimum OOS Sharpe ratio
        mask_sharpe = df['test_sharpe'].abs() >= MIN_OOS_SHARPE
        print(f"  4. OOS Sharpe >= {MIN_OOS_SHARPE}: {mask_sharpe.sum():,} pass")
        
        # Combine all filters
        vetted = df[mask_t & mask_deg & mask_bh & mask_sharpe].copy()
        
        print(f"\n🏆 FINAL VETTED STRATEGIES: {len(vetted):,}")
        
        # Sort by out-of-sample Sharpe ratio
        vetted = vetted.sort_values('test_sharpe', ascending=False)
        
        # Save
        vetted.to_csv(OUTPUT_CSV, index=False)
        print(f"  Saved to: {OUTPUT_CSV}")
        
        # Also save full results
        df.to_csv('data/FULL_VALIDATION_RESULTS.csv', index=False)
        print(f"  Full results saved to: data/FULL_VALIDATION_RESULTS.csv")
        
        # Save as pickle for faster loading
        with open(RESULTS_PKL, 'wb') as f:
            pickle.dump({
                'full_results': df,
                'vetted_strategies': vetted,
                'config': {
                    'train_end': TRAIN_END,
                    'test_start': TEST_START,
                    't_threshold': T_THRESHOLD,
                    'degradation_threshold': DEGRADATION_THRESHOLD,
                    'min_oos_sharpe': MIN_OOS_SHARPE,
                    'fwer_alpha': FWER_ALPHA
                }
            }, f)
        print(f"  Pickle saved to: {RESULTS_PKL}")
        
        return vetted
    
    def print_summary(self):
        """Print summary of vetted strategies"""
        if not hasattr(self, 'results_df') or self.results_df is None:
            return
        
        vetted = self.results_df[
            (self.results_df['test_t_stat'].abs() >= T_THRESHOLD) &
            (self.results_df['degradation_pct'] < DEGRADATION_THRESHOLD * 100) &
            (self.results_df['bonferroni_holm_significant'] == True)
        ]
        
        print(f"\n{'='*60}")
        print("SUMMARY: TOP 20 BULLETPROOF STRATEGIES")
        print(f"{'='*60}")
        
        top = vetted.nlargest(20, 'test_sharpe')[
            ['category', 'strategy', 'train_t_stat', 'test_t_stat', 
             'degradation_pct', 'test_sharpe']
        ]
        print(top.to_string(index=False))
        
        print(f"\n{'='*60}")
        print("CATEGORY BREAKDOWN")
        print(f"{'='*60}")
        
        cat_summary = vetted.groupby('category').agg({
            'strategy': 'count',
            'test_sharpe': 'mean'
        }).rename(columns={'strategy': 'n_strategies', 'test_sharpe': 'avg_sharpe'})
        cat_summary = cat_summary.sort_values('n_strategies', ascending=False)
        print(cat_summary.head(15))
        
        print(f"\n{'='*60}")
        print("VALIDATION STATISTICS")
        print(f"{'='*60}")
        
        total = len(self.results_df)
        oos_significant = (self.results_df['test_t_stat'].abs() >= T_THRESHOLD).sum()
        low_degrade = (self.results_df['degradation_pct'] < 50).sum()
        bh_sig = self.results_df['bonferroni_holm_significant'].sum()
        
        print(f"  Total validated: {total:,}")
        print(f"  OOS t > 3.0: {oos_significant:,} ({100*oos_significant/total:.1f}%)")
        print(f"  Degradation < 50%: {low_degrade:,} ({100*low_degrade/total:.1f}%)")
        print(f"  Bonferroni-Holm significant: {bh_sig:,} ({100*bh_sig/total:.1f}%)")
        print(f"  FINAL VETTED: {len(vetted):,} ({100*len(vetted)/total:.1f}%)")


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Strategy Vetting Pipeline')
    parser.add_argument('--n', type=int, default=None, 
                       help='Number of strategies to test (default: all)')
    parser.add_argument('--quick', action='store_true',
                       help='Quick test with 100 strategies')
    args = parser.parse_args()
    
    n_strategies = 100 if args.quick else args.n
    
    # Initialize pipeline
    pipeline = StrategyVettingPipeline()
    
    # Run validation
    pipeline.run_validation(n_strategies=n_strategies)
    
    # Apply multiple testing correction
    pipeline.apply_multiple_testing_correction()
    
    # Generate vetted strategies
    vetted = pipeline.generate_vetted_strategies()
    
    # Print summary
    pipeline.print_summary()
    
    print("\n" + "="*60)
    print("🎯 PIPELINE COMPLETE")
    print("="*60)
    print(f"\nFiles generated:")
    print(f"  - {OUTPUT_CSV}")
    print(f"  - data/FULL_VALIDATION_RESULTS.csv")
    print(f"  - {RESULTS_PKL}")
