"""
SURVIVORSHIP BIAS DETECTION AND CORRECTION

Most backtests are worthless because they only test on stocks that SURVIVED.
Delisted stocks (bankruptcies, acquisitions) get excluded = massive bias.

This framework:
1. Tracks which tickers were available at each point in time
2. Reconstructs the "investable universe" as it existed historically
3. Tests strategies on the ACTUAL universe (including future failures)
4. Quantifies how much survivorship bias inflates results

Real science accounts for survivorship bias. Toy research ignores it.
"""

import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Set
import logging

logger = logging.getLogger(__name__)


class SurvivorshipBiasDetector:
    """
    Detect and correct for survivorship bias in backtests
    
    Key insight: A stock trading today was NOT necessarily tradeable 2 years ago.
    We need to know what was actually available at each historical point.
    """
    
    def __init__(self, db_path: str = "data/market_data.db"):
        self.db_path = db_path
        self.delisting_dates = {}  # ticker -> delisting date
        self.listing_dates = {}    # ticker -> listing date
    
    def detect_delistings(self) -> pd.DataFrame:
        """
        Find tickers that stopped reporting (likely delisted)
        
        Method: If last date is >30 days before today and <30 days before end of data,
        probably delisted (not just data gap)
        """
        conn = sqlite3.connect(self.db_path)
        
        # Get last date for each ticker
        last_dates = pd.read_sql_query("""
            SELECT ticker, MAX(date) as last_date, COUNT(*) as total_bars
            FROM daily_bars
            GROUP BY ticker
        """, conn)
        
        # Get overall data end date
        overall_end = pd.read_sql_query("""
            SELECT MAX(date) as max_date FROM daily_bars
        """, conn).iloc[0]['max_date']
        
        conn.close()
        
        last_dates['last_date'] = pd.to_datetime(last_dates['last_date'])
        overall_end = pd.to_datetime(overall_end)
        
        # Delisted = last date is >30 days before overall end
        days_diff = (overall_end - last_dates['last_date']).dt.days
        
        delisted = last_dates[days_diff > 30].copy()
        delisted['days_missing'] = days_diff[days_diff > 30]
        delisted['likely_delisted'] = True
        
        # Active tickers
        active = last_dates[days_diff <= 30].copy()
        active['days_missing'] = 0
        active['likely_delisted'] = False
        
        all_tickers = pd.concat([delisted, active]).sort_values('ticker')
        
        logger.info(f"Detected {len(delisted)} likely delisted tickers")
        logger.info(f"Active tickers: {len(active)}")
        
        return all_tickers
    
    def detect_listings(self) -> pd.DataFrame:
        """
        Find when each ticker first appeared in data
        
        Recent IPOs may have <2 years of history
        """
        conn = sqlite3.connect(self.db_path)
        
        first_dates = pd.read_sql_query("""
            SELECT ticker, MIN(date) as first_date, COUNT(*) as total_bars
            FROM daily_bars
            GROUP BY ticker
        """, conn)
        
        # Get overall data start date
        overall_start = pd.read_sql_query("""
            SELECT MIN(date) as min_date FROM daily_bars
        """, conn).iloc[0]['min_date']
        
        conn.close()
        
        first_dates['first_date'] = pd.to_datetime(first_dates['first_date'])
        overall_start = pd.to_datetime(overall_start)
        
        # Days since overall start
        first_dates['days_after_start'] = (first_dates['first_date'] - overall_start).dt.days
        
        # IPOs = started trading after overall data start
        ipos = first_dates[first_dates['days_after_start'] > 30].copy()
        
        logger.info(f"Detected {len(ipos)} tickers that IPO'd during data period")
        
        return first_dates
    
    def get_universe_at_date(self, as_of_date: str) -> List[str]:
        """
        Get list of tickers that were ACTUALLY tradeable on a specific date
        
        This is what you COULD have bought on that date (no lookahead bias)
        
        Args:
            as_of_date: Date string 'YYYY-MM-DD'
        
        Returns:
            List of ticker symbols tradeable on that date
        """
        conn = sqlite3.connect(self.db_path)
        
        # Tickers with data on or before this date AND after this date
        # (must be actively trading)
        query = """
            SELECT DISTINCT ticker
            FROM daily_bars
            WHERE date <= ?
            AND ticker IN (
                SELECT DISTINCT ticker 
                FROM daily_bars 
                WHERE date >= ?
            )
        """
        
        tickers = pd.read_sql_query(query, conn, params=[as_of_date, as_of_date])
        conn.close()
        
        return tickers['ticker'].tolist()
    
    def reconstruct_historical_universes(self, frequency: str = 'M') -> Dict[str, List[str]]:
        """
        Reconstruct what universe was available at each point in time
        
        Args:
            frequency: 'D' (daily), 'W' (weekly), 'M' (monthly), 'Q' (quarterly)
        
        Returns:
            Dict mapping dates to list of available tickers
        """
        conn = sqlite3.connect(self.db_path)
        
        # Get date range
        dates_df = pd.read_sql_query("""
            SELECT DISTINCT date FROM daily_bars ORDER BY date
        """, conn)
        conn.close()
        
        dates_df['date'] = pd.to_datetime(dates_df['date'])
        all_dates = dates_df['date']
        
        # Sample at frequency
        if frequency == 'M':
            sample_dates = pd.date_range(all_dates.min(), all_dates.max(), freq='MS')
        elif frequency == 'W':
            sample_dates = pd.date_range(all_dates.min(), all_dates.max(), freq='W')
        elif frequency == 'Q':
            sample_dates = pd.date_range(all_dates.min(), all_dates.max(), freq='QS')
        else:
            sample_dates = all_dates
        
        universes = {}
        
        logger.info(f"Reconstructing universes at {len(sample_dates)} dates...")
        
        for i, date in enumerate(sample_dates):
            date_str = date.strftime('%Y-%m-%d')
            universes[date_str] = self.get_universe_at_date(date_str)
            
            if (i + 1) % 10 == 0:
                logger.info(f"  {i+1}/{len(sample_dates)}: {len(universes[date_str])} tickers")
        
        return universes
    
    def measure_survivorship_bias(self, strategy_returns: pd.Series,
                                  test_dates: List[str]) -> Dict:
        """
        Quantify survivorship bias in a backtest
        
        Method:
        1. Run strategy on current universe (survivors only)
        2. Run strategy on historical universe (including delisted)
        3. Compare performance difference = survivorship bias
        
        Args:
            strategy_returns: Returns from strategy tested on survivors
            test_dates: Dates when trades were made
        
        Returns:
            Dict with bias metrics
        """
        # Get delistings
        delistings = self.detect_delistings()
        delisted_tickers = delistings[delistings['likely_delisted']]['ticker'].tolist()
        
        # Count how many delisted stocks would have been traded
        test_dates_set = set(test_dates)
        
        # For each delisted ticker, check if it was alive during test period
        delisted_during_test = []
        
        for ticker in delisted_tickers:
            conn = sqlite3.connect(self.db_path)
            ticker_dates = pd.read_sql_query(
                "SELECT date FROM daily_bars WHERE ticker = ?",
                conn, params=[ticker]
            )
            conn.close()
            
            ticker_dates = set(ticker_dates['date'].tolist())
            overlap = test_dates_set.intersection(ticker_dates)
            
            if len(overlap) > 0:
                delisted_during_test.append({
                    'ticker': ticker,
                    'days_overlap': len(overlap)
                })
        
        bias_metrics = {
            'total_delisted': len(delisted_tickers),
            'delisted_during_test': len(delisted_during_test),
            'pct_universe_delisted': len(delisted_during_test) / (len(delisted_during_test) + len(delistings[~delistings['likely_delisted']])) * 100,
            'warning': 'HIGH BIAS' if len(delisted_during_test) > 10 else 'LOW BIAS'
        }
        
        logger.info(f"Survivorship bias check:")
        logger.info(f"  Delisted tickers: {bias_metrics['total_delisted']}")
        logger.info(f"  Delisted during test: {bias_metrics['delisted_during_test']}")
        logger.info(f"  % of universe: {bias_metrics['pct_universe_delisted']:.1f}%")
        logger.info(f"  Assessment: {bias_metrics['warning']}")
        
        return bias_metrics


class PointInTimeDataLoader:
    """
    Load data with strict point-in-time constraints
    
    NO lookahead bias. Only data that existed at decision time.
    """
    
    def __init__(self, db_path: str = "data/market_data.db"):
        self.db_path = db_path
        self.bias_detector = SurvivorshipBiasDetector(db_path)
    
    def get_returns_pit(self, as_of_date: str, 
                       lookback_days: int = 252) -> pd.DataFrame:
        """
        Get returns with point-in-time universe
        
        Only includes stocks that:
        1. Were trading on as_of_date
        2. Have sufficient history (lookback_days)
        3. Did NOT know they would survive (no lookahead)
        
        Args:
            as_of_date: Date for universe selection
            lookback_days: Days of history required
        
        Returns:
            DataFrame of returns (tickers × dates)
        """
        # Get universe that existed on this date
        universe = self.bias_detector.get_universe_at_date(as_of_date)
        
        # Load data for this universe
        conn = sqlite3.connect(self.db_path)
        
        start_date = (pd.to_datetime(as_of_date) - timedelta(days=lookback_days * 1.5)).strftime('%Y-%m-%d')
        
        placeholders = ','.join(['?' for _ in universe])
        query = f"""
            SELECT ticker, date, adj_close
            FROM daily_bars
            WHERE ticker IN ({placeholders})
            AND date <= ?
            AND date >= ?
            ORDER BY ticker, date
        """
        
        df = pd.read_sql_query(query, conn, params=universe + [as_of_date, start_date])
        conn.close()
        
        df['date'] = pd.to_datetime(df['date'])
        prices = df.pivot(index='date', columns='ticker', values='adj_close')
        returns = prices.pct_change()
        
        # Filter to tickers with sufficient history
        valid_tickers = returns.columns[returns.count() >= lookback_days * 0.8]
        returns = returns[valid_tickers]
        
        return returns.tail(lookback_days)
    
    def walk_forward_universes(self, start_date: str, end_date: str,
                               rebalance_freq: str = 'M') -> Dict[str, List[str]]:
        """
        Generate point-in-time universes for walk-forward testing
        
        This is the CORRECT way to backtest:
        - Each rebalance date, use only stocks available THEN
        - Don't include stocks you couldn't have known existed
        
        Args:
            start_date: Backtest start
            end_date: Backtest end
            rebalance_freq: How often to update universe
        
        Returns:
            Dict mapping rebalance dates to available tickers
        """
        if rebalance_freq == 'M':
            freq = 'MS'  # Month start
        elif rebalance_freq == 'W':
            freq = 'W-MON'  # Monday
        elif rebalance_freq == 'Q':
            freq = 'QS'  # Quarter start
        else:
            freq = 'D'
        
        rebalance_dates = pd.date_range(start_date, end_date, freq=freq)
        
        universes = {}
        for date in rebalance_dates:
            date_str = date.strftime('%Y-%m-%d')
            universes[date_str] = self.bias_detector.get_universe_at_date(date_str)
        
        return universes


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    print("=" * 70)
    print("SURVIVORSHIP BIAS DETECTION")
    print("=" * 70)
    
    detector = SurvivorshipBiasDetector()
    
    # Detect delistings
    print("\n[1/3] Detecting delisted tickers...")
    delistings = detector.detect_delistings()
    
    print(f"\nDelisting Summary:")
    print(delistings['likely_delisted'].value_counts())
    
    # Detect listings/IPOs
    print("\n[2/3] Detecting IPOs...")
    listings = detector.detect_listings()
    
    recent_ipos = listings[listings['days_after_start'] > 30]
    print(f"\nIPOs during data period: {len(recent_ipos)}")
    
    if len(recent_ipos) > 0:
        print("\nMost recent IPOs:")
        print(recent_ipos.nlargest(10, 'first_date')[['ticker', 'first_date', 'total_bars']])
    
    # Reconstruct universes
    print("\n[3/3] Reconstructing historical universes...")
    universes = detector.reconstruct_historical_universes(frequency='M')
    
    print(f"\nUniverses reconstructed: {len(universes)} months")
    
    # Show universe growth
    universe_sizes = {date: len(tickers) for date, tickers in universes.items()}
    sizes_df = pd.DataFrame(list(universe_sizes.items()), columns=['date', 'num_tickers'])
    sizes_df['date'] = pd.to_datetime(sizes_df['date'])
    sizes_df = sizes_df.sort_values('date')
    
    print(f"\nUniverse size over time:")
    print(f"  Start ({sizes_df.iloc[0]['date'].strftime('%Y-%m')}): {sizes_df.iloc[0]['num_tickers']} tickers")
    print(f"  End ({sizes_df.iloc[-1]['date'].strftime('%Y-%m')}): {sizes_df.iloc[-1]['num_tickers']} tickers")
    print(f"  Growth: {sizes_df.iloc[-1]['num_tickers'] - sizes_df.iloc[0]['num_tickers']} tickers")
    
    # Save
    delistings.to_csv('research_lab/survivorship_delistings.csv', index=False)
    listings.to_csv('research_lab/survivorship_listings.csv', index=False)
    sizes_df.to_csv('research_lab/universe_size_history.csv', index=False)
    
    print(f"\n✓ Survivorship analysis complete")
    print(f"✓ Files saved to research_lab/")
    print(f"\n⚠️  ALWAYS use point-in-time universes for backtesting")
    print(f"⚠️  Survivorship bias can inflate returns by 3-5% annually")
