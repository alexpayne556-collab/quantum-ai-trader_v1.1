"""
Data Validator - Quality checks for market data

Validates OHLCV data for:
- Missing data gaps
- Outliers and anomalies
- Stock splits detection
- Volume anomalies
- Data consistency
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Tuple, List, Dict

from .config import *

logger = logging.getLogger(__name__)


class DataValidator:
    """
    Validate market data quality.
    
    Checks:
    - Sufficient data volume
    - Missing dates
    - Price outliers
    - Volume anomalies
    - Data consistency
    """
    
    def __init__(self):
        """Initialize validator."""
        logger.debug("DataValidator initialized")
    
    def validate_ticker_data(
        self,
        df: pd.DataFrame,
        ticker: str
    ) -> Tuple[bool, List[str]]:
        """
        Run all validation checks on ticker data.
        
        Args:
            df: DataFrame with OHLCV data
            ticker: Ticker symbol for reporting
        
        Returns:
            (is_valid, list_of_issues)
        """
        issues = []
        
        # Check 1: Sufficient data
        if len(df) < MIN_TRADING_DAYS:
            issues.append(f"Insufficient data: {len(df)} days (need {MIN_TRADING_DAYS})")
        
        # Check 2: Missing dates
        missing_pct, max_consecutive = self._check_missing_dates(df)
        if missing_pct > MAX_MISSING_DAYS_PCT:
            issues.append(f"Too many missing days: {missing_pct:.1f}% (max {MAX_MISSING_DAYS_PCT}%)")
        
        if max_consecutive > MAX_CONSECUTIVE_MISSING:
            issues.append(f"Consecutive missing days: {max_consecutive} (max {MAX_CONSECUTIVE_MISSING})")
        
        # Check 3: Price outliers
        price_outliers = self._check_price_outliers(df)
        if price_outliers:
            issues.append(f"Price outliers detected: {len(price_outliers)} days")
        
        # Check 4: Volume issues
        avg_volume = df['Volume'].mean()
        if avg_volume < MIN_VOLUME_THRESHOLD:
            issues.append(f"Low volume: {avg_volume:,.0f} (min {MIN_VOLUME_THRESHOLD:,.0f})")
        
        # Check 5: Data consistency
        consistency_issues = self._check_data_consistency(df)
        if consistency_issues:
            issues.extend(consistency_issues)
        
        # Check 6: Potential splits
        splits = self._detect_splits(df)
        if splits:
            issues.append(f"Potential splits detected: {len(splits)} occurrences")
        
        is_valid = len(issues) == 0
        
        if not is_valid:
            logger.warning(f"{ticker} validation issues: {', '.join(issues)}")
        else:
            logger.debug(f"{ticker} passed all validation checks")
        
        return is_valid, issues
    
    def _check_missing_dates(self, df: pd.DataFrame) -> Tuple[float, int]:
        """
        Check for missing trading days.
        
        Returns:
            (missing_percentage, max_consecutive_missing)
        """
        if len(df) < 2:
            return 0.0, 0
        
        # Calculate expected trading days (rough estimate: ~252 per year)
        date_range = (df.index.max() - df.index.min()).days
        expected_days = int(date_range * 252 / 365)
        
        actual_days = len(df)
        missing_days = max(0, expected_days - actual_days)
        missing_pct = (missing_days / expected_days * 100) if expected_days > 0 else 0
        
        # Check for consecutive missing days
        df_sorted = df.sort_index()
        date_diff = df_sorted.index.to_series().diff()
        
        # Business days difference > 4 indicates missing data (weekend + 1 day)
        max_consecutive = 0
        if len(date_diff) > 0:
            consecutive_missing = date_diff.dt.days - 1  # -1 for normal gap
            consecutive_missing = consecutive_missing[consecutive_missing > 3]  # Ignore weekends
            if len(consecutive_missing) > 0:
                max_consecutive = int(consecutive_missing.max() - 2)  # Subtract weekend
        
        return missing_pct, max(0, max_consecutive)
    
    def _check_price_outliers(self, df: pd.DataFrame) -> List[str]:
        """
        Detect price outliers (possible data errors).
        
        Returns:
            List of dates with outliers
        """
        outliers = []
        
        # Check for negative or zero prices
        if (df['Close'] <= 0).any():
            outliers.extend(df[df['Close'] <= 0].index.strftime('%Y-%m-%d').tolist())
        
        # Check for extreme daily returns
        returns = df['Close'].pct_change()
        extreme_returns = abs(returns) > (MAX_DAILY_RETURN_PCT / 100)
        
        if extreme_returns.any():
            outlier_dates = df[extreme_returns].index.strftime('%Y-%m-%d').tolist()
            outliers.extend(outlier_dates)
        
        # Check for prices below minimum
        if (df['Close'] < MIN_PRICE).any():
            low_price_dates = df[df['Close'] < MIN_PRICE].index.strftime('%Y-%m-%d').tolist()
            outliers.extend(low_price_dates)
        
        return list(set(outliers))
    
    def _check_data_consistency(self, df: pd.DataFrame) -> List[str]:
        """
        Check for data consistency issues.
        
        Returns:
            List of consistency issues
        """
        issues = []
        
        # High should be >= Low
        if (df['High'] < df['Low']).any():
            issues.append("High < Low detected")
        
        # Close should be between High and Low
        invalid_close = (df['Close'] > df['High']) | (df['Close'] < df['Low'])
        if invalid_close.any():
            issues.append("Close outside High/Low range")
        
        # Open should be between High and Low
        if 'Open' in df.columns:
            invalid_open = (df['Open'] > df['High']) | (df['Open'] < df['Low'])
            if invalid_open.any():
                issues.append("Open outside High/Low range")
        
        # Volume should be non-negative
        if (df['Volume'] < 0).any():
            issues.append("Negative volume detected")
        
        # Check for NaN values
        if df[['Open', 'High', 'Low', 'Close', 'Volume']].isna().any().any():
            issues.append("NaN values detected")
        
        return issues
    
    def _detect_splits(self, df: pd.DataFrame) -> List[str]:
        """
        Detect potential stock splits (large price jumps).
        
        Returns:
            List of dates with potential splits
        """
        # Calculate daily returns
        returns = df['Close'].pct_change()
        
        # Splits often show as 50% or 100% changes (2:1 split, 3:1 split, etc.)
        # But yfinance usually adjusts for splits, so this is a sanity check
        potential_splits = []
        
        # Look for exact 50%, 66.67%, 75% drops (splits) or inverse (reverse splits)
        split_ratios = [0.5, 0.333, 0.25, -0.5, -1.0, -2.0]
        
        for ratio in split_ratios:
            matches = abs(returns - ratio) < 0.05  # Within 5% of split ratio
            if matches.any():
                split_dates = df[matches].index.strftime('%Y-%m-%d').tolist()
                potential_splits.extend(split_dates)
        
        return list(set(potential_splits))
    
    def generate_quality_report(
        self,
        df: pd.DataFrame,
        ticker: str
    ) -> Dict:
        """
        Generate comprehensive quality report for a ticker.
        
        Returns:
            Dictionary with quality metrics
        """
        is_valid, issues = self.validate_ticker_data(df, ticker)
        
        missing_pct, max_consecutive = self._check_missing_dates(df)
        price_outliers = self._check_price_outliers(df)
        splits = self._detect_splits(df)
        
        report = {
            'ticker': ticker,
            'is_valid': is_valid,
            'issues': issues,
            'metrics': {
                'total_days': len(df),
                'date_range': f"{df.index.min().date()} to {df.index.max().date()}",
                'missing_pct': missing_pct,
                'max_consecutive_missing': max_consecutive,
                'price_outliers': len(price_outliers),
                'potential_splits': len(splits),
                'avg_volume': float(df['Volume'].mean()),
                'avg_price': float(df['Close'].mean()),
                'price_range': f"${df['Close'].min():.2f} - ${df['Close'].max():.2f}"
            }
        }
        
        return report


if __name__ == '__main__':
    # Test the validator
    import yfinance as yf
    
    print("🧪 Testing DataValidator...")
    print("="*80)
    
    # Download test data
    print("\n1. Testing with AAPL...")
    df = yf.download('AAPL', period='1y', progress=False, auto_adjust=True)
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    
    validator = DataValidator()
    is_valid, issues = validator.validate_ticker_data(df, 'AAPL')
    
    print(f"   Valid: {is_valid}")
    if issues:
        print(f"   Issues: {', '.join(issues)}")
    
    # Generate full report
    print("\n2. Quality report for AAPL:")
    report = validator.generate_quality_report(df, 'AAPL')
    
    print(f"   Total days: {report['metrics']['total_days']}")
    print(f"   Date range: {report['metrics']['date_range']}")
    print(f"   Missing %: {report['metrics']['missing_pct']:.1f}%")
    print(f"   Avg volume: {report['metrics']['avg_volume']:,.0f}")
    print(f"   Price range: {report['metrics']['price_range']}")
    
    print("\n" + "="*80)
    print("✅ DataValidator test complete")
