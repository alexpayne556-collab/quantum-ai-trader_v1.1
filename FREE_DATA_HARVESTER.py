#!/usr/bin/env python3
"""
FREE DATA HARVESTER - Comprehensive Market Data Collection System
=================================================================

This system harvests ALL data needed for advanced hypothesis testing
using 100% FREE data sources:

1. yfinance - Unlimited historical OHLCV data
2. FRED API - Unlimited economic indicators
3. Public calendars - Event dates

NO DATA PURCHASE REQUIRED.

Usage:
    python FREE_DATA_HARVESTER.py --full          # Harvest everything
    python FREE_DATA_HARVESTER.py --vix           # VIX data only
    python FREE_DATA_HARVESTER.py --update        # Update existing data
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import os
import warnings
import time
import json
from pathlib import Path
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_DIR = Path("data/free_harvest")
START_DATE = "2010-01-01"  # 15 years of data for robust testing

# Comprehensive ticker universe
TICKER_UNIVERSE = {
    # VIX & Volatility Products
    'volatility': {
        '^VIX': 'CBOE Volatility Index',
        '^VIX3M': 'CBOE 3-Month Volatility Index (VXV)',
        '^VIX9D': 'CBOE 9-Day Volatility Index',
        'VXX': 'iPath Series B S&P 500 VIX Short-Term Futures ETN',
        'UVXY': 'ProShares Ultra VIX Short-Term Futures ETF',
        'SVXY': 'ProShares Short VIX Short-Term Futures ETF',
        'VIXY': 'ProShares VIX Short-Term Futures ETF',
    },
    
    # Major Market Indices
    'indices': {
        '^GSPC': 'S&P 500 Index',
        '^DJI': 'Dow Jones Industrial Average',
        '^IXIC': 'NASDAQ Composite',
        '^RUT': 'Russell 2000',
        '^VIX': 'CBOE Volatility Index',
    },
    
    # Core Market ETFs
    'market_etfs': {
        'SPY': 'SPDR S&P 500 ETF',
        'QQQ': 'Invesco QQQ Trust',
        'IWM': 'iShares Russell 2000 ETF',
        'DIA': 'SPDR Dow Jones Industrial Average ETF',
        'MDY': 'SPDR S&P MidCap 400 ETF',
        'IVV': 'iShares Core S&P 500 ETF',
        'VTI': 'Vanguard Total Stock Market ETF',
        'VOO': 'Vanguard S&P 500 ETF',
    },
    
    # Sector ETFs (SPDR Select Sector)
    'sectors': {
        'XLF': 'Financial Select Sector SPDR',
        'XLK': 'Technology Select Sector SPDR',
        'XLV': 'Health Care Select Sector SPDR',
        'XLE': 'Energy Select Sector SPDR',
        'XLI': 'Industrial Select Sector SPDR',
        'XLP': 'Consumer Staples Select Sector SPDR',
        'XLY': 'Consumer Discretionary Select Sector SPDR',
        'XLU': 'Utilities Select Sector SPDR',
        'XLB': 'Materials Select Sector SPDR',
        'XLRE': 'Real Estate Select Sector SPDR',
        'XLC': 'Communication Services Select Sector SPDR',
    },
    
    # Treasury & Bond ETFs
    'bonds': {
        'TLT': 'iShares 20+ Year Treasury Bond ETF',
        'IEF': 'iShares 7-10 Year Treasury Bond ETF',
        'SHY': 'iShares 1-3 Year Treasury Bond ETF',
        'IEI': 'iShares 3-7 Year Treasury Bond ETF',
        'TIP': 'iShares TIPS Bond ETF',
        'LQD': 'iShares iBoxx $ Investment Grade Corporate Bond ETF',
        'HYG': 'iShares iBoxx $ High Yield Corporate Bond ETF',
        'AGG': 'iShares Core U.S. Aggregate Bond ETF',
        'BND': 'Vanguard Total Bond Market ETF',
        'GOVT': 'iShares U.S. Treasury Bond ETF',
        'MUB': 'iShares National Muni Bond ETF',
        'EMB': 'iShares J.P. Morgan USD Emerging Markets Bond ETF',
    },
    
    # Treasury Yields (FRED proxy via yfinance)
    'yields': {
        '^TNX': '10-Year Treasury Yield',
        '^TYX': '30-Year Treasury Yield',
        '^FVX': '5-Year Treasury Yield',
        '^IRX': '13-Week T-Bill Rate',
    },
    
    # Commodity ETFs
    'commodities': {
        'GLD': 'SPDR Gold Shares',
        'IAU': 'iShares Gold Trust',
        'SLV': 'iShares Silver Trust',
        'USO': 'United States Oil Fund',
        'UNG': 'United States Natural Gas Fund',
        'DBA': 'Invesco DB Agriculture Fund',
        'DBC': 'Invesco DB Commodity Index Tracking Fund',
        'PDBC': 'Invesco Optimum Yield Diversified Commodity Strategy No K-1 ETF',
        'GSG': 'iShares S&P GSCI Commodity-Indexed Trust',
        'WEAT': 'Teucrium Wheat Fund',
        'CORN': 'Teucrium Corn Fund',
        'SOYB': 'Teucrium Soybean Fund',
    },
    
    # Commodity Futures (direct)
    'futures': {
        'GC=F': 'Gold Futures',
        'SI=F': 'Silver Futures',
        'CL=F': 'Crude Oil Futures',
        'NG=F': 'Natural Gas Futures',
        'ZC=F': 'Corn Futures',
        'ZW=F': 'Wheat Futures',
        'ZS=F': 'Soybean Futures',
        'HG=F': 'Copper Futures',
    },
    
    # International ETFs
    'international': {
        'EFA': 'iShares MSCI EAFE ETF',
        'EEM': 'iShares MSCI Emerging Markets ETF',
        'VGK': 'Vanguard FTSE Europe ETF',
        'VWO': 'Vanguard FTSE Emerging Markets ETF',
        'IEMG': 'iShares Core MSCI Emerging Markets ETF',
        'FXI': 'iShares China Large-Cap ETF',
        'EWJ': 'iShares MSCI Japan ETF',
        'EWG': 'iShares MSCI Germany ETF',
        'EWU': 'iShares MSCI United Kingdom ETF',
        'INDA': 'iShares MSCI India ETF',
        'EWZ': 'iShares MSCI Brazil ETF',
        'EWY': 'iShares MSCI South Korea ETF',
        'EWT': 'iShares MSCI Taiwan ETF',
        'EWH': 'iShares MSCI Hong Kong ETF',
        'EWA': 'iShares MSCI Australia ETF',
        'EWC': 'iShares MSCI Canada ETF',
    },
    
    # Currency ETFs
    'currencies': {
        'UUP': 'Invesco DB US Dollar Index Bullish Fund',
        'UDN': 'Invesco DB US Dollar Index Bearish Fund',
        'FXE': 'Invesco CurrencyShares Euro Trust',
        'FXY': 'Invesco CurrencyShares Japanese Yen Trust',
        'FXB': 'Invesco CurrencyShares British Pound Sterling Trust',
        'FXA': 'Invesco CurrencyShares Australian Dollar Trust',
        'FXC': 'Invesco CurrencyShares Canadian Dollar Trust',
    },
    
    # Leverage & Inverse ETFs (for vol analysis)
    'leveraged': {
        'TQQQ': 'ProShares UltraPro QQQ (3x)',
        'SQQQ': 'ProShares UltraPro Short QQQ (-3x)',
        'UPRO': 'ProShares UltraPro S&P 500 (3x)',
        'SPXU': 'ProShares UltraPro Short S&P 500 (-3x)',
        'TNA': 'Direxion Daily Small Cap Bull 3X',
        'TZA': 'Direxion Daily Small Cap Bear 3X',
    },
    
    # Factor ETFs
    'factors': {
        'MTUM': 'iShares MSCI USA Momentum Factor ETF',
        'QUAL': 'iShares MSCI USA Quality Factor ETF',
        'VLUE': 'iShares MSCI USA Value Factor ETF',
        'SIZE': 'iShares MSCI USA Size Factor ETF',
        'USMV': 'iShares MSCI USA Min Vol Factor ETF',
    },
}

# ============================================================================
# DATA HARVESTING FUNCTIONS
# ============================================================================

def create_data_directory():
    """Create data directory structure."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for category in TICKER_UNIVERSE.keys():
        (DATA_DIR / category).mkdir(exist_ok=True)
    print(f"✓ Data directory created: {DATA_DIR}")


def harvest_ticker(ticker: str, start_date: str = START_DATE) -> pd.DataFrame:
    """Harvest OHLCV data for a single ticker."""
    try:
        data = yf.Ticker(ticker).history(start=start_date, auto_adjust=True)
        if len(data) < 100:
            return None
        
        # Clean column names
        data.columns = [c.lower().replace(' ', '_') for c in data.columns]
        data.index.name = 'date'
        
        # Add calculated fields
        data['returns'] = data['close'].pct_change()
        data['log_returns'] = np.log(data['close'] / data['close'].shift(1))
        data['realized_vol_5d'] = data['returns'].rolling(5).std() * np.sqrt(252)
        data['realized_vol_20d'] = data['returns'].rolling(20).std() * np.sqrt(252)
        
        return data
    except Exception as e:
        return None


def harvest_category(category: str, tickers: dict, verbose: bool = True) -> dict:
    """Harvest all tickers in a category."""
    results = {}
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"HARVESTING: {category.upper()}")
        print('='*60)
    
    for ticker, name in tickers.items():
        data = harvest_ticker(ticker)
        if data is not None:
            results[ticker] = data
            # Save individual file
            filepath = DATA_DIR / category / f"{ticker.replace('^', 'INDEX_').replace('=', '_')}.csv"
            data.to_csv(filepath)
            if verbose:
                print(f"  ✓ {ticker}: {len(data)} days -> {filepath.name}")
        else:
            if verbose:
                print(f"  ✗ {ticker}: No data")
        
        time.sleep(0.1)  # Be nice to Yahoo's servers
    
    return results


def harvest_all(verbose: bool = True) -> dict:
    """Harvest ALL data from the entire universe."""
    create_data_directory()
    
    all_data = {}
    total_tickers = sum(len(t) for t in TICKER_UNIVERSE.values())
    harvested = 0
    
    print(f"\n{'='*80}")
    print(f"FREE DATA HARVESTER - Collecting {total_tickers} instruments")
    print('='*80)
    
    for category, tickers in TICKER_UNIVERSE.items():
        category_data = harvest_category(category, tickers, verbose)
        all_data[category] = category_data
        harvested += len(category_data)
    
    # Create combined price matrix
    print(f"\n{'='*60}")
    print("CREATING COMBINED PRICE MATRIX")
    print('='*60)
    
    all_prices = {}
    for category, data in all_data.items():
        for ticker, df in data.items():
            clean_ticker = ticker.replace('^', '').replace('=F', '')
            all_prices[clean_ticker] = df['close']
    
    price_matrix = pd.DataFrame(all_prices)
    price_matrix.to_csv(DATA_DIR / "COMBINED_PRICE_MATRIX.csv")
    print(f"  ✓ Combined price matrix: {price_matrix.shape}")
    
    # Create combined returns matrix
    returns_matrix = price_matrix.pct_change()
    returns_matrix.to_csv(DATA_DIR / "COMBINED_RETURNS_MATRIX.csv")
    print(f"  ✓ Combined returns matrix: {returns_matrix.shape}")
    
    # Summary
    print(f"\n{'='*80}")
    print(f"HARVEST COMPLETE")
    print('='*80)
    print(f"""
    Total instruments harvested: {harvested}/{total_tickers}
    Date range: {START_DATE} to {datetime.now().strftime('%Y-%m-%d')}
    Data directory: {DATA_DIR}
    
    Files created:
    - Individual CSVs in category folders
    - COMBINED_PRICE_MATRIX.csv
    - COMBINED_RETURNS_MATRIX.csv
    """)
    
    # Save metadata
    metadata = {
        'harvest_date': datetime.now().isoformat(),
        'start_date': START_DATE,
        'total_instruments': harvested,
        'categories': {k: list(v.keys()) for k, v in all_data.items()}
    }
    with open(DATA_DIR / "harvest_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return all_data


def load_harvested_data() -> tuple:
    """Load previously harvested data."""
    price_matrix = pd.read_csv(DATA_DIR / "COMBINED_PRICE_MATRIX.csv", index_col=0, parse_dates=True)
    returns_matrix = pd.read_csv(DATA_DIR / "COMBINED_RETURNS_MATRIX.csv", index_col=0, parse_dates=True)
    return price_matrix, returns_matrix


# ============================================================================
# FRED DATA (Economic Indicators)
# ============================================================================

def harvest_fred_data(api_key: str = None) -> pd.DataFrame:
    """
    Harvest economic data from FRED.
    
    Key series:
    - VIXCLS: VIX Index
    - DGS10: 10-Year Treasury Rate
    - DGS2: 2-Year Treasury Rate  
    - FEDFUNDS: Federal Funds Rate
    - UNRATE: Unemployment Rate
    - CPIAUCSL: Consumer Price Index
    - DTWEXBGS: Trade Weighted U.S. Dollar Index
    - T10Y2Y: 10-Year Treasury Minus 2-Year Treasury (Yield Curve)
    """
    try:
        from fredapi import Fred
        fred = Fred(api_key=api_key or os.getenv('FRED_API_KEY'))
        
        series = {
            'VIXCLS': 'VIX Index',
            'DGS10': '10-Year Treasury',
            'DGS2': '2-Year Treasury',
            'DGS30': '30-Year Treasury',
            'DFF': 'Fed Funds Rate',
            'T10Y2Y': 'Yield Curve (10Y-2Y)',
            'T10Y3M': 'Yield Curve (10Y-3M)',
            'UNRATE': 'Unemployment Rate',
            'DTWEXBGS': 'Dollar Index',
        }
        
        print(f"\n{'='*60}")
        print("HARVESTING FRED DATA")
        print('='*60)
        
        fred_data = {}
        for code, name in series.items():
            try:
                data = fred.get_series(code, observation_start=START_DATE)
                fred_data[code] = data
                print(f"  ✓ {name} ({code}): {len(data)} observations")
            except Exception as e:
                print(f"  ✗ {name} ({code}): {str(e)[:30]}")
        
        df = pd.DataFrame(fred_data)
        df.to_csv(DATA_DIR / "FRED_ECONOMIC_DATA.csv")
        print(f"\n  ✓ Saved to FRED_ECONOMIC_DATA.csv")
        
        return df
        
    except ImportError:
        print("Note: fredapi not installed. Install with: pip install fredapi")
        return None


# ============================================================================
# VIX TERM STRUCTURE CALCULATOR
# ============================================================================

def calculate_vix_term_structure(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate VIX term structure metrics for hypothesis testing.
    
    Returns DataFrame with:
    - contango: VIX3M/VIX ratio (>1 = contango, <1 = backwardation)
    - term_structure_slope: Rolling slope of VIX curve
    - vix_percentile: VIX percentile rank (1-year)
    """
    vix = prices.get('VIX', prices.get('^VIX'))
    vix3m = prices.get('VIX3M', prices.get('^VIX3M'))
    
    if vix is None or vix3m is None:
        print("Error: VIX data not found")
        return None
    
    ts = pd.DataFrame(index=vix.index)
    
    # Contango ratio (VIX3M / VIX)
    ts['contango_ratio'] = vix3m / vix
    
    # Term structure state
    ts['term_structure'] = np.where(ts['contango_ratio'] > 1, 'CONTANGO', 'BACKWARDATION')
    
    # VIX percentile (1-year rolling)
    ts['vix_percentile'] = vix.rolling(252).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    
    # Contango percentile
    ts['contango_percentile'] = ts['contango_ratio'].rolling(252).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    
    # VIX momentum
    ts['vix_5d_change'] = vix.pct_change(5)
    ts['vix_20d_change'] = vix.pct_change(20)
    
    # Add raw values
    ts['vix'] = vix
    ts['vix3m'] = vix3m
    
    ts.to_csv(DATA_DIR / "VIX_TERM_STRUCTURE.csv")
    print(f"✓ VIX term structure calculated and saved")
    
    return ts


# ============================================================================
# YIELD CURVE CALCULATOR
# ============================================================================

def calculate_yield_curve(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate yield curve metrics for hypothesis testing.
    
    Returns DataFrame with:
    - yield_10y: 10-Year Treasury Yield
    - yield_2y: 5-Year Treasury Yield (proxy for 2Y)
    - yield_curve: 10Y - 5Y spread
    - curve_state: NORMAL, FLAT, or INVERTED
    """
    y10 = prices.get('TNX', prices.get('^TNX'))
    y30 = prices.get('TYX', prices.get('^TYX'))
    y5 = prices.get('FVX', prices.get('^FVX'))
    y3m = prices.get('IRX', prices.get('^IRX'))
    
    yc = pd.DataFrame(index=y10.index)
    
    yc['yield_10y'] = y10
    yc['yield_30y'] = y30
    yc['yield_5y'] = y5
    yc['yield_3m'] = y3m
    
    # Yield curve spreads
    yc['spread_10y_3m'] = y10 - y3m
    yc['spread_10y_5y'] = y10 - y5
    yc['spread_30y_10y'] = y30 - y10
    
    # Curve state
    yc['curve_state'] = np.where(yc['spread_10y_3m'] > 0.5, 'NORMAL',
                         np.where(yc['spread_10y_3m'] < -0.1, 'INVERTED', 'FLAT'))
    
    # Curve slope change (momentum)
    yc['curve_momentum'] = yc['spread_10y_3m'].diff(20)
    
    yc.to_csv(DATA_DIR / "YIELD_CURVE.csv")
    print(f"✓ Yield curve metrics calculated and saved")
    
    return yc


# ============================================================================
# SECTOR ROTATION CALCULATOR
# ============================================================================

def calculate_sector_rotation(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate sector relative strength for rotation hypothesis.
    """
    sectors = ['XLF', 'XLK', 'XLV', 'XLE', 'XLI', 'XLP', 'XLY', 'XLU', 'XLB', 'XLRE', 'XLC']
    
    sector_data = {}
    for s in sectors:
        if s in prices.columns:
            sector_data[s] = prices[s]
    
    sector_df = pd.DataFrame(sector_data)
    
    # Calculate returns
    returns = sector_df.pct_change()
    
    # Relative strength vs SPY
    if 'SPY' in prices.columns:
        spy = prices['SPY']
        spy_ret = spy.pct_change()
        
        rs = pd.DataFrame()
        for s in sector_data.keys():
            rs[f'{s}_vs_SPY'] = returns[s] - spy_ret
    
    # Momentum scores
    mom = pd.DataFrame()
    for s in sector_data.keys():
        mom[f'{s}_mom_20d'] = sector_df[s].pct_change(20)
        mom[f'{s}_mom_60d'] = sector_df[s].pct_change(60)
    
    # Z-scores for pairs
    pairs = [('XLF', 'XLK'), ('XLE', 'XLV'), ('XLY', 'XLP'), ('XLI', 'XLU')]
    
    zscore = pd.DataFrame(index=sector_df.index)
    for s1, s2 in pairs:
        if s1 in sector_df.columns and s2 in sector_df.columns:
            ratio = sector_df[s1] / sector_df[s2]
            rolling_mean = ratio.rolling(60).mean()
            rolling_std = ratio.rolling(60).std()
            zscore[f'{s1}_{s2}_zscore'] = (ratio - rolling_mean) / rolling_std
    
    zscore.to_csv(DATA_DIR / "SECTOR_PAIR_ZSCORES.csv")
    print(f"✓ Sector rotation metrics calculated and saved")
    
    return zscore


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Free Data Harvester")
    parser.add_argument('--full', action='store_true', help='Harvest all data')
    parser.add_argument('--vix', action='store_true', help='VIX data only')
    parser.add_argument('--fred', action='store_true', help='FRED economic data')
    parser.add_argument('--metrics', action='store_true', help='Calculate derived metrics')
    
    args = parser.parse_args()
    
    if args.full or not any(vars(args).values()):
        # Default: harvest everything
        all_data = harvest_all()
        
        # Load combined data
        prices, returns = load_harvested_data()
        
        # Calculate derived metrics
        vix_ts = calculate_vix_term_structure(prices)
        yc = calculate_yield_curve(prices)
        sectors = calculate_sector_rotation(prices)
        
        print("\n" + "="*80)
        print("DATA HARVEST COMPLETE - READY FOR HYPOTHESIS TESTING")
        print("="*80)
    
    elif args.vix:
        create_data_directory()
        harvest_category('volatility', TICKER_UNIVERSE['volatility'])
    
    elif args.fred:
        create_data_directory()
        harvest_fred_data()
    
    elif args.metrics:
        prices, returns = load_harvested_data()
        calculate_vix_term_structure(prices)
        calculate_yield_curve(prices)
        calculate_sector_rotation(prices)
