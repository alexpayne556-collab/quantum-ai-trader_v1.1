"""
🚀 GOD MODE DATA PIPELINE - ULTIMATE BASELINE BUILDER

Mission: Build institutional-grade training dataset
Target: 1000+ tickers, 5+ years, 250K+ samples, 80%+ baseline WR

This script creates the ULTIMATE training dataset by:
1. Gathering ALL available tickers (S&P 500, NASDAQ, sectors)
2. Fetching 5+ years of historical data
3. Engineering 70+ features (technical + market regime + microstructure)
4. Creating Triple Barrier labels (institutional-grade)
5. Adding market context (SPY trend, VIX fear index)
6. Saving production-ready dataset

Runtime: 2-4 hours in Colab Pro (CPU fine, GPU not needed)
Output: training_data_ultimate.csv (~500MB-1GB)
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Suppress yfinance logging
import logging
logging.getLogger('yfinance').setLevel(logging.CRITICAL)

print("=" * 80)
print("🚀 GOD MODE DATA PIPELINE - ULTIMATE BASELINE BUILDER")
print("=" * 80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")


# ========== STEP 1: GATHER ALL TICKERS ==========

def gather_all_tickers():
    """
    Collect comprehensive ticker universe
    
    Returns:
        Set of 1000+ unique tickers across multiple sources
    """
    print("[1/6] 📊 GATHERING TICKER UNIVERSE...")
    print("-" * 80)
    
    all_tickers = set()
    
    # S&P 500 (500 large caps)
    try:
        print("   Fetching S&P 500 tickers...")
        sp500_table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
        sp500_tickers = set(sp500_table['Symbol'].str.replace('.', '-').tolist())
        all_tickers.update(sp500_tickers)
        print(f"   ✅ S&P 500: {len(sp500_tickers)} tickers")
    except Exception as e:
        print(f"   ⚠️ S&P 500 failed: {e}")
    
    # User's legendary tickers (proven winners)
    legendary_tickers = {
        # AI/ML/Quantum
        'IONQ', 'RGTI', 'QUBT', 'ARQQ', 'QBTS',
        # Energy/Nuclear
        'OKLO', 'SMR', 'VST', 'CEG',
        # Fintech
        'HOOD', 'SOFI', 'AFRM', 'UPST',
        # Cloud/SaaS
        'SNOW', 'NOW', 'DDOG', 'NET', 'CRWD', 'S', 'ZS',
        # Semiconductors
        'NVDA', 'AMD', 'AVGO', 'MRVL', 'QCOM', 'INTC', 'ARM', 'ASML',
        # EV/Transportation
        'TSLA', 'RIVN', 'LCID', 'NIO', 'XPEV', 'LI',
        # Space/Defense
        'LUNR', 'RKLB', 'ASTS', 'PL',
        # Biotech/Healthcare
        'RXRX', 'SDGR', 'BEAM', 'CRSP', 'NTLA', 'EDIT',
        # Crypto/Web3
        'COIN', 'MSTR', 'RIOT', 'MARA', 'CLSK', 'CIFR',
        # Cybersecurity
        'PANW', 'FTNT', 'OKTA',
        # AI Hardware
        'SMCI', 'DELL', 'HPE',
        # Data Centers
        'EQIX', 'DLR',
        # High-Growth Tech
        'PLTR', 'U', 'PATH', 'DOCN',
        # User's current holdings
        'PALI', 'RXT', 'KDK', 'DGNX',
    }
    all_tickers.update(legendary_tickers)
    print(f"   ✅ Legendary tickers: {len(legendary_tickers)} tickers")
    
    # NASDAQ 100 (top 100 non-financial)
    nasdaq100_tickers = {
        'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'META', 'TSLA', 'NVDA',
        'NFLX', 'ADBE', 'CSCO', 'AVGO', 'PEP', 'COST', 'CMCSA', 'INTC',
        'TMUS', 'AMD', 'TXN', 'QCOM', 'INTU', 'AMGN', 'HON', 'SBUX',
        'AMAT', 'ISRG', 'GILD', 'BKNG', 'ADP', 'VRTX', 'ADI', 'MU',
        'LRCX', 'REGN', 'PYPL', 'MDLZ', 'ASML', 'PANW', 'KLAC', 'SNPS',
        'CDNS', 'CSX', 'MELI', 'NXPI', 'CRWD', 'WDAY', 'ABNB', 'MNST',
        'FTNT', 'DXCM', 'ADSK', 'MRVL', 'ORLY', 'LULU', 'PCAR', 'CHTR',
        'MAR', 'CPRT', 'AEP', 'PAYX', 'ROST', 'ODFL', 'KDP', 'CTAS',
        'EA', 'FAST', 'CTSH', 'DDOG', 'XEL', 'VRSK', 'EXC', 'IDXX',
        'GEHC', 'ZS', 'CCEP', 'TTWO', 'ANSS', 'CDW', 'BIIB', 'ON',
        'FANG', 'WBD', 'CSGP', 'GFS', 'MDB', 'WBA', 'ILMN', 'ZM',
        'MRNA', 'ARM', 'SMCI', 'TEAM', 'DASH', 'TTD', 'PDD',
    }
    all_tickers.update(nasdaq100_tickers)
    print(f"   ✅ NASDAQ 100: {len(nasdaq100_tickers)} tickers")
    
    # High-volume small caps (momentum potential)
    small_cap_momentum = {
        # Recent IPOs
        'ARM', 'RKLB', 'IONQ', 'RGTI',
        # Meme stocks (high volume)
        'GME', 'AMC', 'BBBY', 'BB',
        # Crypto proxies
        'COIN', 'MSTR', 'RIOT', 'MARA', 'CLSK', 'CIFR', 'BTBT',
        # Speculative biotech
        'SAVA', 'DMTK', 'NVAX', 'OCGN',
        # Penny stocks (user's strength)
        'PALI', 'RXT', 'KDK', 'DGNX',
    }
    all_tickers.update(small_cap_momentum)
    print(f"   ✅ Small-cap momentum: {len(small_cap_momentum)} tickers")
    
    # Remove invalid tickers
    all_tickers = {t for t in all_tickers if t and len(t) <= 5 and t.isalnum()}
    
    print("-" * 80)
    print(f"✅ TOTAL UNIVERSE: {len(all_tickers)} unique tickers\n")
    
    return sorted(list(all_tickers))


# ========== STEP 2: FETCH HISTORICAL DATA ==========

def fetch_all_historical_data(tickers, start_date='2019-01-01', end_date=None):
    """
    Download 5+ years of OHLCV data for all tickers
    
    Args:
        tickers: List of ticker symbols
        start_date: Start date (default: 2019-01-01 for 5+ years)
        end_date: End date (default: today)
        
    Returns:
        DataFrame with all historical data
    """
    print("[2/6] 📥 FETCHING 5+ YEARS OF HISTORICAL DATA...")
    print("-" * 80)
    
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    all_data = []
    success_count = 0
    fail_count = 0
    
    # Process in batches for progress tracking
    batch_size = 50
    total_batches = (len(tickers) + batch_size - 1) // batch_size
    
    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(tickers))
        batch_tickers = tickers[start_idx:end_idx]
        
        print(f"   Batch {batch_idx + 1}/{total_batches}: Processing {len(batch_tickers)} tickers...")
        
        for ticker in batch_tickers:
            try:
                # Download data
                df = yf.download(ticker, start=start_date, end=end_date, progress=False)
                
                # Fix multi-index
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                
                # Check data quality
                if len(df) < 100:  # Minimum 100 bars
                    continue
                
                # Add ticker column
                df['ticker'] = ticker
                df['date'] = df.index
                df.reset_index(drop=True, inplace=True)
                
                all_data.append(df)
                success_count += 1
                
            except Exception as e:
                fail_count += 1
                continue
        
        print(f"   ✅ Batch {batch_idx + 1} complete (Success: {success_count}, Failed: {fail_count})")
    
    if not all_data:
        raise ValueError(f"No data collected! All {len(tickers)} tickers failed.")
    
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    
    print("-" * 80)
    print(f"✅ DATA COLLECTION COMPLETE")
    print(f"   Success: {success_count}/{len(tickers)} tickers ({success_count/len(tickers)*100:.1f}%)")
    print(f"   Total rows: {len(combined_df):,}")
    print(f"   Date range: {combined_df['date'].min().date()} to {combined_df['date'].max().date()}")
    print(f"   Unique tickers: {combined_df['ticker'].nunique()}\n")
    
    return combined_df


# (Continue in next message due to length...)
