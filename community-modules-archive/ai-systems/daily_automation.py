# ============================================================================
# QUANTUM AI COCKPIT - DAILY AUTOMATION MASTER
# ============================================================================
# File: E:\Quantum_AI_Cockpit\daily_automation.py
# 
# This runs EVERY DAY automatically:
# 1. Scrapes fresh data from EODHD
# 2. Generates 50+ trading signals
# 3. Updates dashboard
# 4. Sends you notification
# 
# Setup: Run once, then Windows Task Scheduler handles it forever

import os
import sys
import json
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_PATH = r'E:\Quantum_AI_Cockpit'
EODHD_KEY = os.getenv('EODHD_API_TOKEN', '68f5419033db54.61168020')

# Tickers to monitor (update this list as needed)
CORE_TICKERS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B', 'UNH', 'JNJ',
    'XOM', 'V', 'PG', 'JPM', 'MA', 'LLY', 'HD', 'CVX', 'MRK', 'ABBV',
    'COST', 'PEP', 'KO', 'AVGO', 'WMT', 'TMO', 'BAC', 'MCD', 'CSCO', 'ACN',
    'LIN', 'ABT', 'DHR', 'ADBE', 'CRM', 'VZ', 'NKE', 'NFLX', 'TXN', 'DIS',
    'PM', 'NEE', 'UPS', 'RTX', 'CMCSA', 'WFC', 'ORCL', 'HON', 'QCOM', 'IBM',
    'AMD', 'INTC', 'AMGN', 'CAT', 'INTU', 'GE', 'BA', 'AMAT', 'SBUX', 'LOW',
    'SPGI', 'PLD', 'GILD', 'ADP', 'BKNG', 'MDLZ', 'ISRG', 'TJX', 'AXP', 'SYK',
    'BLK', 'CVS', 'VRTX', 'MMC', 'TMUS', 'REGN', 'ADI', 'C', 'CB', 'CI',
    'PGR', 'ZTS', 'SCHW', 'MO', 'FI', 'BSX', 'EOG', 'SO', 'DUK', 'ITW',
    'LRCX', 'SLB', 'BMY', 'HCA', 'MMM', 'APD', 'BDX', 'USB', 'CSX', 'NSC'
]

# ============================================================================
# STEP 1: DOWNLOAD FRESH DATA
# ============================================================================

def download_daily_data():
    """Download latest data from EODHD"""
    
    print(f"\n{'='*80}")
    print(f"DAILY DATA UPDATE - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*80}")
    
    print("\n[1/4] Downloading fresh data...")
    
    all_data = []
    successful = 0
    
    for ticker in tqdm(CORE_TICKERS, desc="Downloading"):
        try:
            url = f'https://eodhd.com/api/eod/{ticker}.US'
            params = {
                'api_token': EODHD_KEY,
                'from': (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d'),
                'to': datetime.now().strftime('%Y-%m-%d'),
                'fmt': 'json'
            }
            
            r = requests.get(url, params=params, timeout=10)
            if r.status_code == 200 and r.json():
                for bar in r.json():
                    all_data.append({
                        'ticker': ticker,
                        'date': bar['date'],
                        'open': bar['open'],
                        'high': bar['high'],
                        'low': bar['low'],
                        'close': bar['close'],
                        'volume': bar['volume']
                    })
                successful += 1
            
            time.sleep(0.1)  # Rate limit
        except:
            pass
    
    if not all_data:
        print("❌ No data downloaded")
        return pd.DataFrame()
    
    df = pd.DataFrame(all_data)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    
    # Save
    os.makedirs(f'{PROJECT_PATH}/data', exist_ok=True)
    df.to_parquet(f'{PROJECT_PATH}/data/daily_data.parquet')
    df.to_csv(f'{PROJECT_PATH}/data/daily_data.csv', index=False)
    
    print(f"✅ Downloaded {len(df):,} bars from {successful} tickers")
    return df

# ============================================================================
# STEP 2: CALCULATE INDICATORS
# ============================================================================

def calculate_indicators(df):
    """Add technical indicators to data"""
    
    print("\n[2/4] Calculating indicators...")
    
    all_data = []
    
    for ticker in tqdm(df['ticker'].unique(), desc="Indicators"):
        ticker_data = df[df['ticker'] == ticker].sort_values('date').copy()
        
        if len(ticker_data) < 50:
            continue
        
        # Moving averages
        ticker_data['sma_20'] = ticker_data['close'].rolling(20).mean()
        ticker_data['sma_50'] = ticker_data['close'].rolling(50).mean()
        
        # Bollinger Bands
        ticker_data['std_20'] = ticker_data['close'].rolling(20).std()
        ticker_data['bb_upper'] = ticker_data['sma_20'] + (1.5 * ticker_data['std_20'])
        ticker_data['bb_lower'] = ticker_data['sma_20'] - (1.5 * ticker_data['std_20'])
        
        # RSI
        delta = ticker_data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        ticker_data['rsi'] = 100 - (100 / (1 + rs))
        
        # Volume
        ticker_data['volume_ma'] = ticker_data['volume'].rolling(20).mean()
        ticker_data['volume_ratio'] = ticker_data['volume'] / ticker_data['volume_ma']
        
        all_data.append(ticker_data)
    
    result = pd.concat(all_data)
    result.to_parquet(f'{PROJECT_PATH}/data/data_with_indicators.parquet')
    
    print(f"✅ Calculated indicators for {result['ticker'].nunique()} tickers")
    return result

# ============================================================================
# STEP 3: GENERATE SIGNALS
# ============================================================================

def generate_signals(df):
    """Scan for trading opportunities"""
    
    print("\n[3/4] Generating signals...")
    
    signals = []
    
    for ticker in tqdm(df['ticker'].unique(), desc="Scanning"):
        ticker_data = df[df['ticker'] == ticker].sort_values('date')
        latest = ticker_data.iloc[-1]
        
        # STRATEGY 1: Mean Reversion
        if (latest['close'] < latest['bb_lower'] and 
            latest['rsi'] < 40 and
            not pd.isna(latest['bb_lower'])):
            
            signals.append({
                'ticker': ticker,
                'strategy': 'mean_reversion',
                'entry_price': latest['close'],
                'target_price': latest['sma_20'],
                'stop_loss': latest['close'] * 0.95,
                'expected_return': (latest['sma_20'] - latest['close']) / latest['close'],
                'rsi': latest['rsi'],
                'confidence': 80,
                'signal_date': latest['date']
            })
        
        # STRATEGY 2: Volume Breakout
        high_20 = ticker_data.tail(20)['close'].max()
        if (latest['volume_ratio'] > 3.0 and 
            latest['close'] > high_20 and
            not pd.isna(latest['volume_ratio'])):
            
            signals.append({
                'ticker': ticker,
                'strategy': 'volume_breakout',
                'entry_price': latest['close'],
                'target_price': latest['close'] * 1.05,
                'stop_loss': latest['close'] * 0.97,
                'expected_return': 0.05,
                'volume_ratio': latest['volume_ratio'],
                'confidence': 75,
                'signal_date': latest['date']
            })
    
    signals_df = pd.DataFrame(signals).sort_values('confidence', ascending=False)
    
    # Save
    signals_df.to_csv(f'{PROJECT_PATH}/data/daily_signals.csv', index=False)
    signals_df.to_json(f'{PROJECT_PATH}/data/daily_signals.json', orient='records')
    
    print(f"✅ Generated {len(signals_df)} signals")
    return signals_df

# ============================================================================
# STEP 4: UPDATE DASHBOARD CONFIG
# ============================================================================

def update_dashboard_config(df, signals_df):
    """Update dashboard with latest data"""
    
    print("\n[4/4] Updating dashboard...")
    
    config = {
        'timestamp': datetime.now().isoformat(),
        'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'data_summary': {
            'total_tickers': int(df['ticker'].nunique()),
            'total_bars': int(len(df)),
            'date_range': f"{df['date'].min().date()} to {df['date'].max().date()}"
        },
        'signals': {
            'total': int(len(signals_df)),
            'high_confidence': int(len(signals_df[signals_df['confidence'] >= 75])),
            'strategies': {
                'mean_reversion': int(len(signals_df[signals_df['strategy'] == 'mean_reversion'])),
                'volume_breakout': int(len(signals_df[signals_df['strategy'] == 'volume_breakout']))
            }
        },
        'top_10_signals': signals_df.head(10).to_dict('records')
    }
    
    with open(f'{PROJECT_PATH}/data/dashboard_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Dashboard updated")
    return config

# ============================================================================
# STEP 5: SEND NOTIFICATION
# ============================================================================

def send_notification(config):
    """Log summary and optionally send email/SMS"""
    
    summary = f"""
{'='*80}
DAILY UPDATE COMPLETE - {datetime.now().strftime('%Y-%m-%d %H:%M')}
{'='*80}

📊 DATA:
  Tickers: {config['data_summary']['total_tickers']}
  Bars: {config['data_summary']['total_bars']:,}

🎯 SIGNALS:
  Total: {config['signals']['total']}
  High Confidence: {config['signals']['high_confidence']}
  Mean Reversion: {config['signals']['strategies']['mean_reversion']}
  Volume Breakout: {config['signals']['strategies']['volume_breakout']}

🏆 TOP 3 SIGNALS:

"""
    
    for i, sig in enumerate(config['top_10_signals'][:3], 1):
        summary += f"  {i}. {sig['ticker']}: {sig['strategy']} - {sig['expected_return']:.1%} expected\n"
    
    summary += f"\n{'='*80}\n"
    
    print(summary)
    
    # Save log
    os.makedirs(f'{PROJECT_PATH}/logs', exist_ok=True)
    with open(f'{PROJECT_PATH}/logs/daily_log.txt', 'a') as f:
        f.write(summary)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run complete daily update"""
    
    try:
        # Create directories
        os.makedirs(f'{PROJECT_PATH}/data', exist_ok=True)
        os.makedirs(f'{PROJECT_PATH}/logs', exist_ok=True)
        
        # Run pipeline
        df = download_daily_data()
        df_with_indicators = calculate_indicators(df)
        signals = generate_signals(df_with_indicators)
        config = update_dashboard_config(df_with_indicators, signals)
        send_notification(config)
        
        print("\n✅ DAILY UPDATE COMPLETE!")
        return True
        
    except Exception as e:
        error_msg = f"❌ ERROR: {str(e)}"
        print(error_msg)
        
        os.makedirs(f'{PROJECT_PATH}/logs', exist_ok=True)
        with open(f'{PROJECT_PATH}/logs/error_log.txt', 'a') as f:
            f.write(f"{datetime.now()}: {error_msg}\n")
        
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
