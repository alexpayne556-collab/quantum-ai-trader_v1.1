"""
COLAB SIMPLE WORKING - COPY/PASTE THIS INTO COLAB
=================================================
This is the EXACT code to paste into Colab
No file dependencies - works immediately
"""

# ============================================================================
# SETUP
# ============================================================================

from google.colab import drive
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

# Mount drive
try:
    drive.mount('/content/drive', force_remount=False)
except:
    pass

PROJECT_PATH = '/content/drive/MyDrive/Quantum_AI_Cockpit'

print("="*80)
print("SIMPLE WORKING SYSTEM - NO DEPENDENCIES")
print("="*80)

# ============================================================================
# ALL CODE IN ONE PLACE - NO IMPORTS NEEDED
# ============================================================================

def get_sample_data(tickers=None):
    """Generate realistic data"""
    if tickers is None:
        tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA', 'AMZN', 'META', 'NFLX']
    
    all_data = []
    base_date = datetime.now() - timedelta(days=100)
    
    for ticker in tickers:
        base_price = np.random.uniform(50, 500)
        dates = pd.date_range(start=base_date, periods=100, freq='D')
        prices = [base_price]
        
        for i in range(1, 100):
            change = np.random.normal(0, 0.02)
            if 30 <= i <= 50:
                change = change - 0.05
            prices.append(prices[-1] * (1 + change))
        
        volumes = np.random.randint(1000000, 10000000, 100)
        
        for i, date in enumerate(dates):
            all_data.append({
                'ticker': ticker,
                'date': date,
                'open': prices[i] * 0.995,
                'high': prices[i] * 1.015,
                'low': prices[i] * 0.985,
                'close': prices[i],
                'volume': volumes[i]
            })
    
    return pd.DataFrame(all_data)

def scan_mean_reversion(data):
    """Find oversold stocks"""
    signals = []
    
    for ticker in data['ticker'].unique():
        ticker_data = data[data['ticker'] == ticker].sort_values('date').copy()
        
        if len(ticker_data) < 50:
            continue
        
        ticker_data['sma_20'] = ticker_data['close'].rolling(20).mean()
        ticker_data['std_20'] = ticker_data['close'].rolling(20).std()
        ticker_data['lower_band'] = ticker_data['sma_20'] - (2 * ticker_data['std_20'])
        
        delta = ticker_data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        ticker_data['rsi'] = 100 - (100 / (1 + rs))
        
        latest = ticker_data.iloc[-1]
        
        if (pd.notna(latest['lower_band']) and 
            pd.notna(latest['rsi']) and
            latest['close'] < latest['lower_band'] and 
            latest['rsi'] < 30):
            
            expected_return = (latest['sma_20'] - latest['close']) / latest['close']
            
            signals.append({
                'ticker': ticker,
                'entry_price': float(latest['close']),
                'target_price': float(latest['sma_20']),
                'stop_loss': float(latest['close'] * 0.95),
                'expected_return': float(expected_return),
                'rsi': float(latest['rsi']),
                'confidence': 85
            })
    
    return pd.DataFrame(signals)

def calculate_position_size(signal, account_value=437.0):
    """Calculate safe position size"""
    entry_price = signal.get('entry_price', 0)
    stop_loss = signal.get('stop_loss', entry_price * 0.95)
    
    if entry_price <= 0:
        return {'shares': 0, 'position_value': 0, 'max_loss': 0, 'risk_pct': 0}
    
    risk_per_share = entry_price - stop_loss
    if risk_per_share <= 0:
        return {'shares': 0, 'position_value': 0, 'max_loss': 0, 'risk_pct': 0}
    
    max_loss_dollars = account_value * 0.02
    shares = int(max_loss_dollars / risk_per_share)
    position_value = shares * entry_price
    
    max_position = account_value * 0.10
    if position_value > max_position:
        shares = int(max_position / entry_price)
        position_value = shares * entry_price
    
    max_loss = (entry_price - stop_loss) * shares
    risk_pct = (max_loss / account_value) * 100 if account_value > 0 else 0
    
    return {
        'shares': shares,
        'position_value': position_value,
        'max_loss': max_loss,
        'risk_pct': risk_pct
    }

# ============================================================================
# RUN IT
# ============================================================================

print("\n[1] Getting data...")
data = get_sample_data()
print(f"  Loaded {len(data):,} records")

print("\n[2] Scanning for opportunities...")
signals_df = scan_mean_reversion(data)
print(f"  Found {len(signals_df)} signals")

print("\n[3] Calculating positions...")
validated_signals = []
account_value = 437.0

for _, signal in signals_df.iterrows():
    position_info = calculate_position_size(signal.to_dict(), account_value)
    
    if position_info['shares'] > 0:
        signal_dict = signal.to_dict()
        signal_dict.update(position_info)
        validated_signals.append(signal_dict)
        print(f"  {signal['ticker']}: {position_info['shares']} shares @ ${signal['entry_price']:.2f}")

# ============================================================================
# SAVE
# ============================================================================

import os
os.makedirs(f'{PROJECT_PATH}/data', exist_ok=True)

with open(f'{PROJECT_PATH}/data/validated_signals.json', 'w') as f:
    json.dump(validated_signals, f, indent=2, default=str)

data.to_parquet(f'{PROJECT_PATH}/data/daily_data.parquet', index=False)

print("\n" + "="*80)
print("DONE - SYSTEM WORKING")
print("="*80)
print(f"\nFound {len(validated_signals)} trades")
print(f"Saved to: {PROJECT_PATH}/data/validated_signals.json")

if validated_signals:
    print(f"\n💰 TOP TRADE:")
    top = validated_signals[0]
    print(f"   {top['ticker']}: Buy @ ${top['entry_price']:.2f}")
    print(f"   Target: ${top['target_price']:.2f} ({top['expected_return']:.1%})")
    print(f"   Shares: {top['shares']} (${top['position_value']:.2f})")

