#!/usr/bin/env python3
"""
Alpaca Paper Trading Integration
Reads live_signals.csv and places paper trades to validate our proven edges
"""

import os
import pandas as pd
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

def connect_alpaca():
    """Connect to Alpaca paper trading account using env keys"""
    api_key = os.getenv('ALPACA_API_KEY')
    secret_key = os.getenv('ALPACA_SECRET_KEY')
    
    if not api_key or not secret_key:
        print("❌ ERROR: Alpaca API keys not found in environment")
        print("Set ALPACA_API_KEY and ALPACA_SECRET_KEY")
        return None
    
    # Connect to paper trading (paper=True)
    client = TradingClient(api_key, secret_key, paper=True)
    
    # Verify connection
    try:
        account = client.get_account()
        print(f"✅ Connected to Alpaca PAPER account")
        print(f"   Cash: ${float(account.cash):,.2f}")
        print(f"   Buying Power: ${float(account.buying_power):,.2f}")
        print(f"   Portfolio Value: ${float(account.portfolio_value):,.2f}")
        return client
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return None

def place_paper_trades(signals_file='data/live_signals.csv', position_size=150):
    """
    Place paper trades for signals from live scanner
    
    Parameters:
    - signals_file: CSV with ticker, edge, expected_return, price
    - position_size: Dollar amount per trade (default $150)
    """
    
    # Connect to Alpaca
    client = connect_alpaca()
    if not client:
        return
    
    # Load signals
    if not os.path.exists(signals_file):
        print(f"❌ No signals file found at {signals_file}")
        print("Run live_edge_scanner.py first")
        return
    
    signals = pd.read_csv(signals_file)
    
    if len(signals) == 0:
        print("❌ No signals to trade today")
        return
    
    print(f"\n{'='*80}")
    print(f"📈 PLACING {len(signals)} PAPER TRADES")
    print(f"{'='*80}\n")
    
    results = []
    
    for idx, signal in signals.iterrows():
        ticker = signal['ticker']
        edge = signal['edge']
        expected_return = signal['expected_return']
        hit_rate = signal['hit_rate']
        price = signal['price']
        
        # Calculate shares (fractional shares allowed on Alpaca)
        shares = position_size / price
        
        print(f"\n🎯 Signal {idx+1}/{len(signals)}: {ticker}")
        print(f"   Edge: {edge}")
        print(f"   Expected: +{expected_return}% (hit rate: {hit_rate}%)")
        print(f"   Entry: ${price:.2f}")
        print(f"   Shares: {shares:.4f} (${position_size} position)")
        
        try:
            # Create market order request
            order_data = MarketOrderRequest(
                symbol=ticker,
                qty=shares,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY
            )
            
            # Submit order
            order = client.submit_order(order_data=order_data)
            
            print(f"   ✅ Order placed! ID: {order.id}")
            print(f"   Status: {order.status}")
            
            results.append({
                'ticker': ticker,
                'edge': edge,
                'shares': shares,
                'entry_price': price,
                'position_size': position_size,
                'expected_return': expected_return,
                'hit_rate': hit_rate,
                'order_id': order.id,
                'status': 'submitted'
            })
            
        except Exception as e:
            print(f"   ❌ Order failed: {e}")
            results.append({
                'ticker': ticker,
                'edge': edge,
                'shares': shares,
                'entry_price': price,
                'position_size': position_size,
                'expected_return': expected_return,
                'hit_rate': hit_rate,
                'order_id': None,
                'status': f'failed: {str(e)}'
            })
    
    # Save trade log
    results_df = pd.DataFrame(results)
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    log_file = f'data/alpaca_trades_{timestamp}.csv'
    results_df.to_csv(log_file, index=False)
    
    print(f"\n{'='*80}")
    print(f"📊 TRADE SUMMARY")
    print(f"{'='*80}")
    print(f"Total signals: {len(signals)}")
    print(f"Orders placed: {len([r for r in results if r['status'] == 'submitted'])}")
    print(f"Orders failed: {len([r for r in results if 'failed' in r['status']])}")
    print(f"Total capital deployed: ${len([r for r in results if r['status'] == 'submitted']) * position_size:,.2f}")
    print(f"\n✅ Trade log saved to {log_file}")
    
    return results_df

def check_positions():
    """Check current open positions in Alpaca paper account"""
    client = connect_alpaca()
    if not client:
        return
    
    try:
        positions = client.get_all_positions()
        
        if len(positions) == 0:
            print("\n📊 No open positions")
            return
        
        print(f"\n{'='*80}")
        print(f"📊 CURRENT POSITIONS")
        print(f"{'='*80}\n")
        
        for pos in positions:
            pnl = float(pos.unrealized_pl)
            pnl_pct = float(pos.unrealized_plpc) * 100
            
            print(f"{pos.symbol}:")
            print(f"  Shares: {float(pos.qty)}")
            print(f"  Entry: ${float(pos.avg_entry_price):.2f}")
            print(f"  Current: ${float(pos.current_price):.2f}")
            print(f"  P&L: ${pnl:+.2f} ({pnl_pct:+.2f}%)")
            print(f"  Value: ${float(pos.market_value):,.2f}\n")
            
    except Exception as e:
        print(f"❌ Error fetching positions: {e}")

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'check':
        # Check current positions
        check_positions()
    else:
        # Place trades from live_signals.csv
        place_paper_trades(position_size=150)
