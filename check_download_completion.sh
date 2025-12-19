#!/bin/bash
# Monitor download and run analysis when complete

while true; do
    # Check if download process is still running
    if ! pgrep -f "START_DOWNLOAD_VERBOSE.py" > /dev/null; then
        echo "✅ Download complete! Running analysis..."
        
        # Run data quality audit
        python data_quality_audit.py
        
        # Show completion stats
        python -c "
import sqlite3
conn = sqlite3.connect('data/market_data.db')
tickers = conn.execute('SELECT COUNT(DISTINCT ticker) FROM ohlcv').fetchone()[0]
rows = conn.execute('SELECT COUNT(*) FROM ohlcv').fetchone()[0]
print(f'\n🎉 DOWNLOAD COMPLETE!')
print(f'📊 Final: {tickers} tickers, {rows:,} rows')
conn.close()
"
        break
    fi
    
    # Check progress every 30 seconds
    sleep 30
done
