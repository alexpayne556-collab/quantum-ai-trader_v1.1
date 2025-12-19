#!/bin/bash
# Auto-monitor download completion

echo "🔍 Monitoring download completion..."
echo "📊 Will notify when done and create DOWNLOAD_COMPLETE.txt"
echo ""

while true; do
    # Check if download process is still running
    if ! pgrep -f "START_DOWNLOAD_VERBOSE.py" > /dev/null; then
        echo ""
        echo "✅✅✅ DOWNLOAD COMPLETE! ✅✅✅"
        echo ""
        
        # Get final stats
        python -c "
import sqlite3
conn = sqlite3.connect('data/market_data.db')
tickers = conn.execute('SELECT COUNT(DISTINCT ticker) FROM ohlcv').fetchone()[0]
rows = conn.execute('SELECT COUNT(*) FROM ohlcv').fetchone()[0]
conn.close()

print(f'📊 Final: {tickers} tickers')
print(f'📈 Total: {rows:,} rows')
print(f'💾 Database: data/market_data.db')
print(f'')
print(f'🎯 READY FOR ANALYSIS!')
" > DOWNLOAD_COMPLETE.txt
        
        cat DOWNLOAD_COMPLETE.txt
        echo ""
        echo "✅ Created: DOWNLOAD_COMPLETE.txt"
        echo ""
        
        # Show next steps
        echo "📋 NEXT STEPS:"
        echo "   1. Run: python data_quality_audit.py"
        echo "   2. Review: data/*_issues.csv"
        echo "   3. Begin analysis"
        echo ""
        
        break
    fi
    
    # Show current progress every check
    python -c "
import sqlite3
conn = sqlite3.connect('data/market_data.db')
current = conn.execute('SELECT COUNT(DISTINCT ticker) FROM ohlcv').fetchone()[0]
conn.close()
total = 10986
pct = (current / total) * 100
remaining = total - current
print(f'⏳ Progress: {current}/{total} ({pct:.1f}%) - {remaining} remaining', end='\r')
"
    
    sleep 30
done
