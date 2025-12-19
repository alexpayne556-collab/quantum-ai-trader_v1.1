#!/bin/bash
# QUICK_STATUS.sh - Check testing progress

echo "======================================================================="
echo "📊 TESTING STATUS CHECK"
echo "======================================================================="
echo ""

# Check if tests are running
if ps aux | grep -q "[D]EEP_FINANCIAL_PHYSICS"; then
    echo "✅ Tests are RUNNING"
    echo ""
    echo "Process:"
    ps aux | grep "[D]EEP_FINANCIAL_PHYSICS" | awk '{print "   PID: " $2 "  CPU: " $3 "%  MEM: " $4 "%  Runtime: " $10}'
    echo ""
else
    echo "⏸️  No tests currently running"
    echo ""
fi

# Show recent log output
if [ -f physics_run.log ]; then
    echo "📝 Latest log output (last 20 lines):"
    echo "-----------------------------------------------------------------------"
    tail -20 physics_run.log
    echo "-----------------------------------------------------------------------"
    echo ""
fi

# Show completed result files
echo "📁 Completed result files:"
ls -lh data/*COMPREHENSIVE.csv data/*_RESULTS.csv data/*_PHYSICS.csv 2>/dev/null | awk '{print "   " $9 " (" $5 ")"}'
echo ""

# Quick stats
if ls data/*COMPREHENSIVE.csv 2>/dev/null | head -1 > /dev/null; then
    echo "📊 Quick analysis:"
    python3 << 'EOF'
import pandas as pd
import glob

files = glob.glob('data/*COMPREHENSIVE.csv') + glob.glob('data/*_RESULTS.csv')
total = 0
sig = 0

for f in files:
    try:
        df = pd.read_csv(f)
        if 't_stat' in df.columns:
            total += len(df)
            sig += len(df[df['t_stat'].abs() > 3.0])
    except:
        pass

if total > 0:
    print(f"   Strategies tested: {total:,}")
    print(f"   Significant (t>3.0): {sig:,} ({sig/total*100:.1f}%)")
EOF
    echo ""
fi

echo "======================================================================="
echo "Commands:"
echo "  Monitor live:  tail -f physics_run.log"
echo "  Analyze now:   python3 analyze_all_results.py"
echo "  Run more:      python3 DEEP_FINANCIAL_PHYSICS.py [categories]"
echo "======================================================================="
echo ""
