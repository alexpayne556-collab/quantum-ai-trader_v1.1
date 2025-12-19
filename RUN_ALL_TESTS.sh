#!/bin/bash
# RUN_ALL_TESTS.sh - Complete comprehensive testing
# This will take 2-4 hours depending on your machine

set -e

echo "======================================================================="
echo "🔬 DEEP FINANCIAL PHYSICS - COMPREHENSIVE TESTING"
echo "======================================================================="
echo ""
echo "This will test 3,000+ strategies across all categories."
echo "Time estimate: 2-4 hours (or 20 min with Shadow PC GPU)"
echo ""
echo "Starting in 5 seconds... (Ctrl+C to cancel)"
sleep 5

cd /workspaces/quantum-ai-trader_v1.1

echo ""
echo "🚀 Starting comprehensive testing..."
echo "📝 Logging to: physics_complete_run.log"
echo ""

python3 DEEP_FINANCIAL_PHYSICS.py all 2>&1 | tee physics_complete_run.log

echo ""
echo "======================================================================="
echo "✅ TESTING COMPLETE!"
echo "======================================================================="
echo ""
echo "Results saved to data/*_COMPREHENSIVE.csv"
echo ""
echo "Next step: Analyze results with:"
echo "  python3 analyze_all_results.py"
echo ""
