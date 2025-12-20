#!/usr/bin/env python3
"""
SHADOW PC BATCH RUNNER - Batches 5-9
=====================================
Optimized for parallel execution on Shadow PC.

Run from conda prompt:
    cd C:\Users\alexf\quantum-ai-trader_v1.1
    python SHADOW_RUN_BATCHES_5_9.py

Accelerators enabled:
- NumPy MKL threading
- Parallel data downloads
- Caching to prevent re-downloads
"""

import os
import sys
import time
import warnings
from datetime import datetime

# Performance optimizations
os.environ['OMP_NUM_THREADS'] = '8'  # Use all cores for NumPy
os.environ['MKL_NUM_THREADS'] = '8'
os.environ['NUMEXPR_NUM_THREADS'] = '8'

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

# Try to use faster yfinance settings
try:
    import yfinance as yf
    yf.pdr_override()  # Use pandas_datareader for speed
except:
    pass

def main():
    print("="*70)
    print("SHADOW PC HYPOTHESIS TESTING - BATCHES 5-9")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # Import executor
    from hypothesis_batch_executor import HypothesisBatchExecutor
    
    # Create executor with caching
    executor = HypothesisBatchExecutor(
        data_cache_path='./hypothesis_data/',
        start_date='2010-01-01'
    )
    
    # Run batches 5-9
    batches_to_run = [5, 6, 7, 8, 9]
    
    all_results = []
    
    for batch_num in batches_to_run:
        print(f"\n{'='*60}")
        print(f"STARTING BATCH {batch_num}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        try:
            results = executor.run_batch(batch_num, save_results=True)
            all_results.append(results)
            
            elapsed = time.time() - start_time
            print(f"\n✅ Batch {batch_num} complete in {elapsed/60:.1f} minutes")
            print(f"   Passed: {results['pass'].sum()}/{len(results)}")
            
        except Exception as e:
            print(f"\n❌ Batch {batch_num} failed: {e}")
            continue
    
    # Combine all results
    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        combined.to_csv('./hypothesis_data/SHADOW_BATCHES_5_9_RESULTS.csv', index=False)
        
        print("\n" + "="*70)
        print("SHADOW PC TESTING COMPLETE!")
        print("="*70)
        print(f"Total hypotheses: {len(combined)}")
        print(f"Passed: {combined['pass'].sum()}")
        print(f"Pass rate: {100*combined['pass'].sum()/len(combined):.1f}%")
        
        # Top winners
        winners = combined[combined['pass']].nlargest(10, 'spread')
        if len(winners) > 0:
            print("\n🏆 TOP WINNERS (Batches 5-9):")
            print(winners[['hypothesis_id', 'name', 'spread', 'sharpe', 'p_value']].to_string(index=False))
        
        print(f"\nResults saved to: ./hypothesis_data/SHADOW_BATCHES_5_9_RESULTS.csv")
    
    print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
