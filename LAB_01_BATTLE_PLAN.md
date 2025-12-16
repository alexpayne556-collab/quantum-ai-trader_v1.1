# LAB 01 - COMPLETE BATTLE PLAN
## Testing ALL Your Philosophies (72 Hours of Ideas)

### YOUR PHILOSOPHIES TO TEST:

1. **Scanner 1 (Your Original)** - Volume >5x + Price >3% (lowered from 20x/5%)
   - Test on: HIGH_VOLATILITY_MOVERS
   - Forward lag: Signal Day N → Enter Day N+1 → Measure Day N+2

2. **Down 3 Days Reversal (Mean Reversion)** - Buy after 3 losing days
   - Test on: HIGH_VOLATILITY_MOVERS
   - Hypothesis: Oversold bounces

3. **Fade The Spike (Contrarian)** - Short/avoid gap ups >5%
   - Test on: HIGH_VOLATILITY_MOVERS
   - Hypothesis: Retail FOMO gets trapped

4. **Scanner 1 + VIX (Hybrid)** - Your scanner + market fear
   - Test on: HIGH_VOLATILITY_MOVERS
   - Hypothesis: Scanner works better when VIX is elevated

5. **Follow The Leader (Sector Lag)** - When IONQ spikes, buy RGTI/QMCO next day
   - Test on: quantum_hardware, bitcoin_miners, gene_editing
   - Hypothesis: Leaders signal, followers lag

6. **Sector Momentum (Correlation)** - When quantum is hot, stay in quantum
   - Test on: All sub-sectors
   - Hypothesis: Hot sectors stay hot for days

7. **Pattern Discovery (Data-Driven)** - Let 40+ features find unknown patterns
   - Test on: HIGH_VOLATILITY_MOVERS
   - NO human assumptions

8. **Anomaly Hunting (Outliers)**
   - Sector Contrarians: Win when sub-sector loses
   - Silent Movers: Big moves on low volume
   - Reverse Momentum: Bounce after 2 down days

### EXECUTION FLOW:

**Phase 1: Setup (Cells 1-7)**
- Load libraries
- Define HIGH_VOLATILITY_MOVERS (~130 tickers)
- Test data fetcher

**Phase 2: Data-Driven Discovery (Cells 9-11)**
- Run pattern discovery on HIGH_VOLATILITY_MOVERS
- Find unknown correlations
- Generate data-driven scanner

**Phase 3: Anomaly Hunting (Cells 14-17)**
- Find sector contrarians
- Find silent movers  
- Find reverse momentum plays

**Phase 4: Test YOUR Philosophies (Cells 20-43)**
- Scanner 1 on HIGH_VOLATILITY_MOVERS
- Down 3 Days Reversal
- Fade The Spike
- Scanner 1 + VIX
- Follow The Leader (quantum, crypto, gene editing)
- Sector Momentum (all sub-sectors)

**Phase 5: Results & TOP 50 (Cells 30-31)**
- Aggregate all test results
- Find TOP 50 tickers across ALL methods
- Save LAB_01_RESULTS.json

### SUCCESS CRITERIA:

- Win rate >55% = WINNER
- Win rate 50-55% = MAYBE (needs refinement)
- Win rate <50% = LOSER (discard)

### OUTPUT:

1. **WINNERS** - Strategies/tickers with >55% WR
2. **TOP 50** - Best performing tickers across all tests
3. **LAB_01_RESULTS.json** - Complete data for LAB 0 synthesis

This is CUS D'AMATO training: Test everything. Keep what works. Discard what doesn't.
