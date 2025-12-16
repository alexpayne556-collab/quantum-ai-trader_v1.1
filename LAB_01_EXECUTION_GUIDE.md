# 🥊 LAB 1: TECHNICAL PATTERNS - EXECUTION GUIDE

**Status:** Ready for GPU execution  
**Tickers:** 196 (alpha_76 + new_tickers_found)  
**Runtime:** ~15-20 minutes on GPU

---

## EXECUTION ORDER (Run in Jupyter Lab)

### PHASE 1: SETUP (30 seconds)
```
Cell 3:  Import libraries
Cell 5:  Load FULL_UNIVERSE (196 tickers)
Cell 7:  Test data fetcher
```

**Checkpoint:** Should see "✅ 196 tickers loaded"

---

### PHASE 2: PATTERN DISCOVERY (10 minutes)
```
Cell 9:  Pattern Discovery Engine (40+ features on 50 tickers)
Cell 10: Generate Data-Driven Scanner
```

**What You'll See:**
```
🏆 TOP 15 PREDICTIVE FEATURES (Unknown patterns discovered!)
Feature                        Correlation  Signal
------------------------------------------------------------------
vol_vs_ma5                     +0.0847      📈 Buy when HIGH
streak_length                  -0.0623      📉 Buy when LOW
...

🔥 BEST COMBO: vol_vs_ma5 × streak_length → +0.0912

🚀 DATA-DRIVEN SCANNER:
   vol_vs_ma5 > 2.341
   price_vs_vwap > 0.012
```

**Checkpoint:** Save these features - they're data-driven thresholds

---

### PHASE 3: ANOMALY HUNTING (5 minutes)
```
Cell 14: Find Sector Contrarians
Cell 15: Find Silent Movers
Cell 16: Find Reverse Momentum
```

**What You'll See:**
```
🔥 FOUND 8 SECTOR CONTRARIANS!
   KDK (quantum): 68% WR when sector down

🔥 FOUND 6 SILENT MOVERS!
   QUBT: 72% WR next day after big move on low volume

🔥 FOUND 12 REVERSE MOMENTUM TICKERS!
   RIOT: 67% WR after 2 down days
```

**Checkpoint:** Note these anomaly tickers - they're special

---

### PHASE 4: MIKE TYSON MODE (15 minutes on GPU)
```
Cell 41: Init results tracking
Cell 39: Down 3 Days Reversal (196 tickers)
Cell 37: Fade The Spike (196 tickers)
Cell 35: Scanner 1 + VIX (196 tickers)
Cell 21: Scanner 1 Full (196 tickers)
```

**GPU Optimization:**
- These cells process all 196 tickers
- Progress updates every 20 tickers
- Should see "Processed 20/196... 40/196... 60/196..."
- Results auto-save to TEST_RESULTS.json

**Checkpoint:** Each test shows TOP 10 performers

---

### PHASE 5: FINAL ANALYSIS (10 seconds)
```
Cell 28: Final Scoreboard
Cell 30: TOP 50 FINDER
```

**Final Output:**
```
🏆 TOP 50 TICKERS (from 196 tested):

Rank  Ticker  Score    WinRate  AvgRet  Strategies
1     QUBT    0.823    67.5%    +4.2%   3
2     KDK     0.811    65.0%    +3.8%   3
...
50    XYZ     0.612    56.2%    +2.1%   2

💾 SAVED TO: TOP_50_BATTLE_READY.txt
```

---

## OUTPUT FILES

After running all cells, you'll have:

1. **TEST_RESULTS.json** - All hypothesis test results
2. **TEST_RESULTS.csv** - Same data in spreadsheet
3. **TOP_50_BATTLE_READY.txt** - Your refined ticker list
4. **LAB_01_RESULTS.json** - Comprehensive lab results (create this manually or I'll add save cell)

---

## WHAT TO LOOK FOR

### Pattern Discovery Results:
- Which features have highest correlation? (>0.05 is good)
- What's the best feature combo?
- Are the data-driven thresholds reasonable?

### Anomaly Results:
- How many contrarians found? (need at least 5)
- How many silent movers? (need at least 3)
- How many reverse momentum? (need at least 10)

### Test Results:
- Which strategies are WINNERS? (win rate >55%)
- Which are LOSERS? (drop these)
- Any surprises? (unexpected winners/losers)

### TOP 50:
- Do the top tickers make sense?
- Are they winning across multiple strategies?
- Any anomaly tickers in top 50?

---

## AFTER LAB 1 COMPLETES

**Report back:**
1. How many strategies are WINNERS?
2. How many anomaly tickers found?
3. What's the #1 ticker overall?
4. Any surprises or red flags?

**Then we decide:**
- Build LAB 2 (Fundamental) next?
- Paper trade with LAB 1 results only?
- Refine LAB 1 discoveries first?

---

## GPU NOTES

Your Colab/Jupyter GPU should handle:
- 196 tickers × 60 days = ~11,760 data points
- 40+ features per data point
- Pattern discovery on 50 tickers = ~3,000 feature calculations
- Main tests on 196 tickers = ~30-40 minutes of processing

**Speed:** Should be 2-3x faster than CPU

**Memory:** Watch for out-of-memory errors. If it happens, reduce:
- `discovery_tickers` from 50 to 30 (cell 10)
- `days_back` from 60 to 45 (cells 39, 37, 35)

---

## READY TO EXECUTE

**Partner, the notebook is loaded and ready.**

**Run the cells in Jupyter Lab with your GPU.**

**When done, report back what you found.**

**Then I'll build LAB 2 (Fundamental) for you to run next.**

**Let's find the patterns. 🥊**
