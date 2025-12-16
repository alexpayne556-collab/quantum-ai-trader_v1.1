# 🥊 EXECUTION PLAN: 196 TICKER UNIVERSE TESTING

**Date:** December 15, 2025  
**Status:** Ready to Execute  
**Partner:** You (Cus) + Me (Mike)

---

## THE INSIGHT (You Were Right)

**Your Point:** "We can't test on 3 shitty tickers - our system might not find anything valuable"

**The Fix:** 
- OLD: Testing 28 tickers across 6 sectors
- NEW: Testing **196 tickers** from full universe
- THEN: Refine to **TOP 50** that actually win

---

## WHAT'S LOADED

### Ticker Sources:
1. **alpha_76_watchlist.txt** - 76 tickers (your core watchlist)
2. **data/new_tickers_found.txt** - 120 tickers (expanded universe)
3. **Total:** 196 unique tickers

### Sample Universe:
```
A, ABBV, ACHR, ADBE, AEVA, AKYA, ALKT, AMBA, AMPL, AMSC, AMZN...
KDK, QUBT, RGTI, IONQ, RKLB, ASTS, RIOT, MARA, CLSK, COIN...
```

---

## THE TESTS (All Running on 196 Tickers)

### 1. KDK Live Analysis (Real Event - Just Happened)
- **Cell 22-25:** KDK dip analysis
- Perplexity institutional detector
- DeepSeek 5-filter check
- Your gut vs AI comparison
- **Purpose:** Test on REAL event happening now

### 2. Down 3 Days Reversal (Ridiculous Hypothesis)
- **Cell 38:** Tests all 196 tickers
- Find 3-day losing streaks → day 4 bounce?
- Shows **TOP 10 tickers** for this strategy
- **Output:** Win rate, avg return, best performers

### 3. Fade The Spike (Contrarian)
- **Cell 36:** Tests all 196 tickers
- Gap up >5% → fade by next day?
- Shows **TOP 10 faders** (best to short/avoid)
- **Output:** Fade rate, avg return, worst performers

### 4. Scanner 1 + VIX (Hybrid - Your Gut + Fear)
- **Cell 34:** Tests all 196 tickers
- Your Scanner 1 (Volume >20x + Price >5%) + High VIX
- Shows **TOP 10 tickers** for combo
- **Output:** Scanner alone vs Scanner + VIX comparison

### 5. Scanner 1 Full Universe
- **Cell 20:** Tests all 196 tickers
- Volume >20x + Price >5% breakouts
- Shows **TOP 15 tickers** for Scanner 1
- **Output:** Win rate, signals per ticker

### 6. Follow The Leader (Your Sector Insight)
- **Cell 32:** Quantum, Crypto, EVs
- When leader spikes >8%, followers catch up?
- **Output:** Best leader-follower pairs

### 7. Sector Momentum Correlation
- **Cell 44-47:** Quantum, Crypto, EVs
- Do sector stocks move together?
- **Output:** Correlation scores, both-up days

---

## THE REFINEMENT (Cell 29: TOP 50 FINDER)

**What It Does:**
1. Collects results from ALL tests
2. Scores each ticker by:
   - Average win rate across strategies
   - Average return across strategies
   - Number of strategies it worked in
3. **Composite Score:** `(WinRate * 0.6) + (StrategyCount * 0.2) + (Return * 0.02)`
4. Ranks top 196 → Saves **TOP 50**

**Output File:** `TOP_50_BATTLE_READY.txt`

**What You Get:**
```
Rank  Ticker  Score    WinRate  AvgRet  Strategies  Tests
1     QUBT    0.823    67.5%    +4.2%   3           down3, scanner, vix
2     KDK     0.811    65.0%    +3.8%   3           fade, scanner, vix
...
50    XYZ     0.612    56.2%    +2.1%   2           scanner, down3
```

---

## EXECUTION ORDER

### Phase 1: Setup (Cells 1-7)
```python
# Run these first
1. Imports
2. Battle Plan (markdown)
3. Load FULL_UNIVERSE (196 tickers)
4. Data loader test
```

### Phase 2: KDK Live Event (Cells 22-25)
```python
# Test on real data happening NOW
- KDK price display
- Perplexity detector verdict
- DeepSeek filter check
- Your gut vs AI
```

### Phase 3: Institutional Detector Tests (Cells 9-11)
```python
# Original tests on KDK + QUBT
- Test Perplexity's detector
- Component breakdowns
```

### Phase 4: DeepSeek Filter Tests (Cells 14-16)
```python
# Test on RKLB + ASTS
- 5-filter forward testing
- Individual filter impact
```

### Phase 5: Mike Tyson Mode (Cells 40-38)
```python
# The big tests on 196 tickers (will take 10-20 min)
40. Results tracking system init
38. Down 3 Days Reversal (all 196)
36. Fade The Spike (all 196)
34. Scanner 1 + VIX (all 196)
20. Scanner 1 Full (all 196)
32. Follow The Leader (quantum, crypto)
```

### Phase 6: Sector Tests (Cells 44-47)
```python
# Correlation analysis
- Quantum sector momentum
- Crypto miners momentum
- EVs momentum
```

### Phase 7: Final Analysis (Cells 27, 29)
```python
27. FINAL SCOREBOARD (all test results)
29. TOP 50 FINDER (refined battle list)
```

---

## EXPECTED RUNTIME

**Fast Tests (Cells 1-25):** ~2-3 minutes
- Loads data
- KDK analysis
- Institutional detector
- DeepSeek filters

**Heavy Tests (Cells 38, 36, 34, 20):** ~15-20 minutes
- Processing 196 tickers
- 60 days of data each
- Multiple strategies
- Forward testing with lag

**Total Runtime:** ~20-25 minutes for full test suite

---

## OUTPUTS

### Files Created:
1. **TEST_RESULTS.json** - All hypothesis test results
2. **TEST_RESULTS.csv** - Same data in spreadsheet format
3. **TOP_50_BATTLE_READY.txt** - Refined ticker list (50 winners)

### Data You'll See:

**For Each Test:**
- Win rate (% of profitable trades)
- Avg return (mean % gain/loss)
- Sample size (number of signals)
- TOP 10-15 performers for that strategy
- VERDICT: WINNER / MAYBE / LOSER
- LESSON: What we learned

**Final Scoreboard:**
- All hypotheses ranked by win rate
- Winners to keep
- Maybes to refine
- Losers to drop
- Key lessons learned

**Top 50 List:**
- Ranked by composite score
- Shows which strategies each ticker wins at
- Ready for paper trading

---

## SUCCESS CRITERIA

### WINNERS (Keep These):
- Win rate >55%
- Avg return >2%
- Sample size >10 trades
- Appears in multiple strategies

### MAYBES (Refine These):
- Win rate 50-55%
- Avg return 1-2%
- Sample size 5-10 trades
- Shows promise but needs work

### LOSERS (Drop These):
- Win rate <50%
- Avg return <1%
- Inconsistent across strategies

---

## NEXT STEPS (After Testing)

### If 3+ Strategies are WINNERS:
1. Build hybrid system combining best components
2. Use TOP 50 ticker list
3. Paper trade on Alpaca
4. Target: 55-65% win rate

### If 1-2 Strategies are WINNERS:
1. Focus on refining those approaches
2. Test variations (thresholds, windows)
3. Add filters to improve win rate
4. Use TOP 50 for focused testing

### If 0 Strategies are WINNERS:
1. Analyze lessons learned from losers
2. Generate new hypotheses from insights
3. Test additional approaches
4. Expand universe further if needed

---

## THE PHILOSOPHY (From THE_PARTNERSHIP.md)

**You (Cus D'Amato):**
- Guide the testing strategy
- Point out what I miss
- Challenge with "prove it"
- Maintain discipline

**Me (Mike Tyson):**
- Execute tests systematically
- Save all results
- Learn from losers
- No ego - drop what fails
- Build from proven components

**Goal:** Find the 50 tickers that WIN with our strategies, then BATTLE.

**Mindset:** "Discipline is doing what you hate to do, but nonetheless doing it like you love it." - Mike Tyson

---

## READY TO EXECUTE

**Partner, the notebook is loaded with 196 tickers.**

**Just run the cells in order. Results auto-save. TOP 50 gets generated.**

**Then we KNOW what works, and we battle with the best.**

**Let's knock 'em out. 🥊**
