# LAB 01 - COMPLETE TEST SUITE
## ALL YOUR PHILOSOPHIES + MY ADDITIONS (Past Week Synthesized)

**Date:** December 16, 2025  
**Mission:** Test EVERYTHING you've researched. No human bias. Data decides.

---

## 🎯 YOUR PHILOSOPHIES (From Past Week)

### 1. **Scanner 1: Volume Breakout** (Your Original)
- **Trigger:** Volume >5x + Price >3% (lowered from 20x/5% - too strict)
- **Philosophy:** "Extreme volume = insider knowledge"
- **Source:** DAY4_CLUE_LOG.md - Your core pattern
- **Test:** Forward lag (Day N signal → Enter N+1 → Measure N+2)

### 2. **Pre-Event Volume** (Your Discovery)
- **Trigger:** 2x+ volume for 2+ days, price flat <3%
- **Philosophy:** "Volume precedes price - catch accumulation phase"
- **Source:** DAY4 - "47% of big moves had volume spike BEFORE price"
- **Test:** Does this actually predict next-day breakout?

### 3. **Repeat Winners** (Volatility Clustering)
- **Trigger:** Tickers with 3+ big moves in 12 months
- **Philosophy:** "Hot money follows hot stocks"
- **Source:** DAY4 - "Some tickers move multiple times per year"
- **Test:** Does being a repeat mover predict next move?

### 4. **Down 3 Days Reversal** (Mean Reversion)
- **Trigger:** Buy after 3 consecutive losing days
- **Philosophy:** "Oversold bounces - retail capitulates"
- **Test:** Win rate on day 4 bounce

### 5. **Fade The Spike** (Contrarian)
- **Trigger:** Gap up >5% → Short/avoid
- **Philosophy:** "Retail FOMO gets trapped"
- **Test:** Do gap-ups fade or continue?

### 6. **Follow The Leader** (Sector Lag)
- **Trigger:** When IONQ spikes >8%, buy RGTI/QMCO next day
- **Philosophy:** "Leaders signal, followers lag"
- **Source:** Your sector correlation idea (KILLED on broad sectors, but TEST on tight sub-sectors)
- **Test:** quantum_hardware, bitcoin_miners, gene_editing

### 7. **Sympathy Plays** (Correlation Arbitrage)
- **Trigger:** QS pops → Buy SLDP (laggard)
- **Philosophy:** "Shit rolls downhill - sector correlation edge"
- **Source:** SYMPATHY_PLAY_STRATEGY.md
- **Test:** When leader +10%, does laggard catch up in 2-3 days?

### 8. **Sector Momentum** (Hot Sectors Stay Hot)
- **Trigger:** When quantum is hot, stay in quantum
- **Philosophy:** "Momentum clustering"
- **Test:** Multi-day sector correlation

---

## 🔬 MY ADDITIONS (What I See That You're Missing)

### 9. **Time-of-Day Patterns** (NOT TESTED YET)
- **Hypothesis:** Small caps move differently morning vs afternoon
- **Source:** DAY4 - "Time-of-day patterns for penny stock volatility?"
- **Test:** Do first-hour signals outperform last-hour?
- **MY INSIGHT:** You have the data but haven't split by time

### 10. **Extreme Volume Events** (>100x)
- **Hypothesis:** Volume >100x predicts HUGE moves
- **Source:** DAY4 - "Discovered in exploration"
- **Test:** What % of >100x volume events are profitable?
- **MY INSIGHT:** This is different from Scanner 1 (>5x) - extreme outliers

### 11. **Gap Continuation vs Reversal**
- **Hypothesis:** Size of gap matters
- **Test:** Small gaps (<3%) reverse, big gaps (>7%) continue?
- **Source:** DAY4 - "Gap continuation rate by size/volume"
- **MY INSIGHT:** Not all gaps are equal - test thresholds

### 12. **Options Flow Precursor** (From Sympathy Strategy)
- **Hypothesis:** Unusual options volume predicts moves
- **Source:** SYMPATHY - "MSOS Calls bought BEFORE Trump news"
- **Test:** Can we detect this with free data? (Options volume ratio)
- **MY INSIGHT:** You mentioned it but never tested it

### 13. **Combination Scoring** (Multi-Signal)
- **Hypothesis:** 4-6 aligned signals > single signal
- **Source:** DAY4 - "Professional quant approach"
- **Test:** Score each ticker (volume + price + gaps + repeat winner)
- **MY INSIGHT:** This is YOUR edge - test score thresholds (3/4/5/6 points)

### 14. **Insider/Whale Detection** (Dark Pool Volume)
- **Hypothesis:** Large block trades = institutional accumulation
- **Source:** SYMPATHY - "THH had dark pool spike before +61%"
- **Test:** Can we detect "The Whale Whisper" with volume clustering?
- **MY INSIGHT:** Silent movers + block size analysis

### 15. **Sector Contrarians** (Already in notebook)
- **Hypothesis:** Tickers that win when their sub-sector loses
- **Source:** Your original idea
- **Test:** Find negative correlation plays

---

## 📊 TEST UNIVERSE STRATEGY

### HIGH_VOLATILITY_MOVERS (~130 tickers)
- Quantum, crypto, biotech, space, AVs, small caps
- These ACTUALLY move (not AAPL, banks)

### ALPHA_76 PREMIUM (76 tickers)
- From ALPHA_76_SECTOR_RESEARCH.md
- ARK-validated, catalyst-rich
- Institutional backing

### COMBINED UNIVERSE (190 tickers)
- All of the above
- Broader coverage, more signals

---

## 🎯 EXECUTION PRIORITY

### PHASE 1: Core Tests (Run First)
1. Scanner 1 (Volume >5x, Price >3%)
2. Pre-Event Volume (2x for 2+ days, flat price)
3. Repeat Winners (3+ moves in 12mo)
4. Down 3 Days Reversal
5. Fade The Spike
6. Pattern Discovery (40+ features, data-driven)

### PHASE 2: Sector Tests
7. Follow The Leader (sub-sectors)
8. Sympathy Plays (leader/laggard pairs)
9. Sector Momentum
10. Sector Contrarians

### PHASE 3: Advanced Signals
11. Extreme Volume (>100x)
12. Gap Analysis (by size)
13. Combination Scoring (multi-signal)
14. Silent Movers / Whale Detection
15. Anomaly Hunting (reverse momentum)

---

## ✅ SUCCESS CRITERIA

**WINNER:** Win rate >55%, Avg return >2%
**MAYBE:** Win rate 50-55% (needs refinement)
**LOSER:** Win rate <50% (discard immediately)

**TOP 50:** Best tickers across ALL tests (multi-dimensional scoring)

---

## 🥊 CUS D'AMATO TRAINING PHILOSOPHY

1. **Test Everything** - No sacred cows
2. **Kill What Fails** - Be ruthless
3. **Keep What Works** - Build arsenal
4. **Multi-Dimensional** - One dimension isn't enough (your key insight)
5. **Data Decides** - Not opinions
6. **Learn The Enemy** - Market behavior, not our assumptions

---

## 📁 OUTPUT

1. **LAB_01_RESULTS.json** - All test results
2. **TOP_50_BATTLE_READY.txt** - Best tickers
3. **WINNERS.txt** - Strategies that work (>55% WR)
4. **LOSERS.txt** - Strategies to kill (<50% WR)
5. **MAYBE.txt** - Strategies to refine (50-55% WR)

This feeds into LAB 0 (Master Synthesis) for final integration.

---

## 🔥 MY VALUE-ADD

**What I'm bringing that you didn't ask for:**

1. **Time-of-day analysis** - Your data has timestamps, use them
2. **Gap size thresholds** - Not all gaps are equal
3. **Extreme volume separate test** - >100x is different from >5x
4. **Options flow proxy** - Volume clustering patterns
5. **Combination scoring** - Multi-signal approach (your DAY4 idea, not implemented yet)
6. **Sub-sector precision** - KDK ≠ robotaxis (your insight, I'm executing it)
7. **Systematic execution flow** - Order matters for testing

**This is the complete synthesis of your week. Every idea. Tested unbiasedly.**
