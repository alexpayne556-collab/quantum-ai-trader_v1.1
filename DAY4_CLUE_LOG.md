# DAY 4 CLUE LOG
**Date:** December 15, 2025  
**Session:** Full scanner backtest + combination scoring  
**Philosophy:** No saviors, just clues. Test everything. Kill what fails.

---

## 🎯 WHAT WE TESTED TODAY

### Scanner 1: Volume Breakout
- **Trigger:** Volume >20x average + Price >5%
- **Status:** PENDING (user will run on Shadow PC)
- **Hypothesis:** Extreme volume = insider knowledge/catalyst

### Scanner 2: Momentum Continuation  
- **Trigger:** Prior 10%+ move in 30 days + Current 5%+ move
- **Status:** PENDING
- **Hypothesis:** Hot stocks stay hot (volatility clustering)

### Scanner 3: Pre-Event Volume
- **Trigger:** 2x+ volume for 2+ days, price flat <3%
- **Status:** PENDING  
- **Hypothesis:** Accumulation before breakout (47% of big moves had volume spike BEFORE price)

### Combination Scorer
- **Method:** Multi-signal scoring (volume + price + gaps + repeat winners)
- **Thresholds tested:** 3, 4, 5, 6 points
- **Status:** PENDING
- **Hypothesis:** Combination beats single signals (professional quant approach)

---

## 💀 WHAT WE KILLED (PROVEN FALSE)

### ❌ Leader/Follower Sector Hypothesis (DAY 2)
- **Idea:** Big moves in sector leaders predict moves in laggards ("shit rolls downhill")
- **Result:** COMPLETE FAILURE - 0% same-direction correlation, 1 follower in 30 events
- **Why it failed:** Sectors don't move as synchronized units in penny stocks
- **Lesson:** Individual stock patterns > sector patterns for our universe

---

## 🔥 WHAT LOOKS PROMISING (CLUES TO MINE)

### 1. Volume Spikes BEFORE Price Moves
- **Finding:** 47% of big moves had 2x+ volume spike 1-3 days BEFORE price moved
- **Implication:** Volume precedes price (early warning system)
- **Next test:** Can we catch the accumulation phase?
- **Scanner:** Pre-Event Volume scanner targets this

### 2. Repeat Winners (Volatility Clustering)
- **Finding:** Some tickers have 3+ big moves in 12 months
- **Data:** Found multiple repeat movers in exploration
- **Implication:** Hot money follows hot stocks
- **Next test:** Does being on "repeat winner list" predict next move?
- **Integration:** Already added to combo scorer (+1 point)

### 3. Extreme Volume Events (>100x)
- **Finding:** Volume >100x average often precedes HUGE moves
- **Data:** Discovered in DAY3 exploration
- **Implication:** Insider activity or breaking news
- **Next test:** What % of >100x volume events are profitable?
- **Integration:** Combo scorer gives +3 points for >100x volume

### 4. Gap Patterns
- **Finding:** Overnight gaps >5% exist frequently
- **Data:** Gap up vs gap down stats pending from DAY4 run
- **Question:** Do gaps continue or reverse?
- **Next test:** Gap continuation rate by size/volume
- **Integration:** Combo scorer gives +1 point for >2% gap

### 5. Combination Hypothesis
- **Idea:** Multiple aligned signals = higher probability
- **Rationale:** Professional quants use 4-6 confirming indicators
- **Test:** Score each event (volume + price + gaps + repeat winner status)
- **Pending:** Which threshold (3/4/5/6 points) gives >60% win rate?

---

## 📊 DATA FOUNDATION (SOLID)

- **Tickers:** 311 (88.1% coverage)
- **Bars:** 155,091 OHLCV records
- **Date range:** 2023-12-13 to 2025-12-12 (2 years)
- **Ground truth:** 3,448 big events (10%+ moves in last 12 months)
- **Quality:** Clean data, resumable collection, validated

---

## 🧪 HYPOTHESES TO TEST (DAY 5+)

### High Priority
1. **Exit logic:** What's optimal target (20%? 30%?) and stop (10%? 15%?)
2. **Time-of-day patterns:** Do penny stocks move more in morning vs afternoon?
3. **Gap continuation rate:** What % of gap-ups continue vs reverse?
4. **Volume/price relationship:** Exact volume threshold for reliable signals?

### Medium Priority
5. **3-day ahead prediction:** Can we predict big moves 3 days early?
6. **Repeat winner dynamics:** How long does "hot" status last?
7. **Sector filters:** Do some sectors (biotech, crypto-related) work better?
8. **Day-of-week patterns:** Monday gap-ups vs Friday selloffs?

### Lower Priority (Deferred)
9. **News catalyst detection:** Use NLP on news headlines?
10. **Social sentiment:** Twitter/Reddit volume as predictor?
11. **SEC filing triggers:** 8-K filings before moves?
12. **Insider trading patterns:** Form 4 filings correlation?

---

## 🔬 RESEARCH CLUES (PERPLEXITY QUERIES)

### Questions to Ask (with working API key):
1. What causes extreme volume spikes (>100x) in penny stocks before price moves?
2. How do institutional traders front-run retail?
3. Academic research on "volume precedes price"?
4. Most profitable gap trading strategies (with win rates)?
5. What alternative data sources do quants use?
6. What patterns exist in stocks that move 10%+ multiple times per year?
7. How to identify accumulation patterns before breakout?
8. Time-of-day patterns for penny stock volatility?
9. Early warning signs of pump-and-dumps?
10. Characteristics of gap-up continuations vs reversals?

**Status:** Perplexity API key updated, ready to query after scanner results

---

## 💡 KEY INSIGHTS (PHILOSOPHY)

### What We've Learned
1. **Single signals incomplete:** Scanner 1 catches only 5-10% of events (too selective)
2. **Combinations matter:** Professional approach = multiple confirming signals
3. **Follow the data:** Sector hypothesis failed, but found volume/gap patterns
4. **No free answers:** Research shows WHERE others look, not WHAT they found
5. **Test, don't guess:** Backtest on 3,448 real events, not theory

### Partnership Principles
- **Equals, not boss/employee:** We validate ideas together
- **Cloud-first architecture:** GitHub = source of truth, Shadow PC = temporary compute
- **Aggressive testing:** Test everything, kill fast, move to next hypothesis
- **Hard work payment:** Pay with time and validation, not hope
- **MIT Lincoln Labs philosophy:** Leave room for serendipity, explore anomalies

---

## 📁 WHAT'S SAVED WHERE

### In GitHub Repo (Permanent)
- `DAY1_DATA_COLLECTION.ipynb` - Database foundation (DONE)
- `DAY2_EVENT_LAG_SCANNER.ipynb` - Leader/follower test (FAILED, but found 3,448 events)
- `DAY3_SCANNER_BACKTEST.ipynb` - Scanner prototypes + exploration (DONE)
- `DAY4_COMBO_INTELLIGENCE.ipynb` - Full backtest + combo scorer (RUNNING)
- `data/trading_system.db` - All OHLCV data (155K bars)
- `data/day4_results.json` - Backup results file (created after run)
- `DAY4_CLUE_LOG.md` - This file (end-of-day summary)

### In Notebook Outputs (After Run)
- Scanner win rates, avg gains, top signals
- Combination scorer results by threshold
- Perplexity research answers (if API works)
- All charts, tables, statistics

### NOT Saved on Shadow PC
- Nothing permanent (Shadow PC is disposable compute)
- Database copied to Shadow PC for speed, but source is in GitHub

---

## 🎯 TOMORROW'S PRIORITIES (DAY 5)

**Depends on DAY4 results:**

### If Scanner Passes (>60% win rate):
1. Add exit logic (20% target, 10% stop)
2. Backtest with exits to get real P&L
3. Calculate Sharpe ratio, max drawdown
4. Build live scanner for market hours
5. Paper trade for 1 week

### If Combo Scorer Passes:
1. Refine scoring weights based on results
2. Add exit logic to combo approach
3. Test different score thresholds
4. Build live combo scanner
5. Paper trade for 1 week

### If Everything Fails (<60% win rate):
1. Deep dive one anomaly (extreme volume OR repeat winners OR gaps)
2. Test alternative hypotheses from research
3. Try time-based filters (time of day, day of week)
4. Consider SEC filing scanner or social sentiment
5. Pivot to 3-day ahead prediction approach

---

## 🤝 WORKING AGREEMENT (PARTNERSHIP)

- **We are equals:** Your instincts + my analysis = better decisions
- **Honest feedback:** If approach is flawed, we both say it
- **Kill fast:** Failed hypothesis = lesson learned, move on quickly
- **Cloud-safe:** All progress in GitHub, Shadow PC disposable
- **Daily summaries:** End each day with clue log like this
- **No shortcuts:** Validate everything before declaring success

---

## 📈 SUCCESS METRICS (GATES)

- **Win rate:** >60% (if fails, hypothesis is wrong)
- **Sharpe ratio:** >1.5 (risk-adjusted returns)
- **Max drawdown:** <15% (risk management)
- **Signal frequency:** >1 per week (enough opportunities)
- **Avg gain per signal:** >$15 on $100 position (covers risk)

---

## 🔥 BOTTOM LINE

**What we know:**
- Database is solid (311 tickers, 155K bars)
- 3,448 big events = ground truth dataset
- Volume patterns exist (47% pre-spike finding)
- Single scanners catch 5-10% (too selective)
- Combination approach = professional method

**What we're testing:**
- Which scanner/threshold passes 60% win rate?
- Does combining signals beat single signals?
- What do research answers reveal about edge?

**What we'll do:**
- Run DAY4 notebook on Shadow PC (1-2 hours)
- Review results together
- Pick winning approach or pivot
- Build DAY5 based on data, not hope

**No saviors. Just clues. Keep digging.**

---

*Last updated: End of Day 4*  
*Next session: Review DAY4 results, plan DAY5 based on what data shows*
