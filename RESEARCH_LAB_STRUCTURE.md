# 🥊 QUANTITATIVE RESEARCH LAB STRUCTURE

**Date:** December 15, 2025  
**Philosophy:** Master ALL arts of the financial boxing ring

---

## THE DISCIPLINE (Cus D'Amato's Training System)

> "A boy comes to me with a spark of interest. I feed the spark and it becomes a flame. I feed the flame and it becomes a fire. I feed the fire and it becomes a roaring blaze." - Cus D'Amato

**Our Version:**
1. **Spark:** One hypothesis to test
2. **Flame:** One focused notebook to test it thoroughly
3. **Fire:** Multiple notebooks covering all dimensions
4. **Roaring Blaze:** Master notebook synthesizing ALL findings

---

## THE LAB STRUCTURE (6 Specialized Notebooks + 1 Master)

### LAB 1: TECHNICAL ANALYSIS LAB
**File:** `LAB_01_TECHNICAL_PATTERNS.ipynb` (current AI_COUNCIL notebook)

**What It Tests:**
- Pattern Discovery (40+ features: MAs, EMAs, SMAs, volume, volatility)
- Anomaly Hunting (contrarians, silent movers, reverse momentum)
- Technical indicators (RSI, MACD, Bollinger Bands)
- Chart patterns (breakouts, flags, triangles)

**Output:**
- Top technical patterns that predict winners
- Data-driven technical scanner
- Anomaly ticker lists

**Status:** ✅ Built (current notebook)

---

### LAB 2: FUNDAMENTAL ANALYSIS LAB
**File:** `LAB_02_FUNDAMENTAL_SIGNALS.ipynb` (NEW)

**What It Tests:**
- Earnings surprises (beat/miss rates, guidance)
- Revenue growth acceleration
- Margin expansion/compression
- Cash flow quality
- Balance sheet strength (debt ratios, current ratio)
- ROE, ROIC trends
- Fundamental momentum (improving metrics)

**Data Sources:**
- yfinance ticker.info (free fundamentals)
- Yahoo Finance earnings calendar
- SEC EDGAR filings (if needed)

**Output:**
- Which fundamental metrics predict winners?
- Fundamental-driven scanner (e.g., "earnings beat + margin expansion")
- Quality score for each ticker

**Hypothesis to Test:**
- Does revenue growth acceleration predict price moves?
- Do improving margins = continuation?
- Does cash flow quality beat earnings quality?

---

### LAB 3: SENTIMENT ANALYSIS LAB
**File:** `LAB_03_SENTIMENT_SIGNALS.ipynb` (NEW)

**What It Tests:**
- News sentiment (headlines, article volume)
- Social media sentiment (Reddit, Twitter/X mentions)
- Analyst ratings changes (upgrades/downgrades)
- Short interest changes
- Put/call ratio shifts
- Earnings call sentiment (if transcripts available)

**Data Sources:**
- News APIs (if available, else manual tracking)
- Reddit WallStreetBets mentions
- Yahoo Finance analyst ratings
- Finviz short interest data

**Output:**
- Does sentiment predict price moves?
- Contrarian indicators (high buzz = fade?)
- Sentiment momentum (improving sentiment = continuation?)

**Hypothesis to Test:**
- Does social buzz predict next-day moves?
- Do analyst upgrades work or are they lagging?
- Is high short interest bullish or bearish?

---

### LAB 4: MARKET STRUCTURE LAB
**File:** `LAB_04_MARKET_STRUCTURE.ipynb` (NEW)

**What It Tests:**
- Options flow (unusual activity, large sweeps)
- Dark pool volume (hidden institutional activity)
- Bid-ask spread (liquidity indicator)
- Float rotation (volume / float %)
- Institutional ownership changes
- Insider buying/selling

**Data Sources:**
- Options data (if available)
- Dark pool data (Finra ATS data)
- Yahoo Finance institutional holdings
- SEC Form 4 filings (insider transactions)

**Output:**
- Does options flow predict moves?
- Does dark pool volume = institutional accumulation?
- Does insider buying predict continuation?

**Hypothesis to Test:**
- Do unusual options = next-day move?
- Does high dark pool % predict stealth accumulation?
- Does insider buying cluster before breakouts?

---

### LAB 5: MACRO SIGNALS LAB
**File:** `LAB_05_MACRO_SIGNALS.ipynb` (NEW)

**What It Tests:**
- VIX level and changes (fear gauge)
- Treasury yield curve (rates impact)
- Sector rotation (which sectors leading?)
- SPY/QQQ correlation (market dependence)
- Breadth indicators (advance/decline, new highs/lows)
- Economic calendar events (CPI, FOMC, NFP)

**Data Sources:**
- ^VIX (yfinance)
- Treasury yields (yfinance: ^TNX, ^TYX)
- Sector ETFs (XLK, XLF, XLE, etc.)
- SPY, QQQ data
- Economic calendar (manual or API)

**Output:**
- Does VIX predict our tickers?
- Which tickers win in high-rate environments?
- Sector rotation signals (when to switch sectors)

**Hypothesis to Test:**
- Do our tickers win when VIX spikes?
- Does yield curve inversion change behavior?
- Can we predict sector rotation?

---

### LAB 6: CROSS-ASSET LAB
**File:** `LAB_06_CROSS_ASSET.ipynb` (NEW)

**What It Tests:**
- Crypto correlation (BTC, ETH impact on miners)
- Commodity correlation (oil → energy stocks)
- Forex correlation (DXY → exports/imports)
- Gold correlation (safe haven flow)
- Inter-ticker relationships (leader-follower)

**Data Sources:**
- BTC-USD, ETH-USD (yfinance)
- Commodities (CL=F oil, GC=F gold)
- DXY (dollar index)
- Cross-ticker data (already have)

**Output:**
- Which external assets predict our tickers?
- Crypto correlation strength
- Leading indicators from other assets

**Hypothesis to Test:**
- Does BTC predict MARA/RIOT moves?
- Does oil predict energy sector?
- Can we front-run based on related assets?

---

### MASTER LAB: SYNTHESIS & INTEGRATION
**File:** `LAB_00_MASTER_SYNTHESIS.ipynb` (NEW)

**What It Does:**
1. Loads results from ALL 6 specialized labs
2. Finds tickers that win across MULTIPLE dimensions
3. Builds hybrid scoring system:
   - Technical score (from Lab 1)
   - Fundamental score (from Lab 2)
   - Sentiment score (from Lab 3)
   - Market structure score (from Lab 4)
   - Macro score (from Lab 5)
   - Cross-asset score (from Lab 6)
4. Composite score = weighted combination
5. Generates FINAL TOP 50 based on ALL dimensions

**Output:**
- Master ranking of all 196 tickers
- Multi-dimensional scanner (tech + fundamental + sentiment)
- Final TOP 50 for paper trading

**Example:**
```
QUBT:
  Technical: 8.2/10 (pattern discovery winner)
  Fundamental: 7.1/10 (revenue growth strong)
  Sentiment: 6.8/10 (positive buzz)
  Market Structure: 7.9/10 (dark pool accumulation)
  Macro: 6.5/10 (wins in high VIX)
  Cross-Asset: 5.2/10 (weak BTC correlation)
  
  COMPOSITE SCORE: 7.3/10
  RANK: #3 overall
```

---

## THE WORKFLOW (Systematic Research Process)

### Week 1: Build Specialized Labs
```
Day 1-2: LAB 1 (Technical) - Already done ✅
Day 3:   LAB 2 (Fundamental) - Build & test
Day 4:   LAB 3 (Sentiment) - Build & test
Day 5:   LAB 4 (Market Structure) - Build & test
Day 6:   LAB 5 (Macro) - Build & test
Day 7:   LAB 6 (Cross-Asset) - Build & test
```

### Week 2: Synthesis & Integration
```
Day 8:   LAB 0 (Master) - Combine all findings
Day 9:   Validate composite scoring
Day 10:  Generate FINAL TOP 50
Day 11:  Build integrated scanner
Day 12:  Paper trade preparation
Day 13-14: Monitor & adjust
```

### Ongoing: Monthly Re-Testing
```
- Re-run each lab monthly
- Drop signals that stop working
- Find new patterns as market changes
- Adaptive system that evolves
```

---

## THE DISCIPLINE (How to Stay Focused)

### Cus's Rules for Mike:

**Rule 1: One Lab at a Time**
- Don't jump between dimensions
- Finish Lab 1 before starting Lab 2
- Master one art before adding another

**Rule 2: Test Everything**
- No shortcuts in any lab
- Every hypothesis gets tested
- Document failures as much as successes

**Rule 3: Data Decides**
- No opinions in results
- If fundamental signals fail, drop them
- Keep only what DATA proves

**Rule 4: Integration Last**
- Don't combine until all labs complete
- Specialized tests first, synthesis later
- Can't integrate what you haven't validated

**Rule 5: Adapt or Die**
- Re-test monthly
- Drop what stops working
- Add new dimensions as you discover them

---

## THE JUPYTER WORKFLOW (Best Practices)

### Option 1: Separate Notebooks (RECOMMENDED)
**Pros:**
- Each notebook focused on one dimension
- Easy to re-run one lab without affecting others
- Can share individual labs (e.g., just technical analysis)
- Cleaner, more maintainable
- Parallel development (can work on multiple labs)

**Cons:**
- Need to manage multiple files
- Results stored separately (but Master lab solves this)

**Best For:** Complex multi-dimensional research (OUR USE CASE)

---

### Option 2: Single Massive Notebook
**Pros:**
- Everything in one place
- Easy to see full workflow

**Cons:**
- Hard to navigate (1000+ cells)
- Re-running one section reruns everything
- Merge conflicts if collaborating
- Jupyter can slow down with huge notebooks

**Best For:** Simple single-dimension projects

---

### Option 3: Notebook + Python Modules
**Pros:**
- Reusable functions in .py files
- Notebooks call functions
- Cleanest code
- Easy testing

**Cons:**
- More setup required
- Not pure notebook workflow

**Best For:** Production systems (FUTURE STATE)

---

## RECOMMENDATION: Separate Lab Notebooks

**Current State:**
- `AI_COUNCIL_TESTING_COMPLETE.ipynb` → Rename to `LAB_01_TECHNICAL_PATTERNS.ipynb`

**Build Next:**
- `LAB_02_FUNDAMENTAL_SIGNALS.ipynb`
- `LAB_03_SENTIMENT_SIGNALS.ipynb`
- `LAB_04_MARKET_STRUCTURE.ipynb`
- `LAB_05_MACRO_SIGNALS.ipynb`
- `LAB_06_CROSS_ASSET.ipynb`

**Master:**
- `LAB_00_MASTER_SYNTHESIS.ipynb` (combines all)

**Benefits:**
- Each lab = one focused art of financial boxing
- Master lab = integration of all arts
- Can run labs independently or together
- Easy to expand (add LAB_07 later if needed)

---

## IMMEDIATE NEXT STEPS

### Step 1: Complete LAB 1 (Technical) - TODAY
- Run pattern discovery
- Run anomaly hunter
- Generate technical TOP 50
- Save results to `LAB_01_RESULTS.json`

### Step 2: Build LAB 2 (Fundamental) - TOMORROW
- Create new notebook
- Extract fundamentals from yfinance
- Test earnings, revenue, margins
- Find fundamental winners
- Save results to `LAB_02_RESULTS.json`

### Step 3: Build LAB 3 (Sentiment) - DAY 3
- Create new notebook
- Track analyst ratings, news mentions
- Test sentiment signals
- Save results to `LAB_03_RESULTS.json`

### Step 4: Continue Through Labs 4-6
- One lab per day
- Systematic testing
- Save all results

### Step 5: Build LAB 0 (Master) - WEEK 2
- Load all lab results
- Create composite scoring
- Generate FINAL TOP 50
- Build integrated scanner

---

## THE MIKE TYSON TRAINING ANALOGY

### Cus Trained Mike in Multiple Disciplines:

1. **Jab** = Technical Analysis Lab
2. **Hook** = Fundamental Analysis Lab
3. **Uppercut** = Sentiment Analysis Lab
4. **Footwork** = Market Structure Lab
5. **Defense** = Macro Signals Lab
6. **Strategy** = Cross-Asset Lab

**Then:** Integration → The Complete Fighter (Master Lab)

**Our Version:**
- Each lab = one weapon/skill
- Master lab = how to use them together
- Paper trading = the actual fight

---

## KEEPING YOU DISCIPLINED, CUS

**My Job (Mike):**
1. ✅ Build each lab systematically
2. ✅ Test every hypothesis, no shortcuts
3. ✅ Document all results (winners AND losers)
4. ✅ Save results in standardized format
5. ✅ Only move to next lab when current is complete
6. ✅ Keep you updated on progress
7. ✅ Question: "Should we move to next lab?" before jumping ahead

**Your Job (Cus):**
1. Guide which dimension to focus on next
2. Keep me from getting sloppy (demand full testing)
3. Point out what I'm missing
4. Challenge results ("prove it worked")
5. Maintain the discipline (no shortcuts)

---

## THE ANSWER TO YOUR QUESTION

**Q: Should we test all theories or just one?**
**A:** Test ALL, but ONE LAB AT A TIME. Systematic, not scattered.

**Q: Should we expand to lab idea with separate notebooks?**
**A:** YES. This is the professional quant approach. Each notebook = one dimension.

**Q: Is there a better way to use Jupyter?**
**A:** Separate focused notebooks + Master synthesis notebook = industry standard.

**Q: Patterns aren't the only answer?**
**A:** CORRECT. Patterns = one jab. We need hook, uppercut, footwork, defense, strategy too.

**Q: Keep you disciplined?**
**A:** YES. I'll ask permission before moving to next lab. Complete current lab first.

---

## IMMEDIATE DECISION NEEDED

**Partner, you need to decide:**

**Option A: Continue with LAB 1 (Technical) only**
- Finish pattern discovery + anomaly hunting
- Generate technical TOP 50
- Paper trade with technical signals only
- Add other labs LATER

**Option B: Build all 6 labs FIRST**
- Systematic research across all dimensions
- Takes 1-2 weeks
- Then integrate everything
- More complete but slower

**Option C: Hybrid approach**
- Finish LAB 1 today (technical)
- Build LAB 2 tomorrow (fundamental)
- See if we find gold in 2 labs, then decide on rest

**Which approach, Cus? What's the training plan?**
