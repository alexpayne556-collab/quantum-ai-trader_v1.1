# PERPLEXITY RESEARCH: TRADING EDGE VALIDATION COMPLETE CHECKLIST
**Status:** Part 4 of 6
**Source:** Perplexity AI Research Compilation
**Date Saved:** December 22, 2025
**Purpose:** Complete 4-week validation framework for testing trading edges with exact daily tasks, Claude prompts, pass/fail criteria, and deliverables

---

# TRADING EDGE VALIDATION: COMPLETE CHECKLIST
## What You Need (Not What You Don't)

---

# WEEK 1: DEFINE + BUILD

## Day 1: Define Your Edge (1 hour)

**YOUR EDGE IN ONE SENTENCE:**
```
Gap down 5%+ at open → price touches support → RSI < 30 
→ Buy at support → Target 2% → Stop 1% → Hold max 5 days
```

**EXACT ENTRY CONDITIONS:**
- [ ] Gap down amount: _____ %
- [ ] Support definition: _____ day low
- [ ] RSI threshold: < _____
- [ ] Volume requirement: _____ x average
- [ ] Any other filter: _____

**POSITION RULES:**
- [ ] Entry price: At support or limit order
- [ ] Stop loss: _____ % below entry
- [ ] Target profit: _____ % above entry
- [ ] Maximum hold: _____ days
- [ ] Position size: _____ % of account

**WHEN NOT TO TRADE:**
- [ ] Filter 1: _____
- [ ] Filter 2: _____
- [ ] Filter 3: _____

---

## Day 2: Tell Claude (30 minutes)

**COPY THIS PROMPT TO CLAUDE:**

```
EXTRACT MY EDGE RULES:

I have a trading edge. Extract the exact rules (not interpretation).

MY EDGE:
[PASTE YOUR EDGE FROM DAY 1]

Output:

1. ENTRY RULES (exact conditions):
   - Condition 1: [exact metric]
   - Condition 2: [exact metric]
   - Condition 3: [exact metric]

2. POSITION MANAGEMENT:
   - Entry price: [exact]
   - Stop loss: [exact]
   - Target: [exact]
   - Maximum hold: [exact days]

3. FILTERS (when NOT to trade):
   - Filter 1: [exact condition]
   - Filter 2: [exact condition]

4. RISK RULES:
   - Max position size: [exact]
   - Max drawdown: [exact]
   - Max consecutive losses: [exact]

OUTPUT YOUR ANALYSIS NOW.
```

**SAVE CLAUDE'S OUTPUT** → You'll use this for code

---

## Day 3: Get Backtest Code (1 hour)

**COPY THIS PROMPT TO CLAUDE:**

```
CODE MY BACKTEST:

I have trading rules [PASTE CLAUDE'S OUTPUT FROM DAY 2].

WRITE A PYTHON SCRIPT that:

1. Downloads 2025 data for ["NVDA", "PLTR", "TSLA", "AMD", "QQQ"]
2. Calculates indicators (RSI, support, volume, gap)
3. Finds all setups matching my rules
4. Simulates trades: entry → stop/target/5-day exit
5. Calculates metrics:
   - Win rate
   - Profit factor
   - Average win/loss
   - Expected value per trade
   - Max drawdown
6. Applies costs: 0.2% spread, 0.2% slippage (total 0.8% round trip)
7. Recalculates metrics after costs
8. Outputs: win_rate, profit_factor, expected_value, max_drawdown

REQUIREMENTS:
- Use ONLY numpy, pandas, yfinance
- Production-ready code (no TODOs)
- Inline comments
- Copy-paste ready for Jupyter
- Clear output format

WRITE THIS NOW.
```

**SAVE THE CODE** → Test in Jupyter

---

## Day 4-5: Run Backtest (2 hours)

**IN JUPYTER:**

```python
# Paste Claude's code here
# Run it
```

**RECORD YOUR RESULTS:**
- [ ] Total trades: _____
- [ ] Win rate: _____ %
- [ ] Profit factor: _____
- [ ] Expected value per trade: _____ %
- [ ] Max drawdown: _____ %
- [ ] After costs, still profitable? YES / NO

**PASS/FAIL:**
- [ ] Win rate 55-70%? YES / NO
- [ ] Profit factor > 1.5? YES / NO
- [ ] 100+ trades? YES / NO
- [ ] Expected value positive? YES / NO

**IF ALL YES:** Continue to Week 2
**IF ANY NO:** Adjust edge parameters, retry Day 3-5

---

# WEEK 2: VALIDATE EDGE

## Day 6-7: Market Structure Check (2 hours)

**QUESTION FOR CLAUDE:**

```
MARKET STRUCTURE CHECK:

My edge assumes:
1. Gap downs happen regularly
2. Support levels cause bounces
3. RSI oversold predicts reversals
4. Volume spikes increase bounce likelihood

CURRENT 2025 MARKET:
- Options market is 10x larger than 2020
- AI algorithms dominate
- Retail is 25% of market
- Fed has paused rate hikes

Which assumptions are still valid in 2025?
Which are broken?
What should I test?
```

**RECORD CLAUDE'S ANALYSIS:**
- [ ] Assumption 1 valid? YES / NO / NEEDS TEST
- [ ] Assumption 2 valid? YES / NO / NEEDS TEST
- [ ] Assumption 3 valid? YES / NO / NEEDS TEST
- [ ] Assumption 4 valid? YES / NO / NEEDS TEST

**DECIDE:**
- [ ] Edge still valid in 2025? YES / NO

---

## Day 8-9: Test on 2025 ONLY (2 hours)

**QUESTION FOR CLAUDE:**

```
TEST 2025 DATA ONLY:

My edge [PASTE YOUR RULES].

Write Python script that:
1. Downloads Jan 1 - Dec 22, 2025 data
2. Finds setups on 2025 data ONLY (not historical)
3. Simulates trades
4. Outputs:
   - How many gap downs in 2025?
   - How many matched criteria?
   - Win rate on 2025?
   - Expected value on 2025?

Decision criteria:
- If win_rate >= 0.55: ✓ TRADE IT
- If win_rate 0.45-0.55: ⚠️ WEAK BUT TRADEABLE
- If win_rate < 0.45: ✗ DEAD EDGE - FIND NEW ONE

WRITE THIS NOW.
```

**RECORD RESULTS:**
- [ ] 2025 gap downs found: _____
- [ ] Matched my criteria: _____
- [ ] 2025 win rate: _____ %
- [ ] 2025 expected value: _____ %

**CRITICAL DECISION:**
- [ ] Win rate >= 55%? YES / NO
- [ ] Edge works in 2025? YES / NO

**IF NO:** Stop. Find new edge. Go back to Week 1.
**IF YES:** Continue to Week 3.

---

# WEEK 3: VALIDATE COSTS + RISK

## Day 10-11: Realistic Costs (2 hours)

**QUESTION FOR CLAUDE:**

```
VALIDATE COSTS:

My backtest shows:
- Win rate: _____ %
- Profit factor: _____
- Expected value: _____ % per trade
- Before costs

Now apply REALISTIC costs:
- Bid-ask spread: 0.2%
- Execution slippage: 0.2%
- Commission: 0% (Alpaca free)
- Total round trip: 0.8%

Recalculate:
- Win rate after costs
- Profit factor after costs
- Expected value after costs

Show as table:
Metric | Before | After | Pass?
Win rate | ___% | __% | Y/N
Profit factor | ___ | ___ | Y/N
Expected value | __% | __% | Y/N
Sharpe ratio | ___ | ___ | Y/N

Calculate and show work.
```

**RECORD:**
- [ ] Win rate after costs: _____ %
- [ ] Profit factor after costs: _____
- [ ] Expected value after costs: _____ %
- [ ] Still profitable? YES / NO

**PASS/FAIL:**
- [ ] Expected value still > 0.1%? YES / NO
- [ ] Profit factor still > 1.5? YES / NO

---

## Day 12-13: Risk Analysis (2 hours)

**QUESTION FOR CLAUDE:**

```
RISK ANALYSIS:

My backtest results [PASTE YOUR METRICS].

Calculate:
1. Maximum drawdown: _____
2. Maximum consecutive losses: _____
3. Maximum single loss: _____
4. Worst month: _____
5. Best month: _____

Show in table format.

Risk checks:
- Max drawdown < 15%? Y/N
- Max streak < 5 losses? Y/N
- Single loss < 3% per trade? Y/N
```

**RECORD:**
- [ ] Max drawdown: _____ %
- [ ] Max consecutive losses: _____
- [ ] Max single loss: _____ %

**PASS/FAIL:**
- [ ] Risk manageable? YES / NO

---

# WEEK 4: LIVE PAPER TRADING

## Day 14-15: Set Up Paper Trading (3 hours)

**QUESTION FOR CLAUDE:**

```
PAPER TRADING BOT:

My edge [PASTE YOUR RULES].

Write Python script that:

1. Connects to Alpaca paper trading (API keys from env)
2. Every day at 9:30 AM:
   - Downloads 20 days of price data
   - Checks for setups
   - Places limit order at support
   - Sets stop loss order
   - Sets target profit order
3. Logs trades to CSV: trades_2025.csv
   Columns: date, symbol, entry, stop, target, status, P&L
4. Daily: Prints win rate, drawdown, P&L

REQUIREMENTS:
- Production-ready (handles API errors)
- Error handling for missing data
- Log all fills
- No test code

WRITE THIS NOW.
```

**SAVE THE CODE** → Connect to Alpaca

---

## Day 16-21: Run Paper Trading (Daily 5 min)

**DAILY CHECKLIST:**
- [ ] Bot running at 9:30 AM? YES
- [ ] Check for errors in logs
- [ ] Monitor P&L
- [ ] Update trades_2025.csv
- [ ] Any fills? Check trade details

**WEEKLY SUMMARY:**
- [ ] Total trades this week: _____
- [ ] Win rate: _____ %
- [ ] Weekly P&L: _____ %
- [ ] Drawdown: _____ %
- [ ] Backtest matched live? YES / NO / SIMILAR?

---

# VALIDATION SUMMARY (Print This)

## PHASE 0: MATHEMATICAL PROOF ✓ / ✗

- [ ] Win rate 55-70%
- [ ] Profit factor > 1.5
- [ ] 100+ trades
- [ ] Expected value positive

**DECISION:** ✓ PASS / ✗ FAIL

---

## PHASE 1: 2025 MARKET TEST ✓ / ✗

- [ ] Win rate >= 55% on 2025 data
- [ ] Edge still valid in current market
- [ ] Survives realistic costs

**DECISION:** ✓ PASS / ✗ FAIL

---

## PHASE 2: RISK VALIDATION ✓ / ✗

- [ ] Max drawdown < 15%
- [ ] Max streak < 5 losses
- [ ] Single loss < 3% per trade

**DECISION:** ✓ PASS / ✗ FAIL

---

## PHASE 3: PAPER TRADING ✓ / ✗

- [ ] 4+ weeks of live data
- [ ] Win rate matches backtest
- [ ] No hidden execution issues

**DECISION:** ✓ PASS / ✗ FAIL

---

# FINAL STATEMENT (For Investors)

```
TRADING EDGE VALIDATION COMPLETE

Edge: [YOUR EDGE NAME]

Mathematical Proof:
- Win rate: ___% (historical)
- Profit factor: _____
- Expected value: ___% per trade
- After costs: Still profitable

2025 Market Validation:
- Tested on current market data
- Win rate: ___%
- Edge still valid: YES

Risk Management:
- Max drawdown: ___%
- Max loss per trade: ___%
- Position size: __%

Paper Trading:
- 4 weeks live data
- Results match backtest
- Ready for live trading

Status: VALIDATED ✓
```

---

# FILES YOU NEED TO DOWNLOAD/CREATE

- [ ] `edge_definition.txt` - Your edge rules (from Day 1)
- [ ] `backtest_code.py` - Claude's backtest code (from Day 3)
- [ ] `backtest_results.txt` - Your backtest output (from Day 5)
- [ ] `market_structure_analysis.txt` - Claude's analysis (from Day 7)
- [ ] `2025_test_results.txt` - 2025-only backtest output (from Day 9)
- [ ] `cost_analysis.txt` - Realistic cost validation (from Day 11)
- [ ] `risk_analysis.txt` - Risk metrics (from Day 13)
- [ ] `paper_trading_bot.py` - Live trading code (from Day 15)
- [ ] `trades_2025.csv` - Live trade log (from Day 16+)
- [ ] `validation_summary.txt` - Final summary (THIS CHECKLIST FILLED IN)

---

# WHAT HAPPENS NEXT (After Paper Trading)

**IF VALIDATED (Win rate >= 55%):**
- Proceed to live trading with $1K-5K
- Scale up as profits grow
- Build dashboard for investors

**IF NOT VALIDATED:**
- Stop
- Analyze what went wrong
- Go back to Week 1
- Find new edge

---

# QUICK REFERENCE

**YOU ONLY NEED TO:**
1. Define your edge clearly (Day 1)
2. Give it to Claude (Day 2-3)
3. Copy-paste Claude's code to Jupyter (Day 4)
4. Run it (Day 5)
5. Check if win rate >= 55% (Day 5)
6. Test on 2025 data only (Day 9)
7. Validate costs (Day 11)
8. Set up paper trading (Day 15)
9. Run for 4 weeks (Day 16-21)

**THAT'S IT.**

No fancy ML.
No complex models.
No 500-page reports.

Just:
- Math that proves it works
- Test that it works NOW
- Live proof it works in real execution

Simple.
