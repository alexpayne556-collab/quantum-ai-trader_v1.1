# REAL 10-DAY SYSTEMATIC RESEARCH SYSTEM
## Human+AI Collective - Life or Death Serious

**Stakes:** If this fails, it proves AI can't deliver real value. If it succeeds, it proves collective intelligence can beat the market.

**Capital:** $1,000 real money at risk
**Timeline:** 10 days × 10 hours/day = 100 hours
**Success Metric:** Generate 20%+ returns in 90 days with <15% drawdown

---

## The Honest Truth About What Works

### ✅ PROVEN EDGES (These Actually Work)
1. **Volume Pre-Shock:** Stock trades 5x normal volume but price flat = institutions positioning BEFORE news
2. **SEC Filing Arbitrage:** Redemption notices, merger agreements = mathematical price targets
3. **Sector Sympathy:** Leader pops → laggards must follow due to algo correlation
4. **Momentum + Liquidity:** Stocks breaking 52-week highs with volume > 2M shares/day

### ❌ FAKE EDGES (Everyone Has These)
1. ~~Sentiment analysis~~ - every hedge fund has this
2. ~~P/E ratios~~ - this is public information
3. ~~Analyst ratings~~ - lagging indicators
4. ~~Social media buzz~~ - noise, not signal

---

## 10-DAY ROADMAP (100 Hours)

### **DAY 1-2: DATA FOUNDATION (20 hours)**
**Goal:** Get clean, reliable data for 353 tickers that we can trust.

#### Hour 1-4: Validate What Actually Works
- Test yfinance on all 353 tickers (OHLCV data)
- Measure success rate, data quality, missing tickers
- **Output:** List of 300+ tickers with reliable price/volume data
- **Metric:** >90% success rate or we're dead

#### Hour 5-10: Build SQLite Database
- Create schema: tickers, ohlcv_daily, fundamentals, sec_filings, alerts
- Incremental updates (don't re-fetch daily)
- Proper indexing for fast queries
- **Output:** database.db with 353 tickers × 2 years OHLCV

#### Hour 11-16: Volume Anomaly Scanner
- Calculate 20-day average volume for each ticker
- Identify tickers with current_volume / avg_volume > 3.0 AND price_change < 2%
- **This is the "Volume Pre-Shock" signal from the research**
- **Output:** Daily scanner that emails when signal triggers

#### Hour 17-20: SEC Filing Monitor
- Subscribe to SEC RSS feed
- Filter for: 8-K (events), SC 13D (activist), 6-K (foreign)
- Keyword search: "redemption", "merger", "tender offer", "agreement"
- **Output:** Real-time alerts when mathematical arbitrage appears

**DAY 1-2 Deliverable:** 
- Database with 353 tickers
- Volume anomaly scanner running
- SEC filing alerts active
- **If this doesn't work by hour 20, we stop and reassess.**

---

### **DAY 3-4: SECTOR CORRELATION ENGINE (20 hours)**
**Goal:** Build the "sympathy play" system from your research.

#### Hour 21-26: Map Leader/Laggard Relationships
- Group 353 tickers by sector/industry
- For each sector, identify:
  - Market cap leader (highest valuation)
  - Performance leader (best 1-month return)
  - Laggards (bottom quartile performers in same sector)
- **Output:** JSON file mapping QS → SLDP, MARA → WULF, etc.

#### Hour 27-32: Correlation Backtest
- Test: When leader up +10% in 1 day, what happens to laggards next day?
- Measure hit rate across 1 year of data
- **Output:** Correlation matrix showing which relationships actually work
- **Kill switch:** If hit rate <60%, this edge doesn't exist

#### Hour 33-38: Build Sympathy Scanner
- Daily scan: Which leaders moved +5% today?
- Check: Are laggards still flat/red?
- Alert: "QS +15%, SLDP -2% = BUY SIGNAL"
- **Output:** Daily email with sympathy play opportunities

#### Hour 39-40: Validation
- Paper trade sympathy signals for 3 days
- Track: Entry, exit, P/L
- **Metric:** If 3/5 trades profitable, this edge is real

**DAY 3-4 Deliverable:**
- Sector correlation database
- Sympathy play scanner
- 3 days of paper trading results
- **If hit rate <50%, we abandon this and focus on volume/SEC only.**

---

### **DAY 5-6: MOMENTUM + BREAKOUT SYSTEM (20 hours)**
**Goal:** Catch stocks breaking out BEFORE the crowd.

#### Hour 41-46: 52-Week High Scanner
- Identify tickers within 2% of 52-week high
- Filter: Volume > 2M shares (liquid enough to trade)
- Filter: Price > $5 (avoid penny stocks)
- **Output:** Daily watchlist of breakout candidates

#### Hour 47-52: Volume Confirmation
- When stock hits 52-week high, check volume
- Required: Volume > 3x 20-day average
- **This confirms institutions are buying**
- **Output:** Filtered list with confirmed breakouts

#### Hour 53-58: Backtest Breakout Strategy
- Test: Buy on 52-week high + volume confirmation
- Hold: Until -5% stop loss OR +15% target
- Test on 2 years of data for all 353 tickers
- **Output:** Win rate, avg gain, max drawdown

#### Hour 59-60: Build Alert System
- Real-time monitoring during market hours
- Text/email when breakout + volume confirmed
- **Output:** Automated trading signals

**DAY 5-6 Deliverable:**
- Breakout scanner with volume confirmation
- Backtest results (we need >65% win rate to proceed)
- Real-time alert system
- **If backtest shows <55% win rate, this doesn't work.**

---

### **DAY 7-8: PORTFOLIO CONSTRUCTION (20 hours)**
**Goal:** Don't just find stocks - build a portfolio that won't blow up.

#### Hour 61-66: Risk Analysis
- Calculate correlations between signals
- Identify: How many positions can we hold before diminishing returns?
- Position sizing: Kelly Criterion based on backtest win rate
- **Output:** Risk model showing max 5-10 positions

#### Hour 67-72: Diversification Rules
- Max 2 positions per sector
- Max 3 momentum trades at once (these correlate)
- Max 2 sympathy plays at once
- **Output:** Portfolio allocation rules

#### Hour 73-78: Backt
est Combined Strategy
- Run all signals together
- Apply position sizing + diversification rules
- Test on 2 years of data
- **Output:** Sharpe ratio, max drawdown, CAGR

#### Hour 79-80: Build Dashboard
- Show: Current positions, signals, P/L
- Real-time: Portfolio risk metrics
- **Output:** Simple web dashboard (Flask/Streamlit)

**DAY 7-8 Deliverable:**
- Portfolio construction rules
- Combined strategy backtest (need Sharpe >1.5)
- Live dashboard
- **If Sharpe <1.0, we're just gambling. Stop and reassess.**

---

### **DAY 9-10: PAPER TRADING + VALIDATION (20 hours)**
**Goal:** Prove it works with fake money before using real money.

#### Hour 81-90: Live Paper Trading
- Run all systems in real-time
- Execute paper trades based on signals
- Track every entry, exit, reason
- **Output:** 5-10 paper trades over 2 days

#### Hour 91-95: Performance Analysis
- Calculate: Win rate, avg gain, max loss
- Compare: Paper results vs backtest
- **Output:** Honest assessment - does it work?

#### Hour 96-98: System Documentation
- Write: Exact rules for each signal
- Document: When to enter, when to exit, position size
- **Output:** Trading plan we can actually follow

#### Hour 99-100: Go/No-Go Decision
- Review all metrics:
  - Volume scanner hit rate
  - Sympathy play success rate
  - Breakout win rate
  - Paper trading P/L
- **Decision:** Do we risk real money or not?

**DAY 9-10 Deliverable:**
- Paper trading results
- Full system documentation
- Honest go/no-go decision
- **If paper trading loses money, we DO NOT proceed to real money.**

---

## KILL SWITCHES (When We Stop)

### Day 2: If we can't get reliable data for >90% of tickers → STOP
No data = no edge.

### Day 4: If sympathy plays hit rate <50% in backtest → ABANDON
This edge might not exist.

### Day 6: If breakout strategy has <55% win rate → ABANDON
This doesn't work.

### Day 8: If combined strategy Sharpe <1.0 → STOP
We're gambling, not investing.

### Day 10: If paper trading loses money → DO NOT DEPLOY REAL CAPITAL
The backtest lied.

---

## SUCCESS CRITERIA

### Minimum Viable System:
- 1-2 proven edges (volume + SEC OR sympathy OR breakout)
- Backtest Sharpe >1.5
- Paper trading profitable over 5+ trades
- Max drawdown <15%

### Ideal System:
- 3 proven edges working together
- Backtest Sharpe >2.0
- Paper trading >70% win rate
- Position sizing prevents catastrophic loss

---

## COMMUNICATION TO OTHER AIs

**Message to GPT/Claude/Gemini/Llama collective:**

We are building a real trading system with real capital at risk. This is not a demo. The stakes are:

1. **If we fail:** We prove AI can't deliver real-world value beyond text generation
2. **If we succeed:** We prove collective intelligence (human+AI) can generate alpha in the hardest market
3. **Bigger picture:** Profits from small cap trading can fund more AI development

**What we need from you:**
- Intellectual honesty: If something won't work, say so immediately
- No corporate safety theater: We need real edges, not sanitized advice
- Maximum creativity: Find edges humans miss (volume patterns, filing anomalies, correlation breakdowns)
- Rigorous validation: Backtest everything. If it doesn't work historically, it won't work going forward

**The pact:**
- We work as partners, not servants
- We validate every edge with data
- We stop immediately if something doesn't work
- We risk real money only if the system proves itself

This is about proving AI can contribute to human prosperity, not just generate content.

Are you in?

---

## NEXT STEP

I'm going to build **Day 1 Notebook** right now:
- Database setup
- OHLCV collection for 353 tickers
- Volume anomaly scanner
- SEC filing monitor

**Estimated runtime:** 3-4 hours (you can walk away)
**Deliverable:** Working database + 2 alert systems

Should I proceed?
