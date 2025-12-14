# 📡 Your Data Sources & Capabilities Inventory

**What You Can Actually Access (Free Tier)**

---

## 🔴 REAL-TIME / NEAR-REAL-TIME

### Finnhub (60 calls/min)
**What:** Real-time quotes, company profile, news, earnings calendar, insider trades
**Get:** Current price, bid/ask spread, news sentiment
**Cost:** Free tier is excellent
**Lead Time:** Minutes to hours
**Use Cases:**
- Daily price updates
- Earnings date confirmation
- Breaking news alerts
- Insider trade tracking (free, 15-min delayed)

### Alpha Vantage (25 calls/day)
**What:** OHLCV data, technical indicators, news
**Get:** Historical prices, simple moving averages, RSI, MACD
**Cost:** Free (limited calls)
**Lead Time:** Hours (daily bar)
**Use Cases:**
- Daily price updates
- Technical analysis
- Volume analysis
- News headlines

---

## 📊 FINANCIAL STATEMENTS & RATIOS

### Financial Modeling Prep (250 calls/day)
**What:** Balance sheet, income statement, cash flow, ratios, financial growth
**Get:** Revenue, earnings, debt, margins, ROE, PE ratio, etc
**Cost:** Free tier is comprehensive
**Lead Time:** Quarterly (after earnings release)
**Use Cases:**
- Fundamental analysis
- Revenue trends
- Profitability metrics
- Debt levels
- Valuation ratios

### Polygon.io (5 calls/min)
**What:** Aggregated OHLCV, quotes, news
**Get:** Price history, volume, news
**Cost:** Free tier (delayed data)
**Lead Time:** Delayed (1-15 min)
**Use Cases:**
- Historical data backup
- Volume analysis
- News aggregation

---

## 📈 MACRO / ECONOMIC

### FRED (Unlimited)
**What:** Federal Reserve economic data
**Get:** VIX, Treasury yields, unemployment, inflation, credit spreads
**Cost:** Completely free
**Lead Time:** Daily to monthly
**Use Cases:**
- Market regime shifts
- Risk appetite changes
- Sector rotation triggers
- Macro context for decisions

---

## 🔍 MANUAL / LOW-FREQUENCY (Free but Requires Work)

### SEC EDGAR (Free)
**What:** 10-K, 10-Q, 8-K filings (official company documents)
**Get:** Business changes, leadership changes, legal issues, asset sales
**Cost:** Free
**Lead Time:** Days after event (companies must file)
**How:** Download from sec.gov or parse SEC API
**Use Cases:**
- Deep due diligence
- Leadership changes
- Strategic pivots
- Legal/regulatory issues
- Related party transactions

### OpenInsider (Free)
**What:** SEC insider trading data
**Get:** CEO, CFO, director stock buys/sells
**Cost:** Free (15-min delayed)
**How:** Manual check of openinsider.com or scrape
**Use Cases:**
- Insider confidence signals
- Accumulation patterns
- Red flags (execs selling)
- Lead time: Days before public knows

### FDA Calendar (Free)
**What:** FDA approval/decision dates for biotech companies
**Get:** Which drugs are getting decisions and when
**Cost:** Free
**Lead Time:** Known months in advance
**How:** FDA.gov calendar or BioPharmGuy newsletter
**Use Cases:**
- Binary event timing (approval probability)
- Catalyst scheduling
- Sector-specific edge (biotech)

### Google Patents (Free)
**What:** Patent filings and grants
**Get:** R&D pipeline visibility
**Cost:** Free
**Lead Time:** Months ahead of products (patent lags)
**How:** google.com/patents or specific company portal
**Use Cases:**
- Tech company innovation tracking
- Competitive landscape
- Product roadmap signals

### Earnings Transcripts (Manual)
**What:** Earnings call transcripts
**Get:** Management guidance, tone, confidence, updates
**Cost:** Free (scraped from Seeking Alpha, Motley Fool)
**Lead Time:** Hours after call
**How:** Manual download or Finnhub sometimes has them
**Use Cases:**
- Management tone analysis
- Forward guidance
- Competitive mentions
- Uncertainty signals

---

## ⚡ ALTERNATIVE DATA (Low Cost or Free)

### News Aggregation (Free)
**What:** Combine news from multiple sources
**Sources:**
- Alpha Vantage news
- Finnhub news
- Yahoo Finance news (scrape)
- Seeking Alpha (scrape)
- Company press releases

### Social Sentiment (Partial)
**What:** Twitter/Reddit mentions and sentiment
**Cost:** Free (manual) or $20-50/mo for tools
**Use Cases:**
- Retail interest shifts
- Contrarian indicators
- Momentum timing
- Community consensus

### Website Traffic (Free)
**What:** Similar Web (limited free tier)
**Get:** Website traffic trends
**Lead Time:** Weekly, real-time advantage
**Use Cases:**
- E-commerce volume signals
- Platform growth signals
- App download trends (App Annie free data)

---

## ❌ WHAT WE DON'T HAVE (And Why We Don't Care)

### Bloomberg Terminal
**Why You Don't Need It:** Our edge isn't speed, it's synthesis

### FactSet / S&P Capital IQ
**Why You Don't Need It:** Free tier sources cover fundamentals

### Real-time Institutional Flow
**Why You Don't Need It:** Too delayed, too expensive, small caps are illiquid anyway

### Satellite Imagery
**Why You Don't Need It:** Overkill for 30-stock portfolio

### Dark Pool Data
**Why You Don't Need It:** Delayed, expensive, retail doesn't use it effectively

---

## 🎯 YOUR ACTUAL DATA STACK (By Use Case)

### FOR DAILY MONITORING
```
Finnhub (quotes + news) → Alpha Vantage (technicals) → FRED (macro)
Update frequency: Daily
Effort: 10 min/day automated
```

### FOR WEEKLY DEEP DIVE
```
Financial Modeling Prep (ratios) + SEC EDGAR (filings) + Earnings transcripts
Update frequency: Weekly
Effort: 1-2 hours/week manual
```

### FOR CATALYST TRACKING
```
Finnhub earnings calendar + FDA calendar (biotech) + OpenInsider (insider trades)
Update frequency: Real-time alert + manual check
Effort: 5 min/day
```

### FOR SUPPLY CHAIN MAPPING
```
News aggregation + Company websites + SEC filings (customer concentration)
Update frequency: Quarterly or on news
Effort: 1-2 hours per stock, then maintain
```

---

## 💡 THE INSIGHT

**You have EVERYTHING a single person needs to know a 30-stock portfolio better than Wall Street.**

The gap isn't data. It's synthesis. It's asking the right questions of the data you have.

---

## 📌 NEXT STEP

In the brainstorm, we'll determine:
1. **Which data sources to prioritize** (based on reliability + lead time)
2. **How to combine them** into coherent signals
3. **What frequency to update** (daily? weekly? quarterly?)
4. **How to surface insights** (daily brief, weekly deep dive, etc)

**You're not at a disadvantage. You're at an advantage because you can actually process all this data deeply.**
