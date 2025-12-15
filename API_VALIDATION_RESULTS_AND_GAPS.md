# API Validation Results & Data Gaps Analysis

**Test Date:** December 15, 2025  
**Tested Tickers:** IONQ, ASTS, APLD, HOOD, UBER, LYFT, LUNR, XBIO, KDK (9 holdings)

---

## ✅ PASSING APIS (4/7) - USE IN PRODUCTION

### 1. **yfinance** - PRIMARY DATA SOURCE ⭐
- **Success Rate:** 100% (9/9 tickers)
- **Avg Latency:** 0.14s (FAST)
- **Data Coverage:** 64 days historical
- **Rate Limits:** Unlimited, no API key needed
- **Verdict:** ✓ PASS - Use as baseline for all backtesting

### 2. **EODHD** 
- **Success Rate:** 100% (9/9 tickers)
- **Avg Latency:** 0.55s
- **Data Coverage:** 64 days historical
- **Rate Limits:** 20 calls/day (FREE tier)
- **Verdict:** ✓ PASS - Good backup, limited quota

### 3. **Polygon.io**
- **Success Rate:** 100% (9/9 tickers)
- **Avg Latency:** 0.26s
- **Data Coverage:** 63 days historical
- **Rate Limits:** 5 calls/min (FREE tier)
- **Verdict:** ✓ PASS - Solid alternative

### 4. **Alpha Vantage**
- **Success Rate:** 100% (9/9 tickers)
- **Avg Latency:** 0.46s
- **Data Coverage:** 95 days historical (MOST DATA)
- **Rate Limits:** 5 calls/min, 500/day (FREE tier)
- **Verdict:** ✓ PASS - Best for historical depth

---

## ❌ FAILED APIS (3/7) - NEED FIXES

### 1. **Twelve Data** - RATE LIMIT ISSUE
- **Success Rate:** 88.9% (8/9 tickers - failed on last one)
- **Error:** "You have run out of API credits for the current minute. 9 API credits were used, with the current limit being 8."
- **Root Cause:** FREE tier = 8 calls/minute, we tried 9 tickers at once
- **Solution Options:**
  - Add 8-second delay between calls to stay under limit
  - Test with 8 tickers at a time, batch the 9th
  - Upgrade to paid tier (not recommended)
- **Data Quality:** 90 days when working (excellent)
- **Fix Priority:** MEDIUM (can work around with delays)

### 2. **Finnhub** - 403 FORBIDDEN (ALL TICKERS)
- **Success Rate:** 0% (0/9 tickers)
- **Error:** "FinnhubAPIException(status_code: 403): You don't have access to this resource."
- **Root Cause:** FREE tier may not support historical daily candles for all stocks
- **Possible Issues:**
  - API key invalid/expired
  - FREE tier only supports premium/major stocks
  - Historical candles require paid subscription
- **Solution Options:**
  - Check if API key needs re-verification
  - Try different endpoint (e.g., /quote instead of /stock/candle)
  - Test with SPY/AAPL (major stocks) to confirm key works
  - Replace with alternative API if free tier insufficient
- **Fix Priority:** LOW (have 4 working alternatives)

### 3. **FMP (Financial Modeling Prep)** - 403 FORBIDDEN (ALL TICKERS)
- **Success Rate:** 0% (0/9 tickers)
- **Error:** "HTTP 403"
- **Root Cause:** API key likely invalid or FREE tier doesn't allow historical data
- **Possible Issues:**
  - API key format wrong (needs verification)
  - FREE tier restricted to limited endpoints
  - Account not activated or needs email confirmation
- **Solution Options:**
  - Re-verify API key from FMP dashboard
  - Test with /profile endpoint to confirm key works
  - Check FMP free tier documentation for allowed endpoints
  - Replace if free tier too limited
- **Fix Priority:** LOW (have 4 working alternatives)

---

## 📊 DATA COVERAGE ANALYSIS

### What We Have (Working APIs):
1. **OHLCV Daily Data:** ✅ (yfinance, Alpha Vantage, Polygon, EODHD)
2. **90-day History:** ✅ (Alpha Vantage has 95 days)
3. **Real-time Quotes:** ✅ (Polygon, yfinance)
4. **No Cost:** ✅ (All 4 passing APIs are FREE)

### What We're Missing (Data Gaps):
1. **Fundamentals:** ❌ (P/E, EPS, Market Cap, Revenue, etc.)
2. **News/Sentiment:** ❌ (Company news, analyst ratings)
3. **Options Data:** ❌ (Implied volatility, option chain)
4. **Insider Trading:** ❌ (Executive buys/sells)
5. **Short Interest:** ❌ (Borrow rates, days to cover)
6. **Dark Pool Volume:** ❌ (Off-exchange trades)
7. **Institutional Holdings:** ❌ (13F filings, ownership %)
8. **Earnings Calendar:** ❌ (Upcoming earnings dates/estimates)
9. **Social Media Sentiment:** ❌ (Reddit WallStreetBets, Twitter)
10. **Economic Indicators:** ✅ (Have FRED API key for macro data)

---

## 🔍 QUESTIONS FOR DEEPSEEK/PERPLEXITY PRO

### **Problem 1: Failed API Fixes**
"I tested 7 free market data APIs. Twelve Data hit rate limits (8/min limit, I need 9 tickers), Finnhub returned 403 Forbidden for all historical candle requests (free tier issue?), and FMP returned 403 for all requests. What are:
1. Workarounds for Twelve Data rate limit with 9 tickers?
2. Alternative FREE endpoints on Finnhub that work on free tier?
3. How to verify/fix FMP API key or alternative free fundamental data APIs?
4. Best practices for handling rate limits across multiple free APIs?"

### **Problem 2: Missing Data Sources**
"I have working free APIs for OHLCV data (yfinance, Alpha Vantage, Polygon, EODHD) but need FREE sources for:
1. **Fundamentals** (P/E, EPS, revenue, market cap) - FMP failed, alternatives?
2. **News/Sentiment** - Finnhub news endpoint, alternatives?
3. **Dark Pool Volume** - Any free sources for off-exchange trading data?
4. **Insider Trading** - SEC Form 4 filings, any APIs or scraping methods?
5. **Short Interest** - Free sources for borrow rates, days to cover?
6. **Social Media Sentiment** - Reddit/Twitter sentiment APIs (free tier)?
7. **Options Data** - Free implied volatility or option chain data?
8. **Earnings Calendar** - When companies report, consensus estimates?

What are the BEST free/low-cost APIs or scraping methods for each? Prioritize reliability and ease of use."

### **Problem 3: Data Architecture**
"With 4 working APIs (yfinance primary, Alpha Vantage/Polygon/EODHD backups), how should I architect data collection for a quantitative trading system that:
1. Needs 2+ years historical daily OHLCV for backtesting
2. Tracks 9 current holdings + 100+ watchlist stocks
3. Runs daily scans/signals before market open
4. Must handle rate limits gracefully (5/min, 500/day varies by API)
5. Needs failover if primary API goes down
6. Should cache data to minimize API calls

Best practices for: data pipeline, caching strategy, failover logic, rate limit management?"

### **Problem 4: Alternative Data Sources**
"Beyond traditional APIs, what unconventional FREE data sources can give an edge?
1. SEC EDGAR for 13F/Form 4 scraping (insider/institutional flows)?
2. Reddit API for WallStreetBets sentiment tracking?
3. Unusual Whales/alternative data aggregators (free tiers)?
4. Academic datasets (Fama-French factors, momentum, etc.)?
5. Government data (economic indicators, commodity prices)?
6. Exchange APIs directly (NASDAQ, NYSE developer programs)?

Which are most valuable for retail trader with coding skills but no budget?"

---

## 💡 IMMEDIATE ACTION PLAN

### Phase 1: Fix Failing APIs (Optional - LOW Priority)
1. **Twelve Data Rate Limit:**
   - Test with 8-second delay between calls
   - If works, keep; if not, skip (have 4 alternatives)

2. **Finnhub 403:**
   - Test `/quote` endpoint instead of candles
   - Try SPY/AAPL to verify key works
   - If free tier too limited, REJECT and remove

3. **FMP 403:**
   - Re-verify API key from dashboard
   - Test `/profile/{ticker}` endpoint
   - If fails, REJECT and find alternative for fundamentals

### Phase 2: Fill Critical Data Gaps (HIGH Priority)
1. **Fundamentals:** Find free source (Alpha Vantage has some, Yahoo Finance scraping)
2. **News/Sentiment:** Finnhub news API or Reddit API
3. **Insider Trading:** SEC EDGAR scraping (no API needed)
4. **Dark Pool:** Check if any of 4 working APIs provide this
5. **Short Interest:** FINRA publishes free (delayed) short interest data

### Phase 3: Build Data Pipeline (NEXT Notebook)
1. Create notebook 02: Data collection framework
2. Implement rate limit manager
3. Add caching layer (SQLite or Parquet files)
4. Build failover logic (try yfinance → Alpha Vantage → Polygon)
5. Test with full 100+ ticker watchlist

---

## 📈 SUCCESS METRICS

**CURRENT STATE:**
- ✅ 4/7 APIs passing (57% success rate)
- ✅ OHLCV daily data covered (yfinance primary)
- ✅ 90+ days historical available
- ✅ Zero cost (all free tiers)
- ❌ Missing fundamentals, news, dark pool, insider data
- ❌ No social sentiment tracking yet

**TARGET STATE:**
- ✅ 5+ reliable data sources (1 primary + 4 backups)
- ✅ Fundamentals covered (P/E, revenue, growth rates)
- ✅ News/sentiment pipeline (Finnhub or Reddit)
- ✅ Insider/institutional flow tracking (SEC filings)
- ✅ Dark pool volume (if available)
- ✅ Rate limit handling (<1% API errors)
- ✅ Data cache (reduce API calls by 80%)

---

## 🎯 BOTTOM LINE

**What Works RIGHT NOW:**
- yfinance (100% reliable, fast, unlimited)
- Alpha Vantage (best historical depth: 95 days)
- Polygon (solid alternative)
- EODHD (backup, limited 20/day quota)

**What We Need to Add:**
1. Fundamentals source (critical for valuation models)
2. News/sentiment (for catalyst detection)
3. Insider/dark pool (for institutional edge)
4. Data caching (to avoid rate limit hell)

**Recommendation:**
- Use yfinance as PRIMARY for all OHLCV data
- Use Alpha Vantage for fundamentals (they have some free)
- Build failover: yfinance fails → Alpha Vantage → Polygon → EODHD
- Add Reddit API for sentiment (free tier good enough)
- Scrape SEC EDGAR for insider trades (no API needed)
- Implement 24-hour data cache to minimize API calls

**Action:** Take this document to DeepSeek/Perplexity for solutions to fill gaps.
