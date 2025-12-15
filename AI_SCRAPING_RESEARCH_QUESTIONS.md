# 🤖 AI Research Questions: Stock Fundamental Data Collection

## PROJECT CONTEXT (IMPORTANT - Read First!)

### What We're Building
**Systematic AI Trading Research Engine** - A complete pipeline to research 353 stocks and automatically identify the best 50 trading opportunities using FREE data sources + GPU analysis.

**Mission:**
1. Screen 353 tickers (user's holdings + watchlists)
2. Collect multi-source data on each (OHLCV, fundamentals, insider activity, sentiment)
3. Score them on 6 factors (momentum, sentiment, insider activity, volatility, value, short squeeze)
4. Identify top 50 stocks for daily AI trading signals

### Why This Matters
- **No paid APIs** - Everything must be free (user constraint)
- **353 small-cap stocks** - Not S&P 500 (need stocks with big move potential: IONQ, ASTS, MARA, QUBT, RGTI)
- **Daily batch processing** - Runs overnight, needs to be efficient
- **GPU acceleration** - Using NVIDIA GPU for sentiment analysis (FinBERT model)
- **Fallback system** - If one source fails, automatically try alternatives

### Data Requirements
```
Per Stock:
  - OHLCV: Price, volume, returns, volatility (yfinance ✅)
  - Fundamentals: P/E, short %, insider %, market cap, beta
  - Insider Activity: Recent Form 4 filings (SEC EDGAR ✅)
  - News Sentiment: Headlines + GPU sentiment score (Google News ✅)
```

### Current Status
✅ OHLCV: yfinance works perfectly  
✅ SEC Insider: SEC EDGAR API works  
✅ News: Google News RSS works  
⚠️ **Fundamentals: NEED SOLUTION** (Yahoo scraping fails, need alternative)

---

## Questions for Perplexity Pro (Best for current web scraping methods & alternatives)

### Question 1: yfinance Library Data Coverage
```
I'm using Python's yfinance library to collect stock fundamental data. 
I successfully get: marketCap, trailingPE, forwardPE, shortPercentOfFloat, 
heldPercentInsiders, heldPercentInstitutions, beta

I'm MISSING: pegRatio, and some P/E values for pre-revenue stocks (IONQ, ASTS return -99 forwardPE).

Questions:
1. Does yfinance have ALL available fields in .info? Show me complete list.
2. For pre-revenue stocks, where can I get PEG ratio or alternative valuation metrics?
3. Should I supplement yfinance with another source for missing fields?
4. Is yfinance fast enough for 353 tickers? Any optimization tips?

Show working Python code examples.
```

### Question 2: Alternative Free Data Sources for Fundamentals
```
I need fundamental data (P/E, short %, insider %, market cap, beta, PEG ratio) for 353 stocks, 
especially small-cap/pre-revenue companies (IONQ, ASTS, MARA, QUBT, RGTI, SOUN, BBAI, SERV).

yfinance covers 80% but misses PEG ratio and has issues with pre-revenue stocks.

What are ALL the free sources for fundamentals?
1. Finviz (free version available?)
2. MarketWatch
3. Seeking Alpha
4. Trading View
5. Financial Modeling Prep
6. Alpha Vantage (I know they have limited free tier)
7. Any others?

For each source, provide:
- Which fundamental fields available
- Scraping method (if needed)
- Rate limits
- Quality/freshness of data
- Best for small-cap coverage

I need to pick 2-3 sources as fallbacks.
```

### Question 3: Efficient Batch Collection Strategy
```
I need to collect fundamentals for 353 stocks daily (batch job overnight).

Current approach: yfinance library for each ticker (too slow?)

Requirements:
1. Complete in 30-60 minutes maximum
2. Reliable (handle errors without stopping)
3. Fallback sources if primary source fails
4. Save checkpoints every 50 tickers (recovery)
5. NO payment/authentication

Best practices for:
- Async/parallel requests (which library? aiohttp, asyncio, concurrent.futures?)
- Rate limiting without getting blocked
- Error handling and retry logic
- Merging data from multiple sources
- Detecting data staleness/quality issues

Show complete Python code with all 5 requirements.
```

### Question 4: NEW - All Free Financial Data Aggregators
```
What free financial data aggregators exist that provide:
- Bulk fundamental data download/API access
- Cover small-cap US stocks
- No authentication required
- Can be scraped or have free tier

Examples I know:
- yfinance (partial)
- EDGAR SEC (filings only)
- Google Finance (limited)
- etc.

What am I missing? Any lesser-known sources that work well for small-caps?
```

---

## Questions for DeepSeek (Best for technical implementation)

### Question 1: Robust Yahoo Finance Scraper
```
Write a production-ready Python function to scrape Yahoo Finance key statistics page 
for fundamental data. Requirements:

1. Handle ALL error cases (404, 403, timeout, format changes)
2. Support multiple parsing methods (tables, JSON, BeautifulSoup CSS selectors)
3. Return standardized dict even if some fields missing
4. Include retry logic with exponential backoff
5. Log failures for debugging

Target fields: market_cap, trailing_pe, forward_pe, short_pct, insider_pct, beta

Show complete code with error handling, not pseudocode.
```

### Question 2: Multi-Ticker Batch Scraping
```
Design an efficient system to scrape fundamental data for 353 stock tickers without 
getting blocked or rate-limited. Requirements:

1. Process 353 tickers in 30-60 minutes
2. Save checkpoints every 50 tickers (recovery from failures)
3. Implement polite rate limiting (avoid bans)
4. Use parallel/async requests where safe
5. Store results in pandas DataFrame

Code should include:
- Queue management for failed tickers
- Progress tracking
- Error categorization (network vs parsing vs missing data)
- CSV export with metadata (timestamp, success rate)

Language: Python with requests, BeautifulSoup, pandas, asyncio (if needed)
```

### Question 3: Fallback Data Collection Strategy
```
Create a multi-source data collection system with fallbacks. If Yahoo Finance fails, 
automatically try alternative sources.

Priority order:
1. Yahoo Finance (free, unlimited)
2. yfinance Python library (wrapper around Yahoo)
3. MarketWatch scraping
4. Finviz scraping
5. SEC EDGAR for insider data

Write a master collector function that:
- Tries each source in order
- Merges partial results from multiple sources
- Marks data quality/freshness per field
- Returns best available data even if incomplete

Show Python code with clear source attribution per field.
```

---

## Context to Provide (Copy/Paste This)

**Current Setup:**
- Python 3.11.9
- Libraries: pandas, requests, beautifulsoup4, lxml, yfinance
- Environment: Windows (Shadow PC) + Linux (Codespace)
- GPU: Available for processing

**Current Code Issues:**
```python
# This fails with "No tables found on page"
tables = pd.read_html(response.text)

# Error pattern:
# ValueError: No tables found
# OR KeyError: columns not in index
```

**What We Need:**
- Fundamental data for 353 small-cap stocks (not S&P 500)
- Daily batch processing (run overnight)
- 100% free (no paid APIs)
- Data fields: P/E, short %, insider %, market cap, beta, volume

**Tickers to Test:**
IONQ, ASTS, HOOD, MARA, PLTR, RGTI, QUBT, SOUN, BBAI, SERV

---

## ACTUAL TEST RESULTS - yfinance Coverage

We tested yfinance on 5 real stocks (IONQ, ASTS, HOOD, MARA, PLTR):

### Successfully Retrieved Fields:
✅ marketCap (all 5)
✅ trailingPE (3/5 - IONQ, ASTS missing)
✅ forwardPE (all 5, but some return -99 for pre-revenue)
✅ priceToBook (all 5)
✅ priceToSalesTrailing12Months (all 5)
✅ shortPercentOfFloat (all 5)
✅ shortRatio (all 5)
✅ heldPercentInsiders (all 5)
✅ heldPercentInstitutions (all 5)
✅ beta (all 5)
✅ trailingAnnualDividendYield (all 5)

### Missing Fields:
❌ pegRatio (none of the 5)
❌ priceToBook proxy for value scoring (incomplete for pre-revenue)

### Verdict:
**yfinance covers ~85-90% of needs.** Need fallback source for:
1. PEG ratio (all stocks)
2. Better trailing P/E for pre-revenue companies

### Next Steps:
- Use yfinance as primary (fast, reliable)
- Add 1-2 fallback sources for missing fields
- Use what you get from AIs for optimal combination

---

## Questions to Ask After Getting Answers

1. **If pandas.read_html doesn't work:**
   - "The HTML structure has changed, show me how to use CSS selectors to extract the same data"

2. **If getting blocked:**
   - "I'm getting 403 errors even with proper headers, what proxy or session rotation strategies work?"

3. **If data is missing:**
   - "Some fields return N/A, which alternative sources have better coverage for small-cap stocks?"

4. **If too slow:**
   - "Processing 353 tickers takes 3+ hours, how can I use asyncio to speed this up without getting banned?"

---

## Copy These Exact Prompts

**Perplexity Prompt (Copy This):**
```
I'm building a systematic stock research engine that screens 353 small-cap stocks daily. 

Current data collection:
- OHLCV: yfinance ✅ (fast, reliable)
- News: Google News RSS ✅ (free)
- Insider trades: SEC EDGAR API ✅ (free)
- Fundamentals: NEED SOLUTION ⚠️

I tested yfinance.Ticker().info for fundamentals. It successfully returns:
marketCap, trailingPE, shortPercentOfFloat, heldPercentInsiders, beta, etc.

BUT it's MISSING: PEG ratio, and has issues with pre-revenue stocks (returns -99 for forwardPE).

Test results show yfinance covers ~80% of what I need.

Questions for my 353-ticker batch job:
1. What sources give me the missing 20% (PEG ratio, valuation for pre-revenue stocks)?
2. Should I use yfinance + 1-2 alternative sources for fallback?
3. What's the fastest way to collect fundamentals for 353 stocks daily?
4. Any free aggregators I'm missing that have better small-cap coverage?

Show me working Python code for multi-source collection with fallbacks.

Test stocks: IONQ, ASTS, HOOD, MARA, PLTR (small-caps that yfinance partially misses)
```

**DeepSeek Prompt (Copy This):**
```
Project: Systematic AI stock research engine for 353 tickers.

Goals:
1. Collect 6 data sources per stock (OHLCV, fundamentals, insider, sentiment, etc.)
2. Run daily batch overnight (complete in 30-60 min)
3. All FREE (no paid APIs)
4. Handle failures gracefully (fallback sources)
5. Save checkpoints every 50 tickers

Current solution status:
✅ OHLCV: yfinance works perfectly
✅ Insider trades: SEC EDGAR JSON API works
✅ News scraping: Google News RSS works  
✅ GPU sentiment: FinBERT on GPU (50+ texts/sec)
⚠️ Fundamentals: yfinance covers 80%, need fallback for remaining 20%

yfinance test results show it successfully fetches most fields but MISSES:
- PEG ratio (all stocks)
- Trailing P/E for pre-revenue stocks (IONQ, ASTS return N/A)

Task 1: Write production code for multi-source fundamental collection:
```python
def collect_fundamentals_multigsource(ticker):
    '''
    Try yfinance first, fallback to alternative sources
    Return dict with ALL fields or N/A
    '''
    # Priority: yfinance → Finviz → MarketWatch → None
```

Task 2: Design efficient batch processor for 353 stocks:
- Use asyncio or concurrent.futures for parallelism
- Checkpoint every 50 tickers
- Error categorization (network vs parsing vs missing data)
- Rate limiting (respectful, no blocking)
- Progress tracking
- Save to CSV with metadata

Task 3: Show me which sources to use as fallbacks:
- Where to get PEG ratio?
- Where to get valuation for pre-revenue stocks?
- How to merge data from multiple sources?
- How to handle conflicting data between sources?

Code should be production-ready, not pseudocode.
Target fields: market_cap, trailing_pe, forward_pe, peg_ratio, short_pct, 
insider_pct, beta, price_to_book, price_to_sales

Test tickers: IONQ, ASTS, HOOD, MARA, PLTR, RGTI, QUBT, SOUN, BBAI, SERV
```

---

## Next Steps

1. **Ask both AIs simultaneously** (get different perspectives)
2. **Compare their solutions** (Perplexity = current methods, DeepSeek = robust code)
3. **Test on 5 stocks first** (IONQ, ASTS, HOOD, MARA, PLTR)
4. **Report back errors** to this agent for integration into notebook
5. **Scale to full 353** after validation

---

## Alternative: Use yfinance Library Instead

While asking AIs, also test this backup approach:

```python
import yfinance as yf

def get_fundamentals_yfinance(ticker):
    """Backup method using yfinance library"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        return {
            'ticker': ticker,
            'market_cap': info.get('marketCap', 'N/A'),
            'trailing_pe': info.get('trailingPE', 'N/A'),
            'forward_pe': info.get('forwardPE', 'N/A'),
            'short_pct': info.get('shortPercentOfFloat', 'N/A'),
            'insider_pct': info.get('heldPercentInsiders', 'N/A'),
            'institution_pct': info.get('heldPercentInstitutions', 'N/A'),
            'beta': info.get('beta', 'N/A'),
        }
    except Exception as e:
        return {'ticker': ticker, 'error': str(e)}

# Test this NOW while waiting for AI responses
test = get_fundamentals_yfinance('IONQ')
print(test)
```

This might just work without any scraping! 🚀
