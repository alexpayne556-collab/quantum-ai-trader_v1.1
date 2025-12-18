# Lessons Learned - Bug Documentation & Prevention

**Purpose:** Document every mistake, root cause, and fix to prevent repetition and help future contributors. Open-source institutional knowledge.

---

## Critical Bugs

### BUG-001: OHLC Column Mapping Corruption (CRITICAL)
**Date:** Dec 18, 2025 | **Severity:** CRITICAL | **Status:** ✅ FIXED

**What Happened:** Entire 518MB database had scrambled OHLC columns
- "open" column actually contained "adj_close" data
- "high" contained "close", "low" contained "high", etc.
- All 9,494 tickers affected - entire backtest results would be meaningless

**Root Cause:** yfinance returns MultiIndex columns, direct renaming misaligned positions

**The Fix:**
```python
# WRONG - Direct renaming without handling MultiIndex
df.columns = ['date', 'open', 'high', 'low', 'close', 'adj_close', 'volume', 'ticker']

# CORRECT - Flatten MultiIndex, select by NAME
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)
df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']].copy()
```

**How We Caught It:** Data quality audit found 4.2M "impossible" OHLC values (high < low)

**Prevention:**
- Always flatten MultiIndex before renaming
- Select columns by NAME, never by position
- Run data quality audit after transformations
- Add assertion: `assert (df['high'] >= df['low']).all()`

**Cost:** 36 minutes to re-download entire dataset  
**Saved:** Weeks of false research by catching before analysis

---

### BUG-002: print_flush() Missing **kwargs
**Date:** Dec 18, 2025 | **Status:** ✅ FIXED

**Issue:** `TypeError: got unexpected keyword argument 'end'`

**Fix:** Added `**kwargs` to wrapper function
```python
def print_flush(msg, **kwargs):  # Accept all print() arguments
    print(msg, **kwargs)
```

---

### BUG-003: Invalid yfinance Parameter
**Date:** Dec 18, 2025 | **Status:** ✅ FIXED

**Issue:** Used non-existent parameter `show_errors=False`

**Fix:** Removed invalid parameter, verified with `help(yf.download)`

**Prevention:** Always verify API parameters against current library version

---

### BUG-004: auto_adjust Default Changed
**Date:** Dec 18, 2025 | **Status:** ✅ FIXED

**Issue:** yfinance changed default, removed Adj Close column

**Fix:** Explicitly set `auto_adjust=False`

**Prevention:** Never rely on library defaults - always set explicitly

---

## Key Patterns Learned

**1. Data Quality Audits (From BUG-001)**
```python
def validate_ohlc(df):
    checks = {
        'high >= low': (df['high'] >= df['low']).all(),
        'close in range': ((df['close'] >= df['low']) & (df['close'] <= df['high'])).all(),
        'no negatives': (df[['open', 'high', 'low', 'close']] > 0).all().all()
    }
    return all(checks.values())
```

**2. Explicit Column Selection**
```python
# BAD: df.columns = [...]
# GOOD: df = df[['Named', 'Columns']].copy()
```

**3. MultiIndex Handling**
```python
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)
```

---

## Testing Checklist

Before trusting ANY dataset:
- [ ] Verify column names match expectations
- [ ] Check OHLC sanity (high ≥ low, close in range)
- [ ] Spot-check known values against source
- [ ] Check for nulls/duplicates
- [ ] Validate date ranges
- [ ] Compare sample row to Yahoo Finance website

---

## Statistics

**Bugs Found:** 4 (1 critical, 3 minor)  
**Time Lost:** 40 min re-download  
**Time Saved:** Weeks of false research  
**ROI:** 10 min audit script prevented catastrophic waste

**Philosophy:** "We need to know every anomaly...it could change previously thought laws" - User wisdom that saved the project

---

*Updated: Dec 18, 2025 | Next Review: After each major bug*
