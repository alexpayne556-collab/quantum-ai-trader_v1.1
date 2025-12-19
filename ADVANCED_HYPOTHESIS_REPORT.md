# Advanced Hypothesis Testing Report
## Renaissance-Grade Research Sprint: Day 2

**Date:** December 19, 2025  
**Objective:** Test 4 sophisticated market hypotheses using FREE data  
**Framework:** Walk-forward validation with Monte Carlo significance testing

---

## Executive Summary

### 🔬 What We Tested
Using 101 FREE time series (15+ years of data), we tested 4 advanced market hypotheses:

| Hypothesis | Core Thesis | Result |
|------------|-------------|--------|
| VIX Term Structure | Contango/backwardation predicts volatility ETF returns | ⚠️ Works in-sample, fails walk-forward |
| Intermarket Momentum | Commodity momentum predicts sector returns | ❌ 0/10 relationships beat B&H |
| Yield Curve Regime | Inverted curve signals defensive positioning | ❌ Loses to 60/40 by 0.24 Sharpe |
| Sector Pairs Z-Score | Mean reversion in correlated pairs | ❌ 0/9 pairs beat 50/50 baseline |

### 📊 Final Verdict: NO TACTICAL ALPHA IN ETFs

After rigorous testing:
- **0 out of 4 hypotheses** survived walk-forward validation
- **0 out of 23 individual tests** beat their proper baselines with statistical significance
- ETF markets are **highly efficient** - simple signals have been arbitraged away

---

## Methodology

### Data Infrastructure (100% FREE)
```
✓ yfinance:     101 instruments, 15+ years each
✓ Categories:   Volatility, Bonds, Commodities, Sectors, International
✓ Alignment:    Cleaned to 4,020 unique trading days
✓ Cost:         $0 (unlimited API calls)
```

### Validation Framework
1. **In-Sample Metrics:** Total return, CAGR, Sharpe ratio, max drawdown
2. **Proper Baselines:** Not "not trading" but "random entry" or "passive allocation"
3. **Monte Carlo:** 5,000-10,000 simulations per test
4. **Walk-Forward:** 1-year train, 3-month test, rolling quarterly
5. **Statistical Test:** T-test on out-of-sample excess Sharpe (p < 0.05)

---

## Hypothesis 1: VIX Term Structure Engine

### Theory
VIX futures curve shape (contango vs backwardation) predicts volatility ETF returns.
- **Contango** (VIX < VIX3M): Volatility decay → Short VXX
- **Backwardation** (VIX > VIX3M): Fear spike → Long VXX

### Initial Results (In-Sample)
```
Term Structure Strategy:  +153% total, 0.51 Sharpe
Always Short VXX:         -25% total, 0.35 Sharpe
Buy & Hold VXX:           -98% total, -0.55 Sharpe

✓ Beats naive "always short" baseline!
```

### Regime Analysis
```
HIGH VOLATILITY (SPY vol > 20%):
  Term Structure:  +222% annualized
  Always Short:    -149% annualized
  Difference:      +371% ✓

LOW VOLATILITY:
  Term Structure:  +16% annualized
  Always Short:    +41% annualized
  Difference:      -25% ✗
```

### Monte Carlo Result
- Real Sharpe at **86.4th percentile** vs random signals
- NOT quite significant (need >95%)

### Walk-Forward Result
```
Test Periods: 7
Beat Baseline: 3/7 (43%)
Avg Excess Sharpe: -0.41
T-stat p-value: 0.79

❌ FAILED WALK-FORWARD VALIDATION
```

### Conclusion
The VIX term structure signal shows promise in high-volatility regimes but does not survive out-of-sample testing. The in-sample alpha was likely due to overfitting to specific volatility events (COVID crash, 2022 drawdown).

---

## Hypothesis 2: Intermarket Momentum Cascade

### Theory
Information diffuses slowly across asset classes. Commodity momentum should predict next-week sector returns.

### Pairs Tested
| Leader | Follower | Direction | Result |
|--------|----------|-----------|--------|
| Crude Oil (CL) | Energy (XLE) | Same | ❌ -0.28 excess Sharpe |
| Crude Oil (CL) | Industrials (XLI) | Same | ❌ -0.58 excess Sharpe |
| Gold (GC) | Financials (XLF) | Inverse | ❌ -0.54 excess Sharpe |
| Gold (GC) | Utilities (XLU) | Same | ❌ -0.31 excess Sharpe |
| Gold (GC) | Treasuries (TLT) | Same | ❌ -0.08 excess Sharpe |
| Yields (TNX) | Financials (XLF) | Same | ❌ -0.30 excess Sharpe |
| Yields (TNX) | Treasuries (TLT) | Inverse | ❌ -0.32 excess Sharpe |
| Dollar (UUP) | Emerging (EEM) | Inverse | ❌ -0.15 excess Sharpe |
| Dollar (UUP) | Gold (GLD) | Inverse | ❌ -0.51 excess Sharpe |
| VIX | SPY | Inverse | ❌ -0.54 excess Sharpe |

### Lag Optimization
Tested lags from 1 to 20 days. **No lag produced positive excess Sharpe.**

### Conclusion
Cross-asset information diffusion, if it exists, happens too quickly to capture with daily data. These relationships are priced in within minutes, not days.

---

## Hypothesis 3: Yield Curve Regime Strategy

### Theory
Yield curve shape predicts equity/bond relative performance.
- **Normal/Steep curve** → Risk-on → Favor stocks
- **Inverted curve** → Recession risk → Favor bonds

### Regime Distribution
```
STEEP (spread > 1.5):    1,899 days (47%)
NORMAL:                  1,502 days (37%)
INVERTED (spread < 0):   614 days (15%)
```

### Regime-Conditional Returns
| Regime | SPY Ann | TLT Ann | Winner |
|--------|---------|---------|--------|
| STEEP | +13.0% | +1.6% | SPY (+0.71 Sharpe) |
| NORMAL | +16.4% | +1.6% | SPY (+0.74 Sharpe) |
| INVERTED | +14.6% | +16.2% | TLT (+0.03 Sharpe) |

### Strategy Performance
```
YC Switch Strategy: +546% total, 0.76 Sharpe
60/40 Static:       +370% total, 0.99 Sharpe

Excess Sharpe: -0.24 ❌
```

### Conclusion
The yield curve switch produces higher total returns but **worse risk-adjusted returns** than simple 60/40. The diversification benefit of static allocation beats tactical timing.

---

## Hypothesis 4: Sector Pairs Z-Score Mean Reversion

### Theory
When correlated sectors diverge (Z-score > |2|), they mean-revert. Trade the spread.

### Pairs Tested
| Pair | Strategy Sharpe | Baseline (50/50) | Excess |
|------|-----------------|------------------|--------|
| XLF/XLK | 0.50 | 0.82 | ❌ -0.32 |
| XLE/XLV | -0.26 | 0.59 | ❌ -0.86 |
| XLY/XLP | -0.03 | 0.85 | ❌ -0.88 |
| XLI/XLU | -0.09 | 0.78 | ❌ -0.87 |
| XLB/XLK | 0.39 | 0.77 | ❌ -0.38 |
| IWM/SPY | 0.33 | 0.69 | ❌ -0.36 |
| EEM/SPY | -0.06 | 0.56 | ❌ -0.62 |
| TLT/SPY | 0.16 | 0.95 | ❌ -0.79 |
| GLD/SPY | 0.18 | 1.01 | ❌ -0.82 |

### Monte Carlo Note
XLF/XLK achieved 98th percentile vs random - but this means the signal has STRUCTURE, not ALPHA. The structure is worse than passive allocation.

### Conclusion
Sector pair mean reversion does not work at the ETF level. Static 50/50 allocation consistently beats Z-score timing.

---

## Final Walk-Forward Results

### Initial "Survivors"
Two strategies showed promise in full-sample tests:
1. VIX Term Structure in High Vol: 1.89 Sharpe, 97th percentile
2. Dollar → EEM: 0.31 vs 0.30 Sharpe (marginal)

### Walk-Forward Validation
```
SURVIVOR 1: VIX Term Structure (High Vol)
  Test Periods: 7
  Beat Baseline: 3/7 (43%)
  T-test p-value: 0.79
  Result: ❌ FAILED

SURVIVOR 2: Dollar → EEM
  Test Periods: 59
  Beat Baseline: 24/59 (41%)
  T-test p-value: 0.52
  Result: ❌ FAILED
```

---

## What This Means

### The ETF Market is Efficient
- Simple technical and cross-asset signals do not generate alpha
- Any patterns visible in historical data have been arbitraged away
- This is EXPECTED for highly liquid, institutionally traded markets

### Your Framework Works
The framework successfully:
1. Killed naive hypotheses (intermarket momentum, yield curve timing)
2. Found that "promising" signals collapse under proper testing
3. Applied the same rigor that killed your original 45 edges

### What's NOT Ruled Out
- **Individual stocks**: Less analyst coverage, more noise to exploit
- **Higher-frequency signals**: Intraday patterns (requires different data)
- **Alternative data**: Sentiment, flow, satellite imagery
- **Factor timing**: Momentum, value, quality at factor level
- **Risk management**: Signals may reduce drawdowns even without alpha

---

## Strategic Recommendations

### Path A: Accept Efficiency
Use ETFs for what they're good at:
- **Passive allocation**: 60/40, risk parity, target date
- **Factor exposure**: Momentum (MTUM), Quality (QUAL), Value (VLUE)
- **Cost-efficient diversification**: One trade = entire market

### Path B: Search in Less Efficient Markets
Invest in survivorship-bias-free stock data ($200-500):
- Test same hypotheses on 3,000+ individual stocks
- Dispersion within ETFs may hide stock-level alpha
- Less analyst coverage = more inefficiency

### Path C: Improve Signal Sophistication
Move beyond daily price patterns:
- Machine learning on multi-factor combinations
- Sentiment analysis (news, social media)
- Order flow and market microstructure

---

## Files Created

```
data/free_harvest/                    # 101 instrument CSVs
data/free_harvest/COMBINED_PRICE_MATRIX.csv
data/free_harvest/VIX_TERM_STRUCTURE.csv
data/free_harvest/YIELD_CURVE.csv
data/free_harvest/SECTOR_PAIR_ZSCORES.csv

data/HYPOTHESIS_1_VIX_TERM_STRUCTURE.csv
data/HYPOTHESIS_1_RESULTS.csv
data/HYPOTHESIS_2_INTERMARKET.csv
data/HYPOTHESIS_3_YIELD_CURVE.csv
data/HYPOTHESIS_4_PAIRS.csv
data/SURVIVOR_1_WALKFORWARD.csv
data/SURVIVOR_2_WALKFORWARD.csv

FREE_DATA_HARVESTER.py               # Reusable data collection system
ADVANCED_HYPOTHESIS_REPORT.md        # This report
```

---

## Conclusion

**This sprint was a success.** Not because we found alpha, but because we DIDN'T find alpha where it doesn't exist.

> "In God we trust, all others must bring data." - W. Edwards Deming

The data brought the truth: ETF markets are efficient. Your framework is Renaissance-grade - it kills bad ideas before they cost money.

**Next experiment:** Test these same hypotheses on individual stocks. The inefficiency may be hiding in the dispersion.

---

*Generated by Quantum AI Trader Research Framework v1.1*
