#!/usr/bin/env python3
"""
COMBINED HYPOTHESIS TEST RESULTS
=================================
Batches 1-4 (Codespace) + Batches 5-9 (Shadow PC)
Date: December 20, 2025
"""

# ============================================================================
# COMBINED RESULTS SUMMARY
# ============================================================================

"""
OVERALL: 20 WINNERS out of 116 hypotheses (17.2% pass rate)

============================================================================
🏆 TOP 20 WINNERS - RANKED BY SPREAD
============================================================================

Rank | ID     | Name                      | Spread  | Sharpe | p-value  | Category
-----|--------|---------------------------|---------|--------|----------|----------
  1  | H18    | RSI Extreme (20/80)       | +109.5% |  5.89  | <0.0001  | Mean Reversion
  2  | H19    | Bollinger Band MR         |  +41.3% |  2.47  | <0.0001  | Mean Reversion
  3  | H17    | RSI Mean Reversion        |  +40.7% |  2.27  | <0.0001  | Mean Reversion
  4  | H27E   | Multi-Indicator Oversold  |  +39.5% |  2.20  | <0.0001  | Mean Reversion
  5  | H27    | Post-Large-Move Reversal  |  +39.7% |  1.85  | 0.0986   | Mean Reversion
  6  | H27C   | MA Distance               |  +36.5% |  2.10  | <0.0001  | Mean Reversion
  7  | H73    | Flight to Quality         |  +29.6% |  1.87  | 0.0007   | Cross-Asset ⭐
  8  | H27D   | Consecutive Down Reversal |  +26.4% |  1.75  | 0.0021   | Mean Reversion
  9  | H20    | VIX Mean Reversion        |  +23.1% |  1.24  | <0.0001  | Volatility
 10  | H128   | VIX Turbulence            |  +21.1% |  1.19  | 0.0002   | Creative/NEW ⭐
 11  | H21    | VIX Percentile            |  +18.2% |  1.20  | <0.0001  | Volatility
 12  | H16    | Weekly Reversal           |  +16.5% |  1.24  | 0.0006   | Mean Reversion
 13  | H27B   | Z-Score Mean Reversion    |  +16.2% |  1.15  | 0.0091   | Mean Reversion
 14  | H57    | Bond Leading Indicator    |  +14.0% |  1.25  | <0.0001  | Cross-Asset ⭐
 15  | H53    | Quarter-End Effect        |  +13.3% |  1.76  | 0.0448   | Seasonality
 16  | H62    | Oil-Equity Relationship   |  +12.1% |  1.13  | <0.0001  | Cross-Asset ⭐
 17  | H39    | Cross-Asset Vol Signal    |  +11.6% |  1.09  | 0.0538   | Volatility
 18  | H126   | Small vs Large Cap        |   +6.8% |  1.54  | <0.0001  | Sentiment ⭐
 19  | H49    | September Effect          |   +6.7% |  1.00  | 0.0245   | Seasonality
 20  | H142   | Dollar Valuation          |   +5.4% |  1.55  | 0.0013   | Creative/NEW ⭐

============================================================================
RESULTS BY CATEGORY
============================================================================

Category          | Passed | Total | Rate  | Best Signal
------------------|--------|-------|-------|---------------------------
Mean Reversion    |   9    |  12   | 75.0% | RSI Extreme (+110%)
Volatility        |   3    |  16   | 18.8% | VIX Mean Reversion (+23%)
Cross-Asset       |   4    |  11   | 36.4% | Flight to Quality (+30%)
Creative/Novel    |   2    |  18   | 11.1% | VIX Turbulence (+21%)
Seasonality       |   2    |  12   | 16.7% | Quarter-End (+13%)
Sentiment         |   1    |   9   | 11.1% | Small vs Large (+7%)
Technical         |   0    |  12   |  0.0% | (None passed)
Macro/Rates       |   0    |  15   |  0.0% | (None passed - needs FRED data)
Momentum          |   0    |  11   |  0.0% | (Signal issues)

============================================================================
KEY INSIGHTS
============================================================================

1. MEAN REVERSION DOMINATES (75% pass rate!)
   - RSI-based signals are exceptionally strong
   - Bollinger Bands confirm mean reversion works
   - Multi-indicator combinations improve reliability
   
2. VIX SIGNALS ARE RELIABLE
   - High VIX = good buying opportunity (confirmed)
   - VIX Turbulence (vol of VIX) is novel and works!
   
3. CROSS-ASSET RELATIONSHIPS MATTER
   - Flight to Quality (TLT+GLD up, SPY down) = buy signal
   - Bond Leading Indicator = TLT predicts SPY
   - Oil-Equity relationship adds value
   
4. NEW CREATIVE HYPOTHESES SHOW PROMISE
   - H128 VIX Turbulence: +21% spread (brand new signal!)
   - H142 Dollar Valuation: +5.4% (weak dollar = bullish)

5. WHAT DIDN'T WORK
   - Momentum strategies need refinement
   - Pure technical patterns underperformed
   - Macro signals need FRED API data

============================================================================
RECOMMENDED PORTFOLIO
============================================================================

TOP TIER (Highest Confidence):
- H18: RSI Extreme (20/80)     → +110% spread, 5.89 Sharpe
- H17: RSI Mean Reversion      → +41% spread, 2.27 Sharpe
- H19: Bollinger MR            → +41% spread, 2.47 Sharpe

TIER 2 (Strong Signals):
- H73: Flight to Quality       → +30% spread (cross-asset)
- H20: VIX Mean Reversion      → +23% spread
- H128: VIX Turbulence         → +21% spread (novel!)

TIER 3 (Supporting Signals):
- H57: Bond Leading            → +14% spread
- H53: Quarter-End             → +13% spread
- H62: Oil-Equity              → +12% spread

============================================================================
NEXT STEPS
============================================================================

1. Add FRED API for macro signals (yield curve, PMI, etc.)
2. Fix momentum signal logic (always-on issue)
3. Walk-forward validation on top winners
4. Combine top signals into ensemble strategy
5. Paper trade top 5 for 30 days
"""

# Winners list for programmatic use
WINNERS = [
    {'id': 'H18', 'name': 'RSI Extreme (20/80)', 'spread': 1.095, 'sharpe': 5.89, 'category': 'Mean Reversion'},
    {'id': 'H19', 'name': 'Bollinger Band MR', 'spread': 0.413, 'sharpe': 2.47, 'category': 'Mean Reversion'},
    {'id': 'H17', 'name': 'RSI Mean Reversion', 'spread': 0.407, 'sharpe': 2.27, 'category': 'Mean Reversion'},
    {'id': 'H27E', 'name': 'Multi-Indicator Oversold', 'spread': 0.395, 'sharpe': 2.20, 'category': 'Mean Reversion'},
    {'id': 'H27', 'name': 'Post-Large-Move Reversal', 'spread': 0.397, 'sharpe': 1.85, 'category': 'Mean Reversion'},
    {'id': 'H27C', 'name': 'MA Distance', 'spread': 0.365, 'sharpe': 2.10, 'category': 'Mean Reversion'},
    {'id': 'H73', 'name': 'Flight to Quality', 'spread': 0.296, 'sharpe': 1.87, 'category': 'Cross-Asset'},
    {'id': 'H27D', 'name': 'Consecutive Down Reversal', 'spread': 0.264, 'sharpe': 1.75, 'category': 'Mean Reversion'},
    {'id': 'H20', 'name': 'VIX Mean Reversion', 'spread': 0.231, 'sharpe': 1.24, 'category': 'Volatility'},
    {'id': 'H128', 'name': 'VIX Turbulence', 'spread': 0.211, 'sharpe': 1.19, 'category': 'Creative'},
    {'id': 'H21', 'name': 'VIX Percentile', 'spread': 0.182, 'sharpe': 1.20, 'category': 'Volatility'},
    {'id': 'H16', 'name': 'Weekly Reversal', 'spread': 0.165, 'sharpe': 1.24, 'category': 'Mean Reversion'},
    {'id': 'H27B', 'name': 'Z-Score Mean Reversion', 'spread': 0.162, 'sharpe': 1.15, 'category': 'Mean Reversion'},
    {'id': 'H57', 'name': 'Bond Leading Indicator', 'spread': 0.140, 'sharpe': 1.25, 'category': 'Cross-Asset'},
    {'id': 'H53', 'name': 'Quarter-End Effect', 'spread': 0.133, 'sharpe': 1.76, 'category': 'Seasonality'},
    {'id': 'H62', 'name': 'Oil-Equity Relationship', 'spread': 0.121, 'sharpe': 1.13, 'category': 'Cross-Asset'},
    {'id': 'H39', 'name': 'Cross-Asset Vol Signal', 'spread': 0.116, 'sharpe': 1.09, 'category': 'Volatility'},
    {'id': 'H126', 'name': 'Small vs Large Cap', 'spread': 0.068, 'sharpe': 1.54, 'category': 'Sentiment'},
    {'id': 'H49', 'name': 'September Effect', 'spread': 0.067, 'sharpe': 1.00, 'category': 'Seasonality'},
    {'id': 'H142', 'name': 'Dollar Valuation', 'spread': 0.054, 'sharpe': 1.55, 'category': 'Creative'},
]

if __name__ == "__main__":
    print(__doc__)
    print("\n🏆 TOP 5 WINNERS:")
    for w in WINNERS[:5]:
        print(f"  {w['id']}: {w['name']} → +{w['spread']*100:.1f}% spread, {w['sharpe']:.2f} Sharpe")
