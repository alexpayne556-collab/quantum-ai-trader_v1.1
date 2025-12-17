# DeepSeek Analysis Prompt - Quantum AI Trader Edge Discovery

## CONTEXT
I've spent 12+ hours running systematic backtests on stock patterns. I have Alpaca paper trading connected with $100K. I've discovered 65+ trading edges with statistical validation. I need your help finding what I might be MISSING.

## MY METHODOLOGY
- Used yfinance for 2 years of daily OHLCV data
- Tested crash bounce patterns (-10%, -15%, -20%, -25%, -30% weekly moves)
- Tested momentum continuation (+15%, +20%, +25%, +30%, +40% weekly moves)
- Tested RSI extremes (<20, <25, <30) combined with VIX regimes
- Tested volume surge patterns (>1.5x, >2x, >3x average)
- Tested consecutive down days (3, 4, 5+ days)
- Tested day-of-week patterns (Monday through Friday)
- Tested month-end effects
- Tested gap fill patterns
- Minimum sample size: n≥5 for validity

## MY TOP DISCOVERIES (65+ edges)

### TIER S+ (90%+ Win Rate):
1. RGTI -30% week → 100% bounce, +40.2% avg (n=6)
2. HOOD Vol>2x + Down Day → 100% bounce, +8.9% avg (n=7)
3. QQQ RSI<30 + VIX>30 → 100% WR, +4.4% avg (n=16)
4. NVDA RSI<25 → 100% bounce, +10.4% avg (n=5)
5. SPY 5+ consecutive down days → 100% up in 5D (n=4)
6. AVGO RSI<30 + Vol>1.5x → 93% WR, +9.7% avg (n=14)
7. HOOD RSI<20 → 92% bounce, +5.7% avg (n=12)
8. PLTR +30% week → 91% continue, +4.9% avg (n=11)
9. MSTR -20% week → 90% bounce, +10.5% avg (n=10)

### TIER S (80-89% WR):
- PLTR Vol>2x + Down → 88% bounce
- ARM -20% week → 86% bounce
- DELL -20% week → 83% bounce
- HUT -30% week → 83% bounce
- CHPT -20% week → 82% bounce
- QBTS -30% week → 80% bounce

### KEY DEATH TRAPS IDENTIFIED:
- AMC crash → 29% bounce (AVOID)
- SMR crash → 32% bounce (AVOID)
- RIOT/MARA momentum → 0-7% continue (AVOID)
- SPCE anything → death trap
- TSLA Thursday → 40% WR (AVOID)

### CALENDAR EDGES:
- SPY Monday → 65% WR
- NVDA Monday → 62% WR
- AAPL end of month → 61% WR
- AMD Gap Up ≥3% → 61% fade

## VIX REGIME DISCOVERY
- VIX > 30: RSI<30 strategies hit 93-100% WR
- VIX > 25: RSI<30 strategies hit 82-89% WR
- VIX < 20: RSI strategies degrade significantly
- Current VIX: 16.48 (LOW - not ideal)

## WHAT I NEED FROM YOU

1. **PATTERN GAPS**: What major trading patterns am I NOT testing that have academic/statistical backing?

2. **FEATURE ENGINEERING**: What additional features should I calculate from OHLCV data that might reveal hidden edges?

3. **REGIME ANALYSIS**: Beyond VIX, what other regime indicators should I incorporate? (yield curve, credit spreads, etc.)

4. **SECTOR ROTATION**: Am I missing sector-specific patterns that work better than individual stocks?

5. **TIME DECAY**: Should I be looking at intraday patterns, or is daily granularity optimal?

6. **CORRELATION ANALYSIS**: What correlation patterns between assets might reveal arbitrage opportunities?

7. **SENTIMENT INTEGRATION**: How should I incorporate news sentiment, options flow, or social media data?

8. **RISK MANAGEMENT**: What position sizing or stop-loss rules work best with mean-reversion strategies?

9. **MARKET MICROSTRUCTURE**: Are there patterns around market open, close, or specific times that I should test?

10. **STATISTICAL VALIDATION**: Am I overfitting? What out-of-sample validation methods should I use?

## SPECIFIC QUESTIONS

1. The RGTI -30% → 100% bounce edge seems too good. Is this survivorship bias or a legitimate inefficiency in quantum computing hype stocks?

2. Why do MARA/RIOT momentum plays fail (0-7%) while PLTR momentum works (91%)? What's the underlying mechanism?

3. Should I be combining multiple indicators (RSI + Volume + VIX + Day of Week) into compound signals?

4. What machine learning approaches would help find non-linear edge combinations?

5. Are there any academic papers on retail trader behavior patterns I should study?

## MY DATA SOURCES
- yfinance (free)
- Finnhub (news, insider trades)
- Alpha Vantage (fundamentals)
- FRED (macro data)
- Polygon (if needed)
- FMP (financials)

What gold am I leaving on the table?
