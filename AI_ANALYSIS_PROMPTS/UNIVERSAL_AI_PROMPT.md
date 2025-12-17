# 🤖 COMBINED AI ANALYSIS PROMPT - UNIVERSAL VERSION
## For: ChatGPT, DeepSeek, Perplexity, Claude, Gemini, or any AI

---

## THE SITUATION

I've discovered 65+ statistical trading edges through 12+ hours of systematic backtesting. I have $100K paper trading on Alpaca with 10 positions queued. Before I build automation, I need to know:

**AM I MISSING SOMETHING BIG?**

---

## MY COMPLETE FINDINGS

### Crash Bounce Edges (Buy Extreme Dips):
| Stock | Threshold | Win Rate | Avg Return | Sample |
|-------|-----------|----------|------------|--------|
| RGTI | -30% week | 100% | +40.2% | n=6 |
| QBTS | -30% week | 80% | +23.5% | n=5 |
| DELL | -20% week | 83% | +8.0% | n=6 |
| HUT | -30% week | 83% | +13.9% | n=6 |
| CHPT | -20% week | 82% | +8.4% | n=17 |
| ARM | -20% week | 86% | +6.5% | n=7 |
| MSTR | -20% week | 90% | +10.5% | n=10 |
| LEU | -20% week | 75% | +12.4% | n=8 |
| HOOD | -15% week | 79% | +9.6% | n=14 |

### Momentum Continuation Edges:
| Stock | Threshold | Win Rate | Avg Return | Sample |
|-------|-----------|----------|------------|--------|
| PLTR | +30% week | 91% | +4.9% | n=11 |
| CEG | +20% week | 85% | +2.8% | n=13 |
| VST | +15% week | 79% | +6.1% | n=28 |
| IONQ | +40% week | 77% | +8.5% | n=13 |
| ASTS | +30% week | 71% | +10.5% | n=41 |

### RSI + VIX Combination Edges:
| Stock | Condition | Win Rate | Avg Return | Sample |
|-------|-----------|----------|------------|--------|
| QQQ | RSI<30 + VIX>30 | 100% | +4.4% | n=16 |
| NVDA | RSI<30 + VIX>30 | 93% | +8.2% | n=14 |
| SPY | RSI<30 + VIX>30 | 89% | +3.8% | n=19 |
| NVDA | RSI<25 | 100% | +10.4% | n=5 |
| HOOD | RSI<20 | 92% | +5.7% | n=12 |

### Volume Surge Edges:
| Stock | Condition | Win Rate | Avg Return | Sample |
|-------|-----------|----------|------------|--------|
| HOOD | Vol>2x + Down | 100% | +8.9% | n=7 |
| PLTR | Vol>2x + Down | 88% | +5.4% | n=8 |
| RGTI | Vol>2x + Down | 80% | +17.0% | n=10 |

### Consecutive Down Days:
| Index | Days Down | Win Rate (5D) | Sample |
|-------|-----------|---------------|--------|
| SPY | 5+ days | 100% | n=4 |
| QQQ | 5+ days | 80% | n=5 |

### Calendar Effects:
- SPY Monday: 65% win rate
- AAPL End of Month: 61% win rate
- AMD Gap Up ≥3%: 61% fade (short)

### DEATH TRAPS (Edges that FAIL):
- AMC crash: 29% bounce (AVOID)
- SMR crash: 32% bounce (AVOID)
- RIOT/MARA: 0-28% (AVOID)
- SPCE: death trap
- TSLA Thursday: 40% (AVOID)

---

## CRITICAL DISCOVERY: VIX REGIME

My edges have REGIME DEPENDENCY:
- **VIX > 30**: RSI strategies = 93-100% WR
- **VIX 25-30**: RSI strategies = 82-89% WR
- **VIX < 20**: RSI strategies DEGRADE
- **Current VIX**: 16.48 (LOW)

This means my best edges don't work in current market!

---

## WHAT I NEED FROM YOU

### 1. PATTERN GAPS
What major trading patterns am I NOT testing that have proven statistical edge? Think:
- Mean reversion variations
- Momentum variations
- Market microstructure
- Cross-asset signals
- Alternative data

### 2. STATISTICAL CONCERNS
- Am I overfitting with n=5-17 samples?
- Is 2 years enough data?
- How do I validate out-of-sample?
- Multiple hypothesis testing correction?

### 3. LOW VIX STRATEGIES
Since my best edges need VIX>25, what works when VIX is LOW?
- Different indicators?
- Different stocks?
- Different timeframes?

### 4. COMPOUND SIGNALS
Should I combine indicators?
- RSI + Volume + VIX + Day of Week?
- How to weight each factor?
- Machine learning approach?

### 5. RISK MANAGEMENT
- Optimal position sizing?
- Stop-loss for mean reversion?
- Max drawdown protection?
- Kelly criterion application?

### 6. EDGE DECAY
- How quickly do retail edges decay?
- How to monitor degradation?
- When to retire an edge?

### 7. ALTERNATIVE DATA
What data sources would improve edge detection?
- Options flow
- Short interest
- Insider buying
- Social sentiment
- Dark pool data

### 8. ACADEMIC BACKING
What academic research supports or contradicts my findings?
- Mean reversion papers
- Momentum papers
- Retail behavior studies
- Market microstructure

---

## MY HYPOTHESIS

**Theory**: High-beta retail favorites exhibit mean reversion because:
1. Retail panics at extremes → overselling
2. Institutions buy the dip
3. Price reverts in 5 days

**Question**: Is this sound? What would disprove it?

---

## SPECIFIC QUESTIONS

1. Why does RGTI -30% bounce 100% but RIOT -30% bounces only 28%?

2. Why does PLTR momentum work (91%) but MARA momentum fails (0-7%)?

3. Are these edges real or am I curve-fitting?

4. What's the single most important edge I'm probably missing?

5. If you could add ONE feature to my analysis, what would it be?

---

## OUTPUT REQUEST

Please provide:
1. **#1 Blind Spot**: The biggest thing I'm missing
2. **New Test Ideas**: 3-5 specific backtests to run
3. **Statistical Validation**: How to prove these edges are real
4. **Low VIX Strategy**: What to trade when fear is low
5. **Risk Warning**: What could blow this up

Be direct. I want truth, not comfort.

---

*Data: 2 years daily OHLCV via yfinance*
*Sample minimum: n≥5*
*Holding period: 5 days*
*Paper trading: Alpaca $100K*
