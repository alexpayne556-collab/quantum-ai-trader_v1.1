# Claude Opus Analysis Prompt - Deep Pattern Recognition

## ROLE
You are a senior quantitative researcher at a hedge fund. I'm showing you my systematic trading edge discovery process. Critique it ruthlessly and tell me what I'm missing.

## MY BACKGROUND
- 12+ hours of systematic backtesting
- 145,530 parameter sweeps completed
- 65+ trading edges discovered
- Alpaca paper trading connected ($100K)
- 10 positions queued for market open validation

## COMPLETE EDGE DATABASE

### TIER S+ (100% Win Rate):
```
RGTI -30% week → 100% bounce, +40.2% avg (n=6)
HOOD Vol>2x + Down Day → 100% bounce, +8.9% avg (n=7)
QQQ RSI<30 + VIX>30 → 100% WR, +4.4% avg (n=16)
NVDA RSI<25 → 100% bounce, +10.4% avg (n=5)
SPY 5+ down days → 100% up in 5D (n=4)
```

### TIER S (90-99% Win Rate):
```
AVGO RSI<30 + Vol>1.5x → 93% WR, +9.7% avg (n=14)
NVDA RSI<30 + VIX>30 → 93% WR, +8.2% avg (n=14)
HOOD RSI<20 → 92% bounce, +5.7% avg (n=12)
PLTR +30% week → 91% continue, +4.9% avg (n=11)
MSTR -20% week → 90% bounce, +10.5% avg (n=10)
```

### TIER A (80-89% Win Rate):
```
PLTR Vol>2x + Down → 88% bounce, +5.4% (n=8)
ARM -20% week → 86% bounce, +6.5% (n=7)
CEG +20% week → 85% continue, +2.8% (n=13)
DELL -20% week → 83% bounce, +8.0% (n=6)
HUT -30% week → 83% bounce, +13.9% (n=6)
CHPT -20% week → 82% bounce, +8.4% (n=17)
QBTS -30% week → 80% bounce, +23.5% (n=5)
```

### TIER B (65-79% Win Rate):
```
HOOD -15% week → 79% bounce, +9.6% (n=14)
VST +15% week → 79% continue, +6.1% (n=28)
IONQ +40% week → 77% continue, +8.5% (n=13)
LEU -20% week → 75% bounce, +12.4% (n=8)
AVGO -10% week → 73% bounce, +7.0%
ASTS +30% week → 71% continue, +10.5% (n=41)
S -15% week → 70% bounce, +1.2% (n=10)
RGTI -20% week → 69% bounce, +14.9% (n=26)
CHPT -15% week → 68% bounce, +5.0% (n=41)
SPY Monday → 65% WR, +0.16% avg
```

### DEATH TRAPS (Never Trade):
```
AMC crash → 29% bounce
SMR crash → 32% bounce
RIOT crash → 25-28% bounce
MARA momentum → 0-7% continue
SPCE anything → death
TSLA Thursday → 40% WR
MSTR Day 1 → 36% WR
```

## MY METHODOLOGY

### Data:
- 2 years daily OHLCV (yfinance)
- RSI(14), Volume ratios, Weekly returns
- VIX regime overlay
- Sample size minimum: n≥5

### Testing Framework:
```python
# For each stock:
# 1. Calculate weekly return
# 2. Identify threshold crossings (-10%, -15%, -20%, etc.)
# 3. Calculate forward 5-day return
# 4. Compute win rate and average return
# 5. Require n≥5 for validity
```

### Regime Filtering:
- VIX>30: High fear (RSI strategies excel)
- VIX 20-30: Normal
- VIX<20: Complacent (strategies degrade)

## CRITICAL ANALYSIS REQUESTED

### 1. STATISTICAL RIGOR
- Am I overfitting with n=5-17 samples?
- Should I use bootstrapping or Monte Carlo validation?
- Is 2 years of data enough for regime analysis?
- What's the proper way to adjust for multiple hypothesis testing?

### 2. SURVIVORSHIP BIAS
- RGTI, QBTS, IONQ are all recent IPOs - am I seeing true edges or just survivor bias?
- How do I validate these edges won't disappear?

### 3. REGIME DEPENDENCY
- My best edges require VIX>25 which only happens ~20% of the time
- How do I build a system that works in ALL regimes?
- What do I trade when VIX is low?

### 4. MISSING FACTORS
What important factors am I NOT considering?
- Liquidity/bid-ask spread?
- Short interest?
- Options gamma exposure?
- Sector rotation?
- Macro calendar?
- Fed policy regime?

### 5. PORTFOLIO CONSTRUCTION
- How should I combine these edges?
- What's optimal position sizing?
- How do I avoid correlation clustering?
- Should I use Kelly criterion?

### 6. EXECUTION REALITY
- Will these edges survive transaction costs?
- Is market impact a concern for these stocks?
- Should I use limit orders or market orders?
- What's the optimal entry timing?

### 7. MACHINE LEARNING OPPORTUNITY
- Should I train an ML model on these features?
- What architecture would work best?
- How do I avoid overfitting?
- What's the value-add vs simple rules?

### 8. EDGE DECAY
- How quickly do retail edges decay?
- Is there alpha in being early vs following?
- How do I monitor edge degradation?

### 9. ALTERNATIVE DATA
What alternative data sources would improve edge detection?
- Options flow
- Dark pool prints
- Social sentiment
- Insider transactions
- Institutional holdings
- Short interest changes

### 10. RISK MANAGEMENT
- What's the optimal stop-loss for mean reversion?
- How do I handle a streak of losers?
- What's max position size per edge?
- How do I manage correlation risk?

## HYPOTHESIS FOR YOU TO EVALUATE

**My Theory**: High-beta retail favorites (HOOD, PLTR, RGTI) exhibit mean reversion because retail panic selling creates temporary dislocations that institutional buyers exploit. The edge exists because:
1. Retail sells at extremes (fear/greed)
2. Institutions buy the dip
3. Price reverts to mean within 5 days

**Question**: Is this theory sound? What would disprove it?

## OUTPUT FORMAT

Please provide:
1. **Blind spots**: What major patterns am I missing?
2. **Statistical concerns**: Where is my methodology weak?
3. **Enhancement ideas**: Specific tests I should run
4. **Risk factors**: What could blow up these strategies?
5. **Next steps**: Prioritized list of additional research

Be harsh. I want the truth, not comfort.
