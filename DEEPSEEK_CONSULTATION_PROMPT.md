# DeepSeek AI Consultation Prompt for Trading System

## Instructions
Copy this entire prompt and paste it into DeepSeek (or any other AI). It contains all the context about your validated signals and asks for ticker recommendations.

---

## PROMPT TO PASTE INTO DEEPSEEK:

```
I'm building a systematic trading system with validated signals. I need your help finding ideal stocks for my signal types. Here's my setup:

## MY VALIDATED SIGNALS (All tested through 12 validation gauntlets)

### Signal 1: Weekly Reversal (H16)
- Logic: Buy when 5-day return < -3%
- Best in: Range-bound, mean-reverting stocks
- OOS Return: +21.1%
- Needs: High volatility, frequent drawdowns that recover

### Signal 2: Bollinger Band Mean Reversion (H19)
- Logic: Buy when price below lower Bollinger Band (20-day, 2 std)
- Best in: Volatile stocks that revert to mean
- OOS Return: +16.5%
- Needs: Stocks that respect technical levels, not trending breakouts

### Signal 3: RSI Oversold
- Logic: Buy when RSI(14) < 30
- Best in: Stocks that have sharp selloffs then recover
- Needs: Frequent RSI extremes (>20% of time in oversold/overbought)

### Signal 4: Gap Reversal
- Logic: Buy when gap down > 2%, expect reversal
- Best in: Stocks with frequent overnight gaps
- OOS edge: +30% on SPY, even higher on volatile stocks
- Needs: High pre-market volatility, event-driven moves

### Signal 5: Volume Spike
- Logic: Buy when volume > 2x 20-day average
- Best in: Stocks where volume predicts moves
- Found to work especially well on: MSTR (+115%), AMD (+65%), TSLA (+38%)
- Needs: Normally steady volume with occasional spikes

### Signal 6: VIX-Based (Market timing, SPY/QQQ only)
- Logic: Buy SPY when VIX > 25 (fear = opportunity)
- Also: Buy when VIX volatility spikes (H128)
- Best in: Index ETFs during market fear
- These don't apply to individual stocks

## MY CURRENT WATCHLIST (48 stocks)
```
Your Portfolio: MU, LUNR, KDK, RKLB, CYPH, DGNX, MVST, RR, APLD, IONQ, UUUU, ASTS, HOOD, PALI, ELWS, SMR, DLB, LEU, KRYS, QTUM, WULF, OKLO, AQMS, B, RXRX, FSM, CRDO, ISPR, GRRR, LLY, RCAT, HMY, TM

Added (ideal for signals): SOXL, COIN, MARA, RIOT, CLSK, PLUG, FCEL, BKSY, RDW, BE, MSTR, HUT, SPCE, TQQQ, TECL
```

## CHARACTERISTICS I NEED
1. **High volatility** (>50% annualized) - more opportunities
2. **Mean-reverting behavior** - frequent MA crosses, range-bound
3. **RSI extremes** - spends >15% of time oversold or overbought
4. **Gap frequency** - gaps >2% at least 20x per year
5. **Sufficient liquidity** - >$10M daily dollar volume for execution
6. **Not pure momentum** - stocks that GO UP FOREVER don't trigger my signals

## WHAT I DON'T WANT
- Slow-moving dividend stocks (too stable)
- Pure trending stocks with no pullbacks
- Penny stocks with no volume
- Stocks with bankruptcy risk (my signals won't save them)
- Recently IPO'd with no history

## MY QUESTIONS FOR YOU:

### Question 1: Sector Gaps
Looking at my watchlist, which SECTORS am I missing that would be ideal for mean reversion signals? For example:
- Am I too heavy in any sector?
- What sectors have good mean reversion characteristics that I'm not covering?

### Question 2: Specific Ticker Recommendations
Please recommend 10-15 specific stocks I should ADD to my watchlist. For each:
- Ticker symbol
- Why it fits my signal types
- Which of my signals would work best on it
- Any risks to watch for

### Question 3: Removal Candidates
Looking at my current list, are there any stocks that seem POORLY suited for my signal types? (e.g., too stable, pure trending, etc.)

### Question 4: ETF Opportunities
What sector/thematic ETFs would be ideal for my signals? I already have:
- SOXL (3x semiconductors)
- TQQQ (3x Nasdaq)
- TECL (3x tech)

What other leveraged or volatile ETFs should I consider?

### Question 5: Risk Warnings
Based on my signal types and watchlist, what risks should I be aware of? For example:
- Sector concentration risk
- Correlation between my positions
- Market regime changes that would hurt my signals

Please be specific and practical. I'm not looking for generic advice - I want actionable ticker recommendations that fit my validated signal characteristics.
```

---

## HOW TO USE THIS

1. **Copy everything between the ``` marks above**
2. **Paste into DeepSeek** (or Claude, GPT-4, etc.)
3. **Review the recommendations**
4. **Research each ticker** before adding to watchlist
5. **Run through your signal scanner** to validate

## WHAT TO EXPECT

DeepSeek should provide:
- 10-15 specific ticker recommendations with reasoning
- Sector analysis of your current coverage
- ETF suggestions for your signal types
- Risk warnings based on your portfolio composition

## AFTER GETTING DEEPSEEK'S RESPONSE

Run these commands to add their recommendations:

```bash
# 1. Edit your watchlist config
code watchlist_config.json

# 2. Add the new symbols to the "symbols" array

# 3. Run the ideal ticker scanner to validate them
python IDEAL_TICKER_SCANNER.py

# 4. When ready, run training
# Change status to "READY" in watchlist_config.json
python WATCHLIST_TRAINING_FOUNDATION.py
```

---

## TEMPLATE FOR FOLLOW-UP QUESTIONS

After DeepSeek responds, you might ask:

```
Thanks for the recommendations. A few follow-ups:

1. For [TICKER], what's the average time it spends in RSI oversold territory?

2. You mentioned [SECTOR] - can you give me 3 more specific tickers in that space that have high volatility but aren't pure momentum plays?

3. For the removal candidates you mentioned, would they work better with a different signal type, or should I just remove them entirely?

4. What's your view on [SPECIFIC SECTOR] stocks for mean reversion? Any specific names?
```

---

## ADDITIONAL CONTEXT TO PROVIDE (if needed)

If DeepSeek asks for more context, share this:

```
My system characteristics:
- Holding period: 1-21 days (not HFT, not long-term)
- Position sizing: Equal weight or volatility-adjusted
- Risk management: Stop losses based on ATR
- Portfolio size: ~50 positions monitored, 5-10 active at any time
- Market hours only (no after-hours trading)
- US equities only (no forex, futures, crypto directly)

My validated edge:
- Mean reversion signals work best during HIGH VOLATILITY regimes
- Volume spikes predict reversals in beaten-down names
- Gap reversals work best on stocks with institutional following
- RSI extremes are more reliable on liquid stocks

What hasn't worked:
- Pure momentum following (markets are efficient)
- Overbought signals (selling strength doesn't work as well)
- Low volatility strategies (too little edge)
```
