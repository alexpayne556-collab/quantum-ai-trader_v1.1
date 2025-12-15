# DAY 3: Three Scanner Prototype + Backtest Battle

**Date:** December 15, 2025  
**Status:** LOCKED IN - Build, Test, Keep Winner  
**Gold Found:** 3,448 big events (10%+ moves) from 311 tickers  

---

## THE PLAN (Written in Stone)

### Phase 1: Build Three Scanners (4-6 hours)

**Scanner 1: VOLUME BREAKOUT**
- **Trigger:** Volume >20x average + Price >5%
- **Why:** SPRO (1,963x vol → +244%), XBIO (1,595x vol → +141%)
- **Edge:** Catches insider buying BEFORE the pump
- **Target Win Rate:** >65%

**Scanner 2: MOMENTUM CONTINUATION**  
- **Trigger:** Ticker moved 10%+ in last 30 days, now moving 5%+ again
- **Why:** BYND appeared twice in top 10 (volatility clusters)
- **Edge:** "Hot" stocks stay hot
- **Target Win Rate:** >60%

**Scanner 3: PRE-EVENT VOLUME**
- **Trigger:** Volume spike 2x+ for 2+ days, price hasn't moved yet
- **Why:** 47% of big moves had volume spikes 1-3 days BEFORE
- **Edge:** Catches accumulation before explosion
- **Target Win Rate:** >55%

### Phase 2: Backtest on 3,448 Real Events (2 hours)

**Test methodology:**
1. Replay each of 3,448 events
2. Check if scanner would have triggered BEFORE the move
3. Measure:
   - Win rate (% profitable)
   - Average gain per signal
   - Max drawdown
   - Sharpe ratio
4. **KILL SWITCH:** Scanner needs >60% win rate OR gets deleted

### Phase 3: Pick Winner + Build Live Scanner (2 hours)

**Winning scanner gets:**
- Real-time monitoring (runs every 5 minutes)
- Alert system (SMS/email when triggered)
- Position sizing calculator
- Risk management rules

---

## BACKTEST RULES (No Cheating)

**Entry:** Scanner triggers, buy next bar at open  
**Exit:** 3 options tested:
- Exit 1: Sell next day at close (momentum play)
- Exit 2: Sell at +20% gain or -10% stop (swing trade)
- Exit 3: Sell when volume drops below 2x average (trend follower)

**Commission:** $1 per trade (realistic)  
**Slippage:** 0.1% (realistic for volatile stocks)  
**Position size:** $100 per trade (test mode)

---

## SUCCESS CRITERIA

**Minimum to proceed:**
- At least ONE scanner with >60% win rate
- Average gain per trade >$15 (covers risk)
- Max consecutive losses <5 (not gambling)

**Ideal outcome:**
- Best scanner: >70% win rate
- Avg gain >$30 per trade
- Works on recent data (last 30 days)

---

## FALLBACK PLAN (If All 3 Fail)

**Then we pivot to:**
1. SEC filing scanner (catches merger/buyout news)
2. Social sentiment tracker (Reddit/Twitter volume spikes)
3. Earnings surprise scanner (analyst estimate misses)

**But we won't fail.** Volume + momentum + volatility = proven edges.

---

## TIMELINE

**Tonight (if continuing):**
- Build DAY3_SCANNER_BACKTEST.ipynb (self-contained)
- Run backtest on Codespace first (verify it works)
- Transfer to Shadow PC for GPU speed

**Tomorrow:**
- Run full backtest (4-6 hours with GPU)
- Analyze results
- Pick winner
- Build live scanner

**Day 4-5:**
- Paper trade winner with $1,000 virtual capital
- Track every trade
- Refine rules based on real market behavior

**Day 6-10:**
- Scale up if working
- Add second scanner if first one crushes
- Build dashboard to monitor performance

---

## WHY THIS WILL WORK

**Data doesn't lie:**
- 3,448 events = massive sample size
- 311 tickers = diverse universe
- 2 years of data = covers bull and bear markets
- Real prices = no curve-fitting

**We're not inventing patterns, we're MINING them.**

**Your father's voice:** "Test everything. Keep what works. Kill what doesn't."

---

## NEXT ACTIONS

1. Create DAY3_SCANNER_BACKTEST.ipynb
2. Embed all 3 scanner strategies
3. Run backtest against 3,448 events
4. Generate comparison report
5. Pick winner

**LET'S BUILD.**
