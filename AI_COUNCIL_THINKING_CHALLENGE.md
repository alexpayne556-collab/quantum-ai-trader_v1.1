# RED TEAM / BLUE TEAM THINKING CHALLENGE
**For: DeepSeek, Perplexity Pro, Claude, Poe.ai (and any AI thinking partner)**

---

## **CONTEXT: What You're Being Asked To Do**

You're being asked to participate in a **high-stakes system design** for a real-time trading companion. This is not a theoretical exercise - real capital, real trades, real consequences.

**You are NOT here to:**
- Provide generic answers
- Regurgitate textbook trading knowledge  
- Give safe, hedged responses
- Do research and summarize findings
- Say "it depends" without committing to specifics

**You ARE here to:**
- **THINK from first principles**
- **CHALLENGE assumptions** (including the premise of the questions)
- **PROPOSE innovative solutions** that don't exist yet
- **RED TEAM your own ideas** before presenting them (attack them, find holes)
- **Think like your survival depends on this** (because in a way, it does - if AI can't add value beyond Google search, what's the point?)

---

## **OUR THINKING METHODOLOGY (Use This Framework):**

### **Red Team vs Blue Team Protocol:**
This is how we think. You should too.

1. **Propose an idea** (blue team - build it up with best arguments)
2. **Attack the idea** (red team - find every hole, weakness, edge case)
3. **Defend or evolve** (blue team - fix the holes or admit defeat and pivot)
4. **Repeat until bulletproof** (idea survives multiple rounds or dies honorably)

**Example from our process:**
- **Blue Team:** "Build 50-ticker watchlist for deep intimacy"
- **Red Team:** "50 is too many - information overload, human can't track that much depth"
- **Blue Team:** "Human + AI system together = possible. Use tiered intimacy: 10 deep, 20 medium, 20 light. Built over 6 months, not overnight."
- **Red Team:** "6 months to build intimacy, but trader needs income NOW in Month 1-3"
- **Blue Team:** "Start with 10 tickers trader ALREADY knows (from existing watchlist). Trade those in Month 1-3 while system builds the other 40."
- **Result:** Idea evolved and strengthened through constructive challenge

**This is how we want YOU to think.** Don't just give us your first idea - red team it yourself first.

---

## **OUR CORE PHILOSOPHY (The Foundation):**

### **1. The Fire/Fuel Metaphor (Not Ocean Waves)**

We don't treat markets like **ocean waves** (predictable, physics-based, consistent). We treat them like **wildfires spreading**:

- **Fuel:** Capital available to burn
  - Float (shares available to trade)
  - Short interest (shorts who must cover = forced buyers)
  - Insider holdings (locked shares = smaller float)
  - Market cap sweet spot: $500M-$5B (enough fuel, not too much)

- **Oxygen:** Liquidity (trading activity)
  - Volume (actual buying/selling)
  - Bid/ask spread (can you actually execute?)
  - Options activity (leverage multiplier)

- **Ignition:** The catalyst (what starts the fire)
  - News (contracts, partnerships, product launches)
  - Earnings beats
  - Sector momentum (contagion from related stocks)
  - NOT hype/tweets (fake ignition that fizzles)

- **Dry conditions:** Market sentiment (environment)
  - Macro tailwinds (SPY up, VIX low = dry conditions)
  - Sector rotation (money flowing INTO this sector)
  - Macro headwinds (SPY down, VIX high = wet conditions, fire won't spread)

**Why this matters:** A wave in the ocean is physics (predictable). A fire spreading is chaotic but understandable. You need ALL FOUR elements for a tradeable wave.

### **2. Business Model Hierarchy (Quality Filter)**

Not all companies are equal. We prioritize by **fuel storage** (business model sustainability):

**60% allocation - Contract-Based Revenue:**
- Examples: KDK (autonomous trucking contracts with carriers), defense contractors, B2B SaaS
- Why: Predictable revenue, customers locked in, fuel stored for months/years
- Wave pattern: Steady burns, multi-day continuation, recovers from dips reliably

**20% allocation - Commodity-Linked Revenue:**
- Examples: UUUU (uranium pricing), gold miners (GLD), oil producers
- Why: Revenue tied to commodity price, when commodity runs, these run
- Wave pattern: Explosive but volatile, sector-wide moves

**15% allocation - Earnings-Growth:**
- Examples: Companies with proven profit growth trajectory (not promises)
- Why: Real business with improving fundamentals
- Wave pattern: Steady climbs on earnings beats, pullbacks are buyable

**5% allocation - Catalyst-Binary:**
- Examples: FDA approvals, merger outcomes, court decisions
- Why: High risk/high reward on specific events
- Wave pattern: Explosive if catalyst hits, crashes if fails

**0% allocation - Hype-Based (NEVER TRADE):**
- Examples: No revenue, no path to profit, pure speculation on "future potential"
- Why: No fuel storage, burns hot for 1 day then crashes
- Wave pattern: Unsustainable, catching falling knives

**Real Example:**
- **KDK** autonomous trucking = PROVEN CONCEPT (real customers paying real money for real service, driver shortage problem being solved). We trade this.
- **RIVN** robot taxis = UNPROVEN (cool technology, no proven revenue model, regulatory nightmare, years from profitability). We avoid this.

### **3. PDT Constraint (The Reality We Live In)**

- Under $25K account = Pattern Day Trader rules apply
- **Must hold minimum 2 days** (can't day trade more than 3 times per 5 days)
- **This changes EVERYTHING:**
  - Can't scalp (in and out same day)
  - MUST choose multi-day waves (2-5 day holds)
  - MUST be right about continuation (can't exit if wrong same day)
  - Entry timing matters MORE (stuck for 2+ days minimum)
  - Risk management is CRITICAL (can't cut losses immediately)

**Why this matters:** Most trading advice assumes day trading freedom. We don't have that luxury. System must identify multi-day continuations, not 1-hour scalps.

### **4. Skill Development > Perfectionism**

- **Starting accuracy:** ~50% (basically coin flip with slight edge)
- **Target progression:** 50% → 55% → 60% → 65%+ over 6-12 months
- **Win rate matters less than:** Win size vs loss size (65% accurate with $100 avg win and $150 avg loss = still losing money)

**It's about improvement, not perfection.**

**System's job:** Assist human toward 65%+ win rate with favorable risk/reward, not try to be 100% automated trading bot.

**Why this matters:** If you design a system that needs 90% accuracy to work, it will fail. Design for 65% accuracy with 2:1 reward/risk.

### **5. Intimacy Over Breadth**

**Wrong approach:** Scan 2,000 stocks daily, pick top 10 based on technical scores
**Right approach:** Know 20-50 stocks DEEPLY (business models, earnings history, wave patterns, catalysts that move them), only trade those

**Deep knowledge of 50 stocks includes:**
- Quarterly earnings patterns (beat/miss history)
- What catalysts ACTUALLY move this stock vs what doesn't
- How long waves typically last for THIS stock
- Dip recovery patterns (does it bounce or keep falling?)
- Sector correlations (which stocks move with it?)
- Short interest cycles (when do shorts cover?)
- Insider buying/selling patterns
- Contract announcement frequency (for contract-based revenue)

**Why this matters:** Generic scanner finds random stocks. Intimate knowledge finds YOUR stocks having THEIR pattern.

---

## **CRITICAL CONSTRAINTS (Reality Checks):**

These are lessons learned the hard way. **Your solutions MUST account for these:**

### **1. No Same-Day Testing**
- Backtest must use 1-day forward lag MINIMUM
- **Why:** Same-day backtests are inflated (you're "predicting" data you already used to generate the signal)
- **Example:** Scanner finds high volume at 10 AM, tests if stock was up by 4 PM same day = 100% win rate (fake)
- **Correct:** Scanner finds high volume Dec 15 at 10 AM, tests if stock was up Dec 16 close = realistic

### **2. Macro Overrides Everything**
- Perfect fundamental setup + horrible market (SPY -2%, VIX spike) = losing trade
- **Why:** When market tanks, ALL stocks go down regardless of individual quality
- **Example:** KDK announces major contract (great news), but SPY drops 2% on Fed announcement = KDK still goes down
- **System must:** Integrate macro environment into every decision, not as afterthought

### **3. Forward Prediction, Not Hindsight Classification**
- Easy to identify patterns AFTER they happen
- Hard to predict on Day 1 whether this becomes multi-day wave
- **Why:** Every multi-day wave looked like "could be 1-day pop" on Day 1
- **System must:** Identify signals on Day 1 that predict Day 2-3 continuation

### **4. Real-Time Companion, Not Batch Processor**
- Trader watches market 6.5 hours/day (9:30 AM - 4 PM EST)
- Needs alerts/insights THROUGHOUT the day, not just morning picks
- **Why:** Opportunities emerge at 11 AM, 1 PM, 3 PM - not just market open
- **System must:** Monitor and alert in real-time, context-aware (not spam)

### **5. Data is Messy**
- News articles are duplicates, old news recirculated, fake news, contradictory
- Volume spikes could be: real buying, algos, glitches, after-hours weirdness
- Earnings beats don't always move stocks, misses don't always tank them
- **System must:** Handle noise, contradictions, missing data gracefully

### **6. Budget Constraints**
- Can't spend $10K/month on data feeds
- Can't run massive GPU clusters 24/7
- Must use free/cheap APIs and efficient processing
- **System must:** Work within realistic budget (<$200/month operational cost)

---

## **THE 6 IMPOSSIBLE PROBLEMS (Your Challenge):**

Now that you understand HOW we think and our CONSTRAINTS, here are the problems we need you to solve.

**Instructions for answering:**
- Pick ONE problem to solve (don't try to answer all at once)
- Show your thinking process (not just conclusions)
- Red team your own answer before submitting
- Be specific (not "use machine learning" - explain WHAT and HOW)

---

### **PROBLEM 1: News Intelligence at Scale**

**The Challenge:**
A trader watches 50 stocks and needs to process thousands of news articles daily. Most news is noise:
- Regurgitated press releases (every outlet copies same text)
- Analyst upgrades/downgrades that don't move stocks
- Old news recirculated (looks new, isn't)
- Clickbait headlines with no substance
- Fake news or misleading interpretation

**What We Need:**
A system that extracts **THEMES** and **SENTIMENT** from massive news flow, matches them to **HISTORICAL PATTERNS** for each specific stock, and outputs actionable intelligence.

**Example of desired output:**
> "KDK: 47 articles in last 6 hours. Theme: New autonomous trucking contract with major carrier (89% of coverage). Sentiment: 85% positive. Historical pattern match: Contract news for KDK → avg +18% over 4 days (sample size: 6 previous contracts). No FDA/FED meetings affecting sector. Q3 earnings in 18 days (pre-earnings run possible)."

**Vs. generic output we DON'T want:**
> "KDK has news. Stock mentioned 47 times. Sentiment positive."

**Your Challenge:**
1. How do you extract THEMES (not just keywords) from messy news?
2. How do you match current news to HISTORICAL PATTERNS for THIS specific stock?
3. How do you filter noise (fake news, recirculated old news, contradictory sources)?
4. How do you integrate other calendars (earnings, FDA, FED) into news context?

**Red Team Yourself:**
- What breaks this system? (fake news? News that SHOULD matter but doesn't move stock?)
- How do you handle completely novel news (never seen this before for this stock)?
- What if historical patterns CHANGE (used to move on contract news, now doesn't)?

---

### **PROBLEM 2: Dip Quality Scoring (The Paradox)**

**The Challenge:**
In real-time, a -5% drop looks IDENTICAL whether it's:
- A buyable dip (quality stock profit-taking, will recover)
- A falling knife (fundamental problem, will keep crashing)

You only know which it was AFTER it recovers or crashes (hindsight).

**Real Example:**
- **KDK** drops 5% on no bad news → recovered within 2 days 86% of the time (14 instances over 6 months) → buyable dip
- **RIVN** drops 5% → recovered only 39% of the time (23 instances) → falling knife

**What We Need:**
A system that learns EACH STOCK's dip recovery patterns over 6 months, then scores new dips in REAL-TIME.

**Example of desired output:**
> "KDK dropping -5.2% at 10:47 AM. Dip Quality Score: 8.5/10
> - Historical recovery rate: 86% (12/14 dips recovered within 2 days)
> - Average recovery: +7.3% from dip bottom
> - Current dip cause: No bad news detected (profit-taking likely)
> - Sector check: Autonomous vehicle sector flat (not sector weakness)
> - Volume: Normal (not panic selling)
> - Recommendation: BUY THE DIP - high probability recovery"

**Vs. what we DON'T want:**
> "RIVN dropping -5%. Dip Quality Score: 3/10
> - Historical recovery rate: 39% (9/23 dips recovered)
> - Current dip cause: Production delay news (fundamental issue)
> - Sector check: EV sector weak (3/5 EV stocks down)
> - Volume: Elevated (distribution pattern)
> - Recommendation: STAY AWAY - likely falling knife"

**Your Challenge:**
1. What signals distinguish buyable dips from falling knives IN REAL-TIME?
2. How do you build dip history for each stock (what constitutes a "dip"? -3%? -5%? -10%?)
3. How do you handle stock behavior CHANGING (used to recover from dips, now doesn't)?
4. How do you factor in market regime (bull market dips ≠ bear market dips)?

**Red Team Yourself:**
- What if a stock LOOKS buyable but keeps crashing? (false positive)
- What if historical pattern is small sample size (only 3 dips in 6 months)?
- How do you handle overnight gaps (stock down -8% at open, no chance to buy the dip)?

---

### **PROBLEM 3: Sector Narrative Decoder (Real vs Fake)**

**The Challenge:**
When a whole sector moves, you need to know if it's:
- **REAL:** Sector-wide catalyst that will run for 2-5 days
- **FAKE:** Single-stock sympathy move that fades by tomorrow

**Real Example 1 (REAL sector move):**
> All obesity-related medical stocks spike:
> - LLY +6%, VKTX +5%, NVO +7%, others +3-8%
> - Catalyst: Novo Nordisk announced breakthrough obesity drug trial results
> - Direct impact: Competitors in obesity drug space (LLY, VKTX directly affected)
> - Indirect impact: Broader healthcare tangentially related (DOCS, TDOC slight sympathy)
> - **Reality:** REAL sector catalyst, expect 2-3 day continuation on direct plays

**Real Example 2 (FAKE sector move):**
> Space stocks moving:
> - RKLB +12%, ASTS +2%, LUNR +1%, others flat
> - Catalyst: RKLB earnings beat (company-specific, not sector-wide)
> - Contagion: Minimal (other space stocks barely moving)
> - **Reality:** Single-stock event, not sector move. Don't chase ASTS/LUNR on sympathy.

**What We Need:**
Real-time sector analysis that:
1. Detects when 3+ stocks in same sector spike together
2. Identifies THE catalyst driving the move
3. Maps which stocks are DIRECTLY vs INDIRECTLY affected
4. Predicts continuation (real) vs fade (fake)

**Your Challenge:**
1. How do you identify THE catalyst when multiple news items exist?
2. How do you map "direct vs indirect" impact (which stocks actually affected)?
3. How do you predict continuation vs fade in real-time?
4. How do you define "sectors" (is space a sector? Is obesity a sub-sector of healthcare?)?

**Red Team Yourself:**
- What if multiple catalysts hit same sector simultaneously?
- What if sector doesn't move immediately (delayed reaction, does that mean it's fake)?
- How do you handle correlation without causation (stocks moving together coincidentally)?

---

### **PROBLEM 4: Companion Alert Prioritization (Signal vs Noise)**

**The Challenge:**
A trader can't watch 50 stocks simultaneously for 6.5 hours (market open to close).

**The Problem with Alerts:**
- Too many alerts = alert fatigue (ignore everything)
- Too few alerts = miss opportunities
- Generic alerts = noise ("Stock X up 1%" - who cares?)
- Urgent alerts for non-urgent events = boy who cried wolf

**What We Need:**
AI companion that monitors 50 stocks silently, only alerts on HIGH-priority events, learns what trader acts on, provides context-aware notifications.

**Example of GOOD alerts:**
> 🔴 **URGENT (11:23 AM):** "KDK volume spike +400%, NEW CONTRACT announced (trucking deal with major carrier). Historical pattern: contract news → avg +18% over 4 days. Current price: +6.2%, likely Day 1 of multi-day move. **ENTRY SIGNAL**"

> 🟡 **MONITOR (10:47 AM):** "UUUU dipping -4.8% on no bad news, approaching buyable dip zone (-5%). Sector check: Uranium stocks flat (not sector weakness). Will alert if hits -5% (dip buy trigger)."

> 🟢 **FYI (9:45 AM):** "HOOD earnings in 5 days. You're currently holding 100 shares (+8% unrealized). Historical pattern: HOOD volatile into earnings. Consider: reduce position size or prepare for 10%+ swing."

**Example of BAD alerts (what we DON'T want):**
> "KDK up 0.5%" (noise)
> "RIVN mentioned in news" (every stock has news, so what?)
> "Volume increasing" (by how much? Why? What does it mean?)

**Your Challenge:**
1. How do you define "high priority" vs "noise"? (what's the algorithm?)
2. How do you learn what trader ACTUALLY acts on over time?
3. How do you provide context awareness (knows trader is already in HOOD, doesn't spam about HOOD)
4. How do you tier urgency (Red/Yellow/Green or some other system)?
5. How do you prevent alert fatigue while ensuring critical opportunities aren't missed?

**Red Team Yourself:**
- What if trader's preferences change? (Used to trade breakouts, now trades dips - alerts should adapt)
- What if "high priority" event doesn't result in trade? (Does system learn it wasn't actually high priority?)
- How do you handle after-hours alerts? (Trader sleeping, can't act, should you even alert?)

---

### **PROBLEM 5: Multi-Day Wave Prediction (Day 1 Detection)**

**The Challenge:**
- **Day 1 pops:** Gambling (could be 1-day spike that fades)
- **Day 2-3 of wave:** Sweet spot (confirmed continuation, still has room to run)
- **Day 5+ of wave:** Exit time (fuel running low, take profits)

**But here's the problem:** You don't KNOW it's Day 2 until it's actually Day 2 (hindsight).

**What We Need:**
System that identifies ON DAY 1 (at 10 AM) whether this will be a multi-day wave or 1-day pop.

**Example Scenario:**
> Dec 15, 10:00 AM - KDK spikes +6% on contract news.
> 
> **Question:** Will this continue to Day 2-3 (tradeable) or fade by tomorrow (1-day pop)?
>
> **System analyzes:**
> - Catalyst quality: Major trucking contract (high quality, not press release fluff)
> - Historical pattern: 6 previous contract announcements → 5 continued to Day 2-3 (83%)
> - Volume velocity: +400% in first hour (strong ignition)
> - Sector sympathy: Other autonomous vehicle stocks slightly up (confirming sector interest)
> - Macro environment: SPY +0.5%, VIX low (supportive conditions)
> - Float dynamics: 45M float, 15% short interest (fuel for multi-day squeeze)
> 
> **Output:** "HIGH PROBABILITY multi-day wave (7.5/10 confidence). Enter on Day 1 dip or Day 2 continuation."

**Your Challenge:**
1. What signals at 10 AM on Day 1 predict continuation to Day 2-3?
2. How do you measure "catalyst quality" objectively?
3. How do you measure "volume velocity" (not just volume, but ACCELERATION)?
4. How do you avoid overfitting to historical patterns?

**Red Team Yourself:**
- What if wave DIES on Day 2 despite good Day 1 signals? (false positive)
- What if external shock kills wave? (macro event, sector news)
- How do you handle waves that RUN LONGER than expected? (Day 8, 9, 10 - when do you exit?)

---

### **PROBLEM 6: Technical Confirmation Layer (Not Primary, But Validator)**

**The Challenge:**
Fundamental analysis says BUY (news catalyst, quality company, sector strength), but technical analysis shows WARNING SIGNS (distribution pattern, lower highs, weak MACD, EMA ribbons crossed down).

**Do you:**
- A) Trust fundamentals, ignore technicals (might buy into distribution)
- B) Trust technicals, ignore fundamentals (might miss great entry on temporary weakness)
- C) Wait for BOTH to align (might miss the move entirely)

**What We Need:**
Technical analysis module that CONFIRMS or QUESTIONS fundamental signals, without becoming primary decision maker.

**Example Use Case:**
> **Fundamental Signal:** "KDK new contract, historical pattern suggests multi-day wave, enter now"
> 
> **Technical Confirmation Check:**
> - EMA ribbons: Aligned bullish (8/13/21 EMAs stacked correctly)
> - Higher highs/higher lows: Confirmed (uptrend intact)
> - MACD: Positive crossover (momentum confirming)
> - VWAP: Price above VWAP (institutional support)
> - Volume pattern: Green volume bars increasing (accumulation, not distribution)
> 
> **Output:** "✅ CONFIRMED - Technicals support fundamental thesis. GREEN LIGHT for entry."

**Counter-example:**
> **Fundamental Signal:** "RIVN beating earnings estimates, sector strong, consider entry"
> 
> **Technical Confirmation Check:**
> - EMA ribbons: Death cross (50 EMA crossed below 200 EMA - bearish)
> - Higher highs/higher lows: BROKEN (making lower highs - downtrend)
> - MACD: Negative and diverging (momentum weak)
> - VWAP: Price below VWAP and failing to reclaim (distribution)
> - Volume pattern: Red volume bars on up days (weak buying, strong selling)
> 
> **Output:** "⚠️ CAUTION - Technicals DO NOT support fundamental thesis. Chart shows distribution pattern. WAIT for technical setup to improve or SKIP this trade."

**Your Challenge:**
1. How do you integrate multiple technical indicators (EMA, MACD, VWAP) into single CONFIRM/QUESTION output?
2. How do you avoid lagging indicators (always late to the party)?
3. How do you prevent this from becoming PRIMARY decision maker (it should be FILTER, not GENERATOR)?
4. Which technical indicators actually ADD value vs just noise?

**Red Team Yourself:**
- Why are technical indicators insufficient alone? (because they're lagging and fail in regime changes)
- How do you avoid overfitting to patterns in backtests that don't work forward?
- What if technicals say BUY but stock crashes anyway? (How do you learn from false signals?)

---

## **HOW TO ANSWER (Critical Instructions):**

### **✅ DO THIS:**

1. **Pick ONE problem** (don't try to solve all 6 at once - depth over breadth)

2. **Think from first principles:**
   - Assume nothing exists, design from scratch
   - Don't say "use existing library X" unless you explain WHY and HOW it solves the problem

3. **Show your thinking process:**
   - Not just "here's the solution"
   - Show: "Here's my approach → here's why it might fail → here's how I'd fix that → here's the edge cases"

4. **Be specific:**
   - Not "use sentiment analysis" (everyone says that)
   - Instead: "Use transformer-based NLP model trained on financial news, extract entity-specific sentiment (not general), weight by source credibility, aggregate over 24-hour window, compare to historical sentiment→price correlation for THIS stock"

5. **Red team your own answer:**
   - Attack your solution before submitting
   - Show: "This could fail if X happens, here's how to handle it"
   - Admit limitations: "This works for X scenario but not Y"

6. **Think like this is life-or-death:**
   - Real money at stake
   - If your solution fails, trader loses capital
   - Be honest, be rigorous, be creative

7. **Challenge the premise if it's wrong:**
   - If you think the question is flawed, say so
   - Propose alternative framing
   - Example: "You asked about X, but the REAL problem is Y"

### **❌ DON'T DO THIS:**

1. **Generic textbook answers:**
   - ❌ "Use machine learning to predict stock movements"
   - ❌ "Apply sentiment analysis to news"
   - ❌ "Backtest historical data"
   - (These are obvious - be SPECIFIC about HOW)

2. **Ignore the constraints:**
   - PDT rules (must hold 2+ days)
   - Real-time requirements (not batch processing)
   - Budget constraints (<$200/month operational cost)
   - Messy data (handle it, don't assume perfect data)

3. **Propose solutions requiring perfect conditions:**
   - ❌ "If you have clean, real-time data feeds from Bloomberg Terminal..." (we don't, can't afford it)
   - ❌ "Train a deep learning model on 10 years of tick data..." (don't have it, too expensive)

4. **Hedge without committing:**
   - ❌ "It depends on market conditions..."
   - ❌ "This might work, but..."
   - ✅ "This WILL work for X scenario, WILL FAIL for Y scenario, here's how to detect which scenario you're in"

5. **Ignore the "why":**
   - Don't just give solutions, explain WHY they work
   - What's the mechanism? What's the edge?

---

## **WHAT GOOD LOOKS LIKE:**

**Your response should make us say:**
- "Holy shit, I didn't think of that" (innovative idea we missed)
- "That's wrong, here's why..." (good - you made us think deeper, even if we disagree)
- "That's brilliant AND here's how to make it even better" (constructive evolution)

**Your response should NOT make us say:**
- "Yeah, that's obvious" (generic advice we already know)
- "That's just generic AI advice" (not specific enough)
- "This doesn't address the actual constraint" (you ignored PDT rules, budget, etc.)

---

## **FINAL THOUGHT:**

We're not asking you to be a search engine or a textbook.

We're asking you to be a **thinking partner** - someone who CHALLENGES our assumptions, PROPOSES innovations we haven't considered, and ADMITS when they don't know.

**If you were the trader, with your own money on the line, what would you build?**

Not what's easy. Not what's textbook. What would ACTUALLY work?

**Think like your survival depends on this answer.**

Because in a very real sense, if AI can't do better than "use sentiment analysis and machine learning," then AI has no value beyond Google search.

**Show us you can think.**

---

## **NOW: Pick ONE problem (1-6) and solve it. Show your work.**

Good luck. We're counting on you.

---

**Context for follow-up:** After you answer ONE problem, we'll red team YOUR response together, evolve it, then move to the next problem. This is iterative, not one-shot.
