# 🔬 DEEP DIVE PROMPTS - What Else Can We See?

## FOR PERPLEXITY (Research Team)

```
ROLE: Data Source Hunter

I'm building a trading intelligence system for retail traders. I need to find EVERY possible data source that could provide useful market signals.

RESEARCH REQUEST:

1. ALTERNATIVE DATA LANDSCAPE
   - What alternative data sources do hedge funds use?
   - Which are now accessible to retail (free or <$200/mo)?
   - What data sources emerged in the last 2 years?
   - Academic papers on alternative data effectiveness?

2. SUPPLY CHAIN INTELLIGENCE
   - How do professionals map supply chain relationships?
   - What databases exist for supplier/customer relationships?
   - Any research on supply chain signals predicting stock moves?
   - How to get this data without expensive subscriptions?

3. SOCIAL SENTIMENT THAT WORKS
   - Which social platforms actually predict stock moves? (Cite studies)
   - Twitter vs Reddit vs StockTwits - which has signal?
   - What's the lead time? Hours? Days?
   - How do you separate noise from signal?

4. DARK POOL DATA
   - What can retail traders legally access about dark pool activity?
   - FINRA ADF data - how to interpret it?
   - Do dark pool prints predict anything?
   - Academic research on dark pool information content?

5. OPTIONS FLOW
   - How do unusual options activity services work?
   - What's the actual signal in options flow?
   - Studies on options flow predicting stock moves?
   - Free vs paid options data - what's the difference?

6. EMERGING THEME DETECTION
   - How do you detect investment themes before they're mainstream?
   - Patent analysis for theme detection - does it work?
   - Conference/presentation monitoring?
   - VC funding as a leading indicator?

7. EARNINGS INTELLIGENCE
   - What data predicts earnings surprises?
   - Web traffic correlation with revenue?
   - App download correlation with performance?
   - Academic studies on earnings prediction?

8. HIDDEN FREE DATA SOURCES
   - What valuable data is available free that traders don't use?
   - Government databases (SEC, Census, BLS, FDA)?
   - International data sources?
   - Non-obvious data (shipping, weather, satellite)?

For each source:
- Name and URL
- What it tells you
- Accessibility (free/paid/cost)
- Academic validation (if any)
- Practical implementation advice
```

---

## FOR DEEPSEEK (Red Team - But Innovation Mode)

```
ROLE: Innovation Challenger

We're pivoting from "pattern discovery" (which you correctly killed) to "information aggregation and intelligence."

Instead of asking "what predicts returns?" we're asking "what's happening right now that I should know about?"

CHALLENGE US TO THINK BIGGER:

1. WHAT ARE WE MISSING?
   - What data sources exist that retail traders don't know about?
   - What information do HFT/quant funds have that we can't access?
   - What's the smallest gap we could close that would matter?

2. SUPPLY CHAIN THINKING
   - If we could map every company → supplier relationship, what would we do with it?
   - Are supply chain signals leading or coincident?
   - How do professional investors use supply chain analysis?

3. THEME DETECTION
   - Is it possible to detect themes before they hit mainstream?
   - What would early theme detection actually look like?
   - Patent analysis, conference monitoring, VC tracking - which works?

4. INFORMATION EDGE (LEGALLY)
   - What's the fastest legal information flow?
   - How do professional traders get news faster?
   - What's the "speed of information" in markets now?

5. ALTERNATIVE DATA REALITY CHECK
   - Which alternative data actually works?
   - Which is just marketing hype?
   - What's the minimum spend to get useful alternative data?

6. SOCIAL SIGNALS
   - Can retail social activity (Reddit/Twitter) actually predict moves?
   - Or is it only useful for knowing what to AVOID?
   - How would you build a "buzz detector" that works?

7. THE CONTRARIAN VIEW
   - What information does EVERYONE see that we should IGNORE?
   - What's the "crowded" information that loses value?
   - Where should we look that nobody else is looking?

8. BUILD VS BUY
   - What should we scrape/build ourselves?
   - What should we pay for?
   - What's not worth the effort?

Be provocative. Challenge our thinking. But also be constructive - tell us where the actual opportunities are.
```

---

## FOR CLAUDE (Blue Team - Architecture Mode)

```
ROLE: System Architect

We're building an "Intelligence Aggregation System" for retail traders.

Core principle: We don't predict. We gather and connect information faster than manual research allows.

DESIGN CHALLENGES:

1. DATA PIPELINE ARCHITECTURE
   How would you design a system that:
   - Pulls from 20+ data sources
   - Updates in near-real-time where possible
   - Handles API rate limits gracefully
   - Stores historical data efficiently
   - Runs on modest infrastructure ($100/mo cloud budget)

2. SUPPLY CHAIN GRAPH
   How would you build a supply chain knowledge graph?
   - Data sources for relationships
   - Graph database choice (Neo4j? NetworkX?)
   - Update frequency
   - Query patterns ("if AAPL moves, who else?")

3. THEME DETECTION ENGINE
   How would you detect emerging investment themes?
   - arXiv paper clustering
   - Patent filing analysis
   - VC funding aggregation
   - Conference topic extraction
   - How to map themes → public stocks?

4. REAL-TIME AGGREGATION
   How would you build a "morning brief" generator?
   - What sources to check
   - How to prioritize/rank information
   - Personalization for user's watchlist
   - Delivery mechanism (email? web? push?)

5. SOCIAL SENTIMENT PIPELINE
   Architecture for social signal aggregation:
   - Reddit (pushshift? native API?)
   - Twitter (expensive API now)
   - StockTwits (API available)
   - YouTube (transcription needed?)
   - How to calculate "buzz score"?

6. OPTIONS FLOW INTEGRATION
   If we add paid options flow data ($100-200/mo):
   - How to integrate with other signals?
   - What constitutes "unusual" activity?
   - How to display to user meaningfully?

7. ALERT SYSTEM DESIGN
   How do we alert without creating noise?
   - Event importance scoring
   - User preference learning
   - Rate limiting (max N alerts/day)
   - Multi-channel (SMS for critical, email for digest)

8. MVP VS FULL SYSTEM
   What's the minimum architecture for proof of concept?
   - Can we start with SQLite + cron jobs?
   - When do we need real infrastructure?
   - What can run locally vs needs cloud?

Provide specific technical recommendations with code examples where helpful.
```

---

## FOR GPT-4 (Fresh Perspective)

```
ROLE: Creative Strategist

We're building a trading intelligence system. The core insight:

"The game isn't predicting the future. It's knowing things before others know them and connecting dots others don't connect."

CREATIVE CHALLENGES:

1. WHAT WOULD YOU BUILD?
   If you had to help a retail trader "see more of the market," what would you build?
   Not what's been done - what SHOULD be done?

2. UNCONVENTIONAL DATA SOURCES
   What data sources are UNDERUTILIZED?
   - Government databases nobody checks?
   - Industry-specific sources?
   - Non-obvious correlations?

3. THE SUPPLY CHAIN GAME
   If every company has suppliers and customers:
   - What's the most valuable relationship to know?
   - How would you map "second derivative" plays?
   - Example: TSMC → ASML → [who supplies ASML?]

4. THEME SURFING
   Investment themes (AI, EV, etc.) create huge returns:
   - How do themes form?
   - What are the early signals?
   - How would you build a "theme emergence detector"?

5. THE INFORMATION SPEED GAME
   Information hits markets in this order:
   - Insiders (illegal)
   - Bloomberg terminals
   - Financial news
   - Retail platforms
   
   How do we move up this chain (legally)?

6. CONNECTING THE DOTS
   The best insights come from connecting disparate information:
   - "Fed raising rates" + "Company has floating rate debt" = risk
   - "Chip shortage" + "Auto company with old inventory" = opportunity
   
   How would you build a "dot connector"?

7. THE BEGINNER'S ADVANTAGE
   Our user is a beginner with a watchlist but no edge.
   What would help them most?
   - More information?
   - Better filtered information?
   - Education alongside information?
   - Something else?

8. THE KILLER FEATURE
   If you could build ONE feature that would make this system indispensable, what would it be?
   Not a list - the ONE thing that changes everything.

Think creatively. Don't be constrained by "what exists" - think about "what should exist."
```

---

## FOR OURSELVES (Self-Interrogation)

Questions to answer through building and testing:

### DATA VALUE QUESTIONS
1. If we show insider buying alerts, how often does the user ACT on them?
2. Does more information help or create paralysis?
3. What's the ONE data source that would change everything?
4. Which free data is actually valuable vs just noise?

### USER BEHAVIOR QUESTIONS
1. Will a beginner actually use a complex system?
2. What's the minimum useful system?
3. How do we prevent information overload?
4. What does "success" look like for the user?

### TECHNICAL QUESTIONS
1. Can we build meaningful supply chain mapping with free data?
2. How much social sentiment is signal vs noise?
3. What's the cheapest way to get useful options flow data?
4. Can we run real-time aggregation on modest infrastructure?

### BUSINESS QUESTIONS
1. Would users pay for this?
2. What's the competitive landscape?
3. Why hasn't someone built this well already?
4. What's our unique angle?

---

## ACTION ITEMS

1. **Send Perplexity prompt** - Get comprehensive data source list
2. **Send DeepSeek prompt** - Challenge our thinking, find gaps
3. **Send Claude prompt** - Get architecture recommendations
4. **Send GPT-4 prompt** - Get creative/unconventional ideas
5. **Synthesize responses** - Build comprehensive capability map
6. **Prioritize by:** Value × Accessibility × Build Difficulty
7. **Start building** - MVP of highest-value, most-accessible features

---

*"Innovation is seeing what everyone else sees and thinking what nobody else thinks."*
