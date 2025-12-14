# 🧠 Multi-AI Brainstorm Session: Inside Edge System Design

**Date:** December 14, 2025
**Goal:** Design a small-cap stock intelligence system that gives YOU an edge
**Participants:** DeepSeek, Perplexity Pro, GPT-4, Claude Opus + Your Strategic Input

---

## 📊 YOUR DATA SOURCES (What We Actually Have)

### Market Data (Free Tier)
| Source | Tier | Limits | Best For |
|--------|------|--------|----------|
| Alpha Vantage | Free | 25 calls/day, 5/min | OHLCV, technicals, news |
| Finnhub | Free | 60 calls/min | Real-time quotes, company data |
| Polygon.io | Free | 5 calls/min | Aggregated data |
| Twelve Data | Free | 800 calls/day | Historical data, intraday |
| Financial Modeling Prep | Free | 250 calls/day | Financial statements, ratios |
| EOD Historical | Free | 20 calls/day | Historical closing data |
| FRED | Free | Unlimited | Macro indicators (VIX, yields, etc) |

### AI APIs (You Have Access To)
- Perplexity API (web search + synthesis)
- OpenAI GPT-4 (reasoning, analysis)
- Your manual access to: Claude Opus, DeepSeek (via web)

### What We DON'T Have
- Real-time insider trades (OpenInsider is free but manual)
- Institutional flow data
- Earnings call transcripts (can scrape manually)
- SEC EDGAR (free but slow)
- Supply chain data (must synthesize from news)

---

## 🎯 The Core Problem We're Solving

**NOT:** "Predict returns from patterns"
**YES:** "Know what's happening to 30 stocks better than Wall Street"

### Key Question
> For a small-cap tech/energy/biotech stock:
> - What events are ABOUT to happen?
> - Who depends on this company?
> - What are analyst blind spots?
> - What can we know TODAY that moves prices TOMORROW?

---

## 💡 BRAINSTORM PROMPTS (Send to Each AI)

### PROMPT 1: For DeepSeek (Red Team / Reality Check)
```
You are a skeptical hedge fund PM reviewing our system design.

CONTEXT:
- Single person, retail access, free data tiers
- Focus: 30 small/mid-cap stocks
- Goal: Information synthesis edge (not prediction)
- Budget: $0 for data

QUESTION:
What is the SINGLE MOST RELIABLE SIGNAL we can extract from free data sources 
that institutional traders miss?

Evaluate these potential signals:
1. Earnings surprise probability (using analyst estimates vs historical volatility)
2. Supply chain disruption cascades (mapping supplier/customer relationships)
3. Management tone shifts (analyzing press release language over time)
4. Sector rotation timing (using relative strength, not absolute returns)
5. Institutional accumulation patterns (from broker research summaries)
6. Catalyst clustering (multiple events happening to same company)

For each, provide:
- Realistic win rate on small caps
- How to implement with FREE data
- What could go wrong
- Is it worth our time?

Be harsh. Tell us what won't work.
```

### PROMPT 2: For Perplexity Pro (Information Synthesis)
```
Research and synthesize: How do successful small-cap investors actually find edges?

SPECIFIC QUESTIONS:
1. What data sources do they use that retail CAN access?
   - SEC EDGAR patterns
   - Press releases and news timing
   - Earnings surprise patterns
   - Supply chain news

2. Which "hidden" free data sources exist?
   - Google Patents (tech companies R&D pipeline)
   - Indeed/LinkedIn job postings (hiring/growth signals)
   - FDA calendar (biotech catalysts)
   - Regulatory filings
   - Twitter/Reddit sentiment shifts

3. What "synthesis" techniques beat pure data collection?
   - Connecting dots across sources
   - Timing relative to events
   - Contrarian positioning

Return a list of 8-10 specific data sources we can access for FREE
with implementation difficulty (Easy/Medium/Hard) and lead time (days/weeks/months)
```

### PROMPT 3: For GPT-4 (System Architecture)
```
Design a system architecture for "Inside Edge" - a small-cap intelligence platform.

CONSTRAINTS:
- Single person to operate it
- Free data only (no subscriptions)
- Must work with 30 stocks
- Output: Daily/weekly intelligence brief
- NOT: Buy/sell signals (just context)

REQUIREMENTS:
1. Catalyst Radar: What's about to happen to these 30 stocks?
   - Data needed
   - Update frequency
   - Implementation complexity

2. Supply Chain Map: Who depends on these companies?
   - Data source
   - How to maintain
   - Update cadence

3. Deep Dossier: What do we know that others don't?
   - Quarterly earnings analysis
   - Management team changes
   - Competitive wins/losses
   - Customer concentration

4. Sector Rotation: Which sectors are heating up?
   - Relative strength calculation
   - Leading/lagging indicators
   - Implementation

Provide pseudocode for the core pipeline and data flows.
```

### PROMPT 4: For Claude Opus (Strategic Synthesis)
```
You are a strategic advisor helping someone design an edge in small-cap investing.

PHILOSOPHY:
"Stop trying to beat institutions at their game (speed, data).
Start playing YOUR game: depth, synthesis, patience."

QUESTIONS:
1. What are the 5 MOST IMPORTANT things to know about a small-cap 
   to make good decisions?
   
2. Which of these can be known BEFORE the market knows them?
   
3. What is the optimal frequency to update information?
   - Daily? Weekly? Quarterly?
   - For which variables?
   
4. How do we avoid information overload?
   - What to ignore
   - What to track obsessively
   - What changes the narrative
   
5. What is the ONE METRIC that best indicates you're ahead of the market
   on a small-cap stock?

Provide a strategic framework, not a technical spec.
```

---

## 📋 WHAT WE NEED BACK FROM EACH AI

### From DeepSeek (Red Team)
- [ ] Top 3 most reliable free-data signals
- [ ] Win rate estimate (be realistic)
- [ ] Why hedge funds don't use these (hint: they're not sexy)

### From Perplexity (Research)
- [ ] 10 specific free data sources with links
- [ ] Difficulty rating + time investment for each
- [ ] Which ones have "lead time" (tell us about future moves)

### From GPT-4 (Architecture)
- [ ] System diagram / data flow
- [ ] Pseudocode for catalyst radar
- [ ] Pseudocode for supply chain map
- [ ] Technical implementation roadmap (Phase 1, 2, 3)

### From Claude Opus (Strategy)
- [ ] Top 5 knowledge priorities
- [ ] Update frequency for each
- [ ] One-page "narrative framework" (how to think about stocks)
- [ ] Red flags that change your thesis

---

## 🛠️ THEN: Stock Selection (After Brainstorm)

Once we have the brainstorm outputs, we'll:

1. **Use the brainstorm data** to understand what signals work
2. **Score all small-caps** against those signals
3. **Pick YOUR 15** - sectors/companies you care about
4. **AI finds 15 more** - using the scoring system
5. **Lock 30-stock watchlist** - ready for implementation

---

## 📝 YOUR ASSIGNMENT (Before Brainstorm)

1. **Read this framework** - understand the goal
2. **Pick 3-5 sectors** you want to focus on:
   - Nuclear / Energy infrastructure
   - AI chips / semiconductors
   - Biotech with FDA catalysts
   - Autonomous vehicles
   - Data centers
   - (Your 3-5 picks)

3. **List 10 companies** you know/care about:
   - Doesn't have to be perfect
   - Just things that interest you
   - Size: $500M - $20B market cap

4. **Identify 3 pain points** with your current approach:
   - "I don't know when earnings are"
   - "I can't track supply chain"
   - "Too much noise, hard to focus"
   - etc.

---

## 🎬 EXECUTION PLAN

### Today (Dec 14)
- [ ] You complete your assignment above
- [ ] Run brainstorm prompts with each AI
- [ ] Compile outputs into a decision document

### Tomorrow (Dec 15)
- [ ] Analyze brainstorm outputs
- [ ] Create preliminary system architecture
- [ ] Build stock discovery/scoring system

### Next 2 Days (Dec 16-17)
- [ ] Pick 30 stocks
- [ ] Build Catalyst Radar
- [ ] Test with real data
- [ ] Generate first intelligence brief

### Week 2 (Dec 21+)
- [ ] Build Supply Chain Map
- [ ] Deep Dossier automation
- [ ] Sector rotation tracking
- [ ] Daily/weekly brief loop

---

## 💬 KEY INSIGHT

> "The edge isn't in having more data. It's in knowing what matters.
> A 30-stock portfolio you understand deeply beats a 500-stock portfolio you don't."

**Small caps are where the game gets played.** Institutions don't have time to know them well. YOU do.

---

**Next Step:** Complete your assignment and report back. Then we run the brainstorm.
