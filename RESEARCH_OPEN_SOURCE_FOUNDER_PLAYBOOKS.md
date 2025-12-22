# OPEN SOURCE FOUNDER PLAYBOOKS & RESOURCES
## Research Material - December 23, 2025

**Status:** RESEARCH ONLY - Do not build yet
**Purpose:** Understanding how successful AI founders shared their work
**Contains:** Verified GitHub repos, founder journeys, architecture patterns

---

## VERIFIED OPEN SOURCE RESOURCES

### 1. GPT Engineer by Anton Osika
- **GitHub:** https://github.com/AntonOsika/gpt-engineer
- **Status:** FULLY OPEN SOURCE (MIT License)
- **Stars:** 500K+
- **What it is:** The exact code behind Lovable ($100M+ valuation)

**Key Points:**
- Complete CLI platform for AI code generation
- Real production code anyone can clone and run
- Integration with local models (CodeLlama, Mixtral) - NO API costs
- Can use, modify, sell derivatives

**Commands:**
```bash
pip install gpt-engineer
gpte <project_dir> <model_name>
git clone https://github.com/gpt-engineer-org/gpt-engineer.git
```

---

### 2. AI Cookbook by Dave Ebbelaar
- **GitHub:** https://github.com/daveebbelaar/ai-cookbook
- **Status:** OPEN SOURCE - Copy/paste ready

**What's Inside:**
- RAG (Retrieval-Augmented Generation) examples
- AI agents that take actions
- Claude API integration patterns
- Prompt engineering best practices
- Working code snippets

---

### 3. Building in Public Framework
- **GitHub:** https://github.com/buildinginpublic/buildinpublic
- **Status:** OPEN SOURCE - Roadmap template

**What it is:**
- Complete guide for solo founders
- How to structure progress for public consumption
- Communication strategies
- Templates for updates, milestones

---

### 4. Daniel Nguyen - BoltAI Journey ($0 → $20K/month)
- **Product:** BoltAI (closed source commercial)
- **Journey:** FULLY DOCUMENTED publicly
- **Twitter:** @daniel_nguyenx
- **GitHub:** https://github.com/longseespace (97 public repos)

**The Documented Journey:**
- April 2023: First launch
- Month 1: $1,162 revenue
- Current: $20K+/month (verified)
- Full interview on Founderoo

**What He Shared:**
- How to find first 10 customers
- Tech stack: Swift (Mac native) + Node.js
- Pricing that worked: $20/month freemium
- Growth channels: Twitter, Reddit, Product Hunt
- Built second product (PDF Pals) for same audience

---

## STOCK PREDICTION DASHBOARD ARCHITECTURE (From Research)

**This is a reference architecture, NOT to build now**

### System Overview
```
DASH FRONTEND (UI)
       ↓
FastAPI REST API
       ↓
BACKEND ENGINES:
├─ DATA ENGINE (fetch prices, store in DB)
├─ CATALYST ENGINE (bill tracking, FDA calendar)
├─ INDICATOR ENGINE (leading indicators, alerts)
├─ COMPANY ENGINE (financials, ranking)
├─ VALUATION ENGINE (3-scenario model)
├─ TRADE ENGINE (buy/sell rules)
├─ PERFORMANCE ENGINE (win rate, stats)
└─ ALERT ENGINE (email, webhooks)
       ↓
PostgreSQL DATABASE
```

### Database Tables (Reference)
- catalysts (id, name, bill_url, status, timeline, amount)
- companies (id, ticker, name, sector, cost_per_unit, moat)
- positions (id, ticker, entry_date, entry_price, shares)
- trades (id, ticker, type, date, price, return%, status)
- indicators (id, catalyst_id, name, status, last_check)
- price_history (id, ticker, date, price, volume)
- alerts (id, type, condition, status, triggered_at)

### Tech Stack (Reference)
- Backend: FastAPI + SQLAlchemy + PostgreSQL
- Frontend: Dash (Python) + Plotly
- Data: yfinance, Congress.gov API, FDA.gov API
- Scheduling: APScheduler for background jobs
- Deployment: Docker Compose

---

## KEY INSIGHTS FROM FOUNDER STORIES

### What They Have in Common:
1. **Started with personal problem** - Built for themselves first
2. **Shipped fast** - MVP in 4 weeks, not 4 months
3. **Built in public** - Shared progress weekly
4. **Charged from day 1** - Even $5-$20/month validates demand
5. **Listened obsessively** - Customer feedback drove features
6. **Simple tech stack** - Node.js, Python, PostgreSQL (not Kubernetes)

### Daniel Nguyen's Exact Playbook:
```
Step 1: Personal Problem
└─ Switching between ChatGPT and Claude frustrated him

Step 2: MVP (4 weeks)
├─ Swift for Mac native app
├─ Node.js backend
└─ Enough to use himself

Step 3: Share on Twitter
├─ Posted video demo
├─ Got 100 beta testers in first day
└─ Shipped improvements every week

Step 4: Growth Channels
├─ Twitter (organic)
├─ Reddit (r/MacApps)
├─ Indie Hackers
├─ Sponsored AI newsletters
└─ Paid ads (later)

Step 5: Second Product
├─ Same audience
├─ PDF Pals = $6K/month in 4 months
└─ Total: $20K/month with 2 products
```

---

## COMMUNITIES (Free)

1. **Indie Hackers** - https://www.indiehackers.com
2. **#buildinpublic on Twitter/X** - Real-time founder updates
3. **Product Hunt** - https://www.producthunt.com
4. **Hacker News** - https://news.ycombinator.com
5. **Reddit:** r/SideProject, r/entrepreneur, r/MacApps

---

## RELEVANCE TO OUR PROJECT

**Why this research matters:**
- Shows how others documented their AI product journey
- Open source patterns we can study
- Architecture examples (8-engine system is similar to our research)
- Validation approaches (real users, real feedback)
- Growth strategies that worked

**What to extract later:**
- Code patterns from GPT Engineer
- API integration patterns from AI Cookbook
- Building in public template for our journey
- Dashboard architecture concepts

**NOT for now:**
- We're still in research phase
- Understanding, not implementing
- Days/weeks of research ahead

---

**Saved:** December 23, 2025
**Status:** Research material collected
**Next:** Continue understanding what we have
