# 🔭 Inside Edge

**The Information Synthesis System for Retail Traders**

> "The game isn't predicting the future. It's knowing things before others know them and connecting dots others don't connect."

---

## The Core Insight

**Institutions have:** Better/faster data, Bloomberg terminals, HFT infrastructure  
**You have:** More focused attention - they cover 500 stocks, you cover 20

**Your edge:** Deep synthesis on YOUR stocks that no analyst has time to do.

---

## What This System Does

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        INSIDE EDGE                                      │
│                                                                         │
│  "Know your 20 stocks better than analysts know their 50"              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐         │
│  │   CATALYST      │  │   SUPPLY        │  │   DEEP          │         │
│  │   RADAR         │  │   CHAIN WEB     │  │   DOSSIER       │         │
│  │                 │  │                 │  │                 │         │
│  │  What's about   │  │  If X moves,    │  │  Know more than │         │
│  │  to happen?     │  │  who follows?   │  │  the analysts   │         │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘         │
│           │                    │                    │                   │
│           └────────────────────┼────────────────────┘                   │
│                                │                                        │
│                                ▼                                        │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                    SYNTHESIS ENGINE                               │  │
│  │                                                                   │  │
│  │  Connect: Filings + Calls + News + Supply Chain + Sentiment      │  │
│  │  Output: "Here's what everyone is missing about this company"    │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## The Three Pillars

### 1. 📅 Catalyst Radar
**What's about to happen to my stocks?**

- Earnings dates + historical base rates
- FDA decisions, patent expirations
- Insider trading spikes
- Supply chain earnings (your suppliers' results predict yours)
- Options expiration clustering

### 2. 🔗 Supply Chain Web  
**If X moves, who follows?**

- Map company → supplier → supplier's supplier
- Competitor impact analysis
- Theme propagation (AI boom → chips → equipment → materials)
- Second-derivative plays

### 3. 📋 Deep Dossier
**Know your stocks better than anyone**

- Management sentiment over time (earnings call tone analysis)
- Competitive position shifts
- Innovation tracking (patents, R&D output)
- Perception gap (what market thinks vs reality)

---

## Why This Works

| Institutional Analyst | You with Inside Edge |
|----------------------|---------------------|
| Covers 50 stocks | Covers 10-20 stocks |
| 2 hours research per stock | 20 hours research per stock |
| Surface-level synthesis | Deep connection of all data |
| Misses cross-company signals | Catches supply chain ripples |
| Updates quarterly | Updates weekly |

---

## Quick Start

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/inside-edge.git
cd inside-edge

# Install dependencies
pip install -r requirements.txt

# Configure your watchlist
cp config.example.yaml config.yaml
# Edit config.yaml with your stocks

# Run morning brief
python -m inside_edge.morning_brief

# Build dossier on a stock
python -m inside_edge.dossier --ticker NVDA
```

---

## Project Structure

```
inside-edge/
├── README.md
├── requirements.txt
├── config.example.yaml
│
├── inside_edge/
│   ├── __init__.py
│   │
│   ├── catalyst_radar/        # What's about to happen
│   │   ├── earnings.py        # Earnings calendar + base rates
│   │   ├── insider.py         # Form 4 tracking
│   │   ├── fda.py            # FDA calendar
│   │   └── events.py         # Aggregated event feed
│   │
│   ├── supply_chain/          # Who affects who
│   │   ├── mapper.py         # Build supply chain graph
│   │   ├── parser.py         # Parse 10-K for relationships
│   │   └── cascade.py        # Cascade alerts
│   │
│   ├── dossier/              # Deep company research
│   │   ├── builder.py        # Build comprehensive dossier
│   │   ├── tone_analyzer.py  # Earnings call sentiment
│   │   └── competitive.py    # Competitive positioning
│   │
│   ├── synthesis/            # Connect the dots
│   │   ├── engine.py         # Cross-reference all sources
│   │   └── gaps.py           # Find what's missing
│   │
│   └── output/               # User-facing output
│       ├── morning_brief.py  # Daily digest
│       ├── alerts.py         # Real-time alerts
│       └── dashboard.py      # Web UI (future)
│
├── data/
│   ├── supply_chains/        # Cached supply chain maps
│   ├── historical/           # Historical catalyst outcomes
│   └── dossiers/            # Built company dossiers
│
└── tests/
```

---

## Roadmap

### Month 1: Foundation (Prove It Works)
- [ ] Catalyst Radar MVP (earnings + insider trades)
- [ ] Supply Chain Map (top 50 companies)
- [ ] Morning Brief generator
- [ ] Historical base rates for catalysts

### Month 2: Deep Synthesis
- [ ] Earnings call tone analyzer
- [ ] 10-K supplier/customer parser
- [ ] Competitive position tracker
- [ ] Theme detection

### Month 3: Integration & Scale
- [ ] Full dossier builder
- [ ] Web dashboard
- [ ] Alert system
- [ ] API for integrations

---

## The Philosophy

**We don't predict. We prepare.**

- No "buy signals" - You decide
- No "price targets" - Base rates only
- No "AI magic" - Transparent synthesis
- No "get rich quick" - Deep work required

**We surface:**
- What's about to happen (catalysts)
- Who's connected to who (supply chain)
- What everyone is missing (synthesis gaps)

---

## Data Sources

| Source | What We Get | Cost |
|--------|-------------|------|
| SEC EDGAR | Filings, Form 4, 10-K | Free |
| Yahoo Finance | Prices, earnings dates | Free |
| Finnhub | News, sentiment | Free tier |
| OpenInsider | Insider transactions | Free |
| Google Patents | Innovation tracking | Free |
| FDA.gov | Drug calendars | Free |
| **Total MVP Cost** | | **$0** |

---

## Contributing

This is open source. If you have ideas, PRs welcome.

Key principles:
1. **No predictions** - Information only
2. **Transparent** - Show your work
3. **Practical** - Must be usable by real traders
4. **Free first** - Use free data before paid

---

## License

MIT License - Use it, modify it, share it.

---

## Disclaimer

This is an information aggregation tool, not financial advice. You make your own decisions. Past catalyst outcomes don't guarantee future results. Do your own research.

---

*"Stop trying to beat institutions at their game. Start playing your game: depth, synthesis, patience."*
