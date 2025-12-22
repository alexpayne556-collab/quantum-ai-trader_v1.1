# AI SESSION START FILE - DECEMBER 2025
## Complete Context for Every New Session

**Last Updated**: December 22, 2025
**Purpose**: READ THIS FIRST every session. Contains everything needed to continue work.

---

# 🚨 CRITICAL CONTEXT (READ FIRST)

## The Mission
- **Personal tribute** to MIT Lincoln Labs
- **Continue father's teaching** - systematic thinking, rigorous validation
- **NO RUSHED SYSTEM** - research first, build later
- **Adaptive > Static** - what works today won't work tomorrow

## Core Principles (MEMORIZE THESE)
1. RESEARCH FIRST, BUILD LATER
2. ADAPTIVE > STATIC
3. PROACTIVE > REACTIVE (know what's going to pop, not what did pop)
4. SECTOR ROTATION > STOCK PICKING
5. SIMPLE > COMPLEX (but ticker-specific when needed)
6. CURRENT MARKET > HISTORICAL

## User Profile
- Has Shadow PC with GPU (RTX 2000 Ada, 6839 GFLOPS)
- Conda environment: `quant2026`
- Paper trading with Alpaca ($100K)
- PDT constraint aware (<$25K = 3 day trades per 5 days)
- Timeline: 6 months for institutional-grade system

---

# 📊 CURRENT STATE (December 22, 2025)

## What's Built & Working
| File | Status | Purpose |
|------|--------|---------|
| `PRODUCTION_SYSTEM_2026.py` | ✅ TESTED | Validation framework with all 6 fixes |
| `MASTER_RESEARCH_SYNTHESIS_2026.md` | ✅ COMPLETE | All research consolidated |
| `GPU_BENCHMARK.py` | ✅ WORKS | GPU verification |
| `EXPLORE_WATCHLIST.py` | ✅ WORKS | Data exploration |
| `VALIDATION_FRAMEWORK_2026.py` | ✅ WORKS | Basic validation |

## What's NOT Built Yet
| Component | Priority | Blocking? |
|-----------|----------|-----------|
| Catalyst Tracker | 🔥 HIGH | Yes - no domain signals |
| Ticker-Aware Model | 🔥 HIGH | Yes - no transfer learning |
| Online Learning | MEDIUM | No |
| Live Alpaca Integration | LOW | No - need validation first |

## Shadow PC Commands (QUICK REFERENCE)
```bash
# Activate environment
conda activate quant2026

# Run validation
python PRODUCTION_SYSTEM_2026.py

# Check GPU
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

---

# 🎯 THE ARCHITECTURE WE'RE BUILDING

## Ticker-Aware Transformer (from Perplexity)

```
INPUT LAYER:
├─ Price features (OHLCV) ──────────────────┐
├─ Technical indicators ────────────────────┤
├─ Ticker embedding (RKLB=0, ASTS=1, etc.) ─┼──> SHARED LSTM
└─ Catalyst signals (ticker-specific) ──────┘
                                             │
                                             ▼
                                    ┌────────────────┐
                                    │ ATTENTION      │
                                    │ (what matters?)│
                                    └────────┬───────┘
                                             │
         ┌───────────────────────────────────┼───────────────────────────────────┐
         ▼                                   ▼                                   ▼
    ┌─────────┐                        ┌─────────┐                        ┌─────────┐
    │ RKLB    │                        │ ASTS    │                        │ IONQ    │
    │ Adapter │                        │ Adapter │                        │ Adapter │
    └────┬────┘                        └────┬────┘                        └────┬────┘
         │                                  │                                  │
         ▼                                  ▼                                  ▼
    ┌─────────┐                        ┌─────────┐                        ┌─────────┐
    │ RKLB    │                        │ ASTS    │                        │ IONQ    │
    │ Signal  │                        │ Signal  │                        │ Signal  │
    └─────────┘                        └─────────┘                        └─────────┘

KEY INSIGHT: Shared learning from ALL tickers (1500+ days)
             + Ticker-specific adaptation (learns what moves each)
```

## Why This Works Better
| Problem | Old Approach | New Approach |
|---------|--------------|--------------|
| "Only 500 days of RKLB" | Train on 500 days | Share learning from 1500 days |
| "Different catalysts matter" | One model, hope it works | Per-ticker adapters learn automatically |
| "What actually moved price?" | Black box | Attention weights show which input mattered |

---

# 📋 TICKER-SPECIFIC CATALYSTS

## RKLB (Rocket Lab)
```
CATALYSTS TO TRACK:
├─ Launch schedule (from rocketlabusa.com, spacenews.com)
├─ Launch success/failure (+15-30% on success, -20% on failure)
├─ Government contracts (SAM.gov, DoD press releases)
├─ Neutron milestones (static fire, first flight)
├─ Quarterly earnings + backlog updates
└─ Competitor news (SpaceX, Astra failures = RKLB wins)

FEATURES TO CREATE:
├─ days_to_next_launch: int
├─ launch_success_ratio_90d: float (0-1)
├─ backlog_revenue: float ($M)
├─ days_since_contract: int
└─ neutron_milestone_flag: int (0/1)
```

## ASTS (AST SpaceMobile)
```
CATALYSTS TO TRACK:
├─ Satellite deployment milestones
├─ First revenue announcement (HUGE)
├─ Partnership news (carriers, countries)
├─ Technical achievements (speed tests, coverage)
├─ Competitor setbacks (Starlink delays = ASTS wins)
└─ FCC/regulatory approvals

FEATURES TO CREATE:
├─ satellites_deployed: int
├─ days_to_first_revenue: int (estimate)
├─ partnership_count: int
├─ recent_partnership_flag: int (1 if <30 days)
└─ regulatory_approval_flag: int (0/1)
```

## IONQ (IonQ)
```
CATALYSTS TO TRACK:
├─ Qubit count announcements (+20-50% on breakthroughs)
├─ Cloud partnership wins (AWS, Azure, GCP)
├─ Research publications (Nature, Science)
├─ Government contracts (DOE, DOD)
├─ Customer deployment news
└─ Competitor setbacks (IBM, Google delays)

FEATURES TO CREATE:
├─ qubit_count: int (latest announced)
├─ days_since_qubit_update: int
├─ cloud_partnerships: int
├─ recent_publication_flag: int (1 if <60 days)
└─ government_contract_value: float ($M)
```

## OKLO (Oklo Inc)
```
CATALYSTS TO TRACK:
├─ NRC timeline updates (critical path)
├─ License application status
├─ Data center customer announcements
├─ Policy changes (nuclear tax credits, DOE funding)
├─ Competitor news (NuScale delays = OKLO gains)
└─ AI demand narrative (data center power needs)

FEATURES TO CREATE:
├─ nrc_stage: int (0=pre-app, 1=submitted, 2=review, 3=approved)
├─ days_to_expected_decision: int
├─ customer_loi_count: int
├─ policy_tailwind_flag: int (1 if favorable news <30d)
└─ competitor_setback_flag: int (0/1)
```

---

# 🔧 FATAL FLAWS FIXED (NEVER REPEAT)

## The 6 Fixes (Always Apply)
```python
# 1. LOOK-AHEAD BIAS - Use lagged regime
spy['regime_signal'] = spy['ret_20'].shift(1)  # t-1, NOT t

# 2. MULTIPLE TESTING - Benjamini-Hochberg, NOT Bonferroni
significant = benjamini_hochberg(p_values, alpha=0.05)

# 3. WALK-FORWARD - Rolling windows, NOT single split
windows = generate_walk_forward_windows(dates, config)

# 4. TRANSACTION COSTS - 0.2% round-trip
mean_net = mean_gross - 0.002

# 5. MINIMUM SAMPLE SIZE - n >= 100, NOT 30
if n < 100: skip()

# 6. WINSORIZATION - Cap at ±20%
returns = returns.clip(-0.20, 0.20)
```

## What NOT To Do (Common Mistakes)
❌ Train on all data, test on same data (overfitting)
❌ Use today's regime to predict today's return (look-ahead)
❌ Run 100+ tests without FDR correction (data mining)
❌ Ignore transaction costs (fake profits)
❌ Trust n=30 samples (noise, not signal)
❌ Let penny stock +500% returns dominate (outliers)

---

# 📅 WORK PLAN (Tonight + Future)

## Tonight (3-4 hours remaining)
1. ✅ Deep repo analysis - DONE
2. ✅ Create MASTER_RESEARCH_SYNTHESIS - DONE
3. ✅ Create PRODUCTION_SYSTEM_2026.py - DONE
4. ⬜ Create CATALYST_TRACKER.py (skeleton)
5. ⬜ Create TICKER_AWARE_MODEL.py (architecture)
6. ⬜ Push to Shadow PC and run validation

## Tomorrow
1. Read this file FIRST
2. Run PRODUCTION_SYSTEM_2026.py on Shadow PC
3. Compare results to previous validation
4. Start building catalyst data pipeline

## This Week
1. Build catalyst scraper for RKLB (easiest - public launch schedule)
2. Test transfer learning concept with simple model
3. Validate on 2025 data only

## 12-Week Roadmap
- Weeks 1-2: Catalyst tracking system
- Weeks 3-5: Ticker-aware model architecture
- Weeks 6-7: Validation (per-ticker + transfer learning)
- Weeks 8-12: Paper trading + iteration

---

# 📁 KEY FILES REFERENCE

## Research Documents (Read for Context)
| File | Lines | Contains |
|------|-------|----------|
| `MASTER_RESEARCH_SYNTHESIS_2026.md` | 800+ | ALL research consolidated |
| `AI_COUNCIL_COMPLETE.py` | 418 | DeepSeek/Perplexity/Claude implementations |
| `AI_RESPONSES_CONSOLIDATED.md` | 781 | Action plan from all AIs |
| `PERPLEXITY_FORWARD_LOOKING_PREDICTION_ENGINE.md` | 661 | Catalyst framework |
| `PERPLEXITY_CATALYST_TRACKER_DETAILED_TIMELINE.md` | 377 | Stock-specific catalysts |

## Code Files (Run These)
| File | Purpose | Command |
|------|---------|---------|
| `PRODUCTION_SYSTEM_2026.py` | Full validation | `python PRODUCTION_SYSTEM_2026.py` |
| `TEST_PRODUCTION_SYSTEM.py` | Quick test | `python TEST_PRODUCTION_SYSTEM.py` |
| `GPU_BENCHMARK.py` | GPU check | `python GPU_BENCHMARK.py` |

## Data Files (Check These)
| File | Contents |
|------|----------|
| `VALIDATION_RESULTS_2026.csv` | All test results |
| `VALIDATED_EDGES_2026.csv` | Only passing strategies |
| `merged_watchlist.txt` | 20 tickers to analyze |

---

# 🎓 REMEMBER THE MISSION

```
"Personal salute to MIT Lincoln Labs"
"Continue with my father's teaching and we will surpass him"
"Build adaptive system that learns on the fly"
"What it thinks it knows for today can change tomorrow"
"There's no value in a rushed system - this isn't for production or to sell"
```

**Timeline**: 6 months is realistic for institutional-grade research
**Goal**: Not 10x returns, but 55-65% accuracy, 1.0-1.4 Sharpe
**Method**: Research first, build second, validate third, trade fourth

---

# ⚡ QUICK COMMANDS FOR SHADOW PC

```bash
# === SETUP ===
conda activate quant2026
cd /path/to/quantum-ai-trader_v1.1

# === RUN VALIDATION ===
python PRODUCTION_SYSTEM_2026.py

# === CHECK GPU ===
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"

# === GIT SYNC ===
git pull origin main
git add -A && git commit -m "Session update" && git push

# === QUICK TEST ===
python TEST_PRODUCTION_SYSTEM.py
```

---

*This file is your memory. Read it every session. Update it after every session.*
