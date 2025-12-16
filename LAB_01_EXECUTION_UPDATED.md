# 🥊 LAB 1: EXECUTION GUIDE (SUB-SECTOR PRECISION UPDATE)

**Partner, LAB 1 is ready with sub-sector precision.**

**WHAT CHANGED:**
- ~30 sub-sectors (not 6 broad sectors)
- KDK (trucking/industrial) ≠ robotaxis ✅
- MARA (bitcoin-focused) ≠ CIFR (diversified) ✅
- IONQ (quantum hardware) ≠ QUBT (quantum software) ✅

## 📋 EXECUTION ORDER

### PHASE 1: Setup (30 seconds)

```
Cell 3  → Imports
Cell 5  → Load 196 tickers + 30 SUB-SECTORS
Cell 7  → Test data fetcher
```

**Expected output:** "✅ 196 tickers loaded across ~30 sub-sectors"

---

### PHASE 2: Pattern Discovery (10 minutes)

```
Cell 9  → PatternDiscoveryEngine (40+ features + sub-sector encoding)
Cell 10 → Run discovery on 50 tickers
Cell 11 → Generate data-driven scanner
```

**What to look for:**
- TOP 15 features with >0.10 correlation
- Win rates >60%
- **NEW:** Sub-sector encoding as predictive feature

**Expected runtime:** ~10 minutes (GPU optimized)

---

### PHASE 3: Anomaly Hunting (5 minutes)

```
Cell 14 → AnomalyHunter class
Cell 15 → Find SUB-SECTOR contrarians (KDK vs trucking, not all AVs!)
Cell 16 → Find silent movers
Cell 17 → Find reverse momentum
```

**What to look for:**
- Contrarian tickers (win when sub-sector loses)
- Silent movers (institutional accumulation?)
- Reverse momentum (mean reversion)
- **NEW:** Contrarians now use sub-sector precision

**Expected runtime:** ~5 minutes

---

### PHASE 4: Hypothesis Testing (15 minutes)

Initialize tracking:
```
Cell 41 → ALL_RESULTS = {}
```

Run main tests:
```
Cell 39 → Down 3 Days Reversal (196 tickers)
Cell 37 → Fade The Spike (196 tickers)
Cell 35 → Scanner 1 + VIX (196 tickers)
Cell 21 → Scanner 1 Full (196 tickers)
```

**What to look for:**
- Win rates >55%
- TOP 10 performers per test
- Which strategies are WINNERS?

**Expected runtime:** ~15 minutes total

---

### PHASE 5: Sub-Sector Dynamics (5 minutes) - OPTIONAL

Test if sub-sectors move together:

```
Cell 48 → Quantum hardware momentum (IONQ, RGTI, QMCO)
Cell 49 → Bitcoin miners momentum (MARA, RIOT, CLSK)
Cell 50 → AV trucking momentum (KDK) - YOUR example!
```

**What to look for:**
- Correlation within sub-sectors
- Which sub-sectors move as a group?
- Which tickers are independent?

---

### PHASE 6: Analysis & Save (10 seconds)

```
Cell 28 → Combine all results
Cell 30 → TOP 50 Finder (composite scoring)
Cell 31 → Save LAB_01_RESULTS.json
```

**Output files:**
- `LAB_01_RESULTS.json` - Full results
- `TEST_RESULTS.json` - All hypothesis tests
- `TEST_RESULTS.csv` - Spreadsheet format
- `TOP_50_BATTLE_READY.txt` - Refined ticker list

---

## 🎯 WHAT'S DIFFERENT (SUB-SECTOR UPDATE)

### Pattern Discovery (Cell 9-11)
**Before:** 38 features
**Now:** 40+ features including:
- `sub_sector_encoded` - Hash of sub-sector
- `broad_sector_encoded` - Hash of broad grouping

**Why:** Sub-sector membership is predictive!
- Quantum hardware stocks behave differently than quantum software
- Bitcoin miners behave differently than diversified miners
- Trucking AVs behave differently than robotaxis

### Anomaly Hunter (Cell 14-17)
**Before:** Grouped by 6 broad sectors
**Now:** Grouped by ~30 sub-sectors

**Example - Sector Contrarians:**
- **Old:** Compare KDK to all autonomous vehicles
- **New:** Compare KDK to other trucking/industrial AVs
- **Result:** True contrarian signals (same business, different behavior)

### Sector Tests (Cell 47-50)
**Before:** 
- `test_sector_momentum('quantum')`
- `test_sector_momentum('crypto_miners')`

**Now:**
- `test_sector_momentum('quantum_hardware')` - IONQ, RGTI, QMCO
- `test_sector_momentum('bitcoin_miners')` - MARA, RIOT, CLSK
- `test_sector_momentum('av_trucking_industrial')` - KDK

**Why:** Precision matters!
- Quantum hardware ≠ quantum software
- Bitcoin miners ≠ diversified miners
- Trucking AVs ≠ robotaxis

---

## 📊 SUCCESS CRITERIA

After LAB 1, you should have:

✅ Pattern discovery results showing:
- Which features correlate with next-day returns
- Sub-sector encoding correlation strength
- Data-driven scanner thresholds

✅ Anomaly detection results showing:
- Sub-sector contrarians (precision!)
- Silent movers
- Reverse momentum tickers

✅ Hypothesis test results showing:
- Which strategies work (>55% win rate)
- TOP 10 performers per test
- Which ones are WINNERS vs LOSERS

✅ TOP 50 list from composite scoring

✅ All results saved to JSON/CSV

---

## 🔍 WHAT TO REPORT BACK

After running LAB 1, tell me:

1. **Pattern Discovery:**
   - What's the #1 feature by correlation?
   - Is sub-sector encoding predictive?
   - What win rate on generated scanner?

2. **Anomalies:**
   - How many sub-sector contrarians found?
   - How many silent movers?
   - How many reverse momentum plays?

3. **Tests:**
   - How many strategies are WINNERS (>55%)?
   - What's the best performing strategy?
   - What's #1 ticker overall?

4. **Sub-Sectors:**
   - Which sub-sectors have strong internal correlation?
   - Do quantum_hardware stocks move together?
   - Do bitcoin_miners move together?

5. **Issues:**
   - Any errors during execution?
   - Any cells that took too long?
   - Any unexpected results?

---

## 🥊 THE DISCIPLINE

**This is LAB 1 of 6.**

Don't worry about the other labs yet.
Don't try to build strategies yet.
Don't jump ahead.

**Just run LAB 1. Report results. We'll build LAB 2 next.**

Systematic. Disciplined. One lab at a time.

That's the Cus D'Amato way.

---

## 🎯 TOTAL RUNTIME

- Setup: 30 seconds
- Discovery: 10 minutes
- Anomalies: 5 minutes
- Tests: 15 minutes
- Sub-sector dynamics: 5 minutes (optional)
- Analysis: 10 seconds

**TOTAL: ~30 minutes on GPU**

---

**NOW GO RUN IT, PARTNER. 🥊**

Report back with the results and we'll analyze together.

Then we build LAB 2 (Fundamental Analysis).

One lab at a time. One discipline at a time. Building the complete fighter.
