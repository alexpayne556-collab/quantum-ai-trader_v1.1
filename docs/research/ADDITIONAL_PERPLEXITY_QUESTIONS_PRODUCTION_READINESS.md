# ⚙️ ADDITIONAL PERPLEXITY QUESTIONS: Production Readiness + Research Gaps

**Purpose**: Ask after the main 25 questions to close remaining risks before building. Keep answers concise and implementation-ready.
**When**: Tomorrow, after Batch 3 if time permits; otherwise first thing in afternoon.

---

## 🧠 New Questions (Advanced / Missing in Core 25)
1) **Timestamp & Causality Guardrails**
```
How to align free data timestamps (yfinance OHLCV, Reddit/Trends, EDGAR filings) to guarantee no look-ahead leakage when generating features and labels? Provide a timestamp alignment checklist and code pattern to enforce it.
```

2) **Event Windows & Labels**
```
For 5-15d horizons with catalysts, what is the best labeling scheme (rolling forward returns, hit ratios to barrier, survival/hazard models) that avoids double-counting overlapping events? Provide formulas + example code.
```

3) **Execution & Slippage Modeling (Free Data Only)**
```
Using only free data, what is a defensible slippage model for US equities (close-to-next-open, midquote proxy, spread-based) and how to parameterize it by regime/volume? Provide default haircuts for AI/Quantum/Robotaxi sectors.
```

4) **Structural Break / Regime-Shift Detection**
```
What tests (e.g., CUSUM, Bai-Perron) should run daily/weekly to detect structural breaks in feature → return relationships, and how to auto-throttle position sizes when breaks are detected?
```

5) **Drift & Health Monitoring in Production**
```
Minimal monitoring stack: which drift metrics (PSI, KL, KS), thresholds, and alert rules per feature group (patterns, regimes, catalysts)? Provide a lean alert template for Streamlit/Grafana.
```

6) **Ablation & Attribution Plan**
```
What is the fastest ablation test plan (weekend run) to quantify incremental Sharpe from: patterns only, +regimes, +research features, +calibration, +position sizing? Provide experiment matrix and stopping rules.
```

7) **Sequence Models vs Tabular Ensembles**
```
Given small samples (~100 trades/regime) and 40-60 features, when do TCN/LSTM/transformers beat tree ensembles, and how to regularize/early-stop them to avoid overfitting? Provide default hyperparameters.
```

8) **Options / Pre-Catalyst Proxies When OI Data Missing**
```
If free options OI/volume is sparse, what proxies (price/volume accelerations, gap filters, realized vol spikes, Form 4 clusters) best substitute for pre-catalyst detection? Provide top 5 heuristics.
```

9) **Portfolio Construction Overlay**
```
How to map per-ticker forecast → portfolio weights using only forecaster outputs + sector Sharpe + regime? Compare fractional Kelly, volatility targeting, and capped risk-parity at 5% max per name.
```

10) **Scenario & Stress Testing**
```
What historical scenarios (e.g., 2020 crash, 2022 rates shock, 2023 AI melt-up) should be replayed, and how to simulate them with free data to validate forecaster robustness? Provide a 1-day script outline.
```

11) **Latency & Recompute Budget**
```
For daily close runs on 100+ tickers, how to budget compute (<1s/ticker) and choose caching/materialization strategies for heavy features (breadth, cross-asset) while keeping determinism for backtests?
```

12) **Governance / Reproducibility**
```
What lightweight experiment tracking (artifact hashes, config versioning) should be added so every forecast/backtest is reproducible without adding paid services? Provide a minimal checklist.
```

---

## 📦 Production Module Checklist (to confirm with answers)
- `research_features.py`: deterministic feature store (timestamp-safe), caching knobs.
- `integrated_meta_learner.py`: regime/sector-aware ensemble with fallback weights when data is missing.
- `confidence_calibrator.py`: per-regime calibration curves + live recalibration trigger.
- `position_sizer.py`: fractional-Kelly/vol-target hybrid with caps and sector guards.
- `execution_sim.py`: free-data slippage model + borrow cost stub.
- `drift_monitor.py`: PSI/KS drift checks + alert hooks.
- `ablation_runner.py`: weekend experiment matrix for attribution.
- `scenario_tester.py`: replay 2020/2022/2023 scenarios with free data.
- `ops_checklist.md`: timestamp/causality checklist + reproducibility steps.

---

**How to use**: Drop these questions into a fourth Perplexity batch after Batch 3, or prioritize top 5 (1,3,4,5,9) if time is tight. Capture answers into `PERPLEXITY_FORECASTER_ARCHITECTURE.md` alongside the main 25.
