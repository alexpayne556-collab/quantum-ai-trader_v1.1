from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO_ROOT_DEFAULT = Path(__file__).resolve().parents[1]
repo_str = str(REPO_ROOT_DEFAULT)
if repo_str not in sys.path:
    sys.path.insert(0, repo_str)

from research.earnings_event_study import (
    EarningsStudyConfig,
    batch_event_study,
    summarize_by_eps_surprise,
)
from research.universe import build_universe, load_universe_300


def main() -> int:
    ap = argparse.ArgumentParser(description="Batch earnings-reaction event study (yfinance-backed)")
    ap.add_argument("--repo-root", default=str(REPO_ROOT_DEFAULT), help="repo root")
    ap.add_argument("--max-tickers", type=int, default=300, help="max tickers to process")
    ap.add_argument("--limit-events", type=int, default=8, help="earnings events per ticker")
    ap.add_argument("--horizon-days", type=int, default=5, help="drift horizon in trading days")
    ap.add_argument("--benchmark", default="SPY", help="benchmark ticker for abnormal returns (set empty to disable)")
    ap.add_argument(
        "--timing-aware",
        dest="timing_aware",
        action="store_true",
        default=True,
        help="align event day based on earnings time (after-close shifts to next trading day)",
    )
    ap.add_argument(
        "--no-timing-aware",
        dest="timing_aware",
        action="store_false",
        help="disable earnings time alignment (legacy behavior)",
    )
    ap.add_argument("--use-cache", action="store_true", default=True)
    ap.add_argument("--no-cache", dest="use_cache", action="store_false")
    ap.add_argument("--source", choices=["universe300", "all"], default="universe300")
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve()
    cache_dir = repo_root / "data" / "earnings_event_study_cache"
    run_dir = repo_root / "data" / "earnings_event_study_runs" / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.source == "universe300":
        u = load_universe_300(repo_root)
        tickers = u.tickers
        meta = u.meta
    else:
        tickers = build_universe(repo_root)
        meta = None

    tickers = tickers[: args.max_tickers]
    print(f"Repo: {repo_root}")
    print(f"Tickers: {len(tickers)} (source={args.source})")
    print(f"Cache dir: {cache_dir}")
    print(f"Run dir: {run_dir}")

    config = EarningsStudyConfig(
        cache_dir=cache_dir,
        limit_events=args.limit_events,
        horizon_days=args.horizon_days,
        use_cache=args.use_cache,
        benchmark_ticker=(args.benchmark.strip().upper() if args.benchmark.strip() else None),
        timing_aware=bool(args.timing_aware),
    )

    events, failures = batch_event_study(tickers, config=config, progress_every=25)

    events_path = run_dir / "events.csv"
    failures_path = run_dir / "failures.csv"
    summary_path = run_dir / "summary_by_eps_surprise.csv"

    events.to_csv(events_path, index=False)
    failures.to_csv(failures_path, index=False)

    summary = summarize_by_eps_surprise(events, horizon_days=args.horizon_days, include_abnormal=True)
    if len(summary):
        # Flatten columns for CSV friendliness
        summary_flat = summary.copy()
        summary_flat.columns = ["_".join([c for c in col if c]) for col in summary_flat.columns.to_flat_index()]
        summary_flat.to_csv(summary_path)

    # Optional: attach meta (sector/industry) to events for downstream analysis
    if meta is not None and len(meta) and len(events):
        try:
            meta_cols = [c for c in ["ticker", "sector", "industry", "market_cap_category"] if c in meta.columns]
            enriched = events.merge(meta[meta_cols], on="ticker", how="left")
            enriched.to_csv(run_dir / "events_enriched.csv", index=False)
        except Exception:
            pass

    print("Done")
    print(f"Events rows: {len(events)}")
    print(f"Failures: {len(failures)}")
    if len(summary):
        print("Summary saved:", summary_path)
    else:
        print("No EPS surprise coverage; summary empty")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
