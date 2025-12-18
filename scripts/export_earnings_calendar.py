from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT_DEFAULT = Path(__file__).resolve().parents[1]
repo_str = str(REPO_ROOT_DEFAULT)
if repo_str not in sys.path:
    sys.path.insert(0, repo_str)

try:
    from dotenv import load_dotenv

    load_dotenv(dotenv_path=REPO_ROOT_DEFAULT / ".env")
except Exception:
    pass

from research.earnings_calendar import EarningsCalendarConfig, build_earnings_calendar, save_calendar
from research.universe import build_universe, load_universe_300


def main() -> int:
    ap = argparse.ArgumentParser(description="Export upcoming earnings calendar for a ticker universe")
    ap.add_argument("--repo-root", default=str(REPO_ROOT_DEFAULT))
    ap.add_argument("--days-ahead", type=int, default=30)
    ap.add_argument("--max-tickers", type=int, default=300)
    ap.add_argument("--source", choices=["universe300", "all"], default="universe300")
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve()

    finnhub_key = os.getenv("FINNHUB_API_KEY", "").strip() or None

    if args.source == "universe300":
        u = load_universe_300(repo_root)
        tickers = u.tickers
    else:
        tickers = build_universe(repo_root)

    tickers = tickers[: args.max_tickers]

    cfg = EarningsCalendarConfig(days_ahead=args.days_ahead, finnhub_api_key=finnhub_key)
    df = build_earnings_calendar(tickers, config=cfg)

    run_dir = repo_root / "data" / "events" / "earnings_calendar" / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = run_dir / "earnings_calendar.csv"
    save_calendar(df, out_path=out_path)

    print(f"Repo: {repo_root}")
    print(f"Tickers: {len(tickers)}")
    print(f"Rows: {len(df)}")
    print(f"Output: {out_path}")
    if finnhub_key:
        print("Source: finnhub (market calendar filtered to your tickers)")
    else:
        print("Source: yfinance fallback (per-ticker next earnings; less reliable)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
