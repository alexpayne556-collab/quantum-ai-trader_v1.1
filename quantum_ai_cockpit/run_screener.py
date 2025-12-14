"""Executable entry point for the Quantum AI Cockpit screener."""
from __future__ import annotations

import argparse
import sys
try:
    from config.improvement_loader import load_improvements
except Exception:
    def load_improvements(path=None):
        return {}, {}
import logging
from datetime import datetime
from pathlib import Path

from .config import SIGNAL_THRESHOLD
from .screener import scan_universe
from .utils import ensure_directory

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def run(output_dir: Path, period: str, threshold: float) -> Path | None:
    df = scan_universe(period=period, score_threshold=threshold)

    if df.empty:
        logger.info("No qualifying setups found (threshold=%.1f)", threshold)
        return None

    logger.info("Qualified setups:\n%s", df.to_string(index=False))

    ensure_directory(output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"qualified_setups_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    logger.info("Exported %d setups to %s", len(df), csv_path)
    return csv_path

def get_runtime_weights():
    weights, _ = load_improvements()
    return weights


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Quantum AI screener.")
    parser.add_argument(
        "--period",
        default="6mo",
        help="Historical period to request from yfinance (default: 6mo)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=SIGNAL_THRESHOLD,
        help="Minimum score required to qualify (default: config.SIGNAL_THRESHOLD)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/outputs",
        help="Directory to store CSV exports",
    )
    args = parser.parse_args()

    run(Path(args.output_dir), args.period, args.threshold)


if __name__ == "__main__":
    main()
