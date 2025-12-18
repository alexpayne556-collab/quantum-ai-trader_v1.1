from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


_TICKER_RE = re.compile(r"^[A-Z0-9][A-Z0-9.\-]{0,9}$")


def normalize_ticker(value: str) -> str | None:
    if value is None:
        return None
    s = str(value).strip().upper()
    if not s or s in {"N/A", "NA", "NONE", "NULL", "NAN"}:
        return None
    s = re.sub(r"[^A-Z0-9.\-]", "", s)
    if not s:
        return None
    if not _TICKER_RE.match(s):
        return None
    return s


def load_tickers_from_txt(path: Path) -> list[str]:
    if not path.exists():
        return []
    out: list[str] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        t = normalize_ticker(line)
        if t:
            out.append(t)
    return out


@dataclass(frozen=True)
class Universe:
    tickers: list[str]
    meta: pd.DataFrame | None = None


def load_universe_300(repo_root: Path) -> Universe:
    path = repo_root / "data" / "ticker_universe_300.csv"
    if not path.exists():
        return Universe(tickers=[])

    df = pd.read_csv(path)
    if "ticker" not in df.columns:
        return Universe(tickers=[])

    df = df.copy()
    df["ticker"] = df["ticker"].map(normalize_ticker)
    df = df[df["ticker"].notna()].drop_duplicates(subset=["ticker"]).reset_index(drop=True)
    return Universe(tickers=df["ticker"].tolist(), meta=df)


def build_universe(repo_root: Path, *, include_watchlists: bool = True, include_best2y: bool = True) -> list[str]:
    tickers = set()

    # Canonical universe (if present)
    u300 = load_universe_300(repo_root)
    tickers |= set(u300.tickers)

    # Watchlists
    if include_watchlists:
        for wl in [
            "alpha_76_watchlist.txt",
            "watchlist.txt",
            "merged_watchlist.txt",
            "small_caps_watchlist.txt",
            "ELITE_20_TICKERS.txt",
            "TOP_50_TICKERS.txt",
            "TOP_50_TICKERS_CLEAN.txt",
            "TOP_100_TICKERS.txt",
            "FINAL_50_TICKERS.txt",
        ]:
            tickers |= set(load_tickers_from_txt(repo_root / wl))

    # Session BEST_2Y (optional, but useful)
    if include_best2y:
        sessions_dir = repo_root / "sessions"
        if sessions_dir.exists():
            # pick most recent BEST_2Y.csv found
            best2y_paths = sorted(
                sessions_dir.glob("**/BEST_2Y.csv"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if best2y_paths:
                try:
                    df = pd.read_csv(best2y_paths[0])
                    if "ticker" in df.columns:
                        for v in df["ticker"].tolist():
                            t = normalize_ticker(v)
                            if t:
                                tickers.add(t)
                except Exception:
                    pass

    return sorted(tickers)
