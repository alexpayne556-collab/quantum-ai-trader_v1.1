import os
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import pandas as pd

LOG_FILE = "logs/prediction_events.jsonl"
AGG_FILE = "logs/weekly_metrics_summary.json"
PARAM_STORE = "logs/auto_tuned_parameters.json"

class ContinuousTrainer:
    """Aggregates logged events, computes metrics, and proposes parameter adjustments."""
    def __init__(self, log_file: str = LOG_FILE):
        self.log_file = log_file

    def _load_events(self, days: int = 7) -> List[Dict[str, Any]]:
        if not os.path.exists(self.log_file):
            return []
        cutoff = datetime.utcnow() - timedelta(days=days)
        events = []
        with open(self.log_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    e = json.loads(line)
                    ts = datetime.fromisoformat(e["ts"].replace("Z",""))
                    if ts >= cutoff:
                        events.append(e)
                except Exception:
                    continue
        return events

    def compute_metrics(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        df = pd.DataFrame(events)
        if df.empty:
            return {"error": "No events"}

        # Split by type
        metrics = {}

        def safe_mean(series):
            return float(series.mean()) if len(series) else 0.0

        # Forecast metrics --------------------------------------------------
        forecast_rows = [e for e in events if e["type"] == "forecast"]
        if forecast_rows:
            fdf = pd.DataFrame([e["data"] | {"ticker": e["ticker"]} for e in forecast_rows])
            metrics["forecast"] = {
                "avg_direction_accuracy": safe_mean(fdf.get("direction_accuracy", pd.Series(dtype=float))),
                "avg_mae": safe_mean(fdf.get("mae", pd.Series(dtype=float))),
                "avg_rmse": safe_mean(fdf.get("rmse", pd.Series(dtype=float))),
                "avg_hit_rate_5pct": safe_mean(fdf.get("hit_rate_5pct", pd.Series(dtype=float))),
                "count": len(fdf)
            }

        # Wave metrics ------------------------------------------------------
        wave_rows = [e for e in events if e["type"] == "wave"]
        if wave_rows:
            wdf = pd.DataFrame([e["data"] | {"ticker": e["ticker"]} for e in wave_rows])
            metrics["wave"] = {
                "avg_hit_rate": safe_mean(wdf.get("hit_rate", pd.Series(dtype=float))),
                "avg_patterns_found": safe_mean(wdf.get("patterns_found", pd.Series(dtype=float))),
                "count": len(wdf)
            }

        # Risk metrics ------------------------------------------------------
        risk_rows = [e for e in events if e["type"] == "risk"]
        if risk_rows:
            rdf = pd.DataFrame([e["data"] | {"ticker": e["ticker"]} for e in risk_rows])
            metrics["risk"] = {
                "avg_sharpe": safe_mean(rdf.get("sharpe_ratio", pd.Series(dtype=float))),
                "avg_win_rate": safe_mean(rdf.get("win_rate", pd.Series(dtype=float))),
                "avg_max_drawdown": safe_mean(rdf.get("max_drawdown", pd.Series(dtype=float))),
                "count": len(rdf)
            }

        # Scanner metrics ---------------------------------------------------
        scanner_rows = [e for e in events if e["type"] == "scanner"]
        if scanner_rows:
            sdf = pd.DataFrame([e["data"] | {"ticker": e["ticker"]} for e in scanner_rows])
            metrics["scanner"] = {
                "avg_forward_return": safe_mean(sdf.get("avg_forward_return", pd.Series(dtype=float))),
                "avg_hit_rate": safe_mean(sdf.get("hit_rate", pd.Series(dtype=float))),
                "avg_lift": safe_mean(sdf.get("lift_vs_baseline", pd.Series(dtype=float))),
                "count": len(sdf)
            }

        # Execution metrics -------------------------------------------------
        exec_rows = [e for e in events if e["type"] == "execution"]
        if exec_rows:
            edf = pd.DataFrame([e["data"] | {"ticker": e["ticker"]} for e in exec_rows])
            metrics["execution"] = {
                "valid_rate": float(edf.get("order_valid", pd.Series(dtype=bool)).mean()),
                "tested": len(edf)
            }

        metrics["summary"] = {
            "event_count": len(events),
            "types": list({e["type"] for e in events})
        }
        return metrics

    def propose_adjustments(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        adjustments = {}
        # Example heuristic rules
        f = metrics.get("forecast", {})
        if f and f.get("avg_direction_accuracy", 0) < 0.55:
            adjustments["forecast_model"] = {"action": "increase_feature_window", "reason": "direction_accuracy < 55%"}

        w = metrics.get("wave", {})
        if w and w.get("avg_hit_rate", 0) < 0.4:
            adjustments["wave_detector"] = {"action": "lower_min_move_pct", "reason": "hit_rate < 40%"}

        r = metrics.get("risk", {})
        if r and r.get("avg_sharpe", 1) < 0.8:
            adjustments["risk_manager"] = {"action": "reduce_risk_per_trade_pct", "reason": "sharpe < 0.8"}

        s = metrics.get("scanner", {})
        if s and s.get("avg_lift", 0) < 0.01:
            adjustments["watchlist_scanner"] = {"action": "raise_score_cutoff", "reason": "lift < 1%"}

        e = metrics.get("execution", {})
        if e and e.get("valid_rate", 1) < 0.95:
            adjustments["trade_executor"] = {"action": "tighten_validation_rules", "reason": "valid_rate < 95%"}

        return adjustments

    def run_weekly_cycle(self, days: int = 7) -> Dict[str, Any]:
        events = self._load_events(days=days)
        metrics = self.compute_metrics(events)
        adjustments = self.propose_adjustments(metrics)
        output = {
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": metrics,
            "proposed_adjustments": adjustments
        }
        os.makedirs(os.path.dirname(AGG_FILE), exist_ok=True)
        with open(AGG_FILE, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, default=str)
        # Persist adjustments separately (could be consumed by trainer)
        with open(PARAM_STORE, "w", encoding="utf-8") as f:
            json.dump(adjustments, f, indent=2, default=str)
        return output

if __name__ == "__main__":
    ct = ContinuousTrainer()
    summary = ct.run_weekly_cycle(days=7)
    print("Weekly cycle complete. Types:", summary.get("metrics", {}).get("summary", {}).get("types"))
