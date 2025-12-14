import os
import json
from datetime import datetime
from typing import Optional, Dict, Any

class PredictionLogger:
    """Unified JSONL event logger for model outputs and decisions.

    Each call to log_event writes one JSON line with a common schema:
    {
        "ts": ISO timestamp,
        "type": event category (forecast|wave|risk|scanner|execution|meta),
        "ticker": optional ticker symbol,
        "data": payload dict
    }
    """
    def __init__(self, log_path: str = "logs/prediction_events.jsonl"):
        self.log_path = log_path
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

    def _write(self, record: Dict[str, Any]) -> None:
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")

    def log_event(self, event_type: str, ticker: Optional[str], data: Dict[str, Any]) -> None:
        record = {
            "ts": datetime.utcnow().isoformat(),
            "type": event_type,
            "ticker": ticker,
            "data": data
        }
        self._write(record)

    # Convenience wrappers -------------------------------------------------
    def log_forecast(self, ticker: str, horizon: int, metrics: Dict[str, Any]) -> None:
        self.log_event("forecast", ticker, {"horizon": horizon, **metrics})

    def log_wave_pattern(self, ticker: str, hit_rate: float, patterns_found: int, params: Dict[str, Any]) -> None:
        self.log_event("wave", ticker, {
            "hit_rate": hit_rate,
            "patterns_found": patterns_found,
            "params": params
        })

    def log_risk_decision(self, ticker: str, config: Dict[str, Any], metrics: Dict[str, Any]) -> None:
        self.log_event("risk", ticker, {"config": config, **metrics})

    def log_scanner_flag(self, ticker: str, config: Dict[str, Any], metrics: Dict[str, Any]) -> None:
        self.log_event("scanner", ticker, {"config": config, **metrics})

    def log_execution_validation(self, ticker: str, order_valid: bool, violations: Any) -> None:
        self.log_event("execution", ticker, {"order_valid": order_valid, "violations": violations})

    def log_meta(self, label: str, payload: Dict[str, Any]) -> None:
        self.log_event("meta", label, payload)
