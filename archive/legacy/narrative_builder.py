from typing import Dict, Any

def build_narrative(features: Dict[str, Any], outputs: Dict[str, Any]) -> Dict[str, Any]:
    p_up = outputs.get("p_up", 0.5)
    edge_score = outputs.get("edge_score", 50)
    trend = features.get("trend_features", {}).get("regime_class", "CHOP")
    trader_action = features.get("ai_trader_action", "FLAT")
    headline = "Neutral bias"
    if p_up > 0.6:
        headline = "Mildly bullish"
    if p_up > 0.7:
        headline = "Bullish bias"
    if p_up < 0.4:
        headline = "Mildly bearish"
    if p_up < 0.3:
        headline = "Bearish bias"

    alt = "If price loses recent support, bias flips to neutral."
    if trend == "BEAR" and p_up > 0.55:
        alt = "Up-bias against bear regime; tighten stops or half-size."

    drift = outputs.get("confidence_drift", "stable")

    return {
        "headline": headline,
        "alt_scenario": alt,
        "confidence_drift": drift,
        "key_factors": outputs.get("key_factors", [])
    }
