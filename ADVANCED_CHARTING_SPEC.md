# Advanced Charting Spec (MVP)

## Overlay Schemas

- pattern_overlays: list of objects
  - id: string
  - ticker: string
  - pattern_type: string (elliott_impulse|flag|triangle|double_top|double_bottom|breakout|...)
  - stage: number or enum
  - created_at: ISO string
  - updated_at: ISO string
  - confidence: number (0-1)
  - status: "active"|"invalidated"
  - geometry: array (price/time points)

- forecast_cone: object
  - id: string
  - horizon_bars: number
  - quantiles: { q10: number[], q25: number[], q50: number[], q75: number[], q90: number[] }

- risk_zones: object
  - entry: number
  - stops: number[]
  - targets: number[]
  - size_hint: number (0-1 fraction of equity)

## Visual Rules
- Patterns: newest opaque (~0.9), fade with age to ~0.2.
- Forecast cone: filled band q25-q75, lines for q10/q50/q90.
- Risk zones: entry line thin; stop band semi-transparent red; targets with T1/T2/T3 markers.

## Confidence Bar (per ticker)
- trend, pattern, volatility, trader_agreement each 0-1.
- Render as stacked bar; width proportional to normalized weight.

## Timeline Strip (below x-axis)
- Events: pattern_start, pattern_complete, invalidation, alert, trade_intent, trade_fill.
- Event object: { id, ts, ticker, event_type, ref_pattern_id?, ref_order_id? }
- Hover highlights chart; click opens detail panel.

## Real-time Simulation
- Client → server: { type: "simulate_trade", ticker, entry_price, size_hint, side }
- Server response (<150ms): { expected_rr, probability_up, cone_targets, suggested_stop, summary }

## Pattern Evolution Ghosting
- Keep invalidated patterns for last 100-200 bars; low alpha (0.1-0.2); mark invalidation point.

## Multi-timeframe Stack View
- For 1h/4h/daily: trend_slope, regime_class.
- Render mini-bars/arrows beside main chart.

## Narrative Panel
- headline, alt_scenario, confidence_drift, key_factors.

## WebSocket Event Schema (common fields)
- type, ticker, module, ts, version, confidence, correlation_id, payload
- Delta ops for overlays: { op: add|update|remove, id, data }

## Client Interactions
- subscribe: { type: "subscribe", tickers: string[], streams: string[] }
- feedback: { type: "feedback", event_id, user_id, label: "useful"|"not_useful" }
- simulate_trade: per above

## Backfill REST
- GET /history/{ticker}/events?types=pattern,forecast&since=&limit=
- GET /state/{ticker}
