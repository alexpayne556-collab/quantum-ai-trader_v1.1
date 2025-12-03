# Quantum AI Cockpit — Frontend Integration Guide

## Overview
This guide shows you how to easily connect your React Vite dashboard to the Quantum AI Cockpit backend and display Plotly graphs for every module.

## Quick Start

### 1. Import the API Client
```javascript
import { apiClient } from './utils/apiClient';
```

### 2. Use the React Hooks
```javascript
import { useDashboard, useForecast } from './hooks/useQuantumAPI';

function MyComponent() {
  const { data, plotlyData, loading, error } = useDashboard('AAPL');
  
  return (
    <PlotlyGraph 
      ticker="AAPL"
      forecastData={data?.forecast}
      height={500}
    />
  );
}
```

## API Client (`apiClient.js`)

The `apiClient` provides a unified interface to all backend endpoints:

### Methods

#### `getForecast(symbol)`
Fetches forecast data with visual_context for Plotly graphs.
```javascript
const forecast = await apiClient.getForecast('AAPL');
// Returns: { symbol, trend, confidence, forecast_days, visual_context, ... }
```

#### `getDashboard(symbol)`
Fetches complete dashboard data including forecast, opportunities, and AI recommendations.
```javascript
const dashboard = await apiClient.getDashboard('AAPL');
// Returns: { symbol, forecast, opportunities, ai_recommendation, market_open, ... }
```

#### `getPrediction(symbol)`
Gets price prediction with confidence and signals.
```javascript
const prediction = await apiClient.getPrediction('AAPL');
// Returns: { symbol, prediction, confidence, signals, timestamp }
```

#### `formatForPlotly(forecastData)`
Formats forecast data for PlotlyGraph component.
```javascript
const plotlyData = apiClient.formatForPlotly(forecastData);
// Returns: { traces: [...], layout: {...} }
```

## React Hooks (`useQuantumAPI.js`)

### `useForecast(symbol, autoFetch = true)`
Hook for fetching forecast data.

**Returns:**
- `data` - Raw forecast data
- `loading` - Loading state
- `error` - Error message
- `refetch` - Function to refetch data
- `plotlyData` - Pre-formatted Plotly data

**Example:**
```javascript
const { data, plotlyData, loading } = useForecast('AAPL');

if (loading) return <div>Loading...</div>;

return <PlotlyGraph ticker="AAPL" forecastData={data} />;
```

### `useDashboard(symbol, autoFetch = true)`
Hook for fetching complete dashboard data.

**Returns:**
- `data` - Raw dashboard data
- `loading` - Loading state
- `error` - Error message
- `refetch` - Function to refetch
- `plotlyData` - Pre-formatted Plotly data
- `opportunities` - Trading opportunities array
- `aiRecommendation` - AI recommendation object
- `marketOpen` - Boolean for market status
- `trend` - Trend string (bullish/bearish/neutral)
- `confidence` - Confidence score (0-1)
- `signals` - Technical signals object
- `metrics` - Technical metrics object

**Example:**
```javascript
const { 
  plotlyData, 
  opportunities, 
  aiRecommendation,
  marketOpen 
} = useDashboard('AAPL');

return (
  <div>
    <PlotlyGraph ticker="AAPL" forecastData={data?.forecast} />
    {opportunities.map(op => (
      <div key={op.date}>
        {op.kind}: Entry ${op.entry}, RR {op.rr}
      </div>
    ))}
  </div>
);
```

### `useQuantumWebSocket(endpoint)`
Hook for WebSocket real-time updates.

**Example:**
```javascript
const { data, isConnected } = useQuantumWebSocket('/ws/trading-signals');
```

### `usePortfolio()` and `useWatchlist()`
Hooks for portfolio and watchlist data.

## PlotlyGraph Component

The `PlotlyGraph` component automatically renders charts from `visual_context`:

```javascript
<PlotlyGraph
  ticker="AAPL"
  forecastData={forecastData}  // Must have visual_context
  height={500}
  theme="dark"
  aiInsight={aiRecommendation}  // Optional: for sentiment styling
/>
```

**Required Props:**
- `forecastData` - Object with `visual_context.traces` and `visual_context.layout`

**Optional Props:**
- `ticker` - Symbol for display
- `height` - Chart height in pixels
- `theme` - "dark" or "light"
- `aiInsight` - AI recommendation for sentiment-based styling

## Example Component

See `QuantumDashboardExample.jsx` for a complete example showing:
- Forecast chart with PlotlyGraph
- Trading opportunities display
- AI recommendations
- Market status indicators
- Technical metrics

## Data Format

### Forecast Data Structure
```javascript
{
  symbol: "AAPL",
  status: "ok",
  trend: "bullish",
  confidence: 0.85,
  forecast_days: [150.2, 151.5, 152.1, ...],
  visual_context: {
    traces: [
      { type: "candlestick", ... },
      { type: "scatter", name: "EMA 8", ... },
      { type: "scatter", name: "EMA 21", ... },
      { type: "scatter", name: "14-Day Forecast", ... }
    ],
    layout: {
      title: "AAPL – 14D Fusior Forecast",
      template: "plotly_dark",
      xaxis: { title: "Date" },
      yaxis: { title: "Price ($)" }
    }
  },
  opportunities: [
    {
      kind: "dip",
      date: "2025-01-15",
      entry: 150.0,
      stop: 148.0,
      target: 154.0,
      rr: 2.0,
      confidence: 0.75,
      rationale: "..."
    }
  ],
  ai_recommendation: {
    stance: "accumulate_on_dips",
    summary: "Don't chase. Likely pullback first...",
    rationale: ["..."],
    next_actions: ["Plan: BUY the dip near 150.00..."]
  },
  market_open: true
}
```

## Environment Variables

Create a `.env` file in the frontend directory:

```env
VITE_API_URL=http://127.0.0.1:8090
VITE_WS_URL=ws://127.0.0.1:8090
```

## Troubleshooting

### No Chart Displaying
1. Check that `forecastData.visual_context.traces` is an array
2. Verify backend is running on the correct port
3. Check browser console for errors

### NaN/Infinity Errors
The backend automatically sanitizes NaN/Inf values. If you see errors:
1. Check backend logs for serialization issues
2. Verify `json_sanitize.py` is being used

### WebSocket Not Connecting
1. Verify `VITE_WS_URL` is correct
2. Check backend WebSocket endpoint exists
3. Check browser console for connection errors

## Backend Endpoints

All endpoints return JSON with proper `visual_context` for Plotly:

- `GET /api/forecast/{symbol}` - Forecast with visual_context
- `GET /api/dashboard?symbol={symbol}` - Complete dashboard data
- `POST /predict` - Price prediction
- `GET /api/portfolio` - Portfolio data
- `GET /api/watchlist` - Watchlist data
- `WebSocket /ws/trading-signals` - Real-time signals

## Next Steps

1. Import `useDashboard` hook in your component
2. Pass `forecastData` to `PlotlyGraph`
3. Display opportunities and AI recommendations
4. Add WebSocket for real-time updates

For a complete example, see `QuantumDashboardExample.jsx`.

