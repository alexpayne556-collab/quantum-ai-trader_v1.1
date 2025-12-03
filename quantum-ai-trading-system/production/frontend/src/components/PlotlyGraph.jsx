/**
 * Quantum AI Cockpit — Stable Plotly Graph Component
 * 🎨 No flicker, keeps last good frame, coerces axes correctly
 * ============================================================
 */

import React, { useEffect, useRef, useState } from "react";
import Plotly from "plotly.js-dist-min";

// Coerce visual_context client-side (safety net)
function coerceVisualContext(vc) {
  if (!vc) return { traces: [], layout: {} };
  
  const traces = Array.isArray(vc.traces) ? [...vc.traces] : [];
  const layout = vc.layout || {};
  
  // Assign yaxis to traces
  for (const t of traces) {
    if (!t.yaxis) {
      const nm = (t.name || "").toLowerCase();
      const typ = (t.type || "").toLowerCase();
      
      if (typ.includes("candlestick") || ["price", "ema", "close", "ohlc"].some(k => nm.includes(k))) {
        t.yaxis = "y";
      } else if (nm.includes("rsi")) {
        t.yaxis = "y2";
      } else if (["macd", "signal", "hist"].some(k => nm.includes(k))) {
        t.yaxis = "y3";
      } else {
        t.yaxis = "y";
      }
    }
  }
  
  // Ensure layout has proper axes
  if (!layout.yaxis) {
    layout.yaxis = { title: "Price", domain: [0.30, 1.00] };
  }
  if (!layout.yaxis2) {
    layout.yaxis2 = { title: "RSI", overlaying: "y", position: 1, range: [0, 100], showgrid: false };
  }
  if (!layout.yaxis3) {
    layout.yaxis3 = { title: "MACD", domain: [0.00, 0.25] };
  }
  if (!layout.xaxis) {
    layout.xaxis = { title: "Date", rangeslider: { visible: false } };
  }
  layout.template = layout.template || "plotly_dark";
  layout.margin = layout.margin || { l: 50, r: 50, t: 40, b: 40 };
  
  return { traces, layout };
}

export default function PlotlyGraph({
  ticker = "AAPL",
  data = null,
  forecastData = null,
  height = 560,
  theme = "dark",
}) {
  const chartRef = useRef(null);
  const lastGoodDataRef = useRef(null);
  const [isLoading, setIsLoading] = useState(true);
  
  useEffect(() => {
    if (!chartRef.current) return;
    
    // Extract visual_context from forecastData or data
    let vc = null;
    if (forecastData?.visual_context) {
      vc = forecastData.visual_context;
    } else if (data?.visual_context) {
      vc = data.visual_context;
    } else if (forecastData) {
      vc = forecastData;
    }
    
    // Coerce visual_context
    const coerced = coerceVisualContext(vc);
    const traces = coerced.traces || [];
    const layout = coerced.layout || {};
    
    // Only update if we have traces
    if (traces.length > 0) {
      // Update last good data
      lastGoodDataRef.current = { traces, layout };
      setIsLoading(false);
      
      // Use Plotly.react for smooth updates (keeps last good frame)
      Plotly.react(chartRef.current, traces, layout, {
        responsive: true,
        displayModeBar: true,
        displaylogo: false,
        modeBarButtonsToRemove: ["lasso2d", "select2d"],
        doubleClick: "reset",
        scrollZoom: true,
      });
    } else if (lastGoodDataRef.current) {
      // Keep last good frame if no new data
      const { traces, layout } = lastGoodDataRef.current;
      Plotly.react(chartRef.current, traces, layout, {
        responsive: true,
        displayModeBar: true,
        displaylogo: false,
        modeBarButtonsToRemove: ["lasso2d", "select2d"],
        doubleClick: "reset",
        scrollZoom: true,
      });
    } else {
      // No data and no last good frame - show empty chart
      setIsLoading(true);
    }
    
    // Handle window resize
    const handleResize = () => {
      if (chartRef.current && lastGoodDataRef.current) {
        Plotly.Plots.resize(chartRef.current);
      }
    };
    
    window.addEventListener("resize", handleResize);
    
    return () => {
      window.removeEventListener("resize", handleResize);
      // Don't purge on unmount - keep chart for smooth transitions
    };
  }, [ticker, forecastData, data, theme]);
  
  return (
    <div style={{ position: "relative", width: "100%", height: `${height}px` }}>
      {isLoading && !lastGoodDataRef.current && (
        <div
          style={{
            position: "absolute",
            top: "50%",
            left: "50%",
            transform: "translate(-50%, -50%)",
            color: "#00ffaa",
            fontSize: "14px",
          }}
        >
          Loading chart...
        </div>
      )}
      <div ref={chartRef} style={{ width: "100%", height: "100%" }} />
    </div>
  );
}
