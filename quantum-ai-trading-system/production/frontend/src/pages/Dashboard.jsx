/**
 * Quantum AI Cockpit — Dashboard Page (Stable)
 * 📊 Debounced symbol, guarded fetch, keeps last good visual_context
 * ===================================================================
 */

import React, { useState, useEffect, useRef, useCallback } from "react";
import PlotlyGraph from "../components/PlotlyGraph";
import TickerSearch from "../components/TickerSearch";
import { guardedFetchJson, unwrap } from "../lib/api";

function Dashboard({ onSymbolChange }) {
  const [symbol, setSymbol] = useState("AAPL");
  const [debouncedSymbol, setDebouncedSymbol] = useState("AAPL");
  const [forecastVC, setForecastVC] = useState(null);
  const [fallbackData, setFallbackData] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  
  const lastGoodVCRef = useRef(null);
  const debounceTimeoutRef = useRef(null);

  // Debounce symbol changes
  useEffect(() => {
    if (debounceTimeoutRef.current) {
      clearTimeout(debounceTimeoutRef.current);
    }
    
    debounceTimeoutRef.current = setTimeout(() => {
      setDebouncedSymbol(symbol);
    }, 500);
    
    return () => {
      if (debounceTimeoutRef.current) {
        clearTimeout(debounceTimeoutRef.current);
      }
    };
  }, [symbol]);

  // Fetch forecast when debounced symbol changes
  useEffect(() => {
    const fetchForecast = async () => {
      setIsLoading(true);
      setError(null);
      
      try {
        const json = await guardedFetchJson(`/api/forecast/${debouncedSymbol}`);
        
        // Unwrap response
        const forecast = unwrap(json);
        
        // Extract visual_context
        const vc = forecast?.visual_context || json?.visual_context;
        
        if (vc && vc.traces && vc.traces.length > 0) {
          // Update last good visual_context
          lastGoodVCRef.current = vc;
          setForecastVC(vc);
        } else if (lastGoodVCRef.current) {
          // Keep last good frame if no new visual_context
          setForecastVC(lastGoodVCRef.current);
        }
        
        // Store fallback data
        setFallbackData(forecast);
      } catch (err) {
        if (err.message !== "stale") {
          console.error("Forecast fetch error:", err);
          setError(err.message);
          // Don't clear chart on error - keep last good frame
        }
      } finally {
        setIsLoading(false);
      }
    };
    
    if (debouncedSymbol) {
      fetchForecast();
    }
  }, [debouncedSymbol]);

  // Handle module run from Sidebar
  const handleRunModule = useCallback(async (moduleId, sym) => {
    setIsLoading(true);
    setError(null);
    
    try {
      const json = await guardedFetchJson(`/api/run/${moduleId}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ symbol: sym || debouncedSymbol }),
      });
      
      const result = unwrap(json);
      const vc = result?.visual_context || json?.visual_context;
      
      if (vc && vc.traces && vc.traces.length > 0) {
        lastGoodVCRef.current = vc;
        setForecastVC(vc);
      }
      // If no visual_context, chart keeps last good frame
    } catch (err) {
      if (err.message !== "stale") {
        console.error("Module run error:", err);
        setError(err.message);
      }
    } finally {
      setIsLoading(false);
    }
  }, [debouncedSymbol]);

  // Handle deep analysis from Sidebar
  const handleRunDeepAnalysis = useCallback(async (moduleIds, sym) => {
    setIsLoading(true);
    setError(null);
    
    try {
      const json = await guardedFetchJson(`/api/deep_analysis/${sym || debouncedSymbol}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ modules: moduleIds }),
      });
      
      // Find first output with visual_context
      const outputs = json.outputs || [];
      for (const output of outputs) {
        const vc = output?.result?.visual_context || output?.visual_context;
        if (vc && vc.traces && vc.traces.length > 0) {
          lastGoodVCRef.current = vc;
          setForecastVC(vc);
          break;
        }
      }
      // If no visual_context found, chart keeps last good frame
    } catch (err) {
      if (err.message !== "stale") {
        console.error("Deep analysis error:", err);
        setError(err.message);
      }
    } finally {
      setIsLoading(false);
    }
  }, [debouncedSymbol]);

  // Expose handlers to parent (if using Sidebar)
  useEffect(() => {
    window.dashboardHandlers = {
      runModule: handleRunModule,
      runDeepAnalysis: handleRunDeepAnalysis,
    };
    return () => {
      delete window.dashboardHandlers;
    };
  }, [handleRunModule, handleRunDeepAnalysis]);

  // Notify parent of symbol changes
  useEffect(() => {
    if (onSymbolChange) {
      onSymbolChange(debouncedSymbol);
    }
  }, [debouncedSymbol, onSymbolChange]);

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center gap-4">
        <h1 className="text-3xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-green-400">
          📊 Dashboard
        </h1>
        <div className="flex-1 max-w-md">
          <TickerSearch
            defaultValue={symbol}
            onSelect={(newSymbol) => setSymbol(newSymbol.toUpperCase())}
          />
        </div>
      </div>

      {error && (
        <div className="p-4 bg-red-500/20 border border-red-500/50 rounded-lg text-red-300">
          Error: {error}
        </div>
      )}

      <div className="glassmorphic-panel rounded-xl p-4">
        <PlotlyGraph
          ticker={debouncedSymbol}
          forecastData={forecastVC ? { visual_context: forecastVC } : null}
          data={fallbackData}
          height={560}
        />
      </div>
    </div>
  );
}

export default Dashboard;
