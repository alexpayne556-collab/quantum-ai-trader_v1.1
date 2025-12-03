/**
 * Quantum AI Cockpit — Deep Analysis Lab Page (Enhanced)
 * 🧠 Multi-module fusion orchestrator page with advanced visuals
 * ==============================================================
 */

import React, { useState, useEffect, useRef, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import PlotlyGraph from "../components/PlotlyGraph";
import SearchBar from "../components/SearchBar";
import ParticleBackground from "../components/ParticleBackground";
import InsightFeed from "../components/InsightFeed";
import { useWebSocketFeed } from "../hooks/useWebSocketFeed";
import "../styles/animations.css";

export default function DeepAnalysisLab() {
  const [symbol, setSymbol] = useState("AAPL");
  const [timeframe, setTimeframe] = useState("1D");
  const [analysisData, setAnalysisData] = useState(null);
  const [aiInsight, setAiInsight] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [expandedRationale, setExpandedRationale] = useState(false);
  const [performanceMetrics, setPerformanceMetrics] = useState({
    fetchTime: 0,
    renderTime: 0,
    fps: 0,
  });

  // Request cancellation
  const abortControllerRef = useRef(null);
  const fetchStartTimeRef = useRef(0);

  // WebSocket feed for real-time updates
  const { data: wsData, isConnected } = useWebSocketFeed("/ws/alerts");
  
  // OPTIMIZED v29: Real-time sentiment state for dynamic visual binding
  const [sentiment, setSentiment] = useState("neutral");
  const [volatility, setVolatility] = useState(0);
  
  // Extract sentiment and volatility from analysis data and WebSocket
  // Support both standardized format (trend, confidence) and legacy format
  useEffect(() => {
    if (aiInsight) {
      // Try standardized format first
      const sentimentValue = aiInsight.sentiment || aiInsight.trend || aiInsight.result?.sentiment || aiInsight.result?.trend;
      if (sentimentValue) {
        setSentiment(sentimentValue.toLowerCase());
      }
    }
    if (analysisData) {
      // Extract volatility from risk data or visual_context
      const visualContext = analysisData.visual_context || analysisData.fusion_forecast?.visual_context;
      if (visualContext?.volatility !== undefined) {
        setVolatility(visualContext.volatility * 100);
      } else if (analysisData.risk) {
        const riskData = analysisData.risk.risk_score || analysisData.risk.result?.risk_score || analysisData.risk.metrics?.risk_score || 0;
        setVolatility((1 - riskData) * 100); // Convert risk to volatility percentage
      }
    }
  }, [aiInsight, analysisData]);

  // Listen for global ticker change events
  useEffect(() => {
    const handleTickerChange = (event) => {
      const newTicker = event.detail?.ticker;
      if (newTicker && newTicker !== symbol) {
        setSymbol(newTicker);
      }
    };

    window.addEventListener("SET_TICKER", handleTickerChange);
    return () => window.removeEventListener("SET_TICKER", handleTickerChange);
  }, [symbol]);

  // Fetch deep analysis data from API with cancellation support
  const fetchAnalysis = useCallback(async (ticker, signal) => {
    setLoading(true);
    setError(null);
    fetchStartTimeRef.current = performance.now();

    try {
      // OPTIMIZED: Try unified /api/deep_analysis endpoint first (uses all integrated modules)
      // Fallback to individual endpoints if needed
      let deepAnalysisData = null;
      try {
        const deepRes = await fetch(`/api/deep_analysis/${ticker}`, { signal });
        if (deepRes.ok) {
          deepAnalysisData = await deepRes.json();
        }
      } catch (e) {
        console.log("Deep analysis endpoint not available, using individual endpoints");
      }

      // If deep analysis returns data, use it; otherwise fetch individual endpoints
      if (deepAnalysisData && !deepAnalysisData.error) {
        // Extract data from deep analysis response (already aggregated)
        // Support both new format (forecast) and legacy (fusion_forecast)
        const fusionData = deepAnalysisData.forecast || deepAnalysisData.fusion_forecast || deepAnalysisData.modules?.forecast || deepAnalysisData.modules?.fusion_forecast;
        const patternData = deepAnalysisData.patterns || deepAnalysisData.pattern_detection || deepAnalysisData.modules?.pattern_detection;
        const sentimentData = deepAnalysisData.sentiment || deepAnalysisData.modules?.sentiment;
        const riskData = deepAnalysisData.risk || deepAnalysisData.modules?.risk;
        const aiData = deepAnalysisData.ai_recommender || deepAnalysisData.modules?.ai_recommender || deepAnalysisData.insights || deepAnalysisData.recommendation;
        
        // Map standardized response format (backend now returns: module, trend, confidence, forecast, metrics, visual_context)
        const chartData = {
          timestamps: fusionData?.metrics?.timestamps || fusionData?.timestamps || fusionData?.result?.timestamps || [],
          open: fusionData?.metrics?.open || fusionData?.open || fusionData?.result?.open || [],
          high: fusionData?.metrics?.high || fusionData?.high || fusionData?.result?.high || [],
          low: fusionData?.metrics?.low || fusionData?.low || fusionData?.result?.low || [],
          close: fusionData?.metrics?.close || fusionData?.close || fusionData?.result?.close || [],
          volume: fusionData?.metrics?.volume || fusionData?.volume || fusionData?.result?.volume || [],
          ema: fusionData?.metrics?.ema || fusionData?.ema || fusionData?.result?.ema || {},
          rsi: Array.isArray(fusionData?.metrics?.rsi) ? fusionData.metrics.rsi : (typeof fusionData?.metrics?.rsi === 'number' ? [fusionData.metrics.rsi] : null),
          macd: fusionData?.metrics?.macd || fusionData?.macd || fusionData?.result?.macd || null,
          forecast: fusionData?.forecast_days || fusionData?.forecast || null,
        };
        
        const aggregated = {
          symbol: ticker,
          timestamp: deepAnalysisData.timestamp || new Date().toISOString(),
          forecast: fusionData?.forecast || fusionData?.metrics?.forecast || fusionData?.result?.forecast || fusionData || null,
          fusion_forecast: fusionData, // Keep for backward compatibility
          pattern_detection: patternData,
          sentiment: sentimentData,
          risk: riskData,
          // Chart data extracted above
          ...chartData,
          patterns: patternData?.metrics?.patterns || patternData?.patterns || patternData?.result?.patterns || [],
          red_detection: riskData?.metrics?.red_detection || riskData?.red_detection || riskData?.result?.red_detection || [],
          visual_context: deepAnalysisData.visual_context || fusionData?.visual_context || {},
        };

        const fetchTime = performance.now() - fetchStartTimeRef.current;
        setPerformanceMetrics((prev) => ({ ...prev, fetchTime }));
        setAnalysisData(aggregated);
        if (aiData && !aiData.error) {
          setAiInsight(aiData);
        }
        return;
      }

      // Fallback: Call individual integrated endpoints concurrently
      // Use current endpoints: /api/forecast (replaces fusion_forecast), /api/run/{module}/{symbol}
      const apiCalls = [
        fetch(`/api/forecast/${ticker}`, { signal }).catch(() => null), // Uses fusior_forecast
        fetch(`/api/run/risk_engine/${ticker}`, { signal }).catch(() => null),
        fetch(`/api/run/ai_recommender/${ticker}`, { signal }).catch(() => null), // Includes sentiment
      ];

      const [forecastRes, riskRes, aiRes] = await Promise.all(apiCalls);

      // Check if request was aborted
      if (signal?.aborted) {
        return;
      }

      const forecastJson = forecastRes?.ok ? await forecastRes.json() : null;
      const riskJson = riskRes?.ok ? await riskRes.json() : null;
      const aiJson = aiRes?.ok ? await aiRes.json() : null;
      
      // Extract data from API responses (handle both /api/run/{module} and direct endpoints)
      const fusionData = forecastJson?.result || forecastJson;
      const riskData = riskJson?.result || riskJson;
      const aiData = aiJson?.result || aiJson;
      
      // Extract sentiment from AI recommendation (includes sentiment analysis)
      const sentimentData = aiData?.sentiment || aiData?.result?.sentiment || null;
      
      // Extract pattern data from deep_analysis if available, otherwise from forecast signals
      const patternData = analysisData?.patterns || analysisData?.modules?.pattern_detection || 
                         (fusionData?.signals ? { signals: fusionData.signals } : null);
      
      // Set AI insight data
      if (aiData && !aiData.error) {
        setAiInsight(aiData);
      }

      // Check for errors in responses (only throw if all failed)
      const errors = [
        forecastJson?.error || fusionData?.error,
        riskJson?.error || riskData?.error,
        aiJson?.error || aiData?.error,
      ].filter(Boolean);
      
      // Only throw if we have no data at all
      if (errors.length > 0 && !fusionData && !riskData && !aiData) {
        throw new Error(`API errors: ${errors.join(", ")}`);
      }

      // Aggregate data with standardized response format mapping
      // Backend now returns: module, trend, confidence, forecast, metrics, visual_context
      const aggregated = {
        symbol: ticker,
        timestamp: new Date().toISOString(),
        forecast: fusionData?.forecast || fusionData?.metrics?.forecast || fusionData?.result?.forecast || fusionData || null,
        fusion_forecast: fusionData, // Keep for backward compatibility
        pattern_detection: patternData,
        sentiment: sentimentData,
        risk: riskData,
        // Extract chart data with fallback to legacy format
        // Support both new standardized format (metrics.forecast) and legacy (forecast, result.forecast)
        timestamps: fusionData?.metrics?.timestamps || fusionData?.timestamps || fusionData?.result?.timestamps || [],
        open: fusionData?.metrics?.open || fusionData?.open || fusionData?.result?.open || [],
        high: fusionData?.metrics?.high || fusionData?.high || fusionData?.result?.high || [],
        low: fusionData?.metrics?.low || fusionData?.low || fusionData?.result?.low || [],
        close: fusionData?.metrics?.close || fusionData?.close || fusionData?.result?.close || [],
        ema: fusionData?.metrics?.ema || fusionData?.ema || fusionData?.result?.ema || {},
        patterns: patternData?.metrics?.patterns || patternData?.patterns || patternData?.result?.patterns || [],
        red_detection: riskData?.metrics?.red_detection || riskData?.red_detection || riskData?.result?.red_detection || [],
        rsi: fusionData?.metrics?.rsi || fusionData?.rsi || fusionData?.result?.rsi || null,
        macd: fusionData?.metrics?.macd || fusionData?.macd || fusionData?.result?.macd || null,
        // Extract visual context for ribbon/breathing animations
        visual_context: fusionData?.visual_context || patternData?.visual_context || {},
      };

      const fetchTime = performance.now() - fetchStartTimeRef.current;
      setPerformanceMetrics((prev) => ({ ...prev, fetchTime }));

      setAnalysisData(aggregated);
      
      // Extract AI insight from forecast or deep analysis
      if (fusionData?.ai_recommendation) {
        const aiRec = fusionData.ai_recommendation;
        setAiInsight({
          sentiment: aiRec.sentiment || (fusionData.trend || "neutral").toLowerCase(),
          emoji: aiRec.emoji || (fusionData.trend === "bullish" ? "🟢🚀" : fusionData.trend === "bearish" ? "🔴⚠️" : "⚫🌀"),
          confidence: aiRec.confidence || fusionData.confidence || 0,
          recommendation: aiRec.action || aiRec.recommendation || "HOLD",
          rationale: aiRec.rationale || "",
        });
      } else if (aiData && !aiData.error) {
        setAiInsight({
          sentiment: aiData.sentiment || aiData.result?.sentiment || "neutral",
          emoji: aiData.emoji || (aiData.sentiment === "bullish" ? "🟢🚀" : aiData.sentiment === "bearish" ? "🔴⚠️" : "⚫🌀"),
          confidence: aiData.confidence || aiData.result?.confidence || 0,
          recommendation: aiData.action || aiData.recommendation || aiData.result?.action || "HOLD",
          rationale: aiData.rationale || aiData.result?.rationale || "",
        });
      } else {
        setAiInsight(null);
      }
    } catch (err) {
      if (err.name === "AbortError") {
        // Request was cancelled, ignore
        return;
      }
      console.error("Deep analysis fetch error:", err);
      setError(err.message);
    } finally {
      if (!signal?.aborted) {
        setLoading(false);
      }
    }
  }, []);

  // Force initial data fetch on mount
  useEffect(() => {
    if (!analysisData) {
      const abortController = new AbortController();
      abortControllerRef.current = abortController;
      fetchAnalysis(symbol, abortController.signal);
    }
  }, []);

  // Effect to fetch data when symbol or timeframe changes
  useEffect(() => {
    // Cancel previous request
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }

    // Create new abort controller
    abortControllerRef.current = new AbortController();

    if (symbol) {
      fetchAnalysis(symbol, abortControllerRef.current.signal);

      // Refresh every 30 seconds
      const interval = setInterval(() => {
        if (symbol) {
          fetchAnalysis(symbol, abortControllerRef.current.signal);
        }
      }, 30000);

      return () => {
        clearInterval(interval);
        if (abortControllerRef.current) {
          abortControllerRef.current.abort();
        }
      };
    }
  }, [symbol, timeframe, fetchAnalysis]);

  // Export snapshot function
  const exportSnapshot = useCallback(async (format = "json") => {
    if (!analysisData) return;

    const snapshot = {
      symbol: analysisData.symbol,
      timestamp: analysisData.timestamp,
      timeframe,
      fusion_forecast: analysisData.fusion_forecast,
      pattern_detection: analysisData.pattern_detection,
      sentiment: analysisData.sentiment,
      risk: analysisData.risk,
      performance: performanceMetrics,
    };

    if (format === "json") {
      const blob = new Blob([JSON.stringify(snapshot, null, 2)], {
        type: "application/json",
      });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `deep_analysis_${symbol}_${new Date().toISOString().split("T")[0]}.json`;
      a.click();
      URL.revokeObjectURL(url);
    } else if (format === "pdf") {
      // PDF export would require a library like jsPDF
      console.warn("PDF export not yet implemented");
    }
  }, [analysisData, timeframe, symbol, performanceMetrics]);

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      className="p-6 space-y-6 min-h-screen relative"
    >
      {/* Subtle particle background */}
      <ParticleBackground particleCount={15} />
      {/* Header with SearchBar and Controls */}
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
        <motion.h1
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          className="text-3xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-green-400"
        >
          🧠 Deep Analysis Lab
        </motion.h1>

        <div className="flex items-center gap-4 flex-wrap">
          <SearchBar onTickerChange={setSymbol} initialTicker={symbol} />

          {/* Timeframe Selector */}
          <div className="flex items-center gap-2 glassmorphic-panel rounded-lg p-2">
            <label className="text-sm text-gray-400 font-mono">Timeframe:</label>
            <select
              value={timeframe}
              onChange={(e) => setTimeframe(e.target.value)}
              className="bg-[rgba(15,15,15,0.9)] border border-cyan-500/30 rounded px-3 py-1 text-white text-sm font-mono focus:outline-none focus:border-cyan-500"
            >
              <option value="1D">1D</option>
              <option value="1W">1W</option>
              <option value="1M">1M</option>
            </select>
          </div>

          {/* Export Button */}
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={() => exportSnapshot("json")}
            className="px-4 py-2 bg-cyan-500/20 border border-cyan-500/50 rounded-lg text-cyan-400 text-sm font-mono hover:bg-cyan-500/30 transition-colors"
            disabled={!analysisData}
          >
            📥 Export Snapshot
          </motion.button>

          {/* WebSocket Status */}
          <div className="flex items-center gap-2">
            <div
              className={`w-3 h-3 rounded-full ${
                isConnected ? "bg-green-500 animate-pulse" : "bg-red-500"
              }`}
              title={isConnected ? "WebSocket Connected" : "WebSocket Disconnected"}
            />
            <span className="text-xs text-gray-400 font-mono">
              {isConnected ? "Live" : "Offline"}
            </span>
          </div>
        </div>
      </div>

      {/* Main Chart with Enhanced Loading State */}
      <AnimatePresence mode="wait">
        {loading ? (
          <motion.div
            key="loading"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="glassmorphic-panel rounded-xl p-8 text-center animate-gradient-shimmer"
            style={{ minHeight: "700px" }}
          >
            <div className="flex flex-col items-center justify-center h-full">
              <motion.div
                animate={{ rotate: 360 }}
                transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
                className="w-16 h-16 border-4 border-cyan-500/30 border-t-cyan-500 rounded-full mb-4"
              />
              <div className="text-neon-green text-lg font-orbitron mb-2">
                Loading Deep Analysis...
              </div>
              <div className="text-gray-400 text-sm font-mono">Fetching data for {symbol}</div>
            </div>
          </motion.div>
        ) : error ? (
          <motion.div
            key="error"
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.95 }}
            className="glassmorphic-panel rounded-xl p-8 text-center border border-red-500/50"
          >
            <div className="text-red-400 text-lg font-orbitron mb-2">Error</div>
            <div className="text-gray-300 font-mono">{error}</div>
          </motion.div>
               ) : analysisData ? (
                 <motion.div
                   key="chart"
                   initial={{ opacity: 0, y: 20 }}
                   animate={{ 
                     opacity: 1, 
                     y: 0,
                     // OPTIMIZED v29: Dynamic border color based on sentiment
                     borderColor: sentiment === "bullish" 
                       ? "rgba(0, 255, 170, 0.5)" 
                       : sentiment === "bearish" 
                       ? "rgba(255, 0, 122, 0.5)" 
                       : "rgba(0, 209, 255, 0.5)"
                   }}
                   exit={{ opacity: 0, y: -20 }}
                   transition={{ duration: 1.5, ease: "easeOut" }}
                   className={`glassmorphic-panel rounded-xl p-4 relative z-10 gpu-accelerated ${
                     sentiment === "bullish" 
                       ? "animate-breathing-green" 
                       : sentiment === "bearish" 
                       ? "animate-breathing-red" 
                       : "animate-breathing-cyan"
                   }`}
                   style={{
                     borderWidth: "2px",
                     borderStyle: "solid",
                     // OPTIMIZED v29: Volatility-based shimmer effect
                     background: volatility > 50
                       ? "linear-gradient(135deg, rgba(0,255,170,0.05) 0%, rgba(0,51,34,0.2) 50%, rgba(0,255,170,0.05) 100%)"
                       : "radial-gradient(circle at center, rgba(0,255,255,0.08), rgba(0,0,0,0.6))",
                     boxShadow: sentiment === "bullish"
                       ? "0 0 30px rgba(0, 255, 170, 0.3), 0 0 60px rgba(0, 255, 170, 0.15), inset 0 0 20px rgba(0, 255, 170, 0.1)"
                       : sentiment === "bearish"
                       ? "0 0 30px rgba(255, 0, 122, 0.3), 0 0 60px rgba(255, 0, 122, 0.15), inset 0 0 20px rgba(255, 0, 122, 0.1)"
                       : "0 0 30px rgba(0, 209, 255, 0.3), 0 0 60px rgba(0, 209, 255, 0.15), inset 0 0 20px rgba(0, 209, 255, 0.1)",
                   }}
                 >
                   <PlotlyGraph 
                     ticker={symbol} 
                     data={analysisData && (analysisData.timestamps?.length > 0 || analysisData.close?.length > 0) ? {
                       timestamps: analysisData.timestamps || [],
                       open: analysisData.open || [],
                       high: analysisData.high || [],
                       low: analysisData.low || [],
                       close: analysisData.close || [],
                       volume: analysisData.volume || [],
                       ema: analysisData.ema || {},
                       rsi: Array.isArray(analysisData.rsi) ? analysisData.rsi : null,
                       macd: analysisData.macd || null,
                       forecast: analysisData.forecast || null,
                     } : null}
                     forecastData={analysisData?.fusion_forecast || analysisData?.forecast || null}
                     aiInsight={aiInsight}
                     height={700} 
                     theme="dark"
                     showLoading={loading}
                   />
                 </motion.div>
               ) : null}
             </AnimatePresence>

      {/* AI Insight Panel */}
      {aiInsight && !error && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
          className={`glassmorphic-panel rounded-xl p-6 border-2 relative z-10 ${
            aiInsight.sentiment === "bullish"
              ? "border-neon-green/50 neon-glow"
              : aiInsight.sentiment === "bearish"
              ? "border-red-500/50"
              : "border-cyan-500/50 neon-glow-blue"
          }`}
          style={{
            boxShadow:
              aiInsight.sentiment === "bullish"
                ? "0 0 20px rgba(0,255,170,0.2)"
                : aiInsight.sentiment === "bearish"
                ? "0 0 20px rgba(255,0,122,0.2)"
                : "0 0 20px rgba(0,209,255,0.2)",
          }}
        >
          <div className="flex items-start justify-between mb-4">
            <div>
              <h3 className="text-2xl font-bold font-orbitron mb-2 flex items-center gap-2">
                <span>{aiInsight.emoji}</span>
                <span>AI Insight</span>
              </h3>
              <p className="text-lg text-gray-300 font-mono">{aiInsight.summary}</p>
            </div>
            <div className="text-right">
              <div className="text-sm text-gray-400 font-mono">Expected Move</div>
              <div
                className={`text-2xl font-bold font-orbitron ${
                  aiInsight.expected_move.startsWith("+")
                    ? "text-neon-green"
                    : aiInsight.expected_move.startsWith("-")
                    ? "text-red-400"
                    : "text-gray-400"
                }`}
              >
                {aiInsight.expected_move}
              </div>
              <div className="text-xs text-gray-500 font-mono mt-1">Horizon: {aiInsight.horizon}</div>
            </div>
          </div>

          {/* Rationale List */}
          <div className="mt-4">
            <button
              onClick={() => setExpandedRationale(!expandedRationale)}
              className="text-sm text-cyan-400 hover:text-cyan-300 font-mono mb-2 flex items-center gap-2"
            >
              <span>{expandedRationale ? "▼" : "▶"}</span>
              <span>Rationale ({aiInsight.rationale?.length || 0} points)</span>
            </button>
            {expandedRationale && aiInsight.rationale && (
              <ul className="space-y-2 mt-2">
                {aiInsight.rationale.map((point, idx) => (
                  <motion.li
                    key={idx}
                    initial={{ opacity: 0, x: -10 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: idx * 0.1 }}
                    className="text-sm text-gray-300 font-mono flex items-start gap-2"
                  >
                    <span className="text-cyan-500 mt-1">•</span>
                    <span>{point}</span>
                  </motion.li>
                ))}
              </ul>
            )}
          </div>

          {/* Consensus and Confidence */}
          <div className="mt-4 flex items-center gap-4 text-xs font-mono">
            <div>
              <span className="text-gray-400">Consensus: </span>
              <span className="text-cyan-400">{(aiInsight.consensus * 100).toFixed(1)}%</span>
            </div>
            <div>
              <span className="text-gray-400">Confidence: </span>
              <span className="text-gold">{(aiInsight.confidence * 100).toFixed(1)}%</span>
            </div>
          </div>
        </motion.div>
      )}

      {/* AI Insight Feed Panel */}
      {aiInsight && !error && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.6 }}
          className="relative z-10"
        >
          <InsightFeed symbol={symbol} maxItems={3} />
        </motion.div>
      )}

      {/* Analysis Cards with Enhanced Styling */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 relative z-10">
        {/* Fusion Forecast Card */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="glassmorphic-panel rounded-xl p-4 border border-cyan-500/20 hover:border-cyan-500/40 transition-all duration-300 hover:shadow-neon"
        >
          <h3 className="text-electric-blue font-semibold mb-2 font-orbitron">
            Forecast
          </h3>
          {analysisData?.forecast || analysisData?.fusion_forecast ? (
            <div className="text-sm space-y-1 font-mono">
              <div>
                Direction:{" "}
                <span className="text-neon-green">
                  {analysisData.forecast?.trend || analysisData.fusion_forecast?.trend ||
                    analysisData.forecast?.direction || analysisData.fusion_forecast?.direction ||
                    analysisData.forecast?.result?.trend || analysisData.fusion_forecast?.result?.direction ||
                    "N/A"}
                </span>
              </div>
              <div>
                Confidence:{" "}
                <span className="text-gold">
                  {analysisData.forecast?.confidence || analysisData.fusion_forecast?.confidence ||
                    analysisData.forecast?.result?.confidence || analysisData.fusion_forecast?.result?.confidence
                    ? `${(
                        (analysisData.forecast?.confidence || analysisData.fusion_forecast?.confidence ||
                          analysisData.forecast?.result?.confidence || analysisData.fusion_forecast?.result?.confidence ||
                          0) * 100
                      ).toFixed(1)}%`
                    : "N/A"}
                </span>
              </div>
            </div>
          ) : (
            <div className="text-gray-400 font-mono">No data</div>
          )}
        </motion.div>

        {/* Pattern Detection Card */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="glassmorphic-panel rounded-xl p-4 border border-magenta/20 hover:border-magenta/40 transition-all duration-300 hover:shadow-neon"
        >
          <h3 className="text-magenta font-semibold mb-2 font-orbitron">Pattern Detection</h3>
          {analysisData?.pattern_detection ? (
            <div className="text-sm space-y-1 font-mono">
              <div>
                Patterns:{" "}
                <span className="text-neon-green">
                  {analysisData.pattern_detection.patterns?.length ||
                    analysisData.pattern_detection.result?.patterns?.length ||
                    0}
                </span>
              </div>
              <div>
                Signal:{" "}
                <span className="text-gold">
                  {analysisData.pattern_detection.signal ||
                    analysisData.pattern_detection.result?.signal ||
                    "N/A"}
                </span>
              </div>
            </div>
          ) : (
            <div className="text-gray-400 font-mono">No data</div>
          )}
        </motion.div>

        {/* Sentiment Card */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="glassmorphic-panel rounded-xl p-4 border border-cyan-500/20 hover:border-cyan-500/40 transition-all duration-300 hover:shadow-neon"
        >
          <h3 className="text-cyan font-semibold mb-2 font-orbitron">Sentiment</h3>
          {analysisData?.sentiment ? (
            <div className="text-sm space-y-1 font-mono">
              <div>
                Score:{" "}
                <span className="text-neon-green">
                  {analysisData.sentiment.sentiment_score ||
                    analysisData.sentiment.result?.sentiment_score
                    ? (
                        analysisData.sentiment.sentiment_score ||
                        analysisData.sentiment.result?.sentiment_score ||
                        0
                      ).toFixed(2)
                    : "N/A"}
                </span>
              </div>
              <div>
                Label:{" "}
                <span className="text-gold">
                  {analysisData.sentiment.sentiment_label ||
                    analysisData.sentiment.result?.sentiment_label ||
                    "N/A"}
                </span>
              </div>
            </div>
          ) : (
            <div className="text-gray-400 font-mono">No data</div>
          )}
        </motion.div>

        {/* Risk Card */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
          className="glassmorphic-panel rounded-xl p-4 border border-red-500/20 hover:border-red-500/40 transition-all duration-300 hover:shadow-neon"
        >
          <h3 className="text-red-400 font-semibold mb-2 font-orbitron">Risk Assessment</h3>
          {analysisData?.risk ? (
            <div className="text-sm space-y-1 font-mono">
              <div>
                Risk Score:{" "}
                <span className="text-magenta">
                  {analysisData.risk.risk_score || analysisData.risk.result?.risk_score
                    ? (
                        analysisData.risk.risk_score ||
                        analysisData.risk.result?.risk_score ||
                        0
                      ).toFixed(2)
                    : "N/A"}
                </span>
              </div>
              <div>
                Volatility:{" "}
                <span className="text-gold">
                  {analysisData.risk.volatility || analysisData.risk.result?.volatility
                    ? (
                        analysisData.risk.volatility ||
                        analysisData.risk.result?.volatility ||
                        0
                      ).toFixed(2)
                    : "N/A"}
                </span>
              </div>
            </div>
          ) : (
            <div className="text-gray-400 font-mono">No data</div>
          )}
        </motion.div>
      </div>

      {/* Performance Metrics (Optional, can be hidden) */}
      {performanceMetrics.fetchTime > 0 && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="text-xs text-gray-500 font-mono text-right"
        >
          Fetch: {performanceMetrics.fetchTime.toFixed(0)}ms
        </motion.div>
      )}
    </motion.div>
  );
}
