/**
 * Quantum AI Cockpit — InsightFeed Component
 * 📊 Real-time AI recommendation feed with sentiment-based styling
 * ================================================================
 */

import React, { useState, useEffect } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { useWebSocketFeed } from "../hooks/useWebSocketFeed";

export default function InsightFeed({ symbol = "AAPL", maxItems = 5 }) {
  const [insights, setInsights] = useState([]);
  const [expandedIndex, setExpandedIndex] = useState(null);
  const [loading, setLoading] = useState(true);

  // WebSocket feed for real-time AI recommendations
  const { data: wsData, isConnected } = useWebSocketFeed("/ws/alerts");

  // Fetch initial insights and listen for updates
  // Use integrated endpoint /api/ai_recommender (or /api/ai_recommendation for backward compatibility)
  useEffect(() => {
    const fetchInsights = async () => {
      try {
        // Try integrated endpoint first, fallback to legacy
        let res = await fetch(`/api/ai_recommender/${symbol}`).catch(() => null);
        if (!res || !res.ok) {
          res = await fetch(`/api/ai_recommendation/${symbol}`).catch(() => null);
        }
        
        if (res && res.ok) {
          const data = await res.json();
          if (data && !data.error) {
            // Map standardized response format (module, trend, confidence, forecast, metrics, visual_context)
            // Support both new format and legacy format
            const insight = {
              ...data,
              // Map standardized fields
              symbol: data.symbol || symbol,
              sentiment: data.sentiment || data.trend || data.result?.sentiment || data.result?.trend || "neutral",
              confidence: data.confidence || data.result?.confidence || 0,
              summary: data.summary || data.result?.summary || data.insight || "",
              expected_move: data.expected_move || data.result?.expected_move || data.metrics?.expected_move || "0.0%",
              horizon: data.horizon || data.result?.horizon || "5 days",
              rationale: data.reasoning || data.rationale || data.result?.rationale || [],
              emoji: data.emoji || data.result?.emoji || "⚫🌀",
              timestamp: data.timestamp || new Date().toISOString(),
              id: data.timestamp || Date.now().toString(),
            };
            
            setInsights((prev) => {
              // Add to beginning, limit to maxItems
              const newInsights = [insight, ...prev.filter(i => i.id !== insight.id)].slice(0, maxItems);
              return newInsights;
            });
          }
        }
      } catch (err) {
        console.error("Insight fetch error:", err);
      } finally {
        setLoading(false);
      }
    };

    fetchInsights();
    
    // Refresh every 30 seconds
    const interval = setInterval(fetchInsights, 30000);
    return () => clearInterval(interval);
  }, [symbol, maxItems]);

  // Update insights from WebSocket
  useEffect(() => {
    if (wsData && wsData.type === "ai_recommendation" && wsData.symbol === symbol) {
      const insight = {
        ...wsData.data,
        timestamp: wsData.timestamp || new Date().toISOString(),
        id: wsData.timestamp || Date.now().toString(),
      };
      
      setInsights((prev) => {
        const newInsights = [insight, ...prev.filter((i) => i.id !== insight.id)].slice(0, maxItems);
        return newInsights;
      });
    }
  }, [wsData, symbol, maxItems]);

  const getSentimentColor = (sentiment) => {
    switch (sentiment) {
      case "bullish":
        return {
          border: "rgba(0, 255, 170, 0.5)",
          glow: "rgba(0, 255, 170, 0.3)",
          bg: "rgba(0, 255, 170, 0.05)",
        };
      case "bearish":
        return {
          border: "rgba(255, 0, 122, 0.5)",
          glow: "rgba(255, 0, 122, 0.3)",
          bg: "rgba(255, 0, 122, 0.05)",
        };
      default:
        return {
          border: "rgba(0, 209, 255, 0.5)",
          glow: "rgba(0, 209, 255, 0.3)",
          bg: "rgba(0, 209, 255, 0.05)",
        };
    }
  };

  const getSentimentGlowClass = (sentiment) => {
    switch (sentiment) {
      case "bullish":
        return "glow-green";
      case "bearish":
        return "glow-red";
      default:
        return "glow-cyan";
    }
  };

  if (loading && insights.length === 0) {
    return (
      <div className="glassmorphic-panel rounded-xl p-6">
        <div className="flex items-center justify-center h-32">
          <div className="w-8 h-8 border-2 border-cyan-500/30 border-t-cyan-500 rounded-full animate-spin" />
        </div>
      </div>
    );
  }

  if (insights.length === 0) {
    return (
      <div className="glassmorphic-panel rounded-xl p-6 text-center">
        <p className="text-gray-400 font-mono">No insights available for {symbol}</p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-xl font-bold font-orbitron text-cyan-400">
          🤖 AI Insight Feed
        </h3>
        <div className="flex items-center gap-2">
          <div
            className={`w-2 h-2 rounded-full ${
              isConnected ? "bg-green-500 animate-pulse" : "bg-red-500"
            }`}
          />
          <span className="text-xs text-gray-400 font-mono">
            {isConnected ? "Live" : "Offline"}
          </span>
        </div>
      </div>

      <AnimatePresence mode="popLayout">
        {insights.map((insight, index) => {
          const sentiment = insight.sentiment || "neutral";
          const colors = getSentimentColor(sentiment);
          const glowClass = getSentimentGlowClass(sentiment);
          const isExpanded = expandedIndex === index;

        return (
          <motion.div
            key={insight.id || index}
            initial={{ opacity: 0, y: 20, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, x: -20, scale: 0.95 }}
            transition={{ 
              type: "spring", 
              stiffness: 300, 
              damping: 25,
              duration: 0.5
            }}
            whileHover={{ scale: 1.02, y: -2 }}
            whileTap={{ scale: 0.98 }}
            className={`glassmorphic-panel rounded-lg p-4 border-2 cursor-pointer transition-all duration-1500 ${glowClass} gpu-accelerated`}
            style={{
              borderColor: colors.border,
              background: colors.bg,
              boxShadow: `0 0 20px ${colors.glow}, inset 0 0 10px ${colors.glow}20`,
              // OPTIMIZED v29: Gradient glow border
              borderImage: `linear-gradient(135deg, ${colors.border}, ${colors.glow}) 1`,
            }}
            onClick={() => setExpandedIndex(isExpanded ? null : index)}
          >
              {/* Header */}
              <div className="flex items-start justify-between mb-2">
                <div className="flex items-center gap-3">
                  <span className="text-3xl">{insight.emoji || "⚫🌀"}</span>
                  <div>
                    <div className="font-bold font-orbitron text-white">
                      {insight.symbol || symbol}
                    </div>
                    <div className="text-sm text-gray-400 font-mono">
                      {new Date(insight.timestamp).toLocaleTimeString()}
                    </div>
                  </div>
                </div>
                <div className="text-right">
                  <div
                    className={`text-lg font-bold font-orbitron ${
                      insight.expected_move?.startsWith("+")
                        ? "text-neon-green"
                        : insight.expected_move?.startsWith("-")
                        ? "text-red-400"
                        : "text-gray-400"
                    }`}
                  >
                    {insight.expected_move || "0.0%"}
                  </div>
                  <div className="text-xs text-gray-500 font-mono">
                    {insight.horizon || "5 days"}
                  </div>
                </div>
              </div>

              {/* Summary */}
              <p className="text-sm text-gray-300 font-mono mb-3">
                {insight.summary || "No summary available"}
              </p>

              {/* Expandable Rationale */}
              <AnimatePresence>
                {isExpanded && insight.rationale && insight.rationale.length > 0 && (
                  <motion.div
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: "auto", opacity: 1 }}
                    exit={{ height: 0, opacity: 0 }}
                    transition={{ duration: 0.3 }}
                    className="overflow-hidden"
                  >
                    <div className="pt-3 border-t border-gray-700/50">
                      <div className="text-xs text-gray-400 font-mono mb-2">
                        Rationale:
                      </div>
                      <ul className="space-y-1">
                        {insight.rationale.map((point, idx) => (
                          <li
                            key={idx}
                            className="text-xs text-gray-300 font-mono flex items-start gap-2"
                          >
                            <span className="text-cyan-500 mt-1">•</span>
                            <span>{point}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>

              {/* Footer with consensus and confidence */}
              <div className="mt-3 flex items-center justify-between text-xs font-mono">
                <div>
                  <span className="text-gray-400">Consensus: </span>
                  <span className="text-cyan-400">
                    {((insight.consensus || 0) * 100).toFixed(1)}%
                  </span>
                </div>
                <div>
                  <span className="text-gray-400">Confidence: </span>
                  <span className="text-gold">
                    {((insight.confidence || 0) * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="text-gray-500">
                  {isExpanded ? "▼" : "▶"} Click to {isExpanded ? "collapse" : "expand"}
                </div>
              </div>
          </motion.div>
        );
        })}
      </AnimatePresence>
    </div>
  );
}
