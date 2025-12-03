/**
 * Quantum AI Cockpit — Watchlist Panel Component
 * 👁️ Real-time watchlist management with WebSocket updates
 * ========================================================
 */

import React, { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { useWebSocketFeed } from "../hooks/useWebSocketFeed";
import PlotlyGraph from "./PlotlyGraph";

// Use relative URL for API calls (Vite proxy will handle routing)
const API_BASE = ""; // Backend API base URL (uses Vite proxy)

export default function WatchlistPanel() {
  const [watchlist, setWatchlist] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [showAddForm, setShowAddForm] = useState(false);
  const [expandedSymbols, setExpandedSymbols] = useState(new Set()); // Track which symbols have graphs expanded
  const [addFormData, setAddFormData] = useState({
    symbol: "",
    notes: "",
  });

  // WebSocket connection for real-time updates
  const { data: wsData, isConnected } = useWebSocketFeed("/ws/watchlist");

  // Load initial watchlist data
  useEffect(() => {
    loadWatchlist();
  }, []);

  // Update watchlist when WebSocket data arrives
  useEffect(() => {
    if (wsData && wsData.type === "watchlist_update" && wsData.data) {
      setWatchlist(wsData.data);
      setLoading(false);
    }
  }, [wsData]);

  const loadWatchlist = async () => {
    try {
      setLoading(true);
      const response = await fetch(`${API_BASE}/api/watchlist`);
      if (!response.ok) {
        throw new Error("Failed to load watchlist");
      }
      const data = await response.json();
      setWatchlist(data);
      setError(null);
    } catch (err) {
      console.error("Error loading watchlist:", err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleAddTicker = async (e) => {
    e.preventDefault();
    try {
      const response = await fetch(`${API_BASE}/api/watchlist/add`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          symbol: addFormData.symbol.toUpperCase(),
          notes: addFormData.notes || "",
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || "Failed to add ticker");
      }

      const result = await response.json();
      setWatchlist(result.watchlist);
      setShowAddForm(false);
      setAddFormData({ symbol: "", notes: "" });
      setError(null);
    } catch (err) {
      console.error("Error adding ticker:", err);
      setError(err.message);
    }
  };

  const handleRemoveTicker = async (symbol) => {
    if (!window.confirm(`Remove ${symbol} from watchlist?`)) {
      return;
    }

    try {
      const response = await fetch(`${API_BASE}/api/watchlist/remove`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ symbol }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || "Failed to remove ticker");
      }

      const result = await response.json();
      setWatchlist(result.watchlist);
      setError(null);
    } catch (err) {
      console.error("Error removing ticker:", err);
      setError(err.message);
    }
  };

  const formatCurrency = (value) => {
    return new Intl.NumberFormat("en-US", {
      style: "currency",
      currency: "USD",
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    }).format(value);
  };

  const formatPercent = (value) => {
    return `${value >= 0 ? "+" : ""}${value.toFixed(2)}%`;
  };

  if (loading && !watchlist) {
    return (
      <div className="glassmorphic-panel rounded-xl p-8 text-center">
        <div className="animate-pulse text-electric-blue">Loading watchlist...</div>
      </div>
    );
  }

  if (error && !watchlist) {
    return (
      <div className="glassmorphic-panel rounded-xl p-8 text-center">
        <div className="text-magenta">Error: {error}</div>
        <button
          onClick={loadWatchlist}
          className="mt-4 px-4 py-2 bg-electric-blue text-white rounded hover:bg-electric-blue/80"
        >
          Retry
        </button>
      </div>
    );
  }

  // Support both old format (tickers array) and new format (symbols array with analysis)
  const tickers = watchlist?.symbols || watchlist?.tickers || [];
  
  const toggleSymbolExpand = (symbol) => {
    setExpandedSymbols(prev => {
      const newSet = new Set(prev);
      if (newSet.has(symbol)) {
        newSet.delete(symbol);
      } else {
        newSet.add(symbol);
      }
      return newSet;
    });
  };

  return (
    <div className="space-y-6">
      {/* Watchlist Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="glassmorphic-panel rounded-xl p-6"
      >
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-2xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-green-400">
            👁️ Watchlist
          </h2>
          <div className="flex items-center gap-2">
            <div
              className={`w-3 h-3 rounded-full ${
                isConnected ? "bg-neon-green animate-pulse" : "bg-magenta"
              }`}
            />
            <span className="text-sm text-gray-400">
              {isConnected ? "Live" : "Offline"}
            </span>
          </div>
        </div>

        <div className="text-sm text-gray-400">
          {tickers.length} ticker{tickers.length !== 1 ? "s" : ""} in watchlist
        </div>
      </motion.div>

      {/* Watchlist Table */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="glassmorphic-panel rounded-xl p-6"
      >
        <div className="flex justify-between items-center mb-4">
          <h3 className="text-xl font-semibold text-electric-blue">Tickers</h3>
          <button
            onClick={() => setShowAddForm(!showAddForm)}
            className="px-4 py-2 bg-neon-green/20 text-neon-green border border-neon-green/40 rounded-lg hover:bg-neon-green/30 transition-all"
          >
            + Add Ticker
          </button>
        </div>

        {/* Add Ticker Form */}
        <AnimatePresence>
          {showAddForm && (
            <motion.form
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              exit={{ opacity: 0, height: 0 }}
              onSubmit={handleAddTicker}
              className="mb-4 p-4 bg-[rgba(0,209,255,0.1)] rounded-lg border border-electric-blue/20"
            >
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <input
                  type="text"
                  placeholder="Symbol (e.g., AAPL)"
                  value={addFormData.symbol}
                  onChange={(e) =>
                    setAddFormData({ ...addFormData, symbol: e.target.value })
                  }
                  className="px-3 py-2 bg-[rgba(15,15,15,0.8)] border border-cyan-500/20 rounded text-white placeholder-gray-500"
                  required
                />
                <input
                  type="text"
                  placeholder="Notes (optional)"
                  value={addFormData.notes}
                  onChange={(e) =>
                    setAddFormData({ ...addFormData, notes: e.target.value })
                  }
                  className="px-3 py-2 bg-[rgba(15,15,15,0.8)] border border-cyan-500/20 rounded text-white placeholder-gray-500"
                />
              </div>
              <div className="flex gap-2 mt-4">
                <button
                  type="submit"
                  className="px-4 py-2 bg-neon-green/20 text-neon-green border border-neon-green/40 rounded-lg hover:bg-neon-green/30 transition-all"
                >
                  Add
                </button>
                <button
                  type="button"
                  onClick={() => setShowAddForm(false)}
                  className="px-4 py-2 bg-magenta/20 text-magenta border border-magenta/40 rounded-lg hover:bg-magenta/30 transition-all"
                >
                  Cancel
                </button>
              </div>
            </motion.form>
          )}
        </AnimatePresence>

        {/* Tickers Table */}
        <div className="overflow-x-auto">
          <table className="w-full text-sm text-left text-gray-300">
            <thead className="text-gray-400 uppercase border-b border-cyan-500/20">
              <tr>
                <th className="py-3 px-4">Symbol</th>
                <th className="py-3 px-4">Current Price</th>
                <th className="py-3 px-4">Price Change</th>
                <th className="py-3 px-4">Price Change %</th>
                <th className="py-3 px-4">Notes</th>
                <th className="py-3 px-4">Added Date</th>
                <th className="py-3 px-4"></th>
              </tr>
            </thead>
            <tbody>
              <AnimatePresence>
                {tickers.map((ticker, index) => {
                  const symbol = ticker.symbol;
                  const isExpanded = expandedSymbols.has(symbol);
                  const analysis = ticker.analysis;
                  
                  return (
                    <React.Fragment key={symbol}>
                      <motion.tr
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        exit={{ opacity: 0, x: 20 }}
                        transition={{ delay: index * 0.05 }}
                        className="border-b border-cyan-500/10 hover:bg-[rgba(0,255,170,0.03)] transition-all cursor-pointer"
                        onClick={() => toggleSymbolExpand(symbol)}
                      >
                        <td className="py-3 px-4 font-semibold text-neon-green flex items-center gap-2">
                          <span>{isExpanded ? "▼" : "▶"}</span>
                          <span>{symbol}</span>
                        </td>
                        <td className="py-3 px-4">
                          {formatCurrency(ticker.current_price || 0)}
                        </td>
                        <td
                          className={`py-3 px-4 ${
                            (ticker.price_change || 0) >= 0
                              ? "text-neon-green"
                              : "text-magenta"
                          }`}
                        >
                          {formatCurrency(ticker.price_change || 0)}
                        </td>
                        <td
                          className={`py-3 px-4 ${
                            (ticker.price_change_pct || 0) >= 0
                              ? "text-neon-green"
                              : "text-magenta"
                          }`}
                        >
                          {formatPercent(ticker.price_change_pct || 0)}
                        </td>
                        <td className="py-3 px-4 text-gray-400">
                          {ticker.notes || "—"}
                        </td>
                        <td className="py-3 px-4 text-gray-400 text-xs">
                          {ticker.added_date
                            ? new Date(ticker.added_date).toLocaleDateString()
                            : "—"}
                        </td>
                        <td className="py-3 px-4 text-right" onClick={(e) => e.stopPropagation()}>
                          <button
                            onClick={() => handleRemoveTicker(symbol)}
                            className="text-magenta hover:text-red-400 transition-colors"
                          >
                            ✖
                          </button>
                        </td>
                      </motion.tr>
                      {/* Plotly Graph Row */}
                      {isExpanded && analysis && (
                        <motion.tr
                          initial={{ opacity: 0, height: 0 }}
                          animate={{ opacity: 1, height: "auto" }}
                          exit={{ opacity: 0, height: 0 }}
                          className="border-b border-cyan-500/10"
                        >
                          <td colSpan={7} className="p-0">
                            <div className="p-4 bg-[rgba(0,0,0,0.2)]">
                              {analysis.error ? (
                                <div className="text-magenta text-sm">Error loading analysis: {analysis.error}</div>
                              ) : (
                                <PlotlyGraph
                                  ticker={symbol}
                                  data={analysis}
                                  aiInsight={analysis.ai_insight}
                                  height={400}
                                />
                              )}
                            </div>
                          </td>
                        </motion.tr>
                      )}
                    </React.Fragment>
                  );
                })}
              </AnimatePresence>
            </tbody>
          </table>
          {tickers.length === 0 && (
            <div className="text-center py-8 text-gray-400">
              No tickers in watchlist. Add your first ticker to get started.
            </div>
          )}
        </div>
      </motion.div>

      {error && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="p-4 bg-magenta/20 border border-magenta/40 rounded-lg text-magenta"
        >
          Error: {error}
        </motion.div>
      )}
    </div>
  );
}

