/**
 * Quantum AI Cockpit — Portfolio Manager Component
 * 💼 Real-time portfolio management with WebSocket updates
 * ========================================================
 */

import React, { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { useWebSocketFeed } from "../hooks/useWebSocketFeed";
import PlotlyGraph from "./PlotlyGraph";
import Plot from "react-plotly.js";

// Use relative URL for API calls (Vite proxy will handle routing)
const API_BASE = ""; // Backend API base URL (uses Vite proxy)

export default function PortfolioManager() {
  const [portfolio, setPortfolio] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [showAddForm, setShowAddForm] = useState(false);
  const [expandedSymbols, setExpandedSymbols] = useState(new Set()); // Track which symbols have graphs expanded
  const [aiRecommendations, setAiRecommendations] = useState({});
  const [plotlyData, setPlotlyData] = useState({});
  const [addFormData, setAddFormData] = useState({
    symbol: "",
    shares: "",
    cost_basis: "",
    sector: "",
  });

  // WebSocket connection for real-time updates
  const { data: wsData, isConnected } = useWebSocketFeed("/ws/portfolio");

  // Load initial portfolio data
  useEffect(() => {
    loadPortfolio();
  }, []);

  // Fetch AI recommendations and plotly data for portfolio positions
  useEffect(() => {
    if (!portfolio || !portfolio.positions) return;
    
    const fetchRecommendations = async () => {
      const positions = portfolio.positions || [];
      const promises = positions.map(async (pos) => {
        const symbol = pos.symbol || pos.ticker;
        try {
          const [aiRes, forecastRes] = await Promise.all([
            fetch(`/api/ai_recommendation/${symbol}`).catch(() => null),
            fetch(`/api/forecast/${symbol}`).catch(() => null)
          ]);
          
          const aiData = aiRes?.ok ? await aiRes.json() : null;
          const forecastData = forecastRes?.ok ? await forecastRes.json() : null;
          
          // Extract plotly data
          const historicalData = forecastData?.metrics || forecastData?.result || {};
          const timestamps = historicalData.timestamps || [];
          const closes = historicalData.close || [];
          
          const plotData = timestamps.length > 0 ? {
            x: timestamps,
            y: closes,
            type: 'scatter',
            mode: 'lines',
            name: symbol,
            line: { color: '#00FFB3', width: 2 }
          } : null;
          
          return { symbol, aiData, plotData };
        } catch (err) {
          console.error(`Error fetching data for ${symbol}:`, err);
          return { symbol, aiData: null, plotData: null };
        }
      });
      
      const results = await Promise.all(promises);
      const aiMap = {};
      const plotMap = {};
      
      results.forEach(({ symbol, aiData, plotData }) => {
        if (aiData) aiMap[symbol] = aiData;
        if (plotData) plotMap[symbol] = plotData;
      });
      
      setAiRecommendations(aiMap);
      setPlotlyData(plotMap);
    };
    
    fetchRecommendations();
    const interval = setInterval(fetchRecommendations, 60000); // Update every minute
    return () => clearInterval(interval);
  }, [portfolio]);

  // Update portfolio when WebSocket data arrives
  useEffect(() => {
    if (wsData && wsData.type === "portfolio_update" && wsData.data) {
      setPortfolio(wsData.data);
      setLoading(false);
    }
  }, [wsData]);

  const loadPortfolio = async () => {
    try {
      setLoading(true);
      const response = await fetch(`${API_BASE}/api/portfolio`);
      if (!response.ok) {
        throw new Error("Failed to load portfolio");
      }
      const data = await response.json();
      setPortfolio(data);
      setError(null);
    } catch (err) {
      console.error("Error loading portfolio:", err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleAddHolding = async (e) => {
    e.preventDefault();
    try {
      const response = await fetch(`${API_BASE}/api/portfolio/add`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          symbol: addFormData.symbol.toUpperCase(),
          shares: parseFloat(addFormData.shares),
          cost_basis: parseFloat(addFormData.cost_basis),
          sector: addFormData.sector || "Unknown",
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || "Failed to add holding");
      }

      const result = await response.json();
      setPortfolio(result.portfolio);
      setShowAddForm(false);
      setAddFormData({ symbol: "", shares: "", cost_basis: "", sector: "" });
      setError(null);
    } catch (err) {
      console.error("Error adding holding:", err);
      setError(err.message);
    }
  };

  const handleRemoveHolding = async (symbol) => {
    if (!window.confirm(`Remove ${symbol} from portfolio?`)) {
      return;
    }

    try {
      const response = await fetch(`${API_BASE}/api/portfolio/remove`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ symbol }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || "Failed to remove holding");
      }

      const result = await response.json();
      setPortfolio(result.portfolio);
      setError(null);
    } catch (err) {
      console.error("Error removing holding:", err);
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

  if (loading && !portfolio) {
    return (
      <div className="glassmorphic-panel rounded-xl p-8 text-center">
        <div className="animate-pulse text-electric-blue">Loading portfolio...</div>
      </div>
    );
  }

  if (error && !portfolio) {
    return (
      <div className="glassmorphic-panel rounded-xl p-8 text-center">
        <div className="text-magenta">Error: {error}</div>
        <button
          onClick={loadPortfolio}
          className="mt-4 px-4 py-2 bg-electric-blue text-white rounded hover:bg-electric-blue/80"
        >
          Retry
        </button>
      </div>
    );
  }

  // Support both old format (holdings array) and new format (symbols array with analysis)
  const holdings = portfolio?.positions || portfolio?.symbols || portfolio?.holdings || [];
  const totalEquity = portfolio?.total_value || portfolio?.total_equity || 0;
  const totalCostBasis = portfolio?.total_cost_basis || 0;
  const totalGainLoss = portfolio?.total_gain_loss || 0;
  const totalGainLossPct = portfolio?.total_gain_loss_pct || 0;
  
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
      {/* Portfolio Summary */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="glassmorphic-panel rounded-xl p-6"
      >
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-2xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-green-400">
            💼 Portfolio Summary
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

        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="bg-[rgba(0,255,170,0.1)] rounded-lg p-4 border border-neon-green/20">
            <div className="text-sm text-gray-400 mb-1">Total Equity</div>
            <div className="text-2xl font-bold text-neon-green">
              {formatCurrency(totalEquity)}
            </div>
          </div>
          <div className="bg-[rgba(0,209,255,0.1)] rounded-lg p-4 border border-electric-blue/20">
            <div className="text-sm text-gray-400 mb-1">Cost Basis</div>
            <div className="text-2xl font-bold text-electric-blue">
              {formatCurrency(totalCostBasis)}
            </div>
          </div>
          <div
            className={`rounded-lg p-4 border ${
              totalGainLoss >= 0
                ? "bg-[rgba(0,255,170,0.1)] border-neon-green/20"
                : "bg-[rgba(255,0,122,0.1)] border-magenta/20"
            }`}
          >
            <div className="text-sm text-gray-400 mb-1">Gain/Loss</div>
            <div
              className={`text-2xl font-bold ${
                totalGainLoss >= 0 ? "text-neon-green" : "text-magenta"
              }`}
            >
              {formatCurrency(totalGainLoss)}
            </div>
          </div>
          <div
            className={`rounded-lg p-4 border ${
              totalGainLossPct >= 0
                ? "bg-[rgba(0,255,170,0.1)] border-neon-green/20"
                : "bg-[rgba(255,0,122,0.1)] border-magenta/20"
            }`}
          >
            <div className="text-sm text-gray-400 mb-1">Gain/Loss %</div>
            <div
              className={`text-2xl font-bold ${
                totalGainLossPct >= 0 ? "text-neon-green" : "text-magenta"
              }`}
            >
              {formatPercent(totalGainLossPct)}
            </div>
          </div>
        </div>
      </motion.div>

      {/* Holdings Table */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="glassmorphic-panel rounded-xl p-6"
      >
        <div className="flex justify-between items-center mb-4">
          <h3 className="text-xl font-semibold text-electric-blue">Holdings</h3>
          <button
            onClick={() => setShowAddForm(!showAddForm)}
            className="px-4 py-2 bg-neon-green/20 text-neon-green border border-neon-green/40 rounded-lg hover:bg-neon-green/30 transition-all"
          >
            + Add Holding
          </button>
        </div>

        {/* Add Holding Form */}
        <AnimatePresence>
          {showAddForm && (
            <motion.form
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              exit={{ opacity: 0, height: 0 }}
              onSubmit={handleAddHolding}
              className="mb-4 p-4 bg-[rgba(0,209,255,0.1)] rounded-lg border border-electric-blue/20"
            >
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
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
                  type="number"
                  step="0.01"
                  placeholder="Shares (fractional ok)"
                  value={addFormData.shares}
                  onChange={(e) =>
                    setAddFormData({ ...addFormData, shares: e.target.value })
                  }
                  className="px-3 py-2 bg-[rgba(15,15,15,0.8)] border border-cyan-500/20 rounded text-white placeholder-gray-500"
                  required
                />
                <input
                  type="number"
                  step="0.01"
                  placeholder="Cost Basis ($)"
                  value={addFormData.cost_basis}
                  onChange={(e) =>
                    setAddFormData({ ...addFormData, cost_basis: e.target.value })
                  }
                  className="px-3 py-2 bg-[rgba(15,15,15,0.8)] border border-cyan-500/20 rounded text-white placeholder-gray-500"
                  required
                />
                <input
                  type="text"
                  placeholder="Sector (optional)"
                  value={addFormData.sector}
                  onChange={(e) =>
                    setAddFormData({ ...addFormData, sector: e.target.value })
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

        {/* Holdings Table */}
        <div className="overflow-x-auto">
          <table className="w-full text-sm text-left text-gray-300">
            <thead className="text-gray-400 uppercase border-b border-cyan-500/20">
              <tr>
                <th className="py-3 px-4">Symbol</th>
                <th className="py-3 px-4">Shares</th>
                <th className="py-3 px-4">Cost Basis</th>
                <th className="py-3 px-4">Current Price</th>
                <th className="py-3 px-4">Equity</th>
                <th className="py-3 px-4">Gain/Loss</th>
                <th className="py-3 px-4">Gain/Loss %</th>
                <th className="py-3 px-4">Sector</th>
                <th className="py-3 px-4"></th>
              </tr>
            </thead>
            <tbody>
              <AnimatePresence>
                {holdings.map((holding, index) => {
                  const symbol = holding.symbol;
                  const isExpanded = expandedSymbols.has(symbol);
                  const analysis = holding.analysis;
                  
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
                        <td className="py-3 px-4">{holding.shares?.toFixed(4) || 0}</td>
                        <td className="py-3 px-4">
                          {formatCurrency(holding.cost_basis || 0)}
                        </td>
                        <td className="py-3 px-4">
                          {formatCurrency(holding.current_price || 0)}
                        </td>
                        <td className="py-3 px-4">
                          {formatCurrency(holding.equity || 0)}
                        </td>
                        <td
                          className={`py-3 px-4 ${
                            (holding.gain_loss || 0) >= 0
                              ? "text-neon-green"
                              : "text-magenta"
                          }`}
                        >
                          {formatCurrency(holding.gain_loss || 0)}
                        </td>
                        <td
                          className={`py-3 px-4 ${
                            (holding.gain_loss_pct || 0) >= 0
                              ? "text-neon-green"
                              : "text-magenta"
                          }`}
                        >
                          {formatPercent(holding.gain_loss_pct || 0)}
                        </td>
                        <td className="py-3 px-4 text-gray-400">
                          {holding.sector || "Unknown"}
                        </td>
                        <td className="py-3 px-4 text-right" onClick={(e) => e.stopPropagation()}>
                          <button
                            onClick={() => handleRemoveHolding(symbol)}
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
                          <td colSpan={9} className="p-0">
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
          {holdings.length === 0 && (
            <div className="text-center py-8 text-gray-400">
              No holdings yet. Add your first holding to get started.
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

