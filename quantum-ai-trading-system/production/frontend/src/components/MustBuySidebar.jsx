/**
 * Quantum AI Cockpit — Must Buy Sidebar
 * Scrapes news to find good trading opportunities
 */

import React, { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import ProfitCalculator from "./ProfitCalculator";
import TradeSimulator from "./TradeSimulator";

export default function MustBuySidebar() {
  const [opportunities, setOpportunities] = useState([]);
  const [loading, setLoading] = useState(true);
  const [expanded, setExpanded] = useState(null);
  const [showCalculator, setShowCalculator] = useState(false);
  const [showSimulator, setShowSimulator] = useState(false);
  const [selectedOpportunity, setSelectedOpportunity] = useState(null);

  useEffect(() => {
    const fetchOpportunities = async () => {
      try {
        const response = await fetch("/api/must_buy?limit=20");
        const data = await response.json();
        if (data.status === "success" && data.data) {
          setOpportunities(data.data);
        }
        setLoading(false);
      } catch (error) {
        console.error("Failed to fetch must-buy opportunities:", error);
        setLoading(false);
      }
    };

    fetchOpportunities();
    const interval = setInterval(fetchOpportunities, 300000); // Update every 5 minutes

    return () => clearInterval(interval);
  }, []);

  const handleSymbolClick = (symbol) => {
    // Dispatch event for global ticker change
    window.dispatchEvent(
      new CustomEvent("SET_TICKER", { detail: { ticker: symbol } })
    );
  };

  if (loading) {
    return (
      <div className="w-full h-full bg-[rgba(15,15,15,0.95)] backdrop-blur-md border border-cyan-500/20 rounded-xl shadow-[0_0_25px_rgba(0,255,170,0.1)] p-4">
        <div className="text-cyan-400 text-sm">Loading opportunities...</div>
      </div>
    );
  }

  return (
    <div className="w-full h-full bg-[rgba(15,15,15,0.95)] backdrop-blur-md border border-cyan-500/20 rounded-xl shadow-[0_0_25px_rgba(0,255,170,0.1)] p-4 overflow-y-auto">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-bold bg-gradient-to-r from-cyan-400 to-green-400 bg-clip-text text-transparent">
          💎 Must Buy
        </h2>
        <span className="text-xs text-cyan-500/70">
          {opportunities.length} opportunities
        </span>
      </div>

      <div className="space-y-3">
        <AnimatePresence>
          {opportunities.map((opp, index) => (
            <motion.div
              key={`${opp.symbol}-${index}`}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ delay: index * 0.05 }}
              className="bg-[rgba(0,255,170,0.05)] border border-green-500/20 rounded-lg p-3 cursor-pointer hover:border-green-500/40 transition-all"
              onClick={() => handleSymbolClick(opp.symbol)}
            >
              <div className="flex items-start justify-between mb-2">
                <div className="flex items-center gap-2">
                  <span className="text-lg font-bold text-green-400">
                    {opp.symbol}
                  </span>
                  <span className="text-xs px-2 py-1 bg-green-500/20 text-green-400 rounded">
                    {opp.sentiment_label}
                  </span>
                </div>
                <div className="text-right">
                  <div className="text-xs text-green-400 font-semibold">
                    {Math.round(opp.confidence * 100)}% confidence
                  </div>
                  <div className="text-xs text-cyan-500/70">
                    {opp.sentiment_score > 0.7 ? "🔥" : "⭐"}
                  </div>
                </div>
              </div>

              <div className="text-sm text-gray-300 mb-2 line-clamp-2">
                {opp.title}
              </div>

              <div className="flex items-center justify-between text-xs">
                <span className="text-cyan-500/70">{opp.source}</span>
                <span className="text-gray-500">
                  {opp.published_at
                    ? new Date(opp.published_at).toLocaleDateString()
                    : "Recent"}
                </span>
              </div>

              {/* Trading Signals Display */}
              {opp.trading_signal && (
                <div className="mt-2 pt-2 border-t border-green-500/20 space-y-1">
                  <div className="grid grid-cols-2 gap-2 text-xs">
                    <div>
                      <span className="text-gray-400">Entry:</span>
                      <span className="text-neon-green ml-1 font-mono">${opp.trading_signal.entry_price?.toFixed(2)}</span>
                    </div>
                    <div>
                      <span className="text-gray-400">Target:</span>
                      <span className="text-gold ml-1 font-mono">${opp.trading_signal.take_profit?.toFixed(2)}</span>
                    </div>
                    <div>
                      <span className="text-gray-400">Stop:</span>
                      <span className="text-magenta ml-1 font-mono">${opp.trading_signal.stop_loss?.toFixed(2)}</span>
                    </div>
                    <div>
                      <span className="text-gray-400">R/R:</span>
                      <span className="text-cyan-400 ml-1 font-mono">{opp.trading_signal.risk_reward_ratio?.toFixed(2)}:1</span>
                    </div>
                  </div>
                  <div className="text-xs text-gray-400 mt-1">
                    Position: {opp.trading_signal.position_size_pct?.toFixed(1)}% | 
                    Timing: <span className="text-gold">{opp.trading_signal.entry_timing}</span>
                  </div>
                </div>
              )}

              {expanded === index && (
                <motion.div
                  initial={{ height: 0, opacity: 0 }}
                  animate={{ height: "auto", opacity: 1 }}
                  exit={{ height: 0, opacity: 0 }}
                  className="mt-2 pt-2 border-t border-green-500/20"
                >
                  <p className="text-xs text-gray-400 mb-2">{opp.description}</p>
                  <div className="text-xs text-green-400">
                    Reason: {opp.reason}
                  </div>
                  {opp.url && (
                    <a
                      href={opp.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-xs text-cyan-400 hover:text-cyan-300 mt-1 block"
                      onClick={(e) => e.stopPropagation()}
                    >
                      Read more →
                    </a>
                  )}
                </motion.div>
              )}

              <div className="flex gap-2 mt-2">
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    setSelectedOpportunity(opp);
                    setShowCalculator(true);
                  }}
                  className="flex-1 text-xs bg-gold/20 border border-gold/40 rounded px-2 py-1 text-gold hover:bg-gold/30 transition-all"
                >
                  Calculate
                </button>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    setShowSimulator(true);
                  }}
                  className="flex-1 text-xs bg-cyan-500/20 border border-cyan-500/40 rounded px-2 py-1 text-cyan-400 hover:bg-cyan-500/30 transition-all"
                >
                  Simulate
                </button>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    setExpanded(expanded === index ? null : index);
                  }}
                  className="text-xs text-cyan-500 hover:text-cyan-300 px-2"
                >
                  {expanded === index ? "▲" : "▼"}
                </button>
              </div>
            </motion.div>
          ))}
        </AnimatePresence>

        {opportunities.length === 0 && (
          <div className="text-center text-gray-500 py-8">
            <div className="text-4xl mb-2">🔍</div>
            <div>No opportunities found</div>
            <div className="text-xs mt-2">
              Check back later for news-based opportunities
            </div>
          </div>
        )}
      </div>

      {/* Profit Calculator Modal */}
      {showCalculator && selectedOpportunity?.trading_signal && (
        <ProfitCalculator
          initialEntry={selectedOpportunity.trading_signal.entry_price}
          initialExit={selectedOpportunity.trading_signal.take_profit}
          initialShares={Math.floor(selectedOpportunity.trading_signal.position_size_shares)}
          onClose={() => {
            setShowCalculator(false);
            setSelectedOpportunity(null);
          }}
        />
      )}

      {/* Trade Simulator */}
      {showSimulator && (
        <div className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className="bg-[rgba(15,15,15,0.95)] border border-cyan-500/30 rounded-xl shadow-[0_0_30px_rgba(0,255,170,0.2)] p-6 max-w-2xl w-full max-h-[90vh] overflow-y-auto">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-electric-blue text-xl font-semibold font-orbitron">
                Trade Simulator
              </h2>
              <button
                onClick={() => setShowSimulator(false)}
                className="text-gray-400 hover:text-red-400 transition-colors"
              >
                ✕
              </button>
            </div>
            <TradeSimulator />
          </div>
        </div>
      )}
    </div>
  );
}

