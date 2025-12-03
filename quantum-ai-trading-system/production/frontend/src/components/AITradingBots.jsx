// frontend/src/components/AITradingBots.jsx
import React, { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";

export default function AITradingBots() {
  const [bots, setBots] = useState([]);
  const [loading, setLoading] = useState(true);
  const [newBotName, setNewBotName] = useState("");
  const [selectedBot, setSelectedBot] = useState(null);
  const [trades, setTrades] = useState([]);

  useEffect(() => {
    fetchBots();
    const interval = setInterval(fetchBots, 10000); // Update every 10 seconds
    return () => clearInterval(interval);
  }, []);

  const fetchBots = async () => {
    try {
      const response = await fetch("/api/trading_bots");
      if (response.ok) {
        const data = await response.json();
        if (data.status === "success") {
          setBots(data.bots || []);
        }
      }
      setLoading(false);
    } catch (error) {
      console.error("Failed to fetch trading bots:", error);
      setLoading(false);
    }
  };

  const createBot = async () => {
    if (!newBotName.trim()) return;
    
    try {
      const response = await fetch(`/api/trading_bots/create?name=${encodeURIComponent(newBotName)}`, {
        method: "POST"
      });
      if (response.ok) {
        const data = await response.json();
        if (data.status === "success") {
          setNewBotName("");
          fetchBots();
        }
      }
    } catch (error) {
      console.error("Failed to create bot:", error);
    }
  };

  const analyzeAndTrade = async (botId, symbols) => {
    try {
      const response = await fetch(`/api/trading_bots/${botId}/analyze?symbols=${symbols.join(",")}`, {
        method: "POST"
      });
      if (response.ok) {
        fetchBots();
      }
    } catch (error) {
      console.error("Failed to analyze and trade:", error);
    }
  };

  if (loading) {
    return (
      <div className="bg-[rgba(15,15,15,0.9)] backdrop-blur-md border border-cyan-500/20 rounded-xl shadow-[0_0_25px_rgba(0,255,170,0.1)] p-4">
        <div className="text-cyan-400 text-sm">Loading trading bots...</div>
      </div>
    );
  }

  return (
    <div className="bg-[rgba(15,15,15,0.9)] backdrop-blur-md border border-cyan-500/20 rounded-xl shadow-[0_0_25px_rgba(0,255,170,0.1)] p-4">
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-electric-blue text-lg font-semibold tracking-wider font-orbitron">
          🤖 AI Trading Bots
        </h2>
        <div className="flex gap-2">
          <input
            type="text"
            placeholder="Bot name"
            value={newBotName}
            onChange={(e) => setNewBotName(e.target.value)}
            onKeyPress={(e) => e.key === "Enter" && createBot()}
            className="bg-[rgba(0,0,0,0.5)] border border-cyan-500/20 rounded px-2 py-1 text-sm text-neon-green w-32"
          />
          <button
            onClick={createBot}
            className="text-neon-green border border-neon-green/40 px-3 py-1 rounded hover:bg-neon-green/10 transition-all text-sm"
          >
            + Create Bot
          </button>
        </div>
      </div>

      <div className="space-y-3">
        {bots.length === 0 ? (
          <div className="text-center text-gray-400 py-8 text-sm">
            No trading bots created yet. Create one to start automated trading.
          </div>
        ) : (
          bots.map((botStatus) => {
            const bot = botStatus.bot;
            const portfolioValue = botStatus.portfolio_value || bot.balance;
            const totalReturn = botStatus.total_return || 0;
            const totalReturnPct = botStatus.total_return_pct || 0;
            const isProfitable = totalReturn > 0;

            return (
              <motion.div
                key={bot.id}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className="p-4 bg-[rgba(0,0,0,0.3)] rounded-lg border border-cyan-500/20 hover:border-cyan-500/40 transition-all"
              >
                <div className="flex justify-between items-start mb-3">
                  <div>
                    <div className="flex items-center gap-2 mb-1">
                      <h3 className="text-neon-green font-semibold">{bot.name}</h3>
                      <span className={`text-xs px-2 py-1 rounded ${
                        bot.active ? "bg-neon-green/20 text-neon-green" : "bg-gray-500/20 text-gray-400"
                      }`}>
                        {bot.active ? "ACTIVE" : "INACTIVE"}
                      </span>
                    </div>
                    <div className="text-xs text-gray-400">
                      Created: {new Date(bot.created_at).toLocaleDateString()}
                    </div>
                  </div>
                  <button
                    onClick={() => setSelectedBot(selectedBot === bot.id ? null : bot.id)}
                    className="text-cyan-400 hover:text-cyan-300 text-xs"
                  >
                    {selectedBot === bot.id ? "▲" : "▼"}
                  </button>
                </div>

                <div className="grid grid-cols-4 gap-2 mb-3">
                  <div>
                    <div className="text-xs text-gray-400">Balance</div>
                    <div className="text-sm font-semibold font-mono text-electric-blue">
                      ${bot.balance.toFixed(2)}
                    </div>
                  </div>
                  <div>
                    <div className="text-xs text-gray-400">Portfolio Value</div>
                    <div className="text-sm font-semibold font-mono text-cyan-400">
                      ${portfolioValue.toFixed(2)}
                    </div>
                  </div>
                  <div>
                    <div className="text-xs text-gray-400">Total Return</div>
                    <div className={`text-sm font-semibold font-mono ${
                      isProfitable ? "text-neon-green" : "text-magenta"
                    }`}>
                      {totalReturn >= 0 ? "+" : ""}${totalReturn.toFixed(2)}
                    </div>
                  </div>
                  <div>
                    <div className="text-xs text-gray-400">Return %</div>
                    <div className={`text-sm font-semibold font-mono ${
                      isProfitable ? "text-neon-green" : "text-magenta"
                    }`}>
                      {totalReturnPct >= 0 ? "+" : ""}{totalReturnPct.toFixed(2)}%
                    </div>
                  </div>
                </div>

                <div className="grid grid-cols-3 gap-2 mb-3">
                  <div>
                    <div className="text-xs text-gray-400">Trades</div>
                    <div className="text-sm font-mono text-gold">
                      {bot.total_trades}
                    </div>
                  </div>
                  <div>
                    <div className="text-xs text-gray-400">Win Rate</div>
                    <div className="text-sm font-mono text-gold">
                      {(bot.win_rate * 100).toFixed(1)}%
                    </div>
                  </div>
                  <div>
                    <div className="text-xs text-gray-400">Positions</div>
                    <div className="text-sm font-mono text-cyan-400">
                      {Object.keys(bot.positions || {}).length}
                    </div>
                  </div>
                </div>

                <AnimatePresence>
                  {selectedBot === bot.id && (
                    <motion.div
                      initial={{ opacity: 0, height: 0 }}
                      animate={{ opacity: 1, height: "auto" }}
                      exit={{ opacity: 0, height: 0 }}
                      className="mt-3 pt-3 border-t border-cyan-500/20"
                    >
                      <div className="mb-2">
                        <div className="text-xs text-gray-400 mb-1">Current Positions:</div>
                        {Object.keys(bot.positions || {}).length > 0 ? (
                          <div className="space-y-1">
                            {Object.entries(bot.positions || {}).map(([symbol, shares]) => (
                              <div key={symbol} className="flex justify-between text-xs">
                                <span className="text-neon-green">{symbol}</span>
                                <span className="text-gray-300">{shares.toFixed(2)} shares</span>
                              </div>
                            ))}
                          </div>
                        ) : (
                          <div className="text-xs text-gray-500">No positions</div>
                        )}
                      </div>
                      <button
                        onClick={() => analyzeAndTrade(bot.id, ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"])}
                        className="w-full bg-neon-green/20 border border-neon-green/40 rounded px-3 py-2 text-neon-green hover:bg-neon-green/30 transition-all text-sm"
                      >
                        Analyze & Trade (Top 5 Stocks)
                      </button>
                    </motion.div>
                  )}
                </AnimatePresence>
              </motion.div>
            );
          })
        )}
      </div>
    </div>
  );
}

