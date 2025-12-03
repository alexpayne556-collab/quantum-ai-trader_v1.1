// frontend/src/components/TradeSimulator.jsx
import React, { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";

const STORAGE_KEY = "quantum_trade_simulator_trades";

export default function TradeSimulator() {
  const [trades, setTrades] = useState([]);
  const [showAddForm, setShowAddForm] = useState(false);
  const [newTrade, setNewTrade] = useState({
    symbol: "",
    entryPrice: "",
    exitPrice: "",
    shares: "",
    entryDate: new Date().toISOString().split('T')[0],
    exitDate: "",
    status: "open" // open, closed
  });
  
  // Load trades from localStorage
  useEffect(() => {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved) {
      try {
        setTrades(JSON.parse(saved));
      } catch (e) {
        console.error("Failed to load trades:", e);
      }
    }
  }, []);
  
  // Save trades to localStorage
  useEffect(() => {
    if (trades.length > 0) {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(trades));
    }
  }, [trades]);
  
  const addTrade = () => {
    if (!newTrade.symbol || !newTrade.entryPrice || !newTrade.shares) return;
    
    const trade = {
      id: Date.now(),
      ...newTrade,
      entryPrice: parseFloat(newTrade.entryPrice),
      exitPrice: newTrade.exitPrice ? parseFloat(newTrade.exitPrice) : null,
      shares: parseFloat(newTrade.shares),
      createdAt: new Date().toISOString()
    };
    
    setTrades([...trades, trade]);
    setNewTrade({
      symbol: "",
      entryPrice: "",
      exitPrice: "",
      shares: "",
      entryDate: new Date().toISOString().split('T')[0],
      exitDate: "",
      status: "open"
    });
    setShowAddForm(false);
  };
  
  const closeTrade = (id) => {
    const exitPrice = prompt("Enter exit price:");
    if (!exitPrice) return;
    
    setTrades(trades.map(trade => 
      trade.id === id 
        ? { ...trade, exitPrice: parseFloat(exitPrice), exitDate: new Date().toISOString().split('T')[0], status: "closed" }
        : trade
    ));
  };
  
  const deleteTrade = (id) => {
    setTrades(trades.filter(t => t.id !== id));
  };
  
  // Calculate metrics
  const closedTrades = trades.filter(t => t.status === "closed");
  const totalPL = closedTrades.reduce((sum, t) => {
    const pl = (t.exitPrice - t.entryPrice) * t.shares;
    return sum + pl;
  }, 0);
  
  const winningTrades = closedTrades.filter(t => (t.exitPrice - t.entryPrice) * t.shares > 0);
  const winRate = closedTrades.length > 0 ? (winningTrades.length / closedTrades.length) * 100 : 0;
  
  const avgProfit = winningTrades.length > 0
    ? winningTrades.reduce((sum, t) => sum + (t.exitPrice - t.entryPrice) * t.shares, 0) / winningTrades.length
    : 0;
  
  return (
    <div className="bg-[rgba(15,15,15,0.9)] backdrop-blur-md border border-cyan-500/20 rounded-xl shadow-[0_0_25px_rgba(0,255,170,0.1)] p-4">
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-electric-blue text-lg font-semibold tracking-wider font-orbitron">
          Trade Simulator
        </h2>
        <button
          onClick={() => setShowAddForm(!showAddForm)}
          className="text-neon-green border border-neon-green/40 px-3 py-1 rounded hover:bg-neon-green/10 transition-all text-sm"
        >
          + Add Trade
        </button>
      </div>
      
      {/* Add Trade Form */}
      <AnimatePresence>
        {showAddForm && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            className="mb-4 p-3 bg-[rgba(0,0,0,0.5)] rounded border border-cyan-500/20"
          >
            <div className="grid grid-cols-2 gap-2 mb-2">
              <input
                type="text"
                placeholder="Symbol"
                value={newTrade.symbol}
                onChange={(e) => setNewTrade({...newTrade, symbol: e.target.value.toUpperCase()})}
                className="bg-[rgba(0,0,0,0.5)] border border-cyan-500/20 rounded px-2 py-1 text-sm text-neon-green"
              />
              <input
                type="number"
                placeholder="Entry Price"
                value={newTrade.entryPrice}
                onChange={(e) => setNewTrade({...newTrade, entryPrice: e.target.value})}
                className="bg-[rgba(0,0,0,0.5)] border border-cyan-500/20 rounded px-2 py-1 text-sm text-neon-green"
              />
              <input
                type="number"
                placeholder="Shares"
                value={newTrade.shares}
                onChange={(e) => setNewTrade({...newTrade, shares: e.target.value})}
                className="bg-[rgba(0,0,0,0.5)] border border-cyan-500/20 rounded px-2 py-1 text-sm text-neon-green"
              />
              <input
                type="date"
                value={newTrade.entryDate}
                onChange={(e) => setNewTrade({...newTrade, entryDate: e.target.value})}
                className="bg-[rgba(0,0,0,0.5)] border border-cyan-500/20 rounded px-2 py-1 text-sm text-neon-green"
              />
            </div>
            <button
              onClick={addTrade}
              className="w-full bg-neon-green/20 border border-neon-green/40 rounded px-3 py-1 text-neon-green hover:bg-neon-green/30 transition-all text-sm"
            >
              Add Trade
            </button>
          </motion.div>
        )}
      </AnimatePresence>
      
      {/* Performance Metrics */}
      {closedTrades.length > 0 && (
        <div className="grid grid-cols-3 gap-2 mb-4 p-3 bg-[rgba(0,0,0,0.3)] rounded">
          <div>
            <div className="text-xs text-gray-400">Total P&L</div>
            <div className={`text-sm font-semibold font-mono ${totalPL >= 0 ? 'text-neon-green' : 'text-magenta'}`}>
              ${totalPL.toFixed(2)}
            </div>
          </div>
          <div>
            <div className="text-xs text-gray-400">Win Rate</div>
            <div className="text-sm font-semibold font-mono text-gold">
              {winRate.toFixed(1)}%
            </div>
          </div>
          <div>
            <div className="text-xs text-gray-400">Avg Profit</div>
            <div className={`text-sm font-semibold font-mono ${avgProfit >= 0 ? 'text-neon-green' : 'text-magenta'}`}>
              ${avgProfit.toFixed(2)}
            </div>
          </div>
        </div>
      )}
      
      {/* Trade List */}
      <div className="space-y-2 max-h-96 overflow-y-auto">
        {trades.length === 0 ? (
          <div className="text-center text-gray-400 py-8 text-sm">
            No trades yet. Add a trade to start tracking.
          </div>
        ) : (
          trades.map((trade) => {
            const currentPrice = trade.exitPrice || trade.entryPrice;
            const pl = (currentPrice - trade.entryPrice) * trade.shares;
            const plPct = ((currentPrice - trade.entryPrice) / trade.entryPrice) * 100;
            
            return (
              <motion.div
                key={trade.id}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className="p-3 bg-[rgba(0,0,0,0.3)] rounded border border-cyan-500/10 hover:border-cyan-500/30 transition-all"
              >
                <div className="flex justify-between items-start mb-2">
                  <div>
                    <div className="font-semibold text-neon-green">{trade.symbol}</div>
                    <div className="text-xs text-gray-400">
                      {trade.shares} shares @ ${trade.entryPrice.toFixed(2)}
                    </div>
                  </div>
                  <div className="text-right">
                    <div className={`font-mono font-semibold ${pl >= 0 ? 'text-neon-green' : 'text-magenta'}`}>
                      ${pl >= 0 ? '+' : ''}{pl.toFixed(2)}
                    </div>
                    <div className={`text-xs font-mono ${plPct >= 0 ? 'text-neon-green' : 'text-magenta'}`}>
                      {plPct >= 0 ? '+' : ''}{plPct.toFixed(2)}%
                    </div>
                  </div>
                </div>
                <div className="flex gap-2 mt-2">
                  {trade.status === "open" && (
                    <button
                      onClick={() => closeTrade(trade.id)}
                      className="flex-1 bg-gold/20 border border-gold/40 rounded px-2 py-1 text-xs text-gold hover:bg-gold/30 transition-all"
                    >
                      Close
                    </button>
                  )}
                  <button
                    onClick={() => deleteTrade(trade.id)}
                    className="flex-1 bg-magenta/20 border border-magenta/40 rounded px-2 py-1 text-xs text-magenta hover:bg-magenta/30 transition-all"
                  >
                    Delete
                  </button>
                </div>
              </motion.div>
            );
          })
        )}
      </div>
    </div>
  );
}

