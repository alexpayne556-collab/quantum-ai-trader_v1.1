// frontend/src/components/ProfitCalculator.jsx
import React, { useState, useEffect } from "react";
import { motion } from "framer-motion";

export default function ProfitCalculator({ 
  initialEntry = 0, 
  initialExit = 0, 
  initialShares = 0,
  onClose 
}) {
  const [entryPrice, setEntryPrice] = useState(initialEntry);
  const [exitPrice, setExitPrice] = useState(initialExit);
  const [shares, setShares] = useState(initialShares);
  const [positionSize, setPositionSize] = useState(0);
  
  // Calculate profit/loss
  const profitLoss = (exitPrice - entryPrice) * shares;
  const roi = entryPrice > 0 ? ((exitPrice - entryPrice) / entryPrice) * 100 : 0;
  const riskReward = entryPrice > 0 && exitPrice > entryPrice 
    ? (exitPrice - entryPrice) / (entryPrice * 0.02) // Assuming 2% stop-loss
    : 0;
  
  useEffect(() => {
    setPositionSize(entryPrice * shares);
  }, [entryPrice, shares]);
  
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0, scale: 0.9 }}
      className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-4"
      onClick={onClose}
    >
      <motion.div
        initial={{ y: 20 }}
        animate={{ y: 0 }}
        className="bg-[rgba(15,15,15,0.95)] border border-cyan-500/30 rounded-xl shadow-[0_0_30px_rgba(0,255,170,0.2)] p-6 max-w-md w-full"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-electric-blue text-xl font-semibold font-orbitron">
            Profit Calculator
          </h2>
          {onClose && (
            <button
              onClick={onClose}
              className="text-gray-400 hover:text-red-400 transition-colors"
            >
              ✕
            </button>
          )}
        </div>
        
        <div className="space-y-4">
          {/* Entry Price */}
          <div>
            <label className="block text-sm text-gray-300 mb-1">Entry Price ($)</label>
            <input
              type="number"
              value={entryPrice || ""}
              onChange={(e) => setEntryPrice(parseFloat(e.target.value) || 0)}
              className="w-full bg-[rgba(0,0,0,0.5)] border border-cyan-500/20 rounded px-3 py-2 text-neon-green focus:border-cyan-500 focus:outline-none"
              placeholder="0.00"
            />
          </div>
          
          {/* Exit Price */}
          <div>
            <label className="block text-sm text-gray-300 mb-1">Exit Price ($)</label>
            <input
              type="number"
              value={exitPrice || ""}
              onChange={(e) => setExitPrice(parseFloat(e.target.value) || 0)}
              className="w-full bg-[rgba(0,0,0,0.5)] border border-cyan-500/20 rounded px-3 py-2 text-neon-green focus:border-cyan-500 focus:outline-none"
              placeholder="0.00"
            />
          </div>
          
          {/* Shares */}
          <div>
            <label className="block text-sm text-gray-300 mb-1">Shares/Contracts</label>
            <input
              type="number"
              value={shares || ""}
              onChange={(e) => setShares(parseFloat(e.target.value) || 0)}
              className="w-full bg-[rgba(0,0,0,0.5)] border border-cyan-500/20 rounded px-3 py-2 text-neon-green focus:border-cyan-500 focus:outline-none"
              placeholder="0"
            />
          </div>
          
          {/* Results */}
          <div className="mt-6 pt-4 border-t border-cyan-500/20 space-y-3">
            <div className="flex justify-between">
              <span className="text-gray-400">Position Size:</span>
              <span className="text-electric-blue font-mono">${positionSize.toFixed(2)}</span>
            </div>
            
            <div className="flex justify-between">
              <span className="text-gray-400">Profit/Loss:</span>
              <span className={`font-mono font-semibold ${profitLoss >= 0 ? 'text-neon-green' : 'text-magenta'}`}>
                ${profitLoss >= 0 ? '+' : ''}{profitLoss.toFixed(2)}
              </span>
            </div>
            
            <div className="flex justify-between">
              <span className="text-gray-400">ROI:</span>
              <span className={`font-mono font-semibold ${roi >= 0 ? 'text-neon-green' : 'text-magenta'}`}>
                {roi >= 0 ? '+' : ''}{roi.toFixed(2)}%
              </span>
            </div>
            
            <div className="flex justify-between">
              <span className="text-gray-400">Risk/Reward:</span>
              <span className="text-gold font-mono">{riskReward.toFixed(2)}:1</span>
            </div>
          </div>
          
          {/* Visual Risk/Reward Bar */}
          {riskReward > 0 && (
            <div className="mt-4">
              <div className="text-xs text-gray-400 mb-1">Risk/Reward Ratio</div>
              <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
                <div 
                  className="h-full bg-gradient-to-r from-magenta via-gold to-neon-green transition-all duration-300"
                  style={{ width: `${Math.min(riskReward * 10, 100)}%` }}
                />
              </div>
            </div>
          )}
        </div>
      </motion.div>
    </motion.div>
  );
}

