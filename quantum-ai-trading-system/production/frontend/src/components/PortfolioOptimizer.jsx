// frontend/src/components/PortfolioOptimizer.jsx
import React, { useState, useEffect } from "react";
import { motion } from "framer-motion";

export default function PortfolioOptimizer() {
  const [portfolio, setPortfolio] = useState([]);
  const [loading, setLoading] = useState(true);
  const [recommendations, setRecommendations] = useState([]);
  const [riskTolerance, setRiskTolerance] = useState(2.0); // % risk per trade
  
  useEffect(() => {
    const fetchPortfolio = async () => {
      try {
        const response = await fetch("/api/portfolio");
        if (response.ok) {
          const data = await response.json();
          if (data.status === "success" && data.data) {
            setPortfolio(data.data.positions || []);
            analyzePortfolio(data.data.positions || []);
          }
        }
        setLoading(false);
      } catch (error) {
        console.error("Failed to fetch portfolio:", error);
        setLoading(false);
      }
    };
    
    fetchPortfolio();
    const interval = setInterval(fetchPortfolio, 60000); // Update every minute
    return () => clearInterval(interval);
  }, [riskTolerance]);
  
  const analyzePortfolio = (positions) => {
    if (!positions || positions.length === 0) {
      setRecommendations([]);
      return;
    }
    
    const totalValue = positions.reduce((sum, p) => sum + (p.current_value || p.value || 0), 0);
    const recommendations = [];
    
    // Analyze each position
    positions.forEach((position) => {
      const currentValue = position.current_value || position.value || 0;
      const currentPct = totalValue > 0 ? (currentValue / totalValue) * 100 : 0;
      const symbol = position.symbol || position.ticker;
      
      // Get risk assessment
      const risk = position.risk_score || 0.5;
      const volatility = position.volatility || 0.02;
      
      // Calculate suggested position size based on risk tolerance
      const suggestedPct = Math.min(riskTolerance * (1 - risk) * 10, 20); // Max 20% per position
      
      if (currentPct > suggestedPct * 1.2) {
        recommendations.push({
          symbol,
          action: "TRIM",
          currentPct: currentPct.toFixed(1),
          suggestedPct: suggestedPct.toFixed(1),
          reason: `Position is ${(currentPct - suggestedPct).toFixed(1)}% overweight`,
          priority: "high"
        });
      } else if (currentPct < suggestedPct * 0.8 && risk < 0.6) {
        recommendations.push({
          symbol,
          action: "ADD",
          currentPct: currentPct.toFixed(1),
          suggestedPct: suggestedPct.toFixed(1),
          reason: `Position is ${(suggestedPct - currentPct).toFixed(1)}% underweight`,
          priority: "medium"
        });
      }
    });
    
    // Diversification analysis
    const sectors = {};
    positions.forEach((p) => {
      const sector = p.sector || "Unknown";
      sectors[sector] = (sectors[sector] || 0) + (p.current_value || p.value || 0);
    });
    
    const sectorConcentration = Object.values(sectors).reduce((max, val) => 
      Math.max(max, (val / totalValue) * 100), 0
    );
    
    if (sectorConcentration > 40) {
      recommendations.push({
        symbol: "PORTFOLIO",
        action: "DIVERSIFY",
        currentPct: sectorConcentration.toFixed(1),
        suggestedPct: "30",
        reason: `Sector concentration is ${sectorConcentration.toFixed(1)}% - consider diversifying`,
        priority: "high"
      });
    }
    
    setRecommendations(recommendations);
  };
  
  // Calculate portfolio metrics
  const totalValue = portfolio.reduce((sum, p) => sum + (p.current_value || p.value || 0), 0);
  const totalPL = portfolio.reduce((sum, p) => {
    const cost = p.cost_basis || p.entry_price * (p.shares || 0) || 0;
    const value = p.current_value || p.value || 0;
    return sum + (value - cost);
  }, 0);
  const totalPLPct = portfolio.reduce((sum, p) => {
    const cost = p.cost_basis || p.entry_price * (p.shares || 0) || 0;
    if (cost === 0) return sum;
    const value = p.current_value || p.value || 0;
    return sum + ((value - cost) / cost) * (p.current_value || p.value || 0) / totalValue;
  }, 0);
  
  if (loading) {
    return (
      <div className="bg-[rgba(15,15,15,0.9)] backdrop-blur-md border border-cyan-500/20 rounded-xl shadow-[0_0_25px_rgba(0,255,170,0.1)] p-4">
        <div className="text-cyan-400 text-sm">Loading portfolio...</div>
      </div>
    );
  }
  
  return (
    <div className="bg-[rgba(15,15,15,0.9)] backdrop-blur-md border border-cyan-500/20 rounded-xl shadow-[0_0_25px_rgba(0,255,170,0.1)] p-4">
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-electric-blue text-lg font-semibold tracking-wider font-orbitron">
          Portfolio Optimizer
        </h2>
        <div className="flex items-center gap-2">
          <label className="text-xs text-gray-400">Risk:</label>
          <input
            type="range"
            min="1"
            max="5"
            step="0.5"
            value={riskTolerance}
            onChange={(e) => setRiskTolerance(parseFloat(e.target.value))}
            className="w-20"
          />
          <span className="text-xs text-cyan-400 font-mono">{riskTolerance}%</span>
        </div>
      </div>
      
      {/* Portfolio Summary */}
      <div className="grid grid-cols-3 gap-2 mb-4 p-3 bg-[rgba(0,0,0,0.3)] rounded">
        <div>
          <div className="text-xs text-gray-400">Total Value</div>
          <div className="text-sm font-semibold font-mono text-electric-blue">
            ${totalValue.toFixed(2)}
          </div>
        </div>
        <div>
          <div className="text-xs text-gray-400">Total P&L</div>
          <div className={`text-sm font-semibold font-mono ${totalPL >= 0 ? 'text-neon-green' : 'text-magenta'}`}>
            ${totalPL >= 0 ? '+' : ''}{totalPL.toFixed(2)}
          </div>
        </div>
        <div>
          <div className="text-xs text-gray-400">Return</div>
          <div className={`text-sm font-semibold font-mono ${totalPLPct >= 0 ? 'text-neon-green' : 'text-magenta'}`}>
            {totalPLPct >= 0 ? '+' : ''}{totalPLPct.toFixed(2)}%
          </div>
        </div>
      </div>
      
      {/* Recommendations */}
      {recommendations.length > 0 ? (
        <div className="space-y-2">
          <div className="text-sm text-gray-300 mb-2 font-semibold">Recommendations:</div>
          {recommendations.map((rec, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              className={`p-3 rounded border ${
                rec.priority === "high" 
                  ? "bg-magenta/10 border-magenta/30" 
                  : "bg-gold/10 border-gold/30"
              }`}
            >
              <div className="flex justify-between items-start mb-1">
                <div className="font-semibold text-neon-green">{rec.symbol}</div>
                <span className={`text-xs px-2 py-1 rounded ${
                  rec.action === "TRIM" 
                    ? "bg-magenta/20 text-magenta" 
                    : rec.action === "ADD"
                    ? "bg-neon-green/20 text-neon-green"
                    : "bg-gold/20 text-gold"
                }`}>
                  {rec.action}
                </span>
              </div>
              <div className="text-xs text-gray-400 mb-1">{rec.reason}</div>
              <div className="text-xs text-gray-300">
                Current: {rec.currentPct}% → Suggested: {rec.suggestedPct}%
              </div>
            </motion.div>
          ))}
        </div>
      ) : (
        <div className="text-center text-gray-400 py-4 text-sm">
          Portfolio is well-balanced. No recommendations at this time.
        </div>
      )}
      
      {/* Position List */}
      {portfolio.length > 0 && (
        <div className="mt-4 pt-4 border-t border-cyan-500/20">
          <div className="text-sm text-gray-300 mb-2 font-semibold">Positions:</div>
          <div className="space-y-1 max-h-48 overflow-y-auto">
            {portfolio.map((position, index) => {
              const symbol = position.symbol || position.ticker;
              const value = position.current_value || position.value || 0;
              const pct = totalValue > 0 ? (value / totalValue) * 100 : 0;
              const pl = (position.current_value || position.value || 0) - 
                        (position.cost_basis || position.entry_price * (position.shares || 0) || 0);
              
              return (
                <div
                  key={index}
                  className="flex justify-between items-center p-2 bg-[rgba(0,0,0,0.3)] rounded text-xs"
                >
                  <div>
                    <div className="font-semibold text-neon-green">{symbol}</div>
                    <div className="text-gray-400">{pct.toFixed(1)}%</div>
                  </div>
                  <div className="text-right">
                    <div className="font-mono">${value.toFixed(2)}</div>
                    <div className={`font-mono ${pl >= 0 ? 'text-neon-green' : 'text-magenta'}`}>
                      {pl >= 0 ? '+' : ''}${pl.toFixed(2)}
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}
      
      {portfolio.length === 0 && (
        <div className="text-center text-gray-400 py-8 text-sm">
          No positions in portfolio
        </div>
      )}
    </div>
  );
}

