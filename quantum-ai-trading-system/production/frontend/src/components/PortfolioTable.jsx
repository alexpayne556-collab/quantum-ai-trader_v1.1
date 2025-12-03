// frontend/src/components/PortfolioTable.jsx
import React, { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import PortfolioChartRow from "./PortfolioChartRow";
import { getPortfolio } from "../api/client";

export default function PortfolioTable() {
  const [portfolio, setPortfolio] = useState([]);
  const [expanded, setExpanded] = useState(null);
  const [chartDataMap, setChartDataMap] = useState({});
  const [forecastDataMap, setForecastDataMap] = useState({});
  const [aiInsightMap, setAiInsightMap] = useState({});

  // Fetch portfolio data
  useEffect(() => {
    const fetchPortfolio = async () => {
      try {
        const portfolioData = await getPortfolio();
        if (portfolioData && portfolioData.positions) {
          // Map positions to portfolio format
          const positions = portfolioData.positions.map((pos) => ({
            ticker: pos.symbol,
            shares: pos.shares || pos.quantity || 0,
            avgCost: pos.avg_cost || pos.average_cost || pos.cost_basis || 0,
            currentPrice: pos.current_price || pos.price || 0,
          }));
          setPortfolio(positions);
        } else if (portfolioData && Array.isArray(portfolioData)) {
          setPortfolio(portfolioData);
        }
      } catch (err) {
        console.error("Portfolio fetch error:", err);
        // Fallback to empty array
        setPortfolio([]);
      }
    };
    fetchPortfolio();
    const interval = setInterval(fetchPortfolio, 30000);
    return () => clearInterval(interval);
  }, []);

  const handleAdd = () => {
    const ticker = prompt("Enter ticker symbol:")?.toUpperCase();
    const shares = parseFloat(prompt("Enter number of shares (fractional ok):"));
    const avgCost = parseFloat(prompt("Enter average cost per share:"));
    if (ticker && shares && avgCost) {
      setPortfolio([...portfolio, { ticker, shares, avgCost }]);
    }
  };

  const handleRemove = (ticker) => {
    setPortfolio(portfolio.filter((p) => p.ticker !== ticker));
  };

  const toggleExpand = (ticker) =>
    setExpanded(expanded === ticker ? null : ticker);

  return (
    <div className="bg-[rgba(15,15,15,0.9)] backdrop-blur-md border border-cyan-500/20 rounded-xl shadow-[0_0_25px_rgba(0,255,170,0.1)] p-4">
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-electric-blue text-lg font-semibold tracking-wider">
          Portfolio
        </h2>
        <button
          onClick={handleAdd}
          className="text-neon-green border border-neon-green/40 px-3 py-1 rounded hover:bg-neon-green/10 transition-all"
        >
          + Add
        </button>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-sm text-left text-gray-300">
          <thead className="text-gray-400 uppercase border-b border-cyan-500/10">
            <tr>
              <th className="py-2 px-3">Ticker</th>
              <th className="py-2 px-3">Shares</th>
              <th className="py-2 px-3">Avg Cost</th>
              <th className="py-2 px-3">PnL</th>
              <th className="py-2 px-3">Forecast</th>
              <th className="py-2 px-3"></th>
            </tr>
          </thead>
          <tbody>
            {portfolio.map((item) => {
              const currentPrice = item.currentPrice || item.avgCost;
              const pnl = ((currentPrice - item.avgCost) * item.shares).toFixed(2);
              const pnlColor =
                pnl > 0 ? "text-neon-green" : pnl < 0 ? "text-magenta" : "text-gray-400";
              const chartData = chartDataMap[item.ticker];
              const forecastData = forecastDataMap[item.ticker];
              const aiInsight = aiInsightMap[item.ticker];
              
              // Determine if good buy (green background)
              const isGoodBuy = aiInsight?.recommendation?.includes("BUY") && 
                               (aiInsight?.confidence || 0) > 0.6;

              return (
                <React.Fragment key={item.ticker}>
                  <tr
                    className={`border-b border-cyan-500/10 hover:bg-[rgba(0,255,170,0.03)] transition-all cursor-pointer ${
                      isGoodBuy ? "bg-[rgba(0,255,0,0.1)]" : ""
                    }`}
                    onClick={() => toggleExpand(item.ticker)}
                  >
                    <td className="py-2 px-3 font-semibold text-neon-green">
                      {item.ticker}
                    </td>
                    <td className="py-2 px-3">{item.shares.toFixed(2)}</td>
                    <td className="py-2 px-3">${item.avgCost.toFixed(2)}</td>
                    <td className={`py-2 px-3 ${pnlColor}`}>${pnl}</td>
                    <td className="py-2 px-3">
                      {aiInsight?.recommendation || forecastData?.trend || "—"}
                    </td>
                    <td className="py-2 px-3 text-right">
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          handleRemove(item.ticker);
                        }}
                        className="text-magenta hover:text-red-400"
                      >
                        ✖
                      </button>
                    </td>
                  </tr>

                  <AnimatePresence>
                    {expanded === item.ticker && (
                      <motion.tr
                        initial={{ opacity: 0, height: 0 }}
                        animate={{ opacity: 1, height: "auto" }}
                        exit={{ opacity: 0, height: 0 }}
                        transition={{ duration: 0.3 }}
                      >
                        <td colSpan={6} className="py-3 px-2">
                          <PortfolioChartRow 
                            ticker={item.ticker}
                            onDataLoaded={(chartData, forecastData, aiInsight) => {
                              setChartDataMap(prev => ({ ...prev, [item.ticker]: chartData }));
                              setForecastDataMap(prev => ({ ...prev, [item.ticker]: forecastData }));
                              setAiInsightMap(prev => ({ ...prev, [item.ticker]: aiInsight }));
                            }}
                            chartData={chartData}
                            forecastData={forecastData}
                            aiInsight={aiInsight}
                          />
                        </td>
                      </motion.tr>
                    )}
                  </AnimatePresence>
                </React.Fragment>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
