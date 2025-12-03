/**
 * Quantum AI Cockpit — MSNBC-Style Market Ticker
 * Scrolling ticker component for real-time market data
 */

import React, { useState, useEffect, useRef } from "react";
import { motion } from "framer-motion";

export default function MarketTicker({ limit = 50 }) {
  const [tickerData, setTickerData] = useState([]);
  const [loading, setLoading] = useState(true);
  const tickerRef = useRef(null);

  useEffect(() => {
    const fetchTickerData = async () => {
      try {
        const response = await fetch(`/api/market_ticker?limit=${limit}`);
        const data = await response.json();
        if (data.status === "success" && data.data) {
          setTickerData(data.data);
        }
        setLoading(false);
      } catch (error) {
        console.error("Failed to fetch ticker data:", error);
        setLoading(false);
      }
    };

    fetchTickerData();
    const interval = setInterval(fetchTickerData, 30000); // Update every 30 seconds

    return () => clearInterval(interval);
  }, [limit]);

  if (loading) {
    return (
      <div className="w-full h-12 bg-[rgba(0,0,0,0.8)] border-b border-cyan-500/20 flex items-center justify-center">
        <div className="text-cyan-400 text-sm">Loading ticker...</div>
      </div>
    );
  }

  // Duplicate data for seamless scrolling
  const duplicatedData = [...tickerData, ...tickerData];

  return (
    <div className="w-full h-12 bg-[rgba(0,0,0,0.9)] border-b border-cyan-500/20 overflow-hidden relative">
      <div className="absolute inset-0 bg-gradient-to-r from-transparent via-cyan-500/5 to-transparent pointer-events-none" />
      
      <motion.div
        ref={tickerRef}
        className="flex items-center h-full"
        animate={{
          x: [0, -50 * 200], // Scroll based on item width
        }}
        transition={{
          duration: 60, // 60 seconds for full scroll
          repeat: Infinity,
          ease: "linear",
        }}
      >
        {duplicatedData.map((item, index) => (
          <div
            key={`${item.symbol}-${index}`}
            className="flex items-center px-4 whitespace-nowrap"
            style={{ minWidth: "200px" }}
          >
            <span className="text-xs font-mono text-cyan-400 mr-2">
              {item.symbol}
            </span>
            <span
              className={`text-xs font-semibold mr-1 ${
                item.color === "green"
                  ? "text-green-400"
                  : item.color === "red"
                  ? "text-red-400"
                  : "text-gray-400"
              }`}
            >
              ${item.price.toFixed(2)}
            </span>
            <span
              className={`text-xs ${
                item.change_pct >= 0 ? "text-green-400" : "text-red-400"
              }`}
            >
              {item.change_pct >= 0 ? "+" : ""}
              {item.change_pct.toFixed(2)}%
            </span>
            <span className="text-xs ml-2 text-cyan-500/50">{item.emoji}</span>
          </div>
        ))}
      </motion.div>
    </div>
  );
}

