/**
 * Quantum AI Cockpit — Top Gainers Ticker
 * Scrolling ticker for top gainers
 */

import React, { useState, useEffect } from "react";
import { motion } from "framer-motion";

export default function TopGainersTicker({ limit = 20 }) {
  const [gainers, setGainers] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchGainers = async () => {
      try {
        const response = await fetch(`/api/top_gainers?limit=${limit}`);
        const data = await response.json();
        if (data.status === "success" && data.data) {
          setGainers(data.data);
        }
        setLoading(false);
      } catch (error) {
        console.error("Failed to fetch top gainers:", error);
        setLoading(false);
      }
    };

    fetchGainers();
    const interval = setInterval(fetchGainers, 60000); // Update every minute

    return () => clearInterval(interval);
  }, [limit]);

  if (loading) {
    return (
      <div className="w-full h-10 bg-[rgba(0,0,0,0.8)] border-b border-green-500/20 flex items-center justify-center">
        <div className="text-green-400 text-xs">Loading gainers...</div>
      </div>
    );
  }

  // Duplicate for seamless scrolling
  const duplicatedGainers = [...gainers, ...gainers];

  return (
    <div className="w-full h-10 bg-[rgba(0,0,0,0.9)] border-b border-green-500/20 overflow-hidden relative">
      <div className="absolute left-0 top-0 bottom-0 w-20 bg-gradient-to-r from-[rgba(0,0,0,0.9)] to-transparent z-10 pointer-events-none" />
      <div className="absolute right-0 top-0 bottom-0 w-20 bg-gradient-to-l from-[rgba(0,0,0,0.9)] to-transparent z-10 pointer-events-none" />
      
      <motion.div
        className="flex items-center h-full"
        animate={{
          x: [0, -50 * 180],
        }}
        transition={{
          duration: 40,
          repeat: Infinity,
          ease: "linear",
        }}
      >
        {duplicatedGainers.map((gainer, index) => (
          <div
            key={`${gainer.symbol}-${index}`}
            className="flex items-center px-3 whitespace-nowrap"
            style={{ minWidth: "180px" }}
          >
            <span className="text-xs font-mono text-green-400 mr-2 font-bold">
              #{gainer.rank || index + 1}
            </span>
            <span className="text-xs font-semibold text-green-300 mr-2">
              {gainer.symbol}
            </span>
            <span className="text-xs text-green-400 mr-1">
              ${gainer.price?.toFixed(2) || "0.00"}
            </span>
            <span className="text-xs text-green-500 font-bold">
              +{gainer.change_pct?.toFixed(2) || "0.00"}%
            </span>
            <span className="text-xs ml-2">{gainer.emoji || "📈"}</span>
          </div>
        ))}
      </motion.div>
    </div>
  );
}

