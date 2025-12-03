/**
 * Quantum AI Cockpit — SearchBar Component
 * 🔍 Dynamic ticker search with debounce and context dispatch
 * ==========================================================
 */

import React, { useState, useEffect, useCallback } from "react";
import { motion } from "framer-motion";
import { useDebounce } from "../hooks/useDebounce";

export default function SearchBar({ onTickerChange, initialTicker = "AAPL" }) {
  const [ticker, setTicker] = useState(initialTicker);
  const [isValid, setIsValid] = useState(true);
  const debouncedTicker = useDebounce(ticker, 500);

  // Validate ticker format
  useEffect(() => {
    const validateTicker = (symbol) => {
      if (!symbol || symbol.trim().length === 0) {
        return false;
      }
      // Basic validation: 1-5 uppercase alphanumeric characters
      return /^[A-Z0-9]{1,5}$/.test(symbol.toUpperCase().trim());
    };

    setIsValid(validateTicker(ticker));
  }, [ticker]);

  // Dispatch ticker change event when debounced value changes
  useEffect(() => {
    if (debouncedTicker && isValid && debouncedTicker.toUpperCase() !== initialTicker) {
      const normalizedTicker = debouncedTicker.toUpperCase().trim();
      if (onTickerChange) {
        onTickerChange(normalizedTicker);
      }
      // Dispatch global event for other components
      window.dispatchEvent(
        new CustomEvent("SET_TICKER", { detail: { ticker: normalizedTicker } })
      );
    }
  }, [debouncedTicker, isValid, onTickerChange, initialTicker]);

  const handleChange = (e) => {
    const value = e.target.value.toUpperCase().replace(/[^A-Z0-9]/g, "");
    setTicker(value);
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: -10 }}
      animate={{ opacity: 1, y: 0 }}
      className="relative"
    >
      <div className="relative flex items-center">
        <input
          type="text"
          value={ticker}
          onChange={handleChange}
          placeholder="Enter symbol (e.g., AAPL)"
          className={`px-4 py-2 pr-10 bg-[rgba(15,15,15,0.9)] border rounded-lg text-white font-mono focus:outline-none focus:ring-2 transition-all duration-300 ${
            isValid
              ? "border-cyan-500/30 focus:border-cyan-500 focus:ring-cyan-500/50"
              : "border-red-500/50 focus:border-red-500 focus:ring-red-500/50"
          }`}
          maxLength={5}
        />
        <div className="absolute right-3 flex items-center gap-2">
          {isValid ? (
            <motion.div
              animate={{ scale: [1, 1.2, 1] }}
              transition={{ duration: 1, repeat: Infinity, repeatDelay: 2 }}
              className="w-2 h-2 bg-neon-green rounded-full"
            />
          ) : (
            <div className="w-2 h-2 bg-red-500 rounded-full" />
          )}
        </div>
      </div>
      {!isValid && ticker.length > 0 && (
        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="text-red-400 text-xs mt-1 font-mono"
        >
          Invalid symbol format
        </motion.p>
      )}
    </motion.div>
  );
}

