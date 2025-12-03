/**
 * Quantum AI Cockpit — Ticker Search Component
 * 🔍 Autocomplete ticker search
 * =============================
 */

import React, { useState, useEffect, useRef } from "react";

export default function TickerSearch({ onSelect, defaultValue = "AAPL" }) {
  const [query, setQuery] = useState(defaultValue);
  const [results, setResults] = useState([]);
  const [showResults, setShowResults] = useState(false);
  const [selectedIndex, setSelectedIndex] = useState(-1);
  const inputRef = useRef(null);
  const timeoutRef = useRef(null);
  
  useEffect(() => {
    if (query.length < 1) {
      setResults([]);
      setShowResults(false);
      return;
    }
    
    // Debounce search
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
    }
    
    timeoutRef.current = setTimeout(async () => {
      try {
        const res = await fetch(`/api/tickers/search?query=${encodeURIComponent(query)}&limit=10`);
        const data = await res.json();
        setResults(data.results || []);
        setShowResults(true);
        setSelectedIndex(-1);
      } catch (err) {
        console.error("Ticker search error:", err);
        setResults([]);
      }
    }, 300);
    
    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
    };
  }, [query]);
  
  const handleSelect = (ticker) => {
    setQuery(ticker.symbol);
    setShowResults(false);
    if (onSelect) {
      onSelect(ticker.symbol);
    }
  };
  
  const handleKeyDown = (e) => {
    if (!showResults || results.length === 0) return;
    
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setSelectedIndex((prev) => (prev < results.length - 1 ? prev + 1 : prev));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setSelectedIndex((prev) => (prev > 0 ? prev - 1 : -1));
    } else if (e.key === "Enter" && selectedIndex >= 0) {
      e.preventDefault();
      handleSelect(results[selectedIndex]);
    } else if (e.key === "Escape") {
      setShowResults(false);
    }
  };
  
  return (
    <div className="relative w-full">
      <input
        ref={inputRef}
        type="text"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        onFocus={() => query.length > 0 && setShowResults(true)}
        onBlur={() => setTimeout(() => setShowResults(false), 200)}
        onKeyDown={handleKeyDown}
        placeholder="Search ticker..."
        className="w-full px-4 py-2 bg-black/40 border border-cyan-500/30 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:border-cyan-500 focus:ring-1 focus:ring-cyan-500"
      />
      {showResults && results.length > 0 && (
        <div className="absolute z-50 w-full mt-1 bg-[rgba(13,2,33,0.95)] border border-cyan-500/30 rounded-lg shadow-lg max-h-60 overflow-y-auto">
          {results.map((item, idx) => (
            <div
              key={item.symbol}
              onClick={() => handleSelect(item)}
              className={`px-4 py-2 cursor-pointer hover:bg-cyan-500/10 ${
                idx === selectedIndex ? "bg-cyan-500/20" : ""
              }`}
            >
              <div className="text-cyan-400 font-semibold">{item.symbol}</div>
              <div className="text-gray-400 text-sm">{item.name}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

