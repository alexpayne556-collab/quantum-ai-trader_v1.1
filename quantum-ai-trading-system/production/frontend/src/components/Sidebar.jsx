/**
 * Quantum AI Cockpit — Sidebar Component (Enhanced)
 * 🎯 Module selection with Run / Run Deep Analysis buttons
 * =======================================================
 */

import React, { useEffect, useState } from "react";
import { Link, useLocation } from "react-router-dom";
import { motion } from "framer-motion";
import { useWebSocketFeed } from "../hooks/useWebSocketFeed";
import "./Sidebar.css";

const navItems = [
  { path: "/", label: "🏠 Dashboard", icon: "🏠" },
  { path: "/deep-analysis", label: "🧠 Deep Analysis Lab", icon: "🧠" },
  { path: "/portfolio", label: "💼 Portfolio", icon: "💼" },
  { path: "/watchlist", label: "👁️ Watchlist", icon: "👁️" },
  { path: "/system-health", label: "📊 System Health", icon: "📊" },
  { path: "/system-trust", label: "🎯 System Trust", icon: "🎯" },
];

export default function Sidebar({ onRunModule, onRunDeepAnalysis, currentSymbol = "AAPL" }) {
  const location = useLocation();
  const [activeModule, setActiveModule] = useState(location.pathname);
  const [backendModules, setBackendModules] = useState([]);
  const [selectedModules, setSelectedModules] = useState(new Set());
  const [loading, setLoading] = useState(true);

  // WebSocket connection (resilient, works without WS)
  const { isConnected: wsConnected } = useWebSocketFeed("/ws", {
    autoConnect: true,
    reconnectInterval: 2000,
    maxReconnectAttempts: 20,
  });

  useEffect(() => {
    setActiveModule(location.pathname);
  }, [location]);

  // Fetch backend modules (polling fallback)
  useEffect(() => {
    const fetchModules = async () => {
      try {
        const res = await fetch("/api/system/modules");
        if (!res.ok) {
          // Non-200 response - show disconnected
          setBackendModules([]);
          setLoading(false);
          return;
        }
        const data = await res.json();
        const modules = data.modules || [];
        setBackendModules(modules);
        // If 200 OK but empty, that's fine - just no modules available
      } catch (err) {
        console.error("Failed to fetch modules:", err);
        // Only show error if fetch actually failed (network error, etc.)
      } finally {
        setLoading(false);
      }
    };

    fetchModules();
    const interval = setInterval(fetchModules, 12000);
    return () => clearInterval(interval);
  }, []);

  const handleToggleModule = (moduleId) => {
    setSelectedModules((prev) => {
      const next = new Set(prev);
      if (next.has(moduleId)) {
        next.delete(moduleId);
      } else {
        next.add(moduleId);
      }
      return next;
    });
  };

  const handleRunModule = (moduleId) => {
    if (onRunModule) {
      onRunModule(moduleId, currentSymbol);
    }
  };

  const handleRunDeepAnalysis = () => {
    if (onRunDeepAnalysis && selectedModules.size > 0) {
      onRunDeepAnalysis(Array.from(selectedModules), currentSymbol);
    }
  };

  // Group modules by category
  const byCategory = backendModules.reduce((acc, m) => {
    const cat = m.category || "core";
    if (!acc[cat]) acc[cat] = [];
    acc[cat].push(m);
    return acc;
  }, {});

  return (
    <motion.div
      initial={{ x: -250 }}
      animate={{ x: 0 }}
      transition={{ duration: 0.5, ease: "easeOut" }}
      className="sidebar fixed left-0 top-10 h-[calc(100vh-2.5rem)] w-64 bg-[rgba(13,2,33,0.95)] border-r border-cyan-500/20 shadow-[2px_0_20px_rgba(0,0,0,0.5)] z-30 overflow-y-auto"
      style={{ display: "flex", flexDirection: "column" }}
    >
      <div className="p-4 flex-shrink-0">
        <h2 className="sidebar-title text-electric-blue text-xl font-bold mb-4 border-b border-cyan-500/20 pb-3">
          Navigation
        </h2>
        <ul className="module-list space-y-2">
          {navItems.map((item) => (
            <li key={item.path}>
              <motion.div whileHover={{ scale: 1.05, x: 5 }} whileTap={{ scale: 0.95 }}>
                <Link
                  to={item.path}
                  className={`module-item block px-4 py-3 rounded-lg transition-all duration-300 ${
                    activeModule === item.path
                      ? "bg-cyan-500/20 text-neon-green border-l-4 border-neon-green shadow-neon animate-neon-pulse"
                      : "text-gray-300 hover:bg-cyan-500/10 hover:text-electric-blue"
                  }`}
                >
                  <span className="mr-2">{item.icon}</span>
                  {item.label}
                </Link>
              </motion.div>
            </li>
          ))}
        </ul>
      </div>

      {/* Modules Section */}
      <div className="p-4 flex-1 overflow-y-auto">
        <h3 className="text-electric-blue text-sm font-semibold mb-3 border-b border-cyan-500/20 pb-2">
          Modules
        </h3>
        {loading ? (
          <div className="text-xs text-gray-400 animate-pulse">Loading modules...</div>
        ) : backendModules.length === 0 ? (
          <div className="text-xs text-gray-400">No modules available</div>
        ) : (
          <>
            {Object.entries(byCategory).map(([category, modules]) => (
              <div key={category} className="mb-4">
                <div className="uppercase text-xs tracking-widest opacity-70 mb-1 text-gray-400">
                  {category}
                </div>
                <ul className="space-y-1">
                  {modules.map((module) => {
                    const isSelected = selectedModules.has(module.id);
                    const isRunnable = module.runnable !== false;
                    return (
                      <li key={module.id}>
                        <div className="flex items-center gap-2 px-2 py-1 rounded-lg hover:bg-white/5 transition-colors">
                          {isRunnable && (
                            <input
                              type="checkbox"
                              checked={isSelected}
                              onChange={() => handleToggleModule(module.id)}
                              className="w-3 h-3 rounded border-cyan-500/50 bg-black/40 text-cyan-500 focus:ring-cyan-500"
                            />
                          )}
                          <span className="text-xs text-gray-300 truncate flex-1">
                            {module.name}
                          </span>
                          {isRunnable && (
                            <button
                              onClick={() => handleRunModule(module.id)}
                              className="text-[10px] px-1.5 py-0.5 bg-cyan-500/20 text-cyan-300 rounded hover:bg-cyan-500/30"
                              title="Run module"
                            >
                              ▶
                            </button>
                          )}
                        </div>
                      </li>
                    );
                  })}
                </ul>
              </div>
            ))}
            
            {/* Run Deep Analysis Button */}
            {selectedModules.size > 0 && (
              <div className="mt-4 pt-4 border-t border-cyan-500/20">
                <button
                  onClick={handleRunDeepAnalysis}
                  className="w-full px-4 py-2 bg-gradient-to-r from-cyan-500/20 to-green-500/20 text-cyan-300 rounded-lg hover:from-cyan-500/30 hover:to-green-500/30 transition-all font-semibold text-sm"
                >
                  🚀 Run Deep Analysis ({selectedModules.size})
                </button>
              </div>
            )}
          </>
        )}
      </div>

      {/* System Status */}
      <div className="absolute bottom-4 left-4 right-4 p-4 glassmorphic-panel rounded-lg flex-shrink-0">
        <div className="text-xs text-gray-400 mb-2">System Status</div>
        <div className="flex items-center gap-2">
          <div
            className={`w-2 h-2 rounded-full ${
              wsConnected ? "bg-neon-green animate-pulse" : "bg-magenta"
            }`}
          />
          <span className={`text-sm ${wsConnected ? "text-neon-green" : "text-magenta"}`}>
            {wsConnected ? "Operational" : "Disconnected"}
          </span>
        </div>
      </div>
    </motion.div>
  );
}
