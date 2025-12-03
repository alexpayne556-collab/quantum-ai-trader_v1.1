/**
 * Quantum AI Cockpit — Main App Component
 * 🚀 Main application with routing and system status
 * ==================================================
 */

import React, { useState, useEffect } from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import { motion, AnimatePresence } from "framer-motion";
import Sidebar from "./components/Sidebar";
import SystemStatusRibbon from "./components/SystemStatusRibbon";
import MarketTicker from "./components/MarketTicker";
import TopGainersTicker from "./components/TopGainersTicker";
import Dashboard from "./pages/Dashboard";
import DeepAnalysisLab from "./pages/DeepAnalysisLab";
import SystemHealth from "./pages/SystemHealth";
import SystemTrust from "./pages/SystemTrust";
import Portfolio from "./pages/Portfolio";
import Watchlist from "./pages/Watchlist";
import "./index.css";

function App() {
  const [dashboardHandlers, setDashboardHandlers] = useState(null);
  const [currentSymbol, setCurrentSymbol] = useState("AAPL");

  // Listen for dashboard handlers registration
  useEffect(() => {
    const checkHandlers = setInterval(() => {
      if (window.dashboardHandlers) {
        setDashboardHandlers(window.dashboardHandlers);
        clearInterval(checkHandlers);
      }
    }, 100);
    return () => clearInterval(checkHandlers);
  }, []);

  return (
    <Router>
      <div className="app-container min-h-screen bg-gradient-to-br from-[#0D0221] via-[#050014] to-[#0D0221]">
        {/* System Status Ribbon */}
        <SystemStatusRibbon />

        {/* MSNBC-style Market Ticker */}
        <div className="fixed top-10 left-64 right-0 z-20">
          <MarketTicker limit={50} />
        </div>

        {/* Top Gainers Ticker */}
        <div className="fixed top-[5.5rem] left-64 right-0 z-20">
          <TopGainersTicker limit={20} />
        </div>

        <div className="flex pt-24">
          {/* Sidebar */}
          <Sidebar
            onRunModule={dashboardHandlers?.runModule}
            onRunDeepAnalysis={dashboardHandlers?.runDeepAnalysis}
            currentSymbol={currentSymbol}
          />

          {/* Main Content */}
          <main className="flex-1 ml-64 p-6">
            <AnimatePresence mode="wait">
              <Routes>
                <Route
                  path="/"
                  element={<Dashboard onSymbolChange={setCurrentSymbol} />}
                />
                <Route
                  path="/deep-analysis"
                  element={<DeepAnalysisLab />}
                />
                <Route
                  path="/system-health"
                  element={<SystemHealth />}
                />
                <Route
                  path="/system-trust"
                  element={<SystemTrust />}
                />
                <Route
                  path="/portfolio"
                  element={<Portfolio />}
                />
                <Route
                  path="/watchlist"
                  element={<Watchlist />}
                />
              </Routes>
            </AnimatePresence>
          </main>
        </div>
      </div>
    </Router>
  );
}

export default App;
