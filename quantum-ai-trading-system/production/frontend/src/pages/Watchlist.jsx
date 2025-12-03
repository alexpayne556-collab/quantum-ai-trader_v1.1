/**
 * Quantum AI Cockpit — Watchlist Page
 * 👁️ Watchlist management and monitoring
 * ======================================
 */

import React from "react";
import { motion } from "framer-motion";
import WatchlistPanel from "../components/WatchlistPanel";
import "../styles/animations.css";

export default function Watchlist() {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      className="p-6 space-y-6"
    >
      <h1 className="text-3xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-green-400">
        👁️ Watchlist
      </h1>
      <WatchlistPanel />
    </motion.div>
  );
}

