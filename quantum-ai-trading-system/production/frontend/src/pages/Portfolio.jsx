/**
 * Quantum AI Cockpit — Portfolio Page
 * 💼 Portfolio management and tracking
 * ====================================
 */

import React from "react";
import { motion } from "framer-motion";
import PortfolioManager from "../components/PortfolioManager";
import "../styles/animations.css";

export default function Portfolio() {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      className="p-6 space-y-6"
    >
      <h1 className="text-3xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-green-400">
        💼 Portfolio
      </h1>
      <PortfolioManager />
    </motion.div>
  );
}

