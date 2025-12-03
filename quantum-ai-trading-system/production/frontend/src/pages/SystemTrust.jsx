/**
 * Quantum AI Cockpit — System Trust Page
 * 🎯 Detailed system trust metrics and monitoring
 * ==============================================
 */

import React from "react";
import { motion } from "framer-motion";
import TrustMeter from "../components/TrustMeter";
import "../styles/animations.css";

export default function SystemTrust() {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      className="p-6 space-y-6 min-h-screen"
    >
      <h1 className="text-3xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-green-400 font-orbitron">
        🎯 System Trust
      </h1>
      
      <div className="flex justify-center">
        <TrustMeter />
      </div>
    </motion.div>
  );
}
