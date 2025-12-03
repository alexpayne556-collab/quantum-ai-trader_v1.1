/**
 * Quantum AI Cockpit — System Status Ribbon
 * 📊 Animated status ribbon showing backend module count, health %, and latency
 * ============================================================================
 */

import React, { useState, useEffect } from "react";
import { motion } from "framer-motion";
import "../styles/animations.css";

export default function SystemStatusRibbon() {
  const [status, setStatus] = useState({
    moduleCount: 0,
    healthPercent: 0,
    latency: 0,
    bindings: 0,
  });
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const startTime = performance.now();
        const res = await fetch("/api/system/modules");
        const data = await res.json();
        const latency = Math.round(performance.now() - startTime);

        // Support both new format (objects) and legacy format (strings)
        const modulesRaw = data.modules || [];
        const moduleObjects = modulesRaw.length > 0 && typeof modulesRaw[0] === "object"
          ? modulesRaw
          : modulesRaw.map(name => ({ name, active: true }));
        const activeModules = moduleObjects.filter((m) => m.active).length;
        const healthPercent = moduleObjects.length > 0
          ? Math.round((activeModules / moduleObjects.length) * 100)
          : 0;

        setStatus({
          moduleCount: moduleObjects.length,
          healthPercent,
          latency,
          bindings: Object.keys(data.bindings || {}).length,
        });
      } catch (err) {
        console.error("Status fetch error:", err);
      } finally {
        setLoading(false);
      }
    };

    fetchStatus();
    // Refresh every 5 seconds
    const interval = setInterval(fetchStatus, 5000);
    return () => clearInterval(interval);
  }, []);

  if (loading) {
    return (
      <div className="h-10 bg-[rgba(13,2,33,0.9)] border-b border-cyan-500/20 animate-gradient-shimmer" />
    );
  }

  return (
    <motion.div
      initial={{ y: -40 }}
      animate={{ y: 0 }}
      className="h-10 bg-[rgba(13,2,33,0.9)] border-b border-cyan-500/20 flex items-center justify-center gap-6 text-xs font-mono text-gray-300 shadow-[0_2px_10px_rgba(0,0,0,0.3)]"
    >
      <div className="flex items-center gap-2">
        <span className="text-gray-400">Modules:</span>
        <span className="text-neon-green font-bold">{status.moduleCount}</span>
      </div>
      <div className="flex items-center gap-2">
        <span className="text-gray-400">Health:</span>
        <span
          className={`font-bold ${
            status.healthPercent >= 80
              ? "text-neon-green"
              : status.healthPercent >= 50
              ? "text-gold"
              : "text-magenta"
          }`}
        >
          {status.healthPercent}%
        </span>
      </div>
      <div className="flex items-center gap-2">
        <span className="text-gray-400">Bindings:</span>
        <span className="text-electric-blue font-bold">{status.bindings}</span>
      </div>
      <div className="flex items-center gap-2">
        <span className="text-gray-400">Latency:</span>
        <span
          className={`font-bold ${
            status.latency < 100
              ? "text-neon-green"
              : status.latency < 500
              ? "text-gold"
              : "text-magenta"
          }`}
        >
          {status.latency}ms
        </span>
      </div>
      <div className="flex items-center gap-2">
        <div
          className={`w-2 h-2 rounded-full ${
            status.healthPercent >= 80
              ? "bg-neon-green animate-pulse"
              : "bg-magenta"
          }`}
        />
        <span className="text-gray-400">System Operational</span>
      </div>
    </motion.div>
  );
}

