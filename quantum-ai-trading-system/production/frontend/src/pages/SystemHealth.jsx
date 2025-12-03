/**
 * Quantum AI Cockpit — System Health Page
 * 📊 System status, module health, and performance metrics
 * =======================================================
 */

import React, { useState, useEffect } from "react";
import { motion } from "framer-motion";
import "../styles/animations.css";

export default function SystemHealth() {
  const [moduleStatus, setModuleStatus] = useState(null);
  const [systemHealth, setSystemHealth] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchHealthData = async () => {
      try {
        // Fetch module status
        const modulesRes = await fetch("/api/system/modules");
        const modulesData = await modulesRes.json();
        setModuleStatus(modulesData);

        // Fetch system health
        const healthRes = await fetch("/api/system/health");
        const healthData = await healthRes.json();
        setSystemHealth(healthData);
      } catch (err) {
        console.error("Health data fetch error:", err);
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    fetchHealthData();
    // Refresh every 10 seconds
    const interval = setInterval(fetchHealthData, 10000);
    return () => clearInterval(interval);
  }, []);

  if (loading) {
    return (
      <div className="p-6">
        <div className="animate-gradient-shimmer h-96 rounded-lg" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-6 text-red-400">
        Error loading health data: {error}
      </div>
    );
  }

  // Support both new format (objects) and legacy format (strings)
  const modulesRaw = moduleStatus?.modules || [];
  const modules = modulesRaw.length > 0 && typeof modulesRaw[0] === "object"
    ? modulesRaw
    : modulesRaw.map(name => ({ name, active: true, accuracy: 0.85, partners: [], description: `${name} module` }));
  const bindings = moduleStatus?.bindings || {};

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="p-6 space-y-6"
    >
      {/* Header */}
      <h1 className="text-3xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-green-400">
        📊 System Health
      </h1>

      {/* Overall Health */}
      {systemHealth && (
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          className="glassmorphic-panel rounded-xl p-6 mb-6"
        >
          <div className="text-sm text-gray-400 mb-2">Overall System Health</div>
          <div className="text-5xl font-extrabold mb-2 text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-green-400">
            {Math.round(systemHealth.overall || 0)}%
          </div>
          <div className="text-lg text-gray-300">{systemHealth.summary || "—"}</div>
        </motion.div>
      )}

      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="glassmorphic-panel rounded-xl p-4"
        >
          <div className="text-sm text-gray-400 mb-1">Total Modules</div>
          <div className="text-2xl font-bold text-neon-green">
            {moduleStatus?.modules?.length || 0}
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="glassmorphic-panel rounded-xl p-4"
        >
          <div className="text-sm text-gray-400 mb-1">Healthy Modules</div>
          <div className="text-2xl font-bold text-electric-blue">
            {systemHealth?.by_module ? Object.values(systemHealth.by_module).filter(s => s === "ok").length : 0}
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="glassmorphic-panel rounded-xl p-4"
        >
          <div className="text-sm text-gray-400 mb-1">Failed Modules</div>
          <div className="text-2xl font-bold text-red-400">
            {systemHealth?.by_module ? Object.values(systemHealth.by_module).filter(s => s === "fail").length : 0}
          </div>
        </motion.div>
      </div>

      {/* Module Status Grid */}
      <div className="glassmorphic-panel rounded-xl p-4">
        <h2 className="text-xl font-semibold text-electric-blue mb-4">
          Module Status
        </h2>
        {systemHealth?.by_module ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-2">
            {Object.entries(systemHealth.by_module).map(([id, status], idx) => {
              const module = modules.find(m => (m.id || m.name) === id) || { name: id, group: "core" };
              return (
                <motion.div
                  key={id}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: idx * 0.02 }}
                  className="flex items-center justify-between bg-white/5 rounded-xl p-3 hover:bg-white/10 transition-colors"
                >
                  <span className="truncate text-sm text-gray-300">{module.name || id}</span>
                  <span
                    className={`ml-2 text-xs px-2 py-0.5 rounded-full flex-shrink-0 ${
                      status === "ok"
                        ? "bg-green-500/20 text-green-300"
                        : status === "warn"
                        ? "bg-yellow-500/20 text-yellow-300"
                        : status === "fail"
                        ? "bg-red-500/20 text-red-300"
                        : "bg-slate-500/20 text-slate-300"
                    }`}
                  >
                    {status || "unknown"}
                  </span>
                </motion.div>
              );
            })}
          </div>
        ) : (
          <div className="text-gray-400">No module status data available</div>
        )}
      </div>

      {/* Bindings Overview */}
      <div className="glassmorphic-panel rounded-xl p-4">
        <h2 className="text-xl font-semibold text-magenta mb-4">
          Module Bindings
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {Object.entries(bindings).map(([module, partners], idx) => (
            <motion.div
              key={module}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: idx * 0.1 }}
              className="border border-cyan-500/20 rounded-lg p-3"
            >
              <div className="font-semibold text-electric-blue mb-2">
                {module}
              </div>
              <div className="text-sm text-gray-400 space-y-1">
                {partners.map((partner) => (
                  <div key={partner}>→ {partner}</div>
                ))}
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </motion.div>
  );
}

