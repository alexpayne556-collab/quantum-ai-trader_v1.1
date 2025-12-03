/**
 * Quantum AI Cockpit — Enhanced TrustMeter Component
 * 🎯 Circular neon gauge with animated glow and real-time updates
 * ===============================================================
 */

import React, { useEffect, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { CircularProgressbar, buildStyles } from "react-circular-progressbar";
import "react-circular-progressbar/dist/styles.css";

export default function TrustMeter() {
  const [trust, setTrust] = useState(0);
  const [status, setStatus] = useState("Calibrating...");
  const [loading, setLoading] = useState(true);
  const [previousTrust, setPreviousTrust] = useState(0);

  useEffect(() => {
    async function fetchTrust() {
      try {
        // Try to fetch from system trust endpoint, fallback to modules health
        let res = await fetch("/api/system/trust").catch(() => null);
        
        if (!res || !res.ok) {
          // Fallback: calculate trust from module health
          res = await fetch("/api/system/modules");
          if (res.ok) {
            const data = await res.json();
            const modules = data.modules || [];
            const activeModules = modules.filter((m) => m.active !== false).length;
            const healthPercent = modules.length > 0
              ? Math.round((activeModules / modules.length) * 100)
              : 0;
            
            setPreviousTrust(trust);
            setTrust(healthPercent);
            
            if (healthPercent >= 85) setStatus("Stable 🟢");
            else if (healthPercent >= 60) setStatus("Caution ⚠️");
            else setStatus("Unstable 🔴");
          } else {
            setStatus("Offline ❌");
          }
        } else {
          const data = await res.json();
          const value = data?.trust_score ?? data?.trust ?? 0;
          setPreviousTrust(trust);
          setTrust(value);
          
          if (value >= 85) setStatus("Stable 🟢");
          else if (value >= 60) setStatus("Caution ⚠️");
          else setStatus("Unstable 🔴");
        }
      } catch (err) {
        console.error("Trust fetch error:", err);
        setStatus("Offline ❌");
      } finally {
        setLoading(false);
      }
    }

    fetchTrust();
    const interval = setInterval(fetchTrust, 10000); // Refresh every 10s
    return () => clearInterval(interval);
  }, [trust]);

  // OPTIMIZED v29: Determine color and luminosity based on trust level (real-time updates)
  const color =
    trust >= 85
      ? "#00ffaa" // neon green
      : trust >= 60
      ? "#ffbb00" // gold
      : "#ff007a"; // magenta

  // OPTIMIZED v29: Dynamic glow intensity that scales with confidence
  const glowIntensity = Math.min(trust / 100, 1.0);
  const luminosity = 0.5 + (trust / 100) * 0.5; // Scale from 0.5 to 1.0

  // Animation key for smooth transitions
  const trustChange = trust !== previousTrust;

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.8 }}
      animate={{ 
        opacity: 1, 
        scale: 1,
        filter: `brightness(${luminosity})` // OPTIMIZED v29: Real-time luminosity adjustment
      }}
      transition={{ duration: 1.5, ease: "easeOut" }}
      className="glassmorphic-panel rounded-xl p-6 flex flex-col items-center justify-center text-center shadow-lg relative overflow-hidden gpu-accelerated"
      style={{
        minWidth: "240px",
        background:
          "radial-gradient(circle at center, rgba(0,255,255,0.08), rgba(0,0,0,0.6))",
        // OPTIMIZED v29: Dynamic glow that scales with trust
        boxShadow: `0 0 ${20 * glowIntensity}px ${color}40, 0 0 ${40 * glowIntensity}px ${color}20`,
      }}
    >
      {/* OPTIMIZED v29: Animated background glow with breathing effect */}
      <motion.div
        className="absolute inset-0 rounded-xl"
        animate={{
          boxShadow: [
            `0 0 ${20 * glowIntensity}px ${color}80`,
            `0 0 ${40 * glowIntensity}px ${color}60`,
            `0 0 ${20 * glowIntensity}px ${color}80`,
          ],
          filter: [`brightness(${luminosity})`, `brightness(${luminosity * 1.2})`, `brightness(${luminosity})`],
        }}
        transition={{ repeat: Infinity, duration: 4, ease: "easeInOut" }}
        style={{
          background: `radial-gradient(circle at center, ${color}10, transparent 70%)`,
        }}
      />

      {/* Circular progress gauge */}
      <motion.div
        key={trust}
        initial={trustChange ? { scale: 1.1, rotate: -5 } : false}
        animate={{ scale: 1, rotate: 0 }}
        transition={{ duration: 0.5, ease: "easeOut" }}
        className="relative z-10"
        style={{
          width: "180px",
          height: "180px",
          marginBottom: "1rem",
        }}
      >
        <motion.div
          animate={{
            filter: [
              `drop-shadow(0 0 ${10 * glowIntensity}px ${color})`,
              `drop-shadow(0 0 ${20 * glowIntensity}px ${color})`,
              `drop-shadow(0 0 ${10 * glowIntensity}px ${color})`,
            ],
          }}
          transition={{ repeat: Infinity, duration: 4, ease: "easeInOut" }}
          className="gpu-accelerated"
        >
          <CircularProgressbar
            value={trust}
            text={`${trust.toFixed(0)}%`}
            styles={buildStyles({
              textColor: "#fff",
              pathColor: color,
              trailColor: "rgba(255,255,255,0.1)",
              textSize: "24px",
              pathTransitionDuration: 1.5, // OPTIMIZED v29: Smooth progress animation
              pathTransition: "easeInOut",
            })}
          />
        </motion.div>
      </motion.div>

      {/* Status text with animated color */}
      <AnimatePresence mode="wait">
        <motion.p
          key={status}
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: 10 }}
          className="mt-2 text-lg font-bold font-orbitron"
          style={{ color }}
        >
          {status}
        </motion.p>
      </AnimatePresence>

      {/* Subtitle */}
      <p className="text-xs text-gray-400 mt-2 font-mono">
        System Trust Index
      </p>

      {/* Loading indicator */}
      {loading && (
        <motion.div
          className="absolute top-2 right-2"
          animate={{ rotate: 360 }}
          transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
        >
          <div className="w-3 h-3 border-2 border-cyan-500/30 border-t-cyan-500 rounded-full" />
        </motion.div>
      )}
    </motion.div>
  );
}