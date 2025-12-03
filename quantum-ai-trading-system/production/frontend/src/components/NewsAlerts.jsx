// frontend/src/components/NewsAlerts.jsx
import React, { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";

export default function NewsAlerts({ symbols = [] }) {
  const [alerts, setAlerts] = useState([]);
  const [loading, setLoading] = useState(true);
  const [expanded, setExpanded] = useState(null);

  useEffect(() => {
    if (symbols.length === 0) {
      setLoading(false);
      return;
    }

    const fetchAlerts = async () => {
      try {
        const symbolsStr = symbols.join(",");
        const response = await fetch(`/api/news/alerts?symbols=${symbolsStr}&min_sentiment=0.7`);
        if (response.ok) {
          const data = await response.json();
          if (data.status === "success") {
            setAlerts(data.alerts || []);
          }
        }
        setLoading(false);
      } catch (error) {
        console.error("Failed to fetch news alerts:", error);
        setLoading(false);
      }
    };

    fetchAlerts();
    const interval = setInterval(fetchAlerts, 60000); // Update every minute
    return () => clearInterval(interval);
  }, [symbols]);

  if (loading) {
    return (
      <div className="bg-[rgba(15,15,15,0.9)] backdrop-blur-md border border-cyan-500/20 rounded-xl shadow-[0_0_25px_rgba(0,255,170,0.1)] p-4">
        <div className="text-cyan-400 text-sm">Loading news alerts...</div>
      </div>
    );
  }

  if (alerts.length === 0) {
    return (
      <div className="bg-[rgba(15,15,15,0.9)] backdrop-blur-md border border-cyan-500/20 rounded-xl shadow-[0_0_25px_rgba(0,255,170,0.1)] p-4">
        <h3 className="text-electric-blue text-lg font-semibold mb-2 font-orbitron">
          📰 News Alerts
        </h3>
        <div className="text-center text-gray-400 py-4 text-sm">
          No news alerts at this time
        </div>
      </div>
    );
  }

  return (
    <div className="bg-[rgba(15,15,15,0.9)] backdrop-blur-md border border-cyan-500/20 rounded-xl shadow-[0_0_25px_rgba(0,255,170,0.1)] p-4">
      <div className="flex justify-between items-center mb-4">
        <h3 className="text-electric-blue text-lg font-semibold font-orbitron">
          📰 News Alerts
        </h3>
        <span className="text-xs text-cyan-500/70">
          {alerts.length} alert{alerts.length !== 1 ? 's' : ''}
        </span>
      </div>

      <div className="space-y-2 max-h-96 overflow-y-auto">
        <AnimatePresence>
          {alerts.map((alert, index) => {
            const isBullish = alert.sentiment === "bullish";
            const isHighPriority = alert.priority === "high";

            return (
              <motion.div
                key={`${alert.symbol}-${index}`}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 20 }}
                className={`p-3 rounded-lg border cursor-pointer transition-all ${
                  isBullish
                    ? "bg-[rgba(0,255,0,0.1)] border-green-500/30 hover:border-green-500/50"
                    : "bg-[rgba(255,0,122,0.1)] border-magenta/30 hover:border-magenta/50"
                } ${isHighPriority ? "ring-2 ring-gold/50" : ""}`}
                onClick={() => setExpanded(expanded === index ? null : index)}
              >
                <div className="flex items-start justify-between mb-2">
                  <div className="flex items-center gap-2">
                    <span className="font-semibold text-neon-green">{alert.symbol}</span>
                    <span className={`text-xs px-2 py-1 rounded ${
                      isBullish
                        ? "bg-green-500/20 text-green-400"
                        : "bg-magenta/20 text-magenta"
                    }`}>
                      {alert.sentiment.toUpperCase()}
                    </span>
                    {isHighPriority && (
                      <span className="text-xs px-2 py-1 rounded bg-gold/20 text-gold">
                        HIGH PRIORITY
                      </span>
                    )}
                  </div>
                  <span className="text-xs text-gray-400">
                    {new Date(alert.published_at).toLocaleTimeString()}
                  </span>
                </div>

                <div className="text-sm text-gray-300 mb-2 line-clamp-2">
                  {alert.title}
                </div>

                <div className="flex items-center justify-between text-xs">
                  <span className="text-cyan-500/70">{alert.source}</span>
                  <span className="text-gray-400">
                    Sentiment: {((alert.sentiment_score || 0.5) * 100).toFixed(0)}%
                  </span>
                </div>

                <AnimatePresence>
                  {expanded === index && (
                    <motion.div
                      initial={{ height: 0, opacity: 0 }}
                      animate={{ height: "auto", opacity: 1 }}
                      exit={{ height: 0, opacity: 0 }}
                      className="mt-2 pt-2 border-t border-cyan-500/20"
                    >
                      <a
                        href={alert.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-xs text-cyan-400 hover:text-cyan-300 block"
                        onClick={(e) => e.stopPropagation()}
                      >
                        Read full article →
                      </a>
                    </motion.div>
                  )}
                </AnimatePresence>
              </motion.div>
            );
          })}
        </AnimatePresence>
      </div>
    </div>
  );
}

