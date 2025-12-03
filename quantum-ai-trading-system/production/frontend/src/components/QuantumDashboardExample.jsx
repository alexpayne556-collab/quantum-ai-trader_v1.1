/**
 * Quantum AI Cockpit — Example Dashboard Component
 * =================================================
 * Shows how to use the API client and hooks to display Plotly graphs
 * This is a reference implementation - adapt to your needs
 */

import React from 'react';
import PlotlyGraph from './PlotlyGraph';
import { useDashboard } from '../hooks/useQuantumAPI';
import { motion } from 'framer-motion';

export default function QuantumDashboardExample({ symbol = 'AAPL' }) {
  const {
    data,
    loading,
    error,
    plotlyData,
    opportunities,
    aiRecommendation,
    marketOpen,
    trend,
    confidence,
    signals,
    metrics
  } = useDashboard(symbol);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="text-cyan-400">Loading forecast data...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="text-red-400">Error: {error}</div>
      </div>
    );
  }

  if (!data) {
    return null;
  }

  // Extract forecast data for PlotlyGraph
  const forecastData = data.forecast || {};

  return (
    <div className="space-y-6">
      {/* Main Plotly Chart */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="glassmorphic-panel rounded-xl p-4"
      >
        <PlotlyGraph
          ticker={symbol}
          forecastData={forecastData}
          height={500}
          theme="dark"
          aiInsight={aiRecommendation}
        />
      </motion.div>

      {/* Opportunities Card */}
      {opportunities && opportunities.length > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="glassmorphic-panel rounded-xl p-4"
        >
          <h3 className="text-xl font-bold text-cyan-400 mb-4">Trading Opportunities</h3>
          <div className="space-y-2">
            {opportunities.map((op, idx) => (
              <div key={idx} className="border border-cyan-500/20 rounded-lg p-3">
                <div className="flex items-center justify-between">
                  <span className={`font-bold ${op.kind === 'dip' ? 'text-green-400' : 'text-red-400'}`}>
                    {op.kind.toUpperCase()}
                  </span>
                  <span className="text-sm text-gray-400">{op.date}</span>
                </div>
                <div className="mt-2 grid grid-cols-4 gap-2 text-sm">
                  <div>
                    <div className="text-gray-400">Entry</div>
                    <div className="text-cyan-400">${op.entry.toFixed(2)}</div>
                  </div>
                  <div>
                    <div className="text-gray-400">Stop</div>
                    <div className="text-red-400">${op.stop.toFixed(2)}</div>
                  </div>
                  <div>
                    <div className="text-gray-400">Target</div>
                    <div className="text-green-400">${op.target.toFixed(2)}</div>
                  </div>
                  <div>
                    <div className="text-gray-400">RR</div>
                    <div className="text-yellow-400">{op.rr.toFixed(2)}</div>
                  </div>
                </div>
                <div className="mt-2 text-xs text-gray-500">{op.rationale}</div>
              </div>
            ))}
          </div>
        </motion.div>
      )}

      {/* AI Recommendation Card */}
      {aiRecommendation && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="glassmorphic-panel rounded-xl p-4"
        >
          <h3 className="text-xl font-bold text-cyan-400 mb-4">AI Recommendation</h3>
          <div className="space-y-3">
            <div>
              <span className="text-gray-400">Stance: </span>
              <span className={`font-bold ${
                aiRecommendation.stance === 'accumulate_on_dips' ? 'text-green-400' :
                aiRecommendation.stance === 'avoid' ? 'text-red-400' : 'text-yellow-400'
              }`}>
                {aiRecommendation.stance.toUpperCase()}
              </span>
            </div>
            <div className="text-cyan-300">{aiRecommendation.summary}</div>
            {aiRecommendation.rationale && aiRecommendation.rationale.length > 0 && (
              <div>
                <div className="text-sm font-semibold text-gray-400 mb-1">Rationale:</div>
                <ul className="list-disc list-inside space-y-1 text-sm text-gray-300">
                  {aiRecommendation.rationale.map((r, idx) => (
                    <li key={idx}>{r}</li>
                  ))}
                </ul>
              </div>
            )}
            {aiRecommendation.next_actions && aiRecommendation.next_actions.length > 0 && (
              <div>
                <div className="text-sm font-semibold text-gray-400 mb-1">Next Actions:</div>
                <ul className="list-disc list-inside space-y-1 text-sm text-cyan-300">
                  {aiRecommendation.next_actions.map((a, idx) => (
                    <li key={idx}>{a}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        </motion.div>
      )}

      {/* Market Status */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="flex items-center gap-4"
      >
        <div className={`px-4 py-2 rounded-lg ${marketOpen ? 'bg-green-500/20 text-green-400' : 'bg-gray-500/20 text-gray-400'}`}>
          Market: {marketOpen ? 'OPEN' : 'CLOSED'}
        </div>
        <div className="px-4 py-2 rounded-lg bg-cyan-500/20 text-cyan-400">
          Trend: {trend.toUpperCase()}
        </div>
        <div className="px-4 py-2 rounded-lg bg-yellow-500/20 text-yellow-400">
          Confidence: {(confidence * 100).toFixed(0)}%
        </div>
      </motion.div>

      {/* Metrics */}
      {metrics && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="glassmorphic-panel rounded-xl p-4"
        >
          <h3 className="text-xl font-bold text-cyan-400 mb-4">Technical Metrics</h3>
          <div className="grid grid-cols-3 gap-4">
            {metrics.rsi && (
              <div>
                <div className="text-gray-400 text-sm">RSI</div>
                <div className="text-cyan-400 text-lg font-bold">{metrics.rsi.toFixed(2)}</div>
              </div>
            )}
            {metrics.atr_pct && (
              <div>
                <div className="text-gray-400 text-sm">ATR %</div>
                <div className="text-cyan-400 text-lg font-bold">{(metrics.atr_pct * 100).toFixed(2)}%</div>
              </div>
            )}
            {metrics.volatility_20d && (
              <div>
                <div className="text-gray-400 text-sm">Volatility (20d)</div>
                <div className="text-cyan-400 text-lg font-bold">{(metrics.volatility_20d * 100).toFixed(2)}%</div>
              </div>
            )}
          </div>
        </motion.div>
      )}
    </div>
  );
}

