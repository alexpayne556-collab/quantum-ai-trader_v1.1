/**
 * Quantum AI Cockpit — Plotly Dashboard Component
 * ===============================================
 * Advanced Plotly visualization dashboard with multiple chart types
 */

import React, { useState, useEffect, useRef } from 'react';
import { motion } from 'framer-motion';
import PlotlyGraph from './PlotlyGraph';
import { fetchAPISafe } from '../api/fetchHandler';
import { formatForecastData, formatPatternData } from '../utils/schemaFormatter';
import { API_ENDPOINTS } from '../config/apiConfig';

export default function PlotlyDashboard({ symbol = 'AAPL', onSymbolChange }) {
  const [forecastData, setForecastData] = useState(null);
  const [patternData, setPatternData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState('forecast');

  // Fetch data on symbol change
  useEffect(() => {
    const fetchData = async () => {
      setLoading(true);
      setError(null);

      try {
        // Fetch forecast and pattern data concurrently
        const [forecast, pattern] = await Promise.all([
          fetchAPISafe(API_ENDPOINTS.FORECAST(symbol), {}, null),
          fetchAPISafe(API_ENDPOINTS.PATTERNS(symbol), {}, null),
        ]);

        if (forecast) {
          setForecastData(formatForecastData(forecast));
        }

        if (pattern) {
          setPatternData(formatPatternData(pattern));
        }
      } catch (err) {
        console.error('Dashboard data fetch error:', err);
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    if (symbol) {
      fetchData();
    }
  }, [symbol]);

  // Auto-refresh every 30 seconds
  useEffect(() => {
    if (!symbol) return;

    const interval = setInterval(() => {
      const fetchData = async () => {
        try {
          const [forecast, pattern] = await Promise.all([
            fetchAPISafe(API_ENDPOINTS.FORECAST(symbol), {}, null),
            fetchAPISafe(API_ENDPOINTS.PATTERNS(symbol), {}, null),
          ]);

          if (forecast) setForecastData(formatForecastData(forecast));
          if (pattern) setPatternData(formatPatternData(pattern));
        } catch (err) {
          console.error('Auto-refresh error:', err);
        }
      };
      fetchData();
    }, 30000);

    return () => clearInterval(interval);
  }, [symbol]);

  if (loading && !forecastData && !patternData) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-cyan-400 animate-pulse">Loading dashboard data...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-red-500/10 border border-red-500/50 rounded-lg p-4">
        <div className="text-red-400">Error: {error}</div>
      </div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="space-y-4"
    >
      {/* Tab Navigation */}
      <div className="flex gap-2 border-b border-cyan-500/20">
        <button
          onClick={() => setActiveTab('forecast')}
          className={`px-4 py-2 transition-colors ${
            activeTab === 'forecast'
              ? 'text-cyan-400 border-b-2 border-cyan-400'
              : 'text-gray-400 hover:text-cyan-300'
          }`}
        >
          Forecast
        </button>
        <button
          onClick={() => setActiveTab('pattern')}
          className={`px-4 py-2 transition-colors ${
            activeTab === 'pattern'
              ? 'text-cyan-400 border-b-2 border-cyan-400'
              : 'text-gray-400 hover:text-cyan-300'
          }`}
        >
          Patterns
        </button>
        <button
          onClick={() => setActiveTab('combined')}
          className={`px-4 py-2 transition-colors ${
            activeTab === 'combined'
              ? 'text-cyan-400 border-b-2 border-cyan-400'
              : 'text-gray-400 hover:text-cyan-300'
          }`}
        >
          Combined
        </button>
      </div>

      {/* Forecast Chart */}
      {activeTab === 'forecast' && forecastData && (
        <div className="bg-[rgba(15,15,15,0.9)] backdrop-blur-md border border-cyan-500/20 rounded-xl p-4">
          <h3 className="text-xl font-bold text-cyan-400 mb-4">
            {symbol} — 14-Day Forecast
          </h3>
          <PlotlyGraph
            data={[
              {
                x: forecastData.dates,
                y: forecastData.historical,
                type: 'scatter',
                mode: 'lines',
                name: 'Historical',
                line: { color: '#00d1ff' },
              },
              {
                x: forecastData.dates.slice(-14),
                y: forecastData.forecast,
                type: 'scatter',
                mode: 'lines',
                name: 'Forecast',
                line: { color: '#00ffaa', dash: 'dash' },
              },
              {
                x: forecastData.dates.slice(-14),
                y: forecastData.upperBand,
                type: 'scatter',
                mode: 'lines',
                name: 'Upper Band',
                line: { color: '#00ffaa', width: 1, dash: 'dot' },
                showlegend: false,
              },
              {
                x: forecastData.dates.slice(-14),
                y: forecastData.lowerBand,
                type: 'scatter',
                mode: 'lines',
                name: 'Lower Band',
                line: { color: '#00ffaa', width: 1, dash: 'dot' },
                fill: 'tonexty',
                fillcolor: 'rgba(0, 255, 170, 0.1)',
              },
            ]}
            layout={{
              paper_bgcolor: 'rgba(0,0,0,0)',
              plot_bgcolor: 'rgba(0,0,0,0)',
              font: { color: '#00d1ff' },
              xaxis: { gridcolor: 'rgba(0, 209, 255, 0.1)' },
              yaxis: { gridcolor: 'rgba(0, 209, 255, 0.1)' },
              hovermode: 'x unified',
            }}
          />
          {forecastData.fii && (
            <div className="mt-4 text-sm text-gray-400">
              Fusion Intelligence Index (FII): <span className="text-cyan-400">{forecastData.fii.toFixed(2)}</span>
            </div>
          )}
        </div>
      )}

      {/* Pattern Chart */}
      {activeTab === 'pattern' && patternData && (
        <div className="bg-[rgba(15,15,15,0.9)] backdrop-blur-md border border-cyan-500/20 rounded-xl p-4">
          <h3 className="text-xl font-bold text-cyan-400 mb-4">
            {symbol} — Technical Patterns
          </h3>
          <PlotlyGraph
            data={[
              {
                x: patternData.dates || [],
                y: patternData.close,
                type: 'scatter',
                mode: 'lines',
                name: 'Close Price',
                line: { color: '#00d1ff' },
              },
              {
                x: patternData.dates || [],
                y: patternData.bollinger.upper,
                type: 'scatter',
                mode: 'lines',
                name: 'Upper Bollinger',
                line: { color: '#00ffaa', width: 1, dash: 'dot' },
              },
              {
                x: patternData.dates || [],
                y: patternData.bollinger.lower,
                type: 'scatter',
                mode: 'lines',
                name: 'Lower Bollinger',
                line: { color: '#00ffaa', width: 1, dash: 'dot' },
                fill: 'tonexty',
                fillcolor: 'rgba(0, 255, 170, 0.1)',
              },
            ]}
            layout={{
              paper_bgcolor: 'rgba(0,0,0,0)',
              plot_bgcolor: 'rgba(0,0,0,0)',
              font: { color: '#00d1ff' },
              xaxis: { gridcolor: 'rgba(0, 209, 255, 0.1)' },
              yaxis: { gridcolor: 'rgba(0, 209, 255, 0.1)' },
              hovermode: 'x unified',
            }}
          />
          {patternData.summary && (
            <div className="mt-4 text-sm text-gray-400">
              <div className="text-cyan-400 font-semibold mb-2">Pattern Summary:</div>
              <div>{patternData.summary}</div>
            </div>
          )}
        </div>
      )}

      {/* Combined View */}
      {activeTab === 'combined' && (forecastData || patternData) && (
        <div className="space-y-4">
          {forecastData && (
            <div className="bg-[rgba(15,15,15,0.9)] backdrop-blur-md border border-cyan-500/20 rounded-xl p-4">
              <h3 className="text-lg font-bold text-cyan-400 mb-2">Forecast</h3>
              <PlotlyGraph
                data={[
                  {
                    x: forecastData.dates,
                    y: forecastData.historical,
                    type: 'scatter',
                    mode: 'lines',
                    name: 'Price',
                    line: { color: '#00d1ff' },
                  },
                ]}
                layout={{
                  height: 200,
                  paper_bgcolor: 'rgba(0,0,0,0)',
                  plot_bgcolor: 'rgba(0,0,0,0)',
                  font: { color: '#00d1ff', size: 10 },
                  margin: { t: 10, b: 20, l: 40, r: 10 },
                }}
              />
            </div>
          )}
          {patternData && (
            <div className="bg-[rgba(15,15,15,0.9)] backdrop-blur-md border border-cyan-500/20 rounded-xl p-4">
              <h3 className="text-lg font-bold text-cyan-400 mb-2">RSI & MACD</h3>
              <PlotlyGraph
                data={[
                  {
                    x: patternData.dates || [],
                    y: patternData.rsi,
                    type: 'scatter',
                    mode: 'lines',
                    name: 'RSI',
                    line: { color: '#ff6b6b' },
                    yaxis: 'y',
                  },
                  {
                    x: patternData.dates || [],
                    y: patternData.macd,
                    type: 'scatter',
                    mode: 'lines',
                    name: 'MACD',
                    line: { color: '#4ecdc4' },
                    yaxis: 'y2',
                  },
                ]}
                layout={{
                  height: 200,
                  paper_bgcolor: 'rgba(0,0,0,0)',
                  plot_bgcolor: 'rgba(0,0,0,0)',
                  font: { color: '#00d1ff', size: 10 },
                  margin: { t: 10, b: 20, l: 40, r: 40 },
                  yaxis: { domain: [0.55, 1], gridcolor: 'rgba(0, 209, 255, 0.1)' },
                  yaxis2: { domain: [0, 0.45], gridcolor: 'rgba(0, 209, 255, 0.1)' },
                }}
              />
            </div>
          )}
        </div>
      )}
    </motion.div>
  );
}

