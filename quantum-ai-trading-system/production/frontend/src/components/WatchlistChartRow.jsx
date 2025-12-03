/**
 * Watchlist Chart Row Component
 * Fetches and displays chart data for a watchlist symbol
 */

import React, { useEffect, useState } from "react";
import PlotlyGraph from "./PlotlyGraph";
import { getForecast, getAIRecommendation } from "../api/client";

export default function WatchlistChartRow({ 
  ticker, 
  forecast: initialForecast,
  aiRecommendation: initialAiRec
}) {
  const [chartData, setChartData] = useState(null);
  const [forecastData, setForecastData] = useState(initialForecast);
  const [aiInsight, setAiInsight] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchChartData = async () => {
      try {
        setLoading(true);
        
        // Fetch forecast and AI recommendation if not provided
        let forecast = initialForecast;
        let aiRec = initialAiRec;
        
        if (!forecast) {
          forecast = await getForecast(ticker).catch(() => null);
        }
        
        if (!aiRec) {
          aiRec = await getAIRecommendation(ticker).catch(() => null);
        }

        if (forecast) {
          setForecastData(forecast);
          
          // Extract chart data from forecast
          const metrics = forecast.metrics || forecast.result || {};
          const chartDataObj = {
            timestamps: metrics.timestamps || [],
            open: metrics.open || [],
            high: metrics.high || [],
            low: metrics.low || [],
            close: metrics.close || [],
            volume: metrics.volume || [],
            ema: metrics.ema || {},
            rsi: Array.isArray(metrics.rsi) ? metrics.rsi : null,
            macd: metrics.macd || null,
            forecast: forecast.forecast || null,
          };
          setChartData(chartDataObj);
        }

        if (aiRec) {
          setAiInsight({
            sentiment: aiRec.sentiment || aiRec.result?.sentiment || "neutral",
            emoji: aiRec.emoji || (aiRec.sentiment === "bullish" ? "🟢🚀" : aiRec.sentiment === "bearish" ? "🔴⚠️" : "⚫🌀"),
            confidence: aiRec.confidence || aiRec.result?.confidence || 0,
            recommendation: aiRec.recommendation || aiRec.result?.recommendation || "HOLD",
            rationale: aiRec.rationale || aiRec.result?.rationale || "",
            expectedGain: aiRec.expected_move_5d || aiRec.metrics?.expected_move_5d || 0,
            holdingDays: aiRec.holding_horizon_days || aiRec.metrics?.holding_horizon_days || 5,
          });
        }
      } catch (err) {
        console.error(`Error fetching chart data for ${ticker}:`, err);
      } finally {
        setLoading(false);
      }
    };

    fetchChartData();
  }, [ticker, initialForecast, initialAiRec]);

  return (
    <div className="space-y-4">
      <PlotlyGraph
        ticker={ticker}
        data={chartData}
        forecastData={forecastData}
        aiInsight={aiInsight}
        height={400}
        showLoading={loading}
      />
      
      {/* AI Recommendation Display */}
      {aiInsight && (
        <div className="bg-[rgba(0,0,0,0.5)] rounded-lg p-3 border border-cyan-500/20">
          <div className="text-xs text-gray-400 mb-2">AI Recommendation:</div>
          <div className="flex items-center gap-4 text-sm mb-2">
            <span className={`font-semibold ${
              aiInsight.recommendation?.includes("BUY") ? "text-neon-green" :
              aiInsight.recommendation?.includes("SELL") ? "text-magenta" :
              "text-gray-400"
            }`}>
              {aiInsight.recommendation || "HOLD"}
            </span>
            <span className="text-gray-400">
              Confidence: {Math.round((aiInsight.confidence || 0) * 100)}%
            </span>
            {aiInsight.emoji && (
              <span className="text-lg">{aiInsight.emoji}</span>
            )}
          </div>
          {aiInsight.expectedGain > 0 && (
            <div className="text-xs text-neon-green mb-1">
              Expected Gain: +{((aiInsight.expectedGain || 0) * 100).toFixed(1)}% over {aiInsight.holdingDays} days
            </div>
          )}
          {aiInsight.rationale && (
            <div className="text-xs text-gray-500 mt-2">{aiInsight.rationale}</div>
          )}
        </div>
      )}
    </div>
  );
}

