/**
 * Portfolio Chart Row Component
 * Fetches and displays chart data for a portfolio symbol
 */

import React, { useEffect, useState } from "react";
import PlotlyGraph from "./PlotlyGraph";
import { getForecast, getAIRecommendation } from "../api/client";

export default function PortfolioChartRow({ 
  ticker, 
  onDataLoaded,
  chartData: initialChartData,
  forecastData: initialForecastData,
  aiInsight: initialAiInsight 
}) {
  const [chartData, setChartData] = useState(initialChartData);
  const [forecastData, setForecastData] = useState(initialForecastData);
  const [aiInsight, setAiInsight] = useState(initialAiInsight);
  const [loading, setLoading] = useState(!initialChartData);

  useEffect(() => {
    if (initialChartData && initialForecastData) {
      // Data already loaded
      return;
    }

    const fetchChartData = async () => {
      try {
        setLoading(true);
        
        // Fetch forecast and AI recommendation in parallel
        const [forecast, aiRec] = await Promise.all([
          getForecast(ticker).catch(() => null),
          getAIRecommendation(ticker).catch(() => null),
        ]);

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
          });
        }

        // Notify parent of loaded data
        if (onDataLoaded) {
          const metrics = forecast?.metrics || forecast?.result || {};
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
            forecast: forecast?.forecast || null,
          };
          onDataLoaded(chartDataObj, forecast, {
            sentiment: aiRec?.sentiment || aiRec?.result?.sentiment || "neutral",
            emoji: aiRec?.emoji || (aiRec?.sentiment === "bullish" ? "🟢🚀" : aiRec?.sentiment === "bearish" ? "🔴⚠️" : "⚫🌀"),
            confidence: aiRec?.confidence || aiRec?.result?.confidence || 0,
            recommendation: aiRec?.recommendation || aiRec?.result?.recommendation || "HOLD",
            rationale: aiRec?.rationale || aiRec?.result?.rationale || "",
          });
        }
      } catch (err) {
        console.error(`Error fetching chart data for ${ticker}:`, err);
      } finally {
        setLoading(false);
      }
    };

    fetchChartData();
  }, [ticker, onDataLoaded, initialChartData, initialForecastData]);

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
        <div className="bg-[rgba(0,0,0,0.5)] rounded p-3 border border-cyan-500/20">
          <div className="text-xs text-gray-400 mb-2">AI Recommendation:</div>
          <div className="flex items-center gap-4 text-sm">
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
          {aiInsight.rationale && (
            <div className="text-xs text-gray-500 mt-2">{aiInsight.rationale}</div>
          )}
        </div>
      )}
    </div>
  );
}

