// frontend/src/components/WatchlistTable.jsx
import React, { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import WatchlistChartRow from "./WatchlistChartRow";
import { getWatchlist, getForecast, getAIRecommendation } from "../api/client";

export default function WatchlistTable() {
  const [watchlist, setWatchlist] = useState(["AAPL", "TSLA", "BTC-USD"]);
  const [expanded, setExpanded] = useState(null);
  const [dataMap, setDataMap] = useState({});
  const [aiRecommendations, setAiRecommendations] = useState({});

  // Fetch watchlist data and forecast for each ticker
  useEffect(() => {
    const fetchData = async () => {
      try {
        // Fetch watchlist data
        const watchlistData = await getWatchlist();
        
        // Extract tickers from watchlist response
        const tickers = watchlistData?.tickers?.map(t => t.symbol || t) || 
                       watchlistData?.symbols || 
                       watchlistData?.watchlist ||
                       watchlist;
        
        // Update watchlist state if we got tickers from API
        if (Array.isArray(tickers) && tickers.length > 0) {
          setWatchlist(tickers);
        }

        // Fetch forecast and AI recommendation for each ticker
        const forecastPromises = tickers.map(async (ticker) => {
          try {
            const [forecast, aiRec] = await Promise.all([
              getForecast(ticker).catch(() => null),
              getAIRecommendation(ticker).catch(() => null),
            ]);

            return {
              ticker,
              forecast,
              aiRecommendation: aiRec,
            };
          } catch (err) {
            console.error(`Error fetching data for ${ticker}:`, err);
            return { ticker, forecast: null, aiRecommendation: null };
          }
        });

        const results = await Promise.all(forecastPromises);
        
        const forecastMap = {};
        const aiRecMap = {};
        
        results.forEach(({ ticker, forecast, aiRecommendation }) => {
          if (forecast) {
            forecastMap[ticker] = {
              ...forecast,
              forecast_direction: forecast?.trend || forecast?.result?.trend || "—",
              volatility: forecast?.metrics?.volatility_20d || forecast?.metrics?.atr_pct || null
            };
          }
          if (aiRecommendation) {
            aiRecMap[ticker] = aiRecommendation;
          }
        });

        setDataMap(forecastMap);
        setAiRecommendations(aiRecMap);
      } catch (err) {
        console.error("Watchlist fetch error:", err);
      }
    };

    fetchData();
    const interval = setInterval(fetchData, 30000); // Update every 30 seconds
    return () => clearInterval(interval);
  }, []);

  const handleAddTicker = () => {
    const newTicker = prompt("Enter ticker symbol:")?.toUpperCase();
    if (newTicker && !watchlist.includes(newTicker)) {
      setWatchlist([...watchlist, newTicker]);
    }
  };

  const handleRemoveTicker = (ticker) => {
    setWatchlist(watchlist.filter((t) => t !== ticker));
  };

  const toggleExpand = (ticker) => {
    setExpanded(expanded === ticker ? null : ticker);
  };

  return (
    <div className="bg-[rgba(15,15,15,0.9)] backdrop-blur-md border border-cyan-500/20 rounded-xl shadow-[0_0_25px_rgba(0,255,170,0.1)] p-4">
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-electric-blue text-lg font-semibold tracking-wider">
          Watchlist
        </h2>
        <button
          onClick={handleAddTicker}
          className="text-neon-green border border-neon-green/40 px-3 py-1 rounded hover:bg-neon-green/10 transition-all"
        >
          + Add
        </button>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-sm text-left text-gray-300">
          <thead className="text-gray-400 uppercase border-b border-cyan-500/10">
            <tr>
              <th className="py-2 px-3">Ticker</th>
              <th className="py-2 px-3">Forecast</th>
              <th className="py-2 px-3">Sentiment</th>
              <th className="py-2 px-3">Volatility</th>
              <th className="py-2 px-3"></th>
            </tr>
          </thead>
          <tbody>
            {watchlist.map((ticker) => {
              const forecast = dataMap[ticker] || {};
              const volatility = forecast.volatility || "—";
              const aiRec = aiRecommendations[ticker] || {};
              
              // Determine if good buy (green background)
              const isGoodBuy = (aiRec?.recommendation?.includes("BUY") || aiRec?.action?.includes("BUY")) && 
                               (aiRec?.confidence || aiRec?.metrics?.confidence || 0) > 0.6 &&
                               ((aiRec?.expected_move_5d || aiRec?.metrics?.expected_move_5d || 0) > 0);

              return (
                <React.Fragment key={ticker}>
                  <tr
                    className={`border-b border-cyan-500/10 hover:bg-[rgba(0,255,170,0.03)] transition-all cursor-pointer ${
                      isGoodBuy ? "bg-[rgba(0,255,0,0.1)]" : ""
                    }`}
                    onClick={() => toggleExpand(ticker)}
                  >
                    <td className="py-2 px-3 font-semibold text-neon-green">
                      {ticker}
                    </td>
                    <td className="py-2 px-3">
                      {forecast.forecast_direction || "—"}
                    </td>
                    <td className="py-2 px-3">
                      {aiRec?.sentiment || aiRec?.result?.sentiment || "—"}
                    </td>
                    <td className="py-2 px-3">{volatility}</td>
                    <td className="py-2 px-3 text-right">
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          handleRemoveTicker(ticker);
                        }}
                        className="text-magenta hover:text-red-400"
                      >
                        ✖
                      </button>
                    </td>
                  </tr>

                  <AnimatePresence>
                    {expanded === ticker && (
                      <motion.tr
                        initial={{ opacity: 0, height: 0 }}
                        animate={{ opacity: 1, height: "auto" }}
                        exit={{ opacity: 0, height: 0 }}
                        transition={{ duration: 0.3 }}
                      >
                        <td colSpan={5} className="py-3 px-2">
                          <WatchlistChartRow 
                            ticker={ticker}
                            forecast={forecast}
                            aiRecommendation={aiRec}
                          />
                        </td>
                      </motion.tr>
                    )}
                  </AnimatePresence>
                </React.Fragment>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
