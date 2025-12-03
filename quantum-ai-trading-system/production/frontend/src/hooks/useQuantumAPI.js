/**
 * Quantum AI Cockpit — React Hook for Backend API Integration
 * ============================================================
 * Easy-to-use hook for fetching forecast, dashboard, and other data
 * Automatically formats data for PlotlyGraph and other components
 */

import { useState, useEffect, useCallback } from 'react';
import { apiClient } from '../utils/apiClient';

/**
 * Hook for fetching forecast data
 * @param {string} symbol - Ticker symbol
 * @param {boolean} autoFetch - Whether to fetch automatically on mount/symbol change
 * @returns {Object} { data, loading, error, refetch }
 */
export function useForecast(symbol, autoFetch = true) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const fetchForecast = useCallback(async () => {
    if (!symbol) return;
    
    setLoading(true);
    setError(null);
    try {
      const forecastData = await apiClient.getForecast(symbol);
      setData(forecastData);
    } catch (err) {
      setError(err.message);
      console.error('Forecast fetch error:', err);
    } finally {
      setLoading(false);
    }
  }, [symbol]);

  useEffect(() => {
    if (autoFetch) {
      fetchForecast();
    }
  }, [symbol, autoFetch, fetchForecast]);

  return {
    data,
    loading,
    error,
    refetch: fetchForecast,
    plotlyData: data ? apiClient.formatForPlotly(data) : { traces: [], layout: {} }
  };
}

/**
 * Hook for fetching dashboard data
 * @param {string} symbol - Ticker symbol
 * @param {boolean} autoFetch - Whether to fetch automatically
 * @returns {Object} { data, loading, error, refetch, plotlyData, opportunities, aiRecommendation }
 */
export function useDashboard(symbol, autoFetch = true) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const fetchDashboard = useCallback(async () => {
    if (!symbol) return;
    
    setLoading(true);
    setError(null);
    try {
      const dashboardData = await apiClient.getDashboard(symbol);
      setData(dashboardData);
    } catch (err) {
      setError(err.message);
      console.error('Dashboard fetch error:', err);
    } finally {
      setLoading(false);
    }
  }, [symbol]);

  useEffect(() => {
    if (autoFetch) {
      fetchDashboard();
    }
  }, [symbol, autoFetch, fetchDashboard]);

  const formatted = data ? apiClient.formatDashboardData(data) : null;

  return {
    data,
    loading,
    error,
    refetch: fetchDashboard,
    plotlyData: formatted?.plotlyData || { traces: [], layout: {} },
    opportunities: formatted?.opportunities || [],
    aiRecommendation: formatted?.aiRecommendation,
    marketOpen: formatted?.marketOpen || false,
    trend: formatted?.trend || 'neutral',
    confidence: formatted?.confidence || 0.0,
    signals: formatted?.signals || {},
    metrics: formatted?.metrics || {}
  };
}

/**
 * Hook for WebSocket real-time updates
 * @param {string} endpoint - WebSocket endpoint (e.g., '/ws/trading-signals')
 * @returns {Object} { data, isConnected, error }
 */
export function useQuantumWebSocket(endpoint) {
  const [data, setData] = useState(null);
  const [isConnected, setIsConnected] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    const ws = apiClient.connectWebSocket(
      endpoint,
      (message) => {
        setData(message);
        setIsConnected(true);
      },
      (err) => {
        setError(err);
        setIsConnected(false);
      }
    );

    return () => {
      apiClient.disconnectWebSocket(endpoint);
    };
  }, [endpoint]);

  return { data, isConnected, error };
}

/**
 * Hook for portfolio data
 * @returns {Object} { data, loading, error, refetch }
 */
export function usePortfolio() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const fetchPortfolio = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const portfolioData = await apiClient.getPortfolio();
      setData(portfolioData);
    } catch (err) {
      setError(err.message);
      console.error('Portfolio fetch error:', err);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchPortfolio();
  }, [fetchPortfolio]);

  return { data, loading, error, refetch: fetchPortfolio };
}

/**
 * Hook for watchlist data
 * @returns {Object} { data, loading, error, refetch }
 */
export function useWatchlist() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const fetchWatchlist = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const watchlistData = await apiClient.getWatchlist();
      setData(watchlistData);
    } catch (err) {
      setError(err.message);
      console.error('Watchlist fetch error:', err);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchWatchlist();
  }, [fetchWatchlist]);

  return { data, loading, error, refetch: fetchWatchlist };
}

