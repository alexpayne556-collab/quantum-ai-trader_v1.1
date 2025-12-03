/**
 * Quantum AI Cockpit — API Client
 * Centralized API client for backend communication
 */

const API_BASE_URL = "http://127.0.0.1:8090";

/**
 * Generic fetch wrapper with error handling
 */
async function apiFetch(endpoint, options = {}) {
  try {
    const url = endpoint.startsWith("http") ? endpoint : `${API_BASE_URL}${endpoint}`;
    const response = await fetch(url, {
      ...options,
      headers: {
        "Content-Type": "application/json",
        ...options.headers,
      },
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`API Error (${response.status}): ${errorText || response.statusText}`);
    }

    return await response.json();
  } catch (error) {
    console.error(`API fetch error for ${endpoint}:`, error);
    throw error;
  }
}

/**
 * Get forecast data for a symbol
 */
export async function getForecast(symbol) {
  return apiFetch(`/api/forecast/${encodeURIComponent(symbol)}`);
}

/**
 * Get portfolio data
 */
export async function getPortfolio() {
  return apiFetch("/api/portfolio");
}

/**
 * Get watchlist data
 */
export async function getWatchlist() {
  return apiFetch("/api/watchlist");
}

/**
 * Get AI recommendation for a symbol
 */
export async function getAIRecommendation(symbol) {
  return apiFetch(`/api/ai_recommendation/${encodeURIComponent(symbol)}`);
}

/**
 * Get deep analysis for a symbol
 */
export async function getDeepAnalysis(symbol) {
  return apiFetch(`/api/deep_analysis/${encodeURIComponent(symbol)}`);
}

/**
 * Get system diagnostics
 */
export async function getSystemDiagnostics() {
  return apiFetch("/api/system/diagnostics");
}

export default {
  getForecast,
  getPortfolio,
  getWatchlist,
  getAIRecommendation,
  getDeepAnalysis,
  getSystemDiagnostics,
};

