/**
 * Quantum AI Cockpit — API Configuration
 * =======================================
 * Centralized API endpoint and configuration management
 */

// API Base URL (uses Vite proxy in development)
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '';
const WS_BASE_URL = import.meta.env.VITE_WS_BASE_URL || '';

// Backend API endpoints
// NOTE: Updated to use current active endpoints (fusior_forecast replaces fusion_forecast)
export const API_ENDPOINTS = {
  // Forecast endpoints (current)
  FORECAST: (symbol) => `/api/forecast/${symbol}`, // Uses fusior_forecast
  
  // Pattern endpoints - deprecated, use deep_analysis instead
  // PATTERN_DETECTION and PATTERNS removed - use DEEP_ANALYSIS
  
  // Sentiment endpoints - deprecated, included in AI recommendation
  // SENTIMENT and SENTIMENT_DATA removed - use AI_RECOMMENDATION
  
  // Risk endpoints
  RISK: (symbol) => `/api/risk/${symbol}`,
  
  // Deep Red endpoints (integrated)
  DEEP_RED: (symbol) => `/api/deep_red/${symbol}`,
  DEEPRED: (symbol) => `/api/deepred/${symbol}`, // Legacy alias
  
  // AI Recommendation endpoints (integrated)
  AI_RECOMMENDER: (symbol) => `/api/ai_recommender/${symbol}`, // Integrated endpoint
  AI_RECOMMENDATION: (symbol) => `/api/ai_recommendation/${symbol}`, // Legacy alias
  RECOMMENDATION: (symbol) => `/api/recommendation/${symbol}`, // Legacy alias
  
  // Portfolio endpoints
  PORTFOLIO: `/api/portfolio`,
  PORTFOLIO_DATA: `/api/portfolio/data`,
  PORTFOLIO_ADD: `/api/portfolio/add`,
  PORTFOLIO_REMOVE: `/api/portfolio/remove`,
  
  // Watchlist endpoints
  WATCHLIST: `/api/watchlist`,
  WATCHLIST_ADD: `/api/watchlist/add`,
  WATCHLIST_REMOVE: `/api/watchlist/remove`,
  
  // Market endpoints
  MARKET_OVERVIEW: `/api/market_overview`,
  SCREENER: `/api/screener`,
  
  // System endpoints
  SYSTEM_MODULES: `/api/system/modules`,
  SYSTEM_MODULE: (moduleName) => `/api/system/modules/${moduleName}`,
  SYSTEM_BINDINGS: `/api/system/bindings`,
  SYSTEM_HEALTH: `/api/system/health`,
  SYSTEM_TRUST: `/api/system/trust`,
  
  // Deep Analysis endpoint
  DEEP_ANALYSIS: (symbol) => `/api/deep_analysis/${symbol}`,
  
  // Trading Signals endpoint
  TRADING_SIGNALS: (symbol) => `/api/trading_signals/${symbol}`,
  
  // Stock Scraper endpoints
  STOCK_SCRAPER_ROBINHOOD: `/api/stock_scraper/robinhood_top_500`,
  STOCK_SCRAPER_RECOMMENDATIONS: `/api/stock_scraper/recommendations`,
  
  // AI Trading Bot endpoints
  TRADING_BOTS: `/api/trading_bots`,
  TRADING_BOTS_CREATE: `/api/trading_bots/create`,
  TRADING_BOTS_TRADE: (botId) => `/api/trading_bots/${botId}/trade`,
  TRADING_BOTS_ANALYZE: (botId) => `/api/trading_bots/${botId}/analyze`,
  
  // News Scraper endpoints
  NEWS_SYMBOL: (symbol) => `/api/news/${symbol}`,
  NEWS_MARKET: `/api/news/market/general`,
  NEWS_ALERTS: `/api/news/alerts`,
  NEWS_SENTIMENT: (symbol) => `/api/news/sentiment/${symbol}`,
  
  // Market Ticker endpoints
  MARKET_TICKER: `/api/market_ticker`,
  TOP_GAINERS: `/api/market_overview`,
  MUST_BUY: `/api/must_buy`,
};

// WebSocket endpoints
export const WS_ENDPOINTS = {
  ALERTS: `/ws/alerts`,
  PORTFOLIO: `/ws/portfolio`,
  WATCHLIST: `/ws/watchlist`,
  STREAM: `/ws/stream`, // General broadcast stream
  MODULE_STREAM: (module, symbol) => `/ws/${module}/${symbol}`,
};

// API Configuration
export const API_CONFIG = {
  BASE_URL: API_BASE_URL,
  WS_BASE_URL: WS_BASE_URL,
  TIMEOUT: 30000, // 30 seconds
  RETRY_ATTEMPTS: 2,
  RETRY_DELAY: 1000, // 1 second
};

// Request headers
export const API_HEADERS = {
  'Content-Type': 'application/json',
  'Accept': 'application/json',
};

// Default symbols for testing
export const DEFAULT_SYMBOLS = ['AAPL', 'MSFT', 'NVDA', 'TSLA', 'SPY'];

// Polling intervals (in milliseconds)
export const POLLING_INTERVALS = {
  MARKET_DATA: 15000,      // 15 seconds
  PORTFOLIO: 30000,        // 30 seconds
  WATCHLIST: 30000,        // 30 seconds
  SYSTEM_HEALTH: 60000,    // 1 minute
};

export default {
  API_ENDPOINTS,
  WS_ENDPOINTS,
  API_CONFIG,
  API_HEADERS,
  DEFAULT_SYMBOLS,
  POLLING_INTERVALS,
};

