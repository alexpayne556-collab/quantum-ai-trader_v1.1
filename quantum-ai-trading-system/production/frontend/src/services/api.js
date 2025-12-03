// Quantum AI Trading System - API Service
// Handles all backend communication

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8090';

class ApiService {
  constructor() {
    this.baseUrl = API_BASE_URL;
  }

  async request(endpoint, options = {}) {
    const url = `${this.baseUrl}${endpoint}`;
    const config = {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
      ...options,
    };

    try {
      const response = await fetch(url, config);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('API request failed:', error);
      throw error;
    }
  }

  // Elite Trading Engine APIs
  async getQuantScore(symbol) {
    return this.request(`/api/elite/quant-score/${symbol}`);
  }

  async getBayesianSignal(symbol) {
    return this.request(`/api/elite/bayesian-signal/${symbol}`);
  }

  async getRegimeDetection(symbol) {
    return this.request(`/api/elite/regime/${symbol}`);
  }

  async getKellySize(symbol) {
    return this.request(`/api/elite/kelly-size/${symbol}`);
  }

  // Core Trading APIs
  async getAIRecommendation(symbol) {
    return this.request(`/api/ai_recommendation/${symbol}`);
  }

  async getForecast(symbol) {
    return this.request(`/api/forecast/${symbol}`);
  }

  async getRiskAnalysis(symbol) {
    return this.request(`/api/risk/${symbol}`);
  }

  // Market Data APIs
  async getScreener(source = 'robinhood') {
    return this.request(`/api/screener?source=${source}`);
  }

  async getTopGainers(limit = 20) {
    return this.request(`/api/top_gainers?limit=${limit}`);
  }

  async getMarketOverview() {
    return this.request(`/api/market_overview`);
  }

  async getMarketTicker(limit = 50) {
    return this.request(`/api/market_ticker?limit=${limit}`);
  }

  // Advanced Scanner APIs
  async getPreGainerScanner() {
    return this.request(`/api/scanners/pre-gainer`);
  }

  async getDayTradingScanner() {
    return this.request(`/api/scanners/day-trading`);
  }

  async getOpportunityScanner() {
    return this.request(`/api/scanners/opportunity`);
  }

  async getDarkPoolScanner() {
    return this.request(`/api/scanners/dark-pool`);
  }

  async getInsiderTradingScanner() {
    return this.request(`/api/scanners/insider`);
  }

  async getShortSqueezeScanner() {
    return this.request(`/api/scanners/short-squeeze`);
  }

  // Backtesting APIs
  async runBacktest(config) {
    return this.request('/api/backtest', {
      method: 'POST',
      body: JSON.stringify(config),
    });
  }

  async getBacktestResults(id) {
    return this.request(`/api/backtest/results/${id}`);
  }

  // WebSocket connection for real-time data
  createWebSocketConnection(onMessage, onError) {
    const wsUrl = process.env.REACT_APP_WS_URL || 'ws://localhost:8090/ws';
    
    const ws = new WebSocket(wsUrl);
    
    ws.onopen = () => {
      console.log('WebSocket connection established');
    };
    
    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        onMessage(data);
      } catch (error) {
        console.error('Failed to parse WebSocket message:', error);
      }
    };
    
    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      if (onError) onError(error);
    };
    
    ws.onclose = () => {
      console.log('WebSocket connection closed');
    };
    
    return ws;
  }

  // Utility methods
  async healthCheck() {
    try {
      const response = await this.request('/');
      return response.status === '✅ Operational';
    } catch (error) {
      return false;
    }
  }

  async getSystemStatus() {
    return this.request('/api/system/status');
  }
}

// Create singleton instance
const apiService = new ApiService();

// Export individual functions for easier usage
export const fetchQuantScore = (symbol) => apiService.getQuantScore(symbol);
export const fetchBayesianSignal = (symbol) => apiService.getBayesianSignal(symbol);
export const fetchRegimeDetection = (symbol) => apiService.getRegimeDetection(symbol);
export const fetchKellySize = (symbol) => apiService.getKellySize(symbol);
export const fetchAIRecommendation = (symbol) => apiService.getAIRecommendation(symbol);
export const fetchForecast = (symbol) => apiService.getForecast(symbol);
export const fetchRiskAnalysis = (symbol) => apiService.getRiskAnalysis(symbol);
export const fetchScreener = (source) => apiService.getScreener(source);
export const fetchTopGainers = (limit) => apiService.getTopGainers(limit);
export const fetchMarketOverview = () => apiService.getMarketOverview();
export const fetchMarketTicker = (limit) => apiService.getMarketTicker(limit);

// Scanner functions
export const fetchPreGainerScanner = () => apiService.getPreGainerScanner();
export const fetchDayTradingScanner = () => apiService.getDayTradingScanner();
export const fetchOpportunityScanner = () => apiService.getOpportunityScanner();
export const fetchDarkPoolScanner = () => apiService.getDarkPoolScanner();
export const fetchInsiderTradingScanner = () => apiService.getInsiderTradingScanner();
export const fetchShortSqueezeScanner = () => apiService.getShortSqueezeScanner();

// Backtesting functions
export const runBacktest = (config) => apiService.runBacktest(config);
export const getBacktestResults = (id) => apiService.getBacktestResults(id);

// WebSocket and utility functions
export const createWebSocketConnection = (onMessage, onError) => 
  apiService.createWebSocketConnection(onMessage, onError);
export const healthCheck = () => apiService.healthCheck();
export const getSystemStatus = () => apiService.getSystemStatus();

export default apiService;
