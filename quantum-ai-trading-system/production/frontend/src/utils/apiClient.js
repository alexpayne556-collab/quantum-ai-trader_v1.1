/**
 * Quantum AI Cockpit — Frontend API Client Plugin
 * ================================================
 * Unified client for connecting React Vite dashboard to backend
 * Handles all API calls, WebSocket connections, and data formatting
 */

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8090';
const WS_BASE_URL = import.meta.env.VITE_WS_URL || 'ws://127.0.0.1:8090';

/**
 * Main API client class for backend communication
 */
class QuantumAPIClient {
  constructor() {
    this.baseURL = API_BASE_URL;
    this.wsURL = WS_BASE_URL;
    this.wsConnections = new Map();
  }

  /**
   * Fetch forecast data for a symbol
   * @param {string} symbol - Ticker symbol
   * @returns {Promise<Object>} Forecast data with visual_context
   */
  async getForecast(symbol) {
    try {
      const response = await fetch(`${this.baseURL}/api/forecast/${symbol}`);
      if (!response.ok) {
        throw new Error(`Forecast API error: ${response.statusText}`);
      }
      const data = await response.json();
      
      // Ensure visual_context is properly formatted
      if (data.visual_context) {
        return {
          ...data,
          visual_context: {
            traces: Array.isArray(data.visual_context.traces) ? data.visual_context.traces : [],
            layout: data.visual_context.layout || {
              title: `${symbol} – 14D Fusior Forecast`,
              template: "plotly_dark"
            }
          }
        };
      }
      return data;
    } catch (error) {
      console.error('Forecast fetch error:', error);
      throw error;
    }
  }

  /**
   * Fetch dashboard data for a symbol
   * @param {string} symbol - Ticker symbol
   * @returns {Promise<Object>} Dashboard data with forecast.visual_context
   */
  async getDashboard(symbol) {
    try {
      const response = await fetch(`${this.baseURL}/api/dashboard?symbol=${symbol}`);
      if (!response.ok) {
        throw new Error(`Dashboard API error: ${response.statusText}`);
      }
      const data = await response.json();
      
      // Ensure forecast.visual_context is available
      if (data.forecast && !data.forecast.visual_context) {
        // Try to get visual_context from forecast endpoint
        try {
          const forecastData = await this.getForecast(symbol);
          if (forecastData.visual_context) {
            data.forecast.visual_context = forecastData.visual_context;
          }
        } catch (e) {
          console.warn('Could not fetch visual_context separately:', e);
        }
      }
      
      return data;
    } catch (error) {
      console.error('Dashboard fetch error:', error);
      throw error;
    }
  }

  /**
   * Get prediction for a symbol
   * @param {string} symbol - Ticker symbol
   * @returns {Promise<Object>} Prediction data
   */
  async getPrediction(symbol) {
    try {
      const response = await fetch(`${this.baseURL}/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ symbol }),
      });
      if (!response.ok) {
        throw new Error(`Prediction API error: ${response.statusText}`);
      }
      return await response.json();
    } catch (error) {
      console.error('Prediction fetch error:', error);
      throw error;
    }
  }

  /**
   * Get portfolio data
   * @returns {Promise<Object>} Portfolio data
   */
  async getPortfolio() {
    try {
      const response = await fetch(`${this.baseURL}/api/portfolio`);
      if (!response.ok) {
        throw new Error(`Portfolio API error: ${response.statusText}`);
      }
      return await response.json();
    } catch (error) {
      console.error('Portfolio fetch error:', error);
      throw error;
    }
  }

  /**
   * Get watchlist data
   * @returns {Promise<Object>} Watchlist data
   */
  async getWatchlist() {
    try {
      const response = await fetch(`${this.baseURL}/api/watchlist`);
      if (!response.ok) {
        throw new Error(`Watchlist API error: ${response.statusText}`);
    }
      return await response.json();
    } catch (error) {
      console.error('Watchlist fetch error:', error);
      throw error;
    }
  }

  /**
   * Connect to WebSocket for real-time updates
   * @param {string} endpoint - WebSocket endpoint (e.g., '/ws/trading-signals')
   * @param {Function} onMessage - Callback for messages
   * @param {Function} onError - Callback for errors
   * @returns {WebSocket} WebSocket connection
   */
  connectWebSocket(endpoint, onMessage, onError) {
    const ws = new WebSocket(`${this.wsURL}${endpoint}`);
    
    ws.onopen = () => {
      console.log(`WebSocket connected: ${endpoint}`);
    };
    
    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (onMessage) onMessage(data);
      } catch (error) {
        console.error('WebSocket message parse error:', error);
      }
    };
    
    ws.onerror = (error) => {
      console.error(`WebSocket error on ${endpoint}:`, error);
      if (onError) onError(error);
    };
    
    ws.onclose = () => {
      console.log(`WebSocket closed: ${endpoint}`);
      // Auto-reconnect after 3 seconds
      setTimeout(() => {
        if (!this.wsConnections.has(endpoint)) {
          this.connectWebSocket(endpoint, onMessage, onError);
        }
      }, 3000);
    };
    
    this.wsConnections.set(endpoint, ws);
    return ws;
  }

  /**
   * Disconnect WebSocket
   * @param {string} endpoint - WebSocket endpoint
   */
  disconnectWebSocket(endpoint) {
    const ws = this.wsConnections.get(endpoint);
    if (ws) {
      ws.close();
      this.wsConnections.delete(endpoint);
    }
  }

  /**
   * Format forecast data for PlotlyGraph component
   * @param {Object} forecastData - Raw forecast data from API
   * @returns {Object} Formatted data for PlotlyGraph
   */
  formatForPlotly(forecastData) {
    if (!forecastData) {
      return {
        traces: [],
        layout: {
          title: 'No data',
          template: 'plotly_dark'
        }
      };
    }

    // Extract visual_context if available
    const vc = forecastData.visual_context || forecastData.forecast?.visual_context || {};
    
    return {
      traces: Array.isArray(vc.traces) ? vc.traces : [],
      layout: vc.layout || {
        title: `${forecastData.symbol || 'Symbol'} – Forecast`,
        template: 'plotly_dark',
        xaxis: { title: 'Date' },
        yaxis: { title: 'Price' }
      }
    };
  }

  /**
   * Format dashboard data for components
   * @param {Object} dashboardData - Raw dashboard data from API
   * @returns {Object} Formatted data with all required fields
   */
  formatDashboardData(dashboardData) {
    if (!dashboardData) {
      return {
        forecast: null,
        plotlyData: { traces: [], layout: {} },
        opportunities: [],
        aiRecommendation: null,
        marketOpen: false
      };
    }

    const forecast = dashboardData.forecast || {};
    const visualContext = forecast.visual_context || {};
    
    return {
      forecast: forecast,
      plotlyData: {
        traces: Array.isArray(visualContext.traces) ? visualContext.traces : [],
        layout: visualContext.layout || {
          title: `${dashboardData.symbol || 'Symbol'} – Forecast`,
          template: 'plotly_dark'
        }
      },
      opportunities: forecast.opportunities || [],
      aiRecommendation: forecast.ai_recommendation || dashboardData.ai_recommendation || null,
      marketOpen: forecast.market_open || false,
      trend: forecast.trend || 'neutral',
      confidence: forecast.confidence || 0.0,
      signals: forecast.signals || {},
      metrics: forecast.metrics || {}
    };
  }
}

// Export singleton instance
export const apiClient = new QuantumAPIClient();

// Export class for custom instances
export default QuantumAPIClient;

