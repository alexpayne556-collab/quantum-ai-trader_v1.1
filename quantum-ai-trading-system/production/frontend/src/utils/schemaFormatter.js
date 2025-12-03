/**
 * Quantum AI Cockpit — Data Schema Formatter
 * ===========================================
 * Normalizes and formats API responses for consistent frontend consumption
 */

/**
 * Format forecast data for Plotly visualization
 * @param {object} forecastData - Raw forecast API response
 * @returns {object} - Formatted data for Plotly
 */
export function formatForecastData(forecastData) {
  if (!forecastData || !forecastData.result) {
    return {
      dates: [],
      historical: [],
      forecast: [],
      confidence: [],
      error: 'No forecast data available'
    };
  }

  const result = forecastData.result;
  
  return {
    dates: result.dates || result.Dates || [],
    historical: result.historical || result.Close_series || [],
    forecast: result.forecast || result.Forecast || [],
    confidence: result.confidence || result.Confidence || [],
    upperBand: result.upper_band || result.UpperBand || [],
    lowerBand: result.lower_band || result.LowerBand || [],
    symbol: result.symbol || forecastData.symbol || 'UNKNOWN',
    fii: result.fii || result.FII || 0,
    signal: result.signal || result.Signal || 'NEUTRAL',
  };
}

/**
 * Format pattern detection data
 * @param {object} patternData - Raw pattern API response
 * @returns {object} - Formatted pattern data
 */
export function formatPatternData(patternData) {
  if (!patternData || !patternData.payload) {
    return {
      rsi: [],
      macd: [],
      close: [],
      bollinger: { upper: [], lower: [], middle: [] },
      alerts: [],
      summary: 'No pattern data available'
    };
  }

  const payload = patternData.payload;
  
  return {
    rsi: payload.RSI_series || payload.rsi || [],
    macd: payload.MACD_series || payload.macd || [],
    close: payload.Close_series || payload.close || [],
    bollinger: {
      upper: payload.UpperBand || payload.upper_band || [],
      lower: payload.LowerBand || payload.lower_band || [],
      middle: payload.MiddleBand || payload.middle_band || [],
    },
    dates: payload.Dates || payload.dates || [],
    alerts: payload.Alerts || payload.alerts || [],
    summary: payload.Summary || payload.summary || 'No summary available',
    symbol: patternData.symbol || 'UNKNOWN',
  };
}

/**
 * Format sentiment data
 * @param {object} sentimentData - Raw sentiment API response
 * @returns {object} - Formatted sentiment data
 */
export function formatSentimentData(sentimentData) {
  if (!sentimentData) {
    return {
      score: 0,
      label: 'NEUTRAL',
      confidence: 0,
      breakdown: {}
    };
  }

  return {
    score: sentimentData.sentiment_score || sentimentData.score || 0,
    label: sentimentData.sentiment_label || sentimentData.label || 'NEUTRAL',
    confidence: sentimentData.confidence || 0,
    breakdown: sentimentData.breakdown || sentimentData.model_scores || {},
    symbol: sentimentData.symbol || 'UNKNOWN',
  };
}

/**
 * Format risk analysis data
 * @param {object} riskData - Raw risk API response
 * @returns {object} - Formatted risk data
 */
export function formatRiskData(riskData) {
  if (!riskData || !riskData.result) {
    return {
      riskScore: 0,
      var: 0,
      sharpe: 0,
      volatility: 0,
      beta: 0,
    };
  }

  const result = riskData.result;
  
  return {
    riskScore: result.risk_score || result.riskScore || 0,
    var: result.var || result.VaR || 0,
    sharpe: result.sharpe || result.Sharpe || 0,
    volatility: result.volatility || result.Volatility || 0,
    beta: result.beta || result.Beta || 0,
    symbol: riskData.symbol || 'UNKNOWN',
  };
}

/**
 * Format AI recommendation data
 * @param {object} recommendationData - Raw recommendation API response
 * @returns {object} - Formatted recommendation
 */
export function formatRecommendationData(recommendationData) {
  if (!recommendationData) {
    return {
      action: 'HOLD',
      confidence: 0,
      expectedGain: 0,
      rationale: 'No recommendation available',
      emoji: '🤔',
    };
  }

  return {
    action: recommendationData.action || recommendationData.recommendation || 'HOLD',
    confidence: recommendationData.confidence || 0,
    expectedGain: recommendationData.expected_gain || recommendationData.expectedGain || 0,
    rationale: recommendationData.rationale || recommendationData.summary || 'No rationale available',
    emoji: recommendationData.emoji || getEmojiForAction(recommendationData.action),
    symbol: recommendationData.symbol || 'UNKNOWN',
    horizon: recommendationData.horizon || '14d',
  };
}

/**
 * Get emoji for action type
 * @param {string} action - Action type (BUY, SELL, HOLD)
 * @returns {string} - Emoji
 */
function getEmojiForAction(action) {
  const actionUpper = (action || '').toUpperCase();
  if (actionUpper.includes('BUY')) return '💰';
  if (actionUpper.includes('SELL')) return '📉';
  if (actionUpper.includes('HOLD')) return '🤔';
  return '📊';
}

/**
 * Format portfolio data
 * @param {object} portfolioData - Raw portfolio API response
 * @returns {Array} - Formatted portfolio holdings
 */
export function formatPortfolioData(portfolioData) {
  if (!portfolioData || !portfolioData.holdings) {
    return [];
  }

  return portfolioData.holdings.map(holding => ({
    symbol: holding.symbol || 'UNKNOWN',
    shares: holding.shares || holding.qty || 0,
    costBasis: holding.cost_basis || holding.cost || 0,
    currentPrice: holding.current_price || holding.price || 0,
    equity: holding.equity || (holding.shares * holding.current_price) || 0,
    gainLoss: holding.gain_loss || 0,
    gainLossPct: holding.gain_loss_pct || 0,
    sector: holding.sector || 'Unknown',
  }));
}

/**
 * Format watchlist data
 * @param {object} watchlistData - Raw watchlist API response
 * @returns {Array} - Formatted watchlist tickers
 */
export function formatWatchlistData(watchlistData) {
  if (!watchlistData || !watchlistData.tickers) {
    return [];
  }

  return watchlistData.tickers.map(ticker => ({
    symbol: ticker.symbol || 'UNKNOWN',
    currentPrice: ticker.current_price || ticker.price || 0,
    priceChange: ticker.price_change || 0,
    priceChangePct: ticker.price_change_pct || 0,
    notes: ticker.notes || '',
    addedDate: ticker.added_date || ticker.addedDate || null,
  }));
}

/**
 * Normalize API response to standard format
 * @param {object} response - Raw API response
 * @param {string} type - Data type (forecast, pattern, sentiment, etc.)
 * @returns {object} - Normalized response
 */
export function normalizeAPIResponse(response, type) {
  if (!response) return null;

  const formatters = {
    forecast: formatForecastData,
    pattern: formatPatternData,
    sentiment: formatSentimentData,
    risk: formatRiskData,
    recommendation: formatRecommendationData,
    portfolio: formatPortfolioData,
    watchlist: formatWatchlistData,
  };

  const formatter = formatters[type];
  if (!formatter) {
    console.warn(`No formatter for type: ${type}`);
    return response;
  }

  return formatter(response);
}

export default {
  formatForecastData,
  formatPatternData,
  formatSentimentData,
  formatRiskData,
  formatRecommendationData,
  formatPortfolioData,
  formatWatchlistData,
  normalizeAPIResponse,
};

