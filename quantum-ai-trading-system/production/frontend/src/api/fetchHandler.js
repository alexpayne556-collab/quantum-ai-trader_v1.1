/**
 * Quantum AI Cockpit — Centralized API Fetch Handler
 * ===================================================
 * Provides unified error handling, retry logic, and request cancellation
 */

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '';
const DEFAULT_TIMEOUT = 30000; // 30 seconds

/**
 * Centralized fetch handler with error handling and retry logic
 * @param {string} endpoint - API endpoint (e.g., '/api/forecast/AAPL')
 * @param {object} options - Fetch options
 * @returns {Promise<object>} - Parsed JSON response
 */
export async function fetchAPI(endpoint, options = {}) {
  const {
    method = 'GET',
    body = null,
    timeout = DEFAULT_TIMEOUT,
    retries = 2,
    signal = null,
    ...fetchOptions
  } = options;

  const url = endpoint.startsWith('http') ? endpoint : `${API_BASE_URL}${endpoint}`;
  
  // Create abort controller for timeout
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeout);
  
  // Combine signals
  const combinedSignal = signal 
    ? (signal.aborted ? signal : new AbortController().signal)
    : controller.signal;

  let lastError;
  
  for (let attempt = 0; attempt <= retries; attempt++) {
    try {
      const response = await fetch(url, {
        method,
        headers: {
          'Content-Type': 'application/json',
          ...fetchOptions.headers,
        },
        body: body ? JSON.stringify(body) : null,
        signal: combinedSignal,
        ...fetchOptions,
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`API Error ${response.status}: ${errorText || response.statusText}`);
      }

      const data = await response.json();
      return { data, error: null };

    } catch (error) {
      clearTimeout(timeoutId);
      lastError = error;

      // Don't retry on abort or client errors (4xx)
      if (error.name === 'AbortError' || (error.message?.includes('4'))) {
        throw error;
      }

      // Wait before retry (exponential backoff)
      if (attempt < retries) {
        await new Promise(resolve => setTimeout(resolve, Math.pow(2, attempt) * 1000));
      }
    }
  }

  throw lastError || new Error('API request failed after retries');
}

/**
 * Fetch with automatic error handling and default values
 * @param {string} endpoint - API endpoint
 * @param {object} options - Fetch options
 * @param {*} defaultValue - Default value to return on error
 * @returns {Promise<*>} - Response data or default value
 */
export async function fetchAPISafe(endpoint, options = {}, defaultValue = null) {
  try {
    const { data } = await fetchAPI(endpoint, options);
    return data;
  } catch (error) {
    console.error(`API fetch error for ${endpoint}:`, error);
    return defaultValue;
  }
}

/**
 * Batch fetch multiple endpoints concurrently
 * @param {Array<{endpoint: string, options?: object}>} requests - Array of fetch requests
 * @returns {Promise<Array>} - Array of results
 */
export async function fetchBatch(requests) {
  const promises = requests.map(({ endpoint, options = {} }) =>
    fetchAPISafe(endpoint, options, { error: true })
  );
  return Promise.all(promises);
}

/**
 * WebSocket connection helper
 * @param {string} endpoint - WebSocket endpoint (e.g., '/ws/alerts')
 * @param {function} onMessage - Message handler
 * @param {function} onError - Error handler
 * @returns {WebSocket} - WebSocket instance
 */
export function createWebSocket(endpoint, onMessage, onError) {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const host = window.location.host;
  const wsUrl = `${protocol}//${host}${endpoint}`;

  const ws = new WebSocket(wsUrl);

  ws.onopen = () => {
    console.log(`✅ WebSocket connected: ${endpoint}`);
  };

  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      onMessage(data);
    } catch (error) {
      console.error('WebSocket message parse error:', error);
      if (onError) onError(error);
    }
  };

  ws.onerror = (error) => {
    console.error(`WebSocket error on ${endpoint}:`, error);
    if (onError) onError(error);
  };

  ws.onclose = () => {
    console.log(`WebSocket closed: ${endpoint}`);
  };

  return ws;
}

export default {
  fetchAPI,
  fetchAPISafe,
  fetchBatch,
  createWebSocket,
};

