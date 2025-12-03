/**
 * Quantum AI Cockpit — WebSocket Feed Hook
 * 🔌 Real-time data streaming hook for WebSocket connections
 * ==========================================================
 */

import { useState, useEffect, useRef } from "react";

/**
 * Custom hook for WebSocket data feed with multiplex channel support
 * @param {string} endpoint - WebSocket endpoint (e.g., "/ws/fusion_forecast")
 * @param {object} options - Configuration options
 * @param {string} options.channel - Optional channel name for multiplexing (e.g., "forecast", "trust", "sentiment")
 * @returns {object} - { data, error, isConnected, send }
 */
export function useWebSocketFeed(endpoint, options = {}) {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [isConnected, setIsConnected] = useState(false);
  const wsRef = useRef(null);
  const reconnectTimeoutRef = useRef(null);
  const {
    reconnectInterval = 5000,
    maxReconnectAttempts = 10,
    autoConnect = true,
    channel = null, // Multiplex channel name
  } = options;

  const reconnectAttempts = useRef(0);
  const backoffRef = useRef(1000);

  const connect = () => {
    try {
      // Determine WebSocket URL - use Vite proxy
      const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
      const host = window.location.host;
      const wsUrl = `${protocol}//${host}${endpoint}`;

      const ws = new WebSocket(wsUrl);

      ws.onopen = () => {
        console.log(`✅ WebSocket connected: ${endpoint}`);
        setIsConnected(true);
        setError(null);
        reconnectAttempts.current = 0;
        backoffRef.current = 1000; // Reset backoff on successful connection
        
        // Subscribe to channel if multiplexing is enabled
        if (channel) {
          ws.send(JSON.stringify({
            type: "subscribe",
            channel: channel,
          }));
        }
      };

      ws.onmessage = (event) => {
        try {
          const parsedData = JSON.parse(event.data);
          
          // OPTIMIZED v29: Map incoming signals to CSS variables for real-time visual updates
          if (parsedData.sentiment) {
            const sentiment = parsedData.sentiment.toLowerCase();
            const root = document.documentElement;
            
            if (sentiment === "bullish") {
              root.style.setProperty("--glow-color", "rgba(0, 255, 170, 0.8)");
              root.style.setProperty("--panel-opacity", "0.12");
            } else if (sentiment === "bearish") {
              root.style.setProperty("--glow-color", "rgba(255, 0, 122, 0.8)");
              root.style.setProperty("--panel-opacity", "0.12");
            } else {
              root.style.setProperty("--glow-color", "rgba(0, 209, 255, 0.8)");
              root.style.setProperty("--panel-opacity", "0.1");
            }
          }
          
          // Handle multiplexed messages
          if (channel && parsedData.channel === channel) {
            setData(parsedData.data || parsedData);
          } else if (!channel) {
            // Non-multiplexed: use data directly
            setData(parsedData);
          }
        } catch (e) {
          console.error("WebSocket message parse error:", e);
          setError("Failed to parse WebSocket message");
        }
      };

      ws.onerror = (error) => {
        console.error(`❌ WebSocket error on ${endpoint}:`, error);
        setError("WebSocket connection error");
        setIsConnected(false);
        // Close on error to trigger reconnect
        ws.close();
      };

      ws.onclose = () => {
        console.log(`🔌 WebSocket closed: ${endpoint}`);
        setIsConnected(false);

        // Auto-reconnect logic with exponential backoff
        if (
          autoConnect &&
          reconnectAttempts.current < maxReconnectAttempts
        ) {
          reconnectAttempts.current += 1;
          const delay = Math.min(backoffRef.current, 15000);
          backoffRef.current *= 2; // Exponential backoff
          
          reconnectTimeoutRef.current = setTimeout(() => {
            console.log(
              `🔄 Reconnecting... (attempt ${reconnectAttempts.current}/${maxReconnectAttempts}, delay: ${delay}ms)`
            );
            connect();
          }, delay);
        }
      };

      wsRef.current = ws;
    } catch (e) {
      console.error("WebSocket connection error:", e);
      setError(`Failed to connect: ${e.message}`);
      setIsConnected(false);
    }
  };

  const disconnect = () => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
    setIsConnected(false);
  };

  const send = (message) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(message));
    } else {
      console.warn("WebSocket is not connected. Cannot send message.");
    }
  };

  useEffect(() => {
    if (autoConnect) {
      connect();
    }

    return () => {
      disconnect();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [endpoint, autoConnect, channel]);

  return {
    data,
    error,
    isConnected,
    connect,
    disconnect,
    send,
  };
}
