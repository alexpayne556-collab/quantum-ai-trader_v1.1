/**
 * Quantum AI Cockpit — Performance Monitor
 * 📊 Performance monitoring utilities for FPS, render time, and WebSocket latency
 * ===============================================================================
 */

export class PerformanceMonitor {
  constructor() {
    this.metrics = {
      fps: 0,
      renderTime: 0,
      wsLatency: 0,
      memoryUsage: 0,
    };
    this.frameCount = 0;
    this.lastFrameTime = performance.now();
    this.fpsInterval = null;
  }

  startFPSMonitoring() {
    const measureFPS = () => {
      this.frameCount++;
      const now = performance.now();
      const elapsed = now - this.lastFrameTime;

      if (elapsed >= 1000) {
        this.metrics.fps = Math.round((this.frameCount * 1000) / elapsed);
        this.frameCount = 0;
        this.lastFrameTime = now;
      }

      requestAnimationFrame(measureFPS);
    };

    this.lastFrameTime = performance.now();
    measureFPS();
  }

  measureRenderTime(callback) {
    const startTime = performance.now();
    const result = callback();
    const endTime = performance.now();
    this.metrics.renderTime = endTime - startTime;
    return result;
  }

  measureWSLatency(sendTime, receiveTime) {
    this.metrics.wsLatency = receiveTime - sendTime;
    return this.metrics.wsLatency;
  }

  getMemoryUsage() {
    if (performance.memory) {
      this.metrics.memoryUsage = performance.memory.usedJSHeapSize / 1048576; // MB
    }
    return this.metrics.memoryUsage;
  }

  getMetrics() {
    return {
      ...this.metrics,
      memoryUsage: this.getMemoryUsage(),
      timestamp: new Date().toISOString(),
    };
  }

  exportMetrics() {
    return JSON.stringify(this.getMetrics(), null, 2);
  }
}

// Singleton instance
export const performanceMonitor = new PerformanceMonitor();

// Start FPS monitoring automatically
if (typeof window !== 'undefined') {
  performanceMonitor.startFPSMonitoring();
}
