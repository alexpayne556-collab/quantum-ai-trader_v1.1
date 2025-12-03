/**
 * Quantum AI Cockpit — Lazy Loader
 * 🚀 Lazy loading utilities for heavy components
 * ==============================================
 */

import { lazy } from 'react';

// Lazy load Plotly components
export const LazyPlotlyGraph = lazy(() => import('../components/PlotlyGraph'));

// Lazy load heavy chart libraries
export const LazyPlotly = lazy(() => import('plotly.js-dist-min'));

// Lazy load D3 if needed in the future
// export const LazyD3 = lazy(() => import('d3'));

// Loading fallback component
export const LoadingFallback = () => (
  <div className="flex items-center justify-center h-full min-h-[400px]">
    <div className="text-center">
      <div className="animate-spin rounded-full h-12 w-12 border-4 border-cyan-500/30 border-t-cyan-500 mx-auto mb-4" />
      <div className="text-cyan-400 font-mono text-sm">Loading chart...</div>
    </div>
  </div>
);
