/**
 * Quantum AI Cockpit — Error Boundary
 * Catches React errors and displays a friendly error message instead of blank screen
 */

import React from "react";
import { motion } from "framer-motion";

class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null, errorInfo: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true };
  }

  componentDidCatch(error, errorInfo) {
    console.error("ErrorBoundary caught an error:", error, errorInfo);
    this.setState({
      error,
      errorInfo,
    });
  }

  handleReset = () => {
    this.setState({ hasError: false, error: null, errorInfo: null });
  };

  render() {
    if (this.state.hasError) {
      return (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="min-h-screen flex items-center justify-center bg-gradient-to-br from-[#0D0221] via-[#050014] to-[#0D0221] p-8"
        >
          <div className="max-w-2xl w-full bg-[rgba(15,15,15,0.95)] backdrop-blur-md border-2 border-magenta/50 rounded-xl shadow-[0_0_30px_rgba(255,0,122,0.3)] p-8">
            <div className="text-center mb-6">
              <div className="text-6xl mb-4">⚠️</div>
              <h1 className="text-3xl font-bold text-magenta mb-2 font-orbitron">
                Quantum Error Detected
              </h1>
              <p className="text-gray-300 text-lg">
                The cockpit encountered an unexpected issue, but we've contained it.
              </p>
            </div>

            <div className="bg-black/50 rounded-lg p-4 mb-6 border border-cyan-500/20">
              <h2 className="text-neon-green font-semibold mb-2">Error Details:</h2>
              <pre className="text-xs text-gray-400 overflow-auto max-h-40 font-mono">
                {this.state.error?.toString() || "Unknown error"}
              </pre>
            </div>

            {process.env.NODE_ENV === "development" && this.state.errorInfo && (
              <div className="bg-black/50 rounded-lg p-4 mb-6 border border-cyan-500/20">
                <h2 className="text-neon-green font-semibold mb-2">Stack Trace:</h2>
                <pre className="text-xs text-gray-400 overflow-auto max-h-40 font-mono">
                  {this.state.errorInfo.componentStack}
                </pre>
              </div>
            )}

            <div className="flex gap-4 justify-center">
              <button
                onClick={this.handleReset}
                className="px-6 py-3 bg-neon-green/20 text-neon-green border border-neon-green/40 rounded-lg font-semibold hover:bg-neon-green/30 transition-all hover:shadow-[0_0_15px_rgba(0,255,170,0.3)]"
              >
                🔄 Try Again
              </button>
              <button
                onClick={() => window.location.reload()}
                className="px-6 py-3 bg-electric-blue/20 text-electric-blue border border-electric-blue/40 rounded-lg font-semibold hover:bg-electric-blue/30 transition-all hover:shadow-[0_0_15px_rgba(0,209,255,0.3)]"
              >
                🔃 Reload Page
              </button>
            </div>

            <div className="mt-6 text-center text-sm text-gray-500">
              If this persists, check the browser console for more details.
            </div>
          </div>
        </motion.div>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;

