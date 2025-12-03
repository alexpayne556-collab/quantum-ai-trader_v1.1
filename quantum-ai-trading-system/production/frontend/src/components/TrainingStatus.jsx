// frontend/src/components/TrainingStatus.jsx
import React, { useState, useEffect } from "react";
import { motion } from "framer-motion";

export default function TrainingStatus() {
  const [trainingStatus, setTrainingStatus] = useState(null);
  const [loading, setLoading] = useState(true);
  const [training, setTraining] = useState(false);

  useEffect(() => {
    fetchTrainingStatus();
    const interval = setInterval(fetchTrainingStatus, 30000); // Update every 30 seconds
    return () => clearInterval(interval);
  }, []);

  const fetchTrainingStatus = async () => {
    try {
      const response = await fetch("/api/training/status");
      if (response.ok) {
        const data = await response.json();
        setTrainingStatus(data);
      }
      setLoading(false);
    } catch (error) {
      console.error("Failed to fetch training status:", error);
      setLoading(false);
    }
  };

  const startTraining = async () => {
    setTraining(true);
    try {
      const response = await fetch("/api/training/train_all?months=12&force_retrain=true", {
        method: "POST"
      });
      if (response.ok) {
        const data = await response.json();
        // Show success message
        console.log("Training started:", data);
        // Refresh status periodically while training
        const interval = setInterval(() => {
          fetchTrainingStatus();
        }, 10000); // Check every 10 seconds
        
        // Stop checking after 5 minutes or when training completes
        setTimeout(() => {
          clearInterval(interval);
          setTraining(false);
          fetchTrainingStatus();
        }, 300000); // 5 minutes
      } else {
        setTraining(false);
        alert("Failed to start training. Please check the console for errors.");
      }
    } catch (error) {
      console.error("Failed to start training:", error);
      setTraining(false);
      alert("Error starting training: " + error.message);
    }
  };

  if (loading) {
    return (
      <div className="bg-[rgba(15,15,15,0.9)] backdrop-blur-md border border-cyan-500/20 rounded-xl shadow-[0_0_25px_rgba(0,255,170,0.1)] p-4">
        <div className="text-cyan-400 text-sm">Loading training status...</div>
      </div>
    );
  }

  const summary = trainingStatus?.summary;

  return (
    <div className="bg-[rgba(15,15,15,0.9)] backdrop-blur-md border border-cyan-500/20 rounded-xl shadow-[0_0_25px_rgba(0,255,170,0.1)] p-4">
      <div className="flex justify-between items-center mb-4">
        <h3 className="text-electric-blue text-lg font-semibold font-orbitron">
          🎯 Module Training Status
        </h3>
        <button
          onClick={startTraining}
          disabled={training}
          className={`px-4 py-2 rounded text-sm font-semibold transition-all ${
            training
              ? "bg-gray-500/20 text-gray-400 cursor-not-allowed"
              : "bg-neon-green/20 text-neon-green border border-neon-green/40 hover:bg-neon-green/30 hover:shadow-[0_0_15px_rgba(0,255,170,0.3)]"
          }`}
          title="Train all modules with 6+ months historical data until 75%+ accuracy"
        >
          {training ? "🔄 Training..." : "🎯 Train & Tune All Modules"}
        </button>
      </div>

      {!summary ? (
        <div className="text-center text-gray-400 py-8 text-sm">
          No training has been run yet. Click "Start Training" to begin.
        </div>
      ) : (
        <div className="space-y-4">
          {/* Summary Stats */}
          <div className="grid grid-cols-3 gap-2">
            <div className="bg-[rgba(0,0,0,0.3)] rounded p-2">
              <div className="text-xs text-gray-400">Modules</div>
              <div className="text-sm font-semibold text-cyan-400">
                {summary.total_modules}
              </div>
            </div>
            <div className="bg-[rgba(0,0,0,0.3)] rounded p-2">
              <div className="text-xs text-gray-400">Targets Met</div>
              <div className={`text-sm font-semibold ${
                summary.targets_met === summary.total_modules
                  ? "text-neon-green"
                  : "text-gold"
              }`}>
                {summary.targets_met}/{summary.total_modules}
              </div>
            </div>
            <div className="bg-[rgba(0,0,0,0.3)] rounded p-2">
              <div className="text-xs text-gray-400">Avg Accuracy</div>
              <div className={`text-sm font-semibold ${
                summary.average_accuracy >= 0.80
                  ? "text-neon-green"
                  : summary.average_accuracy >= 0.75
                  ? "text-gold"
                  : "text-magenta"
              }`}>
                {(summary.average_accuracy * 100).toFixed(1)}%
              </div>
              <div className="text-xs text-gray-500 mt-1">
                Min: 75%
              </div>
            </div>
          </div>

          {/* Module Results */}
          <div className="space-y-2 max-h-64 overflow-y-auto">
            {summary.module_results && Object.entries(summary.module_results).map(([module, result]) => (
              <motion.div
                key={module}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className={`p-2 rounded border ${
                  result.target_met
                    ? "bg-[rgba(0,255,0,0.1)] border-green-500/30"
                    : "bg-[rgba(255,0,122,0.1)] border-magenta/30"
                }`}
              >
                <div className="flex justify-between items-center">
                  <div className="flex-1">
                    <div className="text-xs font-semibold text-neon-green">{module}</div>
                    <div className="text-xs text-gray-400 mt-1">
                      {result.training_samples} samples | {result.epochs} epochs
                    </div>
                  </div>
                  <div className="text-right">
                    <div className={`text-sm font-semibold font-mono ${
                      result.min_accuracy_met 
                        ? (result.target_met ? "text-neon-green" : "text-gold")
                        : "text-magenta"
                    }`}>
                      {(result.accuracy * 100).toFixed(1)}%
                    </div>
                    <div className="text-xs text-gray-400">
                      {result.min_accuracy_met 
                        ? (result.target_met ? "✅ Target Met" : "✅ Min Met")
                        : "⚠️ Below Min"}
                    </div>
                  </div>
                </div>
                {result.calibration_applied && (
                  <div className="text-xs text-gold mt-1">
                    🔧 Self-calibration applied
                  </div>
                )}
              </motion.div>
            ))}
          </div>

          {/* Training Info */}
          <div className="text-xs text-gray-400 pt-2 border-t border-cyan-500/20">
            <div>Historical Period: {summary.historical_months} months</div>
            <div>Target Accuracy: {(summary.target_accuracy * 100).toFixed(0)}%</div>
            <div>Last Updated: {new Date(summary.timestamp).toLocaleString()}</div>
          </div>
        </div>
      )}
    </div>
  );
}

