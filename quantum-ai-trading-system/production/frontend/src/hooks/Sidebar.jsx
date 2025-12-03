import React from "react";
import "./Sidebar.css";
import { useWebSocketFeed } from "../hooks/useWebSocketFeed";
import apiManifest from "../../../backend/output/api_manifest.json";

/**
 * Sidebar component for Quantum AI Cockpit
 * Displays live module status indicators linked to backend WebSocket streams
 */
export default function Sidebar() {
  const modules = Object.keys(apiManifest);

  return (
    <div className="sidebar">
      <h2 className="sidebar-title">Modules</h2>
      <ul className="module-list">
        {modules.map((m) => {
          const wsUrl = `ws://localhost:8000${apiManifest[m].ws}`;
          const { status } = useWebSocketFeed(wsUrl);

          return (
            <li key={m} className={`module-item ${status}`}>
              {m}
              <span className={`status-dot ${status}`}></span>
            </li>
          );
        })}
      </ul>
    </div>
  );
}
