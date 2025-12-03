/**
 * Quantum AI Cockpit — Cyberpunk Theme Configuration
 * 🎨 Neon colors, fonts, and animation presets
 * ==============================================
 */

export const cyberpunkTheme = {
  colors: {
    neonGreen: "#00ffaa",
    electricBlue: "#00d1ff",
    magenta: "#ff007a",
    gold: "#ffbb00",
    cyan: "#00ffe1",
    darkBg: "#0D0221",
    darkerBg: "#050014",
    cardBg: "rgba(15, 15, 15, 0.9)",
    borderColor: "rgba(0, 255, 170, 0.3)",
    textPrimary: "#ffffff",
    textSecondary: "#cccccc",
    textMuted: "#888888",
  },
  fonts: {
    primary: "'Orbitron', sans-serif",
    mono: "'Share Tech Mono', monospace",
    system: "system-ui, -apple-system, sans-serif",
  },
  shadows: {
    neonGreen: "0 0 20px rgba(0, 255, 170, 0.3)",
    electricBlue: "0 0 20px rgba(0, 209, 255, 0.3)",
    magenta: "0 0 20px rgba(255, 0, 122, 0.3)",
    gold: "0 0 20px rgba(255, 187, 0, 0.3)",
    glow: "0 0 25px rgba(0, 255, 170, 0.1)",
  },
  animations: {
    sidebarHover: {
      scale: 1.05,
      transition: { duration: 0.2 },
    },
    cardEntry: {
      initial: { opacity: 0, y: 20 },
      animate: { opacity: 1, y: 0 },
      transition: { duration: 0.5, ease: "easeOut" },
    },
    chartLoading: {
      shimmer: {
        background: "linear-gradient(90deg, transparent, rgba(0,255,170,0.3), transparent)",
        backgroundSize: "200% 100%",
        animation: "shimmer 2s infinite",
      },
    },
    pageTransition: {
      initial: { opacity: 0, x: -20 },
      animate: { opacity: 1, x: 0 },
      exit: { opacity: 0, x: 20 },
      transition: { duration: 0.3 },
    },
    breathingNeon: {
      animate: {
        boxShadow: [
          "0 0 10px rgba(0, 255, 170, 0.3)",
          "0 0 20px rgba(0, 255, 170, 0.5)",
          "0 0 10px rgba(0, 255, 170, 0.3)",
        ],
      },
      transition: {
        repeat: Infinity,
        duration: 3,
        ease: "easeInOut",
      },
    },
  },
  gradients: {
    darkBg: "linear-gradient(135deg, #0D0221 0%, #050014 100%)",
    cardBg: "linear-gradient(135deg, rgba(15,15,15,0.95) 0%, rgba(5,0,20,0.95) 100%)",
    neonBorder: "linear-gradient(90deg, transparent, rgba(0,255,170,0.5), transparent)",
  },
};

export default cyberpunkTheme;

