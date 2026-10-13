/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // SignBridge AI Design System
        brand: {
          primary:   "#4338CA",
          secondary: "#6366F1",
          accent:    "#14B8A6",
        },
        surface: {
          bg:     "#F8FAFC",
          card:   "#FFFFFF",
          border: "#E2E8F0",
        },
        content: {
          primary:   "#0F172A",
          secondary: "#64748B",
        },
        status: {
          success: "#16A34A",
          warning: "#F59E0B",
          error:   "#DC2626",
        },
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
      },
      boxShadow: {
        card:  "0 1px 3px 0 rgba(0,0,0,0.07), 0 1px 2px -1px rgba(0,0,0,0.05)",
        modal: "0 20px 60px -10px rgba(0,0,0,0.15)",
      },
      borderRadius: {
        xl:  "0.75rem",
        "2xl": "1rem",
        "3xl": "1.5rem",
      },
      animation: {
        "fade-in":    "fadeIn 0.2s ease-out",
        "slide-up":   "slideUp 0.25s ease-out",
        "pulse-slow": "pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite",
      },
      keyframes: {
        fadeIn: {
          "0%":   { opacity: "0" },
          "100%": { opacity: "1" },
        },
        slideUp: {
          "0%":   { opacity: "0", transform: "translateY(8px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
      },
    },
  },
  plugins: [
    require("@tailwindcss/forms"),
  ],
};
