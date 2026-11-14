import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ["var(--font-sans)", "system-ui", "Segoe UI", "sans-serif"],
        mono: ["var(--font-mono)", "ui-monospace", "Consolas", "monospace"],
      },
      colors: {
        // The NHS-blue world. The page ground is blue, panels are pale grey,
        // and every other colour carries a meaning: a verdict, a citation, a
        // warning or the keyboard focus.
        nhs: {
          blue: "#005EB8", // page ground, links on panels, primary data
          dark: "#003087", // header, headings on panels, selected trial
          pale: "#D8DDE0", // recessed areas inside panels
          panel: "#E8EDEE", // panels (never white)
          line: "#AEB7BD", // hairlines on panels
          unknown: "#AEB7BD", // the "insufficient info" band: neutral, not a warning
          ink: "#212B32", // text on panels
          mid: "#4C6272", // secondary text
          lightgreen: "#78BE20", // likely eligible
          green: "#007F3B", // met, eligible, the primary action
          greendark: "#00401E", // primary button edge
          aqua: "#41B6E6", // citations and the evidence highlight
          aquapale: "#D2EEFA", // highlighted sentence and selected rule
          red: "#D5281B", // not met, excluded, errors
          purple: "#330072", // exclusion rules
          warm: "#FFB81C", // the disclaimer's warning edge
          focus: "#FFEB3B", // keyboard focus, as on NHS services
        },
      },
      boxShadow: {
        card: "0 1px 2px rgba(0, 32, 80, 0.18), 0 10px 28px -14px rgba(0, 32, 80, 0.45)",
      },
    },
  },
  plugins: [],
};

export default config;
