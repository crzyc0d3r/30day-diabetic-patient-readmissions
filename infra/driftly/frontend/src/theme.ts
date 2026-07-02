import { useCallback, useEffect, useState } from "react";
import type { Status } from "./types";

export type ThemeName = "dark" | "light";
const KEY = "driftly-theme";

// Chart colors mirror the CSS design tokens for each theme (README.html). Kept
// as explicit hex (rather than reading getComputedStyle) so Recharts gets stable
// values synchronously on every theme flip.
interface Palette {
  accent: string; // reference series
  violet: string; // current series
  ok: string;
  warn: string;
  alert: string;
  ink2: string;
  ink3: string;
  rule: string;
  bg2: string;
}

const PALETTES: Record<ThemeName, Palette> = {
  dark: {
    accent: "#5b9eff", violet: "#a78bfa", ok: "#4ade80", warn: "#fbbf24",
    alert: "#f472b6", ink2: "#a8a8b4", ink3: "#82828f", rule: "#2c2c38", bg2: "#111116",
  },
  light: {
    accent: "#2563eb", violet: "#7c3aed", ok: "#16a34a", warn: "#d97706",
    alert: "#db2777", ink2: "#3a3a45", ink3: "#5e5e6a", rule: "#cdcdd6", bg2: "#f3f3f6",
  },
};

export function statusColor(p: Palette, s: Status): string {
  return s === "ALERT" ? p.alert : s === "WARN" ? p.warn : p.ok;
}

function readTheme(): ThemeName {
  const attr = document.documentElement.getAttribute("data-theme");
  return attr === "light" ? "light" : "dark";
}

/** Theme state + toggle, persisted to localStorage and applied to <html>. */
export function useTheme() {
  const [theme, setTheme] = useState<ThemeName>(readTheme);

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    try {
      localStorage.setItem(KEY, theme);
    } catch {
      /* ignore storage failures (private mode) */
    }
  }, [theme]);

  const toggle = useCallback(() => setTheme((t) => (t === "dark" ? "light" : "dark")), []);
  const palette = PALETTES[theme];
  return { theme, toggle, palette };
}
