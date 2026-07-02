import type { View } from "../App";

const VIEWS: { id: View; num: string; label: string }[] = [
  { id: "monitor", num: "01", label: "Monitor" },
  { id: "simulator", num: "02", label: "Simulator" },
];

// External links to the rest of the stack, opened on whatever host the dashboard
// is served from (localhost or a remote IP) so they resolve from any browser.
const STACK: { label: string; port: number; path?: string }[] = [
  { label: "MLflow", port: 5000 },
  { label: "Ray Tune", port: 8265 },
  { label: "Airflow", port: 8080 },
  { label: "Grafana", port: 3003 },
  { label: "Prometheus", port: 9090 },
  { label: "API docs", port: 8002, path: "/docs" },
];

interface Props {
  view: View;
  onSwitch: (v: View) => void;
  onToggleTheme: () => void;
  open: boolean;
  onMobileToggle: () => void;
}

// Single adaptive sun/moon icon from README.html (CSS-driven via [data-theme]).
function ThemeIcon() {
  return (
    <svg className="theme-icon" viewBox="0 0 24 24" aria-hidden="true">
      <mask id="driftly-moon-mask">
        <rect x="0" y="0" width="24" height="24" fill="white" />
        <circle className="moon-cut" cx="17" cy="7" r="6" fill="black" />
      </mask>
      <circle className="sun-core" cx="12" cy="12" r="6" fill="currentColor" mask="url(#driftly-moon-mask)" />
      <g className="sun-rays" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
        <line x1="12" y1="1" x2="12" y2="3" />
        <line x1="12" y1="21" x2="12" y2="23" />
        <line x1="3.5" y1="3.5" x2="5.2" y2="5.2" />
        <line x1="18.8" y1="18.8" x2="20.5" y2="20.5" />
        <line x1="1" y1="12" x2="3" y2="12" />
        <line x1="21" y1="12" x2="23" y2="12" />
        <line x1="3.5" y1="20.5" x2="5.2" y2="18.8" />
        <line x1="18.8" y1="5.2" x2="20.5" y2="3.5" />
      </g>
    </svg>
  );
}

export function Nav({ view, onSwitch, onToggleTheme, open, onMobileToggle }: Props) {
  const host = typeof window !== "undefined" ? window.location.hostname : "localhost";
  return (
    <>
      <button className={"mobile-toggle" + (open ? " open" : "")} aria-label="Toggle index" onClick={onMobileToggle}>
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
          <line x1="3" y1="6" x2="21" y2="6" />
          <line x1="3" y1="12" x2="21" y2="12" />
          <line x1="3" y1="18" x2="15" y2="18" />
        </svg>
      </button>

      <aside className={"nav" + (open ? " open" : "")} id="nav">
        <div className="nav-head">
          <div className="brand">
            <div className="brand-glyph">d</div>
            <div className="brand-name">driftly<span>drift monitor + simulator</span></div>
          </div>
          <button className="theme-toggle" id="themeToggle" onClick={onToggleTheme} aria-label="Toggle theme">
            <ThemeIcon />
          </button>
        </div>

        <nav aria-label="Driftly views">
          <div className="nav-eyebrow">driftly</div>
          <ul className="nav-list">
            {VIEWS.map((v) => (
              <li key={v.id}>
                <a
                  href={`#${v.id}`}
                  className={view === v.id ? "active" : ""}
                  onClick={(e) => { e.preventDefault(); onSwitch(v.id); }}
                >
                  <span className="num">{v.num}</span>
                  <span>{v.label}</span>
                </a>
              </li>
            ))}
          </ul>

          <div className="nav-eyebrow">stack</div>
          <ul className="nav-list">
            {STACK.map((s) => (
              <li key={s.label}>
                <a href={`http://${host}:${s.port}${s.path ?? ""}`} target="_blank" rel="noreferrer">
                  <span className="num">&#8599;</span>
                  <span>{s.label}</span>
                </a>
              </li>
            ))}
          </ul>
        </nav>
      </aside>
    </>
  );
}
