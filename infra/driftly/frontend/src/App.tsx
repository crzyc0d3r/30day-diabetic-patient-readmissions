import { useState } from "react";
import { useTheme } from "./theme";
import { Nav } from "./components/Nav";
import { Hero } from "./components/Hero";
import { MonitorView } from "./components/MonitorView";
import { SimulatorView } from "./components/SimulatorView";

export type View = "monitor" | "simulator";

export default function App() {
  const { toggle, palette } = useTheme();
  const [view, setView] = useState<View>("monitor");
  const [navOpen, setNavOpen] = useState(false);

  return (
    <div className="shell">
      <main className="doc">
        <Hero />
        {view === "monitor" ? <MonitorView palette={palette} /> : <SimulatorView />}
      </main>

      <Nav
        view={view}
        onSwitch={(v) => { setView(v); setNavOpen(false); }}
        onToggleTheme={toggle}
        open={navOpen}
        onMobileToggle={() => setNavOpen((o) => !o)}
      />
    </div>
  );
}
