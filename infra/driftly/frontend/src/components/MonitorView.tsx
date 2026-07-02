import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "../api";
import type { useTheme } from "../theme";
import type { HistoryRun, MonitorResult } from "../types";
import { VerdictCard } from "./VerdictCard";
import { FeatureTable } from "./FeatureTable";
import { DistributionPlot } from "./DistributionPlot";
import { TrendChart } from "./TrendChart";

type Palette = ReturnType<typeof useTheme>["palette"];
const POLL_MS = 20000;

function Opener({ num, light, accent }: { num: string; light: string; accent: string }) {
  return (
    <div className="opener">
      <div className="num">{num}</div>
      <h2><span className="light">{light}</span> <span className="accent">{accent}</span></h2>
    </div>
  );
}

function rank(a: { status: string; psi: number | null }, b: { status: string; psi: number | null }) {
  const order: Record<string, number> = { ALERT: 2, WARN: 1, OK: 0 };
  const d = (order[b.status] ?? 0) - (order[a.status] ?? 0);
  return d !== 0 ? d : (b.psi ?? 0) - (a.psi ?? 0);
}

export function MonitorView({ palette }: { palette: Palette }) {
  const [result, setResult] = useState<MonitorResult | null>(null);
  const [runs, setRuns] = useState<HistoryRun[]>([]);
  const [selectedFeature, setSelectedFeature] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [updatedAt, setUpdatedAt] = useState<Date | null>(null);
  const featRef = useRef<string | null>(null);
  featRef.current = selectedFeature;

  const poll = useCallback(async () => {
    try {
      const r = await api.monitor();
      setResult(r);
      setError(null);
      setUpdatedAt(new Date());
      if (!featRef.current) {
        const worst = [...r.features].sort(rank)[0];
        if (worst) setSelectedFeature(worst.name);
      }
    } catch (e: any) {
      setError(String(e?.message ?? e));
    }
  }, []);

  useEffect(() => {
    poll();
    const id = setInterval(poll, POLL_MS);
    return () => clearInterval(id);
  }, [poll]);

  useEffect(() => {
    api.history(selectedFeature ?? undefined).then(setRuns).catch(() => setRuns([]));
  }, [selectedFeature, result]);

  const focused = result?.features.find((f) => f.name === selectedFeature) ?? null;

  return (
    <>
      <section className="section" id="monitor">
        <Opener num="Monitor" light="Live drift of the" accent="running dataset" />
        <div style={{ display: "flex", gap: 12, alignItems: "center", marginBottom: 18, flexWrap: "wrap" }}>
          <span className="pill ok" style={{ background: "var(--accent-soft)", color: "var(--accent)" }}>● live</span>
          <span className="muted" style={{ fontSize: 12 }}>
            auto-refresh every {POLL_MS / 1000}s
            {updatedAt && ` · updated ${updatedAt.toLocaleTimeString()}`}
            {result && ` · running batch changed ${new Date(result.running_batch_mtime).toLocaleString()}`}
          </span>
          <button className="btn" style={{ padding: "6px 12px", fontSize: 12 }} onClick={poll}>Refresh now</button>
        </div>
        {error && <div className="callout error"><p>{error}</p></div>}
        {result ? <VerdictCard result={result} /> : !error && <p className="empty">Loading current drift…</p>}
      </section>

      <section className="section" id="monitor-features">
        <Opener num="Monitor" light="Per-feature" accent="heatmap" />
        {result
          ? <FeatureTable result={result} selected={selectedFeature} onSelect={setSelectedFeature} />
          : <p className="empty">Waiting for the first reading.</p>}
      </section>

      <section className="section" id="monitor-distribution">
        <Opener num="Monitor" light="Distribution" accent="overlay" />
        {focused
          ? <DistributionPlot feature={focused} palette={palette} />
          : <p className="empty">Select a feature in the heatmap to overlay its reference-vs-current distribution.</p>}
      </section>

      <section className="section" id="monitor-trend">
        <Opener num="Monitor" light="Drift" accent="trend" />
        <TrendChart runs={runs} feature={selectedFeature} palette={palette} />
      </section>
    </>
  );
}
