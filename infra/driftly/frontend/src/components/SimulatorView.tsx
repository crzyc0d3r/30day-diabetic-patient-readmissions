import { useEffect, useState } from "react";
import { api } from "../api";
import type { AirflowStatus, InjectResult, Scenario } from "../types";

function Opener({ num, light, accent }: { num: string; light: string; accent: string }) {
  return (
    <div className="opener">
      <div className="num">{num}</div>
      <h2><span className="light">{light}</span> <span className="accent">{accent}</span></h2>
    </div>
  );
}

export function SimulatorView() {
  const [scenarios, setScenarios] = useState<Scenario[]>([]);
  const [scenario, setScenario] = useState("mixed_severe");
  const [severity, setSeverity] = useState(1.0);
  const [triggerAirflow, setTriggerAirflow] = useState(true);
  const [af, setAf] = useState<AirflowStatus | null>(null);
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<InjectResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.scenarios().then(setScenarios).catch(() => setScenarios([]));
    api.airflowStatus().then(setAf).catch(() => setAf(null));
  }, []);

  const run = async (overrideScenario?: string) => {
    const scen = overrideScenario ?? scenario;
    setBusy(true);
    setError(null);
    try {
      const r = await api.inject({
        scenario: scen,
        severity,
        trigger: triggerAirflow && !!af?.configured,
      });
      setResult(r);
    } catch (e: any) {
      setError(String(e?.message ?? e));
    } finally {
      setBusy(false);
    }
  };

  const desc = scenarios.find((s) => s.name === scenario)?.description;

  return (
    <section className="section" id="simulator">
      <Opener num="Simulator" light="Inject drift into the" accent="running dataset" />
      <p className="muted" style={{ fontSize: 13, marginTop: -8, marginBottom: 18, maxWidth: 720 }}>
        Generates the chosen scenario from the reference and writes it to{" "}
        <code>data/incoming/current.csv</code> — the live slot the Monitor reads and Airflow's{" "}
        <code>scheduled_drift_check</code> watches. With the trigger on, it also fires that DAG,
        which re-validates drift and cascades to <code>retrain_on_drift</code> on a confirmed ALERT.
      </p>

      <div className="controls">
        <div className="field">
          <label htmlFor="scenario">Scenario</label>
          <select id="scenario" value={scenario} onChange={(e) => setScenario(e.target.value)} disabled={busy}>
            {scenarios.map((s) => (
              <option key={s.name} value={s.name}>{s.name}</option>
            ))}
          </select>
        </div>

        <div className="field">
          <label htmlFor="severity">Severity · {severity.toFixed(1)}×</label>
          <input
            id="severity" type="range" min={0.2} max={3} step={0.1} value={severity}
            onChange={(e) => setSeverity(parseFloat(e.target.value))} disabled={busy}
            style={{ minWidth: 200 }}
          />
        </div>

        <div className="field">
          <label htmlFor="trig">Trigger Airflow</label>
          <label style={{ display: "inline-flex", alignItems: "center", gap: 8, fontSize: 13, color: "var(--ink-2)" }}>
            <input
              id="trig" type="checkbox" checked={triggerAirflow && !!af?.configured}
              onChange={(e) => setTriggerAirflow(e.target.checked)}
              disabled={busy || !af?.configured}
            />
            {af?.configured ? `run ${af.dag_id}` : "Airflow not configured"}
          </label>
        </div>

        <button className="btn primary" onClick={() => run()} disabled={busy}>
          {busy ? <span className="spinner" /> : null}
          {busy ? "Injecting…" : "Inject drift"}
        </button>
        <button className="btn" onClick={() => run("none")} disabled={busy} title="Reset the running batch to a clean resample">
          Reset to clean
        </button>
      </div>

      {desc && <p className="muted" style={{ fontSize: 12.5, marginTop: 10 }}>{desc}</p>}
      {error && <div className="callout error" style={{ marginTop: 16 }}><p>{error}</p></div>}

      {result && (
        <div className="cell" style={{ marginTop: 24, maxWidth: "var(--content-max)" }}>
          <div className="k" style={{ fontFamily: "var(--mono)", fontSize: 10, letterSpacing: "0.14em", textTransform: "uppercase", color: "var(--ink-3)" }}>
            injected
          </div>
          <div style={{ fontSize: 15, marginTop: 6 }}>
            Wrote <b>{result.injected.scenario}</b> ({result.injected.severity.toFixed(1)}×,{" "}
            {result.injected.rows.toLocaleString()} rows) to the live slot. Switch to{" "}
            <b>Monitor</b> to watch it land.
          </div>
          {result.triggered && result.airflow ? (
            <div style={{ fontSize: 13, marginTop: 10, color: "var(--ink-2)" }}>
              Triggered <code>{result.airflow.dag_id}</code> ·{" "}
              run <code>{result.airflow.dag_run_id}</code> ({result.airflow.state}) ·{" "}
              <a href={result.airflow.run_url} target="_blank" rel="noreferrer">open in Airflow ↗</a>
            </div>
          ) : (
            <div className="muted" style={{ fontSize: 12.5, marginTop: 10 }}>
              {result.trigger_error
                ? `Trigger failed: ${result.trigger_error}`
                : result.trigger_note ?? "Pipeline not triggered (toggle off)."}
            </div>
          )}
        </div>
      )}
    </section>
  );
}
