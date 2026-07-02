import type {
  AirflowStatus,
  BatchInfo,
  ComputeResult,
  HistoryRun,
  InjectResult,
  MonitorResult,
  Scenario,
} from "./types";

// Same-origin: nginx (prod) / vite proxy (dev) forwards /api -> driftly-api.
async function json<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      if (body?.detail) detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* non-JSON error body */
    }
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

export const api = {
  health: () => fetch("/api/health").then(json<{ status: string }>),

  batches: () => fetch("/api/batches").then(json<BatchInfo[]>),

  features: () => fetch("/api/features").then(json<{ name: string; type: string }[]>),

  computeBatch: (batch: string) =>
    fetch("/api/compute", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ batch }),
    }).then(json<ComputeResult>),

  computeUpload: (file: File) => {
    const form = new FormData();
    form.append("file", file);
    return fetch("/api/compute/upload", { method: "POST", body: form }).then(json<ComputeResult>);
  },

  history: (feature?: string) =>
    fetch("/api/history" + (feature ? `?feature=${encodeURIComponent(feature)}` : "")).then(
      json<HistoryRun[]>,
    ),

  // --- Monitor: live drift of the running dataset ---
  monitor: () => fetch("/api/monitor").then(json<MonitorResult>),

  // --- Simulator: inject drift + (optionally) trigger Airflow ---
  scenarios: () => fetch("/api/simulator/scenarios").then(json<Scenario[]>),

  airflowStatus: () => fetch("/api/airflow/status").then(json<AirflowStatus>),

  inject: (body: { scenario: string; severity: number; trigger: boolean }) =>
    fetch("/api/simulator/inject", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }).then(json<InjectResult>),
};
