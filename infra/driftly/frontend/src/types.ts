export type Status = "OK" | "WARN" | "ALERT";

export interface BatchInfo {
  name: string;
  rows: number | null;
  is_current: boolean;
}

export interface NumericHistogram {
  bins: number[];
  reference: number[];
  current: number[];
}
export interface CategoricalHistogram {
  categories: string[];
  reference: number[];
  current: number[];
}
export type Histogram = NumericHistogram | CategoricalHistogram;

export interface FeatureResult {
  name: string;
  type: "numeric" | "categorical";
  wasserstein: number | null;
  psi: number | null;
  kl: number | null;
  status: Status;
  histogram: Histogram;
}

export interface ThresholdBand {
  warn: number;
  alert: number;
}

export interface ComputeResult {
  computed_at: string;
  batch: string;
  reference_rows: number;
  current_rows: number;
  verdict: Status;
  summary: { n_features: number; ok: number; warn: number; alert: number };
  thresholds: { psi: ThresholdBand; wasserstein: ThresholdBand; kl: ThresholdBand };
  features: FeatureResult[];
  warning?: string;
}

export interface HistoryFeature {
  name: string;
  wasserstein: number | null;
  psi: number | null;
  kl: number | null;
  status: Status;
}
export interface HistoryRun {
  id: number;
  computed_at: string;
  batch: string;
  verdict: Status;
  ok: number;
  warn: number;
  alert: number;
  reference_rows: number;
  current_rows: number;
  feature?: HistoryFeature | null;
}

export function isCategorical(h: Histogram): h is CategoricalHistogram {
  return (h as CategoricalHistogram).categories !== undefined;
}

// --- two-view extensions: Monitor + Simulator ---
export interface MonitorResult extends ComputeResult {
  running_batch: string;
  running_batch_mtime: string;
}
export interface Scenario {
  name: string;
  description: string;
}
export interface AirflowStatus {
  configured: boolean;
  ui_url: string;
  dag_id: string;
}
export interface AirflowRun {
  dag_id: string;
  dag_run_id: string;
  state: string;
  run_url: string;
}
export interface InjectResult {
  injected: { scenario: string; severity: number; rows: number; path: string };
  triggered: boolean;
  airflow?: AirflowRun;
  trigger_note?: string;
  trigger_error?: string;
}
