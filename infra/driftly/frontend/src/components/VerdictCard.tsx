import type { ComputeResult } from "../types";

function fmtTime(iso: string): string {
  const d = new Date(iso);
  return isNaN(d.getTime()) ? iso : d.toLocaleString();
}

export function VerdictCard({ result }: { result: ComputeResult }) {
  const v = result.verdict.toLowerCase() as "ok" | "warn" | "alert";
  const { summary } = result;
  const drifted = summary.warn + summary.alert;
  return (
    <>
      <div className={`verdict ${v}`}>
        <div className="verdict-badge">{result.verdict}</div>
        <div className="verdict-meta">
          <Stat k="Batch" v={result.batch} />
          <Stat k="Drifted features" v={`${drifted} / ${summary.n_features}`} />
          <Stat k="Warn / Alert" v={`${summary.warn} / ${summary.alert}`} />
          <Stat k="Reference rows" v={result.reference_rows.toLocaleString()} />
          <Stat k="Current rows" v={result.current_rows.toLocaleString()} />
          <Stat k="Computed" v={fmtTime(result.computed_at)} />
        </div>
      </div>
      {result.warning && <div className="callout"><p>{result.warning}</p></div>}
    </>
  );
}

function Stat({ k, v }: { k: string; v: string }) {
  return (
    <div>
      <div className="k">{k}</div>
      <div className="v">{v}</div>
    </div>
  );
}
