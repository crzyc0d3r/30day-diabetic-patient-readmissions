import type { ComputeResult, FeatureResult, ThresholdBand } from "../types";

function band(value: number | null, t: ThresholdBand): string {
  if (value == null) return "";
  if (value >= t.alert) return "m-alert";
  if (value >= t.warn) return "m-warn";
  return "";
}

function metric(value: number | null): string {
  return value == null ? "—" : value.toFixed(3);
}

interface Props {
  result: ComputeResult;
  selected: string | null;
  onSelect: (name: string) => void;
}

export function FeatureTable({ result, selected, onSelect }: Props) {
  const { thresholds, features } = result;
  return (
    <>
      <div className="tbl-wrap">
        <table className="tbl">
          <thead>
            <tr>
              <th>Feature</th>
              <th>Type</th>
              <th className="num">Wasserstein</th>
              <th className="num">PSI</th>
              <th className="num">KL</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody>
            {features.map((f: FeatureResult) => (
              <tr
                key={f.name}
                className={selected === f.name ? "selected" : ""}
                onClick={() => onSelect(f.name)}
              >
                <td>{f.name}</td>
                <td>
                  <span className="type-tag">{f.type}</span>
                </td>
                <td className={`metric ${band(f.wasserstein, thresholds.wasserstein)}`}>
                  {metric(f.wasserstein)}
                </td>
                <td className={`metric ${band(f.psi, thresholds.psi)}`}>{metric(f.psi)}</td>
                <td className={`metric ${band(f.kl, thresholds.kl)}`}>{metric(f.kl)}</td>
                <td>
                  <span className={`pill ${f.status.toLowerCase()}`}>{f.status}</span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="muted" style={{ fontSize: 12, marginTop: -20 }}>
        Click a feature to see its reference-vs-current distribution below.
      </p>
    </>
  );
}
