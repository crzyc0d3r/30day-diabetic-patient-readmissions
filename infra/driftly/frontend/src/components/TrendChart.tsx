import {
  CartesianGrid,
  Line,
  LineChart,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { HistoryRun } from "../types";
import type { useTheme } from "../theme";

type Palette = ReturnType<typeof useTheme>["palette"];

function shortTime(iso: string): string {
  const d = new Date(iso);
  return isNaN(d.getTime()) ? iso : d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

interface Props {
  runs: HistoryRun[];
  feature: string | null;
  palette: Palette;
}

export function TrendChart({ runs, feature, palette }: Props) {
  if (runs.length === 0) {
    return (
      <div className="panel">
        <p className="empty">No history yet. Each compute is recorded here to build a trend.</p>
      </div>
    );
  }

  const perFeature = !!feature && runs.some((r) => r.feature);
  const data = runs.map((r) => ({
    label: shortTime(r.computed_at),
    drifted: r.warn + r.alert,
    alert: r.alert,
    wasserstein: r.feature?.wasserstein ?? null,
    psi: r.feature?.psi ?? null,
    kl: r.feature?.kl ?? null,
  }));

  return (
      <div className="panel">
        <p className="panel-sub">
          {perFeature
            ? `Per-run metric values for ${feature} (select a different feature above to repivot).`
            : "Drifted-feature count per recorded run. Select a feature above to chart its metrics instead."}
        </p>
        <div style={{ width: "100%", height: 280 }}>
          <ResponsiveContainer>
            <LineChart data={data} margin={{ top: 6, right: 12, left: 0, bottom: 4 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={palette.rule} vertical={false} />
              <XAxis dataKey="label" tick={{ fill: palette.ink3, fontSize: 10 }} stroke={palette.rule} />
              <YAxis tick={{ fill: palette.ink3, fontSize: 10 }} stroke={palette.rule} allowDecimals={!perFeature ? false : true} />
              <Tooltip
                contentStyle={{ background: palette.bg2, border: `1px solid ${palette.rule}`, borderRadius: 8, fontSize: 12 }}
                labelStyle={{ color: palette.ink2 }}
              />
              <Legend wrapperStyle={{ fontSize: 12 }} />
              {perFeature ? (
                <>
                  <Line type="monotone" name="Wasserstein" dataKey="wasserstein" stroke={palette.accent} strokeWidth={2} dot={{ r: 2 }} connectNulls />
                  <Line type="monotone" name="PSI" dataKey="psi" stroke={palette.violet} strokeWidth={2} dot={{ r: 2 }} connectNulls />
                  <Line type="monotone" name="KL" dataKey="kl" stroke={palette.warn} strokeWidth={2} dot={{ r: 2 }} connectNulls />
                </>
              ) : (
                <>
                  <Line type="monotone" name="drifted features" dataKey="drifted" stroke={palette.accent} strokeWidth={2} dot={{ r: 2 }} />
                  <Line type="monotone" name="alerts" dataKey="alert" stroke={palette.alert} strokeWidth={2} dot={{ r: 2 }} />
                </>
              )}
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
  );
}
