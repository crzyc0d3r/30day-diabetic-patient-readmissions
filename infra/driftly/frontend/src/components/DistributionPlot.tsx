import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { FeatureResult } from "../types";
import { isCategorical } from "../types";
import type { useTheme } from "../theme";

type Palette = ReturnType<typeof useTheme>["palette"];

function buildData(feature: FeatureResult): { label: string; reference: number; current: number }[] {
  const h = feature.histogram;
  if (isCategorical(h)) {
    return h.categories.map((c, i) => ({
      label: c,
      reference: h.reference[i] ?? 0,
      current: h.current[i] ?? 0,
    }));
  }
  // numeric: label each bar by its bin's left edge (rounded)
  return h.reference.map((_, i) => {
    const lo = h.bins[i];
    return {
      label: Number.isFinite(lo) ? lo.toPrecision(3) : String(lo),
      reference: h.reference[i] ?? 0,
      current: h.current[i] ?? 0,
    };
  });
}

export function DistributionPlot({ feature, palette }: { feature: FeatureResult; palette: Palette }) {
  const data = buildData(feature);
  return (
      <div className="panel">
        <p className="panel-sub">
          <b style={{ color: "var(--ink)" }}>{feature.name}</b> · reference vs current — normalized
          frequency per bin ({feature.type}). Divergence between the two series is the drift the
          metrics above quantify.
        </p>
        <div style={{ width: "100%", height: 300 }}>
          <ResponsiveContainer>
            <BarChart data={data} margin={{ top: 6, right: 12, left: 0, bottom: 4 }} barGap={1}>
              <CartesianGrid strokeDasharray="3 3" stroke={palette.rule} vertical={false} />
              <XAxis dataKey="label" tick={{ fill: palette.ink3, fontSize: 10 }} stroke={palette.rule}
                interval="preserveStartEnd" />
              <YAxis tick={{ fill: palette.ink3, fontSize: 10 }} stroke={palette.rule}
                tickFormatter={(v) => `${Math.round(v * 100)}%`} />
              <Tooltip
                contentStyle={{ background: palette.bg2, border: `1px solid ${palette.rule}`, borderRadius: 8, fontSize: 12 }}
                labelStyle={{ color: palette.ink2 }}
                formatter={(v: number) => `${(v * 100).toFixed(1)}%`}
              />
              <Legend wrapperStyle={{ fontSize: 12 }} />
              <Bar name="reference" dataKey="reference" fill={palette.accent} radius={[2, 2, 0, 0]} />
              <Bar name="current" dataKey="current" fill={palette.violet} radius={[2, 2, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
  );
}
