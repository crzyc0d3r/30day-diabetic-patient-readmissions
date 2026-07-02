import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { FeatureTable } from "../components/FeatureTable";
import type { ComputeResult } from "../types";

const result: ComputeResult = {
  computed_at: "2026-06-01T12:00:00Z",
  batch: "spike",
  reference_rows: 1000,
  current_rows: 500,
  verdict: "ALERT",
  summary: { n_features: 2, ok: 0, warn: 1, alert: 1 },
  thresholds: {
    psi: { warn: 0.1, alert: 0.2 },
    wasserstein: { warn: 0.1, alert: 0.25 },
    kl: { warn: 0.1, alert: 0.25 },
  },
  features: [
    {
      name: "time_in_hospital", type: "numeric",
      wasserstein: 0.42, psi: 0.31, kl: 0.27, status: "ALERT",
      histogram: { bins: [0, 1, 2], reference: [0.5, 0.5], current: [0.2, 0.8] },
    },
    {
      name: "race", type: "categorical",
      wasserstein: null, psi: 0.14, kl: 0.05, status: "WARN",
      histogram: { categories: ["a", "b"], reference: [0.6, 0.4], current: [0.5, 0.5] },
    },
  ],
};

describe("FeatureTable", () => {
  it("renders a row per feature with metrics and statuses", () => {
    render(<FeatureTable result={result} selected={null} onSelect={vi.fn()} />);

    expect(screen.getByText("time_in_hospital")).toBeInTheDocument();
    expect(screen.getByText("race")).toBeInTheDocument();

    // status pills
    expect(screen.getByText("ALERT")).toBeInTheDocument();
    expect(screen.getByText("WARN")).toBeInTheDocument();

    // categorical wasserstein renders as an em dash placeholder
    expect(screen.getByText("—")).toBeInTheDocument();

    // a numeric metric is formatted to 3 dp
    expect(screen.getByText("0.420")).toBeInTheDocument();
  });

  it("invokes onSelect when a row is clicked", () => {
    const onSelect = vi.fn();
    render(<FeatureTable result={result} selected={null} onSelect={onSelect} />);
    screen.getByText("race").closest("tr")!.click();
    expect(onSelect).toHaveBeenCalledWith("race");
  });
});
