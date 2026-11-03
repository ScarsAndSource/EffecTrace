/**
 * mlReport.ts — types + loaders for the offline-trained ML evidence layer.
 *
 * The numbers come from backend/ml/artifacts/report.json (written by `python -m ml.run_all`).
 * lib/mlReportSnapshot.json is a copy of that file so /model-report works in Demo Mode with
 * zero network calls, same pattern as demoScenario.json. When a backend is configured the
 * page upgrades to the live GET /ml/report response.
 *
 * Refresh the snapshot after re-running the pipeline:
 *   cp backend/ml/artifacts/report.json frontend/lib/mlReportSnapshot.json
 */

import type { CausalGraphOutput } from "./types";

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL ?? "";

export interface ClassifierMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  roc_auc: number;
  brier: number;
  confusion_matrix: { tn: number; fp: number; fn: number; tp: number };
}
export interface CvStat {
  mean: number;
  std: number;
}
export interface ModelResult {
  cv5: Record<"accuracy" | "precision" | "recall" | "f1" | "roc_auc", CvStat>;
  test: ClassifierMetrics;
}
export interface ElasticitySpec {
  baseline_churn_prob: number;
  shocked_churn_prob: number;
  abs_change_pp: number;
  elasticity: number;
  ci95: [number, number];
  boot_median: number;
  boot_share_positive: number;
  n_boot: number;
}
export interface ChurnSection {
  dataset: string;
  eda: {
    n_rows: number;
    n_features: number;
    churn_rate: number;
    numeric_corr_with_churn: Record<string, number>;
    churn_rate_by_contract: Record<string, number>;
    churn_rate_by_internet: Record<string, number>;
  };
  split: { train: number; test: number };
  models: Record<string, ModelResult>;
  roc: Record<string, { fpr: number[]; tpr: number[] }>;
  selected_model: string;
  selection_rule: string;
  price_elasticity: {
    price_shock: number;
    definition: string;
    by_spec: Record<string, ElasticitySpec>;
    caveats: string[];
  };
}
export interface ForecastMetrics {
  mae: number;
  rmse: number;
  mape_pct: number;
  mase: number;
}
export interface ForecastSection {
  dataset: string;
  eda: {
    n: number;
    train_months: number;
    test_months: number;
    adf_pvalue: Record<string, number>;
    acf_lags_1_24: number[];
    pacf_lags_1_24: number[];
    acf_pacf_95_band: number;
    transform_rationale: string;
  };
  selection: { grid_size: number; top5: { order: number[]; seasonal_order: number[]; aic: number }[] };
  selected: { order: number[]; seasonal_order: number[] };
  holdout_metrics: Record<string, ForecastMetrics>;
  residual_diagnostics: { ljung_box_pvalue_lag12: number; ljung_box_pvalue_lag24: number };
  interval_coverage_95: number;
  rolling_origin: { origin: string; sarima_mape_pct: number; seasonal_naive_mape_pct: number }[];
  series_for_plot: {
    history: number[];
    actual: number[];
    forecast: number[];
    lower95: number[];
    upper95: number[];
  };
}
export interface MlReport {
  generated_at: string;
  runtime_s: number;
  versions: Record<string, string>;
  churn: ChurnSection;
  forecast: ForecastSection;
}

/** Live report from the backend, or null when no backend / artifact / network. */
export async function fetchLiveReport(): Promise<MlReport | null> {
  if (!BACKEND_URL) return null;
  try {
    const res = await fetch(`${BACKEND_URL}/ml/report`);
    if (!res.ok) return null;
    return (await res.json()) as MlReport;
  } catch {
    return null;
  }
}

export interface CrosscheckEdge {
  target_id: string;
  magnitude_estimate: number;
  citation: string;
  /** Elasticity quoted in the LLM citation ("1% price increase yields ~0.5% ..."), best-effort parse. */
  cited_elasticity: number | null;
}

/** Same idea as POST /ml/crosscheck, computed locally so Demo Mode needs no network. */
export function localCrosscheck(graph: CausalGraphOutput): CrosscheckEdge[] {
  return graph.edges
    .filter((e) => e.source_id === "decision_root" && e.target_id.includes("churn"))
    .map((e) => {
      const m = e.rationale_citation.match(
        /(\d+(?:\.\d+)?)%\s+price increase\s+yields\s+~?(\d+(?:\.\d+)?)%/i
      );
      return {
        target_id: e.target_id,
        magnitude_estimate: e.magnitude_estimate,
        citation: e.rationale_citation,
        cited_elasticity: m ? Number(m[2]) / Number(m[1]) : null,
      };
    });
}

export function verdict(cited: number, ci: [number, number]): "inside" | "above" | "below" {
  if (cited > ci[1]) return "above";
  if (cited < ci[0]) return "below";
  return "inside";
}
