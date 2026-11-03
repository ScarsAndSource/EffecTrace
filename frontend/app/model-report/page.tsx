"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import snapshot from "@/lib/mlReportSnapshot.json";
import { getDemoScenario } from "@/lib/demoMode";
import {
  fetchLiveReport,
  localCrosscheck,
  verdict,
  type ForecastSection,
  type MlReport,
} from "@/lib/mlReport";
import { COLOR } from "@/lib/visualTokens";

const f2 = (v: number) => v.toFixed(2);
const f3 = (v: number) => v.toFixed(3);
const pct = (v: number) => `${(v * 100).toFixed(1)}%`;
const mono = "font-mono tabular-nums";

function Card({ title, note, children }: { title: string; note?: string; children: React.ReactNode }) {
  return (
    <section
      className="rounded-xl border px-5 py-4"
      style={{ background: "var(--color-bg-card)", borderColor: "var(--color-border)" }}
    >
      <h3 className="font-display text-sm font-medium" style={{ color: "var(--color-text-primary)" }}>
        {title}
      </h3>
      {note && (
        <p className="mt-1 text-xs leading-relaxed" style={{ color: "var(--color-text-muted)" }}>
          {note}
        </p>
      )}
      <div className="mt-3">{children}</div>
    </section>
  );
}

function Stat({ label, value, sub, color }: { label: string; value: string; sub: string; color: string }) {
  return (
    <div
      className="rounded-xl border border-l-2 px-4 py-3"
      style={{ background: "var(--color-bg-card)", borderColor: "var(--color-border)", borderLeftColor: color }}
    >
      <div className="text-[11px] uppercase tracking-wide" style={{ color: "var(--color-text-muted)" }}>
        {label}
      </div>
      <div className={`mt-1 text-2xl ${mono}`}>{value}</div>
      <div className="mt-0.5 text-xs" style={{ color: "var(--color-text-secondary)" }}>
        {sub}
      </div>
    </div>
  );
}

function Table({ head, rows, hi }: { head: string[]; rows: (string | number)[][]; hi?: number }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-left text-xs">
        <thead>
          <tr style={{ color: "var(--color-text-muted)" }}>
            {head.map((h) => (
              <th key={h} className="whitespace-nowrap px-2 py-1.5 font-medium">
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr
              key={i}
              style={{
                borderTop: "1px solid var(--color-border-subtle)",
                background: i === hi ? "rgba(139,143,245,0.08)" : undefined,
              }}
            >
              {r.map((c, j) => (
                <td key={j} className={`whitespace-nowrap px-2 py-1.5 ${j > 0 ? mono : ""}`}>
                  {c}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function HBars({ data, fmt, color }: { data: Record<string, number>; fmt: (v: number) => string; color: string }) {
  const max = Math.max(...Object.values(data).map(Math.abs), 1e-9);
  return (
    <div className="space-y-1.5">
      {Object.entries(data).map(([k, v]) => (
        <div key={k} className="flex items-center gap-2 text-xs">
          <span className="w-28 shrink-0 truncate" style={{ color: "var(--color-text-secondary)" }}>
            {k}
          </span>
          <div className="h-2 flex-1 rounded-full" style={{ background: "var(--color-border-subtle)" }}>
            <div className="h-full rounded-full" style={{ width: `${(Math.abs(v) / max) * 100}%`, background: color }} />
          </div>
          <span className={`w-12 text-right ${mono}`}>{fmt(v)}</span>
        </div>
      ))}
    </div>
  );
}

const MODEL_COLOR: Record<string, string> = {
  logistic_regression: COLOR.polarityAmplify,
  random_forest: COLOR.tierHistoricallyPrecedented,
  hist_gradient_boosting: COLOR.tierDataGrounded,
};

function RocChart({ roc }: { roc: MlReport["churn"]["roc"] }) {
  const S = 240;
  const P = 28;
  const x = (v: number) => P + v * (S - P - 8);
  const y = (v: number) => S - P - v * (S - P - 8);
  return (
    <svg viewBox={`0 0 ${S} ${S}`} className="w-full max-w-[280px]" role="img" aria-label="ROC curves">
      <line x1={x(0)} y1={y(0)} x2={x(1)} y2={y(1)} stroke={COLOR.textMuted} strokeDasharray="3 3" />
      <line x1={x(0)} y1={y(0)} x2={x(1)} y2={y(0)} stroke={COLOR.border} />
      <line x1={x(0)} y1={y(0)} x2={x(0)} y2={y(1)} stroke={COLOR.border} />
      {Object.entries(roc).map(([m, p]) => (
        <polyline
          key={m}
          fill="none"
          stroke={MODEL_COLOR[m] ?? COLOR.textSecondary}
          strokeWidth={1.6}
          points={p.fpr.map((fx, i) => `${x(fx)},${y(p.tpr[i])}`).join(" ")}
        />
      ))}
      <text x={S / 2} y={S - 6} textAnchor="middle" fontSize="9" fill={COLOR.textMuted}>
        false positive rate
      </text>
      <text x={8} y={S / 2} fontSize="9" fill={COLOR.textMuted} transform={`rotate(-90 8 ${S / 2})`} textAnchor="middle">
        true positive rate
      </text>
    </svg>
  );
}

function ForecastChart({ fc }: { fc: ForecastSection }) {
  const s = fc.series_for_plot;
  const W = 720;
  const H = 240;
  const P = 30;
  const all = [...s.history, ...s.actual, ...s.upper95, ...s.lower95];
  const lo = Math.min(...all);
  const hi = Math.max(...all);
  const n = s.history.length + s.actual.length;
  const x = (i: number) => P + (i / (n - 1)) * (W - P - 10);
  const y = (v: number) => H - P - ((v - lo) / (hi - lo)) * (H - P - 10);
  const off = s.history.length;
  const line = (arr: number[], start: number) => arr.map((v, i) => `${x(start + i)},${y(v)}`).join(" ");
  const band =
    s.upper95.map((v, i) => `${x(off + i)},${y(v)}`).join(" ") +
    " " +
    [...s.lower95].reverse().map((v, i) => `${x(off + s.lower95.length - 1 - i)},${y(v)}`).join(" ");
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label="24-month hold-out forecast">
      <polygon points={band} fill={COLOR.tierSpeculative} opacity={0.18} />
      <polyline points={line(s.history, 0)} fill="none" stroke={COLOR.textMuted} strokeWidth={1.4} />
      <polyline points={line(s.actual, off)} fill="none" stroke={COLOR.textPrimary} strokeWidth={1.8} />
      <polyline points={line(s.forecast, off)} fill="none" stroke={COLOR.tierSpeculative} strokeWidth={1.8} />
      <line x1={x(off)} x2={x(off)} y1={10} y2={H - P} stroke={COLOR.border} strokeDasharray="3 3" />
      <text x={x(off) + 4} y={20} fontSize="9" fill={COLOR.textMuted}>
        hold-out →
      </text>
    </svg>
  );
}

function AcfBars({ vals, band, label }: { vals: number[]; band: number; label: string }) {
  const W = 340;
  const H = 110;
  const mid = H / 2;
  const bw = (W - 20) / vals.length;
  const sy = (v: number) => (v / 0.6) * (mid - 8);
  return (
    <div>
      <div className="mb-1 text-[11px]" style={{ color: "var(--color-text-muted)" }}>
        {label}
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label={label}>
        <line x1={10} x2={W - 10} y1={mid} y2={mid} stroke={COLOR.border} />
        {[band, -band].map((b) => (
          <line key={b} x1={10} x2={W - 10} y1={mid - sy(b)} y2={mid - sy(b)} stroke={COLOR.tierSpeculative} strokeDasharray="3 3" opacity={0.6} />
        ))}
        {vals.map((v, i) => (
          <rect
            key={i}
            x={10 + i * bw + 1}
            width={bw - 2}
            y={v >= 0 ? mid - sy(v) : mid}
            height={Math.abs(sy(v))}
            fill={Math.abs(v) > band ? COLOR.decisionRoot : COLOR.textMuted}
          />
        ))}
      </svg>
    </div>
  );
}

function ClaimVsEvidence({ report }: { report: MlReport }) {
  const edges = localCrosscheck(getDemoScenario());
  const specs = Object.entries(report.churn.price_elasticity.by_spec);
  const cited = edges.find((e) => e.cited_elasticity !== null)?.cited_elasticity ?? null;
  const W = 640;
  const P = 20;
  const x = (v: number) => P + ((v + 1) / 2) * (W - 2 * P);
  const rowH = 44;
  const H = 30 + specs.length * rowH + 20;
  return (
    <div>
      {edges.map((e) => (
        <p key={e.target_id} className="mb-3 text-xs leading-relaxed" style={{ color: "var(--color-text-secondary)" }}>
          <span className={mono} style={{ color: COLOR.decisionRoot }}>
            decision_root → {e.target_id}
          </span>{" "}
          (LLM magnitude {f2(e.magnitude_estimate)}): “{e.citation}”
        </p>
      ))}
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label="Elasticity intervals versus the LLM claim">
        <line x1={x(0)} x2={x(0)} y1={8} y2={H - 14} stroke={COLOR.textMuted} />
        {cited !== null && (
          <g>
            <line x1={x(cited)} x2={x(cited)} y1={8} y2={H - 14} stroke={COLOR.tierSpeculative} strokeDasharray="4 3" strokeWidth={1.6} />
            <text x={x(cited) + 4} y={16} fontSize="10" fill={COLOR.tierSpeculative}>
              LLM claim ≈ {f2(cited)}
            </text>
          </g>
        )}
        {specs.map(([name, s], i) => {
          const cy = 40 + i * rowH;
          return (
            <g key={name}>
              <line x1={x(Math.max(-1, s.ci95[0]))} x2={x(Math.min(1, s.ci95[1]))} y1={cy} y2={cy} stroke={COLOR.tierDataGrounded} strokeWidth={4} strokeLinecap="round" />
              <circle cx={x(s.elasticity)} cy={cy} r={5} fill={COLOR.bg} stroke={COLOR.tierDataGrounded} strokeWidth={2} />
              <text x={x(-1)} y={cy - 10} fontSize="10" fill={COLOR.textSecondary}>
                {name.replace("_", " ")}: {f2(s.elasticity)} [{f2(s.ci95[0])}, {f2(s.ci95[1])}]
              </text>
            </g>
          );
        })}
        {[-1, -0.5, 0, 0.5, 1].map((t) => (
          <text key={t} x={x(t)} y={H - 2} textAnchor="middle" fontSize="9" fill={COLOR.textMuted}>
            {t}
          </text>
        ))}
      </svg>
      {cited !== null && (
        <p className="mt-2 text-xs leading-relaxed" style={{ color: "var(--color-text-secondary)" }}>
          {specs.map(([name, s]) => `${name.replace("_", " ")}: claim is ${verdict(cited, s.ci95)} the 95% interval`).join(" · ")}
          . Definitions are close but not identical (telecom customers vs. the SaaS study the model cited), so treat
          this as a flag for review, not a refutation.
        </p>
      )}
    </div>
  );
}

export default function ModelReportPage() {
  const [report, setReport] = useState<MlReport>(snapshot as unknown as MlReport);
  const [source, setSource] = useState<"snapshot" | "live">("snapshot");

  useEffect(() => {
    fetchLiveReport().then((r) => {
      if (r) {
        setReport(r);
        setSource("live");
      }
    });
  }, []);

  const { churn, forecast } = report;
  const sel = churn.models[churn.selected_model];
  const base = churn.models["dummy_majority"];
  const fcNames = Object.keys(forecast.holdout_metrics);
  const sarimaKey = fcNames[0];
  const cm = sel.test.confusion_matrix;

  return (
    <main className="mx-auto max-w-5xl space-y-5 px-5 py-8">
      <header className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <Link href="/" className="text-xs" style={{ color: "var(--color-text-muted)" }}>
            ← back
          </Link>
          <h1 className="mt-1 font-display text-3xl font-medium">Model report</h1>
          <p className="mt-1 max-w-2xl text-sm leading-relaxed" style={{ color: "var(--color-text-secondary)" }}>
            The causal graph is an LLM-elicited hypothesis. This page is the empirical layer: models trained offline
            on public datasets, with held-out evaluation, to check what the graph claims.
          </p>
        </div>
        <span className={`rounded-full border px-2.5 py-1 text-[10px] uppercase tracking-wide ${mono}`} style={{ borderColor: "var(--color-border)", color: "var(--color-text-muted)" }}>
          {source === "live" ? "live /ml/report" : "bundled snapshot"} · {report.generated_at.slice(0, 10)}
        </span>
      </header>

      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        <Stat label="Churn ROC-AUC (test)" value={f3(sel.test.roc_auc)} sub={churn.selected_model.replace(/_/g, " ")} color={COLOR.tierDataGrounded} />
        <Stat label="Accuracy vs baseline" value={pct(sel.test.accuracy)} sub={`majority-class baseline ${pct(base.test.accuracy)}`} color={COLOR.tierHistoricallyPrecedented} />
        <Stat label="SARIMA hold-out MAPE" value={`${forecast.holdout_metrics[sarimaKey].mape_pct.toFixed(1)}%`} sub={`24 months · seasonal naive ${forecast.holdout_metrics["seasonal_naive"].mape_pct.toFixed(1)}%`} color={COLOR.decisionRoot} />
        <Stat label="Ljung-Box p (lag 12)" value={f2(forecast.residual_diagnostics.ljung_box_pvalue_lag12)} sub="> 0.05 → residuals ≈ white noise" color={COLOR.polarityAmplify} />
      </div>

      <h2 className="pt-2 font-display text-xl">1 · Churn classification</h2>
      <Card
        title="Models compared"
        note={`${churn.dataset}: ${churn.eda.n_rows.toLocaleString()} customers, ${churn.eda.n_features} features, ${pct(churn.eda.churn_rate)} churn. Stratified ${churn.split.train}/${churn.split.test} split; scaling and one-hot encoding are fitted inside each CV fold (no leakage). Selection rule: ${churn.selection_rule}.`}
      >
        <Table
          head={["model", "CV AUC (mean ± sd)", "accuracy", "precision", "recall", "F1", "test AUC", "Brier"]}
          hi={Object.keys(churn.models).indexOf(churn.selected_model)}
          rows={Object.entries(churn.models).map(([m, r]) => [
            m.replace(/_/g, " "),
            `${f3(r.cv5.roc_auc.mean)} ± ${f3(r.cv5.roc_auc.std)}`,
            f3(r.test.accuracy),
            f3(r.test.precision),
            f3(r.test.recall),
            f3(r.test.f1),
            f3(r.test.roc_auc),
            f3(r.test.brier),
          ])}
        />
        <p className="mt-2 text-xs leading-relaxed" style={{ color: "var(--color-text-muted)" }}>
          Accuracy alone is misleading here: always predicting “no churn” already scores {pct(base.test.accuracy)}.
          The three real models are within one CV standard deviation of each other on AUC, so the choice between
          them is a recall/precision trade-off, not a clear win.
        </p>
      </Card>

      <div className="grid gap-5 md:grid-cols-3">
        <Card title="ROC (held-out test)">
          <RocChart roc={churn.roc} />
          <div className="mt-2 space-y-0.5 text-[11px]">
            {Object.keys(churn.roc).map((m) => (
              <div key={m} className="flex items-center gap-2" style={{ color: "var(--color-text-secondary)" }}>
                <span className="inline-block h-1.5 w-3 rounded" style={{ background: MODEL_COLOR[m] }} />
                {m.replace(/_/g, " ")}
              </div>
            ))}
          </div>
        </Card>
        <Card title={`Confusion matrix · ${churn.selected_model.replace(/_/g, " ")}`} note="Threshold 0.5, test set.">
          <div className={`grid grid-cols-2 gap-1.5 text-center text-sm ${mono}`}>
            {[
              ["TN", cm.tn, COLOR.tierDataGrounded],
              ["FP", cm.fp, COLOR.tierSpeculative],
              ["FN", cm.fn, COLOR.tierSpeculative],
              ["TP", cm.tp, COLOR.tierDataGrounded],
            ].map(([k, v, c]) => (
              <div key={k as string} className="rounded-lg border py-3" style={{ borderColor: c as string, background: "var(--color-bg-elevated)" }}>
                <div className="text-[10px]" style={{ color: "var(--color-text-muted)" }}>{k}</div>
                {v}
              </div>
            ))}
          </div>
        </Card>
        <Card title="EDA: where churn concentrates">
          <div className="mb-1 text-[11px]" style={{ color: "var(--color-text-muted)" }}>by contract</div>
          <HBars data={churn.eda.churn_rate_by_contract} fmt={pct} color={COLOR.tierSpeculative} />
          <div className="mb-1 mt-3 text-[11px]" style={{ color: "var(--color-text-muted)" }}>by internet service</div>
          <HBars data={churn.eda.churn_rate_by_internet} fmt={pct} color={COLOR.polarityAmplify} />
        </Card>
      </div>

      <h2 className="pt-2 font-display text-xl">2 · SARIMA forecasting</h2>
      <Card
        title="Stationarity and order selection"
        note={`${forecast.dataset}, n=${forecast.eda.n} (train ${forecast.eda.train_months} / hold-out ${forecast.eda.test_months}). ${forecast.eda.transform_rationale}`}
      >
        <div className="grid gap-5 md:grid-cols-2">
          <Table
            head={["series", "ADF p-value"]}
            rows={Object.entries(forecast.eda.adf_pvalue).map(([k, v]) => [k.replace(/_/g, " "), f3(v)])}
          />
          <Table
            head={["top AIC (train only)", "AIC"]}
            hi={0}
            rows={forecast.selection.top5.map((g) => [`(${g.order.join(",")})(${g.seasonal_order.join(",")})`, f2(g.aic)])}
          />
        </div>
        <div className="mt-4 grid gap-5 md:grid-cols-2">
          <AcfBars vals={forecast.eda.acf_lags_1_24} band={forecast.eda.acf_pacf_95_band} label="ACF of log series, d=1, D=1 (lag 12)" />
          <AcfBars vals={forecast.eda.pacf_lags_1_24} band={forecast.eda.acf_pacf_95_band} label="PACF, same series" />
        </div>
        <p className="mt-2 text-xs" style={{ color: "var(--color-text-muted)" }}>
          Spikes at lag 1 and lag 12 point to MA(1) × seasonal MA(1), matching the AIC-selected airline model.
        </p>
      </Card>

      <Card title="24-month hold-out forecast" note="Grey: train. White: actual. Red: forecast with 95% prediction interval.">
        <ForecastChart fc={forecast} />
        <div className="mt-3">
          <Table
            head={["model", "MAE", "RMSE", "MAPE %", "MASE"]}
            hi={0}
            rows={Object.entries(forecast.holdout_metrics).map(([k, v]) => [k.replace(/_/g, " "), f2(v.mae), f2(v.rmse), f2(v.mape_pct), f3(v.mase)])}
          />
        </div>
        <p className="mt-2 text-xs leading-relaxed" style={{ color: "var(--color-text-muted)" }}>
          Holt-Winters beats the AIC-selected SARIMA on this hold-out, and SARIMA under-forecasts the 1959-60
          acceleration. 95% interval coverage on the 24 hold-out points is {pct(forecast.interval_coverage_95)}; with
          n=24 that says little about calibration.
        </p>
        <div className="mt-3">
          <Table
            head={["rolling origin", "SARIMA MAPE %", "seasonal naive MAPE %"]}
            rows={forecast.rolling_origin.map((r) => [r.origin, f2(r.sarima_mape_pct), f2(r.seasonal_naive_mape_pct)])}
          />
        </div>
      </Card>

      <h2 className="pt-2 font-display text-xl">3 · LLM claim vs. evidence</h2>
      <Card
        title="Price → churn edge in the demo scenario"
        note={`Bootstrap (${churn.price_elasticity.by_spec[Object.keys(churn.price_elasticity.by_spec)[0]].n_boot} resamples) elasticity of mean predicted churn to a ${pct(churn.price_elasticity.price_shock)} price shock, from a logistic model fitted on real customers. Dot = point estimate, bar = 95% CI.`}
      >
        <ClaimVsEvidence report={report} />
        <ul className="mt-3 list-disc space-y-1 pl-5 text-xs leading-relaxed" style={{ color: "var(--color-text-muted)" }}>
          {churn.price_elasticity.caveats.map((c) => (
            <li key={c}>{c}</li>
          ))}
        </ul>
      </Card>

      <Card title="What this is not">
        <p className="text-xs leading-relaxed" style={{ color: "var(--color-text-secondary)" }}>
          These models are validated on their own public datasets. They are not trained on EffecTrace scenarios, the
          forecaster is not wired into the Monte Carlo propagation, and the cross-check covers only pricing decisions.
          Versions: {Object.entries(report.versions).map(([k, v]) => `${k} ${v}`).join(" · ")}.
        </p>
      </Card>
    </main>
  );
}
