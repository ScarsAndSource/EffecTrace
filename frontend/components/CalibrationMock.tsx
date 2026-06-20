import { TIER_COLOR } from "@/lib/visualTokens";

/**
 * CalibrationMock.tsx — static, fabricated visual prop (Section 15, Issue #16 / P1).
 *
 * Makes the long-term "outcome_calibrations" moat tangible on stage without a real
 * pipeline behind it. Every number below is hand-picked for the visual, not computed —
 * outcome_calibrations is an empty table at hackathon time (Section 12). This component
 * must never be wired to a live query; if that table ever gets a real read path,
 * build a new component rather than smuggling a live fetch in here.
 */

interface CalibrationRow {
  label: string;
  domain: string;
  before: { p10: number; p90: number };
  after: { p10: number; p90: number };
}

const MOCK_ROWS: CalibrationRow[] = [
  { label: "Customer churn rate", domain: "Customer", before: { p10: -0.41, p90: 0.08 }, after: { p10: -0.29, p90: -0.04 } },
  { label: "Support ticket volume", domain: "Operations", before: { p10: 0.05, p90: 0.62 }, after: { p10: 0.21, p90: 0.47 } },
  { label: "Competitor price response", domain: "Competitive", before: { p10: -0.18, p90: 0.55 }, after: { p10: 0.12, p90: 0.4 } },
];

function Bar({
  p10,
  p90,
  color,
  height = "h-2.5",
}: {
  p10: number;
  p90: number;
  color: string;
  height?: string;
}) {
  const toPct = (v: number) => ((v + 1) / 2) * 100; // map [-1,1] -> [0,100]
  const left = toPct(p10);
  const width = toPct(p90) - left;
  return (
    <div className={`relative ${height} w-full rounded-full`} style={{ background: "var(--color-border-subtle)" }}>
      <div className="absolute h-full rounded-full" style={{ left: `${left}%`, width: `${width}%`, background: color }} />
    </div>
  );
}

export default function CalibrationMock() {
  return (
    <div
      className="rounded-xl border border-l-2 px-5 py-4 font-sans"
      style={{
        background: "var(--color-bg-card)",
        borderColor: "var(--color-border)",
        borderLeftColor: TIER_COLOR.data_grounded,
      }}
    >
      <div className="flex items-center justify-between">
        <div className="font-display text-sm font-medium" style={{ color: "var(--color-text-primary)" }}>
          Confidence band, before vs. after 50 logged outcomes
        </div>
        <span
          className="rounded-full border px-2 py-0.5 font-mono text-[10px] uppercase tracking-wide"
          style={{ borderColor: "var(--color-border)", color: "var(--color-text-muted)" }}
        >
          Illustrative
        </span>
      </div>
      <p className="mt-1 text-xs leading-relaxed" style={{ color: "var(--color-text-muted)" }}>
        Every real calibration starts wide, like the gray-only bars below. Each logged outcome tightens it.
        This is a mock of the trend, not a live read from outcome_calibrations.
      </p>

      <div className="mt-4 space-y-4">
        {MOCK_ROWS.map((row) => (
          <div key={row.label}>
            <div className="mb-1 flex items-center justify-between text-xs">
              <span style={{ color: "var(--color-text-secondary)" }}>{row.label}</span>
              <span className="font-mono" style={{ color: "var(--color-text-muted)" }}>
                {row.domain}
              </span>
            </div>
            <div className="space-y-1.5">
              <div className="flex items-center gap-2">
                <span
                  className="w-12 text-right font-mono text-[9px]"
                  style={{ color: "var(--color-text-muted)" }}
                >
                  before
                </span>
                <div className="flex-1">
                  <Bar
                    p10={row.before.p10}
                    p90={row.before.p90}
                    color="var(--color-text-muted)"
                    height="h-3"
                  />
                </div>
              </div>
              <div className="flex items-center gap-2">
                <span
                  className="w-12 text-right font-mono text-[9px]"
                  style={{ color: TIER_COLOR.data_grounded }}
                >
                  after
                </span>
                <div className="flex-1">
                  <Bar
                    p10={row.after.p10}
                    p90={row.after.p90}
                    color={TIER_COLOR.data_grounded}
                    height="h-3"
                  />
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>

      <div className="mt-3 flex items-center gap-4 text-[10px]" style={{ color: "var(--color-text-muted)" }}>
        <span className="flex items-center gap-1">
          <span className="inline-block h-1.5 w-3 rounded-full" style={{ background: "var(--color-text-muted)" }} />
          Before (priors only)
        </span>
        <span className="flex items-center gap-1">
          <span className="inline-block h-1.5 w-3 rounded-full" style={{ background: TIER_COLOR.data_grounded }} />
          After 50 outcomes
        </span>
      </div>
    </div>
  );
}