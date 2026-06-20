import { TIER_COLOR, TIER_LABEL, TIER_DESCRIPTION, COLOR } from "@/lib/visualTokens";
import type { ConfidenceTier } from "@/lib/types";

const TIERS: ConfidenceTier[] = ["data_grounded", "historically_precedented", "speculative"];

export default function ConfidenceLegend() {
  return (
    <div
      className="flex flex-wrap items-center gap-x-5 gap-y-2 rounded-xl border px-4 py-3 font-sans text-xs"
      style={{ background: "var(--color-bg-card)", borderColor: "var(--color-border)" }}
    >
      <span className="shrink-0" style={{ color: "var(--color-text-muted)" }}>
        Confidence
      </span>
      {TIERS.map((tier) => (
        <div key={tier} className="flex items-center gap-1.5" title={TIER_DESCRIPTION[tier]}>
          <span
            className="inline-block h-2.5 w-2.5 rounded-full border-2"
            style={{ borderColor: TIER_COLOR[tier], background: "transparent" }}
          />
          <span style={{ color: "var(--color-text-secondary)" }}>{TIER_LABEL[tier]}</span>
        </div>
      ))}
      <div className="flex items-center gap-1.5" title="The decision being analysed">
        <span
          className="inline-block h-2.5 w-2.5 rounded-full"
          style={{ background: COLOR.decisionRoot }}
        />
        <span style={{ color: "var(--color-text-secondary)" }}>Decision</span>
      </div>
    </div>
  );
}
