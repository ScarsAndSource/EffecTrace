import { TIER_COLOR, TIER_LABEL, TIER_DESCRIPTION, COLOR } from "@/lib/visualTokens";
import type { ConfidenceTier } from "@/lib/types";

const TIERS: ConfidenceTier[] = ["data_grounded", "historically_precedented", "speculative"];

export default function ConfidenceLegend() {
  return (
    <div
      className="flex flex-wrap items-center gap-x-6 gap-y-2 rounded-xl border px-4 py-3 font-sans text-xs"
      style={{ background: "var(--color-bg-card)", borderColor: "var(--color-border)" }}
    >
      {/* Group 1 — Node confidence tiers */}
      <div className="flex flex-wrap items-center gap-3" role="group" aria-label="Node confidence tiers">
        <span
          className="shrink-0 font-mono text-[10px] uppercase tracking-wider"
          style={{ color: "var(--color-text-muted)" }}
        >
          Nodes
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

      {/* Divider */}
      <div className="h-4 w-px" style={{ background: "var(--color-border)" }} />

      {/* Group 2 — Edge polarity */}
      <div className="flex flex-wrap items-center gap-3" role="group" aria-label="Edge polarity">
        <span
          className="shrink-0 font-mono text-[10px] uppercase tracking-wider"
          style={{ color: "var(--color-text-muted)" }}
        >
          Edges
        </span>
        <div className="flex items-center gap-1.5" title="This effect amplifies downstream consequences">
          <span
            className="inline-block h-0.5 w-5 rounded-full"
            style={{ background: COLOR.polarityAmplify }}
          />
          <span style={{ color: "var(--color-text-secondary)" }}>Amplifies</span>
        </div>
        <div className="flex items-center gap-1.5" title="This effect dampens downstream consequences">
          <span
            className="inline-block h-0.5 w-5 rounded-full"
            style={{ background: COLOR.polarityDampen }}
          />
          <span style={{ color: "var(--color-text-secondary)" }}>Dampens</span>
        </div>
      </div>
    </div>
  );
}