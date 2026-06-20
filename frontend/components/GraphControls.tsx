"use client";

import { useMemo } from "react";
import type { CausalGraphOutput, ConfidenceTier } from "@/lib/types";
import { TIER_COLOR, TIER_LABEL } from "@/lib/visualTokens";
import type { GraphViewApi } from "./CausalGraph";

interface GraphControlsProps {
  graph: CausalGraphOutput;
  api: GraphViewApi | null;
  activeTiers: ConfidenceTier[];
  onTiersChange: (tiers: ConfidenceTier[]) => void;
  maxHorizonDays: number | null;
  onHorizonChange: (days: number | null) => void;
}

const ALL_TIERS: ConfidenceTier[] = ["data_grounded", "historically_precedented", "speculative"];

export default function GraphControls({
  graph,
  api,
  activeTiers,
  onTiersChange,
  maxHorizonDays,
  onHorizonChange,
}: GraphControlsProps) {
  // Bounds derived from the actual graph, not hardcoded — count-agnostic the
  // same way the layout is (Section 9.4): works whether horizons span 30-180
  // days or 7-720.
  const horizonBounds = useMemo(() => {
    const days = graph.edges.map((e) => e.time_horizon_days);
    if (days.length === 0) return { min: 0, max: 365 };
    return { min: Math.min(...days), max: Math.max(...days) };
  }, [graph]);

  const allActive = activeTiers.length === ALL_TIERS.length;
  const horizonValue = maxHorizonDays ?? horizonBounds.max;

  function toggleTier(tier: ConfidenceTier) {
    if (activeTiers.includes(tier)) {
      const next = activeTiers.filter((t) => t !== tier);
      // Never allow zero tiers selected — that's an empty graph, not a filter.
      onTiersChange(next.length === 0 ? ALL_TIERS : next);
    } else {
      onTiersChange([...activeTiers, tier]);
    }
  }

  return (
    <div
      className="flex flex-wrap items-center gap-x-5 gap-y-2 rounded-xl border px-4 py-2.5 font-sans text-xs"
      style={{ background: "var(--color-bg-card)", borderColor: "var(--color-border)" }}
    >
      <div className="flex items-center gap-1.5">
        <button
          type="button"
          onClick={() => api?.zoomOut()}
          disabled={!api}
          className="rounded-md border px-2 py-1 disabled:opacity-30"
          style={{ borderColor: "var(--color-border)", color: "var(--color-text-secondary)" }}
          aria-label="Zoom out"
        >
          −
        </button>
        <button
          type="button"
          onClick={() => api?.fitView()}
          disabled={!api}
          className="rounded-md border px-2 py-1 disabled:opacity-30"
          style={{ borderColor: "var(--color-border)", color: "var(--color-text-secondary)" }}
        >
          Fit
        </button>
        <button
          type="button"
          onClick={() => api?.zoomIn()}
          disabled={!api}
          className="rounded-md border px-2 py-1 disabled:opacity-30"
          style={{ borderColor: "var(--color-border)", color: "var(--color-text-secondary)" }}
          aria-label="Zoom in"
        >
          +
        </button>
      </div>

      <div className="h-4 w-px" style={{ background: "var(--color-border)" }} />

      <div className="flex items-center gap-2">
        <span style={{ color: "var(--color-text-muted)" }}>Confidence</span>
        {ALL_TIERS.map((tier) => {
          const on = activeTiers.includes(tier);
          return (
            <button
              key={tier}
              type="button"
              onClick={() => toggleTier(tier)}
              className="flex items-center gap-1 rounded-full border px-2 py-1 transition-opacity"
              style={{
                borderColor: TIER_COLOR[tier],
                opacity: on ? 1 : 0.35,
                color: "var(--color-text-secondary)",
              }}
            >
              <span className="inline-block h-2 w-2 rounded-full" style={{ background: TIER_COLOR[tier] }} />
              {TIER_LABEL[tier]}
            </button>
          );
        })}
        {!allActive && (
          <button
            type="button"
            onClick={() => onTiersChange(ALL_TIERS)}
            className="underline underline-offset-2"
            style={{ color: "var(--color-text-muted)" }}
          >
            Reset
          </button>
        )}
      </div>

      <div className="h-4 w-px" style={{ background: "var(--color-border)" }} />

      <label className="flex items-center gap-2">
        <span style={{ color: "var(--color-text-muted)" }}>Horizon ≤</span>
        <input
          type="range"
          min={horizonBounds.min}
          max={horizonBounds.max}
          value={horizonValue}
          onChange={(e) => {
            const v = parseInt(e.target.value, 10);
            // Snapping back to "max" means "no cap" — keeps the slider's resting
            // state identical to the unfiltered graph rather than an arbitrary number.
            onHorizonChange(v >= horizonBounds.max ? null : v);
          }}
          className="accent-[var(--color-decision-root)]"
        />
        <span className="font-mono" style={{ color: "var(--color-text-primary)" }}>
          {horizonValue}d
        </span>
      </label>
    </div>
  );
}
