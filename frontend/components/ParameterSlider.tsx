"use client";

import { useMemo } from "react";
import type { CausalGraphOutput, ParameterOverrides } from "@/lib/types";
import { topDirectEffectsByMagnitude } from "@/lib/graphPropagation";

interface ParameterSliderProps {
  graph: CausalGraphOutput;
  overrides: ParameterOverrides;
  onChange: (next: ParameterOverrides) => void;
}

const MIN = 0.2;
const MAX = 2.0;
const STEP = 0.1;

export default function ParameterSlider({ graph, overrides, onChange }: ParameterSliderProps) {
  // Top 3 highest-magnitude direct effects (by their decision_root edge magnitude).
  // Sliding one scales THAT node's own outgoing edges — the override contract
  // from graph_service.py / graphPropagation.ts (Section 10.5) — not the incoming
  // edge used purely to rank it here.
  const targets = useMemo(() => topDirectEffectsByMagnitude(graph, 3), [graph]);
  const nodeById = useMemo(() => new Map(graph.nodes.map((n) => [n.id, n])), [graph]);

  if (targets.length === 0) return null;

  function handleSlide(nodeId: string, value: number) {
    onChange({ ...overrides, [nodeId]: value });
  }

  function handleReset() {
    onChange({});
  }

  const hasAnyOverride = Object.keys(overrides).length > 0;

  return (
    <div
      className="fixed inset-x-0 bottom-0 z-10 border-t backdrop-blur-md"
      style={{ background: "rgba(10,11,15,0.88)", borderColor: "var(--color-border)" }}
    >
      <div className="mx-auto flex max-w-5xl flex-wrap items-center gap-x-8 gap-y-3 px-6 py-4">
        <span
          className="shrink-0 font-sans text-[11px] uppercase tracking-wider"
          style={{ color: "var(--color-text-muted)" }}
        >
          Adjust magnitude
        </span>

        {targets.map(({ node_id }) => {
          const node = nodeById.get(node_id);
          if (!node) return null;
          const current = overrides[node_id] ?? 1.0;

          return (
            <label key={node_id} className="flex min-w-[180px] flex-1 flex-col gap-1">
              <span className="flex items-center justify-between font-sans text-xs">
                <span style={{ color: "var(--color-text-secondary)" }}>{node.label}</span>
                <span className="font-mono" style={{ color: "var(--color-text-primary)" }}>
                  {current.toFixed(1)}×
                </span>
              </span>
              <input
                type="range"
                min={MIN}
                max={MAX}
                step={STEP}
                value={current}
                onChange={(e) => handleSlide(node_id, parseFloat(e.target.value))}
                className="accent-[var(--color-decision-root)]"
              />
            </label>
          );
        })}

        <button
          type="button"
          onClick={handleReset}
          disabled={!hasAnyOverride}
          className="shrink-0 rounded-lg border px-3 py-1.5 font-sans text-xs transition-opacity disabled:opacity-30"
          style={{ borderColor: "var(--color-border)", color: "var(--color-text-secondary)" }}
        >
          Reset
        </button>
      </div>
    </div>
  );
}
