"use client";

import { useMemo, useState, useCallback, useEffect } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import CausalGraph, { type GraphViewApi } from "@/components/CausalGraph";
import ConfidenceLegend from "@/components/ConfidenceLegend";
import ParameterSlider from "@/components/ParameterSlider";
import GraphControls from "@/components/GraphControls";
import NarrativePanel, { GenerateMemoButton } from "@/components/NarrativePanel";
import CalibrationMock from "@/components/CalibrationMock";
import { runClientSimulation } from "@/lib/graphPropagation";
import { domainHue } from "@/lib/visualTokens";
import {
  DEMO_MODE,
  DEMO_SESSION_ID,
  getDemoGenerateResult,
  getDemoNarrative,
} from "@/lib/demoMode";
import type {
  CausalGraphOutput,
  OutcomeMap,
  ParameterOverrides,
  ConfidenceTier,
  Domain,
} from "@/lib/types";

function loadFromStorage<T>(key: string): T | null {
  if (typeof window === "undefined") return null;
  try {
    const raw = sessionStorage.getItem(key);
    return raw ? (JSON.parse(raw) as T) : null;
  } catch {
    return null;
  }
}

export default function ScenarioPage() {
  const params = useParams();
  const sessionId = params?.id as string ?? "";

  const [graphApi, setGraphApi] = useState<GraphViewApi | null>(null);
  const [overrides, setOverrides] = useState<ParameterOverrides>({});
  const [activeTiers, setActiveTiers] = useState<ConfidenceTier[]>([
    "data_grounded",
    "historically_precedented",
    "speculative",
  ]);
  const [maxHorizonDays, setMaxHorizonDays] = useState<number | null>(null);
  const [narrativeOpen, setNarrativeOpen] = useState(false);
  const [narrative, setNarrative] = useState<string | null>(null);
  const [narrativeLoading, setNarrativeLoading] = useState(false);
  const [narrativeError, setNarrativeError] = useState<string | null>(null);

  // Load graph data
  const { graph, outcomes } = useMemo(() => {
    if (DEMO_MODE || sessionId === DEMO_SESSION_ID) {
      const result = getDemoGenerateResult();
      return { graph: result.graph, outcomes: result.outcomes };
    }

    const storedGraph = loadFromStorage<CausalGraphOutput>("live_graph");
    const storedOutcomes = loadFromStorage<OutcomeMap>("live_outcomes");
    if (storedGraph && storedOutcomes) {
      return { graph: storedGraph, outcomes: storedOutcomes };
    }

    const demoResult = getDemoGenerateResult();
    return { graph: demoResult.graph, outcomes: demoResult.outcomes };
  }, [sessionId]);

  // Re-compute outcomes when slider overrides change
  const computedOutcomes = useMemo(() => {
    if (Object.keys(overrides).length === 0) return outcomes;
    return runClientSimulation(graph, overrides);
  }, [graph, outcomes, overrides]);

  // Reset overrides when graph changes
  useEffect(() => {
    setOverrides({});
    setNarrative(null);
    setNarrativeError(null);
    setNarrativeOpen(false);
  }, [graph]);

  function handleSliderChange(next: ParameterOverrides) {
    setOverrides(next);
  }

  /**
   * FIX (post-review): this previously had no branch for live mode at all —
   * clicking "Generate Board Memo" against a real backend session silently
   * did nothing. Now calls POST /scenario/narrate via apiClient.ts.
   */
  async function handleGenerateNarrative() {
    if (DEMO_MODE || sessionId === DEMO_SESSION_ID) {
      setNarrativeLoading(true);
      setNarrativeError(null);
      setTimeout(() => {
        setNarrative(getDemoNarrative());
        setNarrativeLoading(false);
        setNarrativeOpen(true);
      }, 600);
      return;
    }

    setNarrativeLoading(true);
    setNarrativeError(null);
    try {
      const { narrateScenario } = await import("@/lib/apiClient");
      const response = await narrateScenario(sessionId);
      setNarrative(response.narrative);
      setNarrativeOpen(true);
    } catch (err) {
      console.error("Narration failed:", err);
      setNarrativeError(
        err instanceof Error ? err.message : "Narration failed. Please try again."
      );
      setNarrativeOpen(true);
    } finally {
      setNarrativeLoading(false);
    }
  }

  // Derived stat-strip values — pure computation, no new state
  const consequenceCount = graph.nodes.filter((n) => n.id !== "decision_root").length;
  const domainCount = new Set(graph.nodes.map((n) => n.domain)).size;
  const speculativeCount = graph.nodes.filter((n) => n.confidence_tier === "speculative").length;
  const horizonMin = Math.min(...graph.edges.map((e) => e.time_horizon_days));
  const horizonMax = Math.max(...graph.edges.map((e) => e.time_horizon_days));

  // Domain distribution bar data
  const domainCounts = graph.nodes
    .filter((n) => n.id !== "decision_root")
    .reduce<Record<string, number>>((acc, n) => {
      acc[n.domain] = (acc[n.domain] ?? 0) + 1;
      return acc;
    }, {});
  const totalDomainNodes = Object.values(domainCounts).reduce((s, c) => s + c, 0);

  return (
    <main
      className="relative min-h-screen pb-24"
      style={{ background: "var(--color-bg)" }}
    >
      <header className="border-b px-6 py-5" style={{ borderColor: "var(--color-border)" }}>
        <Link
          href="/"
          className="mb-3 inline-flex items-center gap-1.5 font-mono text-xs no-underline"
          style={{ color: "var(--color-text-muted)" }}
        >
          ← Butterfly Effect
        </Link>

        <div className="flex items-end justify-between gap-4">
          <div className="min-w-0">
            <h1
              className="mt-1 font-display text-xl font-medium leading-snug"
              style={{ color: "var(--color-text-primary)" }}
            >
              {graph.decision_summary}
            </h1>

            {/* Stat strip — all derived from graph, no new API calls */}
            <div
              className="mt-3 flex flex-wrap gap-x-5 gap-y-1.5 font-mono text-[11px]"
              style={{ color: "var(--color-text-muted)" }}
            >
              <span>
                <span style={{ color: "var(--color-text-secondary)" }}>
                  {consequenceCount}
                </span>{" "}
                consequences
              </span>
              <span>·</span>
              <span>
                <span style={{ color: "var(--color-text-secondary)" }}>
                  {domainCount}
                </span>{" "}
                domains
              </span>
              <span>·</span>
              <span>
                <span style={{ color: "var(--color-text-secondary)" }}>
                  {horizonMin}–{horizonMax}d
                </span>{" "}
                horizon
              </span>
              <span>·</span>
              <span
                style={{
                  color: speculativeCount > 0
                    ? "var(--color-tier-speculative)"
                    : "var(--color-text-muted)",
                }}
              >
                {speculativeCount} speculative
              </span>
            </div>
          </div>

          <div className="shrink-0">
            <GenerateMemoButton
              onClick={() => {
                if (narrative) {
                  setNarrativeOpen(true);
                } else {
                  handleGenerateNarrative();
                }
              }}
            />
          </div>
        </div>
      </header>

      <div className="mx-auto flex max-w-6xl flex-col gap-4 px-6 pt-4">
        <ConfidenceLegend />

        {/* Domain distribution bar */}
        <div
          className="flex h-1.5 w-full overflow-hidden rounded-full gap-px"
          title="Domain distribution across consequences"
        >
          {Object.entries(domainCounts).map(([domain, count]) => (
            <div
              key={domain}
              title={`${domain}: ${count} node${count > 1 ? "s" : ""}`}
              style={{
                width: `${(count / totalDomainNodes) * 100}%`,
                background: `hsl(${domainHue(domain as Domain)}, 60%, 50%)`,
                opacity: 0.7,
              }}
            />
          ))}
        </div>

        <GraphControls
          graph={graph}
          api={graphApi}
          activeTiers={activeTiers}
          onTiersChange={setActiveTiers}
          maxHorizonDays={maxHorizonDays}
          onHorizonChange={setMaxHorizonDays}
        />

        <CausalGraph
          graph={graph}
          outcomes={computedOutcomes}
          activeTiers={activeTiers}
          maxHorizonDays={maxHorizonDays}
          onReady={setGraphApi}
        />

        <CalibrationMock />
      </div>

      <ParameterSlider
        graph={graph}
        overrides={overrides}
        onChange={handleSliderChange}
      />

      <NarrativePanel
        narrative={narrative}
        loading={narrativeLoading}
        error={narrativeError}
        isOpen={narrativeOpen}
        decisionSummary={graph.decision_summary}
        onGenerate={handleGenerateNarrative}
        onClose={() => setNarrativeOpen(false)}
      />
    </main>
  );
}