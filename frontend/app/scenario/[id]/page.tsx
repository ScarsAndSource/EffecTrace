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

  function handleGenerateNarrative() {
    if (DEMO_MODE || sessionId === DEMO_SESSION_ID) {
      setNarrativeLoading(true);
      setTimeout(() => {
        setNarrative(getDemoNarrative());
        setNarrativeLoading(false);
        setNarrativeOpen(true);
      }, 600);
      return;
    }
  }

  return (
    <main
      className="relative min-h-screen pb-24"
      style={{ background: "var(--color-bg)" }}
    >
      <header className="flex items-center justify-between px-6 py-4">
        <Link
          href="/"
          className="font-display text-lg font-medium no-underline"
          style={{ color: "var(--color-text-primary)" }}
        >
          Butterfly Effect
        </Link>
        <span className="font-sans text-xs" style={{ color: "var(--color-text-muted)" }}>
          {graph.decision_summary}
        </span>
      </header>

      <div className="mx-auto flex max-w-6xl flex-col gap-4 px-6">
        <ConfidenceLegend />

        <GraphControls
          graph={graph}
          api={graphApi}
          activeTiers={activeTiers}
          onTiersChange={setActiveTiers}
          maxHorizonDays={maxHorizonDays}
          onHorizonChange={setMaxHorizonDays}
        />

        <div className="flex items-center justify-between">
          <div />
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
