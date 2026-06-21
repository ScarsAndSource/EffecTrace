"use client";

import { useRouter } from "next/navigation";
import { useState, useEffect } from "react";
import DecisionInput from "@/components/DecisionInput";
import { DEMO_MODE, getDemoGenerateResult } from "@/lib/demoMode";

const EXAMPLE_DECISIONS = [
  "Raise SaaS prices 15% across all tiers next quarter",
  "Cut support headcount by 30% to hit Q3 EBITDA targets",
  "Launch a freemium tier to compete with new market entrant",
  "Mandate return-to-office 4 days a week starting January",
];

const LOADING_STAGES = [
  "Mapping direct effects…",
  "Tracing second-order consequences…",
  "Calibrating confidence tiers…",
  "Composing causal graph…",
];

export default function HomePage() {
  const router = useRouter();
  const [loading, setLoading] = useState(false);
  const [inputValue, setInputValue] = useState("");
  const [stageIndex, setStageIndex] = useState(0);
  const [error, setError] = useState<string | null>(null);

  // Cycle through loading stage copy while a request is in flight.
  useEffect(() => {
    if (!loading) {
      setStageIndex(0);
      return;
    }
    const interval = setInterval(() => {
      setStageIndex((i) => (i + 1) % LOADING_STAGES.length);
    }, 1200);
    return () => clearInterval(interval);
  }, [loading]);

  async function handleSubmit(decisionText: string) {
    setLoading(true);
    setError(null);

    if (DEMO_MODE) {
      const sessionId = `demo-${Date.now()}`;
      const result = getDemoGenerateResult();
      sessionStorage.setItem(`session_${sessionId}_graph`, JSON.stringify(result.graph));
      sessionStorage.setItem(`session_${sessionId}_outcomes`, JSON.stringify(result.outcomes));
      sessionStorage.setItem(`session_${sessionId}_decision`, decisionText);
      router.push(`/scenario/${sessionId}`);
      return;
    }

    try {
      const { generateScenario } = await import("@/lib/apiClient");
      const response = await generateScenario(decisionText);
      sessionStorage.setItem(`session_${response.session_id}_graph`, JSON.stringify(response.graph));
      sessionStorage.setItem(`session_${response.session_id}_outcomes`, JSON.stringify(response.simulation.outcomes));
      sessionStorage.setItem(`session_${response.session_id}_decision`, decisionText);
      router.push(`/scenario/${response.session_id}`);
    } catch (err) {
      console.error("Generation failed:", err);
      setError(err instanceof Error ? err.message : "Failed to generate scenario. Check that the backend is reachable.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main
      className="flex min-h-screen flex-col items-center justify-center gap-8 px-4"
      style={{ background: "var(--color-bg)" }}
    >
      <div className="text-center max-w-xl">
        <h1
          className="font-display text-4xl font-medium leading-tight sm:text-5xl"
          style={{ color: "var(--color-text-primary)" }}
        >
          EffecTrace
        </h1>
        <p className="mt-3 font-sans text-base leading-relaxed" style={{ color: "var(--color-text-secondary)" }}>
          Type one business decision. Watch a causal web of its likely cascading consequences
          bloom outward in real time — color-coded by confidence, adjustable by slider,
          narrated into a board memo on demand.
        </p>
      </div>

      <DecisionInput
        onSubmit={handleSubmit}
        loading={loading}
        value={inputValue}
        onChange={setInputValue}
      />

      {error && (
        <div
          className="w-full max-w-2xl rounded-xl border px-4 py-3 font-sans text-xs"
          style={{ borderColor: "var(--color-tier-speculative)", color: "var(--color-tier-speculative)", background: "rgba(251,113,133,0.06)" }}
        >
          {error}
        </div>
      )}

      <div className="flex max-w-2xl flex-wrap justify-center gap-2">
        {EXAMPLE_DECISIONS.map((ex) => (
          <button
            key={ex}
            type="button"
            onClick={() => setInputValue(ex)}
            className="rounded-full border px-3 py-1 font-sans text-xs transition-colors hover:border-[var(--color-decision-root)]"
            style={{ borderColor: "var(--color-border)", color: "var(--color-text-secondary)" }}
          >
            {ex.length > 45 ? ex.slice(0, 42) + "…" : ex}
          </button>
        ))}
      </div>

      {loading && (
        <div className="flex flex-col items-center gap-3">
          <div className="flex gap-2">
            <div className="h-2.5 w-2.5 animate-pulse rounded-full" style={{ background: "var(--color-decision-root)" }} />
            <div className="h-2.5 w-2.5 animate-pulse rounded-full delay-75" style={{ background: "var(--color-decision-root)" }} />
            <div className="h-2.5 w-2.5 animate-pulse rounded-full delay-150" style={{ background: "var(--color-decision-root)" }} />
          </div>
          <span className="font-sans text-xs" style={{ color: "var(--color-text-muted)" }}>
            {LOADING_STAGES[stageIndex]}
          </span>
        </div>
      )}

      <div className="mt-8 text-center">
        <span
          className="rounded-full border px-3 py-1 font-mono text-[10px] uppercase tracking-wider"
          style={{ borderColor: "var(--color-tier-data-grounded)", color: "var(--color-tier-data-grounded)" }}
        >
          {DEMO_MODE ? "Demo Mode — no live backend" : "Live mode"}
        </span>
      </div>
    </main>
  );
}