"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";
import DecisionInput from "@/components/DecisionInput";
import { DEMO_MODE, getDemoGenerateResult } from "@/lib/demoMode";

export default function HomePage() {
  const router = useRouter();
  const [loading, setLoading] = useState(false);

  async function handleSubmit(decisionText: string) {
    setLoading(true);

    if (DEMO_MODE) {
      const result = getDemoGenerateResult();
      // Store in sessionStorage for the scenario page to pick up
      sessionStorage.setItem("demo_graph", JSON.stringify(result.graph));
      sessionStorage.setItem("demo_outcomes", JSON.stringify(result.outcomes));
      sessionStorage.setItem("demo_session_id", result.session_id);
      router.push(`/scenario/${result.session_id}`);
      return;
    }

    try {
      const { generateScenario } = await import("@/lib/apiClient");
      const response = await generateScenario(decisionText);
      sessionStorage.setItem("live_graph", JSON.stringify(response.graph));
      sessionStorage.setItem("live_outcomes", JSON.stringify(response.simulation.outcomes));
      sessionStorage.setItem("live_session_id", response.session_id);
      router.push(`/scenario/${response.session_id}`);
    } catch (err) {
      console.error("Generation failed, falling back to demo mode:", err);
      const result = getDemoGenerateResult();
      sessionStorage.setItem("demo_graph", JSON.stringify(result.graph));
      sessionStorage.setItem("demo_outcomes", JSON.stringify(result.outcomes));
      sessionStorage.setItem("demo_session_id", result.session_id);
      router.push(`/scenario/${result.session_id}`);
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
          Butterfly Effect
        </h1>
        <p className="mt-3 font-sans text-base leading-relaxed" style={{ color: "var(--color-text-secondary)" }}>
          Type one business decision. Watch a causal web of its likely cascading consequences
          bloom outward in real time \u2014 color-coded by confidence, adjustable by slider,
          narrated into a board memo on demand.
        </p>
      </div>

      <DecisionInput onSubmit={handleSubmit} loading={loading} />

      {loading && (
        <div className="flex flex-col items-center gap-3">
          <div className="flex gap-2">
            <div className="h-2.5 w-2.5 animate-pulse rounded-full" style={{ background: "var(--color-decision-root)" }} />
            <div className="h-2.5 w-2.5 animate-pulse rounded-full delay-75" style={{ background: "var(--color-decision-root)" }} />
            <div className="h-2.5 w-2.5 animate-pulse rounded-full delay-150" style={{ background: "var(--color-decision-root)" }} />
          </div>
          <span className="font-sans text-xs" style={{ color: "var(--color-text-muted)" }}>
            Mapping direct effects\u2026
          </span>
        </div>
      )}

      <div className="mt-8 text-center">
        <span
          className="rounded-full border px-3 py-1 font-mono text-[10px] uppercase tracking-wider"
          style={{ borderColor: "var(--color-tier-data-grounded)", color: "var(--color-tier-data-grounded)" }}
        >
          {DEMO_MODE ? "Demo Mode \u2014 no live backend" : "Live mode"}
        </span>
      </div>
    </main>
  );
}
