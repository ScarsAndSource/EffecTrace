/**
 * demoMode.ts — the demo-mode flag and offline data loaders (Section 8.2 / Section 17).
 *
 * When active, this reads demoScenario.json (Session 3's literal Session 1 output)
 * and a hardcoded copy of Session 2's real narration text — zero fetch calls, works
 * in airplane mode. This is the PRIMARY scripted walkthrough, not an emergency
 * fallback (Section 15, Issue #9): the live path through apiClient.ts is the bonus
 * "and it's not faked" beat performed only after this has already landed.
 *
 * ⚠️ PLACEHOLDER — DEMO_NARRATIVE below is NOT yet real.
 * Replace it with the literal contents of backend/test_outputs/narrate_output.txt
 * (produced by test_narrate.py) before this goes anywhere near a judge. Demo Mode's
 * entire credibility rests on this being captured output, not a paraphrase — same
 * rule that governs demoScenario.json itself. A fourth guard, in the spirit of
 * safe_variance/clamp/ensure_dag: don't let placeholder content reach the stage.
 */

import demoScenarioRaw from "./demoScenario.json";
import { runClientSimulation } from "./graphPropagation";
import type { CausalGraphOutput, OutcomeMap } from "./types";

export const DEMO_SESSION_ID = "demo-saas-pricing-15pct";

/** True when there's no backend configured, or the flag is forced on explicitly.
 * Mirrors apiClient.ts's own BACKEND_URL check so the two files can never disagree
 * about whether a live call is even possible. */
export const DEMO_MODE: boolean =
  process.env.NEXT_PUBLIC_DEMO_MODE === "true" || !process.env.NEXT_PUBLIC_BACKEND_URL;

const DEMO_NARRATIVE_PLACEHOLDER_MARKER = "__PASTE_SESSION_2_NARRATE_OUTPUT_HERE__";

// TODO(pre-demo): paste the literal contents of backend/test_outputs/narrate_output.txt
// here, verbatim, including its inline [Data-Grounded] / [Historically-Precedented] /
// [Speculative] tags. Do not paraphrase, trim, or "clean up" the wording.
const DEMO_NARRATIVE = `${DEMO_NARRATIVE_PLACEHOLDER_MARKER}`;

if (process.env.NODE_ENV !== "production" && DEMO_NARRATIVE.includes(DEMO_NARRATIVE_PLACEHOLDER_MARKER)) {
  // eslint-disable-next-line no-console
  console.warn(
    "[demoMode] DEMO_NARRATIVE is still a placeholder. Paste the real Session 2 " +
      "narrate_output.txt content into frontend/lib/demoMode.ts before demo day."
  );
}

export function getDemoScenario(): CausalGraphOutput {
  return demoScenarioRaw as unknown as CausalGraphOutput;
}

export interface DemoGenerateResult {
  session_id: string;
  graph: CausalGraphOutput;
  outcomes: OutcomeMap;
}

/**
 * Equivalent to a /scenario/generate response, computed entirely client-side.
 * Demo Mode's simulation isn't faked or pre-baked separately — it's the exact
 * same runClientSimulation() the slider uses (Section 10.4), just run once at
 * load with zero overrides. Moving a slider in Demo Mode re-runs this same
 * function with overrides, identically to the live path.
 */
export function getDemoGenerateResult(): DemoGenerateResult {
  const graph = getDemoScenario();
  const outcomes = runClientSimulation(graph, {});
  return { session_id: DEMO_SESSION_ID, graph, outcomes };
}

/** Returns the hardcoded narrative instead of calling /scenario/narrate. */
export function getDemoNarrative(): string {
  return DEMO_NARRATIVE;
}
