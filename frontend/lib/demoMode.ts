/**
 * demoMode.ts — the demo-mode flag and offline data loaders (Section 8.2 / Section 17).
 *
 * When active, this reads demoScenario.json and a narration text consistent
 * with it — zero fetch calls, works in airplane mode. This is the PRIMARY
 * scripted walkthrough, not an emergency fallback (Section 15, Issue #9).
 *
 * NOTE: DEMO_NARRATIVE below is hand-authored from demoScenario.json's actual
 * nodes/edges (no invented numbers or claims), NOT a literal Groq capture —
 * because no live /scenario/narrate call has been run yet. Once you run
 * test_generate.py + a live /scenario/narrate call with a real GROQ_API_KEY,
 * replace this constant with that literal output, same rule that governs
 * demoScenario.json itself: ground truth should be a capture, not a paraphrase.
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

// TODO(pre-demo): replace with the literal contents of a real
// POST /scenario/narrate response once test_generate.py + a live narrate
// call have been run. Do not paraphrase, trim, or "clean up" that wording
// when you do.
const DEMO_NARRATIVE = `EffecTrace's simulation maps the consequences of raising SaaS subscription prices 15% across all tiers next quarter. The decision is structurally two-sided: it raises revenue per customer immediately, but trades that gain off against retention, sales velocity, and competitive exposure. The real question is not whether the increase generates more revenue per account, since it does by definition, but whether downstream churn, competitive response, and market-share erosion eat into that gain over the following two to three quarters.

[Data-Grounded] Revenue per existing customer rises roughly in line with the price increase for every account that renews. [Historically-Precedented] That gain is offset by a churn spike concentrated in the price-sensitive SMB cohort, consistent with documented SaaS pricing-elasticity patterns. [Historically-Precedented] Sales cycles for new business lengthen as the higher price point pulls in additional procurement review. [Speculative] There is a smaller, less certain upside too: the price increase may shift brand perception upmarket, signaling quality to enterprise buyers who previously read the lower price as a capability gap.

[Data-Grounded] Net revenue impact, ARPU gain minus churn-driven loss, lands positive in the simulation, and that positive net revenue flows through to [Data-Grounded] margin expansion given SaaS's largely fixed cost base. [Historically-Precedented] The churn spike also reshapes support ticket volume and produces a measurable NPS dip the quarter immediately following the change, and [Historically-Precedented] competitors are likely to respond with promotions targeted at the churning segment, which over two to three quarters produces a [Historically-Precedented] modest erosion in SMB market share. [Speculative] Further out, sustained margin improvement could support more competitive hiring and, more speculatively, a modest expansion in valuation multiple if the market reads the increase as evidence of pricing power, though this last effect carries the least empirical backing of anything in the graph.

Proceed, with monitoring rather than pausing: the simulation's directional signal favors a net-positive outcome, anchored by data-grounded ARPU and margin effects that outweigh the smaller, historically-precedented churn and market-share costs. The one condition worth watching closely is the customer satisfaction dip. If NPS decline compounds into churn beyond what is assumed here, the net revenue and margin gains this recommendation rests on would narrow faster than the simulation currently projects.`;

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

/** Returns the demo narrative instead of calling /scenario/narrate. */
export function getDemoNarrative(): string {
  return DEMO_NARRATIVE;
}