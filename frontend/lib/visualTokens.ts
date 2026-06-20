/**
 * visualTokens.ts — the design system's color/label vocabulary, in one place.
 * CausalNode, ConfidenceLegend, and CausalGraph all read from here so a palette
 * change never requires hunting through three files for hardcoded hex values.
 *
 * Palette logic: confidence-tier rings (the node border) and edge polarity
 * (the connecting line) are deliberately different hue families — a node's
 * evidence quality and an edge's causal direction are answering two different
 * questions, and a judge should be able to read both at a glance without the
 * colors competing for the same meaning.
 */

import type { ConfidenceTier, Domain } from "./types";

export const COLOR = {
  bg: "#0a0b0f",
  bgElevated: "#12141a",
  bgCard: "#16181f",
  border: "#262932",
  borderSubtle: "#1d1f27",
  textPrimary: "#e8e9ed",
  textSecondary: "#9498a3",
  textMuted: "#5d616e",
  decisionRoot: "#8b8ff5",

  tierDataGrounded: "#34d399",
  tierHistoricallyPrecedented: "#fbbf24",
  tierSpeculative: "#fb7185",

  polarityAmplify: "#22d3ee",
  polarityDampen: "#fb923c",
} as const;

export const TIER_COLOR: Record<ConfidenceTier, string> = {
  data_grounded: COLOR.tierDataGrounded,
  historically_precedented: COLOR.tierHistoricallyPrecedented,
  speculative: COLOR.tierSpeculative,
};

export const TIER_LABEL: Record<ConfidenceTier, string> = {
  data_grounded: "Data-grounded",
  historically_precedented: "Historically precedented",
  speculative: "Speculative",
};

export const TIER_DESCRIPTION: Record<ConfidenceTier, string> = {
  data_grounded: "Backed by peer-reviewed studies, government data, or rigorous industry benchmarks.",
  historically_precedented: "Observed in multiple documented real-world cases, not rigorously quantified.",
  speculative: "Logically sound inference with limited empirical backing.",
};

export function polarityColor(polarity: 1 | -1): string {
  return polarity === 1 ? COLOR.polarityAmplify : COLOR.polarityDampen;
}

/** Stable color per domain so the same domain always reads as the same hue across
 * a session, without needing 9 hand-picked colors — derived from a fixed hue wheel. */
const DOMAIN_ORDER: Domain[] = [
  "Revenue",
  "Operations",
  "HR",
  "Customer",
  "Market",
  "Regulatory",
  "Competitive",
  "Brand",
  "Technology",
];

export function domainHue(domain: Domain): number {
  const idx = DOMAIN_ORDER.indexOf(domain);
  const safeIdx = idx === -1 ? 0 : idx;
  return Math.round((safeIdx / DOMAIN_ORDER.length) * 360);
}
