/**
 * types.ts — Ground truth TypeScript interfaces.
 *
 * Derived from the backend's Pydantic models (models/causal_graph.py and
 * models/scenario.py). Every frontend component reads from this file.
 */

export type Layer = 0 | 1 | 2;
export type ConfidenceTier = "data_grounded" | "historically_precedented" | "speculative";
export type Direction = "positive" | "negative" | "ambiguous";
export type Domain =
  | "Revenue" | "Operations" | "HR" | "Customer" | "Market"
  | "Regulatory" | "Competitive" | "Brand" | "Technology";

export interface CausalNode {
  id: string;
  label: string;
  domain: Domain;
  layer: Layer;
  confidence_tier: ConfidenceTier;
  description: string;
  direction: Direction;
}

export interface RenderableNode extends CausalNode {
  outcome: OutcomeDistribution;
}

export interface OutcomeDistribution {
  mean: number;
  p10: number;
  p90: number;
  std: number;
}

export interface CausalEdge {
  source_id: string;
  target_id: string;
  polarity: 1 | -1;
  magnitude_estimate: number;
  time_horizon_days: number;
  rationale_citation: string;
}

export interface CausalGraphOutput {
  decision_summary: string;
  primary_domain: Domain;
  nodes: CausalNode[];
  edges: CausalEdge[];
}

export type OutcomeMap = Record<string, OutcomeDistribution>;
export type ParameterOverrides = Record<string, number>;

/** POST /scenario/generate */
export interface GenerateRequest {
  decision_text: string;
}
export interface GenerateResponse {
  session_id: string;
  graph: CausalGraphOutput;
  simulation: SimulationResult;
}

/** POST /scenario/narrate */
export interface NarrateRequest {
  session_id: string;
  focus_nodes?: string[];
}
export interface NarrateResponse {
  narrative: string;
  session_id: string;
}

/** POST /scenario/simulate */
export interface SimulateRequest {
  session_id: string;
  parameter_overrides?: ParameterOverrides;
}
export interface SimulationResult {
  outcomes: OutcomeMap;
}
