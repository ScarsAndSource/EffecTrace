/**
 * graphPropagation.ts — Client-side Monte Carlo simulation engine.
 *
 * This is the PRIMARY simulation path for the slider (Section 8.2 / 8.4).
 * The Python backend's run_monte_carlo() is the reference implementation;
 * this file is a 1:1 TypeScript port.
 *
 * Override contract (Section 10.5):
 *   Overriding node X means: scale every edge where source_id == X
 *   by the override factor, then re-propagate the full graph.
 */

import type {
  CausalGraphOutput,
  CausalNode,
  CausalEdge,
  OutcomeMap,
  ParameterOverrides,
} from "./types";

// ──────────────────────────────────────────────
//  Tier variance mapping (mirrors Python's TIER_VARIANCE)
// ──────────────────────────────────────────────

const TIER_VARIANCE: Record<string, number> = {
  data_grounded: 0.05,
  historically_precedented: 0.15,
  speculative: 0.30,
};
const TIER_VARIANCE_DEFAULT = 0.20;

function safeVariance(tier: string): number {
  return TIER_VARIANCE[tier] ?? TIER_VARIANCE_DEFAULT;
}

function clamp(value: number, lo = 0.02, hi = 0.98): number {
  return Math.max(lo, Math.min(hi, value));
}

// ──────────────────────────────────────────────
//  Random number generation for Beta distribution
// ──────────────────────────────────────────────

function gaussianRandom(): number {
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

function sampleGamma(shape: number): number {
  if (shape >= 1) {
    const d = shape - 1 / 3;
    const c = 1 / Math.sqrt(9 * d);
    while (true) {
      let x: number, v: number;
      do {
        x = gaussianRandom();
        v = 1 + c * x;
      } while (v <= 0);
      v = v * v * v;
      const u = Math.random();
      if (u < 1 - 0.0331 * (x * x) * (x * x)) return d * v;
      if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v;
    }
  } else {
    return sampleGamma(shape + 1) * Math.pow(Math.random(), 1 / shape);
  }
}

function sampleBeta(alpha: number, beta: number): number {
  const x = sampleGamma(alpha);
  const y = sampleGamma(beta);
  return x / (x + y);
}

// ──────────────────────────────────────────────
//  Topological sort (Kahn's algorithm) for small DAGs
// ──────────────────────────────────────────────

function topologicalSort(
  nodeIds: string[],
  adjacency: Map<string, string[]>,
): string[] {
  const inDegree = new Map<string, number>();
  for (const id of nodeIds) inDegree.set(id, 0);
  for (const [, targets] of adjacency) {
    for (const t of targets) {
      inDegree.set(t, (inDegree.get(t) ?? 0) + 1);
    }
  }

  const queue: string[] = [];
  for (const [id, deg] of inDegree) {
    if (deg === 0) queue.push(id);
  }

  const order: string[] = [];
  while (queue.length > 0) {
    const cur = queue.shift()!;
    order.push(cur);
    for (const next of adjacency.get(cur) ?? []) {
      const newDeg = (inDegree.get(next) ?? 1) - 1;
      inDegree.set(next, newDeg);
      if (newDeg === 0) queue.push(next);
    }
  }

  return order;
}

// ──────────────────────────────────────────────
//  Main simulation function
// ──────────────────────────────────────────────

export function runClientSimulation(
  graph: CausalGraphOutput,
  overrides: ParameterOverrides,
  nSamples = 500,
): OutcomeMap {
  // Build adjacency from edges
  const nodeSet = new Map<string, CausalNode>();
  for (const n of graph.nodes) nodeSet.set(n.id, n);

  const outgoing = new Map<string, { target: string; edge: CausalEdge }[]>();
  const incoming = new Map<string, string[]>();
  const decisionRoot = graph.nodes.find((n) => n.id === "decision_root");

  for (const e of graph.edges) {
    if (!nodeSet.has(e.source_id) || !nodeSet.has(e.target_id)) continue;
    if (e.source_id === e.target_id) continue;

    if (!outgoing.has(e.source_id)) outgoing.set(e.source_id, []);
    outgoing.get(e.source_id)!.push({ target: e.target_id, edge: e });

    if (!incoming.has(e.target_id)) incoming.set(e.target_id, []);
    incoming.get(e.target_id)!.push(e.source_id);
  }

  const allIds = graph.nodes.map((n) => n.id);
  const adj = new Map<string, string[]>();
  for (const [src, targets] of outgoing) {
    adj.set(src, targets.map((t) => t.target));
  }
  for (const n of allIds) {
    if (!adj.has(n)) adj.set(n, []);
  }

  const topoOrder = topologicalSort(allIds, adj);

  // Monte Carlo loop
  const samples = new Map<string, number[]>();
  for (const id of allIds) samples.set(id, []);

  for (let iter = 0; iter < nSamples; iter++) {
    // Sample each edge's magnitude
    const sampledEffects = new Map<string, number>();
    for (const [src, edges] of outgoing) {
      for (const { target: tgt, edge } of edges) {
        const sourceNode = nodeSet.get(src);
        const sourceTier = sourceNode?.confidence_tier ?? "speculative";
        const tierVar = safeVariance(sourceTier);
        let mag = clamp(edge.magnitude_estimate);

        // Apply override: scale outgoing edges from overridden node
        if (overrides[src] !== undefined) {
          mag = clamp(mag * overrides[src]);
        }

        const alpha = Math.max(0.1, mag / tierVar);
        const betaParam = Math.max(0.1, (1.0 - mag) / tierVar);

        const sampledMag = sampleBeta(alpha, betaParam);
        const key = `${src}->${tgt}`;
        sampledEffects.set(key, sampledMag * edge.polarity);
      }
    }

    // Propagate in topological order
    const nodeValues = new Map<string, number>();
    for (const nodeId of topoOrder) {
      if (nodeId === "decision_root") {
        nodeValues.set(nodeId, 1.0);
        continue;
      }

      const preds = incoming.get(nodeId) ?? [];
      if (preds.length === 0) {
        nodeValues.set(nodeId, 0.0);
        continue;
      }

      let effect = 0;
      for (const pred of preds) {
        const predVal = nodeValues.get(pred) ?? 0;
        const edgeEffect = sampledEffects.get(`${pred}->${nodeId}`) ?? 0;
        effect += predVal * edgeEffect;
      }

      const squashed = Math.tanh(effect);
      nodeValues.set(nodeId, squashed);
    }

    for (const [id, val] of nodeValues) {
      samples.get(id)?.push(val);
    }
  }

  // Compute summary statistics
  const result: OutcomeMap = {};
  for (const [id, vals] of samples) {
    if (vals.length === 0) {
      result[id] = { mean: 0, p10: 0, p90: 0, std: 0 };
      continue;
    }

    const sorted = [...vals].sort((a, b) => a - b);
    const sum = vals.reduce((a, b) => a + b, 0);
    const mean = sum / vals.length;
    const variance = vals.reduce((acc, v) => acc + (v - mean) ** 2, 0) / vals.length;
    const std = Math.sqrt(variance);
    const p10 = sorted[Math.floor(sorted.length * 0.1)];
    const p90 = sorted[Math.floor(sorted.length * 0.9)];

    result[id] = { mean, p10, p90, std };
  }

  return result;
}

/**
 * topDirectEffectsByMagnitude — Returns the top N direct-effect nodes
 * (layer 0, excluding decision_root) sorted by their incoming edge magnitude.
 *
 * Used by ParameterSlider to decide which 3 sliders to show.
 */
export function topDirectEffectsByMagnitude(
  graph: CausalGraphOutput,
  n: number,
): { node_id: string; magnitude: number }[] {
  const directEdges = graph.edges.filter(
    (e) => e.source_id === "decision_root",
  );

  const scored = directEdges.map((e) => ({
    node_id: e.target_id,
    magnitude: e.magnitude_estimate,
  }));

  // Deduplicate by node_id, keep highest magnitude per node
  const best = new Map<string, number>();
  for (const { node_id, magnitude } of scored) {
    const existing = best.get(node_id) ?? 0;
    if (magnitude > existing) best.set(node_id, magnitude);
  }

  return Array.from(best.entries())
    .map(([node_id, magnitude]) => ({ node_id, magnitude }))
    .sort((a, b) => b.magnitude - a.magnitude)
    .slice(0, n);
}
