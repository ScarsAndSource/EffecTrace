"""
graph_service.py — The math layer. LLM output enters here as validated structure;
Monte Carlo simulation exits here as outcome distributions.

Three guards live here that were bugs in the original spec:

1. safe_variance() — uses .get() with a default so an unexpected confidence_tier
   value never throws KeyError. The Literal enum in causal_graph.py prevents
   this at parse time, but this is a defence-in-depth fallback.

2. clamp() — re-asserts magnitude bounds before beta sampling. The Pydantic
   model clamps at parse time; this makes the math layer self-contained.

3. ensure_dag() — enforces the DAG invariant BEFORE nx.topological_sort() is
   called. Without this, a feedback-loop edge from the LLM crashes the entire
   /scenario/generate endpoint with NetworkXUnfeasible. With it, the weakest
   cycle edge is silently removed and the topological sort always succeeds.

FIX (post-review): get_dag_safe_graph() was added because run_monte_carlo()
only ever pruned cycle/dangling edges from its own internal nx.DiGraph copy —
it never wrote that pruning back onto the CausalGraphOutput object that gets
returned to the frontend. That meant the browser could receive an edge list
containing a cycle the backend had already (correctly) worked around for its
own simulation, and graphPropagation.ts's Kahn's-algorithm sort would then
silently drop every node trapped in that cycle to a flat zero outcome the
moment a slider moved — diverging from the number shown a second earlier.
scenario.py now calls get_dag_safe_graph() BEFORE returning or caching a
graph, so the edge set the browser ever sees has already had cycles removed.

Simulation design:
- 500 samples (balance between variance estimate quality and Vercel function time)
- Beta distribution parameterised via method-of-moments: α = mean/variance, β = (1-mean)/variance
- Variance set by the source node's confidence_tier (tighter = higher confidence)
- tanh squashing bounds all node values to [-1, 1]
- p10/p90 computed from sorted sample array (no scipy dependency)
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Optional

from models.causal_graph import CausalGraphOutput, CausalNode, CausalEdge


# ──────────────────────────────────────────────
#  Tier → sampling variance mapping
# ──────────────────────────────────────────────

TIER_VARIANCE: Dict[str, float] = {
    "data_grounded": 0.05,           # Tight: strong empirical backing
    "historically_precedented": 0.15, # Medium: pattern documented, not quantified
    "speculative": 0.30,              # Loose: inference from adjacent evidence
}
TIER_VARIANCE_DEFAULT = 0.20  # Used when confidence_tier is somehow unrecognised


def safe_variance(tier: str) -> float:
    """
    Safe lookup with default. Literal enum already prevents bad values at parse
    time, but this guard ensures the math layer is independently robust.
    """
    return TIER_VARIANCE.get(tier, TIER_VARIANCE_DEFAULT)


def clamp(value: float, lo: float = 0.02, hi: float = 0.98) -> float:
    """
    Clamp magnitude to (lo, hi). Prevents Beta(0, x) or Beta(x, 0) which
    numpy raises ValueError for. Model-level validator already does this;
    this re-assertion makes graph_service self-contained.
    """
    return max(lo, min(hi, value))


# ──────────────────────────────────────────────
#  DAG enforcement
# ──────────────────────────────────────────────

def ensure_dag(G: nx.DiGraph, max_iterations: int = 30) -> nx.DiGraph:
    """
    Remove the lowest-magnitude edge from each detected cycle until the graph
    is acyclic. Feedback loops are real causal behaviour, but nx.topological_sort
    throws NetworkXUnfeasible on any cycle. This resolves the conflict by
    sacrificing the weakest cycle edges.

    max_iterations prevents infinite loops on pathological graphs.
    """
    removed = []
    for _ in range(max_iterations):
        if nx.is_directed_acyclic_graph(G):
            break
        try:
            cycle = nx.find_cycle(G, orientation="original")
        except nx.NetworkXNoCycle:
            break

        # Find the weakest edge in the cycle by magnitude_estimate
        weakest_edge = min(
            cycle,
            key=lambda e: G.edges[e[0], e[1]].get("magnitude_estimate", 0.0)
        )
        u, v = weakest_edge[0], weakest_edge[1]
        removed.append((u, v, G.edges[u, v].get("magnitude_estimate", 0.0)))
        G.remove_edge(u, v)

    if removed:
        print(
            f"[effectrace] ensure_dag removed {len(removed)} cycle edge(s): "
            + ", ".join(f"{u}->{v} (mag={m:.2f})" for u, v, m in removed)
        )

    return G


# ──────────────────────────────────────────────
#  Graph builder
# ──────────────────────────────────────────────

def build_graph(graph_data: CausalGraphOutput) -> nx.DiGraph:
    """
    Convert validated CausalGraphOutput to a networkx DiGraph.
    Silently drops edges whose source_id or target_id doesn't match a node —
    the LLM occasionally invents node references that don't exist.
    """
    G = nx.DiGraph()

    node_ids = {n.id for n in graph_data.nodes}
    for node in graph_data.nodes:
        G.add_node(node.id, **node.model_dump())

    dropped = 0
    for edge in graph_data.edges:
        if edge.source_id not in node_ids or edge.target_id not in node_ids:
            dropped += 1
            continue
        if edge.source_id == edge.target_id:
            dropped += 1
            continue
        G.add_edge(edge.source_id, edge.target_id, **edge.model_dump())

    if dropped:
        print(f"[effectrace] build_graph: dropped {dropped} dangling/self-loop edge(s)")

    return G


# ──────────────────────────────────────────────
#  Pruned-graph export — FIX: keep the response in sync with the simulation
# ──────────────────────────────────────────────

def get_dag_safe_graph(graph_data: CausalGraphOutput) -> CausalGraphOutput:
    """
    Returns a NEW CausalGraphOutput whose edges exactly match what build_graph()
    + ensure_dag() would actually simulate on — i.e. with cycle edges and any
    dangling/self-loop edges already removed.

    Call this once, right after generate_causal_graph() returns, BEFORE the
    graph is cached, persisted, or sent to the frontend. Every downstream
    consumer (the initial response, Redis, Supabase, /scenario/narrate,
    /scenario/simulate, and graphPropagation.ts on the client) then shares
    one consistent edge set — there is no longer a backend-only "secretly
    cleaned" copy that the frontend never sees.
    """
    G = build_graph(graph_data)
    G = ensure_dag(G)
    surviving = {(u, v) for u, v in G.edges()}
    pruned_edges = [e for e in graph_data.edges if (e.source_id, e.target_id) in surviving]
    return graph_data.model_copy(update={"edges": pruned_edges})


# ──────────────────────────────────────────────
#  Monte Carlo simulation
# ──────────────────────────────────────────────

def run_monte_carlo(
    graph_data: CausalGraphOutput,
    n_samples: int = 500,
    parameter_overrides: Optional[Dict[str, float]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Propagate uncertainty through the causal DAG via Monte Carlo simulation.

    For each sample:
    1. Sample each edge's magnitude from Beta(α, β) parameterised by confidence tier.
    2. Propagate node values in topological order: value = tanh(Σ predecessor_value × edge_effect).
    3. tanh bounds all values to [-1, 1].

    parameter_overrides: {node_id: scale_factor}
    Overriding node X means: multiply every edge where source_id == X by scale_factor.
    This is the slider contract — agreed upon before a line of frontend code is written.

    Still independently calls build_graph()/ensure_dag() itself (rather than
    trusting the caller pre-pruned), so this function stays safe to call on
    its own from simulate.py or anywhere else — get_dag_safe_graph() is what
    keeps the RESPONSE edge set in sync, this is what keeps the MATH safe.

    Returns: {node_id: {mean, p10, p90, std}}
    """
    G = build_graph(graph_data)
    G = ensure_dag(G)

    node_ids: List[str] = list(G.nodes())
    outcomes: Dict[str, List[float]] = {nid: [] for nid in node_ids}

    # Pre-compute topological order (only once — it's deterministic on a fixed DAG)
    topo_order: List[str] = list(nx.topological_sort(G))

    # Build adjacency structures once for the inner loop
    predecessors: Dict[str, List[str]] = {
        nid: list(G.predecessors(nid)) for nid in node_ids
    }

    rng = np.random.default_rng()  # Use new-style RNG for reproducibility if seeded

    for _ in range(n_samples):
        # ── Step 1: sample each edge's effective magnitude ──
        sampled_effects: Dict[tuple, float] = {}

        for u, v, data in G.edges(data=True):
            source_tier = G.nodes[u].get("confidence_tier", "speculative")
            tier_var = safe_variance(source_tier)
            mag = clamp(data.get("magnitude_estimate", 0.5))

            # Apply parameter override: scale outgoing edges from the overridden node
            if parameter_overrides and u in parameter_overrides:
                mag = clamp(mag * parameter_overrides[u])

            # Method-of-moments Beta parameterisation
            alpha = mag / tier_var
            beta_p = (1.0 - mag) / tier_var

            # Extra safety: numpy raises on α≤0 or β≤0 even with clamp
            alpha = max(0.1, alpha)
            beta_p = max(0.1, beta_p)

            polarity = data.get("polarity", 1)
            sampled_mag = rng.beta(alpha, beta_p)
            sampled_effects[(u, v)] = sampled_mag * polarity

        # ── Step 2: propagate in topological order ──
        node_values: Dict[str, float] = {}

        for node_id in topo_order:
            if node_id == "decision_root":
                node_values[node_id] = 1.0
                continue

            preds = predecessors.get(node_id, [])
            if not preds:
                # Isolated node (shouldn't exist after validation, but handle it)
                node_values[node_id] = 0.0
                continue

            effect = sum(
                node_values.get(pred, 0.0) * sampled_effects.get((pred, node_id), 0.0)
                for pred in preds
            )

            # tanh squash: bounds output to (-1, 1), prevents explosion across layers
            node_values[node_id] = float(np.tanh(effect))

        for nid, val in node_values.items():
            if nid in outcomes:
                outcomes[nid].append(val)

    # ── Step 3: compute summary statistics ──
    result: Dict[str, Dict[str, float]] = {}

    for nid, vals in outcomes.items():
        if not vals:
            result[nid] = {"mean": 0.0, "p10": 0.0, "p90": 0.0, "std": 0.0}
            continue

        arr = np.array(vals, dtype=np.float64)
        result[nid] = {
            "mean": float(np.mean(arr)),
            "p10": float(np.percentile(arr, 10)),
            "p90": float(np.percentile(arr, 90)),
            "std": float(np.std(arr)),
        }

    return result