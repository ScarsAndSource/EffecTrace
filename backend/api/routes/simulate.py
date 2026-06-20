"""
simulate.py — POST /scenario/simulate

ARCHITECTURE NOTE — read before modifying:
This endpoint is SECONDARY and OPTIONAL. It is NOT on the demo's critical path.

The slider interaction in the frontend runs entirely client-side via graphPropagation.ts
(TypeScript Monte Carlo port). This endpoint exists only for:
  - Optional server-side persistence of a user's explored parameter set.
  - API consumers who want a re-simulation without building a browser client.

Override contract (matches graphPropagation.ts exactly):
  Overriding node X means: multiply every edge where source_id == X
  by the override factor, then re-propagate the full graph from decision_root forward.

This contract is written down here, in graphPropagation.ts, and in the spec's Section 10.5.
Both halves must implement the same rule or the slider's client/server outputs diverge.
"""

from fastapi import APIRouter, HTTPException

from models.scenario import SimulateRequest, SimulationResult, OutcomeDistribution
from models.causal_graph import CausalGraphOutput
from api.services.cache_service import get_cached_scenario
from api.services.graph_service import run_monte_carlo

router = APIRouter()


@router.post("/simulate", response_model=SimulationResult)
async def simulate_scenario(request: SimulateRequest) -> SimulationResult:
    # Fetch from Redis only — this endpoint does not fall back to Supabase.
    # If the session has expired from cache, the caller should re-generate.
    payload = get_cached_scenario(request.session_id)

    if not payload:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "session_not_in_cache",
                "message": (
                    f"Session '{request.session_id}' not found in cache. "
                    "Either the session expired (24h TTL) or was never generated. "
                    "Call POST /scenario/generate first."
                ),
            },
        )

    # Reconstruct typed graph — same path as narrate.py
    try:
        graph = CausalGraphOutput(**payload["graph"])
    except Exception as e:
        raise HTTPException(
            status_code=422,
            detail={"error": "cached_graph_invalid", "message": str(e)},
        )

    # Re-run Monte Carlo with parameter overrides
    # parameter_overrides: {node_id: scale_factor} — applied to outgoing edges
    overrides = request.parameter_overrides if request.parameter_overrides else None

    raw_outcomes = run_monte_carlo(
        graph_data=graph,
        n_samples=500,
        parameter_overrides=overrides,
    )

    outcomes = {
        node_id: OutcomeDistribution(**dist)
        for node_id, dist in raw_outcomes.items()
    }

    return SimulationResult(outcomes=outcomes)
