"""
scenario.py — POST /scenario/generate

Execution sequence:
1. Call Groq (forced tool-calling) → validated CausalGraphOutput
2. get_dag_safe_graph() → prune cycle/dangling edges BEFORE this graph goes
   anywhere else (response, cache, DB) — keeps every downstream consumer,
   including the frontend's client-side slider math, looking at the exact
   same edge set that was actually simulated.
3. run_monte_carlo() on the pruned graph → SimulationResult
4. Cache payload in Upstash Redis (non-fatal if Redis is down)
5. Return full response — frontend has everything it needs to render immediately

On LLM failure after retries: returns 503 with explicit message.
The frontend's Demo Mode catches this and falls back gracefully.
"""

import uuid
from fastapi import APIRouter, HTTPException

from models.scenario import (
    GenerateRequest,
    GenerateResponse,
    SimulationResult,
    OutcomeDistribution,
)
from api.services.llm_service import generate_causal_graph, GraphGenerationError
from api.services.graph_service import run_monte_carlo, get_dag_safe_graph
from api.services.cache_service import cache_scenario
from api.services.db_service import insert_scenario

router = APIRouter()


@router.post("/generate", response_model=GenerateResponse)
async def generate_scenario(request: GenerateRequest) -> GenerateResponse:
    session_id = str(uuid.uuid4())

    # ── 1. LLM graph generation ──────────────────────────────────────────
    try:
        graph = generate_causal_graph(
            decision_text=request.decision_text,
            session_id=session_id,
        )
    except GraphGenerationError as e:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "graph_generation_failed",
                "message": str(e),
                "fallback_hint": "Load demo mode from /lib/demoScenario.json",
            }
        )

    # ── 2. Prune to a DAG-safe graph BEFORE it goes anywhere else ─────────
    graph = get_dag_safe_graph(graph)

    # ── 3. Monte Carlo simulation (on the same pruned graph) ──────────────
    raw_outcomes = run_monte_carlo(
        graph_data=graph,
        n_samples=500,
        parameter_overrides=None,
    )

    outcomes = {
        node_id: OutcomeDistribution(**dist)
        for node_id, dist in raw_outcomes.items()
    }
    simulation = SimulationResult(outcomes=outcomes)

    # ── 4. Cache + persist (both non-fatal) ──────────────────────────────
    payload = {
        "graph": graph.model_dump(),
        "simulation": {
            "outcomes": {k: v.model_dump() for k, v in outcomes.items()}
        },
        "decision_text": request.decision_text,
    }
    cache_scenario(session_id, payload)

    # DB write: non-fatal, after Redis write so response never blocks on Supabase
    insert_scenario(
        session_id=session_id,
        decision_text=request.decision_text,
        graph_json=payload["graph"],
        simulation_json=payload["simulation"],
        decision_domain=graph.primary_domain,
    )

    # ── 5. Return ────────────────────────────────────────────────────────
    return GenerateResponse(
        session_id=session_id,
        graph=graph,
        simulation=simulation,
    )