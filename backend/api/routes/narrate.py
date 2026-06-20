"""
narrate.py — POST /scenario/narrate

Execution sequence:
1. Check narrative cache (Redis) — return immediately on hit, no LLM call.
2. Fetch scenario payload: Redis first, Supabase fallback (handles expired TTL).
3. Reconstruct CausalGraphOutput from cached dict (no re-generation).
4. Build structured context from nodes + simulation outcomes.
5. Call Groq for ~400-word executive brief with inline confidence tags.
6. Cache result in Redis (narrative:{session_id}, 24h TTL).
7. Persist narrative text back to Supabase scenarios row (non-fatal).

The LLM is never asked to invent — it narrates only what the simulation produced.
"""

from fastapi import APIRouter, HTTPException

from models.scenario import NarrateRequest, NarrateResponse
from models.causal_graph import CausalGraphOutput
from api.services.cache_service import (
    get_cached_scenario,
    get_cached_narrative,
    cache_narrative,
)
from api.services.db_service import get_scenario_by_session, update_scenario_narrative
from api.services.llm_service import generate_narrative, NarrativeError

router = APIRouter()


@router.post("/narrate", response_model=NarrateResponse)
async def narrate_scenario(request: NarrateRequest) -> NarrateResponse:
    session_id = request.session_id

    # ── 1. Narrative cache hit ────────────────────────────────────────────
    cached = get_cached_narrative(session_id)
    if cached:
        return NarrateResponse(narrative=cached, session_id=session_id)

    # ── 2. Fetch scenario payload ─────────────────────────────────────────
    # Redis first (TTL = 24h, same window as narrative generation).
    # Supabase fallback covers the rare case where Redis TTL expired but the
    # row still exists (e.g. user returns after 25+ hours).
    payload = get_cached_scenario(session_id)

    if not payload:
        payload = get_scenario_by_session(session_id)

    if not payload:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "session_not_found",
                "message": (
                    f"No scenario found for session_id='{session_id}'. "
                    "Generate a scenario first via POST /scenario/generate."
                ),
            },
        )

    # ── 3. Reconstruct typed graph from cached dict ───────────────────────
    try:
        graph = CausalGraphOutput(**payload["graph"])
    except Exception as e:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "cached_graph_invalid",
                "message": f"Cached graph failed validation: {e}",
            },
        )

    simulation_outcomes: dict = payload.get("simulation", {}).get("outcomes", {})

    # ── 4 + 5. Generate narrative ─────────────────────────────────────────
    try:
        narrative = generate_narrative(
            graph_data=graph,
            simulation_outcomes=simulation_outcomes,
            session_id=session_id,
            focus_nodes=request.focus_nodes,
        )
    except NarrativeError as e:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "narrative_generation_failed",
                "message": str(e),
            },
        )

    # ── 6. Cache narrative ────────────────────────────────────────────────
    cache_narrative(session_id, narrative)

    # ── 7. Persist to DB (non-fatal) ──────────────────────────────────────
    update_scenario_narrative(session_id, narrative)

    return NarrateResponse(narrative=narrative, session_id=session_id)
