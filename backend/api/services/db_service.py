"""
db_service.py — Supabase persistence layer.

Role in the architecture:
- Primary write: called once per successful /scenario/generate, after Redis cache write.
- Primary read: fallback source for /scenario/narrate when Redis TTL has expired.
- outcome_calibrations table: exists, empty at hackathon time — the long-term moat.

All functions are non-fatal. Generation and narration pipelines must never block
on a database failure. Every function either returns a value or None.

Schema (run against Supabase SQL editor before first /scenario/generate call):

CREATE TABLE scenarios (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  session_id TEXT NOT NULL,
  decision_text TEXT NOT NULL,
  decision_domain TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  graph_json JSONB NOT NULL,
  simulation_json JSONB,
  narrative_text TEXT
);

CREATE TABLE outcome_calibrations (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  scenario_id UUID REFERENCES scenarios(id),
  node_id TEXT NOT NULL,
  predicted_mean FLOAT,
  predicted_p10 FLOAT,
  predicted_p90 FLOAT,
  actual_outcome_value FLOAT,
  actual_outcome_note TEXT,
  calibrated_at TIMESTAMPTZ
);

CREATE INDEX idx_scenarios_session ON scenarios(session_id);
CREATE INDEX idx_calibrations_scenario ON outcome_calibrations(scenario_id);
"""

import os
from typing import Optional


def _get_client():
    """
    Lazy Supabase client. Returns None if env vars are unset — callers
    handle None gracefully rather than raising.
    """
    try:
        from supabase import create_client
        url = os.environ.get("SUPABASE_URL", "")
        key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
        if not url or not key:
            return None
        return create_client(url, key)
    except Exception as e:
        print(f"[effectrace] Supabase client init failed: {e}")
        return None


def insert_scenario(
    session_id: str,
    decision_text: str,
    graph_json: dict,
    simulation_json: dict,
    decision_domain: Optional[str] = None,
) -> Optional[str]:
    """
    Persist a generated scenario. Returns the row's UUID or None on failure.

    Called as the last step of /scenario/generate, after Redis write.
    Non-fatal: generation response is already cached; DB failure is logged only.
    """
    client = _get_client()
    if not client:
        return None

    try:
        result = (
            client.table("scenarios")
            .insert({
                "session_id": session_id,
                "decision_text": decision_text,
                "decision_domain": decision_domain,
                "graph_json": graph_json,
                "simulation_json": simulation_json,
            })
            .execute()
        )
        if result.data:
            return result.data[0]["id"]
        return None
    except Exception as e:
        print(f"[effectrace] DB insert_scenario failed (non-fatal): {e}")
        return None


def get_scenario_by_session(session_id: str) -> Optional[dict]:
    """
    Fetch scenario payload by session_id. Used by /scenario/narrate when the
    Redis cache has expired (24h TTL) but the Supabase row still exists.

    Returns the same shape as the Redis payload:
      { "graph": {...}, "simulation": {"outcomes": {...}}, "decision_text": "..." }
    so callers don't need to branch on data source.
    """
    client = _get_client()
    if not client:
        return None

    try:
        result = (
            client.table("scenarios")
            .select("session_id, decision_text, graph_json, simulation_json")
            .eq("session_id", session_id)
            .limit(1)
            .execute()
        )
        if not result.data:
            return None

        row = result.data[0]
        return {
            "graph": row["graph_json"],
            "simulation": row["simulation_json"] or {"outcomes": {}},
            "decision_text": row["decision_text"],
        }
    except Exception as e:
        print(f"[effectrace] DB get_scenario_by_session failed (non-fatal): {e}")
        return None


def update_scenario_narrative(session_id: str, narrative_text: str) -> None:
    """
    Persist the generated narrative back onto the scenarios row.
    Called after /scenario/narrate generates and caches a narrative.
    Non-fatal.
    """
    client = _get_client()
    if not client:
        return

    try:
        client.table("scenarios").update(
            {"narrative_text": narrative_text}
        ).eq("session_id", session_id).execute()
    except Exception as e:
        print(f"[effectrace] DB update_scenario_narrative failed (non-fatal): {e}")
