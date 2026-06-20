"""
cache_service.py — Upstash Redis wrapper.

After the architectural correction: Redis is NOT on the slider's critical path
(slider runs client-side). Redis is used for:
  1. Narration generation (fetch graph without hitting Supabase)
  2. Cross-device or page-refresh session persistence

All functions fail silently — the generation pipeline must not block on cache.
"""

import os
import json
from typing import Optional


def _get_redis():
    try:
        from upstash_redis import Redis
        url = os.environ.get("UPSTASH_REDIS_REST_URL", "")
        token = os.environ.get("UPSTASH_REDIS_REST_TOKEN", "")
        if not url or not token:
            return None
        return Redis(url=url, token=token)
    except Exception as e:
        print(f"[effectrace] Redis client init failed: {e}")
        return None


def cache_scenario(session_id: str, payload: dict, ttl: int = 86400) -> None:
    """Store full scenario payload (graph + simulation) for 24h."""
    r = _get_redis()
    if not r:
        return
    try:
        r.setex(f"scenario:{session_id}", ttl, json.dumps(payload))
    except Exception as e:
        print(f"[effectrace] Cache write failed (non-fatal): {e}")


def get_cached_scenario(session_id: str) -> Optional[dict]:
    r = _get_redis()
    if not r:
        return None
    try:
        raw = r.get(f"scenario:{session_id}")
        return json.loads(raw) if raw else None
    except Exception as e:
        print(f"[effectrace] Cache read failed (non-fatal): {e}")
        return None


def cache_narrative(session_id: str, narrative: str, ttl: int = 86400) -> None:
    r = _get_redis()
    if not r:
        return
    try:
        r.setex(f"narrative:{session_id}", ttl, narrative)
    except Exception as e:
        print(f"[effectrace] Narrative cache write failed (non-fatal): {e}")


def get_cached_narrative(session_id: str) -> Optional[str]:
    r = _get_redis()
    if not r:
        return None
    try:
        raw = r.get(f"narrative:{session_id}")
        return raw if isinstance(raw, str) else None
    except Exception as e:
        print(f"[effectrace] Narrative cache read failed (non-fatal): {e}")
        return None
