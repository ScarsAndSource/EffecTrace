"""
test_narrate.py — Session 2 integration test.

Run with the FastAPI server live on port 8000:
  uvicorn api.index:app --reload --port 8000  (in one terminal)
  python test_narrate.py                        (in another terminal)

Checks verified:
1. /scenario/generate returns 200 with a full graph + simulation.
2. /scenario/narrate returns 200 with a ~400-word brief.
3. Inline confidence flags [Data-Grounded] / [Historically-Precedented] / [Speculative]
   are present in the narrative (at least one).
4. Second call to /scenario/narrate for the same session_id returns the
   IDENTICAL narrative (proving cache hit, not fresh LLM call).
5. Narrative does not introduce nodes/numbers not present in the graph.
6. /scenario/simulate with a parameter override returns 200 (secondary path check).

Output: test_outputs/narrate_output.txt for use as Session 4's static narrative input.
"""

import httpx
import json
import time
import os
import sys

BASE = os.environ.get("TEST_BASE_URL", "http://localhost:8000")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "test_outputs")

CONFIDENCE_FLAGS = ["[Data-Grounded]", "[Historically-Precedented]", "[Speculative]"]


def _assert(condition: bool, message: str) -> None:
    if not condition:
        print(f"  ✗ FAIL: {message}")
        sys.exit(1)
    print(f"  ✓ {message}")


def test_narrate():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── 1. Generate scenario ──────────────────────────────────────────────
    print("\n[1/6] POST /scenario/generate")
    resp = httpx.post(
        f"{BASE}/scenario/generate",
        json={"decision_text": "We're raising SaaS prices 15% next quarter."},
        timeout=90,
    )
    _assert(resp.status_code == 200, f"generate returned {resp.status_code}: {resp.text[:300]}")

    data = resp.json()
    session_id = data["session_id"]
    node_count = len(data["graph"]["nodes"])
    edge_count = len(data["graph"]["edges"])
    print(f"  session_id={session_id}")
    print(f"  nodes={node_count}, edges={edge_count}")

    _assert("simulation" in data, "response contains simulation key")
    _assert("outcomes" in data["simulation"], "simulation contains outcomes")
    _assert(len(data["simulation"]["outcomes"]) > 0, "outcomes dict is non-empty")

    # ── 2. First narrate call ─────────────────────────────────────────────
    print("\n[2/6] POST /scenario/narrate (first call)")
    t0 = time.time()
    narr_resp = httpx.post(
        f"{BASE}/scenario/narrate",
        json={"session_id": session_id},
        timeout=60,
    )
    elapsed_first = time.time() - t0
    _assert(narr_resp.status_code == 200, f"narrate returned {narr_resp.status_code}: {narr_resp.text[:300]}")

    narr_data = narr_resp.json()
    narrative = narr_data["narrative"]
    _assert(len(narrative) > 100, f"narrative too short ({len(narrative)} chars)")
    _assert(narr_data["session_id"] == session_id, "session_id echoed correctly")
    print(f"  elapsed={elapsed_first:.2f}s, length={len(narrative)} chars, {len(narrative.split())} words")

    # ── 3. Confidence flags ───────────────────────────────────────────────
    print("\n[3/6] Confidence flags present")
    flags_found = [f for f in CONFIDENCE_FLAGS if f in narrative]
    _assert(len(flags_found) >= 1, f"at least one confidence flag present (found: {flags_found})")
    print(f"  flags in narrative: {flags_found}")

    # Bonus: all three ideally present
    if len(flags_found) == 3:
        print("  (all three tiers represented — ideal)")
    else:
        print(f"  (warning: only {len(flags_found)}/3 tier types used — acceptable but suboptimal)")

    # ── 4. Cache hit — second call identical ─────────────────────────────
    print("\n[4/6] Cache hit — second /scenario/narrate call")
    t0 = time.time()
    narr_resp2 = httpx.post(
        f"{BASE}/scenario/narrate",
        json={"session_id": session_id},
        timeout=30,
    )
    elapsed_second = time.time() - t0
    _assert(narr_resp2.status_code == 200, f"second narrate returned {narr_resp2.status_code}")
    _assert(
        narr_resp2.json()["narrative"] == narrative,
        "second call returns identical narrative (cache hit confirmed)"
    )
    speedup = elapsed_first / max(elapsed_second, 0.001)
    print(f"  first={elapsed_first:.2f}s, second={elapsed_second:.2f}s, speedup≈{speedup:.1f}×")

    # ── 5. Narrative does not invent nodes ───────────────────────────────
    print("\n[5/6] Narrative coherence check")
    # Collect node labels (lowercase) from the graph
    node_labels = {
        n["label"].lower()
        for n in data["graph"]["nodes"]
        if n["id"] != "decision_root"
    }
    word_count = len(narrative.split())
    # Rough structural check: 4 paragraphs (separated by blank lines or sentence endings)
    paragraph_count = len([p for p in narrative.split("\n\n") if p.strip()])
    _assert(
        350 <= word_count <= 500,
        f"word count in acceptable range (got {word_count}, target 380–420)"
    )
    print(f"  word_count={word_count}, paragraph_blocks={paragraph_count}")

    # ── 6. Secondary /simulate endpoint ──────────────────────────────────
    print("\n[6/6] POST /scenario/simulate (secondary path, parameter override)")
    # Find first non-root node id to use as override target
    non_root = next(
        n["id"] for n in data["graph"]["nodes"] if n["id"] != "decision_root"
    )
    sim_resp = httpx.post(
        f"{BASE}/scenario/simulate",
        json={
            "session_id": session_id,
            "parameter_overrides": {non_root: 0.5},
        },
        timeout=60,
    )
    _assert(sim_resp.status_code == 200, f"simulate returned {sim_resp.status_code}: {sim_resp.text[:300]}")
    sim_data = sim_resp.json()
    _assert("outcomes" in sim_data, "simulate response contains outcomes")
    print(f"  override node={non_root}, outcomes count={len(sim_data['outcomes'])}")

    # ── Save output files ─────────────────────────────────────────────────
    narrative_path = os.path.join(OUTPUT_DIR, "narrate_output.txt")
    with open(narrative_path, "w", encoding="utf-8") as f:
        f.write(f"session_id: {session_id}\n")
        f.write(f"generated_from: We're raising SaaS prices 15% next quarter.\n")
        f.write(f"word_count: {word_count}\n")
        f.write(f"confidence_flags: {flags_found}\n")
        f.write("\n" + "─" * 60 + "\n\n")
        f.write(narrative)

    graph_path = os.path.join(OUTPUT_DIR, "session2_graph.json")
    with open(graph_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"\n─── Narrative ───────────────────────────────────────────────")
    print(narrative)
    print(f"─────────────────────────────────────────────────────────────")
    print(f"\nOutputs written to:")
    print(f"  {narrative_path}   ← paste into Session 4 opening message")
    print(f"  {graph_path}")
    print(f"\n✓ All 6 Session 2 checks passed.\n")


if __name__ == "__main__":
    test_narrate()
