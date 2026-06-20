"""
test_generate.py — Live test script for POST /scenario/generate.

Run AFTER starting the server:
    uvicorn api.index:app --reload --port 8000

Then, in a second terminal:
    python test_generate.py

Writes each raw response to backend/test_outputs/<name>.json and runs the
checks Session 1 was supposed to gate on: confidence_tier values are exactly
one of the three allowed strings, every magnitude_estimate is inside
[0.02, 0.98], decision_root is present, and the nonsense input still returns
valid structure instead of a crash.

Pick the cleanest output file afterward and paste it VERBATIM into
frontend/lib/demoScenario.json — that's the real ground-truth capture step.
"""

import json
import os
import re
import sys
import httpx

BASE_URL = os.environ.get("EFFECTRACE_TEST_URL", "http://localhost:8000")
OUT_DIR = os.path.join(os.path.dirname(__file__), "test_outputs")

DECISIONS = [
    ("price_increase", "We're raising SaaS prices 15% next quarter across all tiers."),
    ("hire_vp", "Hire a new VP of Sales."),
    ("compound_decision",
     "We're laying off 20% of engineering, freezing all non-critical hiring, "
     "and redirecting the saved budget into a new AI features team launching in Q3."),
    ("feedback_loop_bait", "Cut the marketing budget in half to fund engineering headcount."),
    ("nonsense_input", "asdkjf qwoeiur 12903 !!!! purple monday recursion bacon"),
]

VALID_TIERS = {"data_grounded", "historically_precedented", "speculative"}


def slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    client = httpx.Client(timeout=60.0)

    results = []
    for name, decision_text in DECISIONS:
        print(f"\n→ {name}: {decision_text[:60]}...")
        try:
            resp = client.post(f"{BASE_URL}/scenario/generate", json={"decision_text": decision_text})
        except httpx.ConnectError:
            print(f"  ✗ Could not connect to {BASE_URL}. Is uvicorn running?")
            sys.exit(1)

        out_path = os.path.join(OUT_DIR, f"{slugify(name)}.json")

        if resp.status_code != 200:
            print(f"  ✗ HTTP {resp.status_code}: {resp.text[:300]}")
            with open(out_path, "w") as f:
                json.dump({"status_code": resp.status_code, "body": resp.text}, f, indent=2)
            results.append((name, False, resp.status_code))
            continue

        data = resp.json()
        with open(out_path, "w") as f:
            json.dump(data, f, indent=2)

        graph = data["graph"]
        node_ids = {n["id"] for n in graph["nodes"]}
        bad_tiers = {n["confidence_tier"] for n in graph["nodes"]} - VALID_TIERS
        bad_mags = [
            e["magnitude_estimate"] for e in graph["edges"]
            if not (0.02 <= e["magnitude_estimate"] <= 0.98)
        ]

        print(f"  ✓ {len(graph['nodes'])} nodes, {len(graph['edges'])} edges")
        print(f"    decision_root present: {'decision_root' in node_ids}")
        print(f"    bad confidence_tier values: {bad_tiers or 'none'}")
        print(f"    magnitude_estimate outside [0.02, 0.98]: {bad_mags or 'none'}")
        print(f"    saved → {out_path}")

        results.append((name, True, 200))

    print("\n" + "=" * 50)
    print("SUMMARY")
    for name, ok, status in results:
        print(f"  {'✓' if ok else '✗'} {name} ({status})")
    print(f"\nOutput files in: {OUT_DIR}")
    print("Pick the cleanest one and paste it verbatim into frontend/lib/demoScenario.json")


if __name__ == "__main__":
    main()