"""
ml_report.py — READ-ONLY endpoints over the offline ML artifacts.

    GET  /ml/report                 full report.json
    GET  /ml/report/{section}       "churn" | "forecast"
    POST /ml/crosscheck             LLM edge claim vs. empirical price-elasticity estimate

No sklearn / pandas / statsmodels are imported here: the models are trained offline by
`python -m ml.run_all` and this file only serves the JSON they wrote. If the artifact is
missing the endpoints return 503; nothing else in the API is affected.
"""

import json
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from models.causal_graph import CausalGraphOutput

router = APIRouter()

REPORT_PATH = Path(__file__).resolve().parents[2] / "ml" / "artifacts" / "report.json"
PRICE_WORDS = ("price", "pricing", "subscription", "tariff", "fee")


def _load_report() -> dict:
    if not REPORT_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail={"error": "ml_report_missing",
                    "message": "Run `python -m ml.run_all` in backend/ to generate ml/artifacts/report.json."},
        )
    return json.loads(REPORT_PATH.read_text())


@router.get("/report")
async def full_report() -> dict:
    return _load_report()


@router.get("/report/{section}")
async def report_section(section: str) -> dict:
    report = _load_report()
    if section not in ("churn", "forecast"):
        raise HTTPException(status_code=404, detail={"error": "unknown_section", "valid": ["churn", "forecast"]})
    return report[section]


class CrossCheckRequest(BaseModel):
    graph: CausalGraphOutput
    decision_text: Optional[str] = None


@router.post("/crosscheck")
async def crosscheck(req: CrossCheckRequest) -> dict:
    """Show the LLM's decision->churn edge(s) next to what a churn model fitted on real
    customer data implies for a 15% price shock. Only meaningful for pricing decisions."""
    text = f"{req.decision_text or ''} {req.graph.decision_summary}".lower()
    if not any(w in text for w in PRICE_WORDS):
        return {"applicable": False,
                "reason": "Empirical churn model only covers price-change decisions."}

    edges = [
        {"target_id": e.target_id, "polarity": e.polarity, "magnitude_estimate": e.magnitude_estimate,
         "time_horizon_days": e.time_horizon_days, "llm_citation": e.rationale_citation}
        for e in req.graph.edges
        if e.source_id == "decision_root" and "churn" in e.target_id
    ]
    pe = _load_report()["churn"]["price_elasticity"]
    return {
        "applicable": True,
        "llm_edges": edges,
        "empirical": {
            "dataset": "IBM Telco Customer Churn (n=7,043)",
            "price_shock": pe["price_shock"],
            "definition": pe["definition"],
            "by_spec": pe["by_spec"],
            "caveats": pe["caveats"],
        },
        "reading": "Compare the elasticity quoted in llm_citation with empirical.by_spec[*].ci95. "
                   "If the cited figure lies outside both intervals, treat that edge's 'data_grounded' "
                   "tier as unverified.",
    }
