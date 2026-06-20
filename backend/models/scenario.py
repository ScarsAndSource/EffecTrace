"""
scenario.py — Pydantic models for all API request/response shapes.

Every route in routes/scenario.py, routes/narrate.py, and routes/simulate.py
reads from this file. No route file defines its own request/response model.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field

from models.causal_graph import CausalGraphOutput


# ──────────────────────────────────────────────
#  Shared
# ──────────────────────────────────────────────

class OutcomeDistribution(BaseModel):
    mean: float
    p10: float
    p90: float
    std: float


class SimulationResult(BaseModel):
    outcomes: Dict[str, OutcomeDistribution]


# ──────────────────────────────────────────────
#  POST /scenario/generate
# ──────────────────────────────────────────────

class GenerateRequest(BaseModel):
    decision_text: str = Field(
        min_length=1,
        max_length=1000,
        description="A business decision in plain English, e.g. 'We're raising SaaS prices 15% next quarter.'",
    )


class GenerateResponse(BaseModel):
    session_id: str
    graph: CausalGraphOutput
    simulation: SimulationResult


# ──────────────────────────────────────────────
#  POST /scenario/narrate
# ──────────────────────────────────────────────

class NarrateRequest(BaseModel):
    session_id: str
    focus_nodes: Optional[List[str]] = None


class NarrateResponse(BaseModel):
    narrative: str
    session_id: str


# ──────────────────────────────────────────────
#  POST /scenario/simulate
# ──────────────────────────────────────────────

class SimulateRequest(BaseModel):
    session_id: str
    parameter_overrides: Optional[Dict[str, float]] = None


# SimulateResponse is the same shape as SimulationResult
