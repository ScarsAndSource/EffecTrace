"""
causal_graph.py — The schema contract for everything downstream.

All field types use Literal constraints. A value that doesn't match
gets rejected here, before it can propagate into the math layer.
This is the single choke-point between LLM output and simulation.
"""

from typing import Literal, List
from pydantic import BaseModel, Field, field_validator


# ──────────────────────────────────────────────
#  Node model
# ──────────────────────────────────────────────

Domain = Literal[
    "Revenue", "Operations", "HR", "Customer", "Market",
    "Regulatory", "Competitive", "Brand", "Technology"
]

ConfidenceTier = Literal[
    "data_grounded",
    "historically_precedented",
    "speculative"
]

Direction = Literal["positive", "negative", "ambiguous"]


class CausalNode(BaseModel):
    id: str
    label: str
    domain: Domain
    layer: Literal[0, 1, 2]
    confidence_tier: ConfidenceTier
    description: str
    direction: Direction

    @field_validator("label")
    @classmethod
    def label_max_six_words(cls, v: str) -> str:
        words = v.strip().split()
        if len(words) > 8:
            # Trim silently rather than reject — layout breaks on long labels
            return " ".join(words[:6])
        return v.strip()

    @field_validator("id")
    @classmethod
    def id_must_be_snake_case(cls, v: str) -> str:
        # Normalise: lowercase, spaces → underscores
        return v.strip().lower().replace(" ", "_").replace("-", "_")


# ──────────────────────────────────────────────
#  Edge model
# ──────────────────────────────────────────────

class CausalEdge(BaseModel):
    source_id: str
    target_id: str
    polarity: Literal[1, -1]
    magnitude_estimate: float = Field(ge=0.0, le=1.0)
    time_horizon_days: int = Field(gt=0)
    rationale_citation: str

    @field_validator("magnitude_estimate")
    @classmethod
    def clamp_magnitude(cls, v: float) -> float:
        """
        Clamp to (0.02, 0.98) before ANY downstream code sees the value.

        Why: np.random.beta(alpha, beta) crashes with ValueError when either
        parameter is 0. With the parameterisation alpha = mean/variance and
        beta = (1-mean)/variance, a boundary value (0.0 or 1.0) makes one
        parameter exactly 0. The clamp makes this impossible.
        This guard lives in the model, not the simulation, so the math layer
        never needs its own boundary check.
        """
        return max(0.02, min(0.98, v))

    @field_validator("source_id", "target_id")
    @classmethod
    def normalise_id(cls, v: str) -> str:
        return v.strip().lower().replace(" ", "_").replace("-", "_")

    @field_validator("rationale_citation")
    @classmethod
    def citation_must_be_specific(cls, v: str) -> str:
        """
        Reject vague citations. The LLM is instructed to cite real phenomena;
        this validator enforces that the instruction was followed.
        """
        BANNED_GENERICS = [
            "studies show",
            "research indicates",
            "it is well known",
            "generally accepted",
            "common knowledge",
            "experts agree",
        ]
        cleaned = v.strip()
        if len(cleaned) < 25:
            raise ValueError(
                f"rationale_citation too short to be specific ({len(cleaned)} chars): '{cleaned}'"
            )
        lower = cleaned.lower()
        for phrase in BANNED_GENERICS:
            if phrase in lower:
                raise ValueError(
                    f"rationale_citation uses banned generic phrase '{phrase}': '{cleaned}'"
                )
        return cleaned


# ──────────────────────────────────────────────
#  Root graph output
# ──────────────────────────────────────────────

class CausalGraphOutput(BaseModel):
    decision_summary: str
    primary_domain: Domain
    nodes: List[CausalNode]
    edges: List[CausalEdge]

    @field_validator("nodes")
    @classmethod
    def must_contain_decision_root(cls, nodes: List[CausalNode]) -> List[CausalNode]:
        ids = {n.id for n in nodes}
        if "decision_root" not in ids:
            raise ValueError(
                "Graph is missing 'decision_root' node. "
                "Every causal graph must anchor on the decision itself."
            )
        return nodes

    @field_validator("edges")
    @classmethod
    def edges_must_have_valid_endpoints(cls, edges: List[CausalEdge]) -> List[CausalEdge]:
        # We can't cross-validate against nodes here (field order).
        # Cross-validation happens in graph_service.build_graph().
        return edges
