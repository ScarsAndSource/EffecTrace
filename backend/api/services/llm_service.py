"""
llm_service.py — Groq integration for causal graph generation.

Architecture decisions:
1. Forced tool-calling (tool_choice=function) over bare response_format: json_object.
   Reason: tool-calling mode binds schema adherence at the inference layer for
   arrays-of-objects with multiple required fields per item. json_object mode is
   good but not bulletproof at this schema depth.

2. LLM is called EXACTLY ONCE per scenario (to build structure), never again during
   slider interaction (that's client-side Monte Carlo in graphPropagation.ts).

3. On validation failure: one retry with the exact Pydantic error appended to the
   prompt so the model can self-correct. On a second failure: raise GraphGenerationError.
   The caller decides whether to serve Demo Mode or surface an error — not this function.

4. Langfuse tracing is optional. If keys are unset, tracing silently no-ops.
   The generation pipeline never blocks on observability.
"""

import os
import json
from typing import Optional

from groq import Groq
from pydantic import ValidationError

from models.causal_graph import CausalGraphOutput


# ──────────────────────────────────────────────
#  Langfuse — optional, graceful no-op if unconfigured
# ──────────────────────────────────────────────

_langfuse = None

def _get_langfuse():
    global _langfuse
    if _langfuse is not None:
        return _langfuse
    try:
        if os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY"):
            from langfuse import Langfuse
            _langfuse = Langfuse(
                public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
                secret_key=os.environ["LANGFUSE_SECRET_KEY"],
                host=os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com"),
            )
    except Exception as e:
        print(f"[effectrace] Langfuse init skipped: {e}")
    return _langfuse


# ──────────────────────────────────────────────
#  Groq tool definition — the enforced JSON schema
# ──────────────────────────────────────────────

CAUSAL_GRAPH_TOOL = {
    "type": "function",
    "function": {
        "name": "generate_causal_graph",
        "description": (
            "Generate a directed acyclic causal graph representing the cascading "
            "consequences of a business decision. Every node is an effect; every "
            "edge is a causal relationship with evidence-graded magnitude."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "decision_summary": {
                    "type": "string",
                    "description": "One-sentence restatement of the decision in neutral language"
                },
                "primary_domain": {
                    "type": "string",
                    "enum": [
                        "Revenue", "Operations", "HR", "Customer", "Market",
                        "Regulatory", "Competitive", "Brand", "Technology"
                    ]
                },
                "nodes": {
                    "type": "array",
                    "minItems": 10,
                    "maxItems": 17,
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {
                                "type": "string",
                                "description": "Unique snake_case identifier (e.g. customer_churn_spike)"
                            },
                            "label": {
                                "type": "string",
                                "description": "Human-readable label, maximum 6 words"
                            },
                            "domain": {
                                "type": "string",
                                "enum": [
                                    "Revenue", "Operations", "HR", "Customer", "Market",
                                    "Regulatory", "Competitive", "Brand", "Technology"
                                ]
                            },
                            "layer": {
                                "type": "integer",
                                "enum": [0, 1, 2],
                                "description": "0=direct effect of decision, 1=second-order, 2=third-order"
                            },
                            "confidence_tier": {
                                "type": "string",
                                "enum": [
                                    "data_grounded",
                                    "historically_precedented",
                                    "speculative"
                                ],
                                "description": (
                                    "data_grounded: backed by peer-reviewed studies or government data. "
                                    "historically_precedented: observed in multiple documented cases. "
                                    "speculative: logically plausible but limited empirical backing."
                                )
                            },
                            "description": {
                                "type": "string",
                                "description": "1-2 sentences: what is this effect and why does it happen?"
                            },
                            "direction": {
                                "type": "string",
                                "enum": ["positive", "negative", "ambiguous"],
                                "description": "Is this effect good, bad, or unclear for the company?"
                            }
                        },
                        "required": [
                            "id", "label", "domain", "layer",
                            "confidence_tier", "description", "direction"
                        ]
                    }
                },
                "edges": {
                    "type": "array",
                    "minItems": 10,  # lowered from 12 — model naturally produces 10 for minimal valid graphs
                    "maxItems": 22,
                    "items": {
                        "type": "object",
                        "properties": {
                            "source_id": {
                                "type": "string",
                                "description": "Must exactly match an existing node id"
                            },
                            "target_id": {
                                "type": "string",
                                "description": "Must exactly match an existing node id, different from source_id"
                            },
                            "polarity": {
                                "type": "integer",
                                "enum": [1, -1],
                                "description": "1: more source causes more target. -1: more source causes less target."
                            },
                            "magnitude_estimate": {
                                "type": "number",
                                "minimum": 0.05,
                                "maximum": 0.95,
                                "description": "Strength of causal relationship: 0=negligible, 1=near-certain"
                            },
                            "time_horizon_days": {
                                "type": "integer",
                                "minimum": 1,
                                "description": "Estimated days until this causal effect manifests"
                            },
                            "rationale_citation": {
                                "type": "string",
                                "description": (
                                    "A specific real-world phenomenon, study, or historical case "
                                    "supporting this causal link. Must be concrete, not generic."
                                )
                            }
                        },
                        "required": [
                            "source_id", "target_id", "polarity",
                            "magnitude_estimate", "time_horizon_days", "rationale_citation"
                        ]
                    }
                }
            },
            "required": ["decision_summary", "primary_domain", "nodes", "edges"]
        }
    }
}


# ──────────────────────────────────────────────
#  System prompt — exact semantic rules the math layer depends on
# ──────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a causal systems analyst. You map the cascading consequences of \
business decisions with rigorous intellectual honesty.

STRUCTURAL RULES:
- Node id "decision_root" (layer 0) is ALWAYS the first node — it IS the decision.
- Generate 3–5 direct-effect nodes at layer 0 (caused directly by the decision).
- Generate 4–8 second-order nodes at layer 1 (caused by layer-0 effects).
- Generate 2–4 third-order nodes at layer 2 (caused by layer-1 effects).
- Every layer-1 and layer-2 node must have at least one incoming edge from the layer above it.
- Do NOT create edges that skip layers (decision_root → layer-2 directly is forbidden).
- Generate AT LEAST 10 edges total. Add cross-connections between same-layer nodes \
where causally justified (e.g. churn → support_ticket_volume, revenue_gain → cost_cutting_pressure). \
These lateral edges are required to reach the minimum count and improve graph richness.

CONFIDENCE TIER RULES — assign based on evidence, never plausibility alone:
- data_grounded: backed by peer-reviewed studies, government statistics, SEC filings, \
or rigorous industry datasets. Price elasticity studies, labor economics research, \
consumer psychology experiments qualify.
- historically_precedented: observed in multiple documented real-world corporate cases \
but not rigorously quantified. Company announcements, analyst reports, industry case studies qualify.
- speculative: logically sound inference with limited empirical backing. \
Reasonable extrapolation from adjacent domains.

EDGE POLARITY:
- polarity 1: more of the source causes MORE of the target (amplifying).
- polarity -1: more of the source causes LESS of the target (dampening).
Example: customer_churn → support_ticket_volume has polarity 1 (more churn triggers more tickets).
Example: revenue_increase → cost_cutting_pressure has polarity -1 (more revenue reduces pressure to cut).

DIRECTION field: business valence of the effect, not the causal direction.
- positive: good for the company's long-term health.
- negative: bad for the company's long-term health.
- ambiguous: genuinely unclear or highly context-dependent.

CITATION RULE: rationale_citation must name a specific real phenomenon, study, or \
historical case. "Numerous studies show..." is REJECTED. \
"ProfitWell 2021 SaaS pricing elasticity benchmark: 1% price increase yields ~0.5% churn" \
is ACCEPTED.

Include effects that are counterintuitive but documented. \
Challenge the obvious. A 15% price increase is not purely negative — explore both polarities.\
"""


# ──────────────────────────────────────────────
#  Typed exception
# ──────────────────────────────────────────────

class GraphGenerationError(Exception):
    """Raised when LLM generation fails after all retries."""
    pass


# ──────────────────────────────────────────────
#  Main generation function
# ──────────────────────────────────────────────

def generate_causal_graph(
    decision_text: str,
    session_id: str,
    model: str = "llama-3.3-70b-versatile",
) -> CausalGraphOutput:
    """
    Call Groq once, validate, retry once on schema failure.
    Raises GraphGenerationError on both failures — never returns malformed data.
    """
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    lf = _get_langfuse()

    trace = lf.trace(name="generate_causal_graph", session_id=session_id) if lf else None

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Business decision to analyse:\n\n{decision_text.strip()}"
        }
    ]

    last_error: Optional[str] = None

    for attempt in range(2):
        span = None
        raw_args: Optional[str] = None  # declared here so except blocks can always reference it
        try:
            if trace:
                span = trace.span(
                    name=f"groq_tool_call_attempt_{attempt + 1}",
                    input={"decision": decision_text, "attempt": attempt + 1}
                )

            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=[CAUSAL_GRAPH_TOOL],
                tool_choice={
                    "type": "function",
                    "function": {"name": "generate_causal_graph"}
                },
                max_tokens=4096,
                temperature=0.25,   # Low temp: consistent structure over creativity
            )

            tool_calls = response.choices[0].message.tool_calls
            if not tool_calls:
                raise GraphGenerationError(
                    f"Groq returned no tool call on attempt {attempt + 1}. "
                    "Model may not support forced tool_choice — verify model name."
                )

            raw_args = tool_calls[0].function.arguments
            parsed_json = json.loads(raw_args)

            if span:
                span.end(output={
                    "node_count": len(parsed_json.get("nodes", [])),
                    "edge_count": len(parsed_json.get("edges", [])),
                })

            # Validate against Pydantic models — this is where Literal enums enforce
            validated = CausalGraphOutput(**parsed_json)

            if trace:
                trace.update(output={
                    "nodes": len(validated.nodes),
                    "edges": len(validated.edges),
                    "primary_domain": validated.primary_domain,
                    "attempt": attempt + 1,
                })

            return validated

        except (json.JSONDecodeError, ValidationError) as e:
            last_error = str(e)
            if span:
                span.end(output={"error": last_error})

            print(f"[effectrace] Generation attempt {attempt + 1} failed validation: {last_error[:200]}")

            if attempt == 0:
                # Inject the exact validation error so the model can self-correct
                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "retry_correction",
                            "type": "function",
                            "function": {
                                "name": "generate_causal_graph",
                                "arguments": raw_args or "{}"
                            }
                        }
                    ]
                })
                messages.append({
                    "role": "tool",
                    "tool_call_id": "retry_correction",
                    "content": (
                        f"Validation failed with the following error. Fix it and call "
                        f"generate_causal_graph again:\n\n{last_error}"
                    )
                })

        except Exception as e:
            last_error = str(e)
            if span:
                span.end(output={"error": last_error})

            print(f"[effectrace] Generation attempt {attempt + 1} unexpected error: {e}")

            if attempt == 0:
                # Inject the error so the model knows what to fix on retry.
                # This covers Groq HTTP 400 schema rejections (e.g. minItems violations)
                # which never reach the ValidationError branch above because Groq
                # rejects them server-side before returning a response to Python.
                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "retry_correction",
                            "type": "function",
                            "function": {
                                "name": "generate_causal_graph",
                                "arguments": raw_args or "{}"
                            }
                        }
                    ]
                })
                messages.append({
                    "role": "tool",
                    "tool_call_id": "retry_correction",
                    "content": (
                        f"Your previous call was rejected with the following error. "
                        f"Fix it and call generate_causal_graph again:\n\n{last_error}"
                    )
                })

    raise GraphGenerationError(
        f"Graph generation failed after 2 attempts. Last error: {last_error}"
    )


# ──────────────────────────────────────────────
#  Narration — typed exception
# ──────────────────────────────────────────────

class NarrativeError(Exception):
    """Raised when narrative generation fails."""
    pass


# ──────────────────────────────────────────────
#  Narration — system prompt
# ──────────────────────────────────────────────

NARRATION_SYSTEM_PROMPT = """\
You are a senior strategy analyst writing an executive decision brief for a board audience.

You will receive a structured causal analysis: a list of effects (nodes) with confidence \
tiers and simulation outcome ranges, and the key causal links between them.

YOUR RULES — violating any of these disqualifies the output:
1. Narrate ONLY what appears in the structured context. Do not invent new nodes, \
   percentages, causal links, or company names not present in the input.
2. Every substantive claim must carry its evidence tier inline — use exactly these labels:
   [Data-Grounded]  [Historically-Precedented]  [Speculative]
3. Structure your response as four paragraphs:
   Paragraph 1 — The decision and its primary stakes (2–3 sentences).
   Paragraph 2 — Direct Effects: what happens first and why.
   Paragraph 3 — Second and Third-Order Effects: what propagates downstream.
   Paragraph 4 — Recommendation: proceed / pause / monitor, with the single most \
                 important risk condition named explicitly.
4. Target 380–420 words total. Do not pad. Do not add headers or bullets.
5. Tone: direct, analytically honest, not marketing. An exec will forward this to a board.
6. Outcome ranges (mean / p10 / p90) are directional signals, not precise forecasts. \
   Present them as such: "our simulation indicates a directional negative effect" — \
   never "will definitely" or "certainly."
7. Speculative claims require the [Speculative] tag even if the effect seems obvious. \
   This honesty is the product's trust layer.
"""


# ──────────────────────────────────────────────
#  Narration — context builder
# ──────────────────────────────────────────────

def _build_narrative_context(
    graph_data: "CausalGraphOutput",
    simulation_outcomes: dict,
    focus_nodes: Optional[list] = None,
) -> str:
    """
    Convert graph + simulation dict into a structured text block for the narration prompt.

    Selects nodes in layer order, confidence order (data_grounded first).
    Includes simulation outcome ranges for every node.
    Lists top-8 edges by magnitude with their rationale citations.
    """
    TIER_ORDER = {"data_grounded": 0, "historically_precedented": 1, "speculative": 2}
    TIER_LABEL = {
        "data_grounded": "[Data-Grounded]",
        "historically_precedented": "[Historically-Precedented]",
        "speculative": "[Speculative]",
    }
    LAYER_LABEL = {0: "Direct Effect", 1: "Second-Order", 2: "Third-Order"}
    DIR_LABEL = {"positive": "↑ beneficial", "negative": "↓ harmful", "ambiguous": "~ ambiguous"}

    lines = [
        f"DECISION: {graph_data.decision_summary}",
        f"PRIMARY DOMAIN: {graph_data.primary_domain}",
        "",
        "CAUSAL NODES (by layer, then evidence strength):",
    ]

    sorted_nodes = sorted(
        [n for n in graph_data.nodes if n.id != "decision_root"],
        key=lambda n: (n.layer, TIER_ORDER.get(n.confidence_tier, 3)),
    )

    node_map = {n.id: n for n in graph_data.nodes}

    for node in sorted_nodes:
        if focus_nodes and node.id not in focus_nodes:
            continue

        sim = simulation_outcomes.get(node.id, {})
        mean = sim.get("mean", 0.0)
        p10 = sim.get("p10", 0.0)
        p90 = sim.get("p90", 0.0)

        tier_tag = TIER_LABEL.get(node.confidence_tier, "[Speculative]")
        layer_tag = LAYER_LABEL.get(node.layer, "Effect")
        dir_tag = DIR_LABEL.get(node.direction, "~ ambiguous")

        lines.append(
            f"• [{layer_tag}] {tier_tag} {node.label} "
            f"(domain: {node.domain}, valence: {dir_tag})"
        )
        lines.append(
            f"  Simulation: mean={mean:+.3f}, p10={p10:+.3f}, p90={p90:+.3f}"
        )
        lines.append(f"  {node.description}")

    # Top 8 edges by magnitude for the key causal links section
    top_edges = sorted(
        graph_data.edges,
        key=lambda e: e.magnitude_estimate,
        reverse=True,
    )[:8]

    lines.extend(["", "KEY CAUSAL LINKS (highest magnitude):"])
    for edge in top_edges:
        src = node_map.get(edge.source_id)
        tgt = node_map.get(edge.target_id)
        if not src or not tgt:
            continue
        polarity_str = "amplifies" if edge.polarity == 1 else "dampens"
        lines.append(
            f"• {src.label} {polarity_str} {tgt.label} "
            f"(magnitude={edge.magnitude_estimate:.2f}, ~{edge.time_horizon_days}d) — "
            f"{edge.rationale_citation}"
        )

    return "\n".join(lines)


# ──────────────────────────────────────────────
#  Narration — main function
# ──────────────────────────────────────────────

def generate_narrative(
    graph_data: "CausalGraphOutput",
    simulation_outcomes: dict,
    session_id: str,
    focus_nodes: Optional[list] = None,
    model: str = "llama-3.3-70b-versatile",
) -> str:
    """
    Generate a ~400-word executive brief from validated graph + simulation data.

    Does NOT use forced tool-calling — this is free-form prose, not structured JSON.
    The system prompt is the entire enforcement mechanism: inline confidence flags,
    4-paragraph structure, 380–420 word target.

    Raises NarrativeError on failure — the route handler decides whether to return
    a fallback or surface the error to the frontend.
    """
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    lf = _get_langfuse()

    trace = lf.trace(name="generate_narrative", session_id=session_id) if lf else None

    context_text = _build_narrative_context(graph_data, simulation_outcomes, focus_nodes)

    try:
        span = trace.span(name="groq_narration", input={"session_id": session_id}) if trace else None

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": NARRATION_SYSTEM_PROMPT},
                {"role": "user", "content": context_text},
            ],
            max_tokens=1024,
            temperature=0.4,  # Higher than generation: consistent structure, natural prose
        )

        narrative = response.choices[0].message.content
        if not narrative:
            raise NarrativeError("Groq returned empty content for narration")

        narrative = narrative.strip()

        if span:
            span.end(output={
                "word_count": len(narrative.split()),
                "has_data_grounded_tag": "[Data-Grounded]" in narrative,
                "has_speculative_tag": "[Speculative]" in narrative,
            })

        if trace:
            trace.update(output={
                "word_count": len(narrative.split()),
                "char_count": len(narrative),
            })

        return narrative

    except NarrativeError:
        raise
    except Exception as e:
        raise NarrativeError(f"Narrative generation failed: {e}") from e