"""
llm_service.py — Groq integration for causal graph generation.

Architecture decisions:
1. Structured Outputs with strict=True (response_format=json_schema) — NOT tool-calling.
   Reason: strict mode uses constrained decoding, so the model physically cannot emit
   malformed JSON. We originally used forced tool-calling, but on openai/gpt-oss-120b
   that path returned HTTP 400 `tool_use_failed` (garbled keys, unquoted strings) on
   a large share of requests. Groq does not allow tools + structured outputs together.

2. LLM is called EXACTLY ONCE per scenario (to build structure), never again during
   slider interaction (that's client-side Monte Carlo in graphPropagation.ts).

3. On validation failure: one retry with the exact error appended to the
   conversation so the model can self-correct. On a second failure: raise GraphGenerationError.
   The caller decides whether to serve Demo Mode or surface an error — not this function.

4. Langfuse tracing is optional. If keys are unset, tracing silently no-ops.
   The generation pipeline never blocks on observability.
"""

import os
import json
from typing import Optional

from groq import (
    Groq,
    BadRequestError,
    NotFoundError,
    AuthenticationError,
    PermissionDeniedError,
)
from pydantic import ValidationError

from models.causal_graph import CausalGraphOutput


# ──────────────────────────────────────────────
#  Model configuration — single source of truth
# ──────────────────────────────────────────────
#
# Groq shut down llama-3.3-70b-versatile on 2026-08-16 (free/developer tiers).
# Every request to it now returns 404 model_not_found. Groq's recommended
# replacements are openai/gpt-oss-120b and qwen/qwen3.6-27b.
#
# The model ID lives here and is overridable via the GROQ_MODEL env var, so the
# next deprecation is an env-var change in Vercel + redeploy — not a code edit.

DEFAULT_MODEL = "openai/gpt-oss-120b"


def _resolve_model(explicit: Optional[str] = None) -> str:
    """Explicit arg > GROQ_MODEL env var > DEFAULT_MODEL. Blank values are ignored."""
    return (explicit or os.environ.get("GROQ_MODEL") or DEFAULT_MODEL).strip()


def _extra_body_for(model: str) -> dict:
    """
    Model-specific request fields, passed via extra_body.

    gpt-oss models are reasoning models: reasoning tokens are generated BEFORE the
    answer and count against max_tokens. Default effort is "medium"; for a
    schema-filling task like this one "low" is enough and keeps latency down and
    the token budget free for the actual JSON.

    Sent via extra_body (not as a kwarg) because the pinned groq==0.11.0 SDK
    does not know the reasoning_effort parameter and would raise TypeError.
    Only sent to gpt-oss — other models may reject the field.
    """
    if model.startswith("openai/gpt-oss"):
        return {"reasoning_effort": os.environ.get("GROQ_REASONING_EFFORT", "low")}
    return {}


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
                    "description": "Between 10 and 17 nodes inclusive, counting decision_root.",
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
                    "description": "Between 10 and 22 edges inclusive.",
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
                                "description": "Strength of causal relationship, between 0.05 (negligible) and 0.95 (near-certain)"
                            },
                            "time_horizon_days": {
                                "type": "integer",
                                "minimum": 1,
                                "description": "Estimated days until this causal effect manifests (integer, at least 1)"
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


# Keywords that Groq's strict mode documentation does not list as supported.
# They are stripped from the schema SENT to Groq; the equivalent rules are enforced
# afterwards by Pydantic (magnitude / time_horizon) and _check_shape (counts).
_UNSUPPORTED_IN_STRICT = {"minimum", "maximum", "minItems", "maxItems"}


def _to_strict_schema(node):
    """
    Derive a strict-mode-compliant JSON schema from the tool definition:
      - every object: additionalProperties=False and ALL properties required
      - constraint keywords strict mode doesn't document are removed
    CAUSAL_GRAPH_TOOL stays the single source of truth for the structure.
    """
    if isinstance(node, list):
        return [_to_strict_schema(x) for x in node]
    if not isinstance(node, dict):
        return node
    out = {k: _to_strict_schema(v) for k, v in node.items() if k not in _UNSUPPORTED_IN_STRICT}
    if out.get("type") == "object" and isinstance(out.get("properties"), dict):
        out["additionalProperties"] = False
        out["required"] = list(out["properties"].keys())
    return out


GRAPH_JSON_SCHEMA = _to_strict_schema(CAUSAL_GRAPH_TOOL["function"]["parameters"])


def _response_format(strict: bool) -> dict:
    return {
        "type": "json_schema",
        "json_schema": {"name": "causal_graph", "strict": strict, "schema": GRAPH_JSON_SCHEMA},
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
- TOTAL node count, including decision_root, must be between 10 and 17.
- Generate between 10 and 22 edges total. Add cross-connections between same-layer nodes \
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
Challenge the obvious. A 15% price increase is not purely negative — explore both polarities.

OUTPUT FORMAT: respond with exactly one JSON object matching the provided schema. \
No prose, no markdown fences. Every string must be properly quoted and escaped.\
"""


# ──────────────────────────────────────────────
#  Typed exception
# ──────────────────────────────────────────────

class GraphGenerationError(Exception):
    """Raised when LLM generation fails after all retries."""
    pass


# ──────────────────────────────────────────────
#  Post-parse shape check
# ──────────────────────────────────────────────

MIN_NODES, MAX_NODES = 10, 17
MIN_EDGES, MAX_EDGES = 10, 22


class GraphShapeError(ValueError):
    """Graph parsed and validated, but violates node/edge count bounds."""
    pass


def _check_shape(graph: CausalGraphOutput) -> None:
    """
    Count bounds used to be enforced by minItems/maxItems in the tool schema.
    Strict mode doesn't take those keywords, so they are enforced here instead —
    a violation goes through the same one-retry self-correction path.
    """
    n, e = len(graph.nodes), len(graph.edges)
    if not (MIN_NODES <= n <= MAX_NODES):
        raise GraphShapeError(
            f"'nodes' must contain between {MIN_NODES} and {MAX_NODES} items "
            f"(including decision_root); got {n}."
        )
    if not (MIN_EDGES <= e <= MAX_EDGES):
        raise GraphShapeError(
            f"'edges' must contain between {MIN_EDGES} and {MAX_EDGES} items; got {e}."
        )


def _strip_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else t[3:]
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3]
    return t.strip()


# ──────────────────────────────────────────────
#  Main generation function
# ──────────────────────────────────────────────

def generate_causal_graph(
    decision_text: str,
    session_id: str,
    model: Optional[str] = None,
) -> CausalGraphOutput:
    """
    Call Groq once, validate, retry once on failure.
    Raises GraphGenerationError on both failures — never returns malformed data.

    Retry behaviour:
      - JSON/schema/shape failure -> retry with the exact error fed back to the model.
      - HTTP 400 (e.g. strict schema rejected) -> retry once in best-effort mode.
      - 401/403/404 -> fail immediately, retrying cannot help.
    """
    model = _resolve_model(model)
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

    # Strict (constrained decoding) by default. Set GROQ_STRICT_SCHEMA=false to start in
    # best-effort mode, e.g. when trying a model that lacks strict-mode support.
    strict = os.environ.get("GROQ_STRICT_SCHEMA", "true").strip().lower() != "false"

    last_error: Optional[str] = None

    for attempt in range(2):
        span = None
        raw_content: Optional[str] = None  # declared here so except blocks can always reference it
        try:
            if trace:
                span = trace.span(
                    name=f"groq_json_schema_attempt_{attempt + 1}",
                    input={"decision": decision_text, "attempt": attempt + 1, "strict": strict}
                )

            response = client.chat.completions.create(
                model=model,
                messages=messages,
                response_format=_response_format(strict),
                # 8192, not 4096: gpt-oss reasoning tokens count against this budget,
                # and a 17-node / 22-edge graph with citations is ~3-4k tokens on its own.
                max_tokens=8192,
                temperature=0.25,   # Low temp: consistent structure over creativity
                extra_body=_extra_body_for(model),
            )

            choice = response.choices[0]
            raw_content = choice.message.content
            if not raw_content or not raw_content.strip():
                raise GraphShapeError(
                    f"Empty response (finish_reason={choice.finish_reason!r}); "
                    "output may have been truncated by max_tokens."
                )

            parsed_json = json.loads(_strip_fences(raw_content))

            if span:
                span.end(output={
                    "node_count": len(parsed_json.get("nodes", [])),
                    "edge_count": len(parsed_json.get("edges", [])),
                })

            # Pydantic is still the choke-point: Literal enums, citation rules, clamping.
            validated = CausalGraphOutput(**parsed_json)
            _check_shape(validated)

            if trace:
                trace.update(output={
                    "nodes": len(validated.nodes),
                    "edges": len(validated.edges),
                    "primary_domain": validated.primary_domain,
                    "attempt": attempt + 1,
                })

            return validated

        except (json.JSONDecodeError, ValidationError, GraphShapeError) as e:
            last_error = str(e)
            if span:
                span.end(output={"error": last_error})

            print(f"[effectrace] Generation attempt {attempt + 1} failed validation: {last_error[:200]}")

            if attempt == 0:
                # Feed the exact error back so the model can self-correct.
                if raw_content:
                    messages.append({"role": "assistant", "content": raw_content})
                messages.append({
                    "role": "user",
                    "content": (
                        "Your previous JSON was rejected with this error:\n\n"
                        f"{last_error}\n\n"
                        "Return the complete corrected JSON object only."
                    )
                })

        except (NotFoundError, AuthenticationError, PermissionDeniedError) as e:
            # Non-retryable: wrong/retired model ID (404), bad API key (401),
            # or no access to the model (403). A second attempt cannot succeed,
            # so fail immediately instead of burning a retry and a round-trip.
            last_error = str(e)
            if span:
                span.end(output={"error": last_error})
            print(f"[effectrace] Non-retryable Groq error (model={model!r}): {last_error}")
            raise GraphGenerationError(
                f"Groq rejected the request (model={model!r}): {last_error}"
            ) from e

        except BadRequestError as e:
            # HTTP 400. Most likely cause now: strict mode rejected the schema or the
            # model. Drop to best-effort mode for the retry rather than fail outright.
            last_error = str(e)
            if span:
                span.end(output={"error": last_error})
            print(f"[effectrace] Generation attempt {attempt + 1} got 400 (strict={strict}): {last_error[:300]}")
            if strict:
                strict = False

        except Exception as e:
            # Network errors, 429s, 5xx, unexpected response shapes: plain retry.
            last_error = str(e)
            if span:
                span.end(output={"error": last_error})
            print(f"[effectrace] Generation attempt {attempt + 1} unexpected error: {e}")

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
    model: Optional[str] = None,
) -> str:
    """
    Generate a ~400-word executive brief from validated graph + simulation data.

    Does NOT use forced tool-calling — this is free-form prose, not structured JSON.
    The system prompt is the entire enforcement mechanism: inline confidence flags,
    4-paragraph structure, 380–420 word target.

    Raises NarrativeError on failure — the route handler decides whether to return
    a fallback or surface the error to the frontend.
    """
    model = _resolve_model(model)
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
            # 2048, not 1024: a ~400-word brief is ~550 tokens, but gpt-oss reasoning
            # tokens share this budget. At 1024 the model can exhaust it while
            # "thinking" and return empty content (finish_reason="length").
            max_tokens=2048,
            temperature=0.4,  # Higher than generation: consistent structure, natural prose
            extra_body=_extra_body_for(model),
        )

        narrative = response.choices[0].message.content
        if not narrative:
            finish = response.choices[0].finish_reason
            raise NarrativeError(
                f"Groq returned empty content for narration (finish_reason={finish!r}, model={model!r})"
            )

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
