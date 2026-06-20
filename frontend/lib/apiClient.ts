/**
 * apiClient.ts — typed fetch wrappers to the FastAPI backend (Sessions 1 & 2).
 *
 * Not called anywhere yet. Session 3 renders only demoScenario.json end to end —
 * this file exists so its shape is already settled by the time Session 4 (demo
 * mode toggle) and Session 5 (live wiring) need it. Every function signature
 * matches a real route in scenario.py / narrate.py / simulate.py exactly.
 */

import type {
  GenerateRequest,
  GenerateResponse,
  NarrateRequest,
  NarrateResponse,
  SimulateRequest,
  SimulationResult,
} from "./types";

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL ?? "";

export class ApiError extends Error {
  status: number;
  body: unknown;
  constructor(message: string, status: number, body: unknown) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.body = body;
  }
}

async function postJson<TResponse>(path: string, body: unknown): Promise<TResponse> {
  if (!BACKEND_URL) {
    throw new Error(
      "NEXT_PUBLIC_BACKEND_URL is not set. Set it in .env.local before calling the live API."
    );
  }
  const res = await fetch(`${BACKEND_URL}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  let payload: unknown = null;
  try {
    payload = await res.json();
  } catch {
    // Non-JSON error body — fall through with payload = null.
  }

  if (!res.ok) {
    const message =
      payload && typeof payload === "object" && "detail" in payload
        ? JSON.stringify((payload as { detail: unknown }).detail)
        : `Request to ${path} failed with status ${res.status}`;
    throw new ApiError(message, res.status, payload);
  }

  return payload as TResponse;
}

/** POST /scenario/generate — the only network call that ever invokes the LLM. */
export function generateScenario(decisionText: string): Promise<GenerateResponse> {
  const body: GenerateRequest = { decision_text: decisionText };
  return postJson<GenerateResponse>("/scenario/generate", body);
}

/** POST /scenario/narrate — cached server-side after the first call per session. */
export function narrateScenario(
  sessionId: string,
  focusNodes?: string[]
): Promise<NarrateResponse> {
  const body: NarrateRequest = { session_id: sessionId, focus_nodes: focusNodes };
  return postJson<NarrateResponse>("/scenario/narrate", body);
}

/**
 * POST /scenario/simulate — SECONDARY/OPTIONAL endpoint, never on the demo's
 * critical path (Section 8.2). The slider calls runClientSimulation() from
 * graphPropagation.ts directly; this exists only for server-side persistence
 * of an explored parameter set or non-browser API consumers.
 */
export function simulateScenario(
  sessionId: string,
  parameterOverrides?: Record<string, number>
): Promise<SimulationResult> {
  const body: SimulateRequest = { session_id: sessionId, parameter_overrides: parameterOverrides };
  return postJson<SimulationResult>("/scenario/simulate", body);
}
