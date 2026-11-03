"""
index.py — FastAPI application entry point.

Deployed to Vercel as a Python runtime function (see vercel.json).
All routes are mounted with their prefix here; route files contain only logic.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes.scenario import router as scenario_router
from api.routes.narrate import router as narrate_router
from api.routes.simulate import router as simulate_router

app = FastAPI(
    title="EffecTrace API",
    version="1.0.0",
    description="Causal consequence simulation engine. LLM builds structure; math does prediction.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],            # Restrict to your Vercel domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(scenario_router, prefix="/scenario", tags=["scenario"])
app.include_router(narrate_router, prefix="/scenario", tags=["scenario"])
app.include_router(simulate_router, prefix="/scenario", tags=["scenario"])

# Offline-trained ML evidence layer (read-only). Wrapped so it can never take the API down.
try:
    from api.routes.ml_report import router as ml_router
    app.include_router(ml_router, prefix="/ml", tags=["ml"])
except Exception as e:  # pragma: no cover
    print(f"[effectrace] ML report router not loaded: {e}")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "effectrace",
        "version": "1.0.0",
    }
