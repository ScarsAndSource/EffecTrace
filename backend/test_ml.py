"""
test_ml.py — regression + sanity checks for the ML layer. Needs only the API deps (no sklearn).

    cd backend && python test_ml.py
"""

import json
from pathlib import Path

from fastapi.testclient import TestClient

from api.index import app
from api.services.graph_service import get_dag_safe_graph, run_monte_carlo
from models.causal_graph import CausalGraphOutput

client = TestClient(app)
DEMO = json.loads((Path(__file__).parent.parent / "frontend" / "lib" / "demoScenario.json").read_text())


def check(cond: bool, msg: str) -> None:
    print(("  ✓ " if cond else "  ✗ ") + msg)
    assert cond, msg


def main() -> None:
    print("[1] existing API untouched")
    check(client.get("/health").json()["status"] == "ok", "/health ok")
    paths = app.openapi()["paths"]
    check({"/scenario/generate", "/scenario/narrate", "/scenario/simulate"} <= set(paths), "generate / narrate / simulate still mounted")

    print("[2] simulation engine unchanged on demo graph")
    graph = get_dag_safe_graph(CausalGraphOutput(**DEMO))
    out = run_monte_carlo(graph, n_samples=500)
    check(out["decision_root"]["mean"] == 1.0, "decision_root pinned at 1.0")
    check(0.10 < out["net_revenue_impact"]["mean"] < 0.25, f'net_revenue_impact mean {out["net_revenue_impact"]["mean"]:.3f} in expected band')

    print("[3] /ml/report")
    rep = client.get("/ml/report")
    check(rep.status_code == 200, "report served")
    churn, fc = rep.json()["churn"], rep.json()["forecast"]
    sel = churn["selected_model"]
    check(churn["models"][sel]["test"]["roc_auc"] > churn["models"]["dummy_majority"]["test"]["roc_auc"] + 0.3, "selected model beats majority baseline on AUC")
    check(fc["residual_diagnostics"]["ljung_box_pvalue_lag12"] > 0.05, "SARIMA residuals look like white noise")
    check(client.get("/ml/report/forecast").status_code == 200 and client.get("/ml/report/nope").status_code == 404, "section routing")

    print("[4] /ml/crosscheck")
    ok = client.post("/ml/crosscheck", json={"graph": DEMO}).json()
    check(ok["applicable"] and len(ok["llm_edges"]) >= 1, "pricing decision -> LLM churn edge compared with empirical estimate")
    other = dict(DEMO, decision_summary="Hire a new VP of Sales")
    check(client.post("/ml/crosscheck", json={"graph": other}).json()["applicable"] is False, "non-pricing decision -> not applicable")
    print("ALL PASSED")


if __name__ == "__main__":
    main()
