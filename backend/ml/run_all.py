"""
run_all.py — offline training/evaluation entry point.

    cd backend
    pip install -r requirements-ml.txt
    python -m ml.run_all            # writes ml/artifacts/report.json

The API only ever READS that JSON (api/routes/ml_report.py), so the heavy ML
libraries never enter the Vercel function bundle.
"""

import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import statsmodels

from . import churn_pipeline, forecast_pipeline

OUT = Path(__file__).parent / "artifacts" / "report.json"


def main() -> None:
    t0 = time.time()
    print("[ml] churn pipeline ...")
    churn = churn_pipeline.run()
    print("[ml] forecast pipeline ...")
    fc = forecast_pipeline.run()

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "runtime_s": round(time.time() - t0, 1),
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                     "scikit_learn": sklearn.__version__, "statsmodels": statsmodels.__version__},
        "churn": churn,
        "forecast": fc,
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2))
    print(f"[ml] wrote {OUT} in {report['runtime_s']}s")

    sel = churn["selected_model"]
    t = churn["models"][sel]["test"]
    print(f"[ml] churn   selected={sel}  test acc={t['accuracy']} f1={t['f1']} auc={t['roc_auc']}")
    best_key = next(iter(fc["holdout_metrics"]))
    print(f"[ml] forecast {best_key}  hold-out MAPE={fc['holdout_metrics'][best_key]['mape_pct']}%")


if __name__ == "__main__":
    main()
