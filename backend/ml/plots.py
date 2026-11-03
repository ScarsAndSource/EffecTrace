"""
plots.py — figures for slides/viva, drawn purely from artifacts/report.json (no refitting).

    cd backend && python -m ml.plots     # -> ml/artifacts/figures/*.png
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ART = Path(__file__).parent / "artifacts"
FIG = ART / "figures"


def main() -> None:
    r = json.loads((ART / "report.json").read_text())
    c, f = r["churn"], r["forecast"]
    FIG.mkdir(exist_ok=True)

    # 1. churn EDA
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    ax[0].bar(*zip(*c["eda"]["churn_rate_by_contract"].items()), color="#c0504d")
    ax[0].set_title("Churn rate by contract")
    ax[1].bar(*zip(*c["eda"]["churn_rate_by_internet"].items()), color="#4f81bd")
    ax[1].set_title("Churn rate by internet service")
    corr = c["eda"]["numeric_corr_with_churn"]
    ax[2].barh(list(corr), list(corr.values()), color="#9bbb59")
    ax[2].set_title("Correlation with churn")
    fig.tight_layout(); fig.savefig(FIG / "churn_eda.png", dpi=140); plt.close(fig)

    # 2. ROC curves
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for m, pts in c["roc"].items():
        ax.plot(pts["fpr"], pts["tpr"], label=f'{m} (AUC {c["models"][m]["test"]["roc_auc"]})')
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="chance")
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC — held-out test set")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(FIG / "churn_roc.png", dpi=140); plt.close(fig)

    # 3. ACF / PACF of log, d=1, D=12 series
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    band = f["eda"]["acf_pacf_95_band"]
    for a, key, ttl in zip(ax, ("acf_lags_1_24", "pacf_lags_1_24"), ("ACF", "PACF")):
        a.bar(range(1, 25), f["eda"][key], color="#4f81bd")
        a.axhline(band, ls="--", c="r", lw=0.8); a.axhline(-band, ls="--", c="r", lw=0.8)
        a.set_title(f"{ttl} of log(passengers), d=1, D=1 (lag 12)")
    fig.tight_layout(); fig.savefig(FIG / "acf_pacf.png", dpi=140); plt.close(fig)

    # 4. hold-out forecast
    sp = f["series_for_plot"]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(pd.to_datetime(sp["history_dates"]), sp["history"], c="gray", label="train")
    td = pd.to_datetime(sp["test_dates"])
    ax.plot(td, sp["actual"], c="black", label="actual")
    ax.plot(td, sp["forecast"], c="#c0504d", label=f'SARIMA{tuple(f["selected"]["order"])}x{tuple(f["selected"]["seasonal_order"])}')
    ax.fill_between(td, sp["lower95"], sp["upper95"], color="#c0504d", alpha=0.2, label="95% PI")
    ax.set_title("24-month hold-out forecast"); ax.legend(); fig.tight_layout()
    fig.savefig(FIG / "forecast_holdout.png", dpi=140); plt.close(fig)
    print(f"[ml] figures -> {FIG}")


if __name__ == "__main__":
    main()
