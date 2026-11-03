"""
churn_pipeline.py — supervised learning on IBM Telco churn + a price-shock what-if.

Flow: clean -> EDA -> stratified 80/20 split -> 5-fold CV on train (model selection)
      -> held-out test metrics -> interpretable drivers -> price-elasticity estimate
      with bootstrap CI under two model specifications.

All preprocessing (scaling, one-hot) lives INSIDE sklearn Pipelines, so it is fit on
training folds only — no leakage into CV folds or the test set.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, brier_score_loss, confusion_matrix, f1_score,
    make_scorer, precision_score, recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import load_churn

SEED = 42
PRICE_SHOCK = 0.15  # matches the "raise prices 15%" demo decision


# ── cleaning + EDA ────────────────────────────────────────────────────────────

def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    tc = pd.to_numeric(df["TotalCharges"], errors="coerce")
    # The only blanks are brand-new customers (tenure == 0) who have not been billed yet.
    assert (df.loc[tc.isna(), "tenure"] == 0).all(), "unexpected missing TotalCharges"
    df["TotalCharges"] = tc.fillna(0.0)
    df["y"] = (df.pop("Churn") == "Yes").astype(int)
    return df.drop(columns=["customerID"])


def eda(df: pd.DataFrame) -> dict:
    num = df.select_dtypes("number").drop(columns=["y"])
    return {
        "n_rows": int(len(df)),
        "n_features": int(df.shape[1] - 1),
        "churn_rate": round(float(df["y"].mean()), 4),
        "numeric_summary": {
            c: {"mean": round(float(num[c].mean()), 2), "std": round(float(num[c].std()), 2),
                "min": round(float(num[c].min()), 2), "max": round(float(num[c].max()), 2)}
            for c in ["tenure", "MonthlyCharges", "TotalCharges"]
        },
        "numeric_corr_with_churn": {c: round(float(num[c].corr(df["y"])), 3) for c in num.columns},
        "churn_rate_by_contract": df.groupby("Contract")["y"].mean().round(3).to_dict(),
        "churn_rate_by_internet": df.groupby("InternetService")["y"].mean().round(3).to_dict(),
        "mean_monthly_charges_by_churn": {
            ("churned" if k else "retained"): round(float(v), 2)
            for k, v in df.groupby("y")["MonthlyCharges"].mean().items()
        },
    }


# ── modelling ─────────────────────────────────────────────────────────────────

def preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num = X.select_dtypes("number").columns.tolist()
    cat = [c for c in X.columns if c not in num]
    return ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
    ])


def model_zoo(X: pd.DataFrame) -> dict:
    mk = lambda clf: Pipeline([("prep", preprocessor(X)), ("clf", clf)])
    return {
        "dummy_majority": mk(DummyClassifier(strategy="most_frequent")),
        "logistic_regression": mk(LogisticRegression(max_iter=1000, class_weight="balanced")),
        "random_forest": mk(RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, class_weight="balanced_subsample",
            random_state=SEED, n_jobs=-1)),
        "hist_gradient_boosting": mk(HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=200, random_state=SEED)),
    }


def test_metrics(pipe: Pipeline, X, y) -> dict:
    pred, proba = pipe.predict(X), pipe.predict_proba(X)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    return {
        "accuracy": round(accuracy_score(y, pred), 4),
        "precision": round(precision_score(y, pred, zero_division=0), 4),
        "recall": round(recall_score(y, pred, zero_division=0), 4),
        "f1": round(f1_score(y, pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y, proba), 4),
        "brier": round(brier_score_loss(y, proba), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def roc_points(pipe: Pipeline, X, y, n: int = 41) -> dict:
    fpr, tpr, _ = roc_curve(y, pipe.predict_proba(X)[:, 1])
    grid = np.linspace(0, 1, n)
    return {"fpr": grid.round(3).tolist(), "tpr": np.interp(grid, fpr, tpr).round(3).tolist()}


def top_drivers(pipe: Pipeline, k: int = 6) -> dict:
    names = pipe.named_steps["prep"].get_feature_names_out()
    coef = pipe.named_steps["clf"].coef_[0]
    order = np.argsort(coef)
    fmt = lambda i: {"feature": str(names[i]).split("__", 1)[1], "odds_ratio_per_unit": round(float(np.exp(coef[i])), 3)}
    return {
        "raises_churn": [fmt(i) for i in order[::-1][:k]],
        "lowers_churn": [fmt(i) for i in order[:k]],
        "note": "Coefficients are conditional on all other features. MonthlyCharges, TotalCharges and the "
                "service dummies are strongly collinear, so read the contract / tenure / internet-type "
                "drivers; do NOT read the charge coefficients as a price effect (see price_elasticity).",
    }


# ── price what-if ─────────────────────────────────────────────────────────────

def _shock_effect(X: pd.DataFrame, y: np.ndarray) -> tuple:
    """Fit an (unweighted -> roughly calibrated) logistic model, then scale MonthlyCharges
    by (1 + PRICE_SHOCK) for everyone and compare mean predicted churn probability."""
    pipe = Pipeline([("prep", preprocessor(X)), ("clf", LogisticRegression(max_iter=1000))]).fit(X, y)
    base = pipe.predict_proba(X)[:, 1].mean()
    shocked = X.copy()
    shocked["MonthlyCharges"] = shocked["MonthlyCharges"] * (1 + PRICE_SHOCK)
    up = pipe.predict_proba(shocked)[:, 1].mean()
    return base, up


def price_elasticity(df: pd.DataFrame, n_boot: int = 200) -> dict:
    """Relative change in churn probability per relative change in price, bootstrap CI.
    Two specs because the answer is spec-sensitive and that should be visible, not hidden.
    TotalCharges is excluded in both: it is ~ tenure x MonthlyCharges, so it cannot stay
    fixed while price moves."""
    specs = {
        "full_controls": [c for c in df.columns if c not in ("y", "TotalCharges")],
        "compact_controls": ["MonthlyCharges", "tenure", "Contract", "InternetService",
                             "PaperlessBilling", "SeniorCitizen"],
    }
    rng = np.random.default_rng(SEED)
    out = {}
    for name, cols in specs.items():
        X, y = df[cols].reset_index(drop=True), df["y"].to_numpy()
        base, up = _shock_effect(X, y)
        point = ((up - base) / base) / PRICE_SHOCK
        boots = []
        for _ in range(n_boot):
            idx = rng.integers(0, len(X), len(X))
            b, u = _shock_effect(X.iloc[idx].reset_index(drop=True), y[idx])
            boots.append(((u - b) / b) / PRICE_SHOCK)
        out[name] = {
            "baseline_churn_prob": round(float(base), 4),
            "shocked_churn_prob": round(float(up), 4),
            "abs_change_pp": round(float((up - base) * 100), 2),
            "elasticity": round(float(point), 3),
            "ci95": [round(float(np.percentile(boots, 2.5)), 3), round(float(np.percentile(boots, 97.5)), 3)],
            "boot_median": round(float(np.median(boots)), 3),
            "boot_share_positive": round(float(np.mean(np.array(boots) > 0)), 3),
            "n_boot": n_boot,
        }
    return {
        "price_shock": PRICE_SHOCK,
        "definition": "elasticity = (relative change in mean predicted churn probability) / (relative change in MonthlyCharges)",
        "by_spec": out,
        "caveats": [
            "Observational data: this is an association, not a randomised price experiment.",
            "Counterfactual scales MonthlyCharges for everyone; a 15% shock partly extrapolates the observed range.",
            "Estimate is sensitive to which controls are included (see by_spec).",
        ],
    }


# ── entry point ───────────────────────────────────────────────────────────────

def run(n_boot: int = 200) -> dict:
    df = clean(load_churn())
    report = {"dataset": "IBM Telco Customer Churn", "eda": eda(df)}

    X, y = df.drop(columns=["y"]), df["y"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    report["split"] = {"train": int(len(Xtr)), "test": int(len(Xte)), "stratified": True, "seed": SEED}

    scoring = {
        "accuracy": "accuracy",
        "precision": make_scorer(precision_score, zero_division=0),
        "recall": make_scorer(recall_score, zero_division=0),
        "f1": make_scorer(f1_score, zero_division=0),
        "roc_auc": "roc_auc",
    }
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)

    models, fitted = {}, {}
    for name, pipe in model_zoo(X).items():
        cvres = cross_validate(pipe, Xtr, ytr, cv=cv, scoring=scoring, n_jobs=1)
        fitted[name] = pipe.fit(Xtr, ytr)
        models[name] = {
            "cv5": {m: {"mean": round(float(cvres[f"test_{m}"].mean()), 4),
                        "std": round(float(cvres[f"test_{m}"].std()), 4)} for m in scoring},
            "test": test_metrics(fitted[name], Xte, yte),
        }
    report["models"] = models
    report["roc"] = {m: roc_points(fitted[m], Xte, yte) for m in fitted if m != "dummy_majority"}
    # Select on CV ROC-AUC (never on the test set).
    best = max((m for m in models if m != "dummy_majority"), key=lambda m: models[m]["cv5"]["roc_auc"]["mean"])
    report["selected_model"] = best
    report["selection_rule"] = "highest 5-fold CV ROC-AUC on the training split"
    report["top_drivers_logistic"] = top_drivers(fitted["logistic_regression"])
    report["price_elasticity"] = price_elasticity(df, n_boot=n_boot)
    return report
