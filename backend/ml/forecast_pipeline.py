"""
forecast_pipeline.py — classical time-series forecasting (SARIMA) with honest baselines.

Series: monthly airline passengers 1949-01..1960-12 (n=144), the standard SARIMA benchmark.
Role in EffecTrace: a validated *baseline-trajectory* forecaster — what a KPI does
without the decision. It is evaluated on its own; it is not wired into the causal graph math.

Flow: stationarity tests (ADF) -> log + differencing -> ACF/PACF -> AIC grid search on
TRAIN only -> 24-month hold-out vs naive / seasonal-naive / Holt-Winters -> residual
diagnostics (Ljung-Box) -> interval coverage -> rolling-origin check.
"""

import itertools
import warnings

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import acf, adfuller, pacf

from .data import load_air_passengers

HOLDOUT = 24
SEASON = 12


def _metrics(actual: pd.Series, fc: np.ndarray, train: pd.Series) -> dict:
    a, f = actual.to_numpy(), np.asarray(fc)
    mase_scale = np.mean(np.abs(train.to_numpy()[SEASON:] - train.to_numpy()[:-SEASON]))
    return {
        "mae": round(float(np.mean(np.abs(a - f))), 2),
        "rmse": round(float(np.sqrt(np.mean((a - f) ** 2))), 2),
        "mape_pct": round(float(np.mean(np.abs(a - f) / a) * 100), 2),
        "mase": round(float(np.mean(np.abs(a - f)) / mase_scale), 3),
    }


def _fit(log_series: pd.Series, order, seasonal):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SARIMAX(log_series, order=order, seasonal_order=seasonal).fit(disp=False)


def run() -> dict:
    s = load_air_passengers()
    ls = np.log(s)
    train, test = s.iloc[:-HOLDOUT], s.iloc[-HOLDOUT:]
    log_train = np.log(train)

    # ── stationarity + ACF/PACF (EDA) ────────────────────────────────────────
    diffed = ls.diff().diff(SEASON).dropna()
    def adf(x):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # statsmodels FutureWarning about adfuller's return type
            return round(float(adfuller(x)[1]), 4)

    eda = {
        "n": int(len(s)), "start": str(s.index[0].date()), "end": str(s.index[-1].date()),
        "train_months": int(len(train)), "test_months": int(len(test)),
        "adf_pvalue": {
            "raw": adf(s), "log": adf(ls),
            "log_diff1": adf(ls.diff().dropna()),
            "log_diff1_seasonal12": adf(diffed),
        },
        "transform_rationale": "Seasonal swings grow with the level (multiplicative) -> log stabilises variance; "
                               "d=1 removes trend, D=1 (lag 12) removes seasonality.",
        "acf_lags_1_24": [round(float(v), 3) for v in acf(diffed, nlags=24)[1:]],
        "pacf_lags_1_24": [round(float(v), 3) for v in pacf(diffed, nlags=24)[1:]],
        "acf_pacf_95_band": round(float(1.96 / np.sqrt(len(diffed))), 3),
    }

    # ── order selection by AIC on TRAIN only ─────────────────────────────────
    grid = []
    for p, q, P, Q in itertools.product([0, 1, 2], [0, 1, 2], [0, 1], [0, 1]):
        try:
            m = _fit(log_train, (p, 1, q), (P, 1, Q, SEASON))
            grid.append({"order": [p, 1, q], "seasonal_order": [P, 1, Q, SEASON], "aic": round(float(m.aic), 2)})
        except Exception:
            continue
    grid.sort(key=lambda g: g["aic"])
    best = grid[0]
    order, seasonal = tuple(best["order"]), tuple(best["seasonal_order"])

    # ── hold-out forecast ────────────────────────────────────────────────────
    model = _fit(log_train, order, seasonal)
    fc = model.get_forecast(HOLDOUT)
    mean = np.exp(fc.predicted_mean.to_numpy())
    ci = np.exp(fc.conf_int(alpha=0.05).to_numpy())

    naive = np.repeat(train.iloc[-1], HOLDOUT)
    snaive = train.iloc[-SEASON:].to_numpy()[np.arange(HOLDOUT) % SEASON]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        hw = ExponentialSmoothing(train, trend="add", seasonal="mul", seasonal_periods=SEASON).fit()
    hw_fc = hw.forecast(HOLDOUT).to_numpy()

    results = {
        f"sarima{order}x{seasonal}": _metrics(test, mean, train),
        "holt_winters_add_trend_mul_season": _metrics(test, hw_fc, train),
        "seasonal_naive": _metrics(test, snaive, train),
        "naive_last_value": _metrics(test, naive, train),
    }

    # ── residual diagnostics + interval coverage ─────────────────────────────
    resid = model.resid.iloc[SEASON + 1:]  # drop the diffuse-initialisation burn-in
    lb = acorr_ljungbox(resid, lags=[12, 24], return_df=True)
    coverage = float(np.mean((test.to_numpy() >= ci[:, 0]) & (test.to_numpy() <= ci[:, 1])))

    # ── rolling-origin check (fixed order, 12-step horizon) ──────────────────
    roll = []
    for o in (96, 108, 120):
        m = _fit(np.log(s.iloc[:o]), order, seasonal)
        f = np.exp(m.get_forecast(12).predicted_mean.to_numpy())
        act = s.iloc[o:o + 12]
        sn = s.iloc[o - SEASON:o].to_numpy()
        roll.append({"origin": str(s.index[o - 1].date()),
                     "sarima_mape_pct": round(float(np.mean(np.abs(act.to_numpy() - f) / act.to_numpy()) * 100), 2),
                     "seasonal_naive_mape_pct": round(float(np.mean(np.abs(act.to_numpy() - sn) / act.to_numpy()) * 100), 2)})

    return {
        "dataset": "AirPassengers (monthly, 1949-1960)",
        "eda": eda,
        "selection": {"criterion": "AIC on training window only", "grid_size": len(grid), "top5": grid[:5]},
        "selected": {"order": list(order), "seasonal_order": list(seasonal)},
        "holdout_metrics": results,
        "residual_diagnostics": {
            "ljung_box_pvalue_lag12": round(float(lb["lb_pvalue"].iloc[0]), 4),
            "ljung_box_pvalue_lag24": round(float(lb["lb_pvalue"].iloc[1]), 4),
            "reading": "p > 0.05 => no significant leftover autocorrelation (residuals ~ white noise)",
        },
        "interval_coverage_95": round(coverage, 3),
        "rolling_origin": roll,
        "series_for_plot": {
            "history_dates": [str(d.date()) for d in train.index],
            "history": train.round(1).tolist(),
            "test_dates": [str(d.date()) for d in test.index],
            "actual": test.round(1).tolist(),
            "forecast": [round(float(v), 1) for v in mean],
            "lower95": [round(float(v), 1) for v in ci[:, 0]],
            "upper95": [round(float(v), 1) for v in ci[:, 1]],
        },
    }
