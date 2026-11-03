"""
data.py — dataset loading with a local cache.

First run downloads the CSVs into ml/data/ (git-ignored); later runs work offline.
Both datasets are public benchmarks:
  - IBM Telco Customer Churn (7,043 customers, 21 columns, binary target `Churn`)
  - R `datasets::AirPassengers` (monthly airline passengers, 1949-01 .. 1960-12, n=144)
"""

import urllib.request
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent / "data"

SOURCES = {
    "telco_churn.csv": "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv",
    "air_passengers.csv": "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/datasets/AirPassengers.csv",
}


def _path(name: str) -> Path:
    DATA_DIR.mkdir(exist_ok=True)
    p = DATA_DIR / name
    if not p.exists():
        urllib.request.urlretrieve(SOURCES[name], p)
    return p


def load_churn() -> pd.DataFrame:
    return pd.read_csv(_path("telco_churn.csv"))


def load_air_passengers() -> pd.Series:
    raw = pd.read_csv(_path("air_passengers.csv"))
    idx = pd.date_range("1949-01-01", periods=len(raw), freq="MS")
    return pd.Series(raw["value"].values.astype(float), index=idx, name="passengers")
