"""Cleaning and feature engineering for vehicle listings."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

TARGET = "price"
CATEGORICAL = ["make", "model", "body", "drivetrain", "fuel_type", "transmission_type"]
NUMERIC = ["year", "cylinders", "mileage", "doors", "gears", "is_turbo", "is_hybrid_engine"]
FEATURES = CATEGORICAL + NUMERIC

FUEL_MAP = {
    "gasoline": "gasoline",
    "hybrid": "hybrid",
    "phev hybrid fuel": "plug_in_hybrid",
    "electric": "electric",
    "diesel": "diesel",
    "diesel (b20 capable)": "diesel",
    "e85 flex fuel": "flex_fuel",
}


def _gears(text: object) -> float:
    m = re.search(r"(\d+)\s*-?\s*sp", str(text).lower())
    return float(m.group(1)) if m else np.nan


def _transmission_type(text: object) -> str:
    t = str(text).lower()
    if "cvt" in t:
        return "cvt"
    if "manual" in t:
        return "manual"
    if "dsg" in t or "dual" in t or "dct" in t:
        return "dual_clutch"
    if "auto" in t:
        return "automatic"
    return "other"


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    """Drop unusable rows and normalise messy text columns."""
    df = raw.copy()
    df[TARGET] = pd.to_numeric(df.get(TARGET), errors="coerce")
    # a listing price below $1,000 for a new/near-new vehicle is a data-entry error
    df = df[df[TARGET].isna() | (df[TARGET] >= 1000)]
    for col in ("make", "model", "body", "drivetrain"):
        df[col] = df[col].astype("string").str.strip()
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["make"] = df["make"].astype("string")
    out["model"] = df["model"].astype("string")
    out["body"] = df["body"].astype("string")
    out["drivetrain"] = df["drivetrain"].astype("string")
    out["fuel_type"] = df["fuel"].astype("string").str.strip().str.lower().map(FUEL_MAP).fillna("other")
    out["transmission_type"] = df["transmission"].map(_transmission_type)
    out["gears"] = df["transmission"].map(_gears)
    engine = df.get("engine", pd.Series("", index=df.index)).astype("string").str.lower().fillna("")
    out["is_turbo"] = engine.str.contains("turbo").astype(int)
    out["is_hybrid_engine"] = engine.str.contains("hybrid").astype(int)
    for col in ("year", "cylinders", "mileage", "doors"):
        out[col] = pd.to_numeric(df.get(col), errors="coerce")
    for col in CATEGORICAL:  # sklearn wants plain object dtype with np.nan for missing
        out[col] = out[col].astype(object).where(out[col].notna(), np.nan)
    return out[FEATURES]
