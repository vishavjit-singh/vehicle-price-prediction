"""Load the trained pipeline and price vehicles."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from .features import build_features

REQUIRED = ["make", "model", "year"]
OPTIONAL = ["cylinders", "mileage", "doors", "fuel", "transmission", "body", "drivetrain", "engine"]


@lru_cache(maxsize=2)
def load_model(path: str) -> dict:
    return joblib.load(path)


def predict(vehicles: list[dict], model_path: str | Path) -> list[float]:
    for v in vehicles:
        missing = [k for k in REQUIRED if v.get(k) in (None, "")]
        if missing:
            raise ValueError(f"missing required field(s): {', '.join(missing)}")
    df = pd.DataFrame([{k: v.get(k) for k in REQUIRED + OPTIONAL} for v in vehicles])
    bundle = load_model(str(model_path))
    return [round(float(p), 2) for p in bundle["pipeline"].predict(build_features(df))]
