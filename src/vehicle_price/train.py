"""Train, compare and persist the vehicle price model."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score, root_mean_squared_error
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .features import TARGET, build_features, clean
from .model import candidates

log = logging.getLogger("vehicle_price")
ROOT = Path(__file__).resolve().parents[2]
SEED = 42


@dataclass
class TrainResult:
    best: str
    cv: dict
    test: dict
    baseline_v1: dict
    n_train: int
    n_test: int


def load(path: Path) -> pd.DataFrame:
    df = clean(pd.read_csv(path))
    return df[df[TARGET].notna()].reset_index(drop=True)


def _metrics(y_true, y_pred) -> dict:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
        "mape": float(mean_absolute_percentage_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def v1_baseline(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Re-run the repo's original v1 approach on the same split, for an honest before/after.

    v1 label-encoded categoricals and fitted a RandomForest on raw price. LabelEncoder cannot
    handle categories missing from training, so those test rows are scored with an extra code.
    """
    cols = ["make", "model", "transmission", "fuel", "body", "drivetrain"]
    feats = ["year", "cylinders", "mileage"] + cols
    tr, te = train[feats].copy(), test[feats].copy()
    unseen = 0
    for c in cols:
        le = LabelEncoder().fit(tr[c].astype(str))
        known = set(le.classes_)
        mask = ~te[c].astype(str).isin(known)
        unseen += int(mask.sum())
        tr[c] = le.transform(tr[c].astype(str))
        te[c] = np.where(mask, len(known), le.transform(te[c].astype(str).where(~mask, le.classes_[0])))
    tr, te = tr.fillna(tr.mean()), te.fillna(tr.mean())
    sc = StandardScaler().fit(tr)
    rf = RandomForestRegressor(n_estimators=100, random_state=SEED).fit(sc.transform(tr), train[TARGET])
    out = _metrics(test[TARGET], rf.predict(sc.transform(te)))
    out["unseen_category_values_in_test"] = unseen
    return out


def train(data_path: Path, model_path: Path, report_path: Path) -> TrainResult:
    df = load(data_path)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=SEED)
    X_train, y_train = build_features(train_df), train_df[TARGET]
    X_test, y_test = build_features(test_df), test_df[TARGET]

    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
    scores = {}
    for name, pipe in candidates(SEED).items():
        res = cross_validate(pipe, X_train, y_train, cv=cv, n_jobs=1,
                             scoring={"mae": "neg_mean_absolute_error", "r2": "r2"})
        scores[name] = {"cv_mae_mean": float(-res["test_mae"].mean()), "cv_mae_std": float(res["test_mae"].std()),
                        "cv_r2_mean": float(res["test_r2"].mean())}
        log.info("%-24s CV MAE $%.0f ± %.0f  R² %.3f", name, scores[name]["cv_mae_mean"],
                 scores[name]["cv_mae_std"], scores[name]["cv_r2_mean"])

    best = min(scores, key=lambda k: scores[k]["cv_mae_mean"])
    model = clone(candidates(SEED)[best]).fit(X_train, y_train)
    test_metrics = _metrics(y_test, model.predict(X_test))
    baseline = v1_baseline(train_df, test_df)
    log.info("best=%s test MAE $%.0f R² %.3f (v1 baseline MAE $%.0f R² %.3f)", best, test_metrics["mae"],
             test_metrics["r2"], baseline["mae"], baseline["r2"])

    # refit on all labelled data for the shipped artifact
    final = clone(candidates(SEED)[best]).fit(build_features(df), df[TARGET])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": final, "model_name": best, "trained_rows": len(df)}, model_path, compress=3)

    result = TrainResult(best, scores, test_metrics, baseline, len(train_df), len(test_df))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result.__dict__, indent=2))

    # keep the holdout predictions for the report charts
    preds = test_df[["make", "model", "year", TARGET]].copy()
    preds["predicted"] = model.predict(X_test)
    preds.to_csv(report_path.with_name("holdout_predictions.csv"), index=False)
    joblib.dump({"pipeline": model, "X_test": X_test, "y_test": y_test},
                report_path.with_name("holdout_model.joblib"), compress=3)
    return result
