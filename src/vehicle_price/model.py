"""Model definitions: every candidate is one sklearn Pipeline, so preprocessing ships with the model."""
from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .features import CATEGORICAL, NUMERIC


def _preprocessor(scale: bool) -> ColumnTransformer:
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    cat = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
        # rare categories are pooled; unseen ones at prediction time map to the pooled bucket
        ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=3, sparse_output=False)),
    ])
    return ColumnTransformer([("num", Pipeline(num_steps), NUMERIC), ("cat", cat, CATEGORICAL)])


def _log_target(reg) -> TransformedTargetRegressor:
    # prices are right-skewed; fitting on log(price) stops expensive vehicles dominating the loss
    return TransformedTargetRegressor(regressor=reg, func=np.log1p, inverse_func=np.expm1)


def candidates(seed: int = 42) -> dict[str, Pipeline]:
    return {
        "ridge": Pipeline([("prep", _preprocessor(scale=True)), ("model", _log_target(Ridge(alpha=3.0)))]),
        "random_forest": Pipeline([
            ("prep", _preprocessor(scale=False)),
            ("model", _log_target(RandomForestRegressor(n_estimators=300, min_samples_leaf=2, n_jobs=-1,
                                                        random_state=seed))),
        ]),
        "hist_gradient_boosting": Pipeline([
            ("prep", _preprocessor(scale=False)),
            ("model", _log_target(HistGradientBoostingRegressor(learning_rate=0.06, max_iter=500,
                                                                l2_regularization=0.1, random_state=seed))),
        ]),
    }
