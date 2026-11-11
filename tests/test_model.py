import json
from pathlib import Path

import pytest

from vehicle_price.predict import predict
from vehicle_price.train import train

DATA = Path(__file__).resolve().parents[1] / "data" / "vehicles.csv"


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    d = tmp_path_factory.mktemp("run")
    result = train(DATA, d / "model.joblib", d / "reports" / "metrics.json")
    return result, d


def test_training_beats_v1_baseline(trained):
    result, d = trained
    assert result.test["r2"] > 0.6
    assert result.test["mae"] < result.baseline_v1["mae"]
    saved = json.loads((d / "reports" / "metrics.json").read_text())
    assert saved["best"] in saved["cv"]


def test_predicts_known_vehicle_in_plausible_range(trained):
    _, d = trained
    price = predict([{"make": "Jeep", "model": "Wagoneer", "year": 2024, "cylinders": 6, "fuel": "Gasoline",
                      "body": "SUV", "drivetrain": "Four-wheel Drive"}], d / "model.joblib")[0]
    assert 40_000 < price < 120_000


def test_unseen_make_and_model_do_not_crash(trained):
    _, d = trained  # v1 raised ValueError from LabelEncoder here
    price = predict([{"make": "Rivian", "model": "R9", "year": 2025, "fuel": "Electric"}], d / "model.joblib")[0]
    assert price > 0


def test_missing_required_field(trained):
    _, d = trained
    with pytest.raises(ValueError, match="model"):
        predict([{"make": "Jeep", "year": 2024}], d / "model.joblib")
