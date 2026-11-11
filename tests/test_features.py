import pandas as pd

from vehicle_price.features import FEATURES, _gears, _transmission_type, build_features, clean


def test_gear_parsing():
    assert _gears("8-Speed Automatic") == 8
    assert _gears("6-Spd Aisin F21-250 PHEV Auto Trans") == 6
    assert pd.isna(_gears("Automatic"))


def test_transmission_type():
    assert _transmission_type("Automatic CVT") == "cvt"
    assert _transmission_type("7-Speed DSG Automatic with Tiptronic") == "dual_clutch"
    assert _transmission_type("10-Speed Automatic") == "automatic"
    assert _transmission_type(None) == "other"


def test_clean_drops_placeholder_prices():
    df = clean(pd.DataFrame({"price": ["0", "45000", "abc"], "make": ["a"] * 3, "model": ["m"] * 3,
                             "body": ["SUV"] * 3, "drivetrain": ["AWD"] * 3}))
    assert df["price"].tolist()[0] == 45000
    assert len(df) == 2  # "abc" becomes NaN and is kept for prediction-only use


def test_build_features_shape_and_mapping():
    raw = pd.DataFrame([{"make": "Jeep", "model": "Wagoneer", "year": 2024, "fuel": "PHEV Hybrid Fuel",
                         "transmission": "8-Speed Automatic", "engine": "16V GDI DOHC Turbo Hybrid",
                         "body": "SUV", "drivetrain": "Four-wheel Drive", "cylinders": 4, "mileage": 5, "doors": 4}])
    f = build_features(raw)
    assert list(f.columns) == FEATURES
    row = f.iloc[0]
    assert row["fuel_type"] == "plug_in_hybrid" and row["gears"] == 8
    assert row["is_turbo"] == 1 and row["is_hybrid_engine"] == 1
