"""`python -m vehicle_price train | report | predict`"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import click

from .predict import predict as predict_prices
from .report import render
from .train import ROOT, train as train_model

DATA = ROOT / "data" / "vehicles.csv"
MODEL = ROOT / "models" / "vehicle_price_pipeline.joblib"
REPORTS = ROOT / "reports"
IMAGES = ROOT / "docs" / "images"


@click.group()
def cli() -> None:
    """Vehicle price prediction."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)-5s %(message)s")


@cli.command()
@click.option("--data", type=click.Path(path_type=Path), default=DATA, show_default=True)
def train(data: Path) -> None:
    """Cross-validate candidates, pick the best, save one pipeline file."""
    r = train_model(data, MODEL, REPORTS / "metrics.json")
    click.echo(f"best model: {r.best}  holdout MAE ${r.test['mae']:,.0f}  R² {r.test['r2']:.3f}")


@cli.command()
def report() -> None:
    """Render README charts from the latest training run."""
    for p in render(REPORTS, IMAGES):
        click.echo(f"wrote {p}")


@cli.command()
@click.option("--make", required=True)
@click.option("--model", "model_name", required=True)
@click.option("--year", type=int, required=True)
@click.option("--mileage", type=float)
@click.option("--cylinders", type=float)
@click.option("--fuel")
@click.option("--transmission")
@click.option("--body")
@click.option("--drivetrain")
@click.option("--doors", type=float)
@click.option("--json", "as_json", is_flag=True, help="Print JSON instead of text.")
def predict(make, model_name, year, mileage, cylinders, fuel, transmission, body, drivetrain, doors,
            as_json) -> None:
    """Estimate the price of one vehicle."""
    v = {"make": make, "model": model_name, "year": year, "mileage": mileage, "cylinders": cylinders,
         "fuel": fuel, "transmission": transmission, "body": body, "drivetrain": drivetrain, "doors": doors}
    price = predict_prices([v], MODEL)[0]
    click.echo(json.dumps({**v, "predicted_price_usd": price}) if as_json else f"Estimated price: ${price:,.0f}")


def main() -> None:
    cli()
