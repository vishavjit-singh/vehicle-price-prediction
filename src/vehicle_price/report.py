"""Charts for the README, computed from the training report and holdout predictions."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402

INK, MUTED, ACCENT, SIGNAL, GRID, BAD = "#0e1a2b", "#5b6676", "#2747d6", "#c97d06", "#e3e7ee", "#c6373a"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": GRID,
                     "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.titlesize": 13,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": MUTED, "figure.dpi": 150})
USD = FuncFormatter(lambda x, _: f"${x / 1000:,.0f}k")
NAMES = {"ridge": "Ridge (linear)", "random_forest": "Random forest", "hist_gradient_boosting": "Gradient boosting"}


def model_comparison(report: dict, out: Path) -> None:
    rows = [(NAMES[k], v["cv_mae_mean"], v["cv_mae_std"]) for k, v in report["cv"].items()]
    rows.sort(key=lambda r: r[1], reverse=True)
    fig, ax = plt.subplots(figsize=(8, 3.6))
    colors = [ACCENT if n == NAMES[report["best"]] else "#b8c3d8" for n, _, _ in rows]
    ax.barh([r[0] for r in rows], [r[1] for r in rows], xerr=[r[2] for r in rows], color=colors,
            error_kw={"ecolor": MUTED, "capsize": 3})
    for i, (_, m, s) in enumerate(rows):
        ax.text(m + s, i, f"  ${m:,.0f}", va="center", fontsize=9, color=INK)
    v1 = report["baseline_v1"]["mae"]
    ax.axvline(v1, color=BAD, ls="--", lw=1.4)
    ax.text(v1, len(rows) - 0.45, f" v1 holdout MAE ${v1:,.0f}", color=BAD, fontsize=8.5, va="bottom")
    ax.xaxis.set_major_formatter(USD)
    ax.set_xlabel("5-fold cross-validated mean absolute error (lower is better)")
    ax.set_title("Model comparison")
    ax.grid(axis="x", color=GRID)
    fig.tight_layout()
    fig.savefig(out / "model_comparison.png")
    plt.close(fig)


def actual_vs_predicted(preds: pd.DataFrame, report: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 6))
    ax.scatter(preds["price"], preds["predicted"], s=16, alpha=0.65, color=ACCENT, edgecolors="none")
    lo, hi = 0, max(preds["price"].max(), preds["predicted"].max()) * 1.05
    ax.plot([lo, hi], [lo, hi], color=SIGNAL, lw=1.6, label="perfect prediction")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.xaxis.set_major_formatter(USD)
    ax.yaxis.set_major_formatter(USD)
    ax.set_xlabel("actual listing price")
    ax.set_ylabel("predicted price")
    t = report["test"]
    ax.set_title(f"Holdout set: R² {t['r2']:.3f} · MAE ${t['mae']:,.0f}")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(color=GRID)
    fig.tight_layout()
    fig.savefig(out / "actual_vs_predicted.png")
    plt.close(fig)


def error_distribution(preds: pd.DataFrame, out: Path) -> None:
    pct = (preds["predicted"] - preds["price"]) / preds["price"] * 100
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.hist(pct.clip(-60, 60), bins=40, color="#b8c3d8", edgecolor="white")
    within = (pct.abs() <= 10).mean()
    ax.axvspan(-10, 10, color=ACCENT, alpha=0.12)
    ax.text(0, ax.get_ylim()[1] * 0.9, f"{within:.0%} of holdout vehicles\npriced within ±10%",
            ha="center", va="top", fontsize=9.5, color=INK)
    ax.set_xlabel("prediction error (% of actual price, clipped at ±60%)")
    ax.set_ylabel("vehicles")
    ax.set_title("How far off are predictions?")
    fig.tight_layout()
    fig.savefig(out / "error_distribution.png")
    plt.close(fig)


def feature_importance(holdout: dict, out: Path) -> None:
    res = permutation_importance(holdout["pipeline"], holdout["X_test"], holdout["y_test"], n_repeats=10,
                                 random_state=42, scoring="neg_mean_absolute_error")
    imp = pd.Series(res.importances_mean, index=holdout["X_test"].columns).sort_values()
    imp = imp[imp > 0]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(imp.index, imp.values, color=ACCENT)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.set_xlabel("increase in MAE when the feature is shuffled")
    ax.set_title("Permutation feature importance (holdout)")
    ax.grid(axis="x", color=GRID)
    fig.tight_layout()
    fig.savefig(out / "feature_importance.png")
    plt.close(fig)


def render(report_dir: Path, out: Path) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    report = json.loads((report_dir / "metrics.json").read_text())
    preds = pd.read_csv(report_dir / "holdout_predictions.csv")
    model_comparison(report, out)
    actual_vs_predicted(preds, report, out)
    error_distribution(preds, out)
    feature_importance(joblib.load(report_dir / "holdout_model.joblib"), out)
    np.random.seed(0)
    return sorted(out.glob("*.png"))
