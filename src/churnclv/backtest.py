"""Walk-forward backtest: how stable is churn performance across test periods and seasons?

Each fold scores one monthly test cutoff. Models are trained on every earlier monthly
cutoff whose label window has closed by the test cutoff (an expanding window with a purge
gap of one horizon), so no training label ever overlaps the period being tested.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .churn import evaluate, fit_and_evaluate
from .clv import bgnbd_churn_score
from .features import snapshot

log = logging.getLogger("churnclv")

PROBABILISTIC = ("Logistic regression", "Gradient boosting")


def folds(
    cutoffs: list[pd.Timestamp],
    horizon_days: int,
    min_train: int = 4,
) -> list[tuple[pd.Timestamp, list[pd.Timestamp]]]:
    """(test cutoff, training cutoffs) pairs; training labels end on or before the test cutoff."""
    cutoffs = sorted(pd.Timestamp(c) for c in cutoffs)
    gap = pd.Timedelta(days=horizon_days)
    out = []
    for test in cutoffs:
        train = [c for c in cutoffs if c + gap <= test]
        if len(train) >= min_train:
            out.append((test, train))
    return out


def monthly_cutoffs(tx: pd.DataFrame, start: str, horizon_days: int) -> list[pd.Timestamp]:
    """Month starts from `start` whose label window fits inside the data."""
    last = tx["InvoiceDate"].max().normalize() + pd.Timedelta(days=1 - horizon_days)
    return list(pd.date_range(start, last, freq="MS"))


def backtest(
    tx: pd.DataFrame,
    cutoffs: list[pd.Timestamp],
    horizon_days: int = 90,
    min_train: int = 4,
    seed: int = 42,
) -> pd.DataFrame:
    """One row per (test cutoff, model) with ranking metrics and predicted vs actual churn."""
    snaps = {c: snapshot(tx, c, horizon_days) for c in cutoffs}
    rows = []
    for test_cut, train_cuts in folds(cutoffs, horizon_days, min_train):
        train = pd.concat([snaps[c] for c in train_cuts], ignore_index=True)
        test = snaps[test_cut]
        y = test["churned"].to_numpy()
        results, preds, _ = fit_and_evaluate(train, test, seed)
        bg = bgnbd_churn_score(tx, test_cut, test["CustomerID"], horizon_days)
        results["BG/NBD, no labels"] = evaluate(y, bg, probabilistic=False)
        log.info(
            "Backtest %s: %d train cutoffs, %d test customers, AUC %s",
            test_cut.date(),
            len(train_cuts),
            len(test),
            ", ".join(f"{m} {r['roc_auc']:.3f}" for m, r in results.items()),
        )
        for model, r in results.items():
            rows.append(
                {
                    "test_cutoff": test_cut,
                    "model": model,
                    "train_cutoffs": len(train_cuts),
                    "train_rows": len(train),
                    "test_customers": len(test),
                    "churn_rate": float(y.mean()),
                    "mean_predicted": float(preds[model].mean())
                    if model in PROBABILISTIC
                    else None,
                    **r,
                }
            )
    return pd.DataFrame(rows)


def summarise(bt: pd.DataFrame) -> dict:
    """Mean, spread and range of each metric per model, plus how often each model wins."""
    metrics = ["roc_auc", "pr_auc", "lift_top_10pct"]
    agg = bt.groupby("model")[metrics].agg(["mean", "std", "min", "max"]).round(4)
    out = {m: {k: agg.loc[m, k].to_dict() for k in metrics} for m in agg.index.get_level_values(0)}
    best = bt.loc[bt.groupby("test_cutoff")["roc_auc"].idxmax(), "model"].value_counts()
    for m in out:
        out[m]["folds_best_auc"] = int(best.get(m, 0))
    for m in PROBABILISTIC:
        gap = bt.loc[bt["model"] == m, "mean_predicted"] - bt.loc[bt["model"] == m, "churn_rate"]
        out[m]["calibration_gap"] = {
            "mean": round(float(gap.mean()), 4),
            "mean_abs": round(float(gap.abs().mean()), 4),
        }
    return out


def to_records(bt: pd.DataFrame) -> list[dict]:
    df = bt.assign(test_cutoff=bt["test_cutoff"].dt.date.astype(str)).round(4)
    return df.replace({np.nan: None}).to_dict(orient="records")
