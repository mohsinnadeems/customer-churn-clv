"""Churn models, metrics and explanations."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from .features import FEATURES

# Heavy-tailed features get a signed log transform for the linear model.
_LOG = [
    "recency_days",
    "tenure_days",
    "orders_total",
    "orders_90d",
    "orders_365d",
    "revenue_90d",
    "revenue_365d",
    "avg_order_value",
    "mean_gap_days",
    "recency_gap_ratio",
    "spend_trend",
    "products_365d",
]


def _signed_log(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    for c in _LOG:
        X[c] = np.sign(X[c]) * np.log1p(np.abs(X[c]))
    return X


def make_models(seed: int = 42) -> dict:
    return {
        "Logistic regression": make_pipeline(
            FunctionTransformer(_signed_log),
            SimpleImputer(strategy="median", add_indicator=True),
            StandardScaler(),
            LogisticRegression(max_iter=2000, C=1.0),
        ),
        "Gradient boosting": HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=400,
            max_leaf_nodes=15,
            min_samples_leaf=100,
            l2_regularization=1.0,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=seed,
        ),
    }


def lift_at(y: np.ndarray, score: np.ndarray, frac: float = 0.1) -> float:
    """Churn rate among the top `frac` highest-scored customers, divided by the base rate."""
    n = max(int(len(y) * frac), 1)
    top = np.argsort(-score)[:n]
    return float(np.mean(y[top]) / np.mean(y))


def evaluate(y: np.ndarray, p: np.ndarray, probabilistic: bool = True) -> dict:
    out = {
        "roc_auc": roc_auc_score(y, p),
        "pr_auc": average_precision_score(y, p),
        "lift_top_10pct": lift_at(y, p, 0.1),
    }
    out["brier"] = brier_score_loss(y, p) if probabilistic else None
    return {k: (None if v is None else round(float(v), 4)) for k, v in out.items()}


def fit_and_evaluate(train: pd.DataFrame, test: pd.DataFrame, seed: int = 42):
    """Fit each model on training snapshots and score the held-out snapshot."""
    Xtr, ytr = train[FEATURES], train["churned"].to_numpy()
    Xte, yte = test[FEATURES], test["churned"].to_numpy()
    results, preds, fitted = {}, {}, {}

    # Baseline: rank customers by days since their last purchase.
    preds["Recency rule"] = Xte["recency_days"].to_numpy(dtype=float)
    results["Recency rule"] = evaluate(yte, preds["Recency rule"], probabilistic=False)

    for name, model in make_models(seed).items():
        model.fit(Xtr, ytr)
        p = model.predict_proba(Xte)[:, 1]
        preds[name], fitted[name] = p, model
        results[name] = evaluate(yte, p)
    return results, preds, fitted


def importance(model, X: pd.DataFrame, y: np.ndarray, seed: int = 42) -> pd.Series:
    """Permutation importance: drop in ROC AUC when each feature is shuffled."""
    r = permutation_importance(model, X, y, scoring="roc_auc", n_repeats=5, random_state=seed)
    return pd.Series(r.importances_mean, index=X.columns).sort_values(ascending=False)


def calibration_table(y: np.ndarray, p: np.ndarray, bins: int = 10) -> pd.DataFrame:
    df = pd.DataFrame({"y": y, "p": p})
    df["bin"] = pd.qcut(df["p"], bins, labels=False, duplicates="drop")
    return df.groupby("bin").agg(predicted=("p", "mean"), observed=("y", "mean"), n=("y", "size"))
