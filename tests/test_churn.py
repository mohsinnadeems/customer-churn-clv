import numpy as np
import pandas as pd

from churnclv.churn import calibration_table, evaluate, fit_and_evaluate, lift_at
from churnclv.features import snapshot


def test_lift_at():
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    assert lift_at(y, np.arange(10)[::-1].astype(float), 0.2) == 5.0


def test_evaluate_perfect_scores():
    y = np.array([0, 0, 1, 1])
    m = evaluate(y, np.array([0.1, 0.2, 0.8, 0.9]))
    assert m["roc_auc"] == 1.0 and m["pr_auc"] == 1.0


def test_models_learn_synthetic_churn(synthetic_tx):
    train = pd.concat([snapshot(synthetic_tx, c) for c in ["2010-04-01", "2010-05-01"]])
    test = snapshot(synthetic_tx, "2010-07-01")
    results, preds, _ = fit_and_evaluate(train, test)
    assert results["Logistic regression"]["roc_auc"] > 0.8
    assert set(preds) == {"Recency rule", "Logistic regression", "Gradient boosting"}


def test_calibration_table_bins():
    rng = np.random.default_rng(1)
    p = rng.uniform(size=1000)
    y = (rng.uniform(size=1000) < p).astype(int)
    t = calibration_table(y, p, bins=5)
    assert len(t) == 5 and np.allclose(t["predicted"], t["observed"], atol=0.1)
