import pandas as pd
import pytest

from churnclv.features import FEATURES, snapshot


def test_no_future_leakage(synthetic_tx):
    """Changing data after the cutoff must not change any feature."""
    cutoff = pd.Timestamp("2010-06-01")
    a = snapshot(synthetic_tx, cutoff, with_label=False)
    later = synthetic_tx[synthetic_tx["InvoiceDate"] >= cutoff].copy()
    later["Revenue"] *= 100
    tampered = pd.concat([synthetic_tx[synthetic_tx["InvoiceDate"] < cutoff], later])
    b = snapshot(tampered, cutoff, with_label=False)
    pd.testing.assert_frame_equal(a[FEATURES], b[FEATURES])


def test_labels_match_future_purchases(synthetic_tx):
    snap = snapshot(synthetic_tx, "2010-06-01", horizon_days=90)
    loyal = snap[snap["CustomerID"] <= 100]
    lapsed = snap[snap["CustomerID"] > 100]
    assert loyal["churned"].mean() == 0.0
    assert lapsed["churned"].mean() == 1.0


def test_label_window_must_fit_in_data(synthetic_tx):
    with pytest.raises(ValueError):
        snapshot(synthetic_tx, synthetic_tx["InvoiceDate"].max(), horizon_days=90)


def test_only_active_customers(synthetic_tx):
    snap = snapshot(synthetic_tx, "2011-01-01", horizon_days=30, active_days=60)
    assert (snap["recency_days"] <= 60).all()
