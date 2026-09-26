"""Point-in-time customer features and churn labels.

Every feature is computed only from transactions *before* the cutoff date, and the label
only from transactions in the horizon *after* it. This mirrors how the model would be
used in production and prevents leakage from the future.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import orders as to_orders

FEATURES = [
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
    "return_rate",
    "products_365d",
    "months_active_365d",
    "is_uk",
]


def snapshot(
    tx: pd.DataFrame,
    cutoff: str | pd.Timestamp,
    horizon_days: int = 90,
    active_days: int = 365,
    with_label: bool = True,
) -> pd.DataFrame:
    """Features for customers active in the `active_days` before `cutoff`.

    Label `churned` = 1 if the customer makes no purchase in the `horizon_days` after cutoff.
    """
    cutoff = pd.Timestamp(cutoff)
    past = tx[tx["InvoiceDate"] < cutoff]
    o = to_orders(past)
    if o.empty:
        return pd.DataFrame(columns=["CustomerID", *FEATURES])

    d90 = cutoff - pd.Timedelta(days=90)
    d365 = cutoff - pd.Timedelta(days=active_days)
    g = o.groupby("CustomerID")
    f = pd.DataFrame(
        {
            "first": g["date"].min(),
            "last": g["date"].max(),
            "orders_total": g.size(),
            "revenue_total": g["revenue"].sum(),
        }
    )
    f["orders_90d"] = o[o["date"] >= d90].groupby("CustomerID").size()
    f["orders_365d"] = o[o["date"] >= d365].groupby("CustomerID").size()
    f = f[f["orders_365d"].fillna(0) > 0].copy()  # active customers only

    recent = past[past["InvoiceDate"] >= d365]
    net = recent.groupby("CustomerID")["Revenue"]
    f["revenue_365d"] = net.sum()
    f["revenue_90d"] = past[past["InvoiceDate"] >= d90].groupby("CustomerID")["Revenue"].sum()
    gross = recent[~recent["is_return"]].groupby("CustomerID")["Revenue"].sum()
    returned = -recent[recent["is_return"]].groupby("CustomerID")["Revenue"].sum()
    f["return_rate"] = (returned / gross).clip(0, 1)
    f["products_365d"] = recent[~recent["is_return"]].groupby("CustomerID")["StockCode"].nunique()
    months = o[o["date"] >= d365].assign(m=lambda x: x["date"].dt.to_period("M"))
    f["months_active_365d"] = months.groupby("CustomerID")["m"].nunique()
    country = past.groupby("CustomerID")["Country"].agg(lambda s: s.mode().iat[0])
    f["is_uk"] = (country == "United Kingdom").astype(int)

    f["recency_days"] = (cutoff - f["last"]).dt.days
    f["tenure_days"] = (cutoff - f["first"]).dt.days
    f["avg_order_value"] = f["revenue_total"] / f["orders_total"]
    span = (f["last"] - f["first"]).dt.days
    f["mean_gap_days"] = np.where(
        f["orders_total"] > 1, (span / (f["orders_total"] - 1)).clip(lower=1), np.nan
    )
    f["recency_gap_ratio"] = f["recency_days"] / f["mean_gap_days"]
    f["spend_trend"] = f["revenue_90d"].fillna(0) / (f["revenue_365d"].clip(lower=1) / 4)
    f = f.fillna({c: 0 for c in ["orders_90d", "revenue_90d", "return_rate", "products_365d"]})

    out = f[FEATURES].reset_index()
    out.insert(1, "cutoff", cutoff)

    if with_label:
        end = cutoff + pd.Timedelta(days=horizon_days)
        if end > tx["InvoiceDate"].max() + pd.Timedelta(days=1):
            raise ValueError(f"label window ending {end.date()} runs past the end of the data")
        future = tx[(tx["InvoiceDate"] >= cutoff) & (tx["InvoiceDate"] < end) & ~tx["is_return"]]
        buyers = set(future["CustomerID"].unique())
        out["churned"] = (~out["CustomerID"].isin(buyers)).astype(int)
        fut_rev = future.groupby("CustomerID")["Revenue"].sum()
        out["future_revenue"] = out["CustomerID"].map(fut_rev).fillna(0.0)
    return out


def training_set(tx: pd.DataFrame, cutoffs: list[str], horizon_days: int = 90) -> pd.DataFrame:
    """Stack snapshots from several cutoffs. A customer can appear once per cutoff."""
    return pd.concat([snapshot(tx, c, horizon_days) for c in cutoffs], ignore_index=True)
