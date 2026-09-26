"""Monthly acquisition cohorts and retention."""

from __future__ import annotations

import pandas as pd

from .data import orders as to_orders


def retention_matrix(tx: pd.DataFrame, max_months: int = 12) -> pd.DataFrame:
    """Share of each first-purchase cohort that buys again N months later."""
    o = to_orders(tx)
    o["month"] = o["date"].dt.to_period("M")
    first = o.groupby("CustomerID")["month"].min().rename("cohort")
    o = o.join(first, on="CustomerID")
    o["age"] = (o["month"] - o["cohort"]).apply(lambda d: d.n)
    counts = o.groupby(["cohort", "age"])["CustomerID"].nunique().unstack(fill_value=0)
    ret = counts.div(counts[0], axis=0)
    last = o["month"].max()
    # Blank out cells that fall after the end of the data.
    for cohort in ret.index:
        for age in ret.columns:
            if cohort + age > last:
                ret.loc[cohort, age] = float("nan")
    ret.insert(0, "customers", counts[0])
    return ret.loc[:, ["customers", *range(0, max_months + 1)]]
