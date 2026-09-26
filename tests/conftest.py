import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def raw():
    """A tiny raw extract covering the cleaning edge cases."""
    rows = [
        # Invoice, StockCode, Quantity, InvoiceDate, Price, CustomerID, Country
        ("1001", "85123A", 2, "2010-01-05", 5.0, 1, "United Kingdom"),
        ("1001", "22423", 1, "2010-01-05", 10.0, 1, "United Kingdom"),
        ("1001", "POST", 1, "2010-01-05", 15.0, 1, "United Kingdom"),  # fee: dropped
        ("C1002", "85123A", -1, "2010-01-10", 5.0, 1, "United Kingdom"),  # return: kept, negative
        ("1003", "85123A", 3, "2010-02-01", 5.0, None, "United Kingdom"),  # no customer: dropped
        ("1004", "85123A", 1, "2010-02-02", 0.0, 2, "France"),  # zero price: dropped
        ("1005", "22423", 4, "2010-03-01", 10.0, 2, "France"),
        ("1005", "22423", 4, "2010-03-01", 10.0, 2, "France"),  # exact duplicate: dropped
    ]
    df = pd.DataFrame(
        rows,
        columns=[
            "Invoice",
            "StockCode",
            "Quantity",
            "InvoiceDate",
            "Price",
            "Customer ID",
            "Country",
        ],
    )
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    return df


@pytest.fixture
def synthetic_tx():
    """Simulated transactions: loyal customers buy monthly, lapsing ones stop mid-way."""
    rng = np.random.default_rng(0)
    rows, inv = [], 0
    start = pd.Timestamp("2010-01-01")
    for cid in range(1, 201):
        loyal = cid <= 100
        stop = 400 if loyal else rng.integers(60, 140)
        day = int(rng.integers(0, 30))
        while day < stop:
            inv += 1
            rows.append(
                (
                    str(inv),
                    "A",
                    int(rng.integers(1, 5)),
                    start + pd.Timedelta(days=day),
                    float(rng.uniform(5, 50)),
                    cid,
                    "United Kingdom",
                )
            )
            day += int(rng.integers(15, 45))
    df = pd.DataFrame(
        rows,
        columns=[
            "Invoice",
            "StockCode",
            "Quantity",
            "InvoiceDate",
            "Price",
            "CustomerID",
            "Country",
        ],
    )
    from churnclv.data import clean

    return clean(df)
