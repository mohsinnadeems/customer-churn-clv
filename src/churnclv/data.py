"""Download, load and clean the UCI Online Retail II dataset."""

from __future__ import annotations

import io
import logging
import zipfile
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

UCI_URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"

# Stock codes that are fees, postage or adjustments rather than products.
NON_PRODUCT_CODES = {
    "POST",
    "DOT",
    "M",
    "C2",
    "D",
    "S",
    "B",
    "CRUK",
    "BANK CHARGES",
    "AMAZONFEE",
    "ADJUST",
    "ADJUST2",
    "TEST001",
    "TEST002",
    "PADS",
    "gift_0001_10",
    "gift_0001_20",
    "gift_0001_30",
    "gift_0001_40",
    "gift_0001_50",
    "m",
}

_RENAME = {
    "InvoiceNo": "Invoice",
    "UnitPrice": "Price",
    "Customer ID": "CustomerID",
}


def download_uci(dest: str | Path = "data") -> Path:
    """Download the dataset from the UCI repository and return the path to the .xlsx file."""
    import requests

    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    log.info("Downloading %s", UCI_URL)
    r = requests.get(UCI_URL, timeout=120)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        name = next(n for n in z.namelist() if n.endswith(".xlsx"))
        z.extract(name, dest)
    return dest / name


def load_raw(path: str | Path) -> pd.DataFrame:
    """Load raw transactions from .xlsx (both yearly sheets), .csv or .parquet."""
    path = Path(path)
    if path.suffix == ".xlsx":
        sheets = pd.read_excel(path, sheet_name=None, dtype={"Invoice": str, "StockCode": str})
        df = pd.concat(sheets.values(), ignore_index=True)
    elif path.suffix == ".csv":
        df = pd.read_csv(path, dtype={"Invoice": str, "StockCode": str}, encoding="latin-1")
    elif path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"unsupported file type: {path.suffix}")
    return df.rename(columns=_RENAME)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Return one row per product line with signed revenue.

    - Drops rows without a customer ID (they cannot be tracked over time).
    - Drops fees, postage and manual adjustments.
    - Keeps cancellations (invoices starting with 'C') as negative revenue so that
      customer value is net of returns, and flags them.
    - Removes exact duplicate rows.
    """
    out = df.copy()
    out["Invoice"] = out["Invoice"].astype(str)
    out["StockCode"] = out["StockCode"].astype(str)
    out = out.dropna(subset=["CustomerID"])
    out["CustomerID"] = out["CustomerID"].astype(int)
    out["InvoiceDate"] = pd.to_datetime(out["InvoiceDate"])
    out = out[~out["StockCode"].isin(NON_PRODUCT_CODES)]
    out = out[out["Price"] > 0]
    out = out.drop_duplicates()
    out["is_return"] = out["Invoice"].str.startswith("C")
    out = out[(out["is_return"]) | (out["Quantity"] > 0)]
    out["Revenue"] = out["Quantity"] * out["Price"]
    cols = [
        "Invoice",
        "StockCode",
        "Quantity",
        "InvoiceDate",
        "Price",
        "CustomerID",
        "Country",
        "is_return",
        "Revenue",
    ]
    return out[cols].sort_values("InvoiceDate").reset_index(drop=True)


def orders(tx: pd.DataFrame) -> pd.DataFrame:
    """Aggregate product lines to one row per purchase invoice (returns excluded)."""
    p = tx[~tx["is_return"]]
    return (
        p.groupby(["CustomerID", "Invoice"], as_index=False)
        .agg(
            date=("InvoiceDate", "min"),
            revenue=("Revenue", "sum"),
            items=("Quantity", "sum"),
            lines=("StockCode", "nunique"),
        )
        .sort_values("date")
        .reset_index(drop=True)
    )
