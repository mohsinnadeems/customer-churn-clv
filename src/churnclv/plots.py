"""Charts, styled consistently for the README and reports."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

INK, MUTED, GRID = "#1C2733", "#5B6775", "#E3E8ED"
PALETTE = {
    "Recency rule": "#9FB4CA",
    "Logistic regression": "#2F5D8A",
    "Gradient boosting": "#C8412B",
    "BG/NBD, no labels": "#D9912B",
    "BG/NBD + Gamma-Gamma": "#2F5D8A",
    "Last 6 months' spend": "#9FB4CA",
    "Gradient boosting (spend)": "#C8412B",
}
SEGMENT_COLOURS = {
    ("High value", "High risk"): "#C8412B",
    ("High value", "Low risk"): "#2F5D8A",
    ("Other", "High risk"): "#E9B7AB",
    ("Other", "Low risk"): "#C9D6E3",
}

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.titleweight": "bold",
        "axes.titlesize": 12,
        "axes.titlelocation": "left",
        "figure.dpi": 150,
        "savefig.bbox": "tight",
    }
)


def _clean(ax, grid="y"):
    ax.spines[["top", "right"]].set_visible(False)
    if grid:
        ax.grid(axis=grid, color=GRID)
        ax.set_axisbelow(True)


def cohort_heatmap(ret: pd.DataFrame, path: Path) -> None:
    data = ret.drop(columns=["customers", 0]).to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(9, 6.5))
    im = ax.imshow(data, cmap="Blues", vmin=0, vmax=0.5, aspect="auto")
    ax.set_xticks(range(data.shape[1]), [str(c) for c in ret.columns[2:]])
    ax.set_yticks(
        range(len(ret)), [f"{p}  ({n:,})" for p, n in zip(ret.index, ret["customers"], strict=True)]
    )
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if not np.isnan(v):
                ax.text(
                    j,
                    i,
                    f"{v:.0%}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if v > 0.3 else INK,
                )
    ax.set_xlabel("Months since first purchase")
    ax.set_title("Share of each monthly cohort buying again")
    ax.spines[:].set_visible(False)
    fig.colorbar(im, ax=ax, fraction=0.03, format=lambda x, _: f"{x:.0%}")
    fig.savefig(path)
    plt.close(fig)


def roc_plot(y: np.ndarray, preds: dict[str, np.ndarray], aucs: dict[str, float], path: Path):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], color=GRID, lw=1.5, ls="--")
    for name, p in preds.items():
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, lw=2, color=PALETTE.get(name), label=f"{name} (AUC {aucs[name]:.3f})")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Churn prediction on held-out period")
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    _clean(ax, grid="both")
    fig.savefig(path)
    plt.close(fig)


def calibration_plot(tables: dict[str, pd.DataFrame], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], color=GRID, lw=1.5, ls="--", label="Perfect calibration")
    for name, t in tables.items():
        ax.plot(
            t["predicted"], t["observed"], marker="o", lw=2, color=PALETTE.get(name), label=name
        )
    ax.set_xlabel("Predicted churn probability (decile mean)")
    ax.set_ylabel("Observed churn rate")
    ax.set_title("Calibration")
    ax.legend(frameon=False, fontsize=8)
    _clean(ax, grid="both")
    fig.savefig(path)
    plt.close(fig)


def importance_plot(imp: pd.Series, path: Path, top: int = 10) -> None:
    labels = {
        "revenue_365d": "Spend, last 12 months",
        "months_active_365d": "Active months, last 12",
        "mean_gap_days": "Typical days between orders",
        "recency_days": "Days since last order",
        "recency_gap_ratio": "Days since last order ÷ typical gap",
        "revenue_90d": "Spend, last 90 days",
        "return_rate": "Return rate",
        "products_365d": "Distinct products, last 12 months",
        "orders_365d": "Orders, last 12 months",
        "spend_trend": "Spend trend (90 days vs 12-month average)",
        "tenure_days": "Customer age (days)",
        "avg_order_value": "Average order value",
        "orders_90d": "Orders, last 90 days",
        "is_uk": "UK customer",
        "orders_total": "Orders, all time",
    }
    s = imp.head(top)[::-1]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.barh([labels.get(i, i) for i in s.index], s.values, color="#2F5D8A")
    ax.set_xlabel("Drop in ROC AUC when shuffled")
    ax.set_title("What drives the churn model")
    _clean(ax, grid="x")
    fig.savefig(path)
    plt.close(fig)


def capture_plot(
    curves: dict[str, tuple[np.ndarray, np.ndarray]],
    path: Path,
    perfect: tuple[np.ndarray, np.ndarray],
) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    ax.plot(*perfect, color=INK, lw=1.2, ls=":", label="Perfect foresight")
    for name, (x, y) in curves.items():
        ax.plot(x, y, lw=2, color=PALETTE.get(name), label=name)
    ax.plot([0, 1], [0, 1], color=GRID, lw=1.5, ls="--", label="Random")
    ax.axvline(0.2, color=MUTED, lw=0.8)
    ax.text(0.21, 0.05, "Top 20% of customers", color=MUTED, fontsize=8)
    ax.set_xlabel("Share of customers, ranked by predicted value")
    ax.set_ylabel("Share of actual future revenue")
    ax.set_title("Who will be valuable over the next six months?")
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    _clean(ax, grid="both")
    fig.savefig(path)
    plt.close(fig)


def segment_plot(seg: pd.DataFrame, path: Path) -> None:
    """2x2 grid: churn risk (columns) by value (rows)."""
    fig, ax = plt.subplots(figsize=(7, 4.8))
    colours = SEGMENT_COLOURS
    for (value, risk), row in seg.iterrows():
        x = 1 if risk == "High risk" else 0
        y = 1 if value == "High value" else 0
        c = colours[(value, risk)]
        ax.add_patch(plt.Rectangle((x, y), 0.97, 0.95, color=c))
        dark = c in ("#C8412B", "#2F5D8A")
        tc = "white" if dark else INK
        ax.text(x + 0.05, y + 0.78, row["name"], color=tc, fontsize=11, weight="bold")
        ax.text(x + 0.05, y + 0.5, f"{row['customers']:,} customers", color=tc, fontsize=9)
        ax.text(
            x + 0.05, y + 0.32, f"£{row['spend_12m']:,.0f} spent in past year", color=tc, fontsize=9
        )
        ax.text(
            x + 0.05, y + 0.14, f"{row['churn_rate']:.0%} actually churned", color=tc, fontsize=9
        )
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_xticks(
        [0.5, 1.5], ["Less likely to churn (within tier)", "More likely to churn (within tier)"]
    )
    ax.set_yticks([0.5, 1.5], ["Other customers", "Top 25% by\npast-year spend"])
    ax.tick_params(length=0)
    ax.spines[:].set_visible(False)
    ax.set_title("Where to focus retention effort")
    fig.savefig(path)
    plt.close(fig)


def backtest_plot(bt: pd.DataFrame, path: Path) -> None:
    """Top: ROC AUC per test month. Bottom: predicted vs actual churn rate per test month."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True, height_ratios=[3, 2])
    for name, g in bt.groupby("model", sort=False):
        ax1.plot(
            g["test_cutoff"],
            g["roc_auc"],
            marker="o",
            ms=4,
            lw=2,
            color=PALETTE.get(name),
            label=name,
        )
    ax1.set_ylabel("ROC AUC")
    ax1.set_title("Churn model performance across test months")
    ax1.legend(frameon=False, fontsize=8, ncol=2, loc="lower left")
    _clean(ax1)

    actual = bt.drop_duplicates("test_cutoff")
    ax2.plot(
        actual["test_cutoff"],
        actual["churn_rate"],
        color=INK,
        lw=2,
        ls=":",
        marker="o",
        ms=4,
        label="Actual churn rate",
    )
    for name in ("Logistic regression", "Gradient boosting"):
        g = bt[bt["model"] == name]
        ax2.plot(
            g["test_cutoff"],
            g["mean_predicted"],
            lw=2,
            marker="o",
            ms=4,
            color=PALETTE.get(name),
            label=f"Predicted, {name.lower()}",
        )
    ax2.set_ylabel("Churn rate, next 90 days")
    ax2.set_xlabel("Test cutoff (features up to this date, label over the following 90 days)")
    ax2.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax2.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b %Y"))
    ax2.legend(frameon=False, fontsize=8)
    _clean(ax2)
    fig.autofmt_xdate()
    fig.savefig(path)
    plt.close(fig)
