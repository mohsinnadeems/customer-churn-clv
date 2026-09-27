"""End-to-end analysis: churnclv download | run."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor

from . import plots
from .backtest import backtest, monthly_cutoffs, summarise, to_records
from .churn import calibration_table, evaluate, fit_and_evaluate, importance
from .clv import (
    BetaGeo,
    GammaGamma,
    bgnbd_churn_score,
    capture_curve,
    predict_clv,
    revenue_capture,
    rfm_summary,
)
from .cohorts import retention_matrix
from .data import clean, download_uci, load_raw
from .features import FEATURES, snapshot, training_set

log = logging.getLogger("churnclv")

# Study design. Training labels end before the test period starts, so nothing leaks.
TRAIN_CUTOFFS = [str(d.date()) for d in pd.date_range("2010-03-01", "2011-06-01", freq="MS")]
TEST_CUTOFF = "2011-09-01"
HORIZON_DAYS = 90
CLV_CUTOFF = "2011-06-09"  # six-month holdout to the end of the data
CLV_TRAIN_CUTOFFS = ["2010-05-09", "2010-06-09", "2010-07-09"]
BACKTEST_START = TRAIN_CUTOFFS[0]  # first monthly cutoff; folds need 4+ earlier cutoffs


def churn_analysis(tx: pd.DataFrame, figs: Path) -> tuple[dict, pd.DataFrame, str]:
    train = training_set(tx, TRAIN_CUTOFFS, HORIZON_DAYS)
    test = snapshot(tx, TEST_CUTOFF, HORIZON_DAYS)
    log.info("Churn: %d training rows, %d test customers", len(train), len(test))
    results, preds, fitted = fit_and_evaluate(train, test)
    y = test["churned"].to_numpy()

    # Probabilistic comparison: BG/NBD expected purchases in the horizon (fewer = riskier).
    preds["BG/NBD, no labels"] = bgnbd_churn_score(
        tx, TEST_CUTOFF, test["CustomerID"], HORIZON_DAYS
    )
    results["BG/NBD, no labels"] = evaluate(y, preds["BG/NBD, no labels"], probabilistic=False)

    best = max(("Logistic regression", "Gradient boosting"), key=lambda m: results[m]["roc_auc"])
    aucs = {k: v["roc_auc"] for k, v in results.items()}
    plots.roc_plot(y, preds, aucs, figs / "churn_roc.png")
    plots.calibration_plot(
        {m: calibration_table(y, preds[m]) for m in ("Logistic regression", "Gradient boosting")},
        figs / "churn_calibration.png",
    )
    imp = importance(fitted["Gradient boosting"], test[FEATURES], y)
    plots.importance_plot(imp, figs / "churn_importance.png")

    test = test.assign(churn_prob=preds[best])
    summary = {
        "train_rows": len(train),
        "test_customers": len(test),
        "test_churn_rate": round(float(y.mean()), 4),
        "train_churn_rate": round(float(train["churned"].mean()), 4),
        "mean_predicted_churn": round(float(preds[best].mean()), 4),
        "best_model": best,
        "models": results,
        "importance": imp.round(4).to_dict(),
    }
    return summary, test, best


def backtest_analysis(tx: pd.DataFrame, figs: Path) -> dict:
    bt = backtest(tx, monthly_cutoffs(tx, BACKTEST_START, HORIZON_DAYS), HORIZON_DAYS)
    plots.backtest_plot(bt, figs / "backtest.png")
    return {
        "folds": int(bt["test_cutoff"].nunique()),
        "first_test_cutoff": str(bt["test_cutoff"].min().date()),
        "last_test_cutoff": str(bt["test_cutoff"].max().date()),
        "summary": summarise(bt),
        "by_fold": to_records(bt),
    }


def clv_analysis(tx: pd.DataFrame, figs: Path) -> dict:
    cut = pd.Timestamp(CLV_CUTOFF)
    end = tx["InvoiceDate"].max()
    weeks = (end.normalize() - cut).days / 7
    s = rfm_summary(tx, cut)
    bg = BetaGeo().fit(s["frequency"], s["recency"], s["T"])
    gg = GammaGamma().fit(s["frequency"], s["monetary"])
    clv = predict_clv(s, bg, gg, weeks).to_numpy()

    hold = tx[(tx["InvoiceDate"] >= cut) & ~tx["is_return"]].groupby("CustomerID")["Revenue"].sum()
    actual = s.index.map(hold).fillna(0).to_numpy()
    last6 = (
        tx[
            (tx["InvoiceDate"] < cut)
            & (tx["InvoiceDate"] >= cut - pd.Timedelta(days=183))
            & ~tx["is_return"]
        ]
        .groupby("CustomerID")["Revenue"]
        .sum()
    )
    naive = s.index.map(last6).fillna(0).to_numpy()

    # Supervised alternative: learn next-six-month spend from the same season a year earlier.
    horizon = (end.normalize() - cut).days + 1
    tr = pd.concat([snapshot(tx, c, horizon, active_days=10_000) for c in CLV_TRAIN_CUTOFFS])
    te = snapshot(tx, cut, horizon, active_days=10_000, with_label=False).set_index("CustomerID")
    reg = HistGradientBoostingRegressor(
        loss="poisson",
        learning_rate=0.05,
        max_iter=300,
        max_leaf_nodes=15,
        min_samples_leaf=50,
        random_state=42,
    )
    reg.fit(tr[FEATURES], tr["future_revenue"].clip(lower=0))
    ml = pd.Series(reg.predict(te[FEATURES]), index=te.index).reindex(s.index).fillna(0).to_numpy()

    models = {
        "BG/NBD + Gamma-Gamma": clv,
        "Last 6 months' spend": naive,
        "Gradient boosting (spend)": ml,
    }
    out = {}
    for name, pred in models.items():
        out[name] = {
            "mae": round(float(np.mean(np.abs(pred - actual))), 1),
            "spearman": round(float(spearmanr(pred, actual)[0]), 4),
            "top20_revenue_capture": round(revenue_capture(actual, pred, 0.2), 4),
            "predicted_total": round(float(pred.sum())),
        }
    plots.capture_plot(
        {k: capture_curve(actual, v) for k, v in models.items()},
        figs / "clv_capture.png",
        perfect=capture_curve(actual, actual),
    )
    return {
        "cutoff": CLV_CUTOFF,
        "holdout_weeks": round(weeks, 1),
        "customers": len(s),
        "actual_total": round(float(actual.sum())),
        "bgnbd_params": {k: round(float(v), 4) for k, v in bg.params_.items()},
        "gamma_gamma_params": {k: round(float(v), 4) for k, v in gg.params_.items()},
        "models": out,
    }


SEGMENT_NAMES = {
    ("High value", "High risk"): "Act now",
    ("High value", "Low risk"): "Keep happy",
    ("Other", "High risk"): "Low-cost reactivation",
    ("Other", "Low risk"): "Business as usual",
}


def segment(scored: pd.DataFrame) -> pd.DataFrame:
    """Combine model churn risk with each customer's spend over the past 12 months.

    Historical value is used rather than predicted CLV, because predicted CLV already
    discounts customers who look likely to churn, which would empty the high-value,
    high-risk group by construction.
    """
    df = scored.copy()
    df["value"] = np.where(
        df["revenue_365d"] >= df["revenue_365d"].quantile(0.75), "High value", "Other"
    )
    # High spenders rarely churn, so a single risk threshold would leave almost none of them
    # flagged. Instead, rank risk within each value tier: the riskier half of each tier is
    # "high risk". This also makes the split robust to seasonal shifts in the base rate.
    tier_median = df.groupby("value")["churn_prob"].transform("median")
    df["risk"] = np.where(df["churn_prob"] >= tier_median, "High risk", "Low risk")
    df["segment"] = [SEGMENT_NAMES[(v, r)] for v, r in zip(df["value"], df["risk"], strict=True)]
    return df


def run(data: Path, out: Path, with_backtest: bool = True) -> dict:
    figs = out / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    raw = load_raw(data)
    tx = clean(raw)
    log.info(
        "Loaded %d rows, %d after cleaning, %d customers",
        len(raw),
        len(tx),
        tx["CustomerID"].nunique(),
    )

    ret = retention_matrix(tx[tx["InvoiceDate"] < "2011-12-01"])
    plots.cohort_heatmap(ret.iloc[:13], figs / "cohort_retention.png")

    churn, scored, _ = churn_analysis(tx, figs)
    clv = clv_analysis(tx, figs)
    bt = backtest_analysis(tx, figs) if with_backtest else None

    seg = segment(scored)
    table = seg.groupby(["value", "risk"]).agg(
        customers=("CustomerID", "size"),
        spend_12m=("revenue_365d", "sum"),
        churn_rate=("churned", "mean"),
        actual_revenue=("future_revenue", "sum"),
    )
    table["name"] = [SEGMENT_NAMES[i] for i in table.index]
    plots.segment_plot(table, figs / "segments.png")

    metrics = {
        "data": {
            "raw_rows": len(raw),
            "clean_rows": len(tx),
            "customers": int(tx["CustomerID"].nunique()),
            "start": str(tx["InvoiceDate"].min().date()),
            "end": str(tx["InvoiceDate"].max().date()),
        },
        "cohorts": {"month1_retention_median": round(float(ret[1].median()), 4)},
        "churn": churn,
        "clv": clv,
        "segments": table.reset_index().round(4).to_dict(orient="records"),
    }
    if bt is not None:
        metrics["backtest"] = bt
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    log.info("Wrote %s", out / "metrics.json")
    return metrics


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser(prog="churnclv")
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download", help="Download the UCI dataset into data/")
    d.add_argument("--dest", default="data")
    r = sub.add_parser("run", help="Run the full analysis")
    r.add_argument("--data", default="data/online_retail_II.xlsx")
    r.add_argument("--out", default="reports")
    r.add_argument(
        "--skip-backtest", action="store_true", help="Skip the walk-forward backtest (faster)"
    )
    a = p.parse_args(argv)
    if a.cmd == "download":
        print(download_uci(a.dest))
    else:
        run(Path(a.data), Path(a.out), with_backtest=not a.skip_backtest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
