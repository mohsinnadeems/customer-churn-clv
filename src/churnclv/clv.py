"""Customer lifetime value with BG/NBD (purchase frequency) and Gamma-Gamma (spend).

Implemented from the original papers rather than a library:
- Fader, Hardie and Lee (2005), "Counting Your Customers the Easy Way: An Alternative to
  the Pareto/NBD Model", Marketing Science 24(2).
- Fader, Hardie and Lee (2005), "RFM and CLV: Using Iso-Value Curves for Customer Base
  Analysis", Journal of Marketing Research 42(4); Gamma-Gamma spend model per Fader and
  Hardie's note "The Gamma-Gamma Model of Monetary Value" (2013).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import betaln, gammaln, hyp2f1

from .data import orders as to_orders


def rfm_summary(tx: pd.DataFrame, cutoff: str | pd.Timestamp, unit_days: float = 7.0):
    """Per-customer (frequency, recency, T, monetary) as of `cutoff`, in weeks by default.

    frequency: number of repeat purchase days (distinct purchase days minus one)
    recency:   time between first and last purchase
    T:         time between first purchase and cutoff
    monetary:  mean revenue per repeat purchase day (0 for one-time buyers)
    """
    cutoff = pd.Timestamp(cutoff)
    o = to_orders(tx[tx["InvoiceDate"] < cutoff])
    o["day"] = o["date"].dt.normalize()
    daily = o.groupby(["CustomerID", "day"], as_index=False)["revenue"].sum()
    daily = daily[daily["revenue"] > 0]
    g = daily.groupby("CustomerID")
    first, last = g["day"].min(), g["day"].max()
    s = pd.DataFrame(
        {
            "frequency": g.size() - 1,
            "recency": (last - first).dt.days / unit_days,
            "T": (cutoff.normalize() - first).dt.days / unit_days,
        }
    )
    first_day = daily.merge(first.rename("first"), left_on="CustomerID", right_index=True)
    repeat = first_day[first_day["day"] > first_day["first"]]
    s["monetary"] = repeat.groupby("CustomerID")["revenue"].mean()
    s["monetary"] = s["monetary"].fillna(0.0)
    return s[s["T"] > 0]


class BetaGeo:
    """BG/NBD model. Parameters r, alpha (purchase rate) and a, b (dropout)."""

    def __init__(self, penalizer: float = 0.0):
        self.penalizer = penalizer
        self.params_: dict | None = None

    @staticmethod
    def _loglik(params, x, tx, T):
        r, alpha, a, b = params
        a1 = gammaln(r + x) - gammaln(r) + r * np.log(alpha)
        a2 = betaln(a, b + x) - betaln(a, b)
        a3 = -(r + x) * np.log(alpha + T)
        with np.errstate(divide="ignore", invalid="ignore"):
            a4 = np.log(a) - np.log(b + x - 1) - (r + x) * np.log(alpha + tx)
        a4 = np.where(x > 0, a4, -np.inf)
        return a1 + a2 + np.logaddexp(a3, a4)

    def fit(self, frequency, recency, T):
        x, tx, T = (np.asarray(v, dtype=float) for v in (frequency, recency, T))
        if np.any(tx > T):
            raise ValueError("recency cannot exceed T")

        def nll(logp):
            p = np.exp(logp)
            return -self._loglik(p, x, tx, T).mean() + self.penalizer * np.sum(p**2)

        res = minimize(
            nll,
            x0=np.log([0.5, 5.0, 0.5, 2.0]),
            method="Nelder-Mead",
            options={"maxiter": 10000, "xatol": 1e-7, "fatol": 1e-10},
        )
        r, alpha, a, b = np.exp(res.x)
        self.params_ = {"r": r, "alpha": alpha, "a": a, "b": b}
        self.converged_ = bool(res.success)
        return self

    def prob_alive(self, frequency, recency, T):
        p = self.params_
        x, tx, T = (np.asarray(v, dtype=float) for v in (frequency, recency, T))
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = (p["a"] / (p["b"] + x - 1)) * ((p["alpha"] + T) / (p["alpha"] + tx)) ** (
                p["r"] + x
            )
        return np.where(x > 0, 1.0 / (1.0 + ratio), 1.0)

    def expected_purchases(self, t, frequency, recency, T):
        """Expected number of purchases in the next `t` periods, per customer."""
        p = self.params_
        r, alpha, a, b = p["r"], p["alpha"], p["a"], p["b"]
        x, tx, T = (np.asarray(v, dtype=float) for v in (frequency, recency, T))
        z = t / (alpha + T + t)
        hyp = hyp2f1(r + x, b + x, a + b + x - 1, z)
        num = (a + b + x - 1) / (a - 1) * (1 - ((alpha + T) / (alpha + T + t)) ** (r + x) * hyp)
        return num * self.prob_alive(x, tx, T)


class GammaGamma:
    """Gamma-Gamma model of spend per transaction. Fit on repeat customers only."""

    def __init__(self, penalizer: float = 0.0):
        self.penalizer = penalizer
        self.params_: dict | None = None

    def fit(self, frequency, monetary):
        x, m = np.asarray(frequency, float), np.asarray(monetary, float)
        keep = (x > 0) & (m > 0)
        x, m = x[keep], m[keep]

        def nll(logp):
            p, q, v = np.exp(logp)
            ll = (
                gammaln(p * x + q)
                - gammaln(p * x)
                - gammaln(q)
                + q * np.log(v)
                + (p * x - 1) * np.log(m)
                + (p * x) * np.log(x)
                - (p * x + q) * np.log(x * m + v)
            )
            return -ll.mean() + self.penalizer * np.sum(np.exp(logp) ** 2)

        res = minimize(
            nll,
            x0=np.log([1.0, 2.0, 100.0]),
            method="Nelder-Mead",
            options={"maxiter": 10000, "xatol": 1e-7, "fatol": 1e-10},
        )
        p, q, v = np.exp(res.x)
        self.params_ = {"p": p, "q": q, "v": v}
        self.converged_ = bool(res.success)
        return self

    def expected_spend(self, frequency, monetary):
        """Shrinks each customer's observed average towards the population mean."""
        p, q, v = (self.params_[k] for k in ("p", "q", "v"))
        x, m = np.asarray(frequency, float), np.asarray(monetary, float)
        pop_mean = v * p / (q - 1)
        w = p * x / (p * x + q - 1)
        return np.where(x > 0, (1 - w) * pop_mean + w * m, pop_mean)


def predict_clv(summary: pd.DataFrame, bg: BetaGeo, gg: GammaGamma, periods: float):
    """Expected revenue over the next `periods` (undiscounted, short horizon)."""
    e_n = bg.expected_purchases(periods, summary["frequency"], summary["recency"], summary["T"])
    e_m = gg.expected_spend(summary["frequency"], summary["monetary"])
    return pd.Series(e_n * e_m, index=summary.index, name="clv")


def capture_curve(actual: np.ndarray, predicted: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Share of actual revenue captured when customers are ranked by predicted value."""
    order = np.argsort(-predicted)
    cum = np.cumsum(actual[order]) / actual.sum()
    frac = np.arange(1, len(actual) + 1) / len(actual)
    return frac, cum


def revenue_capture(actual: np.ndarray, predicted: np.ndarray, top: float = 0.2) -> float:
    frac, cum = capture_curve(actual, predicted)
    return float(cum[int(np.ceil(top * len(actual))) - 1])
