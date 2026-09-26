import numpy as np
import pandas as pd

from churnclv.clv import BetaGeo, GammaGamma, revenue_capture, rfm_summary


def simulate_bgnbd(n, r, alpha, a, b, T, seed=0):
    """Simulate the BG/NBD generative story to check parameter recovery."""
    rng = np.random.default_rng(seed)
    lam = rng.gamma(r, 1 / alpha, n)
    p = rng.beta(a, b, n)
    x, tx = np.zeros(n), np.zeros(n)
    for i in range(n):
        t, alive = 0.0, True
        while alive:
            t += rng.exponential(1 / lam[i])
            if t > T:
                break
            x[i] += 1
            tx[i] = t
            alive = rng.uniform() > p[i]
    return x, tx, np.full(n, T)


def test_bgnbd_recovers_parameters():
    true = dict(r=0.8, alpha=6.0, a=0.6, b=3.0)
    x, tx, T = simulate_bgnbd(4000, T=52, **true)
    fit = BetaGeo().fit(x, tx, T).params_
    assert abs(fit["r"] - true["r"]) / true["r"] < 0.25
    assert abs(fit["alpha"] - true["alpha"]) / true["alpha"] < 0.3


def test_prob_alive_behaviour():
    bg = BetaGeo()
    bg.params_ = dict(r=0.8, alpha=6.0, a=0.6, b=3.0)
    # One-time buyers are "alive" by definition under BG/NBD.
    assert bg.prob_alive([0], [0], [52])[0] == 1.0
    # A frequent buyer who went quiet long ago is less likely alive than one who bought recently.
    quiet, recent = bg.prob_alive([10, 10], [10, 50], [52, 52])
    assert quiet < recent


def test_expected_purchases_increase_with_horizon():
    bg = BetaGeo()
    bg.params_ = dict(r=0.8, alpha=6.0, a=0.6, b=3.0)
    e = [bg.expected_purchases(t, 5, 40, 52) for t in (4, 13, 26)]
    assert e[0] < e[1] < e[2]


def test_gamma_gamma_shrinks_towards_mean():
    rng = np.random.default_rng(0)
    x = rng.integers(1, 10, 2000)
    m = rng.gamma(3, 50, 2000)
    gg = GammaGamma().fit(x, m)
    pop = gg.expected_spend([0], [0])[0]
    few, many = gg.expected_spend([1, 50], [1000, 1000])
    assert abs(few - pop) < abs(many - pop)  # more history, less shrinkage


def test_rfm_summary(synthetic_tx):
    s = rfm_summary(synthetic_tx, "2010-06-01")
    assert (s["recency"] <= s["T"]).all() and (s["frequency"] >= 0).all()
    assert (s.loc[s["frequency"] == 0, "monetary"] == 0).all()


def test_revenue_capture():
    actual = np.array([100.0, 0, 0, 0, 0])
    assert revenue_capture(actual, np.array([5.0, 1, 1, 1, 1]), 0.2) == 1.0
    assert revenue_capture(actual, pd.Series([0.0, 5, 4, 3, 2]).to_numpy(), 0.2) == 0.0
