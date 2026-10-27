"""Checks that the ETAS likelihood and fit behave on synthetic data."""

import numpy as np

from quakecast import etas


def _simulate(p: etas.ETASParams, t_end: float, m0: float, b: float, seed: int = 0):
    """Branching simulation of a temporal ETAS catalog."""
    rng = np.random.default_rng(seed)
    beta = b * np.log(10.0)
    n_bg = rng.poisson(p.mu * t_end)
    gen = list(zip(rng.uniform(0, t_end, n_bg), m0 + rng.exponential(1 / beta, n_bg), strict=True))
    events = list(gen)
    while gen:
        nxt = []
        for t, m in gen:
            n_kids = rng.poisson(p.K * np.exp(p.alpha * (m - m0)) * p.c ** (1 - p.p) / (p.p - 1))
            u = rng.uniform(size=n_kids)
            dt = p.c * ((1 - u) ** (1 / (1 - p.p)) - 1)
            for d in dt:
                if t + d < t_end:
                    nxt.append((t + d, m0 + rng.exponential(1 / beta)))
        events += nxt
        gen = nxt
    events.sort()
    arr = np.array(events)
    return arr[:, 0], arr[:, 1]


TRUE = etas.ETASParams(mu=0.5, K=0.02, alpha=1.2, c=0.01, p=1.15)


def test_likelihood_prefers_truth():
    """True parameters should score higher than a badly wrong set."""
    t, m = _simulate(TRUE, 1000.0, 3.0, 1.0)
    bad = etas.ETASParams(mu=2.0, K=0.001, alpha=0.5, c=0.5, p=1.5)
    assert etas.log_likelihood(TRUE, t, m, 0, 1000, 3.0) > etas.log_likelihood(
        bad, t, m, 0, 1000, 3.0
    )


def test_fit_recovers_background_rate():
    """Fitted background rate should land near the simulated one."""
    t, m = _simulate(TRUE, 1500.0, 3.0, 1.0, seed=1)
    fitted, _ = etas.fit(t, m, 0.0, 1500.0, 3.0)
    assert 0.3 < fitted.mu < 0.8
    assert etas.branching_ratio(fitted, 1.0) < 1.0


def test_b_value_close_to_one():
    """Aki estimator should recover b on a pure Gutenberg-Richter sample."""
    rng = np.random.default_rng(2)
    mags = np.round(3.0 + rng.exponential(1 / np.log(10), 20000) - 0.05, 1)
    assert abs(etas.b_value(mags, 3.0) - 1.0) < 0.05


def test_expected_count_background_only():
    """With no history the expected count is mu times window length."""
    p = etas.ETASParams(mu=0.4, K=0.1, alpha=1.0, c=0.01, p=1.1)
    assert np.isclose(etas.expected_count(p, np.array([]), np.array([]), 0, 10, 3.0), 4.0)
