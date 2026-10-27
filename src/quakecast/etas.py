"""Temporal ETAS model: the baseline any ML forecaster has to beat."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.optimize import minimize


@dataclass
class ETASParams:
    """Background rate mu (per day), productivity K and alpha, Omori c (days) and p."""

    mu: float
    K: float
    alpha: float
    c: float
    p: float

    def to_dict(self) -> dict:
        """Return the parameters as a plain dict."""
        return asdict(self)


def _productivity(mags: np.ndarray, p: ETASParams, m0: float) -> np.ndarray:
    """Expected direct aftershocks scale per parent event."""
    return p.K * np.exp(p.alpha * (mags - m0))


def _omori_integral(dt: np.ndarray, c: float, p: float) -> np.ndarray:
    """Integral of (s + c)^-p from 0 to dt."""
    if abs(p - 1.0) < 1e-8:
        return np.log1p(dt / c)
    return (c ** (1.0 - p) - (dt + c) ** (1.0 - p)) / (p - 1.0)


def intensity(
    t: np.ndarray, times: np.ndarray, mags: np.ndarray, p: ETASParams, m0: float
) -> np.ndarray:
    """Conditional intensity at each query time given all strictly earlier events."""
    t = np.atleast_1d(t).astype(float)
    k = _productivity(mags, p, m0)
    out = np.full(t.shape, p.mu)
    for idx, ti in enumerate(t):
        past = times < ti
        if past.any():
            out[idx] += np.sum(k[past] * (ti - times[past] + p.c) ** (-p.p))
    return out


def log_likelihood(
    p: ETASParams,
    times: np.ndarray,
    mags: np.ndarray,
    t_start: float,
    t_end: float,
    m0: float,
    block: int = 512,
) -> float:
    """Point-process log-likelihood of events in [t_start, t_end] with full history as triggers."""
    k = _productivity(mags, p, m0)
    target = np.flatnonzero((times >= t_start) & (times <= t_end))
    log_sum = 0.0
    for lo in range(0, target.size, block):
        rows = target[lo : lo + block]
        dt = times[rows][:, None] - times[None, :]
        mask = dt > 0
        trig = np.where(mask, k[None, :] * (np.where(mask, dt, 1.0) + p.c) ** (-p.p), 0.0)
        log_sum += np.sum(np.log(p.mu + trig.sum(axis=1)))
    parents = times < t_end
    a = np.maximum(t_start - times[parents], 0.0)
    b = t_end - times[parents]
    integral = p.mu * (t_end - t_start) + np.sum(
        k[parents] * (_omori_integral(b, p.c, p.p) - _omori_integral(a, p.c, p.p))
    )
    return float(log_sum - integral)


def _unpack(x: np.ndarray) -> ETASParams:
    """Map unconstrained optimiser variables to valid ETAS parameters."""
    return ETASParams(
        mu=float(np.exp(x[0])),
        K=float(np.exp(x[1])),
        alpha=float(np.exp(x[2])),
        c=float(np.exp(x[3])),
        p=float(1.0 + np.exp(x[4])),
    )


def fit(
    times: np.ndarray,
    mags: np.ndarray,
    t_start: float,
    t_end: float,
    m0: float,
    init: ETASParams | None = None,
) -> tuple[ETASParams, float]:
    """Maximum-likelihood ETAS fit; returns parameters and the maximised log-likelihood."""
    n_target = np.sum((times >= t_start) & (times <= t_end))
    init = init or ETASParams(
        mu=max(n_target, 1) / (t_end - t_start) * 0.5, K=0.02, alpha=1.5, c=0.01, p=1.1
    )
    x0 = np.log([init.mu, init.K, init.alpha, init.c, max(init.p - 1.0, 1e-3)])

    def nll(x: np.ndarray) -> float:
        """Negative log-likelihood with a guard against numerical blow-ups."""
        val = log_likelihood(_unpack(x), times, mags, t_start, t_end, m0)
        return -val if np.isfinite(val) else 1e12

    res = minimize(
        nll, x0, method="Nelder-Mead", options={"maxiter": 4000, "xatol": 1e-4, "fatol": 1e-4}
    )
    return _unpack(res.x), float(-res.fun)


def branching_ratio(p: ETASParams, b: float, horizon_days: float = 3652.5) -> float:
    """Mean direct offspring per event within a horizon; below 1 means a stable process."""
    beta = b * np.log(10.0)
    if p.alpha >= beta:
        return float("inf")
    omori = float(_omori_integral(np.array(horizon_days), p.c, p.p))
    return float(p.K * beta / (beta - p.alpha) * omori)


def expected_count(
    p: ETASParams, times: np.ndarray, mags: np.ndarray, t0: float, t1: float, m0: float
) -> float:
    """Expected M>=m0 count in [t0, t1] from background and prior events, no in-window cascades."""
    past = times < t0
    k = _productivity(mags[past], p, m0)
    a = t0 - times[past]
    b = t1 - times[past]
    trig = np.sum(k * (_omori_integral(b, p.c, p.p) - _omori_integral(a, p.c, p.p)))
    return float(p.mu * (t1 - t0) + trig)


def b_value(mags: np.ndarray, m0: float, dm: float = 0.1) -> float:
    """Aki-Utsu maximum-likelihood Gutenberg-Richter b-value with binning correction."""
    m = mags[mags >= m0]
    return float(np.log10(np.e) / (m.mean() - (m0 - dm / 2.0)))


def prob_at_least_one(n_expected: float, m0: float, m_target: float, b: float) -> float:
    """Poisson probability of at least one M>=m_target event given an M>=m0 expected count."""
    n_target = n_expected * 10.0 ** (-b * (m_target - m0))
    return float(1.0 - np.exp(-n_target))
