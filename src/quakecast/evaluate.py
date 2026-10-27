"""Proper scores and CSEP-style consistency tests for count and probability forecasts."""

from __future__ import annotations

import numpy as np
from scipy import stats


def n_test(n_expected: float, n_observed: int) -> dict:
    """CSEP Poisson N-test quantiles; either below 0.025 means the forecast is rejected."""
    delta1 = float(stats.poisson.sf(n_observed - 1, n_expected))
    delta2 = float(stats.poisson.cdf(n_observed, n_expected))
    return {
        "n_expected": float(n_expected),
        "n_observed": int(n_observed),
        "delta1_P_ge_obs": delta1,
        "delta2_P_le_obs": delta2,
        "rejected": bool(min(delta1, delta2) < 0.025),
    }


def poisson_log_score(n_expected: float, n_observed: int) -> float:
    """Log-likelihood of the observed count under the Poisson forecast; higher is better."""
    return float(stats.poisson.logpmf(n_observed, n_expected))


def brier_score(prob: np.ndarray, outcome: np.ndarray) -> float:
    """Mean squared error of binary probability forecasts; lower is better."""
    prob = np.asarray(prob, dtype=float)
    outcome = np.asarray(outcome, dtype=float)
    return float(np.mean((prob - outcome) ** 2))


def information_gain(log_score_model: float, log_score_ref: float, n_events: int) -> float:
    """Information gain per earthquake of a model over a reference such as ETAS."""
    return float((log_score_model - log_score_ref) / max(n_events, 1))
