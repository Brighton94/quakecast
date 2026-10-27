"""Checks for hash commitments and scoring functions."""

import json

from quakecast import commit, evaluate


def test_commit_roundtrip(tmp_path):
    """A committed file verifies, and any edit breaks verification."""
    path, digest = commit.commit_forecast({"expected_count": 3.2}, tmp_path)
    assert commit.verify(path)
    data = json.loads(path.read_text())
    data["record"]["forecast"]["expected_count"] = 9.9
    path.write_text(json.dumps(data))
    assert not commit.verify(path)
    assert len(digest) == 64


def test_n_test_rejects_far_off():
    """Observing 30 events against an expectation of 5 is rejected."""
    assert evaluate.n_test(5.0, 30)["rejected"]
    assert not evaluate.n_test(5.0, 5)["rejected"]


def test_brier():
    """Perfect forecasts score zero."""
    assert evaluate.brier_score([1.0, 0.0], [1, 0]) == 0.0
