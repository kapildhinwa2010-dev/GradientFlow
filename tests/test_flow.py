import numpy as np
import pytest

from gradientflow import flow_report, gradient_health


def test_health_low_risk_for_identity():
    h = gradient_health(np.eye(8))
    assert h["risk"] == "low"
    assert h["condition_number"] == pytest.approx(1.0, abs=1e-9)


def test_health_high_risk_for_ill_conditioned():
    rng = np.random.default_rng(0)
    u, _, vt = np.linalg.svd(rng.standard_normal((16, 16)), full_matrices=False)
    ranks = np.arange(1, 17)
    W = (u * (1.0 / ranks ** 8)) @ vt
    h = gradient_health(W)
    assert h["risk"] == "high"


def test_health_rejects_1d():
    with pytest.raises(ValueError):
        gradient_health(np.ones(4))


def test_flow_report_ranks_highest_risk():
    layers = {"good": np.eye(8), "bad": np.ones((4, 4))}
    out = flow_report(layers)
    assert out["highest_risk"] == "bad"
    assert set(out) == {"good", "bad", "highest_risk"}


def test_flow_report_rejects_empty():
    with pytest.raises(ValueError):
        flow_report({})
