import numpy as np
import pytest

from gwbackpop.inference.single_event import _combined_support_gate_penalty, _support_gate_penalty


@pytest.mark.parametrize("value", [-1.0, 0.5, 2.0])
def test_support_gate_none_delegates_all_tails_to_kde(value):
    assert _support_gate_penalty(value, (0.0, 1.0), "none") is None


@pytest.mark.parametrize("value", [-0.1, 1.1])
def test_support_gate_hard_rejects_below_and_above_bounds(value):
    assert _support_gate_penalty(value, (0.0, 1.0), "hard") == -np.inf


def test_support_gate_hard_allows_within_bounds():
    assert _support_gate_penalty(0.5, (0.0, 1.0), "hard") is None


@pytest.mark.parametrize("value", [-0.01, 1.01])
def test_support_gate_soft_penalizes_below_and_above_symmetrically(value):
    penalty = _support_gate_penalty(value, (0.0, 1.0), "soft")
    assert np.isfinite(penalty)
    assert penalty == pytest.approx(-100.5)


def test_support_gate_soft_allows_within_bounds():
    assert _support_gate_penalty(0.5, (0.0, 1.0), "soft") is None


def test_combined_support_gate_sums_multiple_out_of_bounds_penalties():
    penalty = _combined_support_gate_penalty(
        {"mc": -0.01, "q": 1.01},
        {"mc": (0.0, 1.0), "q": (0.0, 1.0)},
        "soft",
    )
    assert penalty == pytest.approx(-201.0)
