import pytest

from gwbackpop.inference.hierarchical import (
    handle_selection_mc_quality_gates,
    validate_selection_mc_diagnostics,
)


def _row(label="default_pre_nuts", ess=200.0, frac=0.01, top1=0.2, top0p1=0.1):
    return {
        "hyperpoint": label,
        "injection_ess": ess,
        "ess_fraction": frac,
        "top_1pct_alpha_fraction": top1,
        "top_0p1pct_alpha_fraction": top0p1,
    }


def test_bad_selection_diagnostics_raise_by_default():
    rows = [_row(ess=3.0, frac=3.0 / 61519.0, top1=1.0, top0p1=1.0)]
    with pytest.raises(ValueError, match="Selection-MC quality gates failed"):
        handle_selection_mc_quality_gates(rows)


def test_allow_bad_selection_mc_continues_but_marks_invalid():
    rows = [_row(ess=3.0, frac=3.0 / 61519.0, top1=1.0, top0p1=1.0)]
    valid, reason = handle_selection_mc_quality_gates(rows, allow_bad_selection_mc=True)
    assert valid is False
    assert "selection ESS" in reason
    assert "top 1% alpha fraction" in reason


def test_good_selection_diagnostics_pass():
    rows = [_row()]
    valid, reason = validate_selection_mc_diagnostics(rows)
    assert valid is True
    assert reason == ""
