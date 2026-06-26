import ast
import math
from pathlib import Path

import pytest


class _NPShim:
    log = staticmethod(math.log)


def _load_lvk_found_subsample_log_scaling():
    """Load the pure helper so tests do not require JAX/NumPyro deps."""
    source = Path(__file__).resolve().parents[1] / "hierarchical_backpop_jax.py"
    module_ast = ast.parse(source.read_text())
    helper = next(
        node for node in module_ast.body
        if isinstance(node, ast.FunctionDef) and node.name == "lvk_found_subsample_log_scaling"
    )
    mod = ast.Module(body=[helper], type_ignores=[])
    ast.fix_missing_locations(mod)
    namespace = {"np": _NPShim}
    exec(compile(mod, str(source), "exec"), namespace)
    return namespace["lvk_found_subsample_log_scaling"]


lvk_found_subsample_log_scaling = _load_lvk_found_subsample_log_scaling()


def _alpha_from_found_contributions(contrib, idx=None):
    if idx is None:
        idx = range(len(contrib))
    log_scale = lvk_found_subsample_log_scaling(len(contrib), len(idx))
    return math.exp(log_scale) * sum(contrib[i] for i in idx)


def test_lvk_subsampling_all_found_matches_no_subsampling():
    found_contrib = [0.5, 1.25, 2.0, 4.25, 8.0]

    alpha_full = _alpha_from_found_contributions(found_contrib)
    alpha_all_subsampled = _alpha_from_found_contributions(
        found_contrib,
        idx=[4, 2, 0, 3, 1],
    )

    assert alpha_all_subsampled == pytest.approx(alpha_full)
    assert lvk_found_subsample_log_scaling(5, 5) == pytest.approx(0.0)


def test_lvk_uniform_subsample_gets_total_over_used_scaling():
    found_contrib = [3.5] * 10
    idx_sub = [0, 3, 6, 9]

    alpha_full = _alpha_from_found_contributions(found_contrib)
    alpha_sub = _alpha_from_found_contributions(found_contrib, idx=idx_sub)

    assert math.exp(lvk_found_subsample_log_scaling(10, 4)) == pytest.approx(2.5)
    assert alpha_sub == pytest.approx(alpha_full)


def test_lvk_subsample_log_scaling_rejects_invalid_counts():
    with pytest.raises(ValueError, match="positive"):
        lvk_found_subsample_log_scaling(0, 1)
    with pytest.raises(ValueError, match="positive"):
        lvk_found_subsample_log_scaling(1, 0)
    with pytest.raises(ValueError, match="cannot exceed"):
        lvk_found_subsample_log_scaling(1, 2)
