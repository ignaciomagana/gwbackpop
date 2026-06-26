import ast
from pathlib import Path

import pytest


def _load_determine_selection_mode():
    """Load only the pure helper so tests do not require JAX/NumPyro deps."""
    source = Path(__file__).resolve().parents[1] / "hierarchical_backpop_jax.py"
    module_ast = ast.parse(source.read_text())
    helper = next(
        node for node in module_ast.body
        if isinstance(node, ast.FunctionDef) and node.name == "determine_selection_mode"
    )
    mod = ast.Module(body=[helper], type_ignores=[])
    ast.fix_missing_locations(mod)
    namespace = {}
    exec(compile(mod, str(source), "exec"), namespace)
    return namespace["determine_selection_mode"]


determine_selection_mode = _load_determine_selection_mode()


def test_selection_mode_none_without_selection_inputs():
    assert determine_selection_mode(None, None) == "none"


def test_selection_mode_lvk_farr_requires_both_inputs():
    assert determine_selection_mode("cosmic.npz", "lvk.hdf5") == "lvk_farr"


def test_lvk_found_path_requires_injections_path():
    with pytest.raises(ValueError, match="requires --injections_path"):
        determine_selection_mode(None, "lvk.hdf5")


def test_cosmic_only_interpolator_mode_hard_errors():
    with pytest.raises(NotImplementedError, match="unimplemented interpolator"):
        determine_selection_mode("cosmic.npz", None)
