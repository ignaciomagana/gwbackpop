import ast
from pathlib import Path

import pytest


def _load_helpers():
    source = Path(__file__).resolve().parents[1] / "src" / "gwbackpop" / "inference" / "hierarchical.py"
    module_ast = ast.parse(source.read_text())
    names = {"_bool_meta", "metadata_model_signature", "validate_selection_model_consistency"}
    helpers = [
        node for node in module_ast.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    mod = ast.Module(body=helpers, type_ignores=[])
    ast.fix_missing_locations(mod)
    namespace = {"warnings": __import__("warnings")}
    exec(compile(mod, str(source), "exec"), namespace)
    return namespace


_helpers = _load_helpers()
validate_selection_model_consistency = _helpers["validate_selection_model_consistency"]


def test_2d_events_with_3d_injection_metadata_raise_clear_error():
    event_meta = [{
        "likelihood_mode": "2D",
        "uses_z_form": False,
        "uses_sfr_prior": False,
        "uses_logZ_given_z_prior": False,
    }]
    injection_meta = {
        "likelihood_mode": "3D",
        "uses_z_form": True,
        "uses_sfr_prior": True,
        "uses_logZ_given_z_prior": True,
    }

    with pytest.raises(ValueError, match="generative models are inconsistent"):
        validate_selection_model_consistency(event_meta, injection_meta)


def test_inconsistent_selection_model_override_records_false():
    ok, message = validate_selection_model_consistency(
        [{"likelihood_mode": "2D", "uses_z_form": False}],
        {"likelihood_mode": "3D", "uses_z_form": True},
        allow_inconsistent=True,
    )

    assert ok is False
    assert "inconsistent" in message
