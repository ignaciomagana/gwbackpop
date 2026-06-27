import ast
import warnings
from pathlib import Path

import pytest


def _load_mass_column_helpers():
    source = Path(__file__).resolve().parents[1] / "src" / "gwbackpop" / "evolution" / "cosmic.py"
    module_ast = ast.parse(source.read_text())
    wanted = {
        "_SOURCE_MASS_ALIASES",
        "_DETECTOR_MASS_ALIASES",
        "_AMBIGUOUS_MASS_COLUMNS",
    }
    body = []
    for node in module_ast.body:
        if isinstance(node, ast.Assign):
            names = {target.id for target in node.targets if isinstance(target, ast.Name)}
            if names & wanted:
                body.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in {"_find_column_pair", "resolve_mass_columns"}:
            body.append(node)
    mod = ast.Module(body=body, type_ignores=[])
    ast.fix_missing_locations(mod)
    namespace = {"warnings": warnings}
    exec(compile(mod, str(source), "exec"), namespace)
    return namespace["resolve_mass_columns"]


resolve_mass_columns = _load_mass_column_helpers()


def test_resolve_mass_columns_prefers_explicit_source_frame():
    samples = {
        "mass_1_source": [30.0],
        "mass_2_source": [20.0],
        "mass_1_detector": [33.0],
        "mass_2_detector": [22.0],
        "luminosity_distance": [500.0],
    }

    info = resolve_mass_columns(samples, mass_frame="auto")

    assert info["frame"] == "source"
    assert info["columns"] == ("mass_1_source", "mass_2_source")
    assert info["warning"] is None


def test_resolve_mass_columns_uses_explicit_detector_frame():
    samples = {
        "mass_1_detector": [33.0],
        "mass_2_detector": [22.0],
        "luminosity_distance": [500.0],
    }

    info = resolve_mass_columns(samples, mass_frame="auto")

    assert info["frame"] == "detector"
    assert info["columns"] == ("mass_1_detector", "mass_2_detector")
    assert info["warning"] is None


def test_resolve_mass_columns_warns_for_ambiguous_auto_case():
    samples = {
        "mass_1": [33.0],
        "mass_2": [22.0],
        "luminosity_distance": [500.0],
    }

    with pytest.warns(UserWarning, match="ambiguous mass_1/mass_2"):
        info = resolve_mass_columns(samples, mass_frame="auto")

    assert info["frame"] == "detector"
    assert info["columns"] == ("mass_1", "mass_2")
    assert "assuming detector-frame" in info["warning"]


def test_resolve_mass_columns_source_override_for_ambiguous_case():
    samples = {
        "mass_1": [30.0],
        "mass_2": [20.0],
        "luminosity_distance": [500.0],
    }

    with pytest.warns(UserWarning, match="--mass_frame source"):
        info = resolve_mass_columns(samples, mass_frame="source")

    assert info["frame"] == "source"
    assert info["columns"] == ("mass_1", "mass_2")


def test_resolve_mass_columns_rejects_missing_requested_frame():
    samples = {
        "mass_1_source": [30.0],
        "mass_2_source": [20.0],
        "luminosity_distance": [500.0],
    }

    with pytest.raises(KeyError, match="detector-frame"):
        resolve_mass_columns(samples, mass_frame="detector")
