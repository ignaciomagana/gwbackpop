import json
import math

import numpy as np
import pytest

from gwbackpop.config import get_backpop_config
from gwbackpop.metadata import load_metadata_prefer_json, save_metadata

hb = pytest.importorskip("gwbackpop.inference.hierarchical")


def test_get_backpop_config_lucky_strikes_shape_and_bounds():
    lo, hi, params, fixed = get_backpop_config("lucky_strikes")

    assert len(params) == 16
    assert lo.shape == hi.shape == (len(params),)
    assert fixed == {}
    assert params[:4] == ["m1", "q", "logtb", "logZ"]
    assert np.all(hi > lo)
    assert lo[params.index("m1")] == pytest.approx(2.0)
    assert hi[params.index("m1")] == pytest.approx(150.0)


def test_get_backpop_config_zform_adds_only_z_form_dimension():
    lo, hi, params, fixed = get_backpop_config("bbh_no_kicks_zform")

    assert params == ["m1", "q", "logtb", "logZ", "z_form", "alpha_1", "alpha_2", "flim_1", "flim_2"]
    assert fixed["vk1"] == 0.0 and fixed["vk2"] == 0.0
    assert lo[params.index("z_form")] == pytest.approx(1e-4)
    assert hi[params.index("z_form")] == pytest.approx(20.0)


def test_get_backpop_config_rejects_unknown_config():
    with pytest.raises(ValueError, match="Unknown config"):
        get_backpop_config("not_a_config")


def test_metadata_writes_npz_and_json_round_trip(tmp_path):
    save_metadata(tmp_path, {"array": np.array([1, 2]), "nan": np.float64(np.nan), "flag": np.bool_(True)})

    assert (tmp_path / "metadata.npz").exists()
    assert (tmp_path / "metadata.json").exists()
    loaded = load_metadata_prefer_json(tmp_path)
    assert loaded["array"] == [1, 2]
    assert loaded["nan"] is None
    assert loaded["flag"] is True

    raw = json.loads((tmp_path / "metadata.json").read_text())
    assert raw["array"] == [1, 2]


def test_metadata_refuses_to_overwrite_existing_npz_unless_explicit(tmp_path):
    catalog = tmp_path / "catalog.npz"
    np.savez(catalog, theta=np.ones((1, 2)))

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        save_metadata(catalog, {"metadata_only": True})

    save_metadata(catalog, {"metadata_only": True}, overwrite_existing_npz=True)
    assert "metadata_only" in np.load(catalog, allow_pickle=True).files


def test_toy_hierarchical_likelihood_prefers_matching_population():
    samples = hb.jnp.array([[1.0, 0.5], [1.2, 0.5], [1.1, 0.6], [0.9, 0.4]], dtype=hb.jnp.float64)
    pidx = {"alpha_1": 0, "flim_1": 1}
    lo = hb.jnp.array([0.1, 0.0], dtype=hb.jnp.float64)
    hi = hb.jnp.array([20.0, 1.0], dtype=hb.jnp.float64)
    log_wr_fn = hb.make_log_weight_ratio_fn(samples, pidx, lo, hi)
    ll = hb.make_hierarchical_log_likelihood(
        [log_wr_fn], hb.jnp.array([0.0], dtype=hb.jnp.float64), samples.shape[0], None, 1
    )

    good_np = np.asarray(hb.default_hyperparams(), dtype=float)
    bad_np = good_np.copy()
    good_np[0] = math.log(1.0)
    good_np[1] = 0.25
    bad_np[0] = math.log(5.0)
    bad_np[1] = 0.25
    good = hb.jnp.array(good_np, dtype=hb.jnp.float64)
    bad = hb.jnp.array(bad_np, dtype=hb.jnp.float64)
    assert float(ll(good)) > float(ll(bad))


def test_root_compatibility_wrappers_expose_main():
    import importlib.util
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    for filename in [
        "run_backpop.py",
        "run_injections.py",
        "hierarchical_backpop_jax.py",
        "plot_backpop.py",
        "smoke_test_imports.py",
    ]:
        spec = importlib.util.spec_from_file_location(f"_compat_{filename.replace('.', '_')}", root / filename)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        assert callable(module.main)
