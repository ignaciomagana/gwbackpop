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


def test_numpyro_model_applies_positive_floors_and_packs_population_order():
    numpyro = pytest.importorskip("numpyro")
    captured = []

    def log_likelihood_fn(lp_vec):
        captured.append(np.asarray(lp_vec, dtype=float))
        return hb.jnp.array(0.0, dtype=hb.jnp.float64)

    floors = dict(sig_logalpha_floor=0.05, sigma_v_floor=5.0, beta_shape_floor=0.05)
    model = hb.make_numpyro_model(log_likelihood_fn, **floors)
    seeded = numpyro.handlers.seed(model, hb.jax.random.PRNGKey(123))
    trace = numpyro.handlers.trace(seeded).get_trace()

    assert captured, "model should evaluate the likelihood with a packed lp_vec"
    packed = captured[-1]
    deterministic_values = np.array([np.asarray(trace[name]["value"], dtype=float) for name in hb.POP_PARAM_NAMES])
    np.testing.assert_allclose(packed, deterministic_values)

    assert float(trace["sig_logalpha1"]["value"]) >= floors["sig_logalpha_floor"]
    assert float(trace["sig_logalpha2"]["value"]) >= floors["sig_logalpha_floor"]
    assert float(trace["sigma_v1"]["value"]) >= floors["sigma_v_floor"]
    assert float(trace["sigma_v2"]["value"]) >= floors["sigma_v_floor"]
    for name in ("a_f1", "b_f1", "a_f2", "b_f2"):
        assert float(trace[name]["value"]) >= floors["beta_shape_floor"]
        assert f"{name}_raw" in trace
    assert "sig_logalpha1_raw" in trace
    assert "sig_logalpha2_raw" in trace
    assert "sigma_v1_raw" in trace
    assert "sigma_v2_raw" in trace
