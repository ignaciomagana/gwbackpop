import inspect
import numpy as np
import jax.numpy as jnp
import pytest
from scipy.special import logsumexp

from gwbackpop.inference import hierarchical as h


def _write_toy(path):
    params = np.array(["alpha_1", "alpha_2", "flim_1", "flim_2", "vk1", "vk2", "logtb"], dtype="U16")
    theta = np.array([
        [1.0, 1.2, 0.3, 0.4, 30.0, 40.0, 2.0],
        [1.5, 1.1, 0.5, 0.6, 50.0, 60.0, 3.0],
        [2.0, 1.4, 0.7, 0.5, 70.0, 80.0, 4.0],
    ])
    lo = np.array([0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 1.0])
    hi = np.array([10.0, 10.0, 1.0, 1.0, 500.0, 500.0, 5.0])
    np.savez(path, theta=theta, params=params, lower_bound=lo, upper_bound=hi,
             log_q_proposal=np.array([-8.0, -8.1, -8.2]), pdet=np.array([0.2, 0.3, 0.4]),
             m1_src=np.array([30, 31, 32.]), m2_src=np.array([20, 21, 22.]), z_merger=np.array([.1,.2,.3]),
             N_inj=np.array([10]), N_merge=np.array([3]), kick_proposal_sigma=np.array([50.0]),
             likelihood_mode=np.array(["2D"]), uses_aux_z_form=np.array([False]),
             uses_sfr_prior=np.array([False]), uses_logZ_given_z_prior=np.array([False]),
             metadata=np.array({"likelihood_mode":"2D", "uses_aux_z_form": False, "uses_sfr_prior": False, "uses_logZ_given_z_prior": False}, dtype=object))


def test_shared_catalog_helper_static_terms(tmp_path):
    f = tmp_path / "toy.npz"; _write_toy(f)
    cosmic, meta, ok, msg = h.load_cosmic_merger_catalog_for_selection(str(f), [], True)
    for key in ["theta", "m1_src", "m2_src", "z_merger", "pdet", "params", "lo", "hi", "N_inj", "N_merge", "kick_sigma", "log_q_proposal", "log_pop_static"]:
        assert key in cosmic
    assert ok
    assert np.allclose(cosmic["pdet"], [0.2, 0.3, 0.4])
    assert np.all(np.isfinite(cosmic["log_q_proposal"]))
    assert np.all(np.isfinite(cosmic["log_pop_static"]))
    assert not np.allclose(cosmic["log_pop_static"], 0.0)
    assert np.allclose(cosmic["log_pop_static"], -np.log(4.0))


def test_direct_alpha_static_terms_change_alpha():
    theta = np.array([[1.0], [2.0], [3.0]])
    lo = np.array([0.1]); hi = np.array([10.0]); params = {"alpha_1": 0}
    pdet = np.array([0.2, 0.4, 0.8]); log_q = np.array([-1.0, -1.0, -1.0])
    static = np.array([0.0, 0.5, 1.0]); lp = h.default_hyperparams()
    fn = h.make_log_alpha_direct_pdet_fn(jnp.array(theta), jnp.array(pdet), jnp.array(lo), jnp.array(hi), params, 20, 50.0, jnp.array(log_q), jnp.array(static))
    got = float(fn(lp))
    log_wr = h.compute_log_wr_injections_numpy(lp, theta, ["alpha_1"], lo, hi, 50.0, log_q, static)
    expected = logsumexp(np.log(pdet) + log_wr) - np.log(20)
    assert got == pytest.approx(expected)
    no_static = h.make_log_alpha_direct_pdet_fn(jnp.array(theta), jnp.array(pdet), jnp.array(lo), jnp.array(hi), params, 20, 50.0, jnp.array(log_q), jnp.zeros(3))
    assert float(no_static(lp)) != pytest.approx(got)


def test_lvk_farr_reuses_shared_catalog_helper():
    src = inspect.getsource(h.main)
    lvk_part = src.split('elif selection_mode == "lvk_farr"', 1)[1]
    assert "load_cosmic_merger_catalog_for_selection" in lvk_part
    assert "log_pop_static_arr = np.zeros" not in lvk_part


def test_catalog_helper_reports_missing_required_fields(tmp_path):
    f = tmp_path / "missing_common.npz"
    np.savez(f, theta=np.zeros((2, 1)))
    with pytest.raises(ValueError, match="load_cosmic_merger_catalog_for_selection requires.*params.*lower_bound.*upper_bound.*N_inj.*N_merge"):
        h.load_cosmic_merger_catalog_for_selection(str(f), [], True)


def test_catalog_helper_requires_merger_observables_but_not_pdet(tmp_path):
    f = tmp_path / "no_pdet.npz"
    np.savez(
        f,
        theta=np.zeros((2, 1)),
        params=np.array(["alpha_1"], dtype="U16"),
        lower_bound=np.array([0.1]),
        upper_bound=np.array([10.0]),
        m1_src=np.array([30.0, 31.0]),
        m2_src=np.array([20.0, 21.0]),
        z_merger=np.array([0.1, 0.2]),
        N_inj=np.array([5]),
        N_merge=np.array([2]),
    )
    cosmic, *_ = h.load_cosmic_merger_catalog_for_selection(str(f), [], True)
    assert cosmic["pdet"] is None

    f_missing_obs = tmp_path / "missing_observables.npz"
    np.savez(
        f_missing_obs,
        theta=np.zeros((2, 1)),
        params=np.array(["alpha_1"], dtype="U16"),
        lower_bound=np.array([0.1]),
        upper_bound=np.array([10.0]),
        N_inj=np.array([5]),
        N_merge=np.array([2]),
    )
    with pytest.raises(ValueError, match="direct_pdet and LVK/Farr selection requires.*m1_src.*m2_src.*z_merger"):
        h.load_cosmic_merger_catalog_for_selection(str(f_missing_obs), [], True)
