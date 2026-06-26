import numpy as np
import pytest

hb = pytest.importorskip("gwbackpop.inference.hierarchical")


def test_truncated_logpdfs_jax_numpy_agree_inside_support():
    alpha = np.array([0.1, 0.3, 1.0, 5.0, 20.0])
    vk = np.array([0.0, 5.0, 50.0, 200.0, 500.0])

    alpha_np = hb._lognormal_logpdf_numpy(alpha, 0.2, 0.8, 0.1, 20.0)
    alpha_jax = np.asarray(hb.log_p_alpha_jax(hb.jnp.array(alpha), 0.2, 0.8, 0.1, 20.0))
    np.testing.assert_allclose(alpha_jax, alpha_np, rtol=1e-10, atol=1e-10)

    vk_np = hb._maxwell_logpdf_numpy(vk, 120.0, 0.0, 500.0)
    vk_jax = np.asarray(hb.log_p_vk_jax(hb.jnp.array(vk), 120.0, 0.0, 500.0))
    np.testing.assert_allclose(vk_jax, vk_np, rtol=1e-10, atol=1e-10)


def test_truncated_logpdfs_return_minus_inf_outside_support():
    alpha = np.array([0.099, 20.001])
    vk = np.array([-1.0, 500.001])

    assert np.all(np.isneginf(hb._lognormal_logpdf_numpy(alpha, 0.2, 0.8, 0.1, 20.0)))
    assert np.all(np.isneginf(np.asarray(hb.log_p_alpha_jax(hb.jnp.array(alpha), 0.2, 0.8, 0.1, 20.0))))
    assert np.all(np.isneginf(hb._maxwell_logpdf_numpy(vk, 120.0, 0.0, 500.0)))
    assert np.all(np.isneginf(np.asarray(hb.log_p_vk_jax(hb.jnp.array(vk), 120.0, 0.0, 500.0))))


@pytest.mark.parametrize(
    "kind,args,bounds",
    [
        ("alpha", (0.2, 0.8), (0.1, 20.0)),
        ("vk", (120.0,), (0.0, 500.0)),
    ],
)
def test_truncated_numpy_densities_integrate_to_one(kind, args, bounds):
    lo, hi = bounds
    grid = np.geomspace(lo, hi, 20000) if lo > 0 else np.linspace(lo, hi, 20000)
    if kind == "alpha":
        logpdf = hb._lognormal_logpdf_numpy(grid, *args, lo, hi)
    else:
        logpdf = hb._maxwell_logpdf_numpy(grid, *args, lo, hi)
    integral = np.trapz(np.exp(logpdf), grid)
    assert integral == pytest.approx(1.0, rel=1e-4, abs=1e-4)


def test_flim_clipping_is_boundary_only_not_out_of_support():
    vals = hb.jnp.array([-1e-8, 0.0, 0.5, 1.0, 1.0 + 1e-8])
    got = np.asarray(hb.log_p_flim_jax(vals, 0.7, 1.3))
    assert np.isneginf(got[0])
    assert np.isfinite(got[1:4]).all()
    assert np.isneginf(got[4])
