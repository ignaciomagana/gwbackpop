import pickle
import numpy as np
import pytest

from gwbackpop.selection.snr_pdet import make_snr_proxy_pdet_callable
from gwbackpop.selection.injections import resolve_pdet_callable


def test_snr_proxy_monotonicity_and_bounds():
    pdet = make_snr_proxy_pdet_callable(n_orientation=5000, seed=1)
    vals = pdet(30.0, 30.0, np.array([0.05, 0.2, 0.6]))
    assert np.all(np.isfinite(vals))
    assert np.all((0.0 <= vals) & (vals <= 1.0))
    assert np.all(np.diff(vals) <= 0.0)
    lo = make_snr_proxy_pdet_callable(sensitivity_scale=0.5, n_orientation=5000, seed=1)(30.0, 30.0, 0.2)
    hi = make_snr_proxy_pdet_callable(sensitivity_scale=2.0, n_orientation=5000, seed=1)(30.0, 30.0, 0.2)
    assert hi >= lo


def test_snr_proxy_hard_threshold():
    p = make_snr_proxy_pdet_callable(method="hard_threshold", rho_threshold=10.0, rho_ref=20.0, d_ref_mpc=1000.0, sensitivity_scale=1.0)
    assert p.pdet_from_rho(np.array([11.0]))[0] == 1.0
    assert p.pdet_from_rho(np.array([9.0]))[0] == 0.0


@pytest.mark.parametrize("method", ["orientation_monte_carlo", "logistic"])
def test_snr_proxy_soft_methods_increase_with_rho(method):
    p = make_snr_proxy_pdet_callable(method=method, n_orientation=5000, seed=2)
    vals = p.pdet_from_rho(np.array([1.0, 10.0, 30.0]))
    assert np.all((0.0 <= vals) & (vals <= 1.0))
    assert np.all(np.diff(vals) >= 0.0)


def test_injection_pdet_mode_parsing(tmp_path):
    path, callable_, meta = resolve_pdet_callable("none", None)
    assert path is None and callable_ is None and meta["pdet_mode"] == "none"
    path, callable_, meta = resolve_pdet_callable("snr_proxy", None, snr_n_orientation=1000)
    assert path is None and callable_ is not None and meta["pdet_model_name"] == "semi_analytic_snr_proxy"
    with pytest.raises(ValueError, match="requires"):
        resolve_pdet_callable("pickle", None)
    pkl = tmp_path / "p.pkl"
    pkl.write_bytes(pickle.dumps(abs))
    path, callable_, meta = resolve_pdet_callable("pickle", str(pkl))
    assert path == str(pkl) and callable_ is None and meta["pdet_mode"] == "pickle"
