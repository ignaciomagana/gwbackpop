import numpy as np

import run_injections
from backpop import get_backpop_config


class _FakePool:
    def __init__(self, *args, **kwargs):
        init = kwargs.get("initializer")
        initargs = kwargs.get("initargs", ())
        if init is not None:
            init(*initargs)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def map(self, func, batch):
        lo, hi, params, _fixed = run_injections._load_config("lucky_strikes")
        theta = 0.5 * (lo + hi)
        return [
            {
                "theta": theta,
                "logZ": theta[params.index("logZ")],
                "z_form": 1.0,
                "m1_src": 30.0,
                "m2_src": 20.0,
                "z_merger": 0.2,
                "t_delay": 100.0,
                "pdet": np.nan,
                "log_q_proposal": -1.0,
            }
            for _ in batch
        ]


def test_saved_injection_bounds_match_backpop_config(tmp_path, monkeypatch):
    monkeypatch.setattr(run_injections, "Pool", _FakePool)
    output_path = tmp_path / "injections.npz"

    run_injections.run_campaign(
        pdet_path=None,
        output_path=str(output_path),
        n_inj=1,
        n_workers=1,
        config_name="lucky_strikes",
    )

    lo, hi, params, fixed = get_backpop_config("lucky_strikes")
    saved = np.load(output_path, allow_pickle=True)

    assert saved["config_name"][0] == "lucky_strikes"
    assert saved["params"].tolist() == params
    np.testing.assert_allclose(saved["lower_bound"], lo)
    np.testing.assert_allclose(saved["upper_bound"], hi)
    assert saved["fixed_params"].item() == fixed
    assert saved["likelihood_mode"][0] == "2D"
    assert bool(saved["uses_z_form"][0]) is False
    assert bool(saved["uses_sfr_prior"][0]) is False
    assert bool(saved["uses_logZ_given_z_prior"][0]) is False
