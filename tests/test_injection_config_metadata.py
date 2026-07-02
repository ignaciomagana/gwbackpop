import numpy as np

import gwbackpop.selection.injections as run_injections
from gwbackpop.config import get_backpop_config


class _FakePool:
    initargs = ()

    def __init__(self, *args, **kwargs):
        init = kwargs.get("initializer")
        initargs = kwargs.get("initargs", ())
        type(self).initargs = initargs
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


def _fake_supported_cosmic(monkeypatch):
    caps = {
        "cosmic_popsynth_version": "4.1.0",
        "supports_independent_alpha": True,
        "supports_independent_flim": True,
        "supports_cosmic410_evolv2_signature": True,
        "supported_for_independent_alpha_flim": True,
    }
    monkeypatch.setattr(run_injections, "inspect_cosmic_capabilities", lambda: caps)
    monkeypatch.setattr(
        run_injections,
        "require_supported_cosmic_for_independent_alpha_flim",
        lambda config_name=None, params=None: None,
    )


def test_saved_injection_bounds_match_backpop_config(tmp_path, monkeypatch):
    _fake_supported_cosmic(monkeypatch)
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


def test_run_campaign_accepts_debug_failures(tmp_path, monkeypatch):
    _fake_supported_cosmic(monkeypatch)
    _FakePool.initargs = ()
    monkeypatch.setattr(run_injections, "Pool", _FakePool)
    output_path = tmp_path / "injections.npz"

    run_injections.run_campaign(
        pdet_path=None,
        output_path=str(output_path),
        n_inj=1,
        n_workers=1,
        config_name="lucky_strikes",
        debug_failures=True,
    )

    assert run_injections.DEBUG_FAILURES is True
    assert _FakePool.initargs[4] is True
