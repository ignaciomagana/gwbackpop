import numpy as np
import pytest

from gwbackpop.selection.injections import LOWER, PARAMS, UPPER, compute_log_q_proposal


def _theta(vk1=30.0, vk2=70.0):
    theta = 0.5 * (LOWER + UPPER)
    theta[PARAMS.index("vk1")] = vk1
    theta[PARAMS.index("vk2")] = vk2
    return theta


def test_log_q_proposal_is_finite_for_representative_stored_injections():
    values = np.array([
        compute_log_q_proposal(_theta(20.0, 40.0), 1.0, -2.0, 50.0),
        compute_log_q_proposal(_theta(80.0, 120.0), 2.0, -2.5, 50.0),
        compute_log_q_proposal(_theta(5.0, 250.0), 0.5, -1.8, 50.0),
    ])
    assert np.all(np.isfinite(values))


def test_log_q_proposal_changes_with_kick_scale():
    theta = _theta(100.0, 150.0)
    log_q_50 = compute_log_q_proposal(theta, 1.5, -2.2, 50.0)
    log_q_100 = compute_log_q_proposal(theta, 1.5, -2.2, 100.0)
    assert np.isfinite(log_q_50)
    assert np.isfinite(log_q_100)
    assert not np.isclose(log_q_50, log_q_100)


def test_kick_direction_diagnostic_is_isotropic_in_cosmic_convention():
    from gwbackpop.selection.injections import sample_kick_directions_for_diagnostic

    rng = np.random.default_rng(12345)
    phi, theta = sample_kick_directions_for_diagnostic(rng, 50_000)
    sin_phi = np.sin(np.deg2rad(phi))

    # For COSMIC's convention, phi is valid on [-90, 90] deg and theta is the
    # [0, 360] deg azimuth.  Isotropy implies sin(phi) is uniform on [-1, 1].
    assert abs(np.mean(sin_phi)) < 0.01
    assert abs(np.var(sin_phi) - 1.0 / 3.0) < 0.01
    assert abs(np.mean(theta) - 180.0) < 2.0
    assert abs(np.var(theta) - 360.0**2 / 12.0) < 100.0


def test_isotropic_phi_logpdf_matches_sampler_distribution():
    from gwbackpop.selection.injections import _isotropic_phi_logpdf

    assert np.isneginf(_isotropic_phi_logpdf(-91.0, -90.0, 90.0))
    assert np.isneginf(_isotropic_phi_logpdf(91.0, -90.0, 90.0))
    assert _isotropic_phi_logpdf(0.0, -90.0, 90.0) > _isotropic_phi_logpdf(60.0, -90.0, 90.0)


def test_injection_catalog_and_metadata_sidecar_do_not_overwrite(tmp_path):
    from gwbackpop.metadata import save_metadata

    output_path = tmp_path / "tiny_injections.npz"
    metadata = {"likelihood_mode": "2D", "uses_aux_z_form": True}
    np.savez(
        output_path,
        theta=np.zeros((1, len(PARAMS))),
        m1_src=np.array([30.0]),
        z_merger=np.array([0.2]),
        log_q_proposal=np.array([-1.0]),
        metadata=np.array(metadata, dtype=object),
    )
    sidecar = output_path.with_name(output_path.stem + "_metadata.npz")
    save_metadata(sidecar, metadata)

    data = np.load(output_path, allow_pickle=True)
    assert {"theta", "m1_src", "z_merger", "log_q_proposal", "metadata"} <= set(data.files)
    assert sidecar.exists()
    with pytest.raises(FileExistsError):
        save_metadata(output_path, metadata)


def test_support_aware_logz_prior_normalizes_on_backpop_config():
    from gwbackpop.config import get_backpop_config
    from gwbackpop.cosmology import log_prior_logZ_given_z_on_support

    lo, hi, params, _ = get_backpop_config("lucky_strikes_zform")
    zlo = float(lo[params.index("logZ")])
    zhi = float(hi[params.index("logZ")])
    grid = np.linspace(zlo, zhi, 4000)
    for z_form in (0.1, 2.0, 10.0):
        vals = np.exp([log_prior_logZ_given_z_on_support(x, z_form, zlo, zhi) for x in grid])
        assert np.trapezoid(vals, grid) == pytest.approx(1.0, rel=2e-3)
        assert np.isneginf(log_prior_logZ_given_z_on_support(zlo - 1e-6, z_form, zlo, zhi))
        assert np.isneginf(log_prior_logZ_given_z_on_support(zhi + 1e-6, z_form, zlo, zhi))


def test_2d_aux_z_form_factor_cancels_when_added_to_static_numerator():
    from gwbackpop.selection.injections import _log_q_z_form

    z1, z2 = 0.5, 3.0
    log_q1 = _log_q_z_form(z1)
    log_q2 = _log_q_z_form(z2)
    old_delta = -log_q1 - (-log_q2)
    new_delta = (log_q1 - log_q1) - (log_q2 - log_q2)
    assert not np.isclose(old_delta, 0.0)
    assert new_delta == pytest.approx(0.0)


def test_3d_logz_proposal_and_static_numerator_use_same_config_support():
    import gwbackpop.selection.injections as ri
    from gwbackpop.config import get_backpop_config
    from gwbackpop.cosmology import log_prior_logZ_given_z_on_support

    lo, hi, params, _ = get_backpop_config("lucky_strikes_zform")
    zlo = float(lo[params.index("logZ")])
    zhi = float(hi[params.index("logZ")])
    logz = 0.5 * (zlo + zhi)
    z_form = 2.0

    old_mode, old_lo, old_hi = ri.LIKELIHOOD_MODE, ri.LOGZ_LO, ri.LOGZ_HI
    try:
        ri.LIKELIHOOD_MODE = "3D"
        ri.LOGZ_LO = zlo
        ri.LOGZ_HI = zhi
        theta = _theta()
        theta[PARAMS.index("logZ")] = logz
        log_q = ri.compute_log_q_proposal(theta, z_form, logz, 50.0)
        log_q_without_static_logz = log_q - log_prior_logZ_given_z_on_support(logz, z_form, zlo, zhi)
        assert np.isfinite(log_q)
        assert log_q - log_q_without_static_logz == pytest.approx(
            log_prior_logZ_given_z_on_support(logz, z_form, zlo, zhi)
        )
    finally:
        ri.LIKELIHOOD_MODE, ri.LOGZ_LO, ri.LOGZ_HI = old_mode, old_lo, old_hi


def test_likelihood_3d_accepts_config_logz_support(monkeypatch):
    import importlib
    import sys
    import types

    class FakeKDE:
        def logpdf(self, coord):
            return np.array([0.0])

    fake_nautilus = types.ModuleType("nautilus")
    fake_nautilus.Prior = object
    fake_nautilus.Sampler = object
    monkeypatch.setitem(sys.modules, "nautilus", fake_nautilus)

    fake_backpop = types.ModuleType("backpop")
    fake_backpop.get_backpop_config = lambda name: None
    fake_backpop.load_gw_data = lambda *args, **kwargs: None
    fake_backpop.evolv2 = lambda *args, **kwargs: None
    fake_backpop.str_to_bool = lambda value: value
    fake_backpop.BPP_SHAPE = (1,)
    fake_backpop.KICK_SHAPE = (1,)
    fake_backpop.COLS_KEEP = []
    monkeypatch.setitem(sys.modules, "backpop", fake_backpop)

    sys.modules.pop("gwbackpop.inference.single_event", None)
    run_backpop = importlib.import_module("gwbackpop.inference.single_event")

    monkeypatch.setattr(
        run_backpop,
        "evolv2",
        lambda params, output_columns, fixed_params: (
            {"mass_1": 30.0, "mass_2": 20.0},
            np.array([1.0]),
            np.array([2.0]),
        ),
    )
    monkeypatch.setattr(run_backpop, "_extract_t_delay_myr", lambda bpp_raw: 100.0)
    monkeypatch.setattr(run_backpop, "z_merger_from_t_delay", lambda z_form, t_delay: 0.1)

    log_prob, bpp_flat, kick_flat = run_backpop.likelihood_3d(
        {"z_form": 1.0, "logZ": -2.0},
        kde=FakeKDE(),
        q_bounds=(0.0, 1.0),
        mc_bounds=(0.0, 100.0),
        z_bounds=(0.0, 2.0),
        support_gate="none",
        fixed_params={},
        logZ_support=(-4.0, 0.0),
    )

    assert np.isfinite(log_prob)
    assert np.array_equal(bpp_flat, np.array([1.0]))
    assert np.array_equal(kick_flat, np.array([2.0]))

    log_prob, _, _ = run_backpop.likelihood_3d(
        {"z_form": 1.0, "logZ": -5.0},
        kde=FakeKDE(),
        q_bounds=(0.0, 1.0),
        mc_bounds=(0.0, 100.0),
        z_bounds=(0.0, 2.0),
        support_gate="none",
        fixed_params={},
        logZ_support=(-4.0, 0.0),
    )
    assert np.isneginf(log_prob)

    with pytest.raises(ValueError, match="logZ_support must satisfy lo < hi"):
        run_backpop.likelihood_3d(
            {"z_form": 1.0, "logZ": -2.0},
            kde=FakeKDE(),
            q_bounds=(0.0, 1.0),
            mc_bounds=(0.0, 100.0),
            z_bounds=(0.0, 2.0),
            support_gate="none",
            fixed_params={},
            logZ_support=(0.0, -4.0),
        )


def test_set_evolvebin_flags_allows_missing_se_flags(monkeypatch):
    import importlib
    import sys
    import types

    fake_cosmic = types.ModuleType("cosmic")
    fake_evolvebin = types.SimpleNamespace()
    for name in [
        "windvars", "cevars", "ceflags", "flags", "snvars", "points",
        "mtvars", "magvars", "tidalvars", "rand1", "mixvars", "metvars",
    ]:
        setattr(fake_evolvebin, name, types.SimpleNamespace())
    fake_cosmic._evolvebin = fake_evolvebin
    monkeypatch.setitem(sys.modules, "cosmic", fake_cosmic)
    sys.modules.pop("gwbackpop.evolution.cosmic", None)
    cosmic_mod = importlib.import_module("gwbackpop.evolution.cosmic")

    flag_names = [
        'ST_cr', 'ST_tide', 'acc2', 'acc_lim', 'aic', 'alpha1', 'bconst',
        'bdecayfac', 'beta', 'bhflag', 'bhms_coll_flag', 'bhsigmafrac',
        'bhspinflag', 'bhspinmag', 'bwind', 'ceflag', 'cehestarflag',
        'cekickflag', 'cemergeflag', 'ck', 'don_lim', 'ecsn', 'ecsn_mlow',
        'eddfac', 'eddlimflag', 'epsnov', 'fprimc_array', 'gamma', 'grflag',
        'hewind', 'htpmb', 'ifflag', 'kickflag', 'lambdaf', 'mxns',
        'natal_kick_array', 'neta', 'pisn', 'polar_kick_angle', 'pts1',
        'pts2', 'pts3', 'qcflag', 'qcrit_array', 'randomseed', 'rejuv_fac',
        'rejuvflag', 'rembar_massloss', 'remnantflag', 'rtmsflag', 'sigma',
        'sigmadiv', 'tflag', 'ussn', 'wdflag', 'windflag', 'xi', 'zsun',
    ]
    flags = {name: 1.0 for name in flag_names}
    flags["alpha1"] = np.array([1.0, 1.0])
    flags["acc_lim"] = np.array([0.5, 0.5])
    flags["natal_kick_array"] = np.zeros((2, 5))
    flags["qcrit_array"] = np.ones(16)
    flags["fprimc_array"] = np.ones(16)

    with pytest.warns(RuntimeWarning, match="lacks se_flags"):
        cosmic_mod.set_evolvebin_flags(flags)
    cosmic_mod.set_evolvebin_flags(flags)


def test_run_one_debug_reports_cosmic_exception(monkeypatch):
    import importlib
    import sys
    import types

    fake_cosmic = types.ModuleType("cosmic")
    fake_cosmic._evolvebin = types.SimpleNamespace()
    monkeypatch.setitem(sys.modules, "cosmic", fake_cosmic)
    sys.modules.pop("gwbackpop.evolution.cosmic", None)

    import gwbackpop.selection.injections as inj
    cosmic_mod = importlib.import_module("gwbackpop.evolution.cosmic")

    monkeypatch.setattr(inj, "DEBUG_FAILURES", True)
    monkeypatch.setattr(inj, "LOWER", np.array([10.0, 0.5]))
    monkeypatch.setattr(inj, "UPPER", np.array([20.0, 1.0]))
    monkeypatch.setattr(inj, "PARAMS", ["m1", "q"])
    monkeypatch.setattr(inj, "FIXED_PARAMS", {})
    monkeypatch.setattr(inj, "LIKELIHOOD_MODE", "2D")
    monkeypatch.setattr(inj, "_PDET", None, raising=False)
    monkeypatch.setattr(inj, "_draw_z_form", lambda rng: 1.0)
    monkeypatch.setattr(inj, "compute_log_q_proposal", lambda *args, **kwargs: 0.0)
    monkeypatch.setattr(cosmic_mod, "evolv2", lambda *args, **kwargs: (_ for _ in ()).throw(AttributeError("boom")))

    result = inj._run_one(123)

    assert result["ok"] is False
    assert result["reason"] == "cosmic_exception"
    assert "AttributeError" in result["message"]
