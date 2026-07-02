import sys
import types

import numpy as np
import pytest

from gwbackpop.evolution import cosmic_capabilities as cc


DOC_410 = """zpars,kick_info,bpp_index_out,bcm_index_out = evolv2(kstar,mass,tb,ecc,z,tphysf,dtp,mass0,rad,lumin,massc,radc,menv,renv,ospin,b_0,bacc,tacc,epoch,tms,bhspin,tphys,zpars,kick_info)

kick_info : input rank-2 array('d') with bounds (2,19)
"""


def install_fake_cosmic(monkeypatch, *, version="4.1.0", alpha=(0.5, 5.0), flim=(0.1, 0.9), se_flags=True, doc=DOC_410):
    cosmic = types.ModuleType("cosmic")
    evolvebin = types.ModuleType("cosmic._evolvebin")
    evolvebin.cevars = types.SimpleNamespace(alpha1=np.asarray(alpha))
    evolvebin.mtvars = types.SimpleNamespace(acc_lim=np.asarray(flim))
    evolvebin.col_inds_bpp = np.zeros(3)
    evolvebin.col_inds_bcm = np.zeros(4)

    def evolv2():
        return None

    evolv2.__doc__ = doc
    evolvebin.evolv2 = evolv2
    if se_flags:
        evolvebin.se_flags = True
    cosmic._evolvebin = evolvebin
    cosmic.__version__ = version
    monkeypatch.setitem(sys.modules, "cosmic", cosmic)
    monkeypatch.setitem(sys.modules, "cosmic._evolvebin", evolvebin)
    monkeypatch.setattr(cc.importlib_metadata, "version", lambda name: version)
    return cosmic, evolvebin


def test_vector_alpha_flim_cosmic410_supported(monkeypatch):
    install_fake_cosmic(monkeypatch)
    caps = cc.inspect_cosmic_capabilities()
    assert caps["cosmic_popsynth_version"] == "4.1.0"
    assert caps["cevars_alpha1_shape"] == [2]
    assert caps["mtvars_acc_lim_shape"] == [2]
    assert caps["has_se_flags"] is True
    assert caps["supports_independent_alpha"] is True
    assert caps["supports_independent_flim"] is True
    assert caps["supports_cosmic410_evolv2_signature"] is True
    assert caps["supported_for_independent_alpha_flim"] is True


@pytest.mark.parametrize("alpha,flim", [(0.5, 0.1), ([0.5], [0.1])])
def test_scalar_or_length_one_alpha_flim_unsupported(monkeypatch, alpha, flim):
    install_fake_cosmic(monkeypatch, alpha=alpha, flim=flim)
    caps = cc.inspect_cosmic_capabilities()
    assert caps["supports_independent_alpha"] is False
    assert caps["supports_independent_flim"] is False
    assert caps["supported_for_independent_alpha_flim"] is False


def test_missing_se_flags_unsupported(monkeypatch):
    install_fake_cosmic(monkeypatch, se_flags=False)
    caps = cc.inspect_cosmic_capabilities()
    assert caps["has_se_flags"] is False
    assert caps["supported_for_independent_alpha_flim"] is False


def test_unsupported_376_scalar_fails_require_for_independent_params(monkeypatch):
    install_fake_cosmic(monkeypatch, version="3.7.6", alpha=0.5, flim=0.1)
    with pytest.raises(RuntimeError) as excinfo:
        cc.require_supported_cosmic_for_independent_alpha_flim(
            config_name="bad_config", params=["alpha_2", "flim_2"]
        )
    message = str(excinfo.value)
    assert "bad_config" in message
    assert "3.7.6" in message
    assert "python -m pip install --upgrade --force-reinstall" in message
    assert "python -m pip install --no-deps --force-reinstall -e ." in message


def test_params_without_independent_terms_do_not_raise_when_cosmic_missing(monkeypatch):
    monkeypatch.delitem(sys.modules, "cosmic", raising=False)
    monkeypatch.delitem(sys.modules, "cosmic._evolvebin", raising=False)

    def missing_import(name):
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(cc.importlib, "import_module", missing_import)
    cc.require_supported_cosmic_for_independent_alpha_flim(params=["alpha_1", "flim_1"])


def test_error_message_includes_install_and_reinstall_commands(monkeypatch):
    install_fake_cosmic(monkeypatch, version="3.7.6", alpha=0.5, flim=0.1)
    with pytest.raises(RuntimeError) as excinfo:
        cc.require_supported_cosmic_for_independent_alpha_flim(params=["alpha_2"])
    message = str(excinfo.value)
    assert 'python -m pip install --upgrade --force-reinstall "cosmic-popsynth>=4.1.0,<4.2.0"' in message
    assert "python -m pip install --no-deps --force-reinstall -e ." in message


def test_doctor_report_includes_core_fields(monkeypatch):
    install_fake_cosmic(monkeypatch)
    report = cc.format_cosmic_capabilities_report()
    assert "cosmic_popsynth_version: 4.1.0" in report
    assert "cevars_alpha1_shape: [2]" in report
    assert "mtvars_acc_lim_shape: [2]" in report
    assert "has_se_flags: True" in report
    assert "evolv2_first_doc_line: zpars,kick_info,bpp_index_out,bcm_index_out = evolv2" in report
