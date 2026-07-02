from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pytest


COSMIC410_DOC = """
zpars,kick_info,bpp_index_out,bcm_index_out = evolv2(kstar,mass,tb,ecc,z,tphysf,dtp,mass0,rad,lumin,massc,radc,menv,renv,ospin,b_0,bacc,tacc,epoch,tms,bhspin,tphys,zpars,kick_info)

kick_info : input rank-2 array('d') with bounds (2,19)
"""


def _install_fake_cosmic(monkeypatch, evolvebin):
    fake_cosmic = types.ModuleType("cosmic")
    fake_cosmic._evolvebin = evolvebin
    monkeypatch.setitem(sys.modules, "cosmic", fake_cosmic)
    sys.modules.pop("gwbackpop.evolution.cosmic", None)
    return importlib.import_module("gwbackpop.evolution.cosmic")


class FakeEvolv2Callable:
    def __init__(self, owner, doc):
        self.owner = owner
        self.__doc__ = doc

    def __call__(self, *args):
        return self.owner._evolv2(*args)


class FakeEvolvebin410:
    def __init__(self):
        self.arg_counts = []
        self.received_bkick = False
        self.evolv2 = FakeEvolv2Callable(self, COSMIC410_DOC)

    def _evolv2(self, *args):
        self.arg_counts.append(len(args))
        if len(args) != 24:
            raise TypeError(f"evolv2() takes exactly 24 positional arguments ({len(args)} given)")
        kick_info = args[-1]
        if np.shape(kick_info) != (2, 19):
            raise ValueError(f"0-th dimension must be fixed to 2 but got {np.shape(kick_info)}")
        # bkick was the old final 24th argument and has shape (20,); ensure it was not passed.
        self.received_bkick = np.shape(args[-1]) == (20,)
        return args[-2], kick_info + 1.0, np.array(7), np.array([9])


class FakeEvolvebin25:
    def __init__(self):
        self.arg_counts = []
        self.evolv2 = FakeEvolv2Callable(self, "legacy evolv2 doc")

    def _evolv2(self, *args):
        self.arg_counts.append(len(args))
        if len(args) != 25:
            raise TypeError(f"evolv2() takes exactly 25 positional arguments ({len(args)} given)")
        bkick = args[-2]
        kick_info = args[-1]
        assert np.shape(bkick) == (20,)
        assert np.shape(kick_info) == (2, 18)
        return None, 3, np.array(4), kick_info + 2.0


def test_cosmic410_24_fake_uses_no_bkick_and_shape_2x19(monkeypatch):
    fake = FakeEvolvebin410()
    cosmic_mod = _install_fake_cosmic(monkeypatch, fake)
    kick_info = np.zeros((2, 19))

    out, convention = cosmic_mod._call_evolv2_with_supported_abi(
        fake, (*tuple(range(23)), kick_info), np.zeros(20), kick_info
    )

    assert convention == "cosmic410_24"
    assert fake.arg_counts == [24]
    assert fake.received_bkick is False
    bpp_index, bcm_index, kick_arrays, ret_len = cosmic_mod._parse_evolv2_return(
        out, convention, kick_info
    )
    assert (bpp_index, bcm_index, ret_len) == (7, 9, 4)
    assert kick_arrays.shape == (2, 19)


def test_legacy_25_fake_requires_bkick_and_kick_info(monkeypatch):
    fake = FakeEvolvebin25()
    cosmic_mod = _install_fake_cosmic(monkeypatch, fake)
    kick_info = np.zeros((2, 18))

    out, convention = cosmic_mod._call_evolv2_with_supported_abi(
        fake, (*tuple(range(23)), kick_info), np.zeros(20), kick_info
    )

    assert convention == "legacy_25"
    assert fake.arg_counts == [25]
    bpp_index, bcm_index, kick_arrays, ret_len = cosmic_mod._parse_evolv2_return(
        out, convention, kick_info
    )
    assert (bpp_index, bcm_index, ret_len) == (3, 4, 4)
    assert kick_arrays.shape == (2, 18)


def test_non_argument_typeerror_is_not_swallowed(monkeypatch):
    class FakeBad(FakeEvolvebin25):
        def _evolv2(self, *args):
            raise TypeError("ufunc 'isfinite' not supported for the input types")

    fake = FakeBad()
    cosmic_mod = _install_fake_cosmic(monkeypatch, fake)

    # The convention-aware caller no longer retries legacy calls after docstring
    # selection, so internal TypeErrors should propagate unchanged.
    with pytest.raises(TypeError, match="ufunc 'isfinite'"):
        cosmic_mod._call_evolv2_with_supported_abi(
            fake, (*tuple(range(23)), np.zeros((2, 18))), np.zeros(20), np.zeros((2, 18))
        )


def test_vector_alpha_flim_capability_passes_independent_check(monkeypatch):
    fake = types.SimpleNamespace(
        cevars=types.SimpleNamespace(alpha1=np.ones(2)),
        mtvars=types.SimpleNamespace(acc_lim=np.ones(2)),
        se_flags=types.SimpleNamespace(),
        evolv2=FakeEvolv2Callable(types.SimpleNamespace(_evolv2=lambda *args: None), COSMIC410_DOC),
    )
    cosmic_mod = _install_fake_cosmic(monkeypatch, fake)

    caps = cosmic_mod.get_cosmic_capabilities()

    assert caps["cevars_alpha1_shape"] == (2,)
    assert caps["mtvars_acc_lim_shape"] == (2,)
    assert caps["has_se_flags"] is True
    assert caps["evolv2_docstring_first_line"].startswith("zpars,kick_info")
    cosmic_mod.require_independent_alpha_flim_capability()
