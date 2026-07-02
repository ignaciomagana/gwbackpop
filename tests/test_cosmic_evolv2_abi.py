from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pytest


def _install_fake_cosmic(monkeypatch, evolvebin):
    fake_cosmic = types.ModuleType("cosmic")
    fake_cosmic._evolvebin = evolvebin
    monkeypatch.setitem(sys.modules, "cosmic", fake_cosmic)
    sys.modules.pop("gwbackpop.evolution.cosmic", None)
    return importlib.import_module("gwbackpop.evolution.cosmic")


class FakeEvolvebin24:
    def __init__(self):
        self.arg_counts = []

    def evolv2(self, *args):
        self.arg_counts.append(len(args))
        if len(args) != 24:
            raise TypeError(f"evolv2() takes at most 24 arguments ({len(args)} given)")
        return 1, 2, np.zeros((2, 18))


class FakeEvolvebin25:
    def __init__(self):
        self.arg_counts = []

    def evolv2(self, *args):
        self.arg_counts.append(len(args))
        if len(args) != 25:
            raise TypeError(f"evolv2() takes exactly 25 positional arguments ({len(args)} given)")
        return None, 1, 2


class FakeEvolvebinBadTypeError:
    def evolv2(self, *args):
        raise TypeError("ufunc 'isfinite' not supported for the input types")


def test_evolv2_24_arg_fake_passes(monkeypatch):
    fake = FakeEvolvebin24()
    cosmic_mod = _install_fake_cosmic(monkeypatch, fake)

    out, convention = cosmic_mod._call_evolv2_with_supported_abi(
        fake, tuple(range(24)), np.zeros((2, 18))
    )

    assert convention == "24-arg"
    assert len(out) == 3
    assert fake.arg_counts == [24]


def test_evolv2_25_arg_fake_passes(monkeypatch):
    fake = FakeEvolvebin25()
    cosmic_mod = _install_fake_cosmic(monkeypatch, fake)

    out, convention = cosmic_mod._call_evolv2_with_supported_abi(
        fake, tuple(range(24)), np.zeros((2, 18))
    )

    assert convention == "25-arg"
    assert len(out) == 3
    assert fake.arg_counts == [24, 25]


def test_non_argument_typeerror_is_not_swallowed(monkeypatch):
    fake = FakeEvolvebinBadTypeError()
    cosmic_mod = _install_fake_cosmic(monkeypatch, fake)

    with pytest.raises(TypeError, match="ufunc 'isfinite'"):
        cosmic_mod._call_evolv2_with_supported_abi(fake, tuple(range(24)), np.zeros((2, 18)))


def test_vector_alpha_flim_capability_passes_independent_check(monkeypatch):
    fake = types.SimpleNamespace(
        cevars=types.SimpleNamespace(alpha1=np.ones(2)),
        mtvars=types.SimpleNamespace(acc_lim=np.ones(2)),
        se_flags=types.SimpleNamespace(),
    )
    cosmic_mod = _install_fake_cosmic(monkeypatch, fake)

    caps = cosmic_mod.get_cosmic_capabilities()

    assert caps["cevars_alpha1_shape"] == (2,)
    assert caps["mtvars_acc_lim_shape"] == (2,)
    assert caps["has_se_flags"] is True
    cosmic_mod.require_independent_alpha_flim_capability()
