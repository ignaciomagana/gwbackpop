"""COSMIC capability inspection and enforcement helpers.

These checks keep configurations with independent first/second-episode
``alpha`` or ``flim`` parameters from silently running on COSMIC builds where
those flags are scalar rather than episode-indexed vectors.
"""

from __future__ import annotations

import importlib
from importlib import metadata as importlib_metadata
from typing import Any


_COSMIC410_FIRST_LINE_TOKEN = "zpars,kick_info,bpp_index_out,bcm_index_out = evolv2"
_COSMIC410_KICK_INFO_TOKEN = "kick_info : input rank-2 array('d') with bounds (2,19)"
_INSTALL_COMMAND = 'python -m pip install --upgrade --force-reinstall "cosmic-popsynth>=4.1.0,<4.2.0"'
_REINSTALL_COMMAND = "python -m pip install --no-deps --force-reinstall -e ."


def _version_tuple(version: str | None) -> tuple[int, ...]:
    if not version:
        return ()
    parts: list[int] = []
    for part in str(version).split("."):
        digits = ""
        for char in part:
            if char.isdigit():
                digits += char
            else:
                break
        if digits == "":
            break
        parts.append(int(digits))
    return tuple(parts)


def _version_in_supported_range(version: str | None) -> bool:
    parsed = _version_tuple(version)
    return (4, 1, 0) <= parsed < (4, 2, 0)


def _shape_of(value: Any) -> list[int] | None:
    try:
        import numpy as np

        return [int(dim) for dim in np.asarray(value).shape]
    except Exception:
        shape = getattr(value, "shape", None)
        if shape is None:
            return None
        try:
            return [int(dim) for dim in tuple(shape)]
        except Exception:
            return None


def _is_rank1_size_at_least_two(shape: list[int] | None) -> bool:
    return shape is not None and len(shape) == 1 and shape[0] >= 2


def _cosmic_version(cosmic_module: Any | None = None) -> str | None:
    try:
        return importlib_metadata.version("cosmic-popsynth")
    except importlib_metadata.PackageNotFoundError:
        return str(getattr(cosmic_module, "__version__", "")) or None


def _base_capabilities() -> dict[str, Any]:
    return {
        "cosmic_popsynth_version": None,
        "has_evolvebin": False,
        "has_se_flags": False,
        "cevars_alpha1_shape": None,
        "mtvars_acc_lim_shape": None,
        "col_inds_bpp_shape": None,
        "col_inds_bcm_shape": None,
        "evolv2_first_doc_line": None,
        "supports_independent_alpha": False,
        "supports_independent_flim": False,
        "supports_cosmic410_evolv2_signature": False,
        "supported_for_independent_alpha_flim": False,
    }


def inspect_cosmic_capabilities() -> dict[str, Any]:
    """Return JSON-serializable details about the installed COSMIC backend."""
    caps = _base_capabilities()
    try:
        cosmic = importlib.import_module("cosmic")
        caps["cosmic_popsynth_version"] = _cosmic_version(cosmic)
        evolvebin = importlib.import_module("cosmic._evolvebin")
    except Exception as exc:
        caps["has_evolvebin"] = False
        caps["import_error"] = f"{type(exc).__name__}: {exc}"
        if caps["cosmic_popsynth_version"] is None:
            caps["cosmic_popsynth_version"] = _cosmic_version(None)
        return caps

    caps["has_evolvebin"] = True
    caps["has_se_flags"] = bool(hasattr(evolvebin, "se_flags"))
    caps["cevars_alpha1_shape"] = _shape_of(getattr(getattr(evolvebin, "cevars", None), "alpha1", None))
    caps["mtvars_acc_lim_shape"] = _shape_of(getattr(getattr(evolvebin, "mtvars", None), "acc_lim", None))
    caps["col_inds_bpp_shape"] = _shape_of(getattr(evolvebin, "col_inds_bpp", None))
    caps["col_inds_bcm_shape"] = _shape_of(getattr(evolvebin, "col_inds_bcm", None))

    doc = getattr(getattr(evolvebin, "evolv2", None), "__doc__", None) or ""
    first_line = next((line.strip() for line in doc.splitlines() if line.strip()), "")
    caps["evolv2_first_doc_line"] = first_line or None
    caps["supports_independent_alpha"] = _is_rank1_size_at_least_two(caps["cevars_alpha1_shape"])
    caps["supports_independent_flim"] = _is_rank1_size_at_least_two(caps["mtvars_acc_lim_shape"])
    caps["supports_cosmic410_evolv2_signature"] = (
        _COSMIC410_FIRST_LINE_TOKEN in first_line and _COSMIC410_KICK_INFO_TOKEN in doc
    )
    caps["supported_for_independent_alpha_flim"] = bool(
        _version_in_supported_range(caps["cosmic_popsynth_version"])
        and caps["supports_independent_alpha"]
        and caps["supports_independent_flim"]
        and caps["has_se_flags"]
        and caps["supports_cosmic410_evolv2_signature"]
    )
    return caps


def format_cosmic_capabilities_report(capabilities: dict[str, Any] | None = None) -> str:
    """Format a human-readable COSMIC capability report."""
    caps = inspect_cosmic_capabilities() if capabilities is None else capabilities
    lines = ["COSMIC capability report"]
    for key in _base_capabilities():
        lines.append(f"{key}: {caps.get(key)}")
    if caps.get("import_error"):
        lines.append(f"import_error: {caps['import_error']}")
    return "\n".join(lines)


def require_supported_cosmic_for_independent_alpha_flim(
    config_name: str | None = None,
    params: list[str] | tuple[str, ...] | None = None,
) -> None:
    """Fail fast if active independent alpha/flim params need unsupported COSMIC."""
    active_params = list(params or [])
    needs_alpha = "alpha_2" in active_params
    needs_flim = "flim_2" in active_params
    if not (needs_alpha or needs_flim):
        return

    caps = inspect_cosmic_capabilities()
    if caps["supported_for_independent_alpha_flim"]:
        return

    raise RuntimeError(
        "Unsupported COSMIC installation for independent alpha_2/flim_2 evolution.\n"
        f"config_name: {config_name}\n"
        f"params: {active_params}\n"
        f"detected cosmic-popsynth version: {caps.get('cosmic_popsynth_version')}\n"
        f"detected cevars.alpha1 shape: {caps.get('cevars_alpha1_shape')}\n"
        f"detected mtvars.acc_lim shape: {caps.get('mtvars_acc_lim_shape')}\n"
        f"has_se_flags: {caps.get('has_se_flags')}\n"
        f"evolv2 first doc line: {caps.get('evolv2_first_doc_line')}\n"
        f"install command: {_INSTALL_COMMAND}\n"
        f"reinstall command: {_REINSTALL_COMMAND}"
    )


def main() -> None:
    caps = inspect_cosmic_capabilities()
    print(format_cosmic_capabilities_report(caps))
    raise SystemExit(0 if caps["supported_for_independent_alpha_flim"] else 1)


if __name__ == "__main__":
    main()
