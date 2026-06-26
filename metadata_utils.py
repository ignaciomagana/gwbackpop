"""Helpers for reproducible BackPop metadata products."""
from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import import_module, metadata as importlib_metadata
from pathlib import Path
from typing import Any

import numpy as np


def get_git_commit_hash(repo_path: str | Path = ".") -> str | None:
    """Return the current git commit hash when available."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_path),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None


def get_package_versions(package_names: list[str] | tuple[str, ...]) -> dict[str, str | None]:
    """Collect installed package versions without importing heavy packages when possible."""
    versions: dict[str, str | None] = {}
    aliases = {"cosmic": ("cosmic-popsynth", "cosmic"), "nautilus": ("nautilus-sampler", "nautilus")}
    for name in package_names:
        candidates = aliases.get(name, (name,))
        version = None
        for dist_name in candidates:
            try:
                version = importlib_metadata.version(dist_name)
                break
            except importlib_metadata.PackageNotFoundError:
                continue
        if version is None:
            try:
                module = import_module(name)
                version = getattr(module, "__version__", None)
            except Exception:
                version = None
        versions[name] = version
    return versions


def base_runtime_metadata(repo_path: str | Path = ".") -> dict[str, Any]:
    """Runtime metadata common to all output products."""
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit_hash": get_git_commit_hash(repo_path),
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
    }


def to_jsonable(value: Any) -> Any:
    """Convert numpy/JAX/Python objects into JSON-serializable values."""
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return to_jsonable(value.item())
        return [to_jsonable(v) for v in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if np.isnan(value):
            return None
        if np.isposinf(value):
            return "Infinity"
        if np.isneginf(value):
            return "-Infinity"
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    try:
        import jax
        if isinstance(value, jax.Device):
            return str(value)
    except Exception:
        pass
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def save_metadata(output_dir_or_file: str | Path, metadata: dict[str, Any], *, npz_name: str = "metadata.npz", json_name: str = "metadata.json") -> None:
    """Save metadata as machine-readable NPZ and human-readable JSON.

    If *output_dir_or_file* has suffix ``.npz``, that exact NPZ path is used and
    the JSON sidecar is written next to it with suffix ``.json`` unless
    ``json_name`` is an absolute or explicit filename.
    """
    target = Path(output_dir_or_file)
    if target.suffix == ".npz":
        npz_path = target
        json_path = target.with_suffix(".json") if json_name == "metadata.json" else target.with_name(json_name)
    else:
        target.mkdir(parents=True, exist_ok=True)
        npz_path = target / npz_name
        json_path = target / json_name
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(npz_path, **metadata)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(metadata), f, indent=2, sort_keys=True)
        f.write("\n")


def load_metadata_prefer_json(path: str | Path) -> dict[str, Any]:
    """Load metadata from JSON sidecar when available, otherwise NPZ."""
    path = Path(path)
    json_path = path.with_suffix(".json") if path.suffix == ".npz" else path / "metadata.json"
    npz_path = path if path.suffix == ".npz" else path / "metadata.npz"
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    with np.load(npz_path, allow_pickle=True) as raw:
        return {k: raw[k].item() if raw[k].shape == () else raw[k] for k in raw.files}
