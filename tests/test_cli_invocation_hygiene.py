from pathlib import Path

LEGACY_WRAPPER_PATHS = [
    Path("run_backpop.py"),
    Path("run_injections.py"),
    Path("hierarchical_backpop_jax.py"),
    Path("plot_backpop.py"),
    Path("smoke_test_imports.py"),
    Path("scripts/run_backpop.py"),
    Path("scripts/run_injections.py"),
    Path("scripts/hierarchical_backpop_jax.py"),
    Path("scripts/plot_backpop.py"),
    Path("scripts/smoke_test_imports.py"),
]

FORBIDDEN_STRINGS = [
    "python run_backpop.py",
    "python run_injections.py",
    "python hierarchical_backpop_jax.py",
    "python plot_backpop.py",
    "python smoke_test_imports.py",
    "Legacy compatibility wrapper",
    "backwards-compatible wrappers",
    "old root-level Python wrappers",
    "root-level Python wrappers remain",
]

PATHS = [
    Path("README.md"),
    Path("run_GW150914.sh"),
]


def iter_workflow_files():
    for pattern in [
        "workflows/**/*.sh",
        "workflows/**/*.slurm",
        ".github/workflows/*.yml",
        ".github/workflows/*.yaml",
    ]:
        yield from Path(".").glob(pattern)


def test_legacy_python_wrappers_are_deleted():
    existing = [str(p) for p in LEGACY_WRAPPER_PATHS if p.exists()]
    assert not existing, "Legacy wrapper files should be deleted:\n" + "\n".join(existing)


def test_user_facing_files_use_console_entry_points():
    files = [p for p in PATHS if p.exists()] + [p for p in iter_workflow_files() if p.exists()]
    violations = []
    for path in files:
        text = path.read_text(encoding="utf-8")
        for bad in FORBIDDEN_STRINGS:
            if bad in text:
                violations.append(f"{path}: contains forbidden legacy text {bad!r}")
    assert not violations, "\n".join(violations)
