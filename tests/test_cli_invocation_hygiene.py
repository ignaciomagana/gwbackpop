from pathlib import Path

FORBIDDEN = [
    "python run_backpop.py",
    "python run_injections.py",
    "python hierarchical_backpop_jax.py",
    "python plot_backpop.py",
    "python smoke_test_imports.py",
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


def test_user_facing_files_use_console_entry_points():
    files = [p for p in PATHS if p.exists()] + [p for p in iter_workflow_files() if p.exists()]
    violations = []
    for path in files:
        text = path.read_text(encoding="utf-8")
        for bad in FORBIDDEN:
            if bad in text:
                violations.append(f"{path}: contains forbidden legacy invocation {bad!r}")
    assert not violations, "\n".join(violations)
