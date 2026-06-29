from importlib.metadata import entry_points
from pathlib import Path
import tomllib

EXPECTED = {
    "gwbackpop-run-event": "gwbackpop.cli.run_backpop:main",
    "gwbackpop-run-injections": "gwbackpop.cli.run_injections:main",
    "gwbackpop-run-hierarchical": "gwbackpop.cli.run_hierarchical:main",
    "gwbackpop-calibrate-snr-pdet": "gwbackpop.selection.calibrate_snr_pdet:main",
    "gwbackpop-plot": "gwbackpop.cli.plot_backpop:main",
    "gwbackpop-smoke-test": "gwbackpop.cli.smoke_test:main",
}


def test_console_scripts_registered():
    scripts = entry_points(group="console_scripts")
    names = {ep.name: ep.value for ep in scripts}

    missing = [name for name in EXPECTED if name not in names]
    if missing:
        # Source-tree fallback for non-installed local test runs; CI installs the
        # package before running this test and exercises the actual metadata.
        project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
        names = project["project"]["scripts"]

    assert {name: names[name] for name in EXPECTED} == EXPECTED
