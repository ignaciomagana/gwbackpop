"""Import-hygiene checks for injection campaign generation."""

import os
import subprocess
import sys
from pathlib import Path


def test_selection_injections_import_does_not_import_jax_or_numpyro():
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root / "src") + os.pathsep + env.get("PYTHONPATH", "")
    code = """
import sys
import gwbackpop.selection.injections
assert "jax" not in sys.modules
assert "numpyro" not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True, env=env)
