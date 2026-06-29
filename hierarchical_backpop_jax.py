#!/usr/bin/env python
# Legacy compatibility wrapper. Prefer the installed console command.
"""Compatibility wrapper. Prefer ``gwbackpop-run-hierarchical``."""
from pathlib import Path
import sys

_SRC = Path(__file__).resolve().parent / "src"
if _SRC.exists():
    sys.path.insert(0, str(_SRC))


def main(*args, **kwargs):
    from gwbackpop.cli.run_hierarchical import main as _main

    return _main(*args, **kwargs)


if __name__ == "__main__":
    main()
