#!/usr/bin/env python
"""Compatibility wrapper. Prefer ``gwbackpop-smoke-test``."""
from pathlib import Path
import sys

_SRC = Path(__file__).resolve().parent / "src"
if _SRC.exists():
    sys.path.insert(0, str(_SRC))


def main(*args, **kwargs):
    from gwbackpop.cli.smoke_test import main as _main

    return _main(*args, **kwargs)


if __name__ == "__main__":
    main()
