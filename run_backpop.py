#!/usr/bin/env python
# Legacy compatibility wrapper. Prefer the installed console command.
"""Compatibility wrapper. Prefer ``gwbackpop-run-event``."""
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if _SRC.exists():
    sys.path.insert(0, str(_SRC))

from gwbackpop.inference import single_event as _single_event

globals().update(_single_event.__dict__)

if __name__ == "__main__":
    _single_event.main()
else:
    sys.modules[__name__] = _single_event
