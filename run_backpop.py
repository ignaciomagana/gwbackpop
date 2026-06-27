#!/usr/bin/env python
"""Compatibility wrapper. Prefer ``gwbackpop-run-event``."""
import sys

from gwbackpop.inference import single_event as _single_event

globals().update(_single_event.__dict__)

if __name__ == "__main__":
    _single_event.main()
else:
    sys.modules[__name__] = _single_event
