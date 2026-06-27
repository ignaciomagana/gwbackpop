"""Legacy import compatibility for :mod:`gwbackpop.inference.single_event`.

Prefer the ``gwbackpop-run-event`` console script or
``gwbackpop.inference.single_event`` for new code.  This module keeps existing
``import run_backpop`` callers working when gwbackpop is installed as a package.
"""
from __future__ import annotations

import sys

from gwbackpop.inference import single_event as _single_event

# Expose the implementation module itself so monkeypatching module globals such
# as ``evolv2`` affects functions whose globals live in ``single_event``.
sys.modules[__name__] = _single_event
