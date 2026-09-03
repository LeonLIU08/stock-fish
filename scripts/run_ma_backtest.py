#!/usr/bin/env python3
"""Backward-compatible entry point for the MA-cross backtest."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_spec = importlib.util.spec_from_file_location("run_backtest", _SCRIPT_DIR / "run_backtest.py")
_module = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_module)

if __name__ == "__main__":
    raise SystemExit(_module.main())
