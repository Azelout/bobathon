"""Parkinson true-OFF prediction package.

Exposes `PROJECT_ROOT`, the absolute path to the repo root, so modules
resolve project files (data, configs) without depending on the CWD.
Relies on the package being installed in editable mode (`pip install -e .`).
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]