"""Skore Hub credential loader.

Reads the `.skore` file at the repo root and returns a plain dict so
experiment scripts can authenticate without hard-coding workspace names.
"""

from __future__ import annotations

import json

from parkinson import PROJECT_ROOT


def load_skore_credentials() -> dict:
    """Load Skore Hub credentials from the .skore file at the repo root.

    Returns
    -------
    dict
        Keys: ``hub_url``, ``workspace``, ``workspace_id``, ``api_key``.
    """
    skore_file = PROJECT_ROOT / ".skore"
    with skore_file.open() as f:
        return json.load(f)
