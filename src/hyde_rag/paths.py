"""Filesystem layout. Configs always come from the repo; data, cache and results can be moved.

Set HYDE_RAG_HOME to point data/, cache/ and results/ elsewhere (tests and the CI smoke eval
use this to run the real pipeline on tiny fixtures). Values are read at call time.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"


def home() -> Path:
    return Path(os.environ.get("HYDE_RAG_HOME") or REPO_ROOT)


def beir_dir() -> Path:
    return home() / "data" / "beir"


def index_dir() -> Path:
    return home() / "data" / "indexes"


def cache_dir() -> Path:
    return home() / "cache"


def results_dir() -> Path:
    return home() / "results"
