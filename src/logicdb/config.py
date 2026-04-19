from __future__ import annotations

import os
from pathlib import Path


def _env_path(name: str, default: str) -> str:
    value = str(os.getenv(name, "") or "").strip()
    return value or default


PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent.parent

DEFAULT_SPIDER_ROOT = _env_path("LOGICDB_SPIDER_ROOT", str(REPO_ROOT / "data" / "spider"))
DEFAULT_BIRD_ROOT = _env_path("LOGICDB_BIRD_ROOT", str(REPO_ROOT / "data" / "bird"))
DEFAULT_WTQ_ROOT = _env_path("LOGICDB_WTQ_ROOT", str(REPO_ROOT / "data" / "wtq"))
DEFAULT_TABFACT_ROOT = _env_path("LOGICDB_TABFACT_ROOT", str(REPO_ROOT / "data" / "tabfact"))


def get_openai_base_url() -> str | None:
    value = str(os.getenv("OPENAI_BASE_URL", "") or "").strip()
    return value or None
