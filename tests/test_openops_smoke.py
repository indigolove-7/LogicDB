from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from logicdb import LogicDBConfig, run_openops_demo_from_jsonl


def test_openops_demo_applies_real_actions(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parent.parent
    subprocess.run(
        [sys.executable, str(repo_root / "examples" / "build_demo_duckdb.py")],
        check=True,
        cwd=repo_root,
    )

    cfg = LogicDBConfig(
        duckdb_path=str(repo_root / "examples" / "demo" / "demo.duckdb"),
        workdir=str(tmp_path / "openops_run"),
    )
    result = run_openops_demo_from_jsonl(cfg, str(repo_root / "examples" / "demo" / "workload.jsonl")).to_dict()

    applied_ids = list(result.get("openops_commit", {}).get("applied_ids", []))
    assert applied_ids
    assert "layout.reorganize" in applied_ids
    assert any(action_id.startswith("index.") for action_id in applied_ids)
    assert any(action_id.startswith("mv.") for action_id in applied_ids)
    assert result.get("layout_paths")
    assert result.get("built_indexes")
    assert result.get("built_mvs")
