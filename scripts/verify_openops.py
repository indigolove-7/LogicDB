from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from logicdb import LogicDBConfig, run_openops_demo_from_jsonl


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    subprocess.run(
        [sys.executable, str(repo_root / "examples" / "build_demo_duckdb.py")],
        check=True,
        cwd=repo_root,
    )
    demo_db = repo_root / "examples" / "demo" / "demo.duckdb"
    demo_workload = repo_root / "examples" / "demo" / "workload.jsonl"
    out_dir = repo_root / "examples" / "demo" / "verify_openops_run"

    cfg = LogicDBConfig(
        duckdb_path=str(demo_db),
        workdir=str(out_dir),
    )
    result = run_openops_demo_from_jsonl(cfg, str(demo_workload)).to_dict()

    selected = list(result.get("openops_plan", {}).get("selected_ids", []))
    applied = list(result.get("openops_commit", {}).get("applied_ids", []))
    commit_records = list(result.get("openops_commit", {}).get("records", []))
    built_indexes = list(result.get("built_indexes", []))
    built_mvs = list(result.get("built_mvs", []))
    layout_paths = list(result.get("layout_paths", []))

    if not selected:
        raise SystemExit("OpenOps verification failed: no actions were selected.")
    if not applied:
        raise SystemExit("OpenOps verification failed: no selected actions were committed.")
    if not commit_records:
        raise SystemExit("OpenOps verification failed: commit records are missing.")
    if "layout.reorganize" not in applied or not layout_paths:
        raise SystemExit("OpenOps verification failed: layout action was not applied.")
    if not any(action_id.startswith("index.") for action_id in applied) or not built_indexes:
        raise SystemExit("OpenOps verification failed: no index action was applied.")
    if not any(action_id.startswith("mv.") for action_id in applied) or not built_mvs:
        raise SystemExit("OpenOps verification failed: no MV action was applied.")

    summary = {
        "selected_ids": selected,
        "applied_ids": applied,
        "built_index_count": len(built_indexes),
        "built_mv_count": len(built_mvs),
        "layout_paths": layout_paths,
        "rolled_back_ids": result.get("openops_commit", {}).get("rolled_back_ids", []),
        "commit_wall_ms": result.get("openops_commit", {}).get("commit_wall_ms", 0.0),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
