from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import duckdb

from logicdb.layout import (
    apply_layout_plan,
    call_llm_for_layout_plan,
    extract_database_stats,
    heuristic_layout_plan,
)
from logicdb.layout.operators import drop_schema_cascade
from logicdb.openops import (
    ActionFamily,
    ActionOutcome,
    OpenOpsAction,
    OpenOpsOrchestrator,
    OrchestrationState,
)
from logicdb.physical import (
    build_indexes,
    build_mvs,
    generate_index_candidates,
    llm_choose_indexes,
    llm_generate_mvs,
    rewrite_with_mvs,
)
from logicdb.physical.cost_model import estimate_future_template_freq
from logicdb.physical.index_duckdb import build_col_to_tables, get_table_stats
from logicdb.physical.index_duckdb import IndexCand
from logicdb.physical.mv_duckdb import build_workload_summary, compute_workload_stats, get_duckdb_schema_info
from logicdb.physical.mv_duckdb import MVCand


@dataclass
class LogicDBConfig:
    duckdb_path: str
    base_schema: str = "main"
    workdir: str = "./logicdb_runs"
    layout_model: str = "gpt-5-mini"
    index_model: str = "gpt-5-mini"
    mv_model: str = "gpt-5-mini"
    reasoning_effort: Optional[str] = None
    max_indexes: int = 6
    max_mvs: int = 6
    maintenance_budget_ms: float = 100000.0
    risk_tolerance: float = 0.7
    layout_variant_name: str = "logicdb_layout"
    layout_variant_schema: str = "logicdb_layout"
    parquet_work_dir: Optional[str] = None
    allow_heuristic_fallbacks: bool = True


@dataclass
class LogicDBRunResult:
    schema_info: str
    table_names: List[str]
    workload_stats: Dict[str, Any]
    notes: List[str] = field(default_factory=list)
    layout_plan: Optional[Dict[str, Any]] = None
    layout_paths: List[str] = field(default_factory=list)
    index_candidates: List[Dict[str, Any]] = field(default_factory=list)
    chosen_indexes: List[Dict[str, Any]] = field(default_factory=list)
    built_indexes: List[Dict[str, Any]] = field(default_factory=list)
    mv_candidates: List[Dict[str, Any]] = field(default_factory=list)
    built_mvs: List[Dict[str, Any]] = field(default_factory=list)
    openops_plan: Dict[str, Any] = field(default_factory=dict)
    openops_commit: Dict[str, Any] = field(default_factory=dict)
    rewrite_demo: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _load_workload_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            rows.append(json.loads(s))
    return rows


def _has_openai_credentials() -> bool:
    return bool(str(os.getenv("OPENAI_API_KEY", "") or "").strip())


def _choose_indexes_fallback(
    candidates: List[Any],
    *,
    max_indexes: int,
) -> List[Any]:
    ranked = sorted(
        candidates,
        key=lambda c: (
            -float(getattr(c, "priority", 0.0) or 0.0),
            float(getattr(c, "estimated_size_mb", 0.0) or 0.0),
            str(getattr(c, "cid", "") or ""),
        ),
    )
    return ranked[: max(0, int(max_indexes))]


def _reset_layout_variant(con: duckdb.DuckDBPyConnection, variant_schema: str) -> None:
    if str(variant_schema or "").strip():
        drop_schema_cascade(con, str(variant_schema))


def _action_for_layout(plan: Dict[str, Any]) -> OpenOpsAction:
    est_apply = 1500.0 + 300.0 * len(plan.get("ops", []))
    return OpenOpsAction(
        action_id="layout.reorganize",
        family=ActionFamily.SCHEMA,
        schema="reorganize_layout(table, spec)",
        target=str(plan.get("variant_name") or "layout"),
        params=plan,
        expected_gain_ms=6000.0,
        expected_apply_ms=est_apply,
        risk_score=0.25,
        is_heavy=True,
        resource_tags={"layout"},
    )


def _action_for_index(idx: Dict[str, Any], n: int) -> OpenOpsAction:
    return OpenOpsAction(
        action_id=f"index.{n}",
        family=ActionFamily.INDEX,
        schema="build_index(table, keys, include, type)",
        target=str(idx.get("table") or ""),
        params=idx,
        expected_gain_ms=2500.0 + 500.0 * len(idx.get("cols") or []),
        expected_apply_ms=800.0,
        risk_score=0.18,
        is_heavy=False,
        resource_tags={f"index:{idx.get('table') or ''}"},
        metadata={"timescale": "maintenance"},
    )


def _action_for_mv(mv: Dict[str, Any], n: int) -> OpenOpsAction:
    return OpenOpsAction(
        action_id=f"mv.{n}",
        family=ActionFamily.PRECACHE,
        schema="create_artifact(name, spec, budget)",
        target=str(mv.get("name") or mv.get("mvid") or ""),
        params=mv,
        expected_gain_ms=4200.0,
        expected_apply_ms=1200.0,
        risk_score=0.22,
        is_heavy=True,
        resource_tags={"mv"},
        metadata={"timescale": "maintenance"},
    )


def _logicdb_executor(
    con: duckdb.DuckDBPyConnection,
    config: LogicDBConfig,
    action_map: Dict[str, Dict[str, Any]],
):
    def _execute(action: OpenOpsAction, _st: OrchestrationState) -> ActionOutcome:
        payload = action_map.get(action.action_id) or {}
        kind = str(payload.get("kind") or "").strip().lower()
        try:
            if kind == "layout":
                plan = payload["plan"]
                _reset_layout_variant(con, str(plan.get("variant_schema") or ""))
                paths = apply_layout_plan(con, plan)
                payload["layout_paths"] = list(paths)
                return ActionOutcome(ok=True, latency_delta_ms=float(action.expected_gain_ms), metadata={"layout_paths": list(paths)})
            if kind == "index":
                cand = payload["candidate"]
                _, applied = build_indexes(con, [cand], config.base_schema, verbose=False)
                payload["applied"] = list(applied)
                ok = bool(applied) and str(applied[0].get("status", "")).startswith("ok")
                err = None if ok else (applied[0].get("status") if applied else "index_build_failed")
                return ActionOutcome(ok=ok, latency_delta_ms=float(action.expected_gain_ms) if ok else 0.0, error=err, metadata={"applied": list(applied)})
            if kind == "mv":
                mv = payload["candidate"]
                build_mvs(con, [mv], verbose=False)
                ok = bool(getattr(mv, "built", False))
                err = None if ok else str(getattr(mv, "build_error", "") or "mv_build_failed")
                return ActionOutcome(ok=ok, latency_delta_ms=float(action.expected_gain_ms) if ok else 0.0, error=err, metadata={"mv_name": getattr(mv, "name", "")})
            return ActionOutcome(ok=False, error=f"unknown_action_kind:{kind}")
        except Exception as exc:
            return ActionOutcome(ok=False, error=f"{type(exc).__name__}: {exc}")

    return _execute


def run_openops_demo(
    config: LogicDBConfig,
    workload: List[Dict[str, Any]],
    *,
    table_names: Optional[List[str]] = None,
) -> LogicDBRunResult:
    _ensure_dir(config.workdir)
    con = duckdb.connect(config.duckdb_path)
    try:
        schema_info, all_tables = get_duckdb_schema_info(con, config.base_schema, table_names)
        workload_stats = compute_workload_stats(con, workload, config.base_schema)
        table_names = table_names or all_tables
        notes: List[str] = []

        # Layout planning
        db_stats = extract_database_stats(con, schema=config.base_schema)
        wl_summary = build_workload_summary(workload, max_queries=40, stats=workload_stats)
        if _has_openai_credentials():
            try:
                layout_result = call_llm_for_layout_plan(
                    workload_features=wl_summary,
                    schema_info=schema_info,
                    model=config.layout_model,
                    db_stats=db_stats,
                    reasoning_effort=config.reasoning_effort,
                )
                notes.append("layout_planner=llm")
            except Exception as exc:
                if not config.allow_heuristic_fallbacks:
                    raise
                notes.append(f"layout_planner=fallback:{exc}")
                layout_result = heuristic_layout_plan(
                    schema_info,
                    db_stats=db_stats,
                    workload_stats=workload_stats,
                    variant_name=config.layout_variant_name,
                    variant_schema=config.layout_variant_schema,
                    work_dir=config.parquet_work_dir or os.path.join(config.workdir, "parquet"),
                )
        elif config.allow_heuristic_fallbacks:
            notes.append("layout_planner=heuristic_fallback")
            layout_result = heuristic_layout_plan(
                schema_info,
                db_stats=db_stats,
                workload_stats=workload_stats,
                variant_name=config.layout_variant_name,
                variant_schema=config.layout_variant_schema,
                work_dir=config.parquet_work_dir or os.path.join(config.workdir, "parquet"),
            )
        else:
            notes.append("layout_planner=disabled_no_key")
            layout_result = {"reasoning": "Skipped: OPENAI_API_KEY missing.", "plan": {"ops": []}}
        layout_plan = dict(layout_result.get("plan") or {})
        layout_plan.setdefault("variant_name", config.layout_variant_name)
        layout_plan.setdefault("variant_schema", config.layout_variant_schema)
        layout_plan["work_dir"] = config.parquet_work_dir or os.path.join(config.workdir, "parquet")
        layout_paths: List[str] = []
        if layout_plan.get("ops"):
            try:
                _reset_layout_variant(con, str(layout_plan.get("variant_schema") or ""))
                layout_paths = apply_layout_plan(con, layout_plan)
            except Exception as exc:
                layout_plan.setdefault("errors", []).append(str(exc))

        # Index planning
        idx_candidates = generate_index_candidates(
            con,
            workload,
            config.base_schema,
            table_names,
            workload_stats=workload_stats,
        )
        if _has_openai_credentials():
            try:
                chosen_indexes = llm_choose_indexes(
                    idx_candidates,
                    schema_info,
                    model=config.index_model,
                    reasoning_effort=config.reasoning_effort,
                    max_indexes=config.max_indexes,
                    workload_stats=workload_stats,
                )
                notes.append("index_planner=llm")
            except Exception as exc:
                if not config.allow_heuristic_fallbacks:
                    raise
                notes.append(f"index_planner=fallback:{exc}")
                chosen_indexes = _choose_indexes_fallback(idx_candidates, max_indexes=config.max_indexes)
        else:
            notes.append("index_planner=priority_fallback")
            chosen_indexes = _choose_indexes_fallback(idx_candidates, max_indexes=config.max_indexes)
        # MV planning
        mv_candidates = llm_generate_mvs(
            con,
            workload,
            schema_info,
            base_schema=config.base_schema,
            model=config.mv_model,
            reasoning_effort=config.reasoning_effort,
            max_mvs=config.max_mvs,
            stats=workload_stats,
        )

        # OpenOps orchestration view
        actions: List[OpenOpsAction] = []
        action_payloads: Dict[str, Dict[str, Any]] = {}
        if layout_plan.get("ops"):
            layout_action = _action_for_layout(layout_plan)
            actions.append(layout_action)
            action_payloads[layout_action.action_id] = {"kind": "layout", "plan": layout_plan}
        for i, idx in enumerate(chosen_indexes, start=1):
            idx_action = _action_for_index(asdict(idx), i)
            actions.append(idx_action)
            action_payloads[idx_action.action_id] = {"kind": "index", "candidate": idx}
        for i, mv in enumerate(mv_candidates[: config.max_mvs], start=1):
            mv_action = _action_for_mv(asdict(mv), i)
            actions.append(mv_action)
            action_payloads[mv_action.action_id] = {"kind": "mv", "candidate": mv}

        future_templates = estimate_future_template_freq(workload, 0, min(len(workload), 500))
        st = OrchestrationState(
            window_id=0,
            schema_epoch=0,
            budget_total_ms=config.maintenance_budget_ms,
            budget_by_family_ms={
                ActionFamily.SCHEMA: config.maintenance_budget_ms * 0.4,
                ActionFamily.INDEX: config.maintenance_budget_ms * 0.3,
                ActionFamily.PRECACHE: config.maintenance_budget_ms * 0.3,
            },
            risk_tolerance=config.risk_tolerance,
            evidence={"future_template_freq": future_templates},
        )
        orch = OpenOpsOrchestrator(min_expected_gain_ms=100.0)
        dag = orch.build_dag(actions)
        selected = orch.select_plan(dag, st)
        commit = orch.commit_plan(
            selected,
            dag,
            st,
            execute_fn=_logicdb_executor(con, config, action_payloads),
        )

        selected_set = set(selected.selected_ids)
        built_indexes: List[Dict[str, Any]] = []
        for rec in commit.records:
            if rec.action_id.startswith("index."):
                payload = action_payloads.get(rec.action_id) or {}
                built_indexes.extend(list(payload.get("applied") or []))

        built_mvs: List[Dict[str, Any]] = []
        for rec in commit.records:
            if rec.action_id.startswith("mv."):
                payload = action_payloads.get(rec.action_id) or {}
                mv = payload.get("candidate")
                if mv is not None and getattr(mv, "built", False):
                    built_mvs.append(asdict(mv))

        if "layout.reorganize" in selected_set:
            layout_paths = list((action_payloads.get("layout.reorganize") or {}).get("layout_paths") or layout_paths)

        rewrite_demo: List[Dict[str, Any]] = []
        for q in workload[: min(5, len(workload))]:
            sql = str(q.get("sql") or q.get("query") or "").strip()
            if not sql:
                continue
            try:
                rewritten_sql, used = rewrite_with_mvs(sql, [payload["candidate"] for payload in action_payloads.values() if payload.get("kind") == "mv" and getattr(payload.get("candidate"), "built", False)])
            except Exception:
                rewritten_sql, used = sql, []
            rewrite_demo.append(
                {
                    "sql": sql,
                    "rewritten_sql": rewritten_sql,
                    "mv_names": [getattr(m, "name", str(m)) for m in used or []],
                }
            )

        result = LogicDBRunResult(
            schema_info=schema_info,
            table_names=table_names,
            workload_stats=workload_stats,
            notes=notes,
            layout_plan=layout_plan,
            layout_paths=layout_paths,
            index_candidates=[asdict(x) for x in idx_candidates[:50]],
            chosen_indexes=[asdict(x) for x in chosen_indexes],
            built_indexes=built_indexes,
            mv_candidates=[asdict(x) for x in mv_candidates],
            built_mvs=built_mvs,
            openops_plan={
                "selected_ids": list(selected.selected_ids),
                "rejected_ids": dict(selected.rejected_ids),
                "score_sum": selected.score_sum,
                "num_actions": len(actions),
            },
            openops_commit={
                "applied_ids": list(commit.applied_ids),
                "rolled_back_ids": list(commit.rolled_back_ids),
                "commit_wall_ms": commit.commit_wall_ms,
                "records": [asdict(r) for r in commit.records],
            },
            rewrite_demo=rewrite_demo,
        )
        return result
    finally:
        con.close()


def run_openops_demo_from_jsonl(
    config: LogicDBConfig,
    workload_jsonl: str,
    *,
    table_names: Optional[List[str]] = None,
) -> LogicDBRunResult:
    workload = _load_workload_jsonl(workload_jsonl)
    return run_openops_demo(config, workload, table_names=table_names)
