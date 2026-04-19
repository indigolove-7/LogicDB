from __future__ import annotations

import argparse
import json
import os
from typing import Any

from logicdb.agent import AgentContext, run_hybrid_agent, run_react_agent
from logicdb.config import (
    DEFAULT_BIRD_ROOT,
    DEFAULT_SPIDER_ROOT,
    DEFAULT_TABFACT_ROOT,
    DEFAULT_WTQ_ROOT,
)
from logicdb.data import (
    get_db_path,
    load_bird_samples,
    load_spider_samples,
    load_spider_tables,
    load_tabfact_samples,
    load_wtq_samples,
)
from logicdb.runner import LogicDBConfig, run_openops_demo_from_jsonl
from logicdb.skills.mine import mine_skills


def _write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def cmd_agent(args: argparse.Namespace) -> int:
    spider_tables = None
    evidence = None
    db_id = args.db_id
    question = args.question
    db_path = args.db_path

    if not question:
        if args.dataset == "spider":
            samples = load_spider_samples(os.path.join(args.spider_root, "dev.json"), limit=args.idx + 1)
            spider_tables = load_spider_tables(os.path.join(args.spider_root, "tables.json"))
        elif args.dataset == "bird":
            samples = load_bird_samples(os.path.join(args.bird_root, "dev.json"), limit=args.idx + 1)
        elif args.dataset == "wtq":
            samples = load_wtq_samples(args.wtq_root, split=args.wtq_split, limit=args.idx + 1)
        else:
            samples = load_tabfact_samples(args.tabfact_root, split=args.tabfact_split, limit=args.idx + 1)
        sample = samples[args.idx]
        question = sample.get("question", sample.get("Question", ""))
        db_id = db_id or sample.get("db_id")
        db_path = db_path or sample.get("db_path")
        evidence = sample.get("evidence")

    if not db_path:
        db_path = get_db_path(db_id, args.dataset, args.spider_root, args.bird_root)
    if args.dataset == "spider" and spider_tables is None:
        tables_path = os.path.join(args.spider_root, "tables.json")
        if os.path.exists(tables_path):
            spider_tables = load_spider_tables(tables_path)

    ctx = AgentContext(
        db_id=db_id,
        db_path=db_path,
        dataset=args.dataset,
        question=question,
        spider_tables=spider_tables,
        evidence=evidence,
        parquet_dir=args.parquet_dir,
        max_turns=args.max_turns,
        model=args.model,
        skill_bank_path=args.skill_bank_path,
        skill_top_k=args.skill_top_k,
        reasoning_effort=args.reasoning_effort,
    )
    result = run_hybrid_agent(ctx) if args.mode == "hybrid" else run_react_agent(ctx)
    if args.out:
        _write_json(args.out, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("ok") else 1


def cmd_demo(args: argparse.Namespace) -> int:
    cfg = LogicDBConfig(
        duckdb_path=args.duckdb_path,
        base_schema=args.base_schema,
        workdir=args.workdir,
        layout_model=args.layout_model,
        index_model=args.index_model,
        mv_model=args.mv_model,
        reasoning_effort=args.reasoning_effort,
        max_indexes=args.max_indexes,
        max_mvs=args.max_mvs,
        maintenance_budget_ms=args.maintenance_budget_ms,
        risk_tolerance=args.risk_tolerance,
        parquet_work_dir=args.parquet_work_dir,
        allow_heuristic_fallbacks=not bool(args.disable_heuristic_fallbacks),
    )
    result = run_openops_demo_from_jsonl(cfg, args.workload_jsonl)
    payload = result.to_dict()
    if args.out:
        _write_json(args.out, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def cmd_mine_skills(args: argparse.Namespace) -> int:
    bank = mine_skills(
        predictions_path=args.predictions,
        trajectory_dir=args.trajectory_dir,
        dataset=args.dataset,
        min_support=args.min_support,
        min_success_rate=args.min_success_rate,
        max_skills=args.max_skills,
        top_keywords=args.top_keywords,
        strict_answer_success=args.strict_answer_success,
    )
    _write_json(args.out, bank)
    print(json.dumps({"skills": len(bank.get("skills", [])), "out": args.out}, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="logicdb")
    sub = parser.add_subparsers(dest="command", required=True)

    p_agent = sub.add_parser("agent", help="Run the foreground analytics agent.")
    p_agent.add_argument("--dataset", choices=["spider", "bird", "wtq", "tabfact"], default="spider")
    p_agent.add_argument("--mode", choices=["code", "hybrid"], default="code")
    p_agent.add_argument("--idx", type=int, default=0)
    p_agent.add_argument("--question")
    p_agent.add_argument("--db-id")
    p_agent.add_argument("--db-path")
    p_agent.add_argument("--model", default="gpt-5-mini")
    p_agent.add_argument("--max-turns", type=int, default=10)
    p_agent.add_argument("--reasoning-effort", default=None)
    p_agent.add_argument("--skill-bank-path", default=None)
    p_agent.add_argument("--skill-top-k", type=int, default=0)
    p_agent.add_argument("--parquet-dir", default=None)
    p_agent.add_argument("--spider-root", default=DEFAULT_SPIDER_ROOT)
    p_agent.add_argument("--bird-root", default=DEFAULT_BIRD_ROOT)
    p_agent.add_argument("--wtq-root", default=DEFAULT_WTQ_ROOT)
    p_agent.add_argument("--tabfact-root", default=DEFAULT_TABFACT_ROOT)
    p_agent.add_argument("--wtq-split", default="pristine-unseen-tables")
    p_agent.add_argument("--tabfact-split", default="val")
    p_agent.add_argument("--out", default=None)
    p_agent.set_defaults(func=cmd_agent)

    p_demo = sub.add_parser("demo-openops", help="Run a compact end-to-end OpenOps demo on DuckDB.")
    p_demo.add_argument("--duckdb-path", required=True)
    p_demo.add_argument("--workload-jsonl", required=True)
    p_demo.add_argument("--base-schema", default="main")
    p_demo.add_argument("--workdir", default="./logicdb_demo")
    p_demo.add_argument("--layout-model", default="gpt-5-mini")
    p_demo.add_argument("--index-model", default="gpt-5-mini")
    p_demo.add_argument("--mv-model", default="gpt-5-mini")
    p_demo.add_argument("--reasoning-effort", default=None)
    p_demo.add_argument("--max-indexes", type=int, default=6)
    p_demo.add_argument("--max-mvs", type=int, default=6)
    p_demo.add_argument("--maintenance-budget-ms", type=float, default=100000.0)
    p_demo.add_argument("--risk-tolerance", type=float, default=0.7)
    p_demo.add_argument("--parquet-work-dir", default=None)
    p_demo.add_argument("--disable-heuristic-fallbacks", action="store_true")
    p_demo.add_argument("--out", default=None)
    p_demo.set_defaults(func=cmd_demo)

    p_skill = sub.add_parser("mine-skills", help="Mine reusable skills from trajectory logs.")
    p_skill.add_argument("--predictions", required=True)
    p_skill.add_argument("--trajectory-dir", required=True)
    p_skill.add_argument("--dataset", required=True)
    p_skill.add_argument("--out", required=True)
    p_skill.add_argument("--min-support", type=int, default=3)
    p_skill.add_argument("--min-success-rate", type=float, default=0.6)
    p_skill.add_argument("--max-skills", type=int, default=32)
    p_skill.add_argument("--top-keywords", type=int, default=6)
    p_skill.add_argument("--strict-answer-success", action="store_true")
    p_skill.set_defaults(func=cmd_mine_skills)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
