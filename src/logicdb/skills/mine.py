#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mine reusable "skills" from successful NL2Analytics trajectories.

A skill is a compact, high-support operator pattern:
- trigger (keywords + intents)
- tool plan (ordered tool sequence)
- short hint + one example

Usage:
  python -m logicdb.skills.mine \
    --predictions runs/wtq/predictions.jsonl \
    --trajectory_dir runs/wtq/trajectories \
    --dataset wtq \
    --out runs/skills/wtq_skill_bank.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from .library import compact_tool_sequence, infer_intent_tags, tokenize_question

try:
    from logicdb.agent.eval import _has_answer_target, answer_match, exec_match
except Exception:
    _has_answer_target = None  # type: ignore
    answer_match = None  # type: ignore
    exec_match = None  # type: ignore


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            s = ln.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
                if isinstance(obj, dict):
                    rows.append(obj)
            except Exception:
                continue
    return rows


def _load_traj_events(trajectory_dir: str, idx: int) -> List[Dict[str, Any]]:
    p = os.path.join(trajectory_dir, f"idx{idx}.jsonl")
    if not os.path.isfile(p):
        return []
    out: List[Dict[str, Any]] = []
    try:
        with open(p, "r", encoding="utf-8") as f:
            for ln in f:
                s = ln.strip()
                if not s:
                    continue
                try:
                    obj = json.loads(s)
                except Exception:
                    continue
                if isinstance(obj, dict):
                    out.append(obj)
    except Exception:
        return []
    return out


def _infer_dataset(rec: Dict[str, Any], fallback: str) -> str:
    ds = str(rec.get("source_dataset") or "").strip().lower()
    if ds:
        return ds
    db_id = str(rec.get("db_id") or "").strip().lower()
    for pref in ("wtq_", "tabfact_", "bird", "spider"):
        if db_id.startswith(pref):
            return pref.replace("_", "")
    return str(fallback or "").strip().lower() or "unknown"


def _is_success(rec: Dict[str, Any], strict_answer_success: bool = True) -> bool:
    if not bool(rec.get("ok", False)):
        return False
    if _has_answer_target is None:
        return True
    try:
        if bool(_has_answer_target(rec)):
            if not strict_answer_success or answer_match is None:
                return bool(rec.get("ok", False))
            match, _, _, _ = answer_match(rec.get("predicted_result"), rec.get("gold_answer"), rec)
            return bool(match)
    except Exception:
        return bool(rec.get("ok", False))
    # SQL-target tasks: verify execution match when gold SQL is available.
    if exec_match is not None:
        try:
            gold_sql = str(rec.get("gold_sql") or "").strip()
            db_path = str(rec.get("db_path") or "").strip()
            if gold_sql and db_path:
                match, _, _, _ = exec_match(rec.get("predicted_result"), gold_sql, db_path)
                return bool(match)
        except Exception:
            return bool(rec.get("ok", False))
    return bool(rec.get("ok", False))


def _choose_source_mode(rec: Dict[str, Any]) -> str:
    mode = str(rec.get("agent_mode") or "").strip().lower()
    if mode == "hybrid":
        hy = rec.get("hybrid")
        if isinstance(hy, dict):
            chosen = str(hy.get("chosen_source") or "").strip().lower()
            if chosen in {"code", "sql"}:
                return chosen
    if mode in {"code", "sql"}:
        return mode
    # Default to code, since trajectories without explicit source are usually code path.
    return "code"


def _extract_tool_sequence(events: List[Dict[str, Any]], source_mode: str) -> List[str]:
    tools: List[str] = []
    source_mode = str(source_mode or "").strip().lower()
    for ev in events:
        if str(ev.get("event") or "") != "act":
            continue
        ev_src = str(ev.get("source") or ev.get("subagent") or "").strip().lower()

        if source_mode:
            if ev_src:
                if ev_src != source_mode:
                    continue
            else:
                # No source annotation appears in code-only trajectories.
                if source_mode != "code":
                    continue

        t = str(ev.get("tool") or "").strip()
        if t:
            tools.append(t)
    return compact_tool_sequence(tools)


def _extract_example_artifact(rec: Dict[str, Any], source_mode: str, max_len: int = 280) -> str:
    sub = rec.get("subagent_outputs")
    if not isinstance(sub, dict):
        return ""
    src = sub.get(source_mode)
    if not isinstance(src, dict):
        return ""
    if source_mode == "code":
        txt = str(src.get("last_execute_code") or "").strip()
    else:
        txt = str(src.get("last_processed_sql") or src.get("last_sql") or "").strip()
    if len(txt) > max_len:
        txt = txt[: max_len - 3].rstrip() + "..."
    return txt


def _build_hint(source_mode: str, seq: List[str], intents: List[str]) -> str:
    notes: List[str] = []
    sseq = set(seq)
    if "get_join_path" in sseq:
        notes.append("Use join-path discovery before merges to avoid guessed keys")
    if source_mode == "code" and {"read_table", "execute_python"}.issubset(sseq):
        notes.append("Read needed table(s) once and finish logic in one execute_python call")
    if source_mode == "sql" and "query_sql" in sseq:
        notes.append("Stop after first successful query_sql unless output shape is clearly wrong")
    if "aggregation" in intents:
        notes.append("For aggregation/count intents, verify filter scope before computing")
    if "boolean" in intents:
        notes.append("For binary judgment tasks, map output strictly to canonical labels")
    if not notes:
        notes.append("Follow the retrieved operator order and keep each step executable")
    return ". ".join(notes[:2]) + "."


def _merge_existing(
    existing: List[Dict[str, Any]],
    fresh: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    key_seen = set()
    merged: List[Dict[str, Any]] = []

    def _k(x: Dict[str, Any]) -> Tuple[str, str, str]:
        return (
            str(x.get("dataset") or "all").strip().lower(),
            str(x.get("source_mode") or "all").strip().lower(),
            str(x.get("tool_pattern") or "").strip().lower(),
        )

    for row in existing + fresh:
        if not isinstance(row, dict):
            continue
        k = _k(row)
        if k in key_seen:
            continue
        key_seen.add(k)
        merged.append(row)
    return merged


def mine_skills(
    *,
    predictions_path: str,
    trajectory_dir: str,
    dataset: str,
    min_support: int,
    min_success_rate: float,
    max_skills: int,
    top_keywords: int,
    strict_answer_success: bool,
) -> Dict[str, Any]:
    preds = _load_jsonl(predictions_path)

    total_by_key: Dict[str, int] = defaultdict(int)
    support_by_key: Dict[str, int] = defaultdict(int)
    turns_sum_by_key: Dict[str, float] = defaultdict(float)
    kw_by_key: Dict[str, Counter] = defaultdict(Counter)
    intent_by_key: Dict[str, Counter] = defaultdict(Counter)
    example_q_by_key: Dict[str, str] = {}
    example_artifact_by_key: Dict[str, str] = {}
    source_by_key: Dict[str, str] = {}
    seq_by_key: Dict[str, List[str]] = {}
    dataset_by_key: Dict[str, str] = {}

    stats = {
        "records_total": len(preds),
        "records_with_trajectory": 0,
        "records_with_actions": 0,
        "records_success": 0,
        "records_success_with_actions": 0,
    }

    for rec in preds:
        idx_raw = rec.get("idx")
        try:
            idx = int(idx_raw)
        except Exception:
            continue

        events = _load_traj_events(trajectory_dir, idx)
        if not events:
            continue
        stats["records_with_trajectory"] += 1

        ds = _infer_dataset(rec, dataset)
        source_mode = _choose_source_mode(rec)
        seq = _extract_tool_sequence(events, source_mode)
        if not seq:
            continue
        stats["records_with_actions"] += 1

        key = f"{ds}|{source_mode}|{'->'.join(seq)}"
        total_by_key[key] += 1
        source_by_key[key] = source_mode
        seq_by_key[key] = seq
        dataset_by_key[key] = ds

        success = _is_success(rec, strict_answer_success=strict_answer_success)
        if success:
            stats["records_success"] += 1
            support_by_key[key] += 1
            stats["records_success_with_actions"] += 1

            q = str(rec.get("question") or "").strip()
            if q and key not in example_q_by_key:
                example_q_by_key[key] = q
            art = _extract_example_artifact(rec, source_mode=source_mode)
            if art and key not in example_artifact_by_key:
                example_artifact_by_key[key] = art
            turns_sum_by_key[key] += float(rec.get("turns", 0) or 0)
            kw_by_key[key].update(tokenize_question(q))
            intent_by_key[key].update(infer_intent_tags(q))

    rows: List[Dict[str, Any]] = []
    for key, support in support_by_key.items():
        total_seen = int(total_by_key.get(key, 0) or 0)
        if total_seen <= 0:
            continue
        success_rate = support / total_seen
        if support < max(1, int(min_support)):
            continue
        if success_rate < float(min_success_rate):
            continue

        seq = seq_by_key.get(key, [])
        source_mode = source_by_key.get(key, "code")
        ds = dataset_by_key.get(key, dataset)
        intent_counter = intent_by_key.get(key, Counter())
        kw_counter = kw_by_key.get(key, Counter())

        dynamic_intent_floor = max(1, int(math.ceil(0.25 * support)))
        intents = [k for k, v in intent_counter.most_common() if v >= dynamic_intent_floor][:4]
        if not intents:
            intents = [k for k, _ in intent_counter.most_common(2)]
        kws = [
            k
            for k, _ in kw_counter.most_common(max(1, int(top_keywords)))
            if not k.isdigit() and len(k) >= 3
        ]

        avg_turns = (turns_sum_by_key.get(key, 0.0) / support) if support > 0 else 0.0
        rows.append(
            {
                "dataset": ds,
                "source_mode": source_mode,
                "tool_pattern": " -> ".join(seq),
                "tool_plan": seq,
                "intent_tags": intents,
                "trigger_keywords": kws,
                "support": int(support),
                "total_seen": int(total_seen),
                "success_rate": round(float(success_rate), 4),
                "avg_turns": round(float(avg_turns), 3),
                "hint": _build_hint(source_mode, seq, intents),
                "example_question": example_q_by_key.get(key, ""),
                "example_artifact": example_artifact_by_key.get(key, ""),
            }
        )

    rows.sort(
        key=lambda x: (
            int(x.get("support", 0) or 0),
            float(x.get("success_rate", 0) or 0),
            -float(x.get("avg_turns", 0) or 0),
        ),
        reverse=True,
    )

    if max_skills > 0:
        rows = rows[:max_skills]

    ds_prefix = str(dataset or "all").strip().lower() or "all"
    for i, r in enumerate(rows, 1):
        src = str(r.get("source_mode") or "all").strip().lower()
        r["skill_id"] = f"{ds_prefix}_{src}_s{i:03d}"

    out = {
        "version": "0.1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "predictions": os.path.abspath(predictions_path),
            "trajectory_dir": os.path.abspath(trajectory_dir),
            "dataset": dataset,
            "strict_answer_success": bool(strict_answer_success),
        },
        "stats": {
            **stats,
            "raw_patterns": len(total_by_key),
            "kept_skills": len(rows),
            "min_support": int(min_support),
            "min_success_rate": float(min_success_rate),
        },
        "skills": rows,
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Mine reusable skills from successful NL2Analytics trajectories")
    ap.add_argument("--predictions", required=True, help="Prediction JSONL from run_batch_eval")
    ap.add_argument("--trajectory_dir", required=True, help="Trajectory dir from run_batch_eval")
    ap.add_argument("--dataset", default="", help="Dataset name used for skill_id prefix")
    ap.add_argument("--out", required=True, help="Output skill bank JSON")
    ap.add_argument("--min_support", type=int, default=5, help="Minimum successful support per skill")
    ap.add_argument("--min_success_rate", type=float, default=0.55, help="Minimum success_rate for kept skills")
    ap.add_argument("--max_skills", type=int, default=64, help="Maximum number of exported skills")
    ap.add_argument("--top_keywords", type=int, default=6, help="Top trigger keywords per skill")
    ap.add_argument(
        "--strict_answer_success",
        action="store_true",
        help="For WTQ/TabFact, only trajectories with answer-level correctness are treated as success",
    )
    ap.add_argument(
        "--append_existing",
        action="store_true",
        help="If --out exists, merge old+new skills by (dataset, source_mode, tool_pattern)",
    )
    args = ap.parse_args()

    if not os.path.isfile(args.predictions):
        raise FileNotFoundError(f"predictions not found: {args.predictions}")
    if not os.path.isdir(args.trajectory_dir):
        raise FileNotFoundError(f"trajectory_dir not found: {args.trajectory_dir}")

    mined = mine_skills(
        predictions_path=args.predictions,
        trajectory_dir=args.trajectory_dir,
        dataset=args.dataset,
        min_support=args.min_support,
        min_success_rate=args.min_success_rate,
        max_skills=args.max_skills,
        top_keywords=args.top_keywords,
        strict_answer_success=bool(args.strict_answer_success),
    )

    if args.append_existing and os.path.isfile(args.out):
        try:
            with open(args.out, "r", encoding="utf-8") as f:
                old = json.load(f)
            old_skills = old.get("skills") if isinstance(old, dict) else []
            if not isinstance(old_skills, list):
                old_skills = []
        except Exception:
            old_skills = []
        merged = _merge_existing(old_skills, mined.get("skills", []))
        for i, row in enumerate(merged, 1):
            if not str(row.get("skill_id") or "").strip():
                ds = str(row.get("dataset") or args.dataset or "all").strip().lower()
                src = str(row.get("source_mode") or "all").strip().lower()
                row["skill_id"] = f"{ds}_{src}_s{i:03d}"
        mined["skills"] = merged
        mined.setdefault("stats", {})["kept_skills"] = len(merged)
        mined.setdefault("stats", {})["merged_with_existing"] = len(old_skills)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(mined, f, ensure_ascii=False, indent=2)

    st = mined.get("stats", {})
    print(
        "[skill-mine] "
        f"records={st.get('records_total', 0)} "
        f"with_traj={st.get('records_with_trajectory', 0)} "
        f"patterns={st.get('raw_patterns', 0)} "
        f"skills={st.get('kept_skills', 0)} -> {os.path.abspath(args.out)}"
    )


if __name__ == "__main__":
    main()
