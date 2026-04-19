#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lightweight skill bank utilities for NL2Analytics.

A "skill" here is a compact plan distilled from successful trajectories.
Runtime usage is retrieval-only (no RL needed):
- offline: mine skill bank from trajectories
- online: retrieve top-k skills for the current question and inject as hints
"""

from __future__ import annotations

import json
import math
import os
import re
import threading
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

_STOPWORDS: Set[str] = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how", "in",
    "is", "it", "of", "on", "or", "that", "the", "to", "was", "were", "what", "when",
    "where", "which", "who", "whom", "why", "with", "within", "without", "into", "than",
    "then", "them", "they", "their", "this", "those", "these", "do", "does", "did", "done",
    "has", "have", "had", "can", "could", "should", "would", "will", "may", "might", "must",
}

_BANK_CACHE: Dict[str, Tuple[float, Dict[str, Any]]] = {}
_BANK_CACHE_LOCK = threading.Lock()


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return float(default)
    try:
        return float(raw)
    except Exception:
        return float(default)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return int(default)
    try:
        return int(raw)
    except Exception:
        return int(default)


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    s = str(raw).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)


def tokenize_question(text: str) -> List[str]:
    if not text:
        return []
    toks = re.findall(r"[a-z0-9]+", str(text).lower())
    return [t for t in toks if len(t) >= 2 and t not in _STOPWORDS]


def infer_intent_tags(question: str) -> List[str]:
    q = str(question or "").lower()
    tags: List[str] = []

    if re.search(r"\b(count|how many|number of|total|sum|average|avg|mean|ratio|percent|percentage)\b", q):
        tags.append("aggregation")
    if re.search(r"\b(max|min|highest|lowest|largest|smallest|most|least|top|best|worst|newest|oldest)\b", q):
        tags.append("superlative")
    if re.search(r"\b(more than|less than|greater than|fewer than|between|compared|compare)\b", q):
        tags.append("comparison")
    if re.search(r"\b(before|after|year|date|month|day|first|last|earliest|latest|during)\b", q):
        tags.append("temporal")
    if re.search(r"\b(both|either|neither|except|not in|intersection|intersect|union|difference)\b", q):
        tags.append("set_ops")
    if re.search(r"\b(true|false|entailed|refuted|support|supported|whether)\b", q):
        tags.append("boolean")
    if not tags:
        tags.append("lookup")
    return tags


def load_skill_bank(path: str) -> Dict[str, Any]:
    """Load skill bank with mtime cache."""
    if not path:
        return {"skills": []}
    ap = os.path.abspath(path)
    if not os.path.isfile(ap):
        return {"skills": []}
    try:
        mtime = os.path.getmtime(ap)
    except Exception:
        mtime = -1.0

    with _BANK_CACHE_LOCK:
        cached = _BANK_CACHE.get(ap)
        if cached and cached[0] == mtime:
            return cached[1]

    try:
        with open(ap, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        data = {"skills": []}

    if not isinstance(data, dict):
        data = {"skills": []}
    if not isinstance(data.get("skills"), list):
        data["skills"] = []

    with _BANK_CACHE_LOCK:
        _BANK_CACHE[ap] = (mtime, data)
    return data


def _safe_list(val: Any) -> List[str]:
    if isinstance(val, list):
        out: List[str] = []
        for x in val:
            s = str(x or "").strip().lower()
            if s:
                out.append(s)
        return out
    return []


def _skill_overlaps(
    skill: Dict[str, Any],
    q_tokens: Set[str],
    q_intents: Set[str],
) -> Tuple[float, float]:
    s_kw = set(_safe_list(skill.get("trigger_keywords")))
    s_tags = set(_safe_list(skill.get("intent_tags")))
    kw_overlap = (len(q_tokens & s_kw) / max(1, len(s_kw))) if s_kw else 0.0
    tag_overlap = (len(q_intents & s_tags) / max(1, len(s_tags))) if s_tags else 0.0
    return kw_overlap, tag_overlap


def _score_skill(
    skill: Dict[str, Any],
    q_tokens: Set[str],
    q_intents: Set[str],
    dataset: str,
    mode: Optional[str],
) -> float:
    s_dataset = str(skill.get("dataset") or "all").strip().lower()
    if s_dataset not in {"all", str(dataset or "").strip().lower()}:
        return -1.0

    s_mode = str(skill.get("source_mode") or "all").strip().lower()
    if mode and s_mode not in {"all", mode.strip().lower()}:
        return -1.0

    kw_overlap, tag_overlap = _skill_overlaps(skill, q_tokens, q_intents)

    support = float(skill.get("support", 0) or 0)
    success_rate = float(skill.get("success_rate", 0) or 0)
    support_bonus = min(1.0, math.log1p(max(0.0, support)) / math.log(64.0))

    # Overlap is primary; support/success are tie-breakers.
    return 2.0 * kw_overlap + 1.3 * tag_overlap + 0.35 * support_bonus + 0.35 * success_rate


def retrieve_skills(
    bank: Dict[str, Any],
    question: str,
    dataset: str,
    mode: Optional[str] = None,
    top_k: int = 3,
) -> List[Dict[str, Any]]:
    if top_k <= 0:
        return []
    skills = bank.get("skills") if isinstance(bank, dict) else None
    if not isinstance(skills, list) or not skills:
        return []

    q_tokens = set(tokenize_question(question))
    q_intents = set(infer_intent_tags(question))

    # Optional retrieval gating (env-driven to keep API stable).
    # Example:
    #   LOGICDB_SKILL_MIN_SCORE=1.2
    #   LOGICDB_SKILL_MIN_SUPPORT=20
    #   LOGICDB_SKILL_MIN_SUCCESS=0.6
    #   LOGICDB_SKILL_REQUIRE_TAG_OVERLAP=1
    #   LOGICDB_SKILL_MAX_TOP_GAP=0.55
    min_score = _env_float("LOGICDB_SKILL_MIN_SCORE", _env_float("AIDB_SKILL_MIN_SCORE", 0.01))
    min_support = _env_int("LOGICDB_SKILL_MIN_SUPPORT", _env_int("AIDB_SKILL_MIN_SUPPORT", 0))
    min_success = _env_float("LOGICDB_SKILL_MIN_SUCCESS", _env_float("AIDB_SKILL_MIN_SUCCESS", 0.0))
    require_tag_overlap = _env_bool("LOGICDB_SKILL_REQUIRE_TAG_OVERLAP", _env_bool("AIDB_SKILL_REQUIRE_TAG_OVERLAP", False))
    require_kw_overlap = _env_bool("LOGICDB_SKILL_REQUIRE_KW_OVERLAP", _env_bool("AIDB_SKILL_REQUIRE_KW_OVERLAP", False))
    max_top_gap = _env_float("LOGICDB_SKILL_MAX_TOP_GAP", _env_float("AIDB_SKILL_MAX_TOP_GAP", 0.0))

    ranked: List[Tuple[float, Dict[str, Any]]] = []
    for s in skills:
        if not isinstance(s, dict):
            continue
        support = float(s.get("support", 0) or 0)
        success_rate = float(s.get("success_rate", 0) or 0)
        if support < float(min_support):
            continue
        if success_rate < float(min_success):
            continue
        kw_overlap, tag_overlap = _skill_overlaps(s, q_tokens, q_intents)
        if require_tag_overlap and tag_overlap <= 0.0:
            continue
        if require_kw_overlap and kw_overlap <= 0.0:
            continue
        sc = _score_skill(s, q_tokens, q_intents, dataset, mode)
        if sc < 0:
            continue
        ranked.append((sc, s))

    ranked.sort(
        key=lambda x: (
            x[0],
            float(x[1].get("support", 0) or 0),
            float(x[1].get("success_rate", 0) or 0),
        ),
        reverse=True,
    )

    out: List[Dict[str, Any]] = []
    top_score: Optional[float] = None
    for score, skill in ranked:
        if score < min_score:
            continue
        if top_score is None:
            top_score = score
        elif max_top_gap > 0.0 and (top_score - score) > max_top_gap:
            # Skip weak secondary matches that often cause negative transfer.
            continue
        row = dict(skill)
        row["_retrieval_score"] = round(score, 4)
        out.append(row)
        if len(out) >= top_k:
            break
    return out


def format_skill_hint(skills: List[Dict[str, Any]], max_chars: int = 900) -> str:
    if not skills:
        return ""
    lines: List[str] = []
    for i, s in enumerate(skills, 1):
        sid = str(s.get("skill_id") or f"skill_{i}")
        plan = str(s.get("tool_pattern") or "")
        support = int(s.get("support", 0) or 0)
        sr = float(s.get("success_rate", 0) or 0)
        kws = _safe_list(s.get("trigger_keywords"))[:5]
        tags = _safe_list(s.get("intent_tags"))[:4]
        hint = str(s.get("hint") or "").strip()

        lines.append(f"{i}) {sid} | plan: {plan or 'n/a'} | support={support}, success={sr:.2f}")
        if kws:
            lines.append(f"   trigger: {', '.join(kws)}")
        if tags:
            lines.append(f"   intents: {', '.join(tags)}")
        if hint:
            lines.append(f"   hint: {hint}")

    text = "\n".join(lines).strip()
    if max_chars > 0 and len(text) > max_chars:
        text = text[: max(0, max_chars - 3)].rstrip() + "..."
    return text


def build_skill_hint(
    *,
    question: str,
    dataset: str,
    mode: Optional[str],
    skill_bank_path: Optional[str],
    top_k: int = 0,
    max_chars: int = 900,
) -> Dict[str, Any]:
    """Retrieve and format skill hints for one question."""
    if not skill_bank_path or top_k <= 0:
        return {"text": "", "skills": [], "skill_ids": [], "count": 0, "bank_path": skill_bank_path}

    bank = load_skill_bank(skill_bank_path)
    hits = retrieve_skills(bank, question=question, dataset=dataset, mode=mode, top_k=top_k)
    hint_text = format_skill_hint(hits, max_chars=max_chars)
    ids = [str(s.get("skill_id") or "") for s in hits if str(s.get("skill_id") or "").strip()]
    return {
        "text": hint_text,
        "skills": hits,
        "skill_ids": ids,
        "count": len(hits),
        "bank_path": os.path.abspath(skill_bank_path),
    }


def compact_tool_sequence(tools: Iterable[str]) -> List[str]:
    """Remove empty names and consecutive duplicates."""
    out: List[str] = []
    prev = None
    for raw in tools:
        t = str(raw or "").strip()
        if not t:
            continue
        if t == prev:
            continue
        out.append(t)
        prev = t
    return out
