#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lightweight deterministic cost-gating heuristics for co-opt candidates.

The goal is not exact cardinality estimation; it is a conservative net-gain
screen to avoid applying candidates that are unlikely to amortize build/apply
cost within the remaining workload horizon.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Tuple


def _env(name: str, legacy_name: str, default: str) -> str:
    return str(os.getenv(name, os.getenv(legacy_name, default)))


MV_GATE_ALLOW_NON_SEED = _env("LOGICDB_MV_GATE_ALLOW_NON_SEED", "AIDB_MV_GATE_ALLOW_NON_SEED", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}


def _env_float(name: str, default: float) -> float:
    try:
        legacy_name = name.replace("LOGICDB_", "AIDB_", 1) if name.startswith("LOGICDB_") else name
        return float(_env(name, legacy_name, str(default)))
    except Exception:
        return float(default)


MV_GATE_NON_SEED_SCORE_MULT = max(1.0, _env_float("LOGICDB_MV_GATE_NON_SEED_SCORE_MULT", 2.0))
MV_GATE_NON_SEED_MIN_SCORE_MS = max(0.0, _env_float("LOGICDB_MV_GATE_NON_SEED_MIN_SCORE_MS", 5000.0))
MV_GATE_NON_SEED_MIN_COVERAGE = min(
    1.0,
    max(0.0, _env_float("LOGICDB_MV_GATE_NON_SEED_MIN_COVERAGE", 0.35)),
)
MV_GATE_PREFERRED_SEED_SCORE_MULT = max(
    0.0,
    _env_float("LOGICDB_MV_GATE_PREFERRED_SEED_SCORE_MULT", 0.50),
)
MV_GATE_PREFERRED_SEED_MIN_COVERAGE = min(
    1.0,
    max(0.0, _env_float("LOGICDB_MV_GATE_PREFERRED_SEED_MIN_COVERAGE", 0.20)),
)
MV_GATE_PREFERRED_SEED_MIN_OBS_MATCHES = max(
    0,
    int(round(_env_float("LOGICDB_MV_GATE_PREFERRED_SEED_MIN_OBS_MATCHES", 12.0))),
)
MV_GATE_PREFERRED_TPCH_SEEDS = {
    "SEED_T1",
    "SEED_T7",
    "SEED_T4",
    "SEED_T10",
    "SEED_T13",
    "SEED_T18",
    "SEED_T6",
    "SEED_T14",
    "SEED_T12",
    "SEED_T5",
    "SEED_T8",
}
INDEX_BUILD_ROWS_PER_S = max(1e3, _env_float("LOGICDB_INDEX_BUILD_ROWS_PER_S", 250000.0))
INDEX_BUILD_ROW_OVERHEAD_MS = max(0.0, _env_float("LOGICDB_INDEX_BUILD_ROW_OVERHEAD_MS", 250.0))


def _as_int(v: Any) -> Optional[int]:
    try:
        iv = int(v)
        return iv if iv >= 0 else None
    except Exception:
        return None


def _template_of(q: Dict[str, Any]) -> Optional[int]:
    t = _as_int(q.get("template"))
    if t is not None:
        return t
    qid = str(q.get("qid") or q.get("query_id") or q.get("id") or "")
    m = re.search(r"\bQ?(\d+)\b", qid, flags=re.IGNORECASE)
    if m:
        return _as_int(m.group(1))
    return None


def _effective_min_gain_ms(
    min_net_gain_ms: float,
    future_span: int,
    horizon_queries: int,
) -> float:
    """Scale threshold by remaining amortization window.

    When remaining queries are far fewer than horizon_queries, a fixed
    threshold can become too conservative and reject candidates with positive
    net benefit. We downscale by coverage ratio while keeping a tiny floor.
    """
    base = max(0.0, float(min_net_gain_ms))
    h = max(1, int(horizon_queries))
    span = max(0, int(future_span))
    ratio = max(0.0, min(1.0, float(span) / float(h)))
    return max(50.0, base * ratio)


def estimate_future_template_freq(
    queries: List[Dict[str, Any]],
    start_idx: int,
    horizon_queries: int,
) -> Dict[int, int]:
    end = min(len(queries), max(start_idx, 0) + max(0, int(horizon_queries)))
    out: Dict[int, int] = {}
    for i in range(max(0, start_idx), end):
        tpl = _template_of(queries[i])
        if tpl is None:
            continue
        out[tpl] = out.get(tpl, 0) + 1
    return out


def estimate_template_freq_range(
    queries: List[Dict[str, Any]],
    start_idx: int,
    end_idx: int,
) -> Dict[int, int]:
    s = max(0, int(start_idx))
    e = min(len(queries), max(s, int(end_idx)))
    out: Dict[int, int] = {}
    for i in range(s, e):
        tpl = _template_of(queries[i])
        if tpl is None:
            continue
        out[tpl] = out.get(tpl, 0) + 1
    return out


def _coverage_factor(
    observed_matches: int,
    future_matches: int,
    floor: float = 0.20,
) -> float:
    fm = max(0, int(future_matches))
    om = max(0, int(observed_matches))
    if fm <= 0:
        return 1.0
    raw = float(om) / float(max(1, fm))
    return max(float(floor), min(1.0, raw))


def _score_terms(
    *,
    gain_ms: float,
    apply_ms: float,
    coverage_factor: float = 1.0,
    uncertainty_mult: float = 1.0,
    switch_penalty_ratio: float = 0.0,
    risk_penalty_ratio: float = 0.0,
    coverage_risk_weight: float = 0.0,
    uncertainty_risk_weight: float = 0.0,
    interference_hits: int = 0,
    interference_penalty_ms_per_hit: float = 0.0,
    churn_penalty_ms: float = 0.0,
) -> Dict[str, float]:
    """Unified joint score decomposition used by all three modules."""
    g = max(0.0, float(gain_ms))
    a = max(0.0, float(apply_ms))
    cov = max(0.0, min(1.0, float(coverage_factor)))
    um = max(0.0, min(1.0, float(uncertainty_mult)))
    sw_ratio = max(0.0, float(switch_penalty_ratio))
    rk_ratio = max(0.0, float(risk_penalty_ratio))
    cov_w = max(0.0, float(coverage_risk_weight))
    unc_w = max(0.0, float(uncertainty_risk_weight))
    ih = max(0, int(interference_hits))
    ih_ms = max(0.0, float(interference_penalty_ms_per_hit))
    churn_ms = max(0.0, float(churn_penalty_ms))

    switch_penalty_ms = a * sw_ratio + churn_ms
    risk_penalty_ms = (
        g * rk_ratio
        + g * max(0.0, 1.0 - cov) * cov_w
        + g * max(0.0, 1.0 - um) * unc_w
    )
    interference_penalty_ms = float(ih) * ih_ms
    score_ms = g - a - switch_penalty_ms - risk_penalty_ms - interference_penalty_ms
    net_gain_ms = g - a
    return {
        "gain_ms": float(g),
        "apply_ms": float(a),
        "switch_penalty_ms": float(switch_penalty_ms),
        "risk_penalty_ms": float(risk_penalty_ms),
        "interference_penalty_ms": float(interference_penalty_ms),
        "score_ms": float(score_ms),
        "net_gain_ms": float(net_gain_ms),
    }


def estimate_template_ms(
    queries: List[Dict[str, Any]],
    observed_per_query_ms: Dict[int, float],
    workload_stats: Optional[Dict[str, Any]] = None,
    default_ms: float = 250.0,
) -> Dict[int, float]:
    """Estimate template latency from observed timings + workload_stats priors.

    Important behavior:
    - When we have partial observations, we still backfill *unseen* templates
      using workload_stats (EXPLAIN-derived relative costs), instead of
      returning early with only observed templates.
    """
    tpl_vals: Dict[int, List[float]] = {}
    for qidx, ms in (observed_per_query_ms or {}).items():
        if qidx < 0 or qidx >= len(queries):
            continue
        tpl = _template_of(queries[qidx])
        if tpl is None:
            continue
        try:
            fms = float(ms)
        except Exception:
            continue
        if fms <= 0:
            continue
        tpl_vals.setdefault(tpl, []).append(fms)

    tpl_ms: Dict[int, float] = {}
    for tpl, vals in tpl_vals.items():
        vals = sorted(vals)
        tpl_ms[tpl] = vals[len(vals) // 2]

    # Backfill unseen templates from workload_stats top_slow_templates
    # (EXPLAIN-derived relative costs).
    if workload_stats and isinstance(workload_stats, dict):
        slow = workload_stats.get("top_slow_templates") or []
        if isinstance(slow, list) and slow:
            max_cost = 0.0
            for item in slow:
                try:
                    max_cost = max(max_cost, float(item[1]))
                except Exception:
                    continue
            if max_cost > 0:
                base_default = float(default_ms)
                if tpl_ms:
                    obs_vals = sorted(float(v) for v in tpl_ms.values())
                    if obs_vals:
                        base_default = max(base_default, obs_vals[len(obs_vals) // 2])
                # Map relative cost rank into [base_default, 4*base_default].
                for item in slow:
                    try:
                        tpl = int(item[0])
                        cost = float(item[1])
                    except Exception:
                        continue
                    rel = max(0.0, min(1.0, cost / max_cost))
                    tpl_ms.setdefault(tpl, float(base_default * (1.0 + 3.0 * rel)))

    return tpl_ms


def _has_range_pred(sql_l: str) -> bool:
    return bool(re.search(r"\bbetween\b|>=|<=|\s<\s|\s>\s", sql_l))


def _has_join(sql_l: str) -> bool:
    return " join " in f" {sql_l} "


def _sql_has_identifier(sql_l: str, ident: str) -> bool:
    i = str(ident or "").strip().lower()
    if not i:
        return False
    # Accept bare and schema-qualified forms; avoid substring collisions
    # such as `part` accidentally matching `partsupp`.
    return re.search(rf"(?:^|[^a-z0-9_])(?:[a-z0-9_]+\.)?{re.escape(i)}(?:$|[^a-z0-9_])", sql_l) is not None


def _speed_for_index(sql_l: str, cols: List[str], base_speedup: float) -> float:
    s = float(base_speedup)
    if cols:
        hit_cnt = sum(1 for c in cols if c and c in sql_l)
        if hit_cnt >= len(cols):
            s *= 1.20
        elif hit_cnt > 0:
            s *= 1.05
    if _has_join(sql_l):
        s *= 1.15
    if _has_range_pred(sql_l):
        s *= 1.10
    return max(0.02, min(0.65, s))


def _index_table_size_factor(row_count: int) -> float:
    rows = max(0, int(row_count))
    if rows <= 0:
        return 1.0
    if rows <= 250_000:
        return 0.20
    if rows <= 1_000_000:
        return 0.40
    if rows <= 5_000_000:
        return 0.70
    return 1.0


def _estimate_index_build_ms(idx: Any, build_mb_per_s: float) -> Tuple[float, float]:
    est_mb = 0.0
    try:
        est_mb = float(getattr(idx, "estimated_size_mb", 0.0) or 0.0)
    except Exception:
        est_mb = 0.0
    try:
        hinted_ms = float(getattr(idx, "_aidb_gate_apply_ms_hint", 0.0) or 0.0)
    except Exception:
        hinted_ms = 0.0
    try:
        row_count = int(getattr(idx, "row_count", 0) or 0)
    except Exception:
        row_count = 0
    cols = list(getattr(idx, "cols", []) or [])
    if est_mb <= 0:
        est_mb = 64.0 + 96.0 * max(1, len(cols))
    mbps = max(1e-6, float(build_mb_per_s))
    build_ms = (est_mb / mbps) * 1000.0 + 150.0
    if row_count > 0:
        row_floor_ms = (float(row_count) / float(INDEX_BUILD_ROWS_PER_S)) * 1000.0 + float(INDEX_BUILD_ROW_OVERHEAD_MS)
        build_ms = max(float(build_ms), float(row_floor_ms))
    if hinted_ms > 0.0:
        build_ms = max(float(build_ms), float(hinted_ms))
    return est_mb, build_ms


def gate_index_candidates(
    candidates: List[Any],
    queries: List[Dict[str, Any]],
    start_idx: int,
    observed_per_query_ms: Dict[int, float],
    workload_stats: Optional[Dict[str, Any]] = None,
    horizon_queries: int = 2000,
    min_net_gain_ms: float = 1000.0,
    build_mb_per_s: float = 250.0,
    base_speedup: float = 0.12,
    switch_penalty_ratio: float = 0.08,
    risk_penalty_ratio: float = 0.02,
    coverage_risk_weight: float = 0.25,
    uncertainty_risk_weight: float = 0.00,
    interference_tables: Optional[List[str]] = None,
    interference_penalty_ms_per_hit: float = 0.0,
    churn_penalty_ms: float = 0.0,
) -> Tuple[List[Any], Dict[str, Any]]:
    future_start = max(0, int(start_idx))
    future_end = min(len(queries), future_start + max(0, int(horizon_queries)))
    future_span = max(0, future_end - future_start)
    observed_start = max(0, future_start - max(1, int(horizon_queries)))
    observed_end = future_start
    no_observed_window = observed_end <= observed_start
    effective_min_gain_ms = _effective_min_gain_ms(min_net_gain_ms, future_span, horizon_queries)
    tpl_ms = estimate_template_ms(queries, observed_per_query_ms, workload_stats)
    global_ms = sorted(tpl_ms.values())[len(tpl_ms) // 2] if tpl_ms else 250.0
    interference_set = {str(x).strip().lower() for x in (interference_tables or []) if str(x).strip()}

    decisions: List[Dict[str, Any]] = []
    kept: List[Any] = []

    for idx in candidates or []:
        table = str(getattr(idx, "table", "") or "").strip().lower()
        cols = [str(c).strip().lower() for c in (getattr(idx, "cols", []) or []) if str(c).strip()]
        try:
            row_count = int(getattr(idx, "row_count", 0) or 0)
        except Exception:
            row_count = 0
        if not table:
            decisions.append({
                "candidate": str(getattr(idx, "cid", "?")),
                "keep": False,
                "net_gain_ms": -1.0,
                "reason": "missing_table",
            })
            continue

        benefit_ms = 0.0
        matched = 0
        observed_matched = 0
        table_size_factor = _index_table_size_factor(row_count)
        for i in range(future_start, future_end):
            q = queries[i]
            sql_l = str((q.get("sql") or q.get("query") or "")).lower()
            if not _sql_has_identifier(sql_l, table):
                continue
            if cols and not any(_sql_has_identifier(sql_l, c) for c in cols):
                continue
            tpl = _template_of(q)
            base_ms = float(tpl_ms.get(tpl, global_ms))
            speed = _speed_for_index(sql_l, cols, base_speedup) * float(table_size_factor)
            benefit_ms += base_ms * speed
            matched += 1

        for i in range(observed_start, observed_end):
            q = queries[i]
            sql_l = str((q.get("sql") or q.get("query") or "")).lower()
            if not _sql_has_identifier(sql_l, table):
                continue
            if cols and not any(_sql_has_identifier(sql_l, c) for c in cols):
                continue
            observed_matched += 1

        coverage_factor = _coverage_factor(observed_matched, matched, floor=0.20)
        benefit_adj_ms = benefit_ms * coverage_factor
        est_mb, build_ms = _estimate_index_build_ms(idx, build_mb_per_s)
        interference_hits = 1 if (table and table in interference_set) else 0
        terms = _score_terms(
            gain_ms=benefit_adj_ms,
            apply_ms=build_ms,
            coverage_factor=coverage_factor,
            uncertainty_mult=1.0,
            switch_penalty_ratio=switch_penalty_ratio,
            risk_penalty_ratio=risk_penalty_ratio,
            coverage_risk_weight=coverage_risk_weight,
            uncertainty_risk_weight=uncertainty_risk_weight,
            interference_hits=interference_hits,
            interference_penalty_ms_per_hit=interference_penalty_ms_per_hit,
            churn_penalty_ms=churn_penalty_ms,
        )
        score = float(terms["score_ms"])
        net = float(terms["net_gain_ms"])
        keep = (matched > 0) and (net > 0.0) and (score >= float(effective_min_gain_ms))
        reasons = []
        if matched == 0:
            reasons.append("no_future_match")
        if net <= 0.0:
            reasons.append("non_positive_net_gain")
        if score < float(effective_min_gain_ms):
            reasons.append("score_below_threshold")
        if coverage_factor < 0.50:
            reasons.append("low_observed_coverage")
        if interference_hits > 0 and float(interference_penalty_ms_per_hit) > 0.0:
            reasons.append("interference_penalized")

        rec = {
            "candidate": str(getattr(idx, "cid", "?")),
            "table": table,
            "cols": cols,
            "matched_queries": matched,
            "benefit_raw_ms": round(benefit_ms, 3),
            "benefit_ms": round(benefit_adj_ms, 3),
            "observed_matched_queries": int(observed_matched),
            "coverage_factor": round(float(coverage_factor), 4),
            "table_size_factor": round(float(table_size_factor), 4),
            "build_ms": round(build_ms, 3),
            "estimated_size_mb": round(est_mb, 3),
            "row_count": int(row_count),
            "net_gain_ms": round(net, 3),
            "gain_ms": round(float(terms["gain_ms"]), 3),
            "apply_ms": round(float(terms["apply_ms"]), 3),
            "switch_penalty_ms": round(float(terms["switch_penalty_ms"]), 3),
            "risk_penalty_ms": round(float(terms["risk_penalty_ms"]), 3),
            "interference_hits": int(interference_hits),
            "interference_penalty_ms": round(float(terms["interference_penalty_ms"]), 3),
            "score_ms": round(score, 3),
            "score": round(score, 3),
            "keep": bool(keep),
            "reasons": reasons,
        }
        decisions.append(rec)
        if keep:
            kept.append(idx)

    decisions.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
    report = {
        "module": "index",
        "start_idx": future_start,
        "future_end": future_end,
        "horizon_queries": int(horizon_queries),
        "min_net_gain_ms": float(min_net_gain_ms),
        "effective_min_gain_ms": float(effective_min_gain_ms),
        "build_mb_per_s": float(build_mb_per_s),
        "base_speedup": float(base_speedup),
        "switch_penalty_ratio": float(switch_penalty_ratio),
        "risk_penalty_ratio": float(risk_penalty_ratio),
        "coverage_risk_weight": float(coverage_risk_weight),
        "uncertainty_risk_weight": float(uncertainty_risk_weight),
        "interference_penalty_ms_per_hit": float(interference_penalty_ms_per_hit),
        "churn_penalty_ms": float(churn_penalty_ms),
        "input_count": len(candidates or []),
        "keep_count": len(kept),
        "kept": len(kept),
        "dropped": max(0, len(candidates or []) - len(kept)),
        "total_net_gain_ms": round(sum(float(x.get("net_gain_ms", 0.0)) for x in decisions if x.get("keep")), 3),
        "total_score_ms": round(sum(float(x.get("score_ms", 0.0)) for x in decisions if x.get("keep")), 3),
        "total_switch_penalty_ms": round(sum(float(x.get("switch_penalty_ms", 0.0)) for x in decisions if x.get("keep")), 3),
        "total_risk_penalty_ms": round(sum(float(x.get("risk_penalty_ms", 0.0)) for x in decisions if x.get("keep")), 3),
        "total_interference_penalty_ms": round(sum(float(x.get("interference_penalty_ms", 0.0)) for x in decisions if x.get("keep")), 3),
        "coverage_factor_avg": round(
            sum(float(x.get("coverage_factor", 1.0)) for x in decisions) / float(max(1, len(decisions))),
            4,
        ),
        "observed_window": {"start_idx": int(observed_start), "end_idx": int(observed_end)},
        "decisions": decisions,
        "template_ms": {str(k): round(v, 3) for k, v in tpl_ms.items()},
        "future_template_freq": {str(k): v for k, v in estimate_future_template_freq(queries, future_start, horizon_queries).items()},
    }
    return kept, report


def _extract_tpl_hints(mv: Any) -> List[int]:
    # Only use structured identifiers. `why` can contain broad/hallucinated
    # "Qxx" mentions and causes severe over-matching.
    text = " ".join([
        str(getattr(mv, "mvid", "") or ""),
        str(getattr(mv, "name", "") or ""),
    ])
    out: List[int] = []
    text_l = text.lower()
    for m in re.findall(r"(?:^|[^a-z0-9])[qt](\d{1,2})(?:$|[^0-9])", text_l):
        try:
            out.append(int(m))
        except Exception:
            pass
    return sorted(set(out))


def _extract_tables(sql: str) -> List[str]:
    if not sql:
        return []
    toks = re.findall(
        r"\b(?:from|join)\s+([a-z_][a-z0-9_]*(?:\.[a-z_][a-z0-9_]*)?)",
        sql.lower(),
    )
    out: List[str] = []
    for t in toks:
        base = str(t).split(".")[-1].strip()
        if base:
            out.append(base)
    return sorted(set(out))


def _estimate_mv_build_ms(mv: Any, build_mb_per_s: float) -> Tuple[float, float]:
    size = 0.0
    for field in ("estimated_size_mb", "size_mb"):
        try:
            size = float(getattr(mv, field, 0.0) or 0.0)
        except Exception:
            size = 0.0
        if size > 0:
            break
    if size <= 0:
        size = 256.0
    mbps = max(1e-6, float(build_mb_per_s))
    build_ms = (size / mbps) * 1000.0 + 300.0
    return size, build_ms


def gate_mv_candidates(
    candidates: List[Any],
    queries: List[Dict[str, Any]],
    start_idx: int,
    observed_per_query_ms: Dict[int, float],
    workload_stats: Optional[Dict[str, Any]] = None,
    horizon_queries: int = 2000,
    min_net_gain_ms: float = 1000.0,
    build_mb_per_s: float = 250.0,
    base_speedup: float = 0.35,
    switch_penalty_ratio: float = 0.06,
    risk_penalty_ratio: float = 0.03,
    coverage_risk_weight: float = 0.30,
    uncertainty_risk_weight: float = 0.20,
    interference_tables: Optional[List[str]] = None,
    interference_penalty_ms_per_hit: float = 0.0,
    churn_penalty_ms: float = 0.0,
) -> Tuple[List[Any], Dict[str, Any]]:
    future_start = max(0, int(start_idx))
    future_end = min(len(queries), future_start + max(0, int(horizon_queries)))
    future_span = max(0, future_end - future_start)
    observed_start = max(0, future_start - max(1, int(horizon_queries)))
    observed_end = future_start
    effective_min_gain_ms = _effective_min_gain_ms(min_net_gain_ms, future_span, horizon_queries)
    tpl_ms = estimate_template_ms(queries, observed_per_query_ms, workload_stats)
    global_ms = sorted(tpl_ms.values())[len(tpl_ms) // 2] if tpl_ms else 250.0
    future_tpl_freq = estimate_future_template_freq(queries, future_start, horizon_queries)
    observed_tpl_freq = estimate_template_freq_range(queries, observed_start, observed_end)
    interference_set = {str(x).strip().lower() for x in (interference_tables or []) if str(x).strip()}

    future_queries = queries[future_start:future_end]
    observed_queries = queries[observed_start:observed_end]

    decisions: List[Dict[str, Any]] = []
    kept: List[Any] = []
    kept_score: Dict[int, float] = {}

    for mv in candidates or []:
        mvid = str(getattr(mv, "mvid", "") or "")
        is_seed_mv = mvid.upper().startswith("SEED_")
        tpl_hints = _extract_tpl_hints(mv)
        mv_tables = _extract_tables(str(getattr(mv, "sql", "") or ""))

        matched = 0
        benefit_ms = 0.0
        observed_matched = 0

        if tpl_hints:
            for tpl in tpl_hints:
                freq = int(future_tpl_freq.get(tpl, 0))
                if freq <= 0:
                    continue
                base_ms = float(tpl_ms.get(tpl, global_ms))
                benefit_ms += freq * base_ms * max(0.05, min(0.80, float(base_speedup)))
                matched += freq
                observed_matched += int(observed_tpl_freq.get(tpl, 0))
        else:
            for q in future_queries:
                sql_l = str((q.get("sql") or q.get("query") or "")).lower()
                if mv_tables:
                    hit = sum(1 for t in mv_tables if _sql_has_identifier(sql_l, t))
                    if hit <= 0:
                        continue
                tpl = _template_of(q)
                base_ms = float(tpl_ms.get(tpl, global_ms))
                speed = max(0.05, min(0.80, float(base_speedup)))
                if "group by" in sql_l and "group by" in str(getattr(mv, "sql", "")).lower():
                    speed *= 1.15
                benefit_ms += base_ms * speed
                matched += 1
            for q in observed_queries:
                sql_l = str((q.get("sql") or q.get("query") or "")).lower()
                if mv_tables:
                    hit = sum(1 for t in mv_tables if _sql_has_identifier(sql_l, t))
                    if hit <= 0:
                        continue
                observed_matched += 1

        # Non-seed MVs come from free-form generation and are noisier in
        # rewrite quality. Discount their expected benefit to reduce
        # false-positive keeps.
        uncertainty_mult = 1.0 if is_seed_mv else 0.35
        coverage_factor = _coverage_factor(observed_matched, matched, floor=0.15)
        benefit_raw_ms = benefit_ms
        benefit_ms *= uncertainty_mult * coverage_factor

        size_mb, build_ms = _estimate_mv_build_ms(mv, build_mb_per_s)
        interference_hits = sum(1 for t in mv_tables if t in interference_set)
        terms = _score_terms(
            gain_ms=benefit_ms,
            apply_ms=build_ms,
            coverage_factor=coverage_factor,
            uncertainty_mult=uncertainty_mult,
            switch_penalty_ratio=switch_penalty_ratio,
            risk_penalty_ratio=risk_penalty_ratio,
            coverage_risk_weight=coverage_risk_weight,
            uncertainty_risk_weight=uncertainty_risk_weight,
            interference_hits=interference_hits,
            interference_penalty_ms_per_hit=interference_penalty_ms_per_hit,
            churn_penalty_ms=churn_penalty_ms,
        )
        score = float(terms["score_ms"])
        net = float(terms["net_gain_ms"])
        keep = (matched > 0) and (score >= float(effective_min_gain_ms))
        preferred_seed = bool(is_seed_mv and (mvid.upper() in MV_GATE_PREFERRED_TPCH_SEEDS))
        force_keep = bool(
            preferred_seed
            and net > 0.0
            and score >= float(effective_min_gain_ms) * float(MV_GATE_PREFERRED_SEED_SCORE_MULT)
            and coverage_factor >= float(MV_GATE_PREFERRED_SEED_MIN_COVERAGE)
            and int(observed_matched) >= int(MV_GATE_PREFERRED_SEED_MIN_OBS_MATCHES)
        )
        if force_keep:
            keep = True
        non_seed_strict_ok = True
        if not is_seed_mv and not MV_GATE_ALLOW_NON_SEED:
            strict_thresh = max(
                float(effective_min_gain_ms) * float(MV_GATE_NON_SEED_SCORE_MULT),
                float(MV_GATE_NON_SEED_MIN_SCORE_MS),
            )
            non_seed_strict_ok = (
                score >= strict_thresh
                and net > 0.0
                and coverage_factor >= float(MV_GATE_NON_SEED_MIN_COVERAGE)
            )
            if non_seed_strict_ok:
                keep = True
            else:
                keep = False
        reasons = []
        if matched == 0:
            reasons.append("no_future_match")
        if score < float(effective_min_gain_ms):
            reasons.append("score_below_threshold")
        if (not is_seed_mv) and (not MV_GATE_ALLOW_NON_SEED) and (not non_seed_strict_ok):
            reasons.append("non_seed_disabled")
        if (not is_seed_mv) and (not MV_GATE_ALLOW_NON_SEED) and non_seed_strict_ok:
            reasons.append("non_seed_promoted")
        if preferred_seed and not force_keep:
            reasons.append("preferred_seed_not_trusted_yet")
        if force_keep:
            reasons.append("preferred_seed")
        if coverage_factor < 0.50:
            reasons.append("low_observed_coverage")
        if interference_hits > 0 and float(interference_penalty_ms_per_hit) > 0.0:
            reasons.append("interference_penalized")

        rec = {
            "candidate": str(getattr(mv, "mvid", "?")),
            "name": str(getattr(mv, "name", "") or ""),
            "is_seed_mv": bool(is_seed_mv),
            "uncertainty_mult": float(uncertainty_mult),
            "coverage_factor": float(round(coverage_factor, 4)),
            "template_hints": tpl_hints,
            "tables": mv_tables,
            "matched_queries": matched,
            "observed_matched_queries": int(observed_matched),
            "benefit_raw_ms": round(benefit_raw_ms, 3),
            "benefit_ms": round(benefit_ms, 3),
            "build_ms": round(build_ms, 3),
            "estimated_size_mb": round(size_mb, 3),
            "net_gain_ms": round(net, 3),
            "gain_ms": round(float(terms["gain_ms"]), 3),
            "apply_ms": round(float(terms["apply_ms"]), 3),
            "switch_penalty_ms": round(float(terms["switch_penalty_ms"]), 3),
            "risk_penalty_ms": round(float(terms["risk_penalty_ms"]), 3),
            "interference_hits": int(interference_hits),
            "interference_penalty_ms": round(float(terms["interference_penalty_ms"]), 3),
            "score_ms": round(score, 3),
            "score": round(score, 3),
            "keep": bool(keep),
            "reasons": reasons,
        }
        decisions.append(rec)
        if keep:
            kept.append(mv)
            kept_score[id(mv)] = float(score)

    decisions.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
    if kept:
        kept.sort(key=lambda m: float(kept_score.get(id(m), float("-inf"))), reverse=True)
    report = {
        "module": "mv",
        "start_idx": future_start,
        "future_end": future_end,
        "horizon_queries": int(horizon_queries),
        "min_net_gain_ms": float(min_net_gain_ms),
        "effective_min_gain_ms": float(effective_min_gain_ms),
        "build_mb_per_s": float(build_mb_per_s),
        "base_speedup": float(base_speedup),
        "switch_penalty_ratio": float(switch_penalty_ratio),
        "risk_penalty_ratio": float(risk_penalty_ratio),
        "coverage_risk_weight": float(coverage_risk_weight),
        "uncertainty_risk_weight": float(uncertainty_risk_weight),
        "interference_penalty_ms_per_hit": float(interference_penalty_ms_per_hit),
        "churn_penalty_ms": float(churn_penalty_ms),
        "input_count": len(candidates or []),
        "keep_count": len(kept),
        "kept": len(kept),
        "dropped": max(0, len(candidates or []) - len(kept)),
        "total_net_gain_ms": round(sum(float(x.get("net_gain_ms", 0.0)) for x in decisions if x.get("keep")), 3),
        "total_score_ms": round(sum(float(x.get("score_ms", 0.0)) for x in decisions if x.get("keep")), 3),
        "total_switch_penalty_ms": round(sum(float(x.get("switch_penalty_ms", 0.0)) for x in decisions if x.get("keep")), 3),
        "total_risk_penalty_ms": round(sum(float(x.get("risk_penalty_ms", 0.0)) for x in decisions if x.get("keep")), 3),
        "total_interference_penalty_ms": round(sum(float(x.get("interference_penalty_ms", 0.0)) for x in decisions if x.get("keep")), 3),
        "coverage_factor_avg": round(
            sum(float(x.get("coverage_factor", 1.0)) for x in decisions) / float(max(1, len(decisions))),
            4,
        ),
        "observed_window": {"start_idx": int(observed_start), "end_idx": int(observed_end)},
        "decisions": decisions,
        "template_ms": {str(k): round(v, 3) for k, v in tpl_ms.items()},
        "future_template_freq": {str(k): v for k, v in future_tpl_freq.items()},
    }
    return kept, report


def _extract_layout_op_columns(op: Dict[str, Any]) -> List[str]:
    cols: List[str] = []
    if not isinstance(op, dict):
        return cols
    op_type = str(op.get("op", "") or "").strip().upper()
    if op_type == "PARTITION":
        parts = op.get("partition_by") or []
        if isinstance(parts, list):
            for p in parts:
                if not isinstance(p, dict):
                    continue
                c = str(p.get("col", "") or "").strip().lower()
                if c:
                    cols.append(c)
    elif op_type in ("CLUSTER_SORT", "ZORDER", "MDDL"):
        keys = op.get("keys") or op.get("cols") or []
        if isinstance(keys, list):
            for k in keys:
                if isinstance(k, dict):
                    c = str(k.get("col", "") or "").strip().lower()
                else:
                    c = str(k or "").strip().lower()
                if c:
                    cols.append(c)
        preds = op.get("predicates") or []
        if isinstance(preds, list):
            for p in preds:
                c = str(p or "").strip().lower()
                if c:
                    cols.append(c)
    return list(dict.fromkeys(cols))


def _has_eq_pred_on_col(sql_l: str, col: str) -> bool:
    c = str(col or "").strip().lower()
    if not c:
        return False
    return re.search(rf"(?:^|[^a-z0-9_]){re.escape(c)}\s*=\s*", sql_l) is not None


def _speed_for_layout(op_type: str, sql_l: str, cols: List[str], base_speedup: float) -> float:
    t = str(op_type or "").strip().upper()
    s = float(base_speedup)
    op_mult = {
        "PARTITION": 0.80,
        "CLUSTER_SORT": 1.05,
        "ZORDER": 1.02,
        "MDDL": 0.98,
        "TUNE_FILE": 0.35,
    }.get(t, 0.70)
    s *= op_mult

    hit_cols = [c for c in cols if c and _sql_has_identifier(sql_l, c)]
    hit_cnt = len(hit_cols)
    has_range = _has_range_pred(sql_l)
    has_join = _has_join(sql_l)

    if t == "PARTITION":
        if hit_cnt > 0 and any(_has_eq_pred_on_col(sql_l, c) for c in hit_cols):
            s *= 1.10
        if hit_cnt > 0 and has_range:
            s *= 1.20
        elif hit_cnt == 0:
            s *= 0.35
        if has_join:
            s *= 0.95
    elif t == "CLUSTER_SORT":
        if hit_cnt >= 2:
            s *= 1.30
        elif hit_cnt == 1:
            s *= 1.10
        else:
            s *= 0.55
        if has_join:
            s *= 1.08
    elif t == "ZORDER":
        if hit_cnt >= 2:
            s *= 1.35
        elif hit_cnt == 1:
            s *= 1.05
        else:
            s *= 0.60
        if has_range:
            s *= 1.10
    elif t == "MDDL":
        if hit_cnt > 0:
            s *= 1.20
        else:
            s *= 0.55
    elif t == "TUNE_FILE":
        # Generic scan throughput gain.
        s *= 1.0
    else:
        s *= 0.60

    if has_range and t in ("PARTITION", "CLUSTER_SORT", "ZORDER", "MDDL"):
        s *= 1.05
    return max(0.01, min(0.75, s))


def _estimate_layout_op_apply_ms(
    op: Dict[str, Any],
    table_size_mb: Optional[Dict[str, float]],
    apply_mb_per_s: float,
) -> Tuple[float, float]:
    t = str((op or {}).get("table", "") or "").strip().lower()
    op_type = str((op or {}).get("op", "") or "").strip().upper()
    size_mb = 0.0
    if isinstance(table_size_mb, dict) and t:
        try:
            size_mb = float(table_size_mb.get(t, 0.0) or 0.0)
        except Exception:
            size_mb = 0.0
    if size_mb <= 0:
        size_mb = 512.0

    op_mult = {
        "PARTITION": 1.40,
        "CLUSTER_SORT": 1.35,
        "ZORDER": 1.55,
        "MDDL": 1.65,
        "TUNE_FILE": 0.35,
    }.get(op_type, 1.0)
    fixed_ms = {
        "PARTITION": 420.0,
        "CLUSTER_SORT": 260.0,
        "ZORDER": 340.0,
        "MDDL": 380.0,
        "TUNE_FILE": 120.0,
    }.get(op_type, 180.0)
    mbps = max(1e-6, float(apply_mb_per_s))
    apply_ms = ((size_mb * op_mult) / mbps) * 1000.0 + fixed_ms
    return size_mb, apply_ms


def gate_layout_plan(
    layout_result: Optional[Dict[str, Any]],
    queries: List[Dict[str, Any]],
    start_idx: int,
    observed_per_query_ms: Dict[int, float],
    workload_stats: Optional[Dict[str, Any]] = None,
    table_size_mb: Optional[Dict[str, float]] = None,
    horizon_queries: int = 2000,
    min_net_gain_ms: float = 1000.0,
    apply_mb_per_s: float = 250.0,
    base_speedup: float = 0.15,
    apply_budget_ms: float = 0.0,
    switch_penalty_ratio: float = 0.12,
    risk_penalty_ratio: float = 0.02,
    coverage_risk_weight: float = 0.30,
    uncertainty_risk_weight: float = 0.00,
    interference_tables: Optional[List[str]] = None,
    interference_penalty_ms_per_hit: float = 0.0,
    churn_penalty_ms: float = 0.0,
    allowed_ops: Optional[List[str]] = None,
    max_table_mb: float = 0.0,
    max_apply_ms_per_op: float = 0.0,
    regret_trigger_ratio: float = 1.0,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    result = dict(layout_result or {})
    plan = dict(result.get("plan") or {})
    raw_ops = list(plan.get("ops") or [])

    future_start = max(0, int(start_idx))
    future_end = min(len(queries), future_start + max(0, int(horizon_queries)))
    future_span = max(0, future_end - future_start)
    observed_start = max(0, future_start - max(1, int(horizon_queries)))
    observed_end = future_start
    no_observed_window = observed_end <= observed_start
    effective_min_gain_ms = _effective_min_gain_ms(min_net_gain_ms, future_span, horizon_queries)
    tpl_ms = estimate_template_ms(queries, observed_per_query_ms, workload_stats)
    global_ms = sorted(tpl_ms.values())[len(tpl_ms) // 2] if tpl_ms else 250.0
    future_tpl_freq = estimate_future_template_freq(queries, future_start, horizon_queries)
    interference_set = {str(x).strip().lower() for x in (interference_tables or []) if str(x).strip()}
    allowed_set: Optional[set] = None
    if isinstance(allowed_ops, list):
        parsed = {str(x).strip().upper() for x in allowed_ops if str(x).strip()}
        if parsed:
            allowed_set = parsed
    max_table_mb = max(0.0, float(max_table_mb))
    max_apply_ms_per_op = max(0.0, float(max_apply_ms_per_op))

    decisions: List[Dict[str, Any]] = []
    for oi, op in enumerate(raw_ops):
        if not isinstance(op, dict):
            continue
        op_type = str(op.get("op", "") or "").strip().upper()
        table = str(op.get("table", "") or "").strip().lower()
        cols = _extract_layout_op_columns(op)
        if not op_type or not table:
            decisions.append({
                "op_idx": int(oi),
                "op": op_type or "?",
                "table": table or "?",
                "keep": False,
                "matched_queries": 0,
                "benefit_ms": 0.0,
                "apply_ms": 0.0,
                "net_gain_ms": -1.0,
                "reasons": ["invalid_op"],
            })
            continue
        if allowed_set is not None and op_type not in allowed_set:
            decisions.append({
                "op_idx": int(oi),
                "op": op_type,
                "table": table,
                "keep": False,
                "matched_queries": 0,
                "benefit_ms": 0.0,
                "apply_ms": 0.0,
                "net_gain_ms": -1.0,
                "reasons": ["op_not_allowed"],
            })
            continue

        size_mb, apply_ms = _estimate_layout_op_apply_ms(op, table_size_mb, apply_mb_per_s)
        if max_table_mb > 0.0 and float(size_mb) > float(max_table_mb):
            decisions.append({
                "op_idx": int(oi),
                "op": op_type,
                "table": table,
                "cols": _extract_layout_op_columns(op),
                "keep": False,
                "matched_queries": 0,
                "benefit_ms": 0.0,
                "apply_ms": round(float(apply_ms), 3),
                "table_size_mb": round(float(size_mb), 3),
                "net_gain_ms": -1.0,
                "reasons": ["table_too_large"],
            })
            continue
        if max_apply_ms_per_op > 0.0 and float(apply_ms) > float(max_apply_ms_per_op):
            decisions.append({
                "op_idx": int(oi),
                "op": op_type,
                "table": table,
                "cols": _extract_layout_op_columns(op),
                "keep": False,
                "matched_queries": 0,
                "benefit_ms": 0.0,
                "apply_ms": round(float(apply_ms), 3),
                "table_size_mb": round(float(size_mb), 3),
                "net_gain_ms": -1.0,
                "reasons": ["apply_ms_too_large"],
            })
            continue

        benefit_ms = 0.0
        matched = 0
        observed_matched = 0
        for qi in range(future_start, future_end):
            q = queries[qi]
            sql_l = str((q.get("sql") or q.get("query") or "")).lower()
            if not _sql_has_identifier(sql_l, table):
                continue
            if op_type in ("PARTITION", "CLUSTER_SORT", "ZORDER", "MDDL") and cols:
                if not any(_sql_has_identifier(sql_l, c) for c in cols):
                    # Allow a weak gain for join-heavy clustered tables.
                    if op_type not in ("CLUSTER_SORT", "ZORDER"):
                        continue
            tpl = _template_of(q)
            base_ms = float(tpl_ms.get(tpl, global_ms))
            speed = _speed_for_layout(op_type, sql_l, cols, base_speedup)
            benefit_ms += base_ms * speed
            matched += 1

        for qi in range(observed_start, observed_end):
            q = queries[qi]
            sql_l = str((q.get("sql") or q.get("query") or "")).lower()
            if not _sql_has_identifier(sql_l, table):
                continue
            if op_type in ("PARTITION", "CLUSTER_SORT", "ZORDER", "MDDL") and cols:
                if not any(_sql_has_identifier(sql_l, c) for c in cols):
                    if op_type not in ("CLUSTER_SORT", "ZORDER"):
                        continue
            observed_matched += 1

        coverage_factor = 1.0 if no_observed_window else _coverage_factor(observed_matched, matched, floor=0.20)
        benefit_raw_ms = benefit_ms
        benefit_ms = benefit_ms * coverage_factor

        interference_hits = 1 if (table and table in interference_set) else 0
        terms = _score_terms(
            gain_ms=benefit_ms,
            apply_ms=apply_ms,
            coverage_factor=coverage_factor,
            uncertainty_mult=1.0,
            switch_penalty_ratio=switch_penalty_ratio,
            risk_penalty_ratio=risk_penalty_ratio,
            coverage_risk_weight=coverage_risk_weight,
            uncertainty_risk_weight=uncertainty_risk_weight,
            interference_hits=interference_hits,
            interference_penalty_ms_per_hit=interference_penalty_ms_per_hit,
            churn_penalty_ms=churn_penalty_ms,
        )
        partition_penalty_ms = 0.0
        if op_type == "PARTITION" and float(size_mb) >= 1024.0:
            # Large-table partition often causes file-fragmentation and join instability
            # on short-horizon workloads; bias gate toward CLUSTER/ZORDER/MDDL first.
            partition_penalty_ms = min(30000.0, max(0.0, 0.25 * float(apply_ms) + 2.0 * float(size_mb)))
        score = float(terms["score_ms"])
        net = float(terms["net_gain_ms"])
        if partition_penalty_ms > 0.0:
            score -= float(partition_penalty_ms)
            net -= float(partition_penalty_ms)
        keep = matched > 0 and score > 0.0
        reasons: List[str] = []
        if matched <= 0:
            reasons.append("no_future_match")
        if score <= 0.0:
            reasons.append("non_positive_score")
        if partition_penalty_ms > 0.0:
            reasons.append("partition_large_table_penalty")
        if (not no_observed_window) and coverage_factor < 0.50:
            reasons.append("low_observed_coverage")
        if interference_hits > 0 and float(interference_penalty_ms_per_hit) > 0.0:
            reasons.append("interference_penalized")
        decisions.append({
            "op_idx": int(oi),
            "op": op_type,
            "table": table,
            "cols": cols,
            "matched_queries": int(matched),
            "observed_matched_queries": int(observed_matched),
            "coverage_factor": round(float(coverage_factor), 4),
            "benefit_raw_ms": round(benefit_raw_ms, 3),
            "benefit_ms": round(benefit_ms, 3),
            "apply_ms": round(apply_ms, 3),
            "table_size_mb": round(size_mb, 3),
            "net_gain_ms": round(net, 3),
            "gain_ms": round(float(terms["gain_ms"]), 3),
            "switch_penalty_ms": round(float(terms["switch_penalty_ms"]), 3),
            "risk_penalty_ms": round(float(terms["risk_penalty_ms"]), 3),
            "interference_hits": int(interference_hits),
            "interference_penalty_ms": round(float(terms["interference_penalty_ms"]), 3),
            "partition_penalty_ms": round(float(partition_penalty_ms), 3),
            "score_ms": round(score, 3),
            "score": round(score, 3),
            "keep": bool(keep),
            "reasons": reasons,
        })

    positive = [d for d in decisions if bool(d.get("keep"))]
    positive.sort(key=lambda d: float(d.get("score", 0.0)), reverse=True)

    selected: List[Dict[str, Any]] = []
    used_apply_ms = 0.0
    budget_ms = max(0.0, float(apply_budget_ms))
    for rec in positive:
        op_apply_ms = float(rec.get("apply_ms", 0.0) or 0.0)
        if budget_ms > 0.0 and op_apply_ms > 0.0 and (used_apply_ms + op_apply_ms) > budget_ms:
            rec["keep"] = False
            reasons = list(rec.get("reasons") or [])
            reasons.append("apply_budget_exceeded")
            rec["reasons"] = reasons
            continue
        selected.append(rec)
        used_apply_ms += op_apply_ms

    selected_op_idx = {int(rec.get("op_idx", -1)) for rec in selected}
    kept_ops = [op for oi, op in enumerate(raw_ops) if oi in selected_op_idx]
    total_net_gain_ms = sum(float(rec.get("net_gain_ms", 0.0) or 0.0) for rec in selected)
    total_score_ms = sum(float(rec.get("score_ms", 0.0) or 0.0) for rec in selected)
    total_benefit_ms = sum(float(rec.get("benefit_ms", 0.0) or 0.0) for rec in selected)
    total_apply_ms = sum(float(rec.get("apply_ms", 0.0) or 0.0) for rec in selected)
    rg_ratio = max(0.0, float(regret_trigger_ratio))
    regret_ok = (rg_ratio <= 0.0) or (total_benefit_ms >= (rg_ratio * total_apply_ms))
    plan_keep = len(kept_ops) > 0 and total_score_ms >= float(effective_min_gain_ms) and regret_ok

    if not plan_keep:
        kept_ops = []
        total_net_gain_ms = 0.0
        total_score_ms = 0.0
        total_benefit_ms = 0.0
        total_apply_ms = 0.0

    plan["ops"] = kept_ops
    result["plan"] = plan

    report = {
        "module": "layout",
        "start_idx": future_start,
        "future_end": future_end,
        "horizon_queries": int(horizon_queries),
        "min_net_gain_ms": float(min_net_gain_ms),
        "effective_min_gain_ms": float(effective_min_gain_ms),
        "apply_mb_per_s": float(apply_mb_per_s),
        "base_speedup": float(base_speedup),
        "apply_budget_ms": float(budget_ms),
        "switch_penalty_ratio": float(switch_penalty_ratio),
        "risk_penalty_ratio": float(risk_penalty_ratio),
        "coverage_risk_weight": float(coverage_risk_weight),
        "uncertainty_risk_weight": float(uncertainty_risk_weight),
        "interference_penalty_ms_per_hit": float(interference_penalty_ms_per_hit),
        "churn_penalty_ms": float(churn_penalty_ms),
        "allowed_ops": sorted(list(allowed_set)) if allowed_set is not None else [],
        "max_table_mb": float(max_table_mb),
        "max_apply_ms_per_op": float(max_apply_ms_per_op),
        "regret_trigger_ratio": float(rg_ratio),
        "regret_ok": bool(regret_ok),
        "input_count": len(raw_ops),
        "keep_count": len(kept_ops),
        "kept": len(kept_ops),
        "dropped": max(0, len(raw_ops) - len(kept_ops)),
        "plan_keep": bool(plan_keep),
        "total_benefit_ms": round(total_benefit_ms, 3),
        "total_apply_ms": round(total_apply_ms, 3),
        "total_net_gain_ms": round(total_net_gain_ms, 3),
        "total_score_ms": round(total_score_ms, 3),
        "total_switch_penalty_ms": round(sum(float(x.get("switch_penalty_ms", 0.0)) for x in selected), 3),
        "total_risk_penalty_ms": round(sum(float(x.get("risk_penalty_ms", 0.0)) for x in selected), 3),
        "total_interference_penalty_ms": round(sum(float(x.get("interference_penalty_ms", 0.0)) for x in selected), 3),
        "total_partition_penalty_ms": round(sum(float(x.get("partition_penalty_ms", 0.0)) for x in selected), 3),
        "coverage_factor_avg": round(
            sum(float(x.get("coverage_factor", 1.0)) for x in decisions) / float(max(1, len(decisions))),
            4,
        ),
        "observed_window": {"start_idx": int(observed_start), "end_idx": int(observed_end)},
        "decisions": sorted(decisions, key=lambda x: float(x.get("score", 0.0)), reverse=True),
        "template_ms": {str(k): round(v, 3) for k, v in tpl_ms.items()},
        "future_template_freq": {str(k): v for k, v in future_tpl_freq.items()},
    }
    return result, report
