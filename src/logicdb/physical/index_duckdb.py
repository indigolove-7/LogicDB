#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
co_opt/index_duckdb.py  (v2 — aggregation-aware covering indexes)

Index optimization for DuckDB (ART indexes) as part of the co-optimization experiment.

DuckDB supports standard SQL CREATE INDEX (Adaptive Radix Tree indexes).
These are useful for:
  - High-selectivity range predicates (e.g., l_shipdate range)
  - Join keys on large tables
  - High-cardinality equality filters
  - Covering indexes for GROUP BY + SELECT (avoid table scan)

New in v2:
  - COVERING indexes: (join/filter key, ...SELECT/GROUP BY cols) for OLAP
    → Allows DuckDB to satisfy aggregation queries entirely from the index
  - GROUP BY column detection: extracts GROUP BY patterns from workload
  - Aggregation column scoring: adds agg_col score for columns in SELECT agg exprs
  - INCLUDE columns (DuckDB 1.1+): CREATE INDEX ... INCLUDE (col1, col2)
    → Tried first; if DuckDB version doesn't support INCLUDE, falls back to regular
  - Richer LLM prompt: shows join graph + GROUP BY frequency + top slow templates

This module is intentionally designed to run concurrently with mv_duckdb.py.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import duckdb
from openai import OpenAI
try:
    import sqlglot
    from sqlglot import exp
except Exception:
    sqlglot = None
    exp = None


# ═══════════════════════════════════════════════════════════════════
# Data classes
# ═══════════════════════════════════════════════════════════════════

@dataclass
class IndexCand:
    cid: str
    table: str
    cols: List[str]          # key columns (order matters)
    include_cols: List[str]  # INCLUDE columns (covering; may be empty)
    index_type: str          # "btree" | "covering"
    category: str            # candidate family for breakdown (e.g., range_single)
    is_partial: bool
    where_clause: str
    why: str
    priority: int = 0
    estimated_size_mb: float = 0.0
    row_count: int = 0

    @property
    def index_name(self) -> str:
        col_str = "_".join(c[:8] for c in self.cols)
        suffix = "_p" if self.is_partial else ("_cov" if self.include_cols else "")
        return f"aidb_idx_{self.table}_{col_str}{suffix}"

    def create_sql(self, schema: str, try_include: bool = True) -> str:
        col_list = ", ".join(f'"{c}"' for c in self.cols)
        base = (f'CREATE INDEX IF NOT EXISTS "{self.index_name}" '
                f'ON {schema}."{self.table}" ({col_list})')
        if self.is_partial and self.where_clause:
            base += f" WHERE {self.where_clause}"
        if self.include_cols and try_include:
            inc_list = ", ".join(f'"{c}"' for c in self.include_cols)
            base += f" INCLUDE ({inc_list})"
        return base

    def drop_sql(self, schema: str) -> str:
        return f'DROP INDEX IF EXISTS "{self.index_name}"'


# ═══════════════════════════════════════════════════════════════════
# Workload feature extraction
# ═══════════════════════════════════════════════════════════════════

def _parse_sql(sql: str):
    if not sqlglot:
        return None
    for dialect in ("duckdb", "postgres", "ansi"):
        try:
            return sqlglot.parse_one(sql, read=dialect)
        except Exception:
            continue
    return None


def _table_name(node: "exp.Table") -> str:
    try:
        name = (node.name or "").strip()
    except Exception:
        name = ""
    if not name:
        try:
            name = str(node.this or "").strip()
        except Exception:
            name = ""
    if "." in name:
        name = name.split(".")[-1]
    return name.strip('"`').lower()


def _resolve_col_table(
    col_expr: "exp.Column",
    alias_to_table: Dict[str, str],
    col_to_tables: Dict[str, Set[str]],
) -> Optional[str]:
    if exp is None:
        return None
    table_alias = (col_expr.table or "").lower()
    col_name = (col_expr.name or "").lower()
    if not col_name:
        return None
    if table_alias:
        return alias_to_table.get(table_alias, table_alias)
    tables = col_to_tables.get(col_name, set())
    if len(tables) == 1:
        return next(iter(tables))
    return None


def extract_index_features(
    queries: List[Dict],
    *,
    col_to_tables: Optional[Dict[str, Set[str]]] = None,
) -> Dict[str, Any]:
    """
    Extract table/column access patterns from queries.
    Returns dict with per-table column frequency scores.

    Scoring dimensions:
      join   : appears in JOIN ON predicate
      range  : appears in range predicate (>=, <=, BETWEEN)
      eq     : appears in equality predicate
      group  : appears in GROUP BY
      agg_col: appears in SELECT aggregation expression (SUM(col), etc.)
    """
    table_col_scores: Dict[str, Dict[str, Dict[str, int]]] = {}

    def _ensure(t: str, c: str) -> None:
        table_col_scores.setdefault(t, {}).setdefault(
            c, {"join": 0, "range": 0, "eq": 0, "in": 0, "group": 0, "order": 0, "agg_col": 0}
        )

    if col_to_tables is None:
        col_to_tables = {}

    for q in queries:
        sql = (q.get("sql") or q.get("query") or "")
        if not sql:
            continue

        parsed = _parse_sql(sql)

        # AST path (preferred): supports unqualified cols and comma joins.
        if parsed is not None and exp is not None:
            alias_to_table: Dict[str, str] = {}
            for t in parsed.find_all(exp.Table):
                base = _table_name(t)
                if not base:
                    continue
                alias = (t.alias_or_name or base).lower()
                alias_to_table[alias] = base

            # JOIN + equality predicates.
            for eq in parsed.find_all(exp.EQ):
                l = eq.left
                r = eq.right
                if isinstance(l, exp.Column) and isinstance(r, exp.Column):
                    lt = _resolve_col_table(l, alias_to_table, col_to_tables)
                    rt = _resolve_col_table(r, alias_to_table, col_to_tables)
                    if lt and rt and lt != rt:
                        lc = (l.name or "").lower()
                        rc = (r.name or "").lower()
                        if lc:
                            _ensure(lt, lc)
                            table_col_scores[lt][lc]["join"] += 1
                        if rc:
                            _ensure(rt, rc)
                            table_col_scores[rt][rc]["join"] += 1
                    else:
                        # column = column on same table or unresolved; treat as filter hints
                        for col_expr in (l, r):
                            t = _resolve_col_table(col_expr, alias_to_table, col_to_tables)
                            c = (col_expr.name or "").lower()
                            if t and c:
                                _ensure(t, c)
                                table_col_scores[t][c]["eq"] += 1
                else:
                    # column = literal (or expression)
                    col_expr = l if isinstance(l, exp.Column) else (r if isinstance(r, exp.Column) else None)
                    if col_expr is not None:
                        t = _resolve_col_table(col_expr, alias_to_table, col_to_tables)
                        c = (col_expr.name or "").lower()
                        if t and c:
                            _ensure(t, c)
                            table_col_scores[t][c]["eq"] += 1

            # Range predicates: >, >=, <, <=, BETWEEN
            for op_cls in (exp.GT, exp.GTE, exp.LT, exp.LTE):
                for op in parsed.find_all(op_cls):
                    col_expr = op.left if isinstance(op.left, exp.Column) else (
                        op.right if isinstance(op.right, exp.Column) else None
                    )
                    if col_expr is None:
                        continue
                    t = _resolve_col_table(col_expr, alias_to_table, col_to_tables)
                    c = (col_expr.name or "").lower()
                    if t and c:
                        _ensure(t, c)
                        table_col_scores[t][c]["range"] += 1
            for op in parsed.find_all(exp.Between):
                col_expr = op.this if isinstance(op.this, exp.Column) else None
                if col_expr is None:
                    continue
                t = _resolve_col_table(col_expr, alias_to_table, col_to_tables)
                c = (col_expr.name or "").lower()
                if t and c:
                    _ensure(t, c)
                    table_col_scores[t][c]["range"] += 1

            # GROUP BY columns (top-level select).
            sel = parsed if isinstance(parsed, exp.Select) else parsed.find(exp.Select)
            if sel is not None:
                grp = sel.args.get("group")
                if grp is not None:
                    for g in grp.expressions or []:
                        if not isinstance(g, exp.Column):
                            continue
                        t = _resolve_col_table(g, alias_to_table, col_to_tables)
                        c = (g.name or "").lower()
                        if t and c:
                            _ensure(t, c)
                            table_col_scores[t][c]["group"] += 1

                # ORDER BY columns (top-level select).
                ord_by = sel.args.get("order")
                if ord_by is not None:
                    for o in ord_by.expressions or []:
                        cexpr = getattr(o, "this", None)
                        if not isinstance(cexpr, exp.Column):
                            continue
                        t = _resolve_col_table(cexpr, alias_to_table, col_to_tables)
                        c = (cexpr.name or "").lower()
                        if t and c:
                            _ensure(t, c)
                            table_col_scores[t][c]["order"] += 1

            # IN predicates (column IN (...))
            for node in parsed.find_all(exp.In):
                col_expr = node.this if isinstance(node.this, exp.Column) else None
                if col_expr is None:
                    continue
                t = _resolve_col_table(col_expr, alias_to_table, col_to_tables)
                c = (col_expr.name or "").lower()
                if t and c:
                    _ensure(t, c)
                    table_col_scores[t][c]["in"] += 1

            # Aggregation columns in SELECT: SUM/COUNT/AVG/MAX/MIN
            for agg in parsed.find_all(exp.AggFunc):
                col_expr = agg.this if isinstance(agg.this, exp.Column) else None
                if col_expr is None:
                    continue
                t = _resolve_col_table(col_expr, alias_to_table, col_to_tables)
                c = (col_expr.name or "").lower()
                if t and c:
                    _ensure(t, c)
                    table_col_scores[t][c]["agg_col"] += 1
            continue

        # Regex fallback (limited to qualified references).
        sql_l = sql.lower()
        for m in re.finditer(r'(\w+)\.(\w+)\s*=\s*(\w+)\.(\w+)', sql_l):
            t1, c1, t2, c2 = m.group(1), m.group(2), m.group(3), m.group(4)
            for t, c in [(t1, c1), (t2, c2)]:
                _ensure(t, c)
                table_col_scores[t][c]["join"] += 1
        for m in re.finditer(r'(\w+)\.(\w+)\s*(?:>=|<=|>|<|between)', sql_l):
            t, c = m.group(1), m.group(2)
            _ensure(t, c)
            table_col_scores[t][c]["range"] += 1
        for m in re.finditer(r'(\w+)\.(\w+)\s*=\s*(?:[\'"\d])', sql_l):
            t, c = m.group(1), m.group(2)
            _ensure(t, c)
            table_col_scores[t][c]["eq"] += 1
        for m in re.finditer(r'(\w+)\.(\w+)\s+in\s*\(', sql_l):
            t, c = m.group(1), m.group(2)
            _ensure(t, c)
            table_col_scores[t][c]["in"] += 1

    return table_col_scores


def get_table_stats(
    con: duckdb.DuckDBPyConnection,
    schema: str,
    table_names: List[str],
) -> Dict[str, int]:
    stats = {}
    for tbl in table_names:
        try:
            n = con.execute(
                f'SELECT COUNT(*) FROM "{schema}"."{tbl}"'
            ).fetchone()[0]
            stats[tbl] = n
        except Exception:
            stats[tbl] = 0
    return stats


def build_col_to_tables(
    con: duckdb.DuckDBPyConnection,
    schema: str,
    table_names: List[str],
) -> Dict[str, Set[str]]:
    """Build {column_name -> {table,...}} mapping for resolving unqualified columns."""
    mapping: Dict[str, Set[str]] = {}
    for tbl in table_names:
        try:
            rows = con.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_schema = ? AND table_name = ?",
                [schema, tbl],
            ).fetchall()
        except Exception:
            continue
        for (col,) in rows:
            c = str(col).lower()
            mapping.setdefault(c, set()).add(tbl.lower())
    return mapping


def get_column_types(
    con: duckdb.DuckDBPyConnection,
    schema: str,
    table_names: List[str],
) -> Dict[str, Dict[str, str]]:
    """Return {table -> {column -> data_type}}."""
    out: Dict[str, Dict[str, str]] = {}
    for tbl in table_names:
        try:
            rows = con.execute(
                "SELECT column_name, data_type FROM information_schema.columns "
                "WHERE table_schema = ? AND table_name = ?",
                [schema, tbl],
            ).fetchall()
        except Exception:
            rows = []
        out[tbl] = {str(c).lower(): str(t).lower() for c, t in rows}
    return out


def _type_size_bytes(dtype: str) -> int:
    """Conservative byte-width estimate for index size approximation."""
    dt = (dtype or "").lower()
    if any(k in dt for k in ("tinyint", "smallint", "short")):
        return 2
    if any(k in dt for k in ("integer", "int", "serial")):
        return 4
    if any(k in dt for k in ("bigint", "hugeint", "ubigint", "long", "double", "timestamp", "date", "time")):
        return 8
    if any(k in dt for k in ("decimal", "numeric")):
        return 16
    if any(k in dt for k in ("uuid",)):
        return 16
    if any(k in dt for k in ("varchar", "char", "string", "text", "blob", "json")):
        return 24
    return 8


def estimate_index_size_mb(
    nrows: int,
    key_cols: List[str],
    include_cols: List[str],
    col_types: Dict[str, str],
) -> float:
    """
    Rough in-memory/index-page size estimate used ONLY for budget gating.
    """
    if nrows <= 0:
        return 0.0
    key_bytes = sum(_type_size_bytes(col_types.get(c.lower(), "")) for c in key_cols)
    include_bytes = sum(_type_size_bytes(col_types.get(c.lower(), "")) for c in include_cols)
    per_row = 16.0 + float(key_bytes) + 0.50 * float(include_bytes)
    est = float(nrows) * per_row * 1.20  # structural overhead factor
    return est / (1024.0 * 1024.0)


def apply_index_storage_budget(
    chosen: List[IndexCand],
    budget_mb: float,
    *,
    ensure_min_kept: int = 1,
    allow_single_over_budget: bool = True,
    over_budget_tolerance: float = 3.0,
) -> Tuple[List[IndexCand], Dict[str, Any]]:
    """
    Hard-cap filter for selected indexes using estimated_size_mb.

    Greedy policy:
      - prioritize higher priority/size ratio
      - keep a tiny order bias to stabilize ties
      - if one candidate overflows, continue trying later/smaller candidates
      - avoid empty plans when budget gate is too strict for all picks
    """
    if budget_mb <= 0 or not chosen:
        return chosen, {
            "enabled": False,
            "budget_mb": float(budget_mb),
            "used_mb": 0.0,
            "kept": len(chosen),
            "dropped": 0,
            "overflow_stopped": False,
            "dropped_candidates": [],
            "floor_applied": False,
            "forced_kept": [],
        }

    ranked: List[Tuple[float, float, float, int, IndexCand]] = []
    for idx, c in enumerate(chosen):
        est = max(float(getattr(c, "estimated_size_mb", 0.0) or 0.0), 0.0)
        pri = float(getattr(c, "priority", 0) or 0.0)
        # Zero-sized estimates should be highly preferred but finite.
        ratio = pri / (est if est > 0.0 else 1e-6)
        # Tiny bonus to preserve original order when ratios are close.
        stable_ratio = ratio + (1e-12 * float(len(chosen) - idx))
        ranked.append((stable_ratio, pri, est, idx, c))

    ranked.sort(key=lambda x: (-x[0], -x[1], x[2], x[3]))

    selected_idx: Set[int] = set()
    used_mb = 0.0
    overflow_stopped = False
    forced_kept: List[Dict[str, Any]] = []
    for _, pri, est, idx, c in ranked:
        if (used_mb + est) > budget_mb:
            overflow_stopped = True
            continue
        selected_idx.add(idx)
        used_mb += est

    # Budget gate can accidentally zero-out index plan (common when freeform picks only very large keys).
    # Keep at least one "seed" index to avoid active=0 whenever possible.
    min_keep = max(0, int(ensure_min_kept))
    if len(selected_idx) < min_keep:
        need = min_keep - len(selected_idx)
        tol = max(1.0, float(over_budget_tolerance))

        while need > 0:
            remaining = max(0.0, float(budget_mb) - float(used_mb))
            candidates_fit = [
                x for x in ranked
                if x[3] not in selected_idx and x[2] <= (remaining + 1e-9)
            ]
            picked: Optional[Tuple[float, float, float, int, IndexCand]] = None
            reason = ""

            if candidates_fit:
                # ranked is already sorted by utility desc.
                picked = candidates_fit[0]
                reason = "floor_keep_in_budget"
            elif allow_single_over_budget:
                candidates_over = [x for x in ranked if x[3] not in selected_idx]
                if not candidates_over:
                    break
                # Smallest index first to minimize budget violation; tie-break by higher priority.
                candidates_over.sort(key=lambda x: (x[2], -x[1], x[3]))
                cand = candidates_over[0]
                if float(cand[2]) <= float(budget_mb) * tol:
                    picked = cand
                    reason = "floor_keep_over_budget"

            if picked is None:
                break

            _, pri, est, idx, c = picked
            selected_idx.add(idx)
            used_mb += float(est)
            forced_kept.append({
                "index_name": str(getattr(c, "index_name", "") or str(getattr(c, "cid", "") or "")),
                "est_mb": float(est),
                "priority": float(pri),
                "reason": reason,
            })
            need -= 1

    kept: List[IndexCand] = [c for i, c in enumerate(chosen) if i in selected_idx]
    dropped_candidates: List[Dict[str, Any]] = []
    for _, pri, est, idx, c in ranked:
        if idx in selected_idx:
            continue
        try:
            idx_name = str(getattr(c, "index_name", "") or "")
        except Exception:
            idx_name = ""
        dropped_candidates.append({
            "index_name": idx_name or str(getattr(c, "cid", "") or ""),
            "est_mb": float(est),
            "priority": float(pri),
            "reason": "over_budget",
        })
    dropped = len(chosen) - len(kept)

    return kept, {
        "enabled": True,
        "budget_mb": float(budget_mb),
        "used_mb": float(used_mb),
        "kept": len(kept),
        "dropped": int(dropped),
        "overflow_stopped": overflow_stopped,
        "dropped_candidates": dropped_candidates,
        "floor_applied": bool(forced_kept),
        "forced_kept": forced_kept,
    }


def _check_include_support(con: duckdb.DuckDBPyConnection) -> bool:
    """Check if current DuckDB version supports INCLUDE in CREATE INDEX."""
    try:
        ver = con.execute("SELECT version()").fetchone()[0]
        parts = re.findall(r'\d+', ver)
        if len(parts) >= 2:
            major, minor = int(parts[0]), int(parts[1])
            return (major, minor) >= (1, 1)
    except Exception:
        pass
    return False


def _adjust_priority_for_cost(
    raw_priority: float,
    est_mb: float,
    *,
    category: str,
    table_rows: int,
    max_table_rows: int,
) -> int:
    """
    Convert raw feature score into a budget-aware priority.
    Goal: improve real keep/apply probability under tight budgets.
    """
    p = max(float(raw_priority or 0.0), 0.0)
    cat = str(category or "").lower()
    nrows = max(int(table_rows or 0), 0)
    max_rows = max(int(max_table_rows or 0), 0)
    est = max(float(est_mb or 0.0), 0.0)

    # Join-only single indexes on very large tables are fragile for OLAP scans.
    if cat == "join_single" and max_rows > 0 and (float(nrows) / float(max_rows)) >= 0.5:
        p *= 0.55

    # Favor direct filter indexes (range/eq/in) since DuckDB ART benefits from selectivity.
    if cat.startswith("range_") or cat.startswith("eq_") or cat.startswith("in_") or "filter" in cat:
        p *= 1.15

    # Size-aware discount: large indexes should provide proportionally stronger signal.
    p *= 1.0 / (1.0 + (est / 1024.0))

    return max(1, int(round(p)))


def generate_index_candidates(
    con: duckdb.DuckDBPyConnection,
    queries: List[Dict],
    schema: str,
    table_names: List[str],
    *,
    min_rows: int = 50_000,
    max_per_table: int = 4,
    max_include_cols: int = 3,
    enable_conj_composite: bool = True,
    workload_stats: Optional[Dict] = None,
) -> List[IndexCand]:
    """
    Generate index candidates from workload features.

    Scoring (OLAP-tuned, range > join > group > eq > agg_col):
      range   × 5
      join    × 3
      group   × 4  (new: GROUP BY cols are very valuable for OLAP)
      eq      × 2
      agg_col × 1  (covered by MV or covering index)

    Covering index strategy (new in v2):
      If a column has high join+range score AND there are co-occurring GROUP BY
      columns on the same table → create (join_key, range_key) + INCLUDE (group_keys).
      This allows DuckDB to satisfy the query entirely from the index.
    """
    col_to_tables = build_col_to_tables(con, schema, table_names)
    table_col_scores = extract_index_features(queries, col_to_tables=col_to_tables)
    table_stats = get_table_stats(con, schema, table_names)
    has_include = _check_include_support(con)

    col_types = get_column_types(con, schema, table_names)
    max_rows_all = max(table_stats.values()) if table_stats else 0

    candidates: List[IndexCand] = []
    cid_counter = 1

    for table in table_names:
        nrows = table_stats.get(table, 0)
        if nrows < min_rows:
            continue
        col_scores = table_col_scores.get(table, {})
        if not col_scores:
            continue

        def score(c: str) -> int:
            s = col_scores.get(c, {})
            return (s.get("range", 0) * 5
                    + s.get("join", 0) * 3
                    + s.get("group", 0) * 4
                    + s.get("order", 0) * 1
                    + s.get("eq", 0) * 2
                    + s.get("in", 0) * 2
                    + s.get("agg_col", 0) * 1)

        all_cols_scored = sorted(
            [(c, score(c), col_scores[c]) for c in col_scores],
            key=lambda x: -x[1],
        )

        per_table = 0
        added_single: List[str] = []

        # ── Single-column indexes ──────────────────────────────────────
        for col, sc, freq in all_cols_scored:
            if per_table >= max_per_table:
                break
            if sc <= 0:
                continue

            idx_type = "range" if freq.get("range", 0) > 0 else (
                "join" if freq.get("join", 0) > 0 else ("in" if freq.get("in", 0) > 0 else "equality")
            )
            category = (
                "range_single" if freq.get("range", 0) > 0 else
                ("join_single" if freq.get("join", 0) > 0 else
                 ("in_single" if freq.get("in", 0) > 0 else "eq_single"))
            )
            est_mb = estimate_index_size_mb(
                nrows=nrows,
                key_cols=[col],
                include_cols=[],
                col_types=col_types.get(table, {}),
            )
            pri = _adjust_priority_for_cost(
                sc,
                est_mb,
                category=category,
                table_rows=nrows,
                max_table_rows=max_rows_all,
            )
            candidates.append(IndexCand(
                cid=f"IX{cid_counter:03d}",
                table=table,
                cols=[col],
                include_cols=[],
                index_type="btree",
                category=category,
                is_partial=False,
                where_clause="",
                why=(f"{idx_type} on {table}.{col} (score={sc}, "
                     f"join={freq.get('join',0)}, range={freq.get('range',0)}, "
                     f"in={freq.get('in',0)}, group={freq.get('group',0)}, {nrows:,} rows)"),
                priority=pri,
                estimated_size_mb=est_mb,
                row_count=nrows,
            ))
            added_single.append(col)
            cid_counter += 1
            per_table += 1

        # ── Composite: top-join + top-range ───────────────────────────
        join_cols  = sorted(
            [c for c in col_scores if col_scores[c].get("join", 0) > 0],
            key=lambda c: -col_scores[c]["join"]
        )
        range_cols = sorted(
            [c for c in col_scores if col_scores[c].get("range", 0) > 0],
            key=lambda c: -col_scores[c]["range"]
        )
        if join_cols and range_cols and per_table < max_per_table:
            top_j = join_cols[0]
            top_r = range_cols[0]
            if top_j != top_r:
                comp_score = col_scores[top_j]["join"] * 3 + col_scores[top_r]["range"] * 5
                est_mb = estimate_index_size_mb(
                    nrows=nrows,
                    key_cols=[top_j, top_r],
                    include_cols=[],
                    col_types=col_types.get(table, {}),
                )
                pri = _adjust_priority_for_cost(
                    comp_score,
                    est_mb,
                    category="composite_join_range",
                    table_rows=nrows,
                    max_table_rows=max_rows_all,
                )
                candidates.append(IndexCand(
                    cid=f"IX{cid_counter:03d}",
                    table=table,
                    cols=[top_j, top_r],
                    include_cols=[],
                    index_type="btree",
                    category="composite_join_range",
                    is_partial=False,
                    where_clause="",
                    why=f"Composite join+range on {table} ({top_j},{top_r}), score={comp_score}",
                    priority=pri,
                    estimated_size_mb=est_mb,
                    row_count=nrows,
                ))
                cid_counter += 1
                per_table += 1

        # ── Composite from frequent filter conjunction (DuckDB partial-index surrogate) ──
        if enable_conj_composite and per_table < max_per_table:
            filter_cols = sorted(
                [c for c in col_scores if (col_scores[c].get("eq", 0) + col_scores[c].get("in", 0)) > 0],
                key=lambda c: -(col_scores[c].get("eq", 0) + col_scores[c].get("in", 0))
            )
            if len(filter_cols) >= 2:
                c1, c2 = filter_cols[0], filter_cols[1]
                conj_score = (
                    (col_scores[c1].get("eq", 0) + col_scores[c1].get("in", 0)) * 2
                    + (col_scores[c2].get("eq", 0) + col_scores[c2].get("in", 0)) * 2
                )
                est_mb = estimate_index_size_mb(
                    nrows=nrows,
                    key_cols=[c1, c2],
                    include_cols=[],
                    col_types=col_types.get(table, {}),
                )
                pri = _adjust_priority_for_cost(
                    conj_score,
                    est_mb,
                    category="composite_filter_pair",
                    table_rows=nrows,
                    max_table_rows=max_rows_all,
                )
                candidates.append(IndexCand(
                    cid=f"IX{cid_counter:03d}",
                    table=table,
                    cols=[c1, c2],
                    include_cols=[],
                    index_type="btree",
                    category="composite_filter_pair",
                    is_partial=False,
                    where_clause="",
                    why=f"Composite frequent-filter pair on {table} ({c1},{c2}), score={conj_score}",
                    priority=pri,
                    estimated_size_mb=est_mb,
                    row_count=nrows,
                ))
                cid_counter += 1
                per_table += 1

        # ── Covering index: (join_key, range_key) INCLUDE (group_cols) ──
        # New in v2: if the table has frequent GROUP BY columns that are NOT
        # join/range keys, create a covering index so DuckDB can skip table scan.
        group_order_cols = sorted(
            [c for c in col_scores
             if (
                 (col_scores[c].get("group", 0) >= 2 or col_scores[c].get("order", 0) >= 2)
                 and c not in (join_cols[:1] + range_cols[:1])
             )],
            key=lambda c: -(col_scores[c].get("group", 0) * 4 + col_scores[c].get("order", 0))
        )[:max_include_cols]  # up to max_include_cols INCLUDE cols

        if group_order_cols and join_cols and per_table < max_per_table:
            key_col = join_cols[0]
            inc_list = group_order_cols
            inc_score = col_scores[key_col]["join"] * 3 + sum(
                (col_scores[c].get("group", 0) * 4 + col_scores[c].get("order", 0))
                for c in inc_list
            )
            inc_for_est = (inc_list if has_include else [])
            est_mb = estimate_index_size_mb(
                nrows=nrows,
                key_cols=[key_col],
                include_cols=inc_for_est,
                col_types=col_types.get(table, {}),
            )
            pri = _adjust_priority_for_cost(
                inc_score,
                est_mb,
                category="covering_group_order",
                table_rows=nrows,
                max_table_rows=max_rows_all,
            )
            candidates.append(IndexCand(
                cid=f"IX{cid_counter:03d}",
                table=table,
                cols=[key_col],
                include_cols=inc_list if has_include else [],
                index_type="covering",
                category="covering_group_order",
                is_partial=False,
                where_clause="",
                why=(f"Covering index on {table} ({key_col}) "
                     f"INCLUDE ({','.join(inc_list)}) for GROUP BY, score={inc_score}"),
                priority=pri,
                estimated_size_mb=est_mb,
                row_count=nrows,
            ))
            cid_counter += 1

    candidates.sort(key=lambda c: -c.priority)
    return candidates


# ═══════════════════════════════════════════════════════════════════
# LLM: Choose best index subset (with richer prompt)
# ═══════════════════════════════════════════════════════════════════

_INDEX_SYSTEM_PROMPT = """\
You are a DuckDB index optimization advisor for OLAP workloads.
DuckDB uses ART (Adaptive Radix Tree) indexes. Key usage patterns:
  - Selective range predicates on large tables (date ranges, ID lookups)
  - Join keys in star/snowflake schema queries
  - High-selectivity equality filters (low-cardinality = bad idea)
  - COVERING indexes: CREATE INDEX idx ON tbl(key_col) INCLUDE (col1, col2)
    → DuckDB can satisfy the query from the index alone, avoiding table scan
    → Ideal for frequently-aggregated columns (GROUP BY + SUM patterns)

Workload statistics:
{workload_stats_section}

Rules:
- Choose at most {max_indexes} indexes from the candidate list
- PREFER: larger tables, stable join/range predicates, and
  covering indexes for frequently-aggregated OLAP patterns
- AVOID: low-selectivity keys and already-covered columns
- For covering indexes: the key column should be a high-selectivity filter key

Output format (strict):
CHOSEN: IX001,IX003,IX007,...

Candidates:
{cand_lines}

Schema info:
{schema_info}
"""

_INDEX_FREEFORM_SYSTEM_PROMPT = """\
You are a DuckDB physical index advisor for OLAP workloads.

Important DuckDB constraints:
- Use only CREATE INDEX on base table columns (ART index family).
- Do NOT use BRIN/GIN/GiST/HASH or engine-specific index methods from PostgreSQL.
- Partial indexes are not reliably supported for this setup. If you need predicate-specific acceleration,
  prefer composite keys using predicate columns.
- INCLUDE columns are optional and should be used conservatively.

Output strict JSON only:
{
  "indexes": [
    {
      "table": "lineitem",
      "cols": ["l_shipdate"],
      "include_cols": ["l_discount"],
      "where": "",
      "why": "short reason"
    }
  ]
}
"""


def _parse_json_robust(txt: str) -> Optional[Any]:
    s = (txt or "").strip()
    if not s:
        return None
    candidates: List[str] = []
    for m in re.finditer(r"```json\s*(.*?)\s*```", s, flags=re.IGNORECASE | re.DOTALL):
        candidates.append(m.group(1).strip())
    for m in re.finditer(r"```\s*(.*?)\s*```", s, flags=re.DOTALL):
        blk = m.group(1).strip()
        if blk and blk not in candidates:
            candidates.append(blk)
    candidates.append(s)
    for c in candidates:
        try:
            return json.loads(c)
        except Exception:
            lb = c.find("{")
            rb = c.rfind("}")
            if lb != -1 and rb != -1 and rb > lb:
                try:
                    return json.loads(c[lb:rb + 1])
                except Exception:
                    pass
            lb = c.find("[")
            rb = c.rfind("]")
            if lb != -1 and rb != -1 and rb > lb:
                try:
                    return json.loads(c[lb:rb + 1])
                except Exception:
                    pass
    return None


def _extract_cols_from_expr(expr_text: str, valid_cols: Set[str]) -> List[str]:
    out: List[str] = []
    for tok in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", expr_text or ""):
        c = tok.lower()
        if c in valid_cols and c not in out:
            out.append(c)
    return out


def _parse_sql_index_shape(raw_sql: str) -> Dict[str, Any]:
    s = (raw_sql or "").strip().rstrip(";")
    out = {"table": "", "cols": [], "include_cols": [], "where": ""}
    if not s:
        return out
    # Very conservative parser for CREATE INDEX ... ON [schema.]table (cols) [INCLUDE (...)] [WHERE ...]
    m = re.search(
        r"(?is)\bon\s+([a-zA-Z_][\w\.]*)\s*\((.*?)\)\s*(?:include\s*\((.*?)\))?\s*(?:where\s+(.*))?$",
        s,
    )
    if not m:
        return out
    table = (m.group(1) or "").strip().strip('"`')
    if "." in table:
        table = table.split(".")[-1]
    cols_raw = (m.group(2) or "").strip()
    inc_raw = (m.group(3) or "").strip()
    where_raw = (m.group(4) or "").strip()
    cols = [x.strip().strip('"`').lower() for x in cols_raw.split(",") if x.strip()]
    inc = [x.strip().strip('"`').lower() for x in inc_raw.split(",") if x.strip()] if inc_raw else []
    out.update({
        "table": table.lower(),
        "cols": cols,
        "include_cols": inc,
        "where": where_raw,
    })
    return out


def llm_choose_indexes_freeform(
    con: duckdb.DuckDBPyConnection,
    queries: List[Dict],
    schema_info: str,
    *,
    schema: str,
    table_names: List[str],
    model: str,
    reasoning_effort: Optional[str] = None,
    max_indexes: int = 8,
    workload_stats: Optional[Dict] = None,
) -> List[IndexCand]:
    """
    Freeform index planning: LLM can propose indexes beyond generated candidates.
    Returned indexes are normalized to DuckDB-supported shapes.
    """
    if max_indexes <= 0:
        return []
    api_key = os.environ.get("OPENAI_API_KEY", "")
    base_url = os.environ.get("OPENAI_BASE_URL", "") or None
    client = OpenAI(api_key=api_key, base_url=base_url)

    tbl_set = {t.lower() for t in table_names}
    col_types = get_column_types(con, schema, table_names)
    table_stats = get_table_stats(con, schema, table_names)
    has_include = _check_include_support(con)

    compact_schema = []
    for t in table_names:
        cols = list(col_types.get(t, {}).keys())
        compact_schema.append({
            "table": t,
            "columns": cols[:80],
            "n_columns": len(cols),
            "row_count": int(table_stats.get(t, 0)),
        })

    q_sample = []
    for q in queries[: min(20, len(queries))]:
        s = (q.get("sql") or q.get("query") or "").strip()
        if s:
            q_sample.append(s[:800])

    ws_obj = {
        "top_slow_templates": (workload_stats or {}).get("top_slow_templates", [])[:8],
        "join_freq": (workload_stats or {}).get("join_freq", [])[:12],
        "group_by_freq": (workload_stats or {}).get("group_by_freq", [])[:20],
    }
    user_prompt = json.dumps(
        {
            "schema": compact_schema,
            "workload_stats": ws_obj,
            "query_samples": q_sample,
            "max_indexes": int(max_indexes),
            "notes": [
                "Prefer high-impact OLAP indexes on large tables.",
                "Use only valid table/column names from schema.",
                "Avoid redundant indexes.",
            ],
        },
        ensure_ascii=False,
    )

    print(f"  [idx_llm_freeform] Calling {model} ...", flush=True)
    t0 = time.perf_counter()
    chat_kwargs: Dict[str, Any] = {}
    if reasoning_effort:
        chat_kwargs["reasoning_effort"] = str(reasoning_effort).strip().lower()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _INDEX_FREEFORM_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        **chat_kwargs,
    )
    elapsed = time.perf_counter() - t0
    txt = (resp.choices[0].message.content or "").strip()
    print(f"  [idx_llm_freeform] Responded in {elapsed:.1f}s", flush=True)

    parsed = _parse_json_robust(txt)
    if parsed is None:
        preview = txt[:2000] if txt else ""
        print("  [idx_llm_freeform] parse failed, returning empty plan", flush=True)
        if preview:
            print(f"  [idx_llm_freeform] raw output preview:\n{preview}", flush=True)
        return []

    items: List[Dict[str, Any]]
    if isinstance(parsed, dict):
        if isinstance(parsed.get("indexes"), list):
            items = parsed["indexes"]
        elif isinstance(parsed.get("index_plan"), list):
            items = parsed["index_plan"]
        else:
            items = []
    elif isinstance(parsed, list):
        items = parsed
    else:
        items = []

    out: List[IndexCand] = []
    seen = set()
    rank = 0
    for it in items:
        if not isinstance(it, dict):
            continue
        rank += 1
        raw_sql = (it.get("sql") or "").strip()
        shaped = _parse_sql_index_shape(raw_sql) if raw_sql else {}
        table = str(it.get("table") or shaped.get("table") or "").strip().strip('"`').lower()
        if "." in table:
            table = table.split(".")[-1]
        if table not in tbl_set:
            continue

        valid_cols = set(col_types.get(table, {}).keys())
        cols = it.get("cols")
        if not isinstance(cols, list):
            cols = shaped.get("cols", [])
        cols = [str(c).strip().strip('"`').lower() for c in cols if str(c).strip()]
        cols = [c for c in cols if c in valid_cols]
        # Handle expression-like tokens by extracting valid identifiers.
        if not cols and raw_sql:
            cols = _extract_cols_from_expr(raw_sql, valid_cols)
        if not cols:
            continue

        include_cols = it.get("include_cols")
        if not isinstance(include_cols, list):
            include_cols = shaped.get("include_cols", [])
        include_cols = [str(c).strip().strip('"`').lower() for c in include_cols if str(c).strip()]
        include_cols = [c for c in include_cols if c in valid_cols and c not in cols]
        if not has_include:
            include_cols = []

        where_clause = str(it.get("where") or shaped.get("where") or "").strip()
        category = "freeform"
        # DuckDB partial-index surrogate: absorb predicate columns into key.
        if where_clause:
            pred_cols = _extract_cols_from_expr(where_clause, valid_cols)
            for c in pred_cols:
                if c not in cols:
                    cols.append(c)
                if len(cols) >= 4:
                    break
            where_clause = ""
            category = "freeform_partial_surrogate"

        # remove duplicate key cols and cap width
        dedup_cols = []
        for c in cols:
            if c not in dedup_cols:
                dedup_cols.append(c)
        cols = dedup_cols[:4]
        include_cols = include_cols[:4]
        if not cols:
            continue

        sig = (table, tuple(cols), tuple(include_cols))
        if sig in seen:
            continue
        seen.add(sig)

        idx_type = "covering" if include_cols else "btree"
        nrows = int(table_stats.get(table, 0))
        why = str(it.get("why") or it.get("reason") or "freeform llm proposal").strip()
        out.append(
            IndexCand(
                cid=f"FX{len(out) + 1:03d}",
                table=table,
                cols=cols,
                include_cols=include_cols,
                index_type=idx_type,
                category=category,
                is_partial=False,
                where_clause="",
                why=f"[freeform] {why}"[:220],
                priority=max(0, 1000 - rank),
                estimated_size_mb=estimate_index_size_mb(
                    nrows=nrows,
                    key_cols=cols,
                    include_cols=include_cols,
                    col_types=col_types.get(table, {}),
                ),
                row_count=nrows,
            )
        )
        if len(out) >= max_indexes:
            break

    return out

def llm_choose_indexes(
    candidates: List[IndexCand],
    schema_info: str,
    *,
    model: str,
    reasoning_effort: Optional[str] = None,
    max_indexes: int = 8,
    workload_stats: Optional[Dict] = None,
) -> List[IndexCand]:
    if not candidates:
        return []

    api_key = os.environ.get("OPENAI_API_KEY", "")
    base_url = os.environ.get("OPENAI_BASE_URL", "") or None
    client = OpenAI(api_key=api_key, base_url=base_url)

    cand_lines = "\n".join(
        f"- {c.cid}: CREATE INDEX ON {c.table}({', '.join(c.cols)})"
        + (f" INCLUDE ({', '.join(c.include_cols)})" if c.include_cols else "")
        + (f"  WHERE {c.where_clause}" if c.is_partial else "")
        + f"  [{c.index_type}/{c.category}] [~{c.estimated_size_mb:.1f}MB]  # {c.why}"
        for c in candidates
    )

    # Build workload stats section
    ws_lines = []
    if workload_stats:
        if workload_stats.get("top_slow_templates"):
            slow_str = ", ".join(
                f"T{t}(cost={c:.0f})" for t, c in workload_stats["top_slow_templates"][:5]
            )
            ws_lines.append(f"Top slow templates: {slow_str}")
        if workload_stats.get("join_freq"):
            jf_str = ", ".join(
                f"{t[0]}×{t[1]}({n}x)" for t, n in workload_stats["join_freq"][:8]
            )
            ws_lines.append(f"Frequent JOINs: {jf_str}")
        if workload_stats.get("group_by_freq"):
            gb_str = ", ".join(
                f"{c}({n}x)" for c, n in workload_stats["group_by_freq"][:10]
            )
            ws_lines.append(f"Frequent GROUP BY: {gb_str}")
        if workload_stats.get("group_by_distinct"):
            gbd_str = ", ".join(
                f"{c}={nd:,} distinct"
                for c, nd in list(workload_stats["group_by_distinct"].items())[:6]
            )
            ws_lines.append(f"GROUP BY cardinalities: {gbd_str}")
    ws_section = "\n".join(ws_lines) if ws_lines else "(none provided)"

    system = _INDEX_SYSTEM_PROMPT.format(
        max_indexes=max_indexes,
        cand_lines=cand_lines,
        schema_info=schema_info,
        workload_stats_section=ws_section,
    )

    print(f"  [idx_llm] Calling {model} for index selection...", flush=True)
    t0 = time.perf_counter()
    chat_kwargs: Dict[str, Any] = {}
    if reasoning_effort:
        chat_kwargs["reasoning_effort"] = str(reasoning_effort).strip().lower()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": "Select the most beneficial indexes for this OLAP workload:"},
        ],
        temperature=0.0,
        **chat_kwargs,
    )
    elapsed = time.perf_counter() - t0
    txt = (resp.choices[0].message.content or "").strip()
    print(f"  [idx_llm] Responded in {elapsed:.1f}s:\n{txt}")

    m = re.search(r"CHOSEN\s*:\s*([A-Za-z0-9_,\s]+)", txt)
    if not m:
        print("  [idx_llm] No CHOSEN line found, using top candidates heuristically.")
        return candidates[:max_indexes]

    ids = {x.strip() for x in m.group(1).split(",") if x.strip()}
    chosen = [c for c in candidates if c.cid in ids]
    return chosen[:max_indexes]


# ═══════════════════════════════════════════════════════════════════
# Build indexes (with INCLUDE fallback)
# ═══════════════════════════════════════════════════════════════════

def build_indexes(
    con: duckdb.DuckDBPyConnection,
    chosen: List[IndexCand],
    schema: str,
    verbose: bool = True,
) -> Tuple[float, List[Dict]]:
    """
    Create DuckDB ART indexes for the chosen candidates.
    For covering indexes: tries INCLUDE syntax first; falls back to regular if unsupported.
    Returns (elapsed_ms, applied_list).
    """
    applied = []
    t0 = time.perf_counter()

    for idx in chosen:
        sql = idx.create_sql(schema, try_include=bool(idx.include_cols))
        try:
            if verbose:
                print(f"  [idx_build] {sql[:140]}", flush=True)
            con.execute(sql)
            applied.append({
                "cid": idx.cid, "table": idx.table, "cols": idx.cols,
                "include_cols": idx.include_cols, "index_type": idx.index_type,
                "category": idx.category, "estimated_size_mb": idx.estimated_size_mb,
                "index_name": idx.index_name, "status": "ok",
            })
            if verbose:
                print(f"  [idx_build]   ok [{idx.index_type}]", flush=True)
        except Exception as e:
            # Fallback: try without INCLUDE
            if idx.include_cols:
                sql_fallback = idx.create_sql(schema, try_include=False)
                try:
                    if verbose:
                        print(f"  [idx_build] INCLUDE unsupported, fallback: {sql_fallback[:120]}",
                              flush=True)
                    con.execute(sql_fallback)
                    applied.append({
                        "cid": idx.cid, "table": idx.table, "cols": idx.cols,
                        "include_cols": [], "index_type": "btree",
                        "category": idx.category, "estimated_size_mb": idx.estimated_size_mb,
                        "index_name": idx.index_name, "status": "ok_no_include",
                    })
                    if verbose:
                        print(f"  [idx_build]   ok (no INCLUDE fallback)", flush=True)
                    continue
                except Exception as e2:
                    e = e2
            applied.append({
                "cid": idx.cid, "table": idx.table, "cols": idx.cols,
                "include_cols": idx.include_cols, "index_type": idx.index_type,
                "category": idx.category, "estimated_size_mb": idx.estimated_size_mb,
                "index_name": idx.index_name, "status": f"error: {e}",
            })
            if verbose:
                print(f"  [idx_build]   FAILED: {e}", flush=True)

    return (time.perf_counter() - t0) * 1000.0, applied


def drop_aidb_indexes(con: duckdb.DuckDBPyConnection, applied: List[Dict]) -> None:
    for op in applied:
        if op.get("status") in ("ok", "ok_no_include"):
            try:
                con.execute(f'DROP INDEX IF EXISTS "{op["index_name"]}"')
            except Exception:
                pass


# ═══════════════════════════════════════════════════════════════════
# Full index pipeline
# ═══════════════════════════════════════════════════════════════════

def run_index_pipeline(
    con: duckdb.DuckDBPyConnection,
    queries: List[Dict],
    schema_info: str,
    schema: str,
    table_names: List[str],
    model: str,
    workdir: str,
    max_indexes: int = 8,
    verbose: bool = True,
    workload_stats: Optional[Dict] = None,
) -> Dict:
    os.makedirs(workdir, exist_ok=True)

    t0 = time.perf_counter()
    candidates = generate_index_candidates(
        con, queries, schema, table_names, workload_stats=workload_stats
    )
    gen_ms = (time.perf_counter() - t0) * 1000.0
    n_cov = sum(1 for c in candidates if c.index_type == "covering")
    print(f"  [idx] Generated {len(candidates)} candidates "
          f"({n_cov} covering) in {gen_ms:.0f}ms")

    with open(os.path.join(workdir, "index_candidates.json"), "w") as f:
        json.dump([{
            "cid": c.cid, "table": c.table, "cols": c.cols,
            "include_cols": c.include_cols, "index_type": c.index_type,
            "category": c.category, "estimated_size_mb": c.estimated_size_mb,
            "why": c.why, "priority": c.priority,
        } for c in candidates], f, indent=2)

    if not candidates:
        return {"n_candidates": 0, "n_built": 0, "chosen": [], "applied": [],
                "llm_ms": 0, "build_ms": 0}

    t1 = time.perf_counter()
    chosen = llm_choose_indexes(
        candidates, schema_info, model=model, max_indexes=max_indexes,
        workload_stats=workload_stats,
    )
    llm_ms = (time.perf_counter() - t1) * 1000.0
    print(f"  [idx] LLM chose {len(chosen)} indexes in {llm_ms/1000:.1f}s")

    with open(os.path.join(workdir, "index_chosen.json"), "w") as f:
        json.dump([{
            "cid": c.cid, "table": c.table, "cols": c.cols,
            "include_cols": c.include_cols, "index_type": c.index_type,
            "category": c.category, "estimated_size_mb": c.estimated_size_mb,
            "sql": c.create_sql(schema), "why": c.why,
        } for c in chosen], f, indent=2)

    build_ms, applied = build_indexes(con, chosen, schema, verbose=verbose)
    n_built = sum(1 for a in applied if a["status"] in ("ok", "ok_no_include"))
    print(f"  [idx] Built {n_built}/{len(chosen)} indexes in {build_ms/1000:.1f}s")

    return {
        "n_candidates": len(candidates),
        "n_built": n_built,
        "chosen": chosen,
        "applied": applied,
        "llm_ms": llm_ms,
        "build_ms": build_ms,
    }
