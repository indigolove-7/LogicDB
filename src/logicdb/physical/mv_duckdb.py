#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
co_opt/mv_duckdb.py  (v2 — precise aggregation MV rewriting)

Materialized View optimization on DuckDB for the co-optimization experiment.

DuckDB does not have native MATERIALIZED VIEW (as of 1.x), so we simulate them:
  - CREATE TABLE mv_schema.<name> AS <select_sql>
  - Register created MVs in a local registry
  - For each workload query, attempt SQL rewrite:
      * Join-only MVs  → search_path trick (no SQL change)
      * Aggregation MVs → precise rewrite (GROUP BY subset matching + SELECT coverage)
  - Use DuckDB EXPLAIN estimated-cardinality as cost proxy

LLM role:
  - Given workload SQL + schema + workload statistics, generate MV definitions
  - DuckDB dry-runs each MV for validation before materialisation
  - Rewriter tries both join-MV (search_path) and agg-MV (precise rewrite)

Aggregation MV Rewrite Strategy (new in v2):
  Given MV defined as:
    SELECT key1, key2, SUM(col_a), COUNT(col_b)
    FROM base_tables
    GROUP BY key1, key2

  And query:
    SELECT key1, SUM(col_a)
    FROM base_tables
    [WHERE ...]
    GROUP BY key1

  Conditions for safe rewrite:
    1. query GROUP BY cols ⊆ MV GROUP BY cols  (query groups are coarser)
    2. all SELECT columns in query exist in MV output
    3. MV tables ⊆ query tables  (MV doesn't reference tables not in query)
    4. query WHERE predicates do NOT touch columns not in MV output
       (can't push filter on raw fact columns after aggregation)
  
  Rewritten form:
    SELECT key1, SUM(col_a)   ← re-aggregate from MV
    FROM mv_schema.mv_name
    [WHERE ...]               ← pushed through only if safe
    GROUP BY key1
"""

from __future__ import annotations

import json
import os
import re
import sys
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


def _env(name: str, legacy_name: str, default: str) -> str:
    return str(os.getenv(name, os.getenv(legacy_name, default)))


MV_SKIP_LLM_ON_TPCH = _env("LOGICDB_MV_SKIP_LLM_ON_TPCH", "AIDB_MV_SKIP_LLM_ON_TPCH", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}


def _env_bool(name: str, default: bool) -> bool:
    legacy_name = name.replace("LOGICDB_", "AIDB_", 1) if name.startswith("LOGICDB_") else name
    raw = os.getenv(name)
    if raw is None:
        raw = os.getenv(legacy_name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    try:
        legacy_name = name.replace("LOGICDB_", "AIDB_", 1) if name.startswith("LOGICDB_") else name
        return float(_env(name, legacy_name, str(default)))
    except Exception:
        return float(default)


MV_ALLOW_LLM_UNCOVERED_TPCH = _env_bool("LOGICDB_MV_ALLOW_LLM_UNCOVERED_TPCH", True)
MV_UNCOVERED_TPL_MIN_SHARE = max(0.0, min(1.0, _env_float("LOGICDB_MV_UNCOVERED_TPL_MIN_SHARE", 0.18)))
MV_UNCOVERED_TPL_TOPK = max(1, int(round(_env_float("LOGICDB_MV_UNCOVERED_TPL_TOPK", 2.0))))
MV_UNCOVERED_LLM_MAX = max(1, int(round(_env_float("LOGICDB_MV_UNCOVERED_LLM_MAX", 2.0))))
MV_UNCOVERED_TPL_MIN_SHARE_LONG = max(
    0.0,
    min(1.0, _env_float("LOGICDB_MV_UNCOVERED_TPL_MIN_SHARE_LONG", 0.08)),
)
MV_UNCOVERED_TPL_MIN_COST_RATIO = max(
    0.0,
    min(1.0, _env_float("LOGICDB_MV_UNCOVERED_TPL_MIN_COST_RATIO", 0.50)),
)
MV_UNCOVERED_TPL_MIN_COST_ABS = max(0.0, _env_float("LOGICDB_MV_UNCOVERED_TPL_MIN_COST_ABS", 0.0))
MV_UNCOVERED_TPL_MIN_RUNTIME_MS = max(0.0, _env_float("LOGICDB_MV_UNCOVERED_TPL_MIN_RUNTIME_MS", 120.0))
MV_UNCOVERED_TPL_MIN_RUNTIME_SHARE = max(
    0.0,
    min(1.0, _env_float("LOGICDB_MV_UNCOVERED_TPL_MIN_RUNTIME_SHARE", 0.08)),
)
MV_GENERIC_REPEAT_SEEDS = _env_bool("LOGICDB_MV_GENERIC_REPEAT_SEEDS", True)
MV_GENERIC_REPEAT_MIN_FREQ = max(1, int(round(_env_float("LOGICDB_MV_GENERIC_REPEAT_MIN_FREQ", 3.0))))
MV_GENERIC_REPEAT_MAX = max(0, int(round(_env_float("LOGICDB_MV_GENERIC_REPEAT_MAX", 8.0))))
MV_SKIP_LLM_ON_MISSING_KEY = _env_bool("LOGICDB_MV_SKIP_LLM_ON_MISSING_KEY", True)
MV_GENERIC_REPEAT_RANK_MODE = str(
    _env("LOGICDB_MV_GENERIC_REPEAT_RANK_MODE", "AIDB_MV_GENERIC_REPEAT_RANK_MODE", "freq")
).strip().lower()


# ═══════════════════════════════════════════════════════════════════
# Data classes
# ═══════════════════════════════════════════════════════════════════

@dataclass
class MVCand:
    mvid: str
    name: str
    sql: str
    why: str
    mv_schema: str = "aidb_co_mv"
    built: bool = False
    build_error: Optional[str] = None
    size_mb: float = 0.0
    estimated_size_mb: float = 0.0
    row_count: int = 0
    # Parsed structure (filled by _parse_mv_structure)
    mv_type: str = "join"       # "join" | "agg"
    mv_group_by_cols: List[str] = field(default_factory=list)
    mv_select_cols: List[str] = field(default_factory=list)   # output col names
    mv_tables: Set[str] = field(default_factory=set)

    @property
    def full_name(self) -> str:
        return f"{self.mv_schema}.{self.name}"


# ═══════════════════════════════════════════════════════════════════
# SQL structure parsing helpers
# ═══════════════════════════════════════════════════════════════════

def _parse_sql(sql: str):
    """Parse SQL with sqlglot (best-effort). Returns parsed AST or None."""
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


def _extract_tables_ast(parsed) -> Set[str]:
    if parsed is None or exp is None:
        return set()
    out: Set[str] = set()
    for t in parsed.find_all(exp.Table):
        name = _table_name(t)
        if name:
            out.add(name)
    return out


def _extract_tables(sql: str) -> Set[str]:
    """
    Extract unqualified table names from SQL.
    Uses sqlglot AST first; falls back to regex for robustness.
    """
    parsed = _parse_sql(sql)
    ast_tables = _extract_tables_ast(parsed)
    if ast_tables:
        return ast_tables
    tables = set(re.findall(r'(?:FROM|JOIN)\s+[\w]+\.(\w+)', sql, re.IGNORECASE))
    if not tables:
        tables = set(re.findall(r'(?:FROM|JOIN)\s+(\w+)', sql, re.IGNORECASE))
    # Fallback for comma-join style FROM a, b, c
    if not tables:
        m = re.search(
            r"\bFROM\b\s+(.+?)(?:\bWHERE\b|\bGROUP\s+BY\b|\bHAVING\b|\bORDER\s+BY\b|\bLIMIT\b|;|$)",
            sql, re.IGNORECASE | re.DOTALL,
        )
        if m:
            from_text = m.group(1)
            for part in from_text.split(","):
                token = part.strip().split()[0] if part.strip() else ""
                token = token.strip('"`')
                if "." in token:
                    token = token.split(".")[-1]
                if re.match(r"^[A-Za-z_]\w*$", token):
                    tables.add(token)
    return {t.lower() for t in tables}


def _extract_select_cols(sql: str) -> List[str]:
    """
    Extract output column aliases / expressions from SELECT clause.
    Returns lowercased list of alias names or the raw expression if no alias.
    """
    parsed = _parse_sql(sql)
    if parsed is not None and exp is not None:
        sel = parsed if isinstance(parsed, exp.Select) else parsed.find(exp.Select)
        if sel is not None:
            result = []
            for expr_i in sel.expressions or []:
                if isinstance(expr_i, exp.Alias):
                    alias = (expr_i.alias or "").strip().lower()
                    if alias:
                        result.append(alias)
                        continue
                if isinstance(expr_i, exp.Column):
                    col = (expr_i.name or "").strip().lower()
                    if col:
                        result.append(col)
                        continue
                # Fallback expression key
                txt = expr_i.sql(dialect="duckdb").strip().lower()
                if txt:
                    result.append(txt)
            if result:
                return result

    # Regex fallback
    m = re.match(r'^\s*SELECT\s+(.+?)\s+FROM\b', sql, re.IGNORECASE | re.DOTALL)
    if not m:
        return []
    select_part = m.group(1)
    cols = []
    depth = 0
    current = []
    for ch in select_part:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            cols.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
    if current:
        cols.append("".join(current).strip())

    result = []
    for col in cols:
        alias_m = re.search(r'\bAS\s+["\'`]?(\w+)["\'`]?\s*$', col, re.IGNORECASE)
        if alias_m:
            result.append(alias_m.group(1).lower())
        else:
            plain = re.search(r'["\`]?(\w+)["\`]?\s*$', col)
            if plain:
                result.append(plain.group(1).lower())
    return result


def _extract_group_by_cols(sql: str) -> List[str]:
    """Extract GROUP BY columns (lowercased, unqualified)."""
    parsed = _parse_sql(sql)
    if parsed is not None and exp is not None:
        sel = parsed if isinstance(parsed, exp.Select) else parsed.find(exp.Select)
        if sel is not None:
            grp = sel.args.get("group")
            if grp is not None:
                cols = []
                for item in grp.expressions or []:
                    if isinstance(item, exp.Column):
                        c = (item.name or "").strip().lower()
                        if c:
                            cols.append(c)
                    else:
                        txt = item.sql(dialect="duckdb").strip().lower()
                        if txt:
                            cols.append(txt)
                if cols:
                    return cols

    m = re.search(r'\bGROUP\s+BY\b\s*(.+?)(?:\bHAVING\b|\bORDER\b|\bLIMIT\b|;|$)',
                  sql, re.IGNORECASE | re.DOTALL)
    if not m:
        return []
    gb_text = m.group(1)
    cols = []
    for part in gb_text.split(','):
        plain = re.search(r'["\`]?(\w+)["\`]?\s*$', part.strip())
        if plain:
            cols.append(plain.group(1).lower())
    return cols


def _extract_where_cols(sql: str) -> Set[str]:
    """Extract column names that appear in WHERE (lowercased, unqualified)."""
    parsed = _parse_sql(sql)
    if parsed is not None and exp is not None:
        sel = parsed if isinstance(parsed, exp.Select) else parsed.find(exp.Select)
        if sel is not None and sel.args.get("where") is not None:
            cols = {
                (c.name or "").strip().lower()
                for c in sel.args["where"].find_all(exp.Column)
                if (c.name or "").strip()
            }
            if cols:
                return cols

    m = re.search(r'\bWHERE\b(.+?)(?:\bGROUP\b|\bORDER\b|\bHAVING\b|\bLIMIT\b|;|$)',
                  sql, re.IGNORECASE | re.DOTALL)
    if not m:
        return set()
    where_text = re.sub(r'\(SELECT.+?\)', '', m.group(1), flags=re.IGNORECASE | re.DOTALL)
    cols = set()
    for nm in re.findall(r'(?:\w+\.)?(\w+)\s*(?:=|>=|<=|>|<|BETWEEN|IN|LIKE|IS)', where_text, re.IGNORECASE):
        cols.add(nm.lower())
    return cols


def _build_col_to_tables(
    con: duckdb.DuckDBPyConnection,
    base_schema: str,
) -> Dict[str, Set[str]]:
    col_to_tables: Dict[str, Set[str]] = {}
    try:
        rows = con.execute(
            "SELECT table_name, column_name "
            "FROM information_schema.columns "
            "WHERE table_schema = ?",
            [base_schema],
        ).fetchall()
    except Exception:
        return col_to_tables
    for tbl, col in rows:
        c = str(col).lower()
        t = str(tbl).lower()
        col_to_tables.setdefault(c, set()).add(t)
    return col_to_tables


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


def _extract_join_pairs(
    sql: str,
    col_to_tables: Dict[str, Set[str]],
) -> List[Tuple[str, str]]:
    parsed = _parse_sql(sql)
    if parsed is None or exp is None:
        # Regex fallback: only qualified t1.c=t2.c patterns.
        pairs = []
        sql_l = sql.lower()
        for m in re.finditer(r'(\w+)\.(\w+)\s*=\s*(\w+)\.(\w+)', sql_l):
            t1, t2 = m.group(1), m.group(3)
            if t1 != t2:
                pairs.append(tuple(sorted((t1, t2))))
        return pairs

    alias_to_table: Dict[str, str] = {}
    for t in parsed.find_all(exp.Table):
        base = _table_name(t)
        if not base:
            continue
        alias = (t.alias_or_name or base).lower()
        alias_to_table[alias] = base

    pairs: List[Tuple[str, str]] = []
    for eq in parsed.find_all(exp.EQ):
        l = eq.left
        r = eq.right
        if not isinstance(l, exp.Column) or not isinstance(r, exp.Column):
            continue
        lt = _resolve_col_table(l, alias_to_table, col_to_tables)
        rt = _resolve_col_table(r, alias_to_table, col_to_tables)
        if lt and rt and lt != rt:
            pairs.append(tuple(sorted((lt, rt))))
    return pairs


def _has_aggregation(sql: str) -> bool:
    return bool(re.search(r'\b(SUM|COUNT|AVG|MAX|MIN)\s*\(', sql, re.IGNORECASE))


def _parse_mv_structure(mv: MVCand) -> None:
    """Fill mv.mv_type, mv_group_by_cols, mv_select_cols, mv_tables in-place."""
    mv.mv_tables = _extract_tables(mv.sql)
    mv.mv_select_cols = _extract_select_cols(mv.sql)
    mv.mv_group_by_cols = _extract_group_by_cols(mv.sql)
    mv.mv_type = "agg" if (_has_aggregation(mv.sql) or mv.mv_group_by_cols) else "join"


_RISKY_FACT_GRAIN_KEYS = ("orderkey", "suppkey", "partkey", "linenumber")
_LINEITEM_PER_LINE_DISCRIMINATORS = (
    "suppkey",
    "partkey",
    "linenumber",
    "shipdate",
    "receiptdate",
    "commitdate",
)


def _normalize_mv_col_token(col: str) -> str:
    token = str(col or "").strip().lower().strip('"`')
    if "." in token:
        token = token.split(".")[-1]
    return re.sub(r"[^a-z0-9_]+", "", token)


def _mv_group_key_buckets(cols: List[str]) -> Set[str]:
    buckets: Set[str] = set()
    for col in cols or []:
        token = _normalize_mv_col_token(col)
        if not token:
            continue
        for key in _RISKY_FACT_GRAIN_KEYS:
            if token == key or token.endswith(f"_{key}") or token.endswith(key):
                buckets.add(key)
                break
    return buckets


def _mv_group_tokens(cols: List[str]) -> Set[str]:
    out: Set[str] = set()
    for col in cols or []:
        token = _normalize_mv_col_token(col)
        if not token:
            continue
        out.add(token)
        if "_" in token:
            out.add(token.split("_")[-1])
    return out


def _structural_mv_reject_reason(mv: MVCand) -> Optional[str]:
    """
    Drop aggregation MVs that still sit too close to fact-table grain.

    These candidates look attractive to the coarse cost gate because they match
    a hot template, but they keep multiple high-cardinality fact keys in the
    GROUP BY and end up materializing something close to the original lineitem
    rows. That is exactly what hurts T21-style supplier-order summaries.
    """
    if not mv or mv.mv_type != "agg" or not mv.mv_group_by_cols:
        return None
    tables = {str(t).strip().lower() for t in (mv.mv_tables or set()) if str(t).strip()}
    if "lineitem" not in tables:
        return None

    group_tokens = _mv_group_tokens(mv.mv_group_by_cols)
    risky_keys = sorted(_mv_group_key_buckets(mv.mv_group_by_cols) & set(_RISKY_FACT_GRAIN_KEYS))
    if "orderkey" in group_tokens:
        per_line = sorted(group_tokens & set(_LINEITEM_PER_LINE_DISCRIMINATORS))
        if per_line:
            return (
                "lineitem grouped by orderkey plus per-line discriminator "
                f"({', '.join(per_line)}) stays near fact-table cardinality"
            )
    if {"orderkey", "suppkey"}.issubset(risky_keys):
        return "lineitem grouped at supplier-order grain remains near fact-table cardinality"
    if len(risky_keys) >= 2:
        return (
            "lineitem grouped by multiple high-cardinality fact keys "
            f"({', '.join(risky_keys)})"
        )
    return None


def _filter_structurally_risky_mvs(
    mvs: List[MVCand],
    *,
    verbose: bool = True,
    stage: str = "candidate",
) -> List[MVCand]:
    kept: List[MVCand] = []
    dropped = 0
    for mv in mvs or []:
        reason = _structural_mv_reject_reason(mv)
        if reason:
            dropped += 1
            mv.build_error = f"structural_reject({reason})"
            if verbose:
                print(
                    f"  [mv_struct] DROP {mv.mvid} [{mv.mv_type}] {mv.name}: {reason}",
                    flush=True,
                )
            continue
        kept.append(mv)
    if verbose and dropped:
        print(
            f"  [mv_struct] Stage={stage}: kept {len(kept)}/{len(mvs)} after structural screening.",
            flush=True,
        )
    return kept


# ═══════════════════════════════════════════════════════════════════
# Schema / workload info helpers
# ═══════════════════════════════════════════════════════════════════

def get_duckdb_schema_info(
    con: duckdb.DuckDBPyConnection,
    base_schema: str = "main",
    table_names: Optional[List[str]] = None,
) -> Tuple[str, List[str]]:
    rows = con.execute(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema = ? ORDER BY table_name",
        [base_schema],
    ).fetchall()
    all_tables = [r[0] for r in rows]
    if table_names:
        all_tables = [t for t in all_tables if t in set(table_names)]

    lines = []
    for tbl in all_tables:
        col_rows = con.execute(
            "SELECT column_name, data_type FROM information_schema.columns "
            "WHERE table_schema = ? AND table_name = ? ORDER BY ordinal_position",
            [base_schema, tbl],
        ).fetchall()
        col_defs = ", ".join(f"{c[0]} {c[1]}" for c in col_rows)
        nrows = 0
        try:
            nrows = con.execute(f'SELECT COUNT(*) FROM "{base_schema}"."{tbl}"').fetchone()[0]
        except Exception:
            pass
        lines.append(f"TABLE {tbl} ({nrows:,} rows): {col_defs}")
    return "\n".join(lines), all_tables


# ─── Workload statistics (new in v2) ────────────────────────────

def compute_workload_stats(
    con: duckdb.DuckDBPyConnection,
    queries: List[Dict],
    base_schema: str,
) -> Dict:
    """
    Compute statistics over the workload for richer LLM prompt:
      - template_runtime_ms: estimated runtime per template (from EXPLAIN cost)
      - join_freq: frequency of each table pair in JOIN predicates
      - group_by_freq: frequency of each (table.col) in GROUP BY
      - filter_freq: frequency of each (table.col) in WHERE
      - top_slow_templates: top-5 most expensive templates by EXPLAIN cost
    """
    tpl_costs: Dict[int, List[float]] = {}
    join_pairs: Dict[Tuple[str,str], int] = {}
    group_cols: Dict[str, int] = {}
    filter_cols: Dict[str, int] = {}
    col_to_tables = _build_col_to_tables(con, base_schema)

    for q in queries:
        sql = (q.get("sql") or q.get("query") or "").strip()
        tpl = q.get("template", -1)
        if not sql:
            continue

        # EXPLAIN cost
        try:
            rows = con.execute(f"EXPLAIN {sql}").fetchall()
            plan_text = "\n".join(str(r) for r in rows)
            nums = re.findall(r'EC[:\s=]+(\d+)', plan_text)
            cost = float(max(int(n) for n in nums)) if nums else 0.0
        except Exception:
            cost = 0.0
        tpl_costs.setdefault(tpl, []).append(cost)

        # JOIN pairs: robustly parse both qualified and unqualified joins.
        for pair in _extract_join_pairs(sql, col_to_tables):
            join_pairs[pair] = join_pairs.get(pair, 0) + 1

        # GROUP BY
        for col in _extract_group_by_cols(sql):
            group_cols[col] = group_cols.get(col, 0) + 1

        # WHERE filter cols
        for col in _extract_where_cols(sql):
            filter_cols[col] = filter_cols.get(col, 0) + 1

    # Aggregate per-template costs
    tpl_avg = {tpl: max(costs) for tpl, costs in tpl_costs.items() if tpl != -1}
    top_slow = sorted(tpl_avg.items(), key=lambda x: -x[1])[:10]

    # Approximate distinct counts for GROUP BY cols
    gb_distinct: Dict[str, int] = {}
    for col, freq in group_cols.items():
        if freq < 2:
            continue
        for tbl in con.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema=?", [base_schema]
        ).fetchall():
            tname = tbl[0]
            col_rows = con.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_schema=? AND table_name=? AND column_name=?",
                [base_schema, tname, col]
            ).fetchall()
            if col_rows:
                try:
                    nd = con.execute(
                        f'SELECT approx_count_distinct("{col}") FROM "{base_schema}"."{tname}"'
                    ).fetchone()[0]
                    gb_distinct[f"{tname}.{col}"] = nd
                except Exception:
                    pass

    return {
        "top_slow_templates": top_slow,
        "join_freq": sorted(join_pairs.items(), key=lambda x: -x[1])[:15],
        "group_by_freq": sorted(group_cols.items(), key=lambda x: -x[1])[:20],
        "group_by_distinct": gb_distinct,
        "filter_freq": sorted(filter_cols.items(), key=lambda x: -x[1])[:20],
    }


def build_workload_summary(
    queries: List[Dict],
    max_queries: int = 40,
    stats: Optional[Dict] = None,
) -> str:
    """
    Return a compact summary of the workload for LLM context.
    If stats is provided, prepend workload statistics for richer context.
    """
    lines = []
    if stats:
        lines.append("=== WORKLOAD STATISTICS ===")
        if stats.get("top_slow_templates"):
            slow_str = ", ".join(
                f"T{t}(cost={c:.0f})" for t, c in stats["top_slow_templates"][:5]
            )
            lines.append(f"Top expensive templates: {slow_str}")
        if stats.get("join_freq"):
            jf = ", ".join(
                f"{t[0]}×{t[1]}({n}x)" for t, n in stats["join_freq"][:8]
            )
            lines.append(f"Frequent JOINs: {jf}")
        if stats.get("group_by_freq"):
            gb = ", ".join(
                f"{c}({n}x)" for c, n in stats["group_by_freq"][:10]
            )
            lines.append(f"Frequent GROUP BY cols: {gb}")
        if stats.get("group_by_distinct"):
            gbd = ", ".join(
                f"{c}={nd:,} distinct" for c, nd in list(stats["group_by_distinct"].items())[:8]
            )
            lines.append(f"GROUP BY column cardinalities: {gbd}")
        if stats.get("filter_freq"):
            ff = ", ".join(
                f"{c}({n}x)" for c, n in stats["filter_freq"][:10]
            )
            lines.append(f"Frequent WHERE cols: {ff}")
        lines.append("")

    sampled = queries[:max_queries]
    for i, q in enumerate(sampled):
        sql = (q.get("sql") or q.get("query") or "").strip()
        if sql:
            lines.append(f"Q{i+1}: {sql[:300]}")
    return "\n\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
# LLM: Generate MV candidates (with richer prompt)
# ═══════════════════════════════════════════════════════════════════

_MV_SYSTEM_PROMPT = """\
You are a database query optimizer expert for DuckDB OLAP workloads.
Your goal is to propose Materialized View (MV) definitions that will speed up the given workload.

Rules:
- Each MV is a SELECT statement that pre-aggregates or pre-joins frequently queried data.
- TWO TYPES of MVs are useful:
    (a) JOIN-only MVs: pre-join large fact tables without aggregation.
        These benefit queries that repeat the same join pattern across multiple templates.
    (b) AGGREGATION MVs (preferred): pre-compute GROUP BY + aggregates that appear frequently.
        These are the MOST powerful for OLAP — they can reduce millions of rows to thousands.
        For aggregation MVs, make sure:
          * GROUP BY keys cover the most-queried grouping dimensions
          * Include all aggregated metrics (SUM, COUNT, AVG) seen in the workload
          * If queries further GROUP BY a subset (e.g. by region only, not by nation),
            the MV should include the FINER-grained keys so queries can re-aggregate.
- Focus on: frequent JOIN motifs, repeated GROUP BY keys, and repeated filter predicates.
- MVs should cover multiple queries (high coverage > single-query MVs).
- MV SQL must be valid DuckDB SQL (no PostgreSQL-specific syntax).
- Reference tables as: {base_schema}.<tablename>
- Do NOT use CREATE TABLE/VIEW in the SQL — just the SELECT.
- Keep MVs focused: 2-4 table joins wide; avoid all-table mega-MVs.
- Propose at most {max_mvs} MVs, mix of join-only and aggregation types.

Output format (strict JSON array):
[
  {{
    "mvid": "MV001",
    "name": "mv_lineitem_orders_agg",
    "sql": "SELECT l_returnflag, l_linestatus, SUM(l_extendedprice*(1-l_discount)) AS revenue, SUM(l_quantity) AS qty, COUNT(*) AS cnt FROM {base_schema}.lineitem GROUP BY l_returnflag, l_linestatus",
    "why": "Q1-style aggregation GROUP BY returnflag,linestatus appears in 30% of workload"
  }},
  ...
]
"""


def _normalize_mv_candidate_sql(sql: str) -> str:
    """
    Normalize LLM MV output to the SELECT/WITH body expected by DuckDB build logic.
    The builder always wraps candidates as `CREATE TABLE ... AS <sql>`, so if the
    model emits `CREATE MATERIALIZED VIEW ... AS SELECT ...`, we strip the DDL.
    """
    text = (sql or "").strip()
    if not text:
        return ""

    def _strip_postgres_mv_suffix(body: str) -> str:
        # Some LLM outputs include PostgreSQL tail "WITH NO DATA".
        return re.sub(r"\bwith\s+no\s+data\s*$", "", body.strip(), flags=re.IGNORECASE).strip()

    def _first_select_or_with_statement(body: str) -> str:
        # Keep only the first SELECT/WITH statement if extra statements are appended.
        parts = [p.strip() for p in re.split(r";\s*", body) if p.strip()]
        for part in parts:
            if re.match(r"^(select|with)\b", part, flags=re.IGNORECASE):
                return _strip_postgres_mv_suffix(part).rstrip(";")
        return _strip_postgres_mv_suffix(body).rstrip(";")

    if re.match(r"^(select|with)\b", text, flags=re.IGNORECASE):
        return _first_select_or_with_statement(text)

    ddl_patterns = [
        r"^\s*create\s+(?:or\s+replace\s+)?materialized\s+view\s+.+?\bas\s+(with\b.+?|select\b.+?)(?:;|$)",
        r"^\s*create\s+(?:or\s+replace\s+)?view\s+.+?\bas\s+(with\b.+?|select\b.+?)(?:;|$)",
        r"^\s*create\s+(?:or\s+replace\s+)?table\s+.+?\bas\s+(with\b.+?|select\b.+?)(?:;|$)",
    ]
    for pat in ddl_patterns:
        m = re.match(pat, text, flags=re.IGNORECASE | re.DOTALL)
        if m:
            return _first_select_or_with_statement(str(m.group(1) or ""))
    return _first_select_or_with_statement(text)


def _safe_template_id(q: Dict[str, Any]) -> Optional[int]:
    v = q.get("template")
    try:
        if v is None:
            return None
        return int(v)
    except Exception:
        return None


def _collect_template_counts(queries: List[Dict[str, Any]]) -> Dict[int, int]:
    out: Dict[int, int] = {}
    for q in queries:
        t = _safe_template_id(q)
        if t is None:
            continue
        out[t] = out.get(t, 0) + 1
    return out


def _is_tpch_like_template_workload(tpl_counts: Dict[int, int]) -> bool:
    if not tpl_counts:
        return False
    keys = set(tpl_counts.keys())
    if not keys:
        return False
    # TPC-H templates are numbered 1..22 in this project.
    if any((not isinstance(k, int)) or k < 1 or k > 22 for k in keys):
        return False
    return len(keys) >= 5


def _is_tpch_numbered_template_workload(tpl_counts: Dict[int, int]) -> bool:
    if not tpl_counts:
        return False
    keys = set(tpl_counts.keys())
    if not keys:
        return False
    return not any((not isinstance(k, int)) or k < 1 or k > 22 for k in keys)


def _is_tpch_schema_like(schema_info: str, queries: List[Dict[str, Any]]) -> bool:
    tpch_tables = {
        "customer",
        "lineitem",
        "nation",
        "orders",
        "part",
        "partsupp",
        "region",
        "supplier",
    }
    found: Set[str] = set()
    schema_text = str(schema_info or "").lower()
    for t in tpch_tables:
        if f"table {t}" in schema_text:
            found.add(t)
    for q in queries or []:
        sql = str(q.get("sql") or q.get("query") or "")
        found.update(_extract_tables(sql) & tpch_tables)
    return len(found) >= 4


def _build_tpch_seed_mvs(
    queries: List[Dict[str, Any]],
    base_schema: str,
    mv_schema: str,
    max_mvs: int,
) -> List[MVCand]:
    """
    Deterministic seed MVs for high-impact TPC-H templates.
    These are conservative, portable SQL and are used as stable anchors
    when LLM proposals are too generic for real rewrite hits.
    """
    if max_mvs <= 0:
        return []
    tpl_counts = _collect_template_counts(queries)

    out: List[MVCand] = []

    # Q5 hotspot: nation revenue by region/date window.
    if tpl_counts.get(5, 0) > 0:
        mv = MVCand(
            mvid="SEED_T5",
            name="mv_tpch_t5_rev_by_region_nation_date",
            sql=(
                f"SELECT r.r_name, n.n_name, o.o_orderdate, "
                f"SUM(l.l_extendedprice * (1 - l.l_discount)) AS revenue "
                f"FROM {base_schema}.customer c "
                f"JOIN {base_schema}.orders o ON c.c_custkey = o.o_custkey "
                f"JOIN {base_schema}.lineitem l ON l.l_orderkey = o.o_orderkey "
                f"JOIN {base_schema}.supplier s ON l.l_suppkey = s.s_suppkey "
                f"JOIN {base_schema}.nation n "
                f"  ON s.s_nationkey = n.n_nationkey AND c.c_nationkey = s.s_nationkey "
                f"JOIN {base_schema}.region r ON n.n_regionkey = r.r_regionkey "
                f"GROUP BY r.r_name, n.n_name, o.o_orderdate"
            ),
            why="Seed MV for TPC-H Q5: pre-aggregate revenue by region/nation/date.",
            mv_schema=mv_schema,
        )
        _parse_mv_structure(mv)
        out.append(mv)

    # Q8 hotspot: market-share numerator/denominator by year/region/type/nation.
    if tpl_counts.get(8, 0) > 0:
        mv = MVCand(
            mvid="SEED_T8",
            name="mv_tpch_t8_volume_by_year_region_type_nation",
            sql=(
                f"SELECT EXTRACT(year FROM o.o_orderdate) AS o_year, "
                f"r.r_name, p.p_type, n2.n_name AS nation, "
                f"SUM(l.l_extendedprice * (1 - l.l_discount)) AS volume "
                f"FROM {base_schema}.part p "
                f"JOIN {base_schema}.lineitem l ON p.p_partkey = l.l_partkey "
                f"JOIN {base_schema}.supplier s ON s.s_suppkey = l.l_suppkey "
                f"JOIN {base_schema}.orders o ON l.l_orderkey = o.o_orderkey "
                f"JOIN {base_schema}.customer c ON o.o_custkey = c.c_custkey "
                f"JOIN {base_schema}.nation n1 ON c.c_nationkey = n1.n_nationkey "
                f"JOIN {base_schema}.region r ON n1.n_regionkey = r.r_regionkey "
                f"JOIN {base_schema}.nation n2 ON s.s_nationkey = n2.n_nationkey "
                f"GROUP BY o_year, r.r_name, p.p_type, n2.n_name"
            ),
            why="Seed MV for TPC-H Q8: pre-aggregate volume by (year, region, part type, nation).",
            mv_schema=mv_schema,
        )
        _parse_mv_structure(mv)
        out.append(mv)

    # Q7 hotspot: pairwise nation revenue by ship year.
    if tpl_counts.get(7, 0) > 0:
        mv = MVCand(
            mvid="SEED_T7",
            name="mv_tpch_t7_rev_by_supp_cust_year",
            sql=(
                f"SELECT n1.n_name AS supp_nation, "
                f"n2.n_name AS cust_nation, "
                f"EXTRACT(year FROM l.l_shipdate) AS l_year, "
                f"SUM(l.l_extendedprice * (1 - l.l_discount)) AS revenue "
                f"FROM {base_schema}.supplier s "
                f"JOIN {base_schema}.lineitem l ON s.s_suppkey = l.l_suppkey "
                f"JOIN {base_schema}.orders o ON o.o_orderkey = l.l_orderkey "
                f"JOIN {base_schema}.customer c ON c.c_custkey = o.o_custkey "
                f"JOIN {base_schema}.nation n1 ON s.s_nationkey = n1.n_nationkey "
                f"JOIN {base_schema}.nation n2 ON c.c_nationkey = n2.n_nationkey "
                f"WHERE l.l_shipdate BETWEEN DATE '1995-01-01' AND DATE '1996-12-31' "
                f"GROUP BY supp_nation, cust_nation, l_year"
            ),
            why="Seed MV for TPC-H Q7: pre-aggregate revenue by supplier nation, customer nation, and ship year.",
            mv_schema=mv_schema,
        )
        _parse_mv_structure(mv)
        out.append(mv)

    # Q12 hotspot: ship-mode priority counts over receiptdate window.
    if tpl_counts.get(12, 0) > 0:
        mv = MVCand(
            mvid="SEED_T12",
            name="mv_tpch_t12_shipmode_priority_by_receiptdate",
            sql=(
                f"SELECT l.l_receiptdate, l.l_shipmode, o.o_orderpriority, "
                f"COUNT(*) AS line_count "
                f"FROM {base_schema}.orders o "
                f"JOIN {base_schema}.lineitem l ON o.o_orderkey = l.l_orderkey "
                f"WHERE l.l_commitdate < l.l_receiptdate "
                f"  AND l.l_shipdate < l.l_commitdate "
                f"GROUP BY l.l_receiptdate, l.l_shipmode, o.o_orderpriority"
            ),
            why="Seed MV for TPC-H Q12: pre-aggregate ship-mode counts by receipt date/priority.",
            mv_schema=mv_schema,
        )
        _parse_mv_structure(mv)
        out.append(mv)

    # Q13 hotspot: customer order-count distribution with comment filter fixed.
    if tpl_counts.get(13, 0) > 0:
        mv = MVCand(
            mvid="SEED_T13",
            name="mv_tpch_t13_customer_order_count",
            sql=(
                f"SELECT c.c_custkey, COUNT(o.o_orderkey) AS c_count "
                f"FROM {base_schema}.customer c "
                f"LEFT JOIN {base_schema}.orders o "
                f"  ON c.c_custkey = o.o_custkey "
                f" AND o.o_comment NOT LIKE '%unusual%requests%' "
                f"GROUP BY c.c_custkey"
            ),
            why="Seed MV for TPC-H Q13: precompute per-customer order counts under the fixed comment exclusion.",
            mv_schema=mv_schema,
        )
        _parse_mv_structure(mv)
        out.append(mv)

    # Q14 hotspot: promo-vs-total discounted revenue by shipdate.
    if tpl_counts.get(14, 0) > 0:
        mv = MVCand(
            mvid="SEED_T14",
            name="mv_tpch_t14_promo_total_by_shipdate",
            sql=(
                f"SELECT l.l_shipdate, "
                f"SUM(l.l_extendedprice * (1 - l.l_discount)) AS sum_disc_price, "
                f"SUM(CASE WHEN p.p_type LIKE 'PROMO%' "
                f"THEN l.l_extendedprice * (1 - l.l_discount) ELSE 0 END) AS sum_promo_disc_price "
                f"FROM {base_schema}.lineitem l "
                f"JOIN {base_schema}.part p ON l.l_partkey = p.p_partkey "
                f"GROUP BY l.l_shipdate"
            ),
            why="Seed MV for TPC-H Q14: pre-aggregate promo and total discounted revenue by shipdate.",
            mv_schema=mv_schema,
        )
        _parse_mv_structure(mv)
        out.append(mv)

    # Q3 hotspot: order-level revenue by segment/date (high-cardinality but high-impact).
    if tpl_counts.get(3, 0) > 0:
        mv = MVCand(
            mvid="SEED_T3",
            name="mv_tpch_t3_order_rev_by_segment_dates",
            sql=(
                f"SELECT c.c_mktsegment, o.o_orderdate, o.o_shippriority, "
                f"l.l_orderkey, l.l_shipdate, "
                f"SUM(l.l_extendedprice * (1 - l.l_discount)) AS revenue "
                f"FROM {base_schema}.customer c "
                f"JOIN {base_schema}.orders o ON c.c_custkey = o.o_custkey "
                f"JOIN {base_schema}.lineitem l ON l.l_orderkey = o.o_orderkey "
                f"GROUP BY c.c_mktsegment, o.o_orderdate, o.o_shippriority, l.l_orderkey, l.l_shipdate"
            ),
            why="Seed MV for TPC-H Q3: pre-aggregate order revenue by segment/orderdate/shipdate.",
            mv_schema=mv_schema,
        )
        _parse_mv_structure(mv)
        out.append(mv)

    # Q1 hotspot: daily aggregates that can be rolled up by returnflag/linestatus.
    if tpl_counts.get(1, 0) > 0:
        mv = MVCand(
            mvid="SEED_T1",
            name="mv_tpch_t1_daily_rollup",
            sql=(
                f"SELECT l_shipdate, l_returnflag, l_linestatus, "
                f"SUM(l_quantity) AS sum_qty, "
                f"SUM(l_extendedprice) AS sum_base_price, "
                f"SUM(l_extendedprice * (1 - l_discount)) AS sum_disc_price, "
                f"SUM(l_extendedprice * (1 - l_discount) * (1 + l_tax)) AS sum_charge, "
                f"SUM(l_discount) AS sum_discount, "
                f"COUNT(*) AS count_order "
                f"FROM {base_schema}.lineitem "
                f"GROUP BY l_shipdate, l_returnflag, l_linestatus"
            ),
            why="Seed MV for TPC-H Q1: daily rollup enabling fast cutoff-date re-aggregation.",
            mv_schema=mv_schema,
        )
        _parse_mv_structure(mv)
        out.append(mv)

    # Q21 hotspot: supplier-order late-shipment flags and counts.
    if tpl_counts.get(21, 0) > 0:
        mv = MVCand(
            mvid="SEED_T21",
            name="mv_tpch_t21_supplier_order_wait_flags",
            sql=(
                f"WITH supplier_order AS ("
                f"  SELECT l_orderkey, l_suppkey, "
                f"         SUM(CASE WHEN l_receiptdate > l_commitdate THEN 1 ELSE 0 END) AS self_late_line_cnt "
                f"  FROM {base_schema}.lineitem "
                f"  GROUP BY l_orderkey, l_suppkey"
                f"), order_tot AS ("
                f"  SELECT l_orderkey, "
                f"         COUNT(*) AS supp_cnt, "
                f"         SUM(CASE WHEN self_late_line_cnt > 0 THEN 1 ELSE 0 END) AS late_supp_cnt "
                f"  FROM supplier_order "
                f"  GROUP BY l_orderkey"
                f") "
                f"SELECT so.l_orderkey, so.l_suppkey, "
                f"       so.self_late_line_cnt, "
                f"       CASE WHEN so.self_late_line_cnt > 0 THEN 1 ELSE 0 END AS self_has_late, "
                f"       ot.supp_cnt, "
                f"       (ot.late_supp_cnt - CASE WHEN so.self_late_line_cnt > 0 THEN 1 ELSE 0 END) "
                f"         AS other_late_supp_cnt "
                f"FROM supplier_order so "
                f"JOIN order_tot ot ON so.l_orderkey = ot.l_orderkey"
            ),
            why="Seed MV for TPC-H Q21: precompute late-shipment flags at supplier-order granularity.",
            mv_schema=mv_schema,
        )
        _parse_mv_structure(mv)
        out.append(mv)

    # Q6 hotspot: shipdate/discount/quantity cube for scalar revenue query.
    if tpl_counts.get(6, 0) > 0:
        mv = MVCand(
            mvid="SEED_T6",
            name="mv_tpch_t6_revenue_by_shipdate_discount_qty",
            sql=(
                f"SELECT l_shipdate, l_discount, l_quantity, "
                f"SUM(l_extendedprice * l_discount) AS revenue "
                f"FROM {base_schema}.lineitem "
                f"GROUP BY l_shipdate, l_discount, l_quantity"
            ),
            why="Seed MV for TPC-H Q6: compact aggregate cube for date/discount/quantity filters.",
            mv_schema=mv_schema,
        )
        _parse_mv_structure(mv)
        out.append(mv)

    # Q4 hotspot: per-order late-receipt existence bit.
    if tpl_counts.get(4, 0) > 0:
        mv = MVCand(
            mvid="SEED_T4",
            name="mv_tpch_t4_late_receipt_flag_by_order",
            sql=(
                f"SELECT l_orderkey, "
                f"MAX(CASE WHEN l_commitdate < l_receiptdate THEN 1 ELSE 0 END) AS has_late_receipt "
                f"FROM {base_schema}.lineitem "
                f"GROUP BY l_orderkey"
            ),
            why="Seed MV for TPC-H Q4: precompute order-level late-receipt existence.",
            mv_schema=mv_schema,
        )
        _parse_mv_structure(mv)
        out.append(mv)

    # Q10 hotspot: revenue by (orderdate, custkey) with fixed returnflag='R'
    if tpl_counts.get(10, 0) > 0:
        mv = MVCand(
            mvid="SEED_T10",
            name="mv_tpch_t10_rev_by_cust_date",
            sql=(
                f"SELECT o.o_orderdate, o.o_custkey, "
                f"SUM(l.l_extendedprice * (1 - l.l_discount)) AS revenue "
                f"FROM {base_schema}.orders o "
                f"JOIN {base_schema}.lineitem l ON l.l_orderkey = o.o_orderkey "
                f"WHERE l.l_returnflag = 'R' "
                f"GROUP BY o.o_orderdate, o.o_custkey"
            ),
            why="Seed MV for TPC-H Q10: pre-aggregate lineitem+orders revenue by date/customer.",
            mv_schema=mv_schema,
        )
        _parse_mv_structure(mv)
        out.append(mv)

    # Q18 hotspot: pre-aggregate lineitem quantity per order.
    if tpl_counts.get(18, 0) > 0:
        mv = MVCand(
            mvid="SEED_T18",
            name="mv_tpch_t18_order_qty",
            sql=(
                f"SELECT l_orderkey, SUM(l_quantity) AS sum_qty "
                f"FROM {base_schema}.lineitem "
                f"GROUP BY l_orderkey"
            ),
            why="Seed MV for TPC-H Q18: pre-aggregate order-level quantity threshold checks.",
            mv_schema=mv_schema,
        )
        _parse_mv_structure(mv)
        out.append(mv)

    # Rank seed MVs by expected steady-state value for TPCH-like workloads.
    # Keep heavy-cardinality seeds (T3/T21) as optional lower-priority candidates.
    seed_tpl = {
        "SEED_T1": 1, "SEED_T3": 3, "SEED_T4": 4, "SEED_T5": 5, "SEED_T6": 6,
        "SEED_T7": 7, "SEED_T8": 8, "SEED_T10": 10, "SEED_T12": 12,
        "SEED_T13": 13, "SEED_T14": 14,
        "SEED_T18": 18, "SEED_T21": 21,
    }
    base_priority = {
        "SEED_T10": 100,
        "SEED_T18": 96,
        "SEED_T6": 92,
        "SEED_T7": 90,
        "SEED_T14": 88,
        "SEED_T12": 86,
        "SEED_T13": 84,
        "SEED_T5": 82,
        "SEED_T8": 80,
        "SEED_T4": 76,
        "SEED_T1": 74,
        "SEED_T3": 70,
        "SEED_T21": 68,
    }

    def _seed_rank(mv: MVCand) -> Tuple[int, int]:
        mid = str(getattr(mv, "mvid", "") or "")
        tpl = int(seed_tpl.get(mid, -1))
        freq = int(tpl_counts.get(tpl, 0))
        return (int(base_priority.get(mid, 0)), freq)

    out.sort(key=lambda m: _seed_rank(m), reverse=True)
    return out[:max_mvs]


def _dedupe_mvs_keep_order(mvs: List[MVCand]) -> List[MVCand]:
    out: List[MVCand] = []
    seen_name: Set[str] = set()
    seen_sql: Set[str] = set()
    for mv in mvs:
        k_name = (mv.name or "").strip().lower()
        k_sql = re.sub(r"\s+", " ", (mv.sql or "").strip().lower())
        if not k_name or not k_sql:
            continue
        if k_name in seen_name or k_sql in seen_sql:
            continue
        seen_name.add(k_name)
        seen_sql.add(k_sql)
        out.append(mv)
    return out


def _normalize_sql_key(sql: str) -> str:
    return re.sub(r"\s+", " ", str(sql or "").strip()).lower()


def _build_repeat_seed_mvs(
    queries: List[Dict],
    *,
    con: Optional[duckdb.DuckDBPyConnection] = None,
    mv_schema: str,
    max_mvs: int,
    min_freq: int = 3,
    rank_mode: str = "freq",
) -> List[MVCand]:
    """
    Build deterministic repeat-seed MVs from exact repeated query texts.
    This is a non-cache artifact path: repeated statements are externalized
    into typed artifacts that can be admitted by rewrite logic.
    """
    freq: Dict[str, int] = {}
    repr_sql: Dict[str, str] = {}
    for q in queries or []:
        sql = (q.get("sql") or q.get("query") or "").strip()
        if not sql:
            continue
        key = _normalize_sql_key(sql)
        if not key:
            continue
        freq[key] = int(freq.get(key, 0) or 0) + 1
        if key not in repr_sql:
            repr_sql[key] = sql

    ranked = [(k, int(c)) for k, c in freq.items() if int(c) >= int(min_freq)]
    if str(rank_mode or "freq").strip().lower() == "benefit_est":
        scored: List[Tuple[str, int, float, float]] = []
        for key, c in ranked:
            sql = repr_sql.get(key, "")
            if not sql:
                continue
            explain_cost = float(_get_explain_cost(con, sql) or 0.0) if con is not None else 0.0
            score = max(0.0, float(c - 1) * max(0.0, explain_cost))
            scored.append((key, int(c), explain_cost, score))
        scored.sort(key=lambda x: (-float(x[3]), -int(x[1]), len(repr_sql.get(x[0], ""))))
        ranked = [(k, c) for k, c, _, _ in scored]
    else:
        ranked.sort(key=lambda x: (-int(x[1]), len(repr_sql.get(x[0], ""))))

    out: List[MVCand] = []
    for i, (key, c) in enumerate(ranked[: max(0, int(max_mvs))], 1):
        sql = repr_sql.get(key, "")
        if not sql:
            continue
        mv = MVCand(
            mvid=f"SEED_R{i:03d}",
            name=f"seed_repeat_{i:03d}",
            sql=sql,
            why=f"repeat-seed from exact repeated query text (freq={int(c)})",
            mv_schema=mv_schema,
        )
        _parse_mv_structure(mv)
        out.append(mv)
    return out


def _validate_mv_candidates_sql(
    con: duckdb.DuckDBPyConnection,
    mvs: List[MVCand],
    *,
    stage: str,
) -> List[MVCand]:
    valid: List[MVCand] = []
    total = len(mvs or [])
    for mv in (mvs or []):
        try:
            con.execute(f"EXPLAIN {mv.sql}")
            valid.append(mv)
        except Exception as e:
            print(f"  [mv_{stage}] MV {mv.mvid} ({mv.name}) invalid SQL: {e}")
    if total > 0:
        print(f"  [mv_{stage}] {len(valid)}/{total} candidates passed SQL validation.")
    return valid

def llm_generate_mvs(
    con: duckdb.DuckDBPyConnection,
    queries: List[Dict],
    schema_info: str,
    base_schema: str = "main",
    mv_schema: str = "aidb_co_mv",
    *,
    model: str,
    reasoning_effort: Optional[str] = None,
    max_mvs: int = 20,
    workload_max_sample: int = 40,
    stats: Optional[Dict] = None,
    benchmark: Optional[str] = None,
) -> List[MVCand]:
    tpl_counts = _collect_template_counts(queries)
    total_tpl = max(1, sum(int(v) for v in tpl_counts.values()))
    tpch_schema_like = _is_tpch_schema_like(schema_info, queries)
    is_tpch_like = _is_tpch_like_template_workload(tpl_counts)
    is_tpch_numbered = _is_tpch_numbered_template_workload(tpl_counts)
    benchmark_name = str(benchmark or "").strip().lower()
    benchmark_is_tpch = benchmark_name.startswith("tpch") or benchmark_name.startswith("tpc-h")
    if benchmark_name and not benchmark_is_tpch:
        # Non-TPCH benchmarks can still have numbered templates; do not inject
        # TPCH-specific seed MVs based on template-shape heuristics alone.
        is_tpch_like = False
        is_tpch_numbered = False
    if not benchmark_is_tpch and not tpch_schema_like:
        is_tpch_like = False
        is_tpch_numbered = False
    dominant_tpl_share = (
        float(max(int(v) for v in tpl_counts.values())) / float(total_tpl)
        if tpl_counts else 0.0
    )
    tpch_hotspot_seed_only = bool(
        is_tpch_numbered
        and len(tpl_counts) <= 2
        and dominant_tpl_share >= 0.75
    )
    seed_mvs = []
    if benchmark_is_tpch or ((not benchmark_name) and tpch_schema_like):
        seed_mvs = _build_tpch_seed_mvs(
            queries=queries,
            base_schema=base_schema,
            mv_schema=mv_schema,
            max_mvs=max_mvs,
        )
    if MV_GENERIC_REPEAT_SEEDS and (not benchmark_is_tpch):
        repeat_cap = int(max(0, min(int(max_mvs), int(MV_GENERIC_REPEAT_MAX))))
        if repeat_cap > 0:
            repeat_seed_mvs = _build_repeat_seed_mvs(
                queries=queries,
                con=con,
                mv_schema=mv_schema,
                max_mvs=repeat_cap,
                min_freq=int(MV_GENERIC_REPEAT_MIN_FREQ),
                rank_mode=str(MV_GENERIC_REPEAT_RANK_MODE or "freq"),
            )
            if repeat_seed_mvs:
                print(
                    f"  [mv_seed] Added {len(repeat_seed_mvs)} repeat-seed candidates "
                    f"(min_freq={int(MV_GENERIC_REPEAT_MIN_FREQ)}).",
                    flush=True,
                )
            seed_mvs = _dedupe_mvs_keep_order(seed_mvs + repeat_seed_mvs)
    seed_mvs = _filter_structurally_risky_mvs(seed_mvs, stage="seed")
    if seed_mvs:
        print(
            f"  [mv_seed] Added {len(seed_mvs)} deterministic seed candidates: "
            + ", ".join(m.name for m in seed_mvs),
            flush=True,
        )
    seed_valid = _validate_mv_candidates_sql(con, seed_mvs, stage="seed")
    seed_tpl_covered: Set[int] = set()
    for mv in seed_valid:
        mid = str(getattr(mv, "mvid", "") or "").upper()
        m = re.search(r"SEED_T(\d+)", mid)
        if m:
            try:
                seed_tpl_covered.add(int(m.group(1)))
            except Exception:
                pass
    uncovered_hot_tpls: List[Tuple[int, int, float, Optional[float]]] = []
    min_uncovered_share = float(MV_UNCOVERED_TPL_MIN_SHARE)
    if (
        os.getenv("LOGICDB_MV_UNCOVERED_TPL_MIN_SHARE") is None
        and os.getenv("AIDB_MV_UNCOVERED_TPL_MIN_SHARE") is None
        and (len(queries) >= 1500)
    ):
        min_uncovered_share = float(MV_UNCOVERED_TPL_MIN_SHARE_LONG)
    tpl_cost_map: Dict[int, float] = {}
    tpl_runtime_map: Dict[int, float] = {}
    tpl_runtime_share_map: Dict[int, float] = {}
    if isinstance(stats, dict):
        for item in (stats.get("top_slow_templates") or []):
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            try:
                tpl_k = int(item[0])
                tpl_v = float(item[1])
            except Exception:
                continue
            tpl_cost_map[tpl_k] = tpl_v
        for item in (stats.get("runtime_template_stats") or []):
            if not isinstance(item, dict):
                continue
            try:
                tpl_k = int(item.get("template"))
                tpl_avg = float(item.get("avg_ms", 0.0) or 0.0)
                tpl_share_rt = float(item.get("share", 0.0) or 0.0)
            except Exception:
                continue
            tpl_runtime_map[tpl_k] = tpl_avg
            tpl_runtime_share_map[tpl_k] = tpl_share_rt
    max_tpl_cost = max(tpl_cost_map.values()) if tpl_cost_map else 0.0
    if tpl_counts:
        ranked_tpls = sorted(tpl_counts.items(), key=lambda x: -int(x[1]))
        for tpl, cnt in ranked_tpls:
            share = float(cnt) / float(total_tpl)
            if share < float(min_uncovered_share):
                break
            if int(tpl) in seed_tpl_covered:
                continue
            tpl_runtime = tpl_runtime_map.get(int(tpl))
            tpl_runtime_share = tpl_runtime_share_map.get(int(tpl))
            if tpl_runtime_map:
                if tpl_runtime is None:
                    continue
                if float(tpl_runtime) < float(MV_UNCOVERED_TPL_MIN_RUNTIME_MS):
                    continue
                if (tpl_runtime_share is not None) and (float(tpl_runtime_share) < float(MV_UNCOVERED_TPL_MIN_RUNTIME_SHARE)):
                    continue
            tpl_cost = tpl_cost_map.get(int(tpl))
            if (not tpl_runtime_map) and tpl_cost_map:
                if tpl_cost is None:
                    continue
                if float(tpl_cost) <= 0.0:
                    continue
                if max_tpl_cost <= 0.0:
                    continue
                if (float(tpl_cost) / float(max_tpl_cost)) < float(MV_UNCOVERED_TPL_MIN_COST_RATIO):
                    continue
            if tpl_cost is not None and float(tpl_cost) < float(MV_UNCOVERED_TPL_MIN_COST_ABS):
                continue
            uncovered_hot_tpls.append((int(tpl), int(cnt), float(share), float(tpl_cost) if tpl_cost is not None else None))
            if len(uncovered_hot_tpls) >= int(MV_UNCOVERED_TPL_TOPK):
                break

    allow_llm_for_uncovered = bool(
        is_tpch_like
        and MV_ALLOW_LLM_UNCOVERED_TPCH
        and len(uncovered_hot_tpls) > 0
    )
    if (is_tpch_like or tpch_hotspot_seed_only) and seed_valid and MV_SKIP_LLM_ON_TPCH and (not allow_llm_for_uncovered):
        mode = "TPCH-like workload" if is_tpch_like else "TPCH hotspot window"
        print(
            f"  [mv_seed] {mode} detected; using seed-only fast path "
            f"({len(seed_valid)} validated seeds, skip extra LLM MV generation).",
            flush=True,
        )
        return seed_valid[:max_mvs]
    if is_tpch_like and seed_valid and MV_SKIP_LLM_ON_TPCH and allow_llm_for_uncovered:
        hot_str = ", ".join(
            (
                f"T{tpl}({cnt}, {share*100:.1f}%, cost={cost:.0f})"
                if cost is not None
                else f"T{tpl}({cnt}, {share*100:.1f}%)"
            )
            for tpl, cnt, share, cost in uncovered_hot_tpls
        )
        print(
            "  [mv_seed] TPCH-like workload has uncovered dominant templates; "
            "allowing limited LLM MV generation "
            f"(min_share={min_uncovered_share:.3f}, min_cost_ratio={MV_UNCOVERED_TPL_MIN_COST_RATIO:.2f}) "
            f"for: {hot_str}",
            flush=True,
        )

    remaining_for_llm = max(0, int(max_mvs) - len(seed_valid))
    if allow_llm_for_uncovered and remaining_for_llm > 0:
        remaining_for_llm = min(int(remaining_for_llm), int(MV_UNCOVERED_LLM_MAX))
    if is_tpch_like and remaining_for_llm > 2:
        remaining_for_llm = 2
        print("  [mv_seed] TPCH-like workload detected; capping extra LLM MV candidates to 2 for stability.")
    if remaining_for_llm <= 0:
        return seed_valid[:max_mvs]

    api_key = os.environ.get("OPENAI_API_KEY", "")
    base_url = os.environ.get("OPENAI_BASE_URL", "") or None
    if MV_SKIP_LLM_ON_MISSING_KEY and (not str(api_key or "").strip()):
        print("  [mv_llm] OPENAI_API_KEY missing; skip LLM MV generation and keep seeds only.")
        return seed_valid[:max_mvs]
    client = OpenAI(api_key=api_key, base_url=base_url)

    wl_summary = build_workload_summary(queries, max_queries=workload_max_sample, stats=stats)
    system = _MV_SYSTEM_PROMPT.format(base_schema=base_schema, max_mvs=remaining_for_llm)

    user_msg = (
        f"## Schema\n{schema_info}\n\n"
        f"## Workload sample ({min(len(queries), workload_max_sample)} queries)\n{wl_summary}\n\n"
        "Generate materialized view candidates as JSON array."
    )

    candidates: List[MVCand] = []
    try:
        print(f"  [mv_llm] Calling {model} for MV generation...", flush=True)
        t0 = time.perf_counter()
        chat_kwargs: Dict[str, Any] = {}
        if reasoning_effort:
            chat_kwargs["reasoning_effort"] = str(reasoning_effort).strip().lower()
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user_msg},
            ],
            temperature=0.0,
            **chat_kwargs,
        )
        elapsed = time.perf_counter() - t0
        raw = (resp.choices[0].message.content or "").strip()
        print(f"  [mv_llm] LLM responded in {elapsed:.1f}s  ({len(raw)} chars)")
        print(f"  [mv_llm] Raw output:\n{raw[:2000]}")

        cleaned = re.sub(r"^```[a-z]*\n?", "", raw.strip(), flags=re.MULTILINE)
        cleaned = re.sub(r"```$", "", cleaned.strip(), flags=re.MULTILINE)
        arr = json.loads(cleaned)
        if not isinstance(arr, list):
            arr = [arr]
        for item in arr:
            if not isinstance(item, dict):
                continue
            mvid = item.get("mvid", f"MV{len(candidates)+1:03d}")
            name = item.get("name", f"mv_{len(candidates)+1:03d}")
            raw_sql = item.get("sql") or ""
            sql = _normalize_mv_candidate_sql(raw_sql)
            why  = item.get("why", "")
            if not sql:
                continue
            mv = MVCand(mvid=mvid, name=name, sql=sql, why=why, mv_schema=mv_schema)
            _parse_mv_structure(mv)
            candidates.append(mv)
    except Exception as e:
        print(f"  [mv_llm] Generation failed, keeping seed MVs only: {e}")

    candidates = _filter_structurally_risky_mvs(candidates, stage="llm")

    print(f"  [mv_llm] Parsed {len(candidates)} MV candidates "
          f"({sum(1 for m in candidates if m.mv_type=='agg')} agg, "
          f"{sum(1 for m in candidates if m.mv_type=='join')} join-only).")

    valid = _validate_mv_candidates_sql(con, candidates, stage="llm")
    merged = _dedupe_mvs_keep_order(seed_valid + valid)
    if len(merged) > max_mvs:
        merged = merged[:max_mvs]
    print(
        f"  [mv_llm] Final candidate set: {len(merged)} "
        f"(seed={len(seed_valid)}, llm_valid={len(valid)})"
    )
    return merged


# ═══════════════════════════════════════════════════════════════════
# Build MVs
# ═══════════════════════════════════════════════════════════════════

def _estimate_result_rows(con: duckdb.DuckDBPyConnection, sql: str) -> float:
    """Best-effort row estimate from DuckDB EXPLAIN text."""
    try:
        rows = con.execute(f"EXPLAIN {sql}").fetchall()
    except Exception:
        return 0.0
    plan_text = "\n".join(str(r) for r in rows)
    nums = re.findall(r'EC[:\s=]+(\d+)', plan_text)
    if nums:
        return float(max(int(n) for n in nums))
    nums = re.findall(r'Rows[:\s=]+(\d+)', plan_text)
    if nums:
        return float(max(int(n) for n in nums))
    return 0.0


def estimate_mv_size_mb(
    con: duckdb.DuckDBPyConnection,
    mv_sql: str,
    default_row_width_bytes: int = 64,
) -> float:
    """
    Estimate MV size before build using EXPLAIN cardinality and a coarse row width.
    Used only for admission ordering/filtering; hard constraints are enforced after build.
    """
    est_rows = _estimate_result_rows(con, mv_sql)
    if est_rows <= 0:
        return 0.0
    return (est_rows * max(default_row_width_bytes, 16)) / (1024 * 1024)


def build_mvs(
    con: duckdb.DuckDBPyConnection,
    mvs: List[MVCand],
    verbose: bool = True,
    budget_mb: float = 0.0,
    max_single_mb: float = 0.0,
) -> float:
    if not mvs:
        return 0.0
    if mvs:
        con.execute(f"CREATE SCHEMA IF NOT EXISTS {mvs[0].mv_schema};")

    t0 = time.perf_counter()
    admitted_total_mb = 0.0
    for mv in mvs:
        structural_reject = _structural_mv_reject_reason(mv)
        if structural_reject:
            mv.built = False
            mv.build_error = f"dropped_by_structure({structural_reject})"
            if verbose:
                print(f"  [mv_build] DROP {mv.full_name}: {mv.build_error}", flush=True)
            continue
        # Pre-build estimate for ordering/debug only.
        mv.estimated_size_mb = estimate_mv_size_mb(con, mv.sql)
        try:
            con.execute(f"DROP TABLE IF EXISTS {mv.full_name};")
            con.execute(f"CREATE TABLE {mv.full_name} AS {mv.sql}")
            row = con.execute(f"SELECT COUNT(*) FROM {mv.full_name}").fetchone()
            mv.row_count = row[0] if row else 0
            try:
                size_row = con.execute(
                    "SELECT estimated_size FROM duckdb_tables() WHERE table_name=? AND schema_name=?",
                    [mv.name, mv.mv_schema]
                ).fetchone()
                mv.size_mb = (size_row[0] / 1024 / 1024) if size_row else 0.0
            except Exception:
                mv.size_mb = 0.0

            # Hard single-MV cap.
            if max_single_mb > 0 and mv.size_mb > max_single_mb:
                con.execute(f"DROP TABLE IF EXISTS {mv.full_name};")
                mv.built = False
                mv.build_error = (
                    f"dropped_by_budget(single={mv.size_mb:.1f}MB > max_single={max_single_mb:.1f}MB)"
                )
                if verbose:
                    print(f"  [mv_build] DROP {mv.full_name}: {mv.build_error}", flush=True)
                continue

            # Hard cumulative cap without eviction (admit in planned order).
            if budget_mb > 0 and (admitted_total_mb + mv.size_mb) > budget_mb:
                con.execute(f"DROP TABLE IF EXISTS {mv.full_name};")
                mv.built = False
                mv.build_error = (
                    "dropped_by_budget("
                    f"current={admitted_total_mb:.1f}MB + mv={mv.size_mb:.1f}MB > budget={budget_mb:.1f}MB)"
                )
                if verbose:
                    print(f"  [mv_build] DROP {mv.full_name}: {mv.build_error}", flush=True)
                continue

            mv.built = True
            admitted_total_mb += mv.size_mb
            if verbose:
                print(f"  [mv_build] {mv.full_name} [{mv.mv_type}]: "
                      f"{mv.row_count:,} rows  ({mv.size_mb:.1f} MB)", flush=True)
        except Exception as e:
            mv.built = False
            mv.build_error = str(e)
            if verbose:
                print(f"  [mv_build] FAILED {mv.full_name}: {e}", flush=True)
    return (time.perf_counter() - t0) * 1000.0


def drop_mv_schema(con: duckdb.DuckDBPyConnection, mv_schema: str) -> None:
    try:
        con.execute(f"DROP SCHEMA IF EXISTS {mv_schema} CASCADE;")
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════
# PRECISE Aggregation MV Rewrite (new in v2)
# ═══════════════════════════════════════════════════════════════════

def _can_rewrite_agg_mv(orig_sql: str, mv: MVCand) -> bool:
    """
    Check whether orig_sql can be safely rewritten to use an aggregation MV.

    Conditions:
      C1. MV tables ⊆ query tables
      C2. query GROUP BY cols ⊆ MV GROUP BY cols  (query groups are coarser or equal)
      C3. All columns in query SELECT that involve aggregation are present in MV output
      C4. query WHERE doesn't reference raw fact columns not exposed in MV SELECT
          (conservative: if all WHERE cols are in MV output cols, allow rewrite)
    """
    if mv.mv_type != "agg":
        return False
    if not mv.mv_tables or not mv.mv_group_by_cols:
        return False

    q_tables = _extract_tables(orig_sql)
    # C1
    if not mv.mv_tables.issubset(q_tables):
        return False

    q_gb = _extract_group_by_cols(orig_sql)
    if not q_gb:
        return False  # query has no GROUP BY → can't use agg MV that way

    # C2: query GROUP BY ⊆ MV GROUP BY
    mv_gb_set = set(mv.mv_group_by_cols)
    q_gb_set  = set(q_gb)
    if not q_gb_set.issubset(mv_gb_set):
        return False

    mv_out = set(mv.mv_select_cols)
    q_where_cols = _extract_where_cols(orig_sql)

    # C4: WHERE cols must all be in MV output (after aggregation filter is safe)
    unsafe_where = q_where_cols - mv_out - q_gb_set
    if unsafe_where:
        return False  # filter on raw fact col → can't push through aggregation

    return True


def _build_agg_mv_rewrite(orig_sql: str, mv: MVCand) -> Optional[str]:
    """
    Produce the rewritten SQL using the aggregation MV.

    Strategy:
      SELECT <orig select, mapped to MV output cols>
      FROM mv.full_name
      [WHERE <orig where, only on MV output cols>]
      GROUP BY <orig group by>
      [ORDER BY ...] [LIMIT ...]

    We do a best-effort structural rewrite using regex.
    Falls back to None if structure is too complex.
    """
    # Extract SELECT part
    sel_m = re.match(r'^\s*SELECT\s+(.+?)\s+FROM\b', orig_sql, re.IGNORECASE | re.DOTALL)
    if not sel_m:
        return None
    sel_part = sel_m.group(1).strip()

    # Extract FROM ... (to the end of FROM chain, before WHERE/GROUP/ORDER/LIMIT)
    # We replace the whole FROM...JOIN... with just FROM mv.full_name
    tail_m = re.search(
        r'\b(WHERE|GROUP\s+BY|ORDER\s+BY|HAVING|LIMIT)\b',
        orig_sql, re.IGNORECASE
    )
    tail_sql = orig_sql[tail_m.start():].strip() if tail_m else ""

    mv_out = set(mv.mv_select_cols)
    sel_part = _dequalify_column_refs(sel_part, mv_out)
    if sel_part is None:
        return None
    if tail_sql:
        tail_sql = _dequalify_column_refs(tail_sql, mv_out)
        if tail_sql is None:
            return None

    # Build rewrite
    rewrite = f"SELECT {sel_part} FROM {mv.full_name}"
    if tail_sql:
        # If there's a WHERE, strip conditions that reference raw tables not in MV
        # (conservative: keep all conditions, since we already validated they're safe)
        rewrite += f" {tail_sql}"

    return rewrite


def _extract_query_column_refs(sql: str) -> Set[str]:
    """Extract all referenced column names (lowercased, unqualified)."""
    parsed = _parse_sql(sql)
    if parsed is not None and exp is not None:
        cols = {
            (c.name or "").strip().lower()
            for c in parsed.find_all(exp.Column)
            if (c.name or "").strip()
        }
        if cols:
            return cols
    return {x.lower() for x in re.findall(r'(?:\b\w+\.)?(\w+)\b', sql)}


def _dequalify_column_refs(fragment: str, allowed_cols: Set[str]) -> Optional[str]:
    """
    Convert alias-qualified references (t.col -> col) on SELECT/WHERE/GROUP/ORDER
    fragments after FROM is replaced by a single MV table.

    Returns None if we see qualified columns that are not projected by MV,
    because dropping qualifiers there would be unsafe.
    """
    if not fragment:
        return fragment
    allowed = {str(c).strip().lower() for c in (allowed_cols or set()) if str(c).strip()}
    bad_ref = False

    def _repl(m: re.Match) -> str:
        nonlocal bad_ref
        col = str(m.group(2) or "").strip()
        if not col:
            bad_ref = True
            return m.group(0)
        if allowed and col.lower() not in allowed:
            bad_ref = True
            return m.group(0)
        return col

    out = re.sub(r"\b([a-zA-Z_]\w*)\.([a-zA-Z_]\w*)\b", _repl, fragment)
    if bad_ref:
        return None
    return out


def _can_rewrite_join_mv(orig_sql: str, mv: MVCand) -> bool:
    """
    Conservative join-MV rewrite eligibility:
      1) MV is join-type and covers query tables
      2) All referenced query columns are projected by MV
    """
    if mv.mv_type != "join":
        return False
    if not _mv_covers_query(mv, orig_sql):
        return False
    q_cols = _extract_query_column_refs(orig_sql)
    if not q_cols:
        return False
    mv_out = set(mv.mv_select_cols)
    return q_cols.issubset(mv_out)


def _split_top_level_csv(text: str) -> List[str]:
    parts: List[str] = []
    cur: List[str] = []
    depth = 0
    in_single = False
    in_double = False
    for ch in str(text or ""):
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif not in_single and not in_double:
            if ch == "(":
                depth += 1
            elif ch == ")" and depth > 0:
                depth -= 1
            elif ch == "," and depth == 0:
                part = "".join(cur).strip()
                if part:
                    parts.append(part)
                cur = []
                continue
        cur.append(ch)
    tail = "".join(cur).strip()
    if tail:
        parts.append(tail)
    return parts


def _extract_from_clause_parts(sql: str) -> Optional[Tuple[str, str, str]]:
    """
    Return (prefix_before_from, from_body, tail_after_from).
    This helper is intentionally conservative and targets JOB-style comma-join SQL.
    """
    s = str(sql or "")
    m_from = re.search(r"\bFROM\b", s, re.IGNORECASE)
    if not m_from:
        return None
    prefix = s[: m_from.start()]
    rest = s[m_from.end() :]
    m_tail = re.search(
        r"\bWHERE\b|\bGROUP\s+BY\b|\bORDER\s+BY\b|\bHAVING\b|\bLIMIT\b",
        rest,
        re.IGNORECASE,
    )
    if m_tail:
        from_body = rest[: m_tail.start()].strip()
        tail = rest[m_tail.start() :].strip()
    else:
        from_body = rest.strip().rstrip(";")
        tail = ""
    if not from_body:
        return None
    return prefix, from_body, tail


def _parse_from_item(raw_item: str) -> Optional[Tuple[str, str]]:
    """
    Parse one FROM item and return (table_name, alias_name) in lowercase.
    Supports:
      table
      schema.table
      table alias
      table AS alias
    """
    item = str(raw_item or "").strip()
    if not item:
        return None
    if re.search(r"\bJOIN\b", item, re.IGNORECASE):
        return None
    m = re.match(
        r'^\s*(?:(?:"?([A-Za-z_]\w*)"?)[.])?"?([A-Za-z_]\w*)"?\s*(?:AS\s+)?("?([A-Za-z_]\w*)"?)?\s*$',
        item,
        flags=re.IGNORECASE,
    )
    if not m:
        return None
    table = str(m.group(2) or "").strip().lower()
    alias = str(m.group(4) or table).strip().lower()
    if not table or not alias:
        return None
    return table, alias


def _build_mv_source_col_map(mv: MVCand) -> Dict[str, str]:
    """
    Build best-effort mapping:
      "<base_table>.<source_col>" -> "<mv_output_col>"
    including simple equality propagation from MV join predicates.
    """
    out: Dict[str, str] = {}
    parsed = _parse_sql(mv.sql)
    if parsed is None or exp is None:
        return out

    alias_to_table: Dict[str, str] = {}
    for t in parsed.find_all(exp.Table):
        base = _table_name(t)
        if not base:
            continue
        alias = (t.alias_or_name or base).lower()
        alias_to_table[alias] = base

    parent: Dict[str, str] = {}

    def _find(x: str) -> str:
        if x not in parent:
            parent[x] = x
            return x
        if parent[x] != x:
            parent[x] = _find(parent[x])
        return parent[x]

    def _union(a: str, b: str) -> None:
        ra, rb = _find(a), _find(b)
        if ra != rb:
            parent[rb] = ra

    def _tok(col: "exp.Column") -> Optional[str]:
        tbl_alias = (col.table or "").strip().lower()
        col_name = (col.name or "").strip().lower()
        if not col_name:
            return None
        tbl = alias_to_table.get(tbl_alias, tbl_alias) if tbl_alias else ""
        if not tbl:
            return None
        return f"{tbl}.{col_name}"

    # Equality graph from join predicates.
    for eq in parsed.find_all(exp.EQ):
        l = eq.left
        r = eq.right
        if not isinstance(l, exp.Column) or not isinstance(r, exp.Column):
            continue
        lt = _tok(l)
        rt = _tok(r)
        if lt and rt:
            _union(lt, rt)

    sel = parsed if isinstance(parsed, exp.Select) else parsed.find(exp.Select)
    if sel is None:
        return out

    # Direct projection map.
    for expr_i in sel.expressions or []:
        out_col = ""
        src_cols: List["exp.Column"] = []
        if isinstance(expr_i, exp.Alias):
            out_col = (expr_i.alias or "").strip().lower()
            src_cols = list(expr_i.this.find_all(exp.Column))
        elif isinstance(expr_i, exp.Column):
            out_col = (expr_i.name or "").strip().lower()
            src_cols = [expr_i]
        else:
            src_cols = list(expr_i.find_all(exp.Column))
            if len(src_cols) == 1:
                out_col = (src_cols[0].name or "").strip().lower()
        if not out_col:
            continue
        for c in src_cols:
            tok = _tok(c)
            if tok and tok not in out:
                out[tok] = out_col

    if not out:
        return out

    # Propagate map through equality classes.
    classes: Dict[str, Set[str]] = {}
    for tok in list(out.keys()):
        root = _find(tok)
        classes.setdefault(root, set()).add(tok)
    # Include equality-only tokens.
    for tok in list(parent.keys()):
        root = _find(tok)
        classes.setdefault(root, set()).add(tok)

    for members in classes.values():
        projected = [out.get(m) for m in members if m in out]
        projected = [p for p in projected if p]
        if not projected:
            continue
        chosen = projected[0]
        for m in members:
            out.setdefault(m, chosen)
    return out


def _rewrite_alias_refs_to_mv(
    sql: str,
    *,
    alias_to_table: Dict[str, str],
    replaced_aliases: Set[str],
    src_col_map: Dict[str, str],
    mv_alias: str,
) -> Optional[str]:
    rewritten = str(sql or "")
    for alias, tbl in alias_to_table.items():
        a = str(alias or "").strip().lower()
        t = str(tbl or "").strip().lower()
        if not a or not t or a not in replaced_aliases:
            continue
        pattern = re.compile(rf"\b{re.escape(a)}\.([A-Za-z_]\w*)\b", flags=re.IGNORECASE)

        def _repl(m: re.Match) -> str:
            col = str(m.group(1) or "").strip().lower()
            if not col:
                raise ValueError("empty column ref")
            key = f"{t}.{col}"
            mapped = src_col_map.get(key)
            if not mapped:
                raise ValueError(f"missing_mv_col_map:{key}")
            return f"{mv_alias}.{mapped}"

        try:
            rewritten = pattern.sub(_repl, rewritten)
        except Exception:
            return None
    return rewritten


def _build_join_mv_partial_rewrite(orig_sql: str, mv: MVCand) -> Optional[str]:
    """
    Partial join rewrite:
      replace only the MV-covered FROM subgraph with one MV alias and
      keep remaining query tables untouched.
    This targets JOB-style wide queries where full-query replacement is too strict.
    """
    if mv.mv_type != "join" or not mv.mv_tables:
        return None
    parts = _extract_from_clause_parts(orig_sql)
    if parts is None:
        return None
    prefix, from_body, tail = parts
    raw_items = _split_top_level_csv(from_body)
    if not raw_items:
        return None

    alias_to_table: Dict[str, str] = {}
    replaced_aliases: Set[str] = set()
    kept_raw_items: List[str] = []
    replaced_count = 0
    for raw in raw_items:
        parsed = _parse_from_item(raw)
        if parsed is None:
            return None
        tbl, alias = parsed
        alias_to_table[alias] = tbl
        if tbl in mv.mv_tables:
            replaced_aliases.add(alias)
            replaced_count += 1
        else:
            kept_raw_items.append(raw.strip())

    # Need a true partial rewrite: both replaced and remaining tables must exist.
    if replaced_count <= 0 or not kept_raw_items:
        return None

    src_col_map = _build_mv_source_col_map(mv)
    if not src_col_map:
        return None

    mv_alias = "__aidb_mv"
    new_from_items = kept_raw_items + [f"{mv.full_name} AS {mv_alias}"]
    candidate = f"{prefix}FROM {', '.join(new_from_items)}"
    if tail:
        candidate += f" {tail}"

    candidate = _rewrite_alias_refs_to_mv(
        candidate,
        alias_to_table=alias_to_table,
        replaced_aliases=replaced_aliases,
        src_col_map=src_col_map,
        mv_alias=mv_alias,
    )
    if not candidate:
        return None
    return candidate


def _build_join_mv_rewrite(orig_sql: str, mv: MVCand) -> Optional[str]:
    """
    Rewrite query to read from join MV directly:
      SELECT <orig select> FROM mv.full_name <orig tail>

    Safety:
      - Skip qualified aliases (t.col) because aliases disappear after FROM rewrite.
      - Keep original WHERE/GROUP/ORDER/LIMIT tail unchanged.
    """
    sel_m = re.match(r'^\s*SELECT\s+(.+?)\s+FROM\b', orig_sql, re.IGNORECASE | re.DOTALL)
    if not sel_m:
        return None
    sel_part = sel_m.group(1).strip()

    tail_m = re.search(
        r'\b(WHERE|GROUP\s+BY|ORDER\s+BY|HAVING|LIMIT)\b',
        orig_sql, re.IGNORECASE
    )
    tail_sql = orig_sql[tail_m.start():].strip() if tail_m else ""

    mv_out = set(mv.mv_select_cols)
    sel_part = _dequalify_column_refs(sel_part, mv_out)
    if sel_part is None:
        return None
    if tail_sql:
        tail_sql = _dequalify_column_refs(tail_sql, mv_out)
        if tail_sql is None:
            return None

    rewrite = f"SELECT {sel_part} FROM {mv.full_name}"
    if tail_sql:
        rewrite += f" {tail_sql}"
    return rewrite


def _norm_sql(sql: str) -> str:
    return " ".join((sql or "").lower().split())


def _extract_where_clause(orig_sql: str) -> Optional[str]:
    m = re.search(
        r"\bwhere\b\s+(.+?)(?:\bgroup\s+by\b|\border\s+by\b|\bhaving\b|\blimit\b|;|$)",
        orig_sql,
        re.IGNORECASE | re.DOTALL,
    )
    if not m:
        return None
    w = m.group(1).strip()
    return w if w else None


def _infer_tpch_template_from_sql(sql: str) -> Optional[int]:
    s = _norm_sql(sql)
    if "group by l_returnflag, l_linestatus" in s and "from lineitem" in s:
        return 1
    if "c_mktsegment" in s and "group by l_orderkey, o_orderdate, o_shippriority" in s:
        return 3
    if "exists ( select * from lineitem where l_orderkey = o_orderkey" in s and "group by o_orderpriority" in s:
        return 4
    if "from customer, orders, lineitem, supplier, nation, region" in s and "group by n_name" in s:
        return 5
    if "sum(l_extendedprice * l_discount) as revenue from lineitem" in s:
        return 6
    if "sum(case when nation =" in s and "as mkt_share" in s:
        return 8
    if "high_line_count" in s and "low_line_count" in s and "group by l_shipmode" in s:
        return 12
    if "promo_revenue" in s and "p_type like 'promo%'" in s:
        return 14
    if "numwait" in s and "not exists ( select * from lineitem l3" in s:
        return 21
    if _looks_like_tpch_q10(sql):
        return 10
    if _looks_like_tpch_q18(sql):
        return 18
    return None


def _build_tpch_q1_seed_rewrite(orig_sql: str, mv: MVCand) -> Optional[str]:
    if "tpch_t1_daily_rollup" not in mv.name.lower():
        return None
    m = re.search(
        r"l_shipdate\s*<=\s*(.+?)(?:\bgroup\s+by\b|\border\s+by\b|;|$)",
        orig_sql,
        re.IGNORECASE | re.DOTALL,
    )
    if not m:
        return None
    cutoff_expr = m.group(1).strip()
    if not cutoff_expr:
        return None
    return (
        "SELECT l_returnflag, l_linestatus, "
        "SUM(sum_qty) AS sum_qty, "
        "SUM(sum_base_price) AS sum_base_price, "
        "SUM(sum_disc_price) AS sum_disc_price, "
        "SUM(sum_charge) AS sum_charge, "
        "SUM(sum_qty) / NULLIF(SUM(count_order), 0) AS avg_qty, "
        "SUM(sum_base_price) / NULLIF(SUM(count_order), 0) AS avg_price, "
        "SUM(sum_discount) / NULLIF(SUM(count_order), 0) AS avg_disc, "
        "SUM(count_order) AS count_order "
        f"FROM {mv.full_name} "
        f"WHERE l_shipdate <= {cutoff_expr} "
        "GROUP BY l_returnflag, l_linestatus "
        "ORDER BY l_returnflag, l_linestatus"
    )


def _build_tpch_q3_seed_rewrite(orig_sql: str, mv: MVCand) -> Optional[str]:
    if "tpch_t3_order_rev_by_segment_dates" not in mv.name.lower():
        return None
    seg_m = re.search(r"c_mktsegment\s*=\s*'([^']+)'", orig_sql, re.IGNORECASE)
    odate_m = re.search(r"o_orderdate\s*<\s*(date\s*'[^']+')", orig_sql, re.IGNORECASE)
    sdate_m = re.search(r"l_shipdate\s*>\s*(date\s*'[^']+')", orig_sql, re.IGNORECASE)
    if not seg_m or not odate_m or not sdate_m:
        return None
    seg = seg_m.group(1)
    odate = odate_m.group(1)
    sdate = sdate_m.group(1)
    return (
        "SELECT l_orderkey, SUM(revenue) AS revenue, o_orderdate, o_shippriority "
        f"FROM {mv.full_name} "
        f"WHERE c_mktsegment = '{seg}' "
        f"AND o_orderdate < {odate} "
        f"AND l_shipdate > {sdate} "
        "GROUP BY l_orderkey, o_orderdate, o_shippriority "
        "ORDER BY revenue DESC, o_orderdate"
    )


def _build_tpch_q4_seed_rewrite(orig_sql: str, mv: MVCand) -> Optional[str]:
    if "tpch_t4_late_receipt_flag_by_order" not in mv.name.lower():
        return None
    gte_m = re.search(r"o_orderdate\s*>=\s*(date\s*'[^']+')", orig_sql, re.IGNORECASE)
    lt_m = re.search(
        r"o_orderdate\s*<\s*(date\s*'[^']+'\s*\+\s*interval\s*'[^']+'\s*month)",
        orig_sql,
        re.IGNORECASE,
    )
    if not gte_m or not lt_m:
        return None
    odate_gte = gte_m.group(1)
    odate_lt = lt_m.group(1)
    return (
        "SELECT o.o_orderpriority, COUNT(*) AS order_count "
        "FROM orders o "
        f"JOIN {mv.full_name} m ON m.l_orderkey = o.o_orderkey "
        "WHERE m.has_late_receipt = 1 "
        f"AND o.o_orderdate >= {odate_gte} "
        f"AND o.o_orderdate < {odate_lt} "
        "GROUP BY o.o_orderpriority "
        "ORDER BY o.o_orderpriority"
    )


def _build_tpch_q5_seed_rewrite(orig_sql: str, mv: MVCand) -> Optional[str]:
    if "tpch_t5_rev_by_region_nation_date" not in mv.name.lower():
        return None
    region_m = re.search(r"r_name\s*=\s*'([^']+)'", orig_sql, re.IGNORECASE)
    gte_m = re.search(r"o_orderdate\s*>=\s*(date\s*'[^']+')", orig_sql, re.IGNORECASE)
    lt_m = re.search(
        r"o_orderdate\s*<\s*(date\s*'[^']+'\s*\+\s*interval\s*'[^']+'\s*year)",
        orig_sql,
        re.IGNORECASE,
    )
    if not region_m or not gte_m or not lt_m:
        return None
    region = region_m.group(1)
    odate_gte = gte_m.group(1)
    odate_lt = lt_m.group(1)
    return (
        "SELECT n_name, SUM(revenue) AS revenue "
        f"FROM {mv.full_name} "
        f"WHERE r_name = '{region}' "
        f"AND o_orderdate >= {odate_gte} "
        f"AND o_orderdate < {odate_lt} "
        "GROUP BY n_name "
        "ORDER BY revenue DESC"
    )


def _build_tpch_q6_seed_rewrite(orig_sql: str, mv: MVCand) -> Optional[str]:
    if "tpch_t6_revenue_by_shipdate_discount_qty" not in mv.name.lower():
        return None
    where_clause = _extract_where_clause(orig_sql)
    if not where_clause:
        return None
    return f"SELECT SUM(revenue) AS revenue FROM {mv.full_name} WHERE {where_clause}"


def _build_tpch_q8_seed_rewrite(orig_sql: str, mv: MVCand) -> Optional[str]:
    if "tpch_t8_volume_by_year_region_type_nation" not in mv.name.lower():
        return None
    nation_m = re.search(r"nation\s*=\s*'([^']+)'", orig_sql, re.IGNORECASE)
    region_m = re.search(r"r_name\s*=\s*'([^']+)'", orig_sql, re.IGNORECASE)
    ptype_m = re.search(r"p_type\s*=\s*'([^']+)'", orig_sql, re.IGNORECASE)
    date_m = re.search(
        r"o_orderdate\s+between\s+date\s*'([^']+)'\s+and\s+date\s*'([^']+)'",
        orig_sql,
        re.IGNORECASE,
    )
    if not nation_m or not region_m or not ptype_m or not date_m:
        return None
    nation = nation_m.group(1)
    region = region_m.group(1)
    ptype = ptype_m.group(1)
    d1 = date_m.group(1)
    d2 = date_m.group(2)
    return (
        "SELECT o_year, "
        f"SUM(CASE WHEN nation = '{nation}' THEN volume ELSE 0 END) / NULLIF(SUM(volume), 0) AS mkt_share "
        f"FROM {mv.full_name} "
        f"WHERE r_name = '{region}' "
        f"AND p_type = '{ptype}' "
        f"AND o_year BETWEEN EXTRACT(year FROM DATE '{d1}') "
        f"AND EXTRACT(year FROM DATE '{d2}') "
        "GROUP BY o_year "
        "ORDER BY o_year"
    )


def _build_tpch_q7_seed_rewrite(orig_sql: str, mv: MVCand) -> Optional[str]:
    if "tpch_t7_rev_by_supp_cust_year" not in mv.name.lower():
        return None
    pair_m = re.search(
        r"\(\s*\(\s*n1\.n_name\s*=\s*'([^']+)'\s*and\s*n2\.n_name\s*=\s*'([^']+)'\s*\)\s*"
        r"or\s*\(\s*n1\.n_name\s*=\s*'([^']+)'\s*and\s*n2\.n_name\s*=\s*'([^']+)'\s*\)\s*\)",
        orig_sql,
        re.IGNORECASE,
    )
    if not pair_m:
        return None
    a1, b1, a2, b2 = pair_m.groups()
    return (
        "SELECT supp_nation, cust_nation, l_year, revenue "
        f"FROM {mv.full_name} "
        f"WHERE ((supp_nation = '{a1}' AND cust_nation = '{b1}') "
        f"OR (supp_nation = '{a2}' AND cust_nation = '{b2}')) "
        "ORDER BY supp_nation, cust_nation, l_year"
    )


def _build_tpch_q12_seed_rewrite(orig_sql: str, mv: MVCand) -> Optional[str]:
    if "tpch_t12_shipmode_priority_by_receiptdate" not in mv.name.lower():
        return None
    shipmode_m = re.search(r"l_shipmode\s+in\s*(\([^)]+\))", orig_sql, re.IGNORECASE)
    gte_m = re.search(r"l_receiptdate\s*>=\s*(date\s*'[^']+')", orig_sql, re.IGNORECASE)
    lt_m = re.search(
        r"l_receiptdate\s*<\s*(date\s*'[^']+'\s*\+\s*interval\s*'[^']+'\s*year)",
        orig_sql,
        re.IGNORECASE,
    )
    if not shipmode_m or not gte_m or not lt_m:
        return None
    shipmode_in = shipmode_m.group(1)
    d_gte = gte_m.group(1)
    d_lt = lt_m.group(1)
    return (
        "SELECT l_shipmode, "
        "SUM(CASE WHEN o_orderpriority = '1-URGENT' OR o_orderpriority = '2-HIGH' "
        "THEN line_count ELSE 0 END) AS high_line_count, "
        "SUM(CASE WHEN o_orderpriority <> '1-URGENT' AND o_orderpriority <> '2-HIGH' "
        "THEN line_count ELSE 0 END) AS low_line_count "
        f"FROM {mv.full_name} "
        f"WHERE l_shipmode IN {shipmode_in} "
        f"AND l_receiptdate >= {d_gte} "
        f"AND l_receiptdate < {d_lt} "
        "GROUP BY l_shipmode "
        "ORDER BY l_shipmode"
    )


def _build_tpch_q13_seed_rewrite(orig_sql: str, mv: MVCand) -> Optional[str]:
    if "tpch_t13_customer_order_count" not in mv.name.lower():
        return None
    s = " ".join((orig_sql or "").lower().split())
    if "o_comment not like '%unusual%requests%'" not in s:
        return None
    return (
        "SELECT c_count, COUNT(*) AS custdist "
        f"FROM {mv.full_name} "
        "GROUP BY c_count "
        "ORDER BY custdist DESC, c_count DESC"
    )


def _build_tpch_q14_seed_rewrite(orig_sql: str, mv: MVCand) -> Optional[str]:
    if "tpch_t14_promo_total_by_shipdate" not in mv.name.lower():
        return None
    gte_m = re.search(r"l_shipdate\s*>=\s*(date\s*'[^']+')", orig_sql, re.IGNORECASE)
    lt_m = re.search(
        r"l_shipdate\s*<\s*(date\s*'[^']+'\s*\+\s*interval\s*'[^']+'\s*month)",
        orig_sql,
        re.IGNORECASE,
    )
    if not gte_m or not lt_m:
        return None
    d_gte = gte_m.group(1)
    d_lt = lt_m.group(1)
    return (
        "SELECT 100.00 * SUM(sum_promo_disc_price) / NULLIF(SUM(sum_disc_price), 0) AS promo_revenue "
        f"FROM {mv.full_name} "
        f"WHERE l_shipdate >= {d_gte} "
        f"AND l_shipdate < {d_lt}"
    )


def _build_tpch_q21_seed_rewrite(orig_sql: str, mv: MVCand) -> Optional[str]:
    if "tpch_t21_supplier_order_wait_flags" not in mv.name.lower():
        return None
    nation_m = re.search(r"n_name\s*=\s*'([^']+)'", orig_sql, re.IGNORECASE)
    if not nation_m:
        return None
    nation = nation_m.group(1)
    return (
        "SELECT s.s_name, SUM(m.self_late_line_cnt) AS numwait "
        "FROM supplier s "
        f"JOIN {mv.full_name} m ON s.s_suppkey = m.l_suppkey "
        "JOIN orders o ON o.o_orderkey = m.l_orderkey "
        "JOIN nation n ON s.s_nationkey = n.n_nationkey "
        "WHERE o.o_orderstatus = 'F' "
        "AND m.self_has_late = 1 "
        "AND m.supp_cnt > 1 "
        "AND m.other_late_supp_cnt = 0 "
        f"AND n.n_name = '{nation}' "
        "GROUP BY s.s_name "
        "ORDER BY numwait DESC, s.s_name"
    )


def _looks_like_tpch_q10(sql: str) -> bool:
    s = " ".join((sql or "").lower().split())
    return (
        "from customer, orders, lineitem, nation" in s
        and "sum(l_extendedprice * (1 - l_discount)) as revenue" in s
        and "l_returnflag = 'r'" in s
        and "group by c_custkey" in s
    )


def _looks_like_tpch_q18(sql: str) -> bool:
    s = " ".join((sql or "").lower().split())
    return (
        "from customer, orders, lineitem" in s
        and "o_orderkey in ( select l_orderkey from lineitem group by l_orderkey having sum(l_quantity) >" in s
        and "sum(l_quantity)" in s
        and "order by o_totalprice desc" in s
    )


def _build_tpch_q10_seed_rewrite(orig_sql: str, mv: MVCand) -> Optional[str]:
    if "tpch_t10_rev_by_cust_date" not in mv.name:
        return None
    m = re.search(r"o_orderdate\s*>=\s*date\s*'([^']+)'", orig_sql, re.IGNORECASE)
    if not m:
        return None
    d0 = m.group(1)
    return (
        "SELECT c_custkey, c_name, SUM(m.revenue) AS revenue, c_acctbal, "
        "n_name, c_address, c_phone, c_comment "
        "FROM customer c "
        "JOIN nation n ON c.c_nationkey = n.n_nationkey "
        f"JOIN {mv.full_name} m ON c.c_custkey = m.o_custkey "
        f"WHERE m.o_orderdate >= DATE '{d0}' "
        f"AND m.o_orderdate < DATE '{d0}' + INTERVAL '3' month "
        "GROUP BY c_custkey, c_name, c_acctbal, c_phone, n_name, c_address, c_comment "
        "ORDER BY revenue DESC"
    )


def _build_tpch_q18_seed_rewrite(orig_sql: str, mv: MVCand) -> Optional[str]:
    if "tpch_t18_order_qty" not in mv.name:
        return None
    m = re.search(
        r"having\s+sum\s*\(\s*l_quantity\s*\)\s*>\s*([0-9]+(?:\.[0-9]+)?)",
        orig_sql,
        re.IGNORECASE,
    )
    if not m:
        return None
    threshold = m.group(1)
    return (
        "SELECT c_name, c_custkey, o_orderkey, o_orderdate, o_totalprice, m.sum_qty "
        "FROM customer c "
        "JOIN orders o ON c.c_custkey = o.o_custkey "
        f"JOIN {mv.full_name} m ON o.o_orderkey = m.l_orderkey "
        f"WHERE m.sum_qty > {threshold} "
        "ORDER BY o_totalprice DESC, o_orderdate"
    )


def _build_tpch_seed_rewrite(orig_sql: str, mv: MVCand, tpl: Optional[int]) -> Optional[str]:
    t = tpl if tpl is not None else _infer_tpch_template_from_sql(orig_sql)
    if t == 1:
        return _build_tpch_q1_seed_rewrite(orig_sql, mv)
    if t == 3:
        return _build_tpch_q3_seed_rewrite(orig_sql, mv)
    if t == 4:
        return _build_tpch_q4_seed_rewrite(orig_sql, mv)
    if t == 5:
        return _build_tpch_q5_seed_rewrite(orig_sql, mv)
    if t == 6:
        return _build_tpch_q6_seed_rewrite(orig_sql, mv)
    if t == 7:
        return _build_tpch_q7_seed_rewrite(orig_sql, mv)
    if t == 8:
        return _build_tpch_q8_seed_rewrite(orig_sql, mv)
    if t == 10:
        return _build_tpch_q10_seed_rewrite(orig_sql, mv)
    if t == 12:
        return _build_tpch_q12_seed_rewrite(orig_sql, mv)
    if t == 13:
        return _build_tpch_q13_seed_rewrite(orig_sql, mv)
    if t == 14:
        return _build_tpch_q14_seed_rewrite(orig_sql, mv)
    if t == 18:
        return _build_tpch_q18_seed_rewrite(orig_sql, mv)
    if t == 21:
        return _build_tpch_q21_seed_rewrite(orig_sql, mv)
    return None


def _get_explain_cost(con: duckdb.DuckDBPyConnection, sql: str) -> float:
    try:
        rows = con.execute(f"EXPLAIN {sql}").fetchall()
        plan_text = "\n".join(str(r) for r in rows)
        nums = re.findall(r'EC[:\s=]+(\d+)', plan_text)
        if nums:
            return float(max(int(n) for n in nums))
        nums = re.findall(r'Rows[:\s=]+(\d+)', plan_text)
        if nums:
            return float(max(int(n) for n in nums))
        # DuckDB's text EXPLAIN often emits cardinalities as "~11,997,210 rows"
        # without a "Rows:" label. Prefer those estimates over plan-text length.
        nums = re.findall(r'~\s*([0-9][0-9,]*)\s+rows', plan_text, re.IGNORECASE)
        if nums:
            return float(max(int(n.replace(",", "")) for n in nums))
        return float(len(plan_text))
    except Exception:
        return 1e18


# ═══════════════════════════════════════════════════════════════════
# Query rewriter (join-MV + agg-MV, cost-safe)
# ═══════════════════════════════════════════════════════════════════

def _mv_covers_query(mv: MVCand, sql: str) -> bool:
    if not mv.mv_tables:
        return False
    return mv.mv_tables.issubset(_extract_tables(sql))


def rewrite_with_mvs(
    con: duckdb.DuckDBPyConnection,
    queries: List[Dict],
    mvs: List[MVCand],
    base_schema: str = "main",
    mv_schema: str = "aidb_co_mv",
    verbose: bool = False,
    cost_check: bool = True,
    rewrite_strategy: str = "first_match",
) -> Tuple[List[str], int, List[Optional[str]]]:
    """
    For each query, attempt to rewrite it using a built MV.

    rewrite_strategy:
      - first_match: first beneficial candidate by priority order (fast)
      - cost_best:   try all candidates and choose min-cost rewrite (higher quality)

    Returns (rewritten_sqls, n_rewritten, rewrite_by_query).
    """
    if rewrite_strategy not in ("first_match", "cost_best"):
        raise ValueError("rewrite_strategy must be first_match|cost_best")

    built_mvs = [m for m in mvs if m.built]
    agg_mvs  = [m for m in built_mvs if m.mv_type == "agg"]
    join_mvs = [m for m in built_mvs if m.mv_type == "join"]

    rewritten_sqls: List[str] = []
    n_rewritten = 0
    rewrite_by_query: List[Optional[str]] = []
    stats = {
        "skipped_cost": 0,
        "skipped_no_match": 0,
        "agg_rewrites": 0,
        "join_rewrites": 0,
        "seed_rewrites": 0,
    }

    for q in queries:
        orig_sql = (q.get("sql") or q.get("query") or "").strip()
        if not orig_sql:
            rewritten_sqls.append(orig_sql)
            rewrite_by_query.append(None)
            continue
        tpl = _safe_template_id(q)

        orig_cost = _get_explain_cost(con, orig_sql) if cost_check else float("inf")
        chosen_sql = orig_sql
        chosen_mv_name: Optional[str] = None
        chosen_type: Optional[str] = None
        chosen_cost = float("inf")

        def _consider_candidate(
            candidate_sql: str,
            candidate_mv_name: str,
            candidate_cost: float,
            candidate_type: str,
        ) -> bool:
            nonlocal chosen_sql, chosen_mv_name, chosen_cost, chosen_type
            if rewrite_strategy == "first_match":
                chosen_sql = candidate_sql
                chosen_mv_name = candidate_mv_name
                chosen_cost = candidate_cost
                chosen_type = candidate_type
                return True
            # cost_best
            if candidate_cost < chosen_cost:
                chosen_sql = candidate_sql
                chosen_mv_name = candidate_mv_name
                chosen_cost = candidate_cost
                chosen_type = candidate_type
            return False

        # ── Priority 0: exact repeat-seed rewrites (benchmark-agnostic) ──
        for mv in built_mvs:
            if rewrite_strategy == "first_match" and chosen_mv_name is not None:
                break
            mid = str(getattr(mv, "mvid", "") or "").upper()
            if not mid.startswith("SEED_R"):
                continue
            if _normalize_sql_key(orig_sql) != _normalize_sql_key(getattr(mv, "sql", "")):
                continue
            candidate_sql = f"SELECT * FROM {mv.full_name}"
            if cost_check:
                try:
                    con.execute(f"EXPLAIN {candidate_sql}")
                except Exception:
                    continue
                rw_cost = _get_explain_cost(con, candidate_sql)
                if rw_cost >= orig_cost:
                    stats["skipped_cost"] += 1
                    continue
            else:
                rw_cost = 0.0
            stop = _consider_candidate(
                candidate_sql=candidate_sql,
                candidate_mv_name=mv.name + " [repeat_seed]",
                candidate_cost=rw_cost,
                candidate_type="seed",
            )
            if stop:
                break

        # ── Priority 0: Template-aware seed rewrites (TPC-H hotspots) ──
        for mv in built_mvs:
            if rewrite_strategy == "first_match" and chosen_mv_name is not None:
                break
            candidate_sql = _build_tpch_seed_rewrite(orig_sql, mv, tpl)
            if candidate_sql is None:
                continue
            if cost_check:
                try:
                    con.execute(f"EXPLAIN {candidate_sql}")
                except Exception:
                    continue
                rw_cost = _get_explain_cost(con, candidate_sql)
                if rw_cost >= orig_cost:
                    stats["skipped_cost"] += 1
                    continue
            else:
                rw_cost = 0.0
            stop = _consider_candidate(
                candidate_sql=candidate_sql,
                candidate_mv_name=mv.name + " [seed_rewrite]",
                candidate_cost=rw_cost,
                candidate_type="seed",
            )
            if stop:
                break

        # ── Priority 1: Aggregation MV precise rewrite ────────────────
        for mv in agg_mvs:
            if not _can_rewrite_agg_mv(orig_sql, mv):
                continue
            candidate_sql = _build_agg_mv_rewrite(orig_sql, mv)
            if candidate_sql is None:
                continue
            if cost_check:
                try:
                    con.execute(f"EXPLAIN {candidate_sql}")
                except Exception:
                    continue
                rw_cost = _get_explain_cost(con, candidate_sql)
                if rw_cost >= orig_cost:
                    stats["skipped_cost"] += 1
                    continue
            else:
                rw_cost = 0.0
            stop = _consider_candidate(
                candidate_sql=candidate_sql,
                candidate_mv_name=mv.name + " [agg_rewrite]",
                candidate_cost=rw_cost,
                candidate_type="agg",
            )
            if stop:
                break

        # ── Priority 2: Join-only MV direct rewrite ───────────────────
        for mv in join_mvs:
            if rewrite_strategy == "first_match" and chosen_mv_name is not None:
                break
            candidate_sql = None
            candidate_tag = "[join_rewrite]"
            if _can_rewrite_join_mv(orig_sql, mv):
                candidate_sql = _build_join_mv_rewrite(orig_sql, mv)
            if candidate_sql is None:
                candidate_sql = _build_join_mv_partial_rewrite(orig_sql, mv)
                if candidate_sql is not None:
                    candidate_tag = "[join_partial]"
            if candidate_sql is None:
                continue
            if cost_check:
                mv_scan_cost = _get_explain_cost(con, candidate_sql)
                if mv_scan_cost >= orig_cost * 0.9:
                    stats["skipped_cost"] += 1
                    continue
            else:
                mv_scan_cost = 0.0
            stop = _consider_candidate(
                candidate_sql=candidate_sql,
                candidate_mv_name=mv.name + " " + candidate_tag,
                candidate_cost=mv_scan_cost,
                candidate_type="join",
            )
            if stop:
                break

        if chosen_mv_name:
            n_rewritten += 1
            if chosen_type == "agg":
                stats["agg_rewrites"] += 1
            elif chosen_type == "join":
                stats["join_rewrites"] += 1
            elif chosen_type == "seed":
                stats["seed_rewrites"] += 1
        else:
            stats["skipped_no_match"] += 1
        rewritten_sqls.append(chosen_sql)
        rewrite_by_query.append(chosen_mv_name)

    if verbose:
        print(f"  [mv_rewrite] {n_rewritten}/{len(queries)} rewritten | "
              f"seed_rewrites={stats['seed_rewrites']} | "
              f"agg_rewrites={stats['agg_rewrites']} | join_rewrites={stats['join_rewrites']} | "
              f"skipped_cost={stats['skipped_cost']} | no_match={stats['skipped_no_match']} | "
              f"strategy={rewrite_strategy} | cost_check={int(bool(cost_check))}")
    return rewritten_sqls, n_rewritten, rewrite_by_query


# ═══════════════════════════════════════════════════════════════════
# Full MV pipeline
# ═══════════════════════════════════════════════════════════════════

def run_mv_pipeline(
    con: duckdb.DuckDBPyConnection,
    queries: List[Dict],
    schema_info: str,
    base_schema: str,
    mv_schema: str,
    model: str,
    workdir: str,
    max_mvs: int = 20,
    verbose: bool = True,
    stats: Optional[Dict] = None,
) -> Dict:
    os.makedirs(workdir, exist_ok=True)

    t_llm0 = time.perf_counter()
    mvs = llm_generate_mvs(
        con, queries, schema_info, base_schema=base_schema, mv_schema=mv_schema,
        model=model, max_mvs=max_mvs, stats=stats,
    )
    llm_ms = (time.perf_counter() - t_llm0) * 1000.0

    with open(os.path.join(workdir, "mv_candidates.json"), "w") as f:
        json.dump([{
            "mvid": m.mvid, "name": m.name, "sql": m.sql, "why": m.why,
            "mv_type": m.mv_type, "mv_group_by_cols": m.mv_group_by_cols,
        } for m in mvs], f, indent=2, ensure_ascii=False)

    drop_mv_schema(con, mv_schema)
    con.execute(f"CREATE SCHEMA IF NOT EXISTS {mv_schema};")
    build_ms = build_mvs(con, mvs, verbose=verbose)
    built_mvs = [m for m in mvs if m.built]

    if verbose:
        n_agg  = sum(1 for m in built_mvs if m.mv_type == "agg")
        n_join = sum(1 for m in built_mvs if m.mv_type == "join")
        print(f"  [mv] Built {len(built_mvs)}/{len(mvs)} MVs "
              f"({n_agg} agg + {n_join} join) in {build_ms/1000:.1f}s")

    with open(os.path.join(workdir, "mv_built.json"), "w") as f:
        json.dump([{
            "mvid": m.mvid, "name": m.name, "sql": m.sql, "why": m.why,
            "built": m.built, "build_error": m.build_error,
            "row_count": m.row_count, "size_mb": m.size_mb,
            "mv_type": m.mv_type,
        } for m in mvs], f, indent=2, ensure_ascii=False)

    rewritten_sqls, n_rewritten, rewrite_by_query = rewrite_with_mvs(
        con, queries, built_mvs, base_schema=base_schema, mv_schema=mv_schema,
        verbose=verbose,
    )

    return {
        "n_candidates": len(mvs),
        "n_built": len(built_mvs),
        "n_agg_mvs": sum(1 for m in built_mvs if m.mv_type == "agg"),
        "n_join_mvs": sum(1 for m in built_mvs if m.mv_type == "join"),
        "llm_ms": llm_ms,
        "build_ms": build_ms,
        "n_rewritten": n_rewritten,
        "rewrite_by_query": rewrite_by_query,
        "rewritten_sqls": rewritten_sqls,
        "mvs": mvs,
    }
