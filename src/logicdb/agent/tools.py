#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NL2Code 工具定义：供 ReAct Agent 调用的工具集
"""

from __future__ import annotations

import json
import os
import io
import csv
import re
import hashlib
import contextlib
import traceback
from collections import deque, OrderedDict
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd
import duckdb

from logicdb.data.loaders import (
    get_db_path,
    spider_schema_to_str,
    load_spider_tables,
)
from .olap_helpers import suggest_olap_llm, format_olap_suggestion, suggest_olap_code


# 安全的 pandas 可用 namespace
SAFE_BUILTINS = {
    "len", "range", "enumerate", "zip", "map", "filter", "sum", "min", "max",
    "abs", "round", "sorted", "list", "dict", "set", "tuple", "str", "int", "float",
    "bool", "None", "True", "False", "print", "isinstance", "any", "all",
    "type", "repr", "object", "getattr", "hasattr",
}
SAFE_MODULES = ["pandas", "numpy", "math", "datetime", "json", "re", "statistics", "time", "ast"]
_PARQUET_DIR_HAS_FILES_CACHE: Dict[str, bool] = {}


def _should_expose_mv_tables() -> bool:
    """Gate MV visibility to avoid prompt bloat unless explicitly enabled."""
    v = str(os.getenv("NL2CODE_EXPOSE_MV_TABLES", "")).strip().lower()
    return v in {"1", "true", "yes", "on"}
_CSV_DF_CACHE: "OrderedDict[str, Tuple[float, pd.DataFrame]]" = OrderedDict()
try:
    _CSV_DF_CACHE_MAX = max(8, int(os.environ.get("NL2CODE_CSV_CACHE_MAX_TABLES", "64")))
except Exception:
    _CSV_DF_CACHE_MAX = 64


def _is_csv_backend(db_path: str) -> bool:
    p = str(db_path or "").lower()
    return p.endswith(".csv")


def _sanitize_identifier(name: str, default: str = "col") -> str:
    out = re.sub(r"[^0-9a-zA-Z_]", "_", str(name or "").strip())
    out = re.sub(r"_+", "_", out).strip("_")
    if not out:
        out = default
    if out[0].isdigit():
        out = f"c_{out}"
    return out


def _dedup_columns(cols: List[str]) -> List[str]:
    seen: Dict[str, int] = {}
    out: List[str] = []
    for i, c in enumerate(cols):
        base = _sanitize_identifier(c, default=f"col_{i}")
        if base not in seen:
            seen[base] = 1
            out.append(base)
        else:
            seen[base] += 1
            out.append(f"{base}_{seen[base]}")
    return out


def _detect_csv_delimiter(csv_path: str) -> str:
    candidates = [",", "\t", "#", ";", "|"]
    try:
        with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
            sample = f.read(4096)
        if not sample:
            return ","
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters="".join(candidates))
            if dialect and getattr(dialect, "delimiter", None) in candidates:
                return dialect.delimiter
        except Exception:
            pass
        first_line = next((ln for ln in sample.splitlines() if ln.strip()), "")
        if not first_line:
            return ","
        counts = {c: first_line.count(c) for c in candidates}
        best = max(counts.items(), key=lambda kv: kv[1])
        return best[0] if best[1] > 0 else ","
    except Exception:
        return ","


def _load_csv_df(csv_path: str) -> pd.DataFrame:
    csv_abs = os.path.abspath(csv_path)
    mtime = os.path.getmtime(csv_abs) if os.path.exists(csv_abs) else -1.0
    cached = _CSV_DF_CACHE.get(csv_abs)
    if cached and cached[0] >= mtime:
        _CSV_DF_CACHE.move_to_end(csv_abs)
        return cached[1]
    if not os.path.exists(csv_abs):
        raise FileNotFoundError(f"CSV not found: {csv_abs}")
    delim = _detect_csv_delimiter(csv_abs)
    try:
        df = pd.read_csv(csv_abs, sep=delim, engine="python")
    except Exception:
        df = pd.read_csv(csv_abs, sep=delim, engine="python", on_bad_lines="skip")
    if df is None:
        df = pd.DataFrame()
    if len(df.columns) == 0:
        df = pd.DataFrame({"value": []})
    df = df.copy()
    df.columns = _dedup_columns([str(c) for c in df.columns])
    _CSV_DF_CACHE[csv_abs] = (mtime, df)
    _CSV_DF_CACHE.move_to_end(csv_abs)
    while len(_CSV_DF_CACHE) > _CSV_DF_CACHE_MAX:
        _CSV_DF_CACHE.popitem(last=False)
    return df


def _csv_table_aliases(db_path: str) -> List[str]:
    stem = os.path.splitext(os.path.basename(db_path))[0]
    alias = _sanitize_identifier(stem, default="table")
    abs_path = os.path.abspath(db_path)
    key = hashlib.sha1(abs_path.encode("utf-8")).hexdigest()[:16]
    aliases = ["data", f"wtq_{key}", f"tabfact_{key}"]
    if alias and alias not in aliases:
        aliases.append(alias)
    return aliases


def _csv_table_matches(table: str, db_path: str) -> bool:
    t = str(table or "").strip().lower()
    if not t:
        return False
    return t in {x.lower() for x in _csv_table_aliases(db_path)}


def _df_to_records(df: pd.DataFrame, max_rows: Optional[int] = None) -> List[Dict[str, Any]]:
    df_use = df if max_rows is None else df.head(max_rows)
    if df_use is None or len(df_use) == 0:
        return []
    clean = df_use.where(pd.notna(df_use), None)
    return clean.to_dict(orient="records")


def _safe_execute_python(code: str, df_refs: Dict[str, pd.DataFrame]) -> tuple[Optional[Any], Optional[str]]:
    """
    在受限环境中执行 Python 代码。
    df_refs: 预加载的 DataFrame 变量名 -> DataFrame
    """
    module_cache: Dict[str, Any] = {"pandas": pd, "pd": pd}

    def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
        root = str(name).split(".", 1)[0]
        if root not in SAFE_MODULES:
            raise ImportError(f"Import of '{name}' is not allowed")
        if root not in module_cache:
            module_cache[root] = __import__(root)
        mod = module_cache[root]
        if fromlist:
            return __import__(name, globals, locals, fromlist, level)
        return mod

    def _run_once(exec_code: str, local_refs: Dict[str, Any]) -> tuple[Optional[Any], Optional[str]]:
        allowed = {
            "pd": pd,
            "pandas": __import__("pandas"),
            "np": __import__("numpy"),
            "numpy": __import__("numpy"),
            "math": __import__("math"),
            "json": __import__("json"),
            "re": __import__("re"),
            "statistics": __import__("statistics"),
            "time": __import__("time"),
            "ast": __import__("ast"),
        }
        allowed.update(local_refs)
        try:
            # 单一命名空间同时作为 globals/locals，这样代码里的 globals() 会包含 df_* 等注入变量，
            # 避免 LLM 写 if 'df_shop' not in globals(): raise ... 时误判
            restricted_builtins = {
                k: getattr(__builtins__, k) if hasattr(__builtins__, k) else __builtins__[k]
                for k in SAFE_BUILTINS
                if k in dir(__builtins__) or (isinstance(__builtins__, dict) and k in __builtins__)
            }
            restricted_builtins["__import__"] = _safe_import
            ns = {"__builtins__": restricted_builtins}
            ns.update(allowed)
            # 注入 globals/locals 使 LLM 写的防御性检查能正确看到 df_*
            ns["globals"] = lambda: ns
            ns["locals"] = lambda: ns
            # Suppress model-generated print noise in batch logs.
            with contextlib.redirect_stdout(io.StringIO()):
                exec(exec_code, ns, ns)
            for name in ("result", "df", "ans", "output", "res"):
                if name in ns and ns[name] is not None:
                    val = ns[name]
                    if isinstance(val, pd.DataFrame):
                        return (
                            val.to_dict(orient="records") if len(val) <= 100 else val.head(100).to_dict(orient="records"),
                            None,
                        )
                    return (val, None)
            return (None, None)
        except Exception as e:
            return (None, f"{type(e).__name__}: {e}\n{traceback.format_exc()}")

    result, err = _run_once(code, df_refs)
    if err is None:
        return result, None

    # Fallback for escaped newlines/tabs in one-line JSON payload code (e.g. "\\n").
    # Some backends return Python code with literal escapes; executing directly causes SyntaxError.
    err_l = err.lower()
    if "\\n" in code and (
        "unexpected character after line continuation character" in err_l
        or "unterminated string literal" in err_l
        or "indentationerror" in err_l
    ):
        normalized = code.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\t", "\t")
        if normalized != code:
            result_n, err_n = _run_once(normalized, df_refs)
            if err_n is None:
                return result_n, None

    # Fallback for category arithmetic/min/max errors: degrade category cols to string and retry once.
    if "categorical is not ordered" in err_l or "category type does not support" in err_l:
        coerced_refs: Dict[str, Any] = {}
        for k, v in df_refs.items():
            if isinstance(v, pd.DataFrame):
                try:
                    df = v.copy()
                    cat_cols = list(df.select_dtypes(include=["category"]).columns)
                    for c in cat_cols:
                        df[c] = df[c].astype(str)
                    coerced_refs[k] = df
                except Exception:
                    coerced_refs[k] = v
            else:
                coerced_refs[k] = v
        result2, err2 = _run_once(code, coerced_refs)
        if err2 is None:
            return result2, None
    return (None, err)


# -----------------------------
# Tool 函数（供 Agent 直接调用）
# -----------------------------


def _sqlite_conn(db_path: str):
    """获取 sqlite3 连接"""
    import sqlite3
    return sqlite3.connect(db_path)


def _query_sqlite_native(sql: str, db_path: str) -> Dict[str, Any]:
    """使用 Python sqlite3 执行 SQL（兼容 Spider/BIRD）"""
    import sqlite3
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.execute(sql)
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description] if cur.description else []
        result = [dict(zip(cols, r)) for r in rows] if cols else [list(r) for r in rows]
        conn.close()
        return {"ok": True, "result": result[:1000], "row_count": len(rows)}
    except Exception as e:
        return {"ok": False, "error": str(e), "result": None}


def _query_csv_duckdb(sql: str, db_path: str) -> Dict[str, Any]:
    """
    Execute SQL directly on a CSV-backed single table using DuckDB in-memory.
    Table aliases include:
    - data (canonical)
    - sanitized filename stem (optional)
    """
    con = None
    try:
        df = _load_csv_df(db_path)
        aliases = _csv_table_aliases(db_path)
        con = duckdb.connect()
        con.register("_csv_df_tmp", df)
        for alias in aliases:
            con.execute(f'CREATE OR REPLACE TEMP VIEW "{alias}" AS SELECT * FROM _csv_df_tmp')
        cur = con.execute(sql)
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description] if cur.description else []
        result = [dict(zip(cols, r)) for r in rows] if cols else [list(r) for r in rows]
        return {
            "ok": True,
            "result": result[:1000],
            "row_count": len(rows),
            "backend": "csv_duckdb",
            "table_aliases": aliases,
        }
    except Exception as e:
        return {"ok": False, "error": str(e), "result": None}
    finally:
        if con is not None:
            try:
                con.close()
            except Exception:
                pass


def tool_query_sql(
    sql: str,
    db_path: str,
    con: Optional[Any] = None,
) -> Dict[str, Any]:
    """在 SQLite 上执行 SQL（使用 sqlite3，兼容 Spider/BIRD）"""
    if _is_csv_backend(db_path):
        return _query_csv_duckdb(sql, db_path)
    return _query_sqlite_native(sql, db_path)


def _parquet_dir_required_ok(parquet_dir: Optional[str]) -> Tuple[bool, Optional[str]]:
    """
    当 parquet_dir 被显式指定时，禁止回退到 SQLite：目录必须存在且有效。
    返回 (True, None) 表示可继续用 Parquet；(False, error_msg) 表示应直接返回错误。
    """
    if parquet_dir is None:
        return True, None  # 未指定 Parquet，允许走 SQLite 路径
    if not os.path.isdir(parquet_dir):
        return False, (
            f"parquet_dir 已指定但目录不存在或不可用: {parquet_dir}。"
            "2Code 使用 Parquet 时不回退到 SQLite，请先对该 db_id 执行离线优化。"
        )
    has_parquet = _PARQUET_DIR_HAS_FILES_CACHE.get(parquet_dir)
    if has_parquet is None:
        has_parquet = False
        for root, _, files in os.walk(parquet_dir):
            if any(f.endswith(".parquet") for f in files):
                has_parquet = True
                break
        _PARQUET_DIR_HAS_FILES_CACHE[parquet_dir] = has_parquet
    if not has_parquet:
        return False, (
            f"parquet_dir 可访问但没有任何 parquet 文件: {parquet_dir}。"
            "请重新运行离线优化（或修复导出失败）后再评测。"
        )
    return True, None


def tool_get_schema(
    db_path: str,
    spider_tables: Optional[Dict[str, Dict]] = None,
    db_id: Optional[str] = None,
    use_mschema: bool = True,
    parquet_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    获取数据库 schema。
    parquet_dir 若被指定则必须存在，否则返回错误（不回退到 SQLite）。
    """
    ok, err = _parquet_dir_required_ok(parquet_dir)
    if not ok:
        return {"ok": False, "error": err}

    expose_mv = _should_expose_mv_tables()

    # Prefer parquet path when provided, even for CSV-backed datasets.
    if parquet_dir and os.path.isdir(parquet_dir):
        tables = _list_parquet_tables(parquet_dir)
        regular_tables = [t for t in tables if not t.startswith("mv_")]
        mv_tables = [t for t in tables if t.startswith("mv_")]
        visible_tables = tables if expose_mv else regular_tables
        manifest = _load_manifest(parquet_dir)
        mv_notes = _mv_semantic_notes(manifest) if expose_mv else {}
        schema_str = _parquet_schema_to_str(parquet_dir, visible_tables, mv_notes=mv_notes)
        return {
            "ok": True,
            "schema": schema_str,
            "tables": regular_tables,
            "mv_tables": mv_tables if expose_mv else [],
            "mv_tables_hidden": 0 if expose_mv else len(mv_tables),
            "format": "parquet",
        }

    if _is_csv_backend(db_path):
        try:
            df = _load_csv_df(db_path)
            tables = ["data"]
            if use_mschema:
                lines: List[str] = ["# Note: only table available is data", "# Table: data"]
                fields: List[str] = []
                for col in df.columns:
                    dtype = str(df[col].dtype).upper()
                    dtype = "TEXT" if "OBJECT" in dtype or "STRING" in dtype else dtype
                    field_line = f"({col}:{dtype}"
                    ex_vals = [x for x in df[col].head(2).tolist() if x is not None and str(x) != "nan"]
                    if ex_vals:
                        ex_str = ", ".join(repr(str(x)[:30]) for x in ex_vals)
                        field_line += f", Examples: [{ex_str}]"
                    field_line += ")"
                    fields.append(field_line)
                lines.append("[" + ",\n".join(fields) + "]")
                schema_str = "\n".join(lines)
                return {
                    "ok": True,
                    "schema": schema_str,
                    "tables": tables,
                    "format": "csv_mschema",
                    "table_aliases": _csv_table_aliases(db_path),
                }
            cols = [f"  {c} ({str(df[c].dtype)})" for c in df.columns]
            return {
                "ok": True,
                "schema": "Only table available: data\n\nTable data:\n" + "\n".join(cols),
                "tables": tables,
                "format": "csv",
                "table_aliases": _csv_table_aliases(db_path),
            }
        except Exception as e:
            return {"ok": False, "error": str(e)}

    if parquet_dir and os.path.isdir(parquet_dir):
        tables = _list_parquet_tables(parquet_dir)
        regular_tables = [t for t in tables if not t.startswith("mv_")]
        mv_tables = [t for t in tables if t.startswith("mv_")]
        visible_tables = tables if expose_mv else regular_tables
        manifest = _load_manifest(parquet_dir)
        mv_notes = _mv_semantic_notes(manifest) if expose_mv else {}
        schema_str = _parquet_schema_to_str(parquet_dir, visible_tables, mv_notes=mv_notes)
        return {
            "ok": True,
            "schema": schema_str,
            "tables": regular_tables,
            "mv_tables": mv_tables if expose_mv else [],
            "mv_tables_hidden": 0 if expose_mv else len(mv_tables),
            "format": "parquet",
        }

    if use_mschema:
        try:
            from logicdb.data.loaders import build_mschema_from_sqlite
            schema_str = build_mschema_from_sqlite(db_path)
            import sqlite3
            conn = sqlite3.connect(db_path)
            tables = [r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            ).fetchall()]
            conn.close()
            return {"ok": True, "schema": schema_str, "tables": tables, "format": "mschema"}
        except Exception as e:
            pass  # fallback
    if spider_tables and db_id and db_id in spider_tables:
        schema_str = spider_schema_to_str(spider_tables[db_id])
        tables = spider_tables[db_id].get("table_names_original", spider_tables[db_id].get("table_names", []))
        return {"ok": True, "schema": schema_str, "tables": tables}
    try:
        import sqlite3
        conn = sqlite3.connect(db_path)
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )
        tables = [r[0] for r in cursor.fetchall()]
        lines = []
        for t in tables:
            cursor = conn.execute(f"PRAGMA table_info({t})")
            cols = [f"  {r[1]} ({r[2]})" for r in cursor.fetchall()]
            lines.append(f"Table {t}:\n" + "\n".join(cols))
        conn.close()
        return {"ok": True, "schema": "\n\n".join(lines), "tables": tables}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def _quote_ident(name: str) -> str:
    """SQLite 标识符加引号（含空格/括号时）"""
    if any(c in name for c in " ()-"):
        return f'"{name}"'
    return name


def tool_get_sample(
    db_path: str,
    table: str,
    n: int = 5,
) -> Dict[str, Any]:
    """获取表的样本行"""
    if _is_csv_backend(db_path):
        try:
            if not _csv_table_matches(table, db_path):
                return {
                    "ok": False,
                    "error": f"unknown table '{table}' for csv backend; use table 'data'",
                    "tables": ["data"],
                }
            n_use = max(1, min(int(n), 5000))
            df = _load_csv_df(db_path)
            return {"ok": True, "result": _df_to_records(df, n_use), "row_count": min(n_use, len(df)), "table": "data"}
        except Exception as e:
            return {"ok": False, "error": str(e), "result": None}
    tq = _quote_ident(table)
    res = _query_sqlite_native(f"SELECT * FROM {tq} LIMIT {n}", db_path)
    if res.get("ok"):
        res["table"] = table
    return res


def tool_list_tables(db_path: str, parquet_dir: Optional[str] = None) -> Dict[str, Any]:
    """列出所有表。parquet_dir 若指定则必须存在，否则报错（不回退 SQLite）。"""
    ok, err = _parquet_dir_required_ok(parquet_dir)
    if not ok:
        return {"ok": False, "error": err}
    expose_mv = _should_expose_mv_tables()
    # Prefer parquet path when provided, even for CSV-backed datasets.
    if parquet_dir and os.path.isdir(parquet_dir):
        all_tables = _list_parquet_tables(parquet_dir)
        regular = [t for t in all_tables if not t.startswith("mv_")]
        mv = [t for t in all_tables if t.startswith("mv_")]
        return {
            "ok": True,
            "tables": regular if not expose_mv else (regular + mv),
            "mv_tables": mv if expose_mv else [],
            "mv_tables_hidden": 0 if expose_mv else len(mv),
        }
    if _is_csv_backend(db_path):
        return {"ok": True, "tables": ["data"], "format": "csv", "table_aliases": _csv_table_aliases(db_path)}
    if parquet_dir and os.path.isdir(parquet_dir):
        all_tables = _list_parquet_tables(parquet_dir)
        regular = [t for t in all_tables if not t.startswith("mv_")]
        mv = [t for t in all_tables if t.startswith("mv_")]
        return {
            "ok": True,
            "tables": regular if not expose_mv else (regular + mv),
            "mv_tables": mv if expose_mv else [],
            "mv_tables_hidden": 0 if expose_mv else len(mv),
        }
    try:
        import sqlite3
        conn = sqlite3.connect(db_path)
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()]
        conn.close()
        return {"ok": True, "tables": tables}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def tool_execute_python(
    code: str,
    df_refs: Optional[Dict[str, pd.DataFrame]] = None,
    sample: Optional[list] = None,
    samples_dict: Optional[Dict[str, list]] = None,
    parquet_df_cache: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    执行 Python/pandas 代码。
    - sample: 最近一次 read_table 的样本行
    - samples_dict: {表名: rows} 供多表 merge（回退用）
    - parquet_df_cache: {表名: DataFrame} parquet 模式下的 categorical 优化 DataFrame，
      优先于 samples_dict 中的行列表重建（性能更优且类型正确）
    """
    df_refs = dict(df_refs or {})

    # 优先使用 parquet_df_cache 中的优化 DataFrame
    if parquet_df_cache:
        for k, df in parquet_df_cache.items():
            safe_name = k.replace("-", "_").replace(" ", "_")
            df_refs[f"df_{safe_name}"] = df
        if sample is not None:
            df_refs["sample_df"] = pd.DataFrame(sample)
            df_refs["sample"] = sample
    else:
        if sample is not None:
            df_refs["sample_df"] = pd.DataFrame(sample)
            df_refs["sample"] = sample
        if samples_dict:
            for k, v in samples_dict.items():
                safe_name = k.replace("-", "_").replace(" ", "_")
                df_refs[f"df_{safe_name}"] = pd.DataFrame(v) if v else pd.DataFrame()
            df_refs["samples"] = samples_dict

    # 大小写兼容：LLM 可能用 df_pets 而表名是 Pets（df_Pets），做双向 alias
    # 规则：对所有已有 df_* 变量，补一个全小写版本（若不冲突）
    aliases: Dict[str, Any] = {}
    for key, val in list(df_refs.items()):
        if key.startswith("df_"):
            lower_key = "df_" + key[3:].lower()
            if lower_key not in df_refs:
                aliases[lower_key] = val
    df_refs.update(aliases)

    result, err = _safe_execute_python(code, df_refs)
    if err:
        return {"ok": False, "error": err, "result": None}
    return {"ok": True, "result": result}


def tool_describe_table(db_path: str, table: str, parquet_dir: Optional[str] = None) -> Dict[str, Any]:
    """获取表的统计信息。parquet_dir 若指定则必须存在，否则报错（不回退 SQLite）。"""
    ok, err = _parquet_dir_required_ok(parquet_dir)
    if not ok:
        return {"ok": False, "error": err}
    # Prefer parquet path when provided, even for CSV-backed datasets.
    if parquet_dir and os.path.isdir(parquet_dir):
        return _describe_parquet_table(parquet_dir, table)
    if _is_csv_backend(db_path):
        try:
            if not _csv_table_matches(table, db_path):
                return {
                    "ok": False,
                    "error": f"unknown table '{table}' for csv backend; use table 'data'",
                    "tables": ["data"],
                }
            df = _load_csv_df(db_path)
            row_count = int(len(df))
            null_counts = {str(c): int(df[c].isna().sum()) for c in df.columns}
            return {
                "ok": True,
                "table": "data",
                "row_count": row_count,
                "null_counts": null_counts,
                "sample": _df_to_records(df, 5),
                "format": "csv",
            }
        except Exception as e:
            return {"ok": False, "error": str(e)}
    if parquet_dir and os.path.isdir(parquet_dir):
        return _describe_parquet_table(parquet_dir, table)
    tq = _quote_ident(table)
    cnt_res = _query_sqlite_native(f"SELECT COUNT(*) AS __cnt FROM {tq}", db_path)
    if not cnt_res.get("ok"):
        return cnt_res
    row_count = int((cnt_res.get("result") or [{}])[0].get("__cnt", 0) or 0)

    cols_res = _query_sqlite_native(f"PRAGMA table_info({tq})", db_path)
    if not cols_res.get("ok"):
        return cols_res
    cols = [str(r.get("name")) for r in (cols_res.get("result") or []) if r.get("name")]

    null_counts: Dict[str, int] = {}
    if cols:
        null_expr = ", ".join(
            f'SUM(CASE WHEN {_quote_ident(c)} IS NULL THEN 1 ELSE 0 END) AS {_quote_ident(c)}'
            for c in cols
        )
        null_res = _query_sqlite_native(f"SELECT {null_expr} FROM {tq}", db_path)
        if null_res.get("ok") and null_res.get("result"):
            first = null_res["result"][0]
            null_counts = {k: int(first.get(k, 0) or 0) for k in cols}

    sample_res = _query_sqlite_native(f"SELECT * FROM {tq} LIMIT 5", db_path)
    if not sample_res.get("ok"):
        return sample_res
    return {
        "ok": True,
        "table": table,
        "row_count": row_count,
        "null_counts": null_counts,
        "sample": sample_res.get("result", []),
    }


def _normalize_tables_arg(tables: Any) -> List[str]:
    if tables is None:
        return []
    if isinstance(tables, str):
        return [t.strip() for t in tables.split(",") if t.strip()]
    if isinstance(tables, (list, tuple)):
        out: List[str] = []
        for t in tables:
            ts = str(t).strip()
            if ts:
                out.append(ts)
        return out
    return []


def _sqlite_list_tables(db_path: str) -> List[str]:
    if _is_csv_backend(db_path):
        return ["data"]
    try:
        import sqlite3
        conn = sqlite3.connect(db_path)
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
        conn.close()
        return [str(r[0]) for r in rows]
    except Exception:
        return []


def _sqlite_table_info(conn, table: str) -> Tuple[List[str], List[str]]:
    """Return (columns, pk_columns)."""
    rows = conn.execute(f'PRAGMA table_info("{table}")').fetchall()
    cols: List[str] = []
    pks: List[str] = []
    for r in rows:
        # pragma table_info: cid, name, type, notnull, dflt_value, pk
        col = str(r[1])
        cols.append(col)
        if int(r[5] or 0) > 0:
            pks.append(col)
    return cols, pks


def _edge_key(a: str, b: str, ca: str, cb: str) -> Tuple[str, str, str, str]:
    return (a.lower(), b.lower(), ca.lower(), cb.lower())


def _shortest_join_paths(
    graph: Dict[str, List[Dict[str, Any]]],
    start_table: str,
    end_table: str,
    max_hops: int,
    top_k: int,
) -> List[Dict[str, Any]]:
    if start_table not in graph or end_table not in graph:
        return []
    if start_table == end_table:
        return [{
            "from": start_table,
            "to": end_table,
            "hops": 0,
            "tables": [start_table],
            "join_chain": [],
            "avg_confidence": 1.0,
        }]

    q = deque()
    q.append((start_table, [start_table], []))  # node, table_path, edge_path
    paths: List[Dict[str, Any]] = []
    seen = {(start_table, 0)}
    while q and len(paths) < top_k:
        node, t_path, e_path = q.popleft()
        if len(e_path) > max_hops:
            continue
        if node == end_table and e_path:
            avg = sum(float(e.get("confidence", 0.5)) for e in e_path) / max(len(e_path), 1)
            paths.append({
                "from": start_table,
                "to": end_table,
                "hops": len(e_path),
                "tables": t_path,
                "join_chain": [str(e.get("on", "")) for e in e_path],
                "avg_confidence": round(avg, 3),
            })
            continue
        if len(e_path) >= max_hops:
            continue
        nbrs = sorted(graph.get(node, []), key=lambda x: float(x.get("confidence", 0.0)), reverse=True)
        for e in nbrs:
            nxt = str(e.get("to"))
            if nxt in t_path:
                continue
            state = (nxt, len(e_path) + 1)
            if state in seen:
                continue
            seen.add(state)
            q.append((nxt, t_path + [nxt], e_path + [e]))

    paths.sort(key=lambda x: (x["hops"], -float(x["avg_confidence"])))
    return paths[:top_k]


def tool_get_join_path(
    db_path: str,
    tables: Optional[Any] = None,
    start_table: Optional[str] = None,
    end_table: Optional[str] = None,
    max_hops: int = 3,
    top_k: int = 8,
) -> Dict[str, Any]:
    """
    Suggest join edges/paths based on FK metadata + column-name matching.
    Helps LLM choose join keys before writing SQL or pandas merge logic.
    """
    if _is_csv_backend(db_path):
        return {
            "ok": True,
            "tables_considered": ["data"],
            "fk_join_edges": [],
            "name_match_edges": [],
            "suggested_paths": [],
            "note": "CSV backend currently exposes a single table ('data'); join-path tool is usually unnecessary.",
        }

    import sqlite3

    all_tables = _sqlite_list_tables(db_path)
    if not all_tables:
        return {"ok": False, "error": "failed to load tables from sqlite"}
    table_set = {t.lower(): t for t in all_tables}

    wanted = _normalize_tables_arg(tables)
    if wanted:
        chosen: List[str] = []
        for t in wanted:
            key = t.lower()
            if key in table_set:
                chosen.append(table_set[key])
        tables_use = sorted(set(chosen))
    else:
        tables_use = all_tables

    if not tables_use:
        return {"ok": False, "error": "no valid tables selected"}

    start = table_set.get(str(start_table).lower(), start_table) if start_table else None
    end = table_set.get(str(end_table).lower(), end_table) if end_table else None

    conn = sqlite3.connect(db_path)
    try:
        col_map: Dict[str, List[str]] = {}
        pk_map: Dict[str, List[str]] = {}
        for t in tables_use:
            cols, pks = _sqlite_table_info(conn, t)
            col_map[t] = cols
            pk_map[t] = pks

        fk_edges: List[Dict[str, Any]] = []
        name_edges: List[Dict[str, Any]] = []
        graph: Dict[str, List[Dict[str, Any]]] = {t: [] for t in tables_use}
        seen_edges = set()

        # FK-derived join edges.
        for t in tables_use:
            rows = conn.execute(f'PRAGMA foreign_key_list("{t}")').fetchall()
            for r in rows:
                ref_t = str(r[2])
                from_c = str(r[3] or "")
                to_c = str(r[4] or from_c)
                if ref_t not in graph or not from_c:
                    continue
                key = _edge_key(t, ref_t, from_c, to_c)
                if key in seen_edges:
                    continue
                seen_edges.add(key)
                on_expr = f'{t}."{from_c}" = {ref_t}."{to_c}"'
                edge = {
                    "left_table": t,
                    "right_table": ref_t,
                    "left_col": from_c,
                    "right_col": to_c,
                    "on": on_expr,
                    "source": "foreign_key",
                    "confidence": 1.0,
                }
                fk_edges.append(edge)
                graph[t].append({"to": ref_t, "on": on_expr, "source": "foreign_key", "confidence": 1.0})
                graph[ref_t].append({
                    "to": t,
                    "on": f'{ref_t}."{to_c}" = {t}."{from_c}"',
                    "source": "foreign_key",
                    "confidence": 1.0,
                })

        # Name-match inferred edges (fallback when FK metadata is sparse/missing).
        tbls = tables_use
        for i in range(len(tbls)):
            t1 = tbls[i]
            c1 = col_map.get(t1, [])
            c1_l = {c.lower(): c for c in c1}
            for j in range(i + 1, len(tbls)):
                t2 = tbls[j]
                c2 = col_map.get(t2, [])
                c2_l = {c.lower(): c for c in c2}
                common_l = sorted(set(c1_l.keys()) & set(c2_l.keys()))
                if not common_l:
                    continue
                scored: List[Tuple[float, str, str]] = []
                for lc in common_l:
                    a = c1_l[lc]
                    b = c2_l[lc]
                    # Skip if FK edge already exists on same pair/cols.
                    if _edge_key(t1, t2, a, b) in seen_edges or _edge_key(t2, t1, b, a) in seen_edges:
                        continue
                    score = 0.55
                    if lc == "id":
                        score = 0.35
                    if lc.endswith("_id") or lc.endswith("id"):
                        score = max(score, 0.8)
                    if "code" in lc or lc.endswith("_code"):
                        score = max(score, 0.85)
                    if a in pk_map.get(t1, []) or b in pk_map.get(t2, []):
                        score += 0.08
                    scored.append((score, a, b))
                scored.sort(reverse=True)
                for score, a, b in scored[:2]:
                    if score < 0.7:
                        continue
                    key = _edge_key(t1, t2, a, b)
                    if key in seen_edges:
                        continue
                    seen_edges.add(key)
                    on_expr = f'{t1}."{a}" = {t2}."{b}"'
                    edge = {
                        "left_table": t1,
                        "right_table": t2,
                        "left_col": a,
                        "right_col": b,
                        "on": on_expr,
                        "source": "name_match",
                        "confidence": round(score, 3),
                    }
                    name_edges.append(edge)
                    graph[t1].append({"to": t2, "on": on_expr, "source": "name_match", "confidence": round(score, 3)})
                    graph[t2].append({
                        "to": t1,
                        "on": f'{t2}."{b}" = {t1}."{a}"',
                        "source": "name_match",
                        "confidence": round(score, 3),
                    })

        suggested_paths: List[Dict[str, Any]] = []
        if start and end:
            if start not in graph or end not in graph:
                return {"ok": False, "error": f"start/end not in selected tables: {start}, {end}"}
            suggested_paths = _shortest_join_paths(
                graph,
                start_table=start,
                end_table=end,
                max_hops=max(1, int(max_hops)),
                top_k=max(1, int(top_k)),
            )
        else:
            direct = sorted(
                fk_edges + name_edges,
                key=lambda x: (0 if x.get("source") == "foreign_key" else 1, -float(x.get("confidence", 0.0))),
            )
            suggested_paths = [{
                "from": e["left_table"],
                "to": e["right_table"],
                "hops": 1,
                "tables": [e["left_table"], e["right_table"]],
                "join_chain": [e["on"]],
                "avg_confidence": e["confidence"],
                "source": e["source"],
            } for e in direct[:max(1, int(top_k))]]

        return {
            "ok": True,
            "tables_considered": tables_use,
            "fk_join_edges": fk_edges[:50],
            "name_match_edges": name_edges[:50],
            "suggested_paths": suggested_paths,
            "note": (
                "Use these ON conditions as strong join candidates. Prefer foreign_key edges first; "
                "use name_match edges as fallback and verify row counts."
            ),
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}
    finally:
        conn.close()


def tool_read_table(
    db_path: str,
    table: str,
    parquet_dir: Optional[str] = None,
    partition_filter: Optional[Dict[str, Any]] = None,
    limit: int = 100,
    mode: str = "sample",
) -> Dict[str, Any]:
    """将表读入并返回样本或全量行（SQLite）。

    默认返回 sample（limit 行，默认 100）。可通过 mode='full' 请求全量读取（仅 SQLite 支持）。
    parquet_dir 若指定则必须存在，否则报错（不回退 SQLite）。
    若表为 Hive 分区表，可传 partition_filter 仅读部分分区（如 {"p_date_year": 2020}）以节省 I/O。
    """
    ok, err = _parquet_dir_required_ok(parquet_dir)
    if not ok:
        return {"ok": False, "error": err}

    try:
        sample_limit = int(limit)
    except Exception:
        sample_limit = 100
    if sample_limit <= 0:
        sample_limit = 100
    sample_limit = max(1, min(sample_limit, 5000))

    read_mode = str(mode or "sample").strip().lower()
    if read_mode not in {"sample", "full"}:
        read_mode = "sample"

    # Prefer parquet path when provided, even for CSV-backed datasets.
    if parquet_dir and os.path.isdir(parquet_dir):
        manifest = _load_manifest(parquet_dir)
        # Parquet path is optimized for preview + lazy full DataFrame cache in agent.
        return _read_parquet_table(
            parquet_dir,
            table,
            partition_filter=partition_filter,
            manifest=manifest,
            sample_limit=sample_limit,
        )

    if _is_csv_backend(db_path):
        try:
            if not _csv_table_matches(table, db_path):
                return {
                    "ok": False,
                    "error": f"unknown table '{table}' for csv backend; use table 'data'",
                    "tables": ["data"],
                }
            df = _load_csv_df(db_path)
            row_count = int(len(df))
            if read_mode == "full":
                if row_count > 250000:
                    return {
                        "ok": False,
                        "error": (
                            f"Table data has {row_count} rows; full load is blocked (>250000). "
                            "Use mode='sample' with a suitable limit."
                        ),
                    }
                rows = _df_to_records(df, None)
            else:
                rows = _df_to_records(df, sample_limit)
            return {
                "ok": True,
                "table": "data",
                "row_count": row_count,
                "sample": rows,
                "read_mode": read_mode,
                "sample_limit": sample_limit if read_mode == "sample" else row_count,
                "sampled": read_mode == "sample" and row_count > len(rows),
                "note": (
                    f"read_table returns {'sampled rows' if read_mode == 'sample' else 'full rows'}; "
                    "do not treat sampled rows as exact global aggregates."
                ),
                "format": "csv",
                "table_aliases": _csv_table_aliases(db_path),
            }
        except Exception as e:
            return {"ok": False, "error": str(e)}

    if parquet_dir and os.path.isdir(parquet_dir):
        manifest = _load_manifest(parquet_dir)
        # Parquet path is optimized for preview + lazy full DataFrame cache in agent.
        return _read_parquet_table(
            parquet_dir,
            table,
            partition_filter=partition_filter,
            manifest=manifest,
            sample_limit=sample_limit,
        )
    tq = _quote_ident(table)
    cnt_res = _query_sqlite_native(f"SELECT COUNT(*) AS __cnt FROM {tq}", db_path)
    if not cnt_res.get("ok"):
        return cnt_res
    row_count = int((cnt_res.get("result") or [{}])[0].get("__cnt", 0) or 0)

    if read_mode == "full":
        if row_count > 250000:
            return {
                "ok": False,
                "error": (
                    f"Table {table} has {row_count} rows; full load is blocked (>250000). "
                    "Use mode='sample' with a suitable limit."
                ),
            }
        sample_res = _query_sqlite_native(f"SELECT * FROM {tq}", db_path)
    else:
        sample_res = _query_sqlite_native(f"SELECT * FROM {tq} LIMIT {sample_limit}", db_path)
    if not sample_res.get("ok"):
        return sample_res
    rows = sample_res.get("result", [])
    return {
        "ok": True,
        "table": table,
        "row_count": row_count,
        "sample": rows,
        "read_mode": read_mode,
        "sample_limit": sample_limit if read_mode == "sample" else row_count,
        "sampled": read_mode == "sample" and row_count > len(rows),
        "note": (
            f"read_table returns {'sampled rows' if read_mode == 'sample' else 'full rows'}; "
            "do not treat sampled rows as exact global aggregates."
        ),
    }


def _load_manifest(parquet_dir: str) -> Optional[Dict]:
    """
    从 parquet_dir 或其上级目录中加载 manifest.json。
    manifest 里包含 LLM 推荐的 MV 语义注解等信息。
    """
    # 先查 parquet_dir 同级（optimized/ → db_work/manifest.json）
    candidate = os.path.join(os.path.dirname(parquet_dir), "manifest.json")
    if os.path.exists(candidate):
        try:
            with open(candidate, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    # 也尝试 parquet_dir 自身
    candidate2 = os.path.join(parquet_dir, "manifest.json")
    if os.path.exists(candidate2):
        try:
            with open(candidate2, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return None


def _mv_semantic_notes(manifest: Optional[Dict]) -> Dict[str, str]:
    """从 manifest 中提取 {mv_name: semantic_note} 映射"""
    if not manifest:
        return {}
    notes: Dict[str, str] = {}
    for rec in manifest.get("mv_records", []):
        name = rec.get("name", "")
        note = rec.get("semantic_note", "")
        if name and note:
            notes[name] = note
    # 兼容旧格式（无 mv_records 只有 mv_tables 列表）
    return notes


def _partition_by_to_hive_columns(partition_by: List[Dict]) -> List[str]:
    """Manifest table_ops partition_by → Hive 分区列名（与 layout_operators 一致）。"""
    names = []
    for item in partition_by:
        col = item.get("col", "")
        tf = item.get("transform", "identity")
        if tf == "identity":
            names.append(f"p_{col}")
        elif tf == "year":
            names.append(f"p_{col}_year")
        elif tf == "month":
            names.append(f"p_{col}_month")
        elif tf == "day":
            names.append(f"p_{col}_day")
        else:
            names.append(f"p_{col}_{tf}")
    return names


def _get_partition_by_for_table(manifest: Optional[Dict], table: str) -> Optional[List[Dict]]:
    """从 manifest.table_ops 取该表的 partition_by（若为 PARTITION 表）。"""
    if not manifest:
        return None
    for op in manifest.get("table_ops", []):
        if (op.get("op") == "PARTITION" and (op.get("table") == table or op.get("dest") == table)):
            return op.get("partition_by") or []
    return None


def _list_parquet_tables(parquet_dir: str) -> List[str]:
    """列出单文件表（*.parquet）与分区表（子目录内含 *.parquet）。"""
    tables = []
    for fn in os.listdir(parquet_dir):
        path = os.path.join(parquet_dir, fn)
        if fn.endswith(".parquet"):
            tables.append(fn[: -len(".parquet")])
        elif os.path.isdir(path):
            for _ in _iter_parquet_in_dir(path):
                tables.append(fn)
                break
    return sorted(set(tables))


def _iter_parquet_in_dir(d: str):
    """递归 yield 目录下所有 .parquet 路径。"""
    for root, _, files in os.walk(d):
        for f in files:
            if f.endswith(".parquet"):
                yield os.path.join(root, f)


def _sample_column_examples(
    con,
    rp_expr: str,
    col_names: List[str],
    sample_size: int = 1000,
    max_examples: int = 3,
    max_val_len: int = 40,
) -> Dict[str, Tuple[List[str], bool]]:
    """
    从 parquet 表采样 sample_size 行，为每列提取最多 max_examples 个非 null 示例值，
    并判断该列是否含 null（仅基于样本，稀疏 null 可能漏检）。

    返回 {col_name: (example_str_list, has_null)}。
    parquet 与 SQLite 的差异：DuckDB 返回 arrow 类型（int64/float64/date/timestamp 等），
    None/NaN/pd.NaT 均视为 null；统一 str() 转换后过滤 nan/none/nat 字样。
    """
    _NULL_STRS = {"nan", "none", "nat", "<na>", ""}
    result: Dict[str, Tuple[List[str], bool]] = {}
    try:
        sample_df = con.execute(f"SELECT * FROM {rp_expr} LIMIT {sample_size}").fetchdf()
    except Exception:
        return result

    for col in col_names:
        if col not in sample_df.columns:
            result[col] = ([], False)
            continue
        series = sample_df[col]
        has_null = bool(series.isnull().any())

        seen: List[str] = []
        seen_set: set = set()
        for v in series:
            if v is None or (isinstance(v, float) and pd.isna(v)):
                continue
            try:
                s = str(v).strip()
            except Exception:
                continue
            if s.lower() in _NULL_STRS:
                continue
            if len(s) > max_val_len:
                s = s[:max_val_len] + "…"
            if s not in seen_set:
                seen_set.add(s)
                seen.append(s)
            if len(seen) >= max_examples:
                break
        result[col] = (seen, has_null)
    return result


def _parquet_schema_to_str(parquet_dir: str, tables: List[str],
                            mv_notes: Optional[Dict[str, str]] = None,
                            index_hints: Optional[Dict[str, Dict]] = None,
                            manifest: Optional[Dict] = None) -> str:
    """
    生成 Parquet 目录的 schema 字符串，包含：
    - 每列的示例值（最多 3 个非 null 值，若存在 null 则附加 NULL 提示）
    - fast-lookup 列（key col，df.loc[val] O(1) 查找）
    - category 列（低基数列，Category dtype 加速 filter/groupby）
    - partition 列（Hive 分区，支持 partition_filter 减少 I/O）
    - MV 表的语义注解
    """
    mv_notes = mv_notes or {}
    index_hints = index_hints or {}
    manifest = manifest or _load_manifest(parquet_dir)
    con = duckdb.connect()
    try:
        lines = []
        regular_tables = [t for t in tables if not t.startswith("mv_")]
        mv_tables = [t for t in tables if t.startswith("mv_")]

        summary_parts = [f"Base tables: {len(regular_tables)} ({', '.join(regular_tables[:8])}{'...' if len(regular_tables) > 8 else ''})."]
        partitioned = [t for t in regular_tables if _get_partition_by_for_table(manifest, t)]
        if partitioned:
            summary_parts.append(f"Partitioned (use partition_filter to load subset): {', '.join(partitioned)}.")
        else:
            summary_parts.append("Partitioned: none.")
        summary_parts.append(f"Materialized views: {len(mv_tables)} (pre-joined; see [MV NOTE] per table).")
        lines.append("-- Storage: " + " ".join(summary_parts))

        # 普通表（单文件或分区目录）
        for t in regular_tables:
            p = os.path.join(parquet_dir, f"{t}.parquet")
            part_dir = os.path.join(parquet_dir, t)
            rp_expr = None
            if os.path.isfile(p):
                p_sql = p.replace("'", "''")
                rp_expr = f"read_parquet('{p_sql}')"
            elif os.path.isdir(part_dir):
                glob_path = os.path.join(part_dir, "**", "*.parquet").replace("'", "''")
                rp_expr = f"read_parquet('{glob_path}', hive_partitioning=true)"
            if not rp_expr:
                continue
            try:
                info = con.execute(f"DESCRIBE SELECT * FROM {rp_expr}").fetchall()
                col_names = [r[0] for r in info]
                col_types = {r[0]: r[1] for r in info}

                # 每列采样示例值
                examples = _sample_column_examples(con, rp_expr, col_names)

                col_lines = []
                for cname in col_names:
                    ctype = col_types[cname]
                    ex_vals, has_null = examples.get(cname, ([], False))
                    display_vals = list(ex_vals)
                    if has_null:
                        display_vals.append("NULL")
                    if display_vals:
                        ex_str = ", ".join(repr(v) for v in display_vals)
                        col_lines.append(f"  {cname} ({ctype}, e.g. {ex_str})")
                    else:
                        col_lines.append(f"  {cname} ({ctype})")

                block = f"Table {t}:\n" + "\n".join(col_lines)

                hints = index_hints.get(t, {})
                hint_parts = []
                idx_col = hints.get("pandas_index_col")
                cat_cols = hints.get("categorical_cols") or []
                if idx_col:
                    hint_parts.append(
                        f"key col: {idx_col} (primary key — use df[df['{idx_col}']==val] to filter by key)"
                    )
                if cat_cols:
                    hint_parts.append(
                        f"category cols: {', '.join(cat_cols)} (Category dtype, fast filter/groupby)"
                    )
                pby = _get_partition_by_for_table(manifest, t)
                if pby:
                    hive_cols = _partition_by_to_hive_columns(pby)
                    hint_parts.append(
                        f"partitioned by {', '.join(hive_cols)} — call read_table with partition_filter"
                        f" e.g. {{\"{hive_cols[0]}\": 2020}} to load only that partition"
                    )
                if hint_parts:
                    block += f"\n  [HINT] {'; '.join(hint_parts)}"
                lines.append(block)
            except Exception:
                continue

        # MV 表 —— 附加语义注解 + 示例值
        if mv_tables:
            lines.append("\n-- Materialized Views (pre-joined tables, read-only) --")
            for t in mv_tables:
                p = os.path.join(parquet_dir, f"{t}.parquet")
                if not os.path.exists(p):
                    continue
                try:
                    p_sql = p.replace("'", "''")
                    rp_expr_mv = f"read_parquet('{p_sql}')"
                    info = con.execute(
                        f"DESCRIBE SELECT * FROM {rp_expr_mv}"
                    ).fetchall()
                    col_names_mv = [r[0] for r in info]
                    col_types_mv = {r[0]: r[1] for r in info}
                    examples_mv = _sample_column_examples(con, rp_expr_mv, col_names_mv)

                    col_lines = []
                    for cname in col_names_mv:
                        ctype = col_types_mv[cname]
                        ex_vals, has_null = examples_mv.get(cname, ([], False))
                        display_vals = list(ex_vals)
                        if has_null:
                            display_vals.append("NULL")
                        if display_vals:
                            ex_str = ", ".join(repr(v) for v in display_vals)
                            col_lines.append(f"  {cname} ({ctype}, e.g. {ex_str})")
                        else:
                            col_lines.append(f"  {cname} ({ctype})")

                    block = f"Table {t}:\n" + "\n".join(col_lines)
                    note = mv_notes.get(t, "")
                    if note:
                        block += f"\n  [MV NOTE] {note}"
                    lines.append(block)
                except Exception:
                    continue

        return "\n\n".join(lines) if lines else "No schema"
    finally:
        con.close()


def _get_index_rec_for_table(parquet_dir: str, table: str) -> Optional[Dict]:
    """从 manifest.index_recs 中取该表的 index 建议（含 categorical_cols / pandas_index_col）"""
    manifest = _load_manifest(parquet_dir)
    if not manifest:
        return None
    for rec in manifest.get("index_recs", []):
        if rec.get("table") == table:
            return rec
    return None


def _parquet_to_pandas_optimized(
    parquet_path: str,
    categorical_hint: Optional[List[str]] = None,
    pandas_index_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    将 Parquet 文件读取为 pandas DataFrame，应用列类型优化：

    1. **Bloom Filter**（写入 Parquet 文件本身，读取端自动生效）：
       DuckDB/pyarrow 读 Parquet 时，row group 内如无目标值则自动跳过 —— 对 LLM 透明。

    2. **Categorical Dtype**（低基数列 → 整数编码）：
       - 优先使用 manifest 的 Index Agent 建议（categorical_hint），
         强制将这些列转为 category，用于 df[df["col"]==val] 的 2-10x 加速。
       - 若无 hint，退回到自动检测（distinct/total < 0.5 且 distinct < 1000）。

    3. **Pandas Index Col**（manifest 建议的 PK/主 FK 列）：
       set_index 后 .loc[val] 为 O(1) 哈希查找，替代 O(n) 扫描。
       注意：set_index 后该列不再是普通 df["col"]，LLM 需通过 df.index 访问。
       因此默认不 set_index，只把建议存在 df.attrs["suggested_index_col"] 供参考。

    LLM 直接写 df[df["col"]==val]，不感知任何底层实现细节。
    """
    import pyarrow.parquet as pq
    table = pq.read_table(parquet_path)
    df = table.to_pandas()

    total = len(df)
    if total == 0:
        return df

    # ── 强制 categorical（来自 Index Agent 建议）──────────────────────────
    forced_cat = set(categorical_hint or [])
    for col in forced_cat:
        if col in df.columns:
            try:
                df[col] = df[col].astype("category")
            except Exception:
                pass

    # ── 自动 categorical（对未被显式建议的列做兜底检测）──────────────────
    for col in df.columns:
        if col in forced_cat:
            continue
        dtype = df[col].dtype
        if dtype == object or str(dtype).startswith("string"):
            n_unique = df[col].nunique(dropna=False)
            if n_unique < 1000 and (n_unique / total) < 0.5:
                try:
                    df[col] = df[col].astype("category")
                except Exception:
                    pass
        elif str(dtype) in ("int64", "Int64", "int32"):
            n_unique = df[col].nunique(dropna=False)
            if n_unique < 500 and (n_unique / total) < 0.3:
                try:
                    df[col] = df[col].astype("category")
                except Exception:
                    pass

    # ── pandas_index_col: 仅存入 attrs 供参考，不 set_index ────────────────────
    # set_index(drop=False) 会导致列名和 index name 同为该列名，
    # pandas merge(on=col) 会报 "is both an index level and a column label" ValueError。
    # 因此仅记录建议，不改变 DataFrame 结构，LLM 正常通过 df["col"] 访问。
    if pandas_index_col and pandas_index_col in df.columns:
        df.attrs["suggested_index_col"] = pandas_index_col

    return df


def _load_parquet_table_to_pandas(
    parquet_dir: str,
    table: str,
    partition_filter: Optional[Dict[str, Any]] = None,
    categorical_hint: Optional[List[str]] = None,
    pandas_index_col: Optional[str] = None,
) -> Optional[Any]:
    """将单文件表或 Hive 分区表整表加载为 pandas DataFrame（供 execute_python 使用）。"""
    single = os.path.join(parquet_dir, f"{table}.parquet")
    part_dir = os.path.join(parquet_dir, table)
    if os.path.isfile(single):
        try:
            return _parquet_to_pandas_optimized(
                single,
                categorical_hint=categorical_hint,
                pandas_index_col=pandas_index_col,
            )
        except Exception:
            # Fallback path for environments without pyarrow or with file-level read issues.
            p_sql = single.replace("'", "''")
            con = duckdb.connect()
            try:
                df = con.execute(f"SELECT * FROM read_parquet('{p_sql}')").fetchdf()
            finally:
                con.close()
            if df is None or df.empty:
                return df
            for col in (categorical_hint or []):
                if col in df.columns:
                    try:
                        df[col] = df[col].astype("category")
                    except Exception:
                        pass
            if pandas_index_col and pandas_index_col in df.columns:
                df.attrs["suggested_index_col"] = pandas_index_col
            return df
    if os.path.isdir(part_dir):
        glob_path = os.path.join(part_dir, "**", "*.parquet")
        glob_sql = glob_path.replace("'", "''")
        con = duckdb.connect()
        try:
            rp = f"read_parquet('{glob_sql}', hive_partitioning=true)"
            where_clause = ""
            if partition_filter and isinstance(partition_filter, dict):
                conds = [f'"{k}" = {repr(v)}' for k, v in partition_filter.items()]
                if conds:
                    where_clause = " WHERE " + " AND ".join(conds)
            df = con.execute(f"SELECT * FROM {rp}{where_clause}").fetchdf()
        finally:
            con.close()
        if df is None or df.empty:
            return df
        # 与 _parquet_to_pandas_optimized 保持一致的 categorical/index 优化
        for col in (categorical_hint or []):
            if col in df.columns:
                try:
                    df[col] = df[col].astype("category")
                except Exception:
                    pass
        if pandas_index_col and pandas_index_col in df.columns:
            df.attrs["suggested_index_col"] = pandas_index_col
        return df
    return None


def _fast_single_parquet_row_count(parquet_path: str) -> int:
    """Fast row count from parquet metadata without scanning full data."""
    try:
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(parquet_path)
        return int(getattr(getattr(pf, "metadata", None), "num_rows", -1) or -1)
    except Exception:
        return -1


def _read_parquet_table(
    parquet_dir: str,
    table: str,
    partition_filter: Optional[Dict[str, Any]] = None,
    manifest: Optional[Dict] = None,
    sample_limit: int = 100,
) -> Dict[str, Any]:
    """
    读单文件表或 Hive 分区表。partition_filter 用于只读部分分区（如 {"p_date_year": 2020}），
    可减少 I/O；仅对分区表有效，列名与 manifest.table_ops PARTITION 一致（p_<col>_year 等）。
    """
    single = os.path.join(parquet_dir, f"{table}.parquet")
    part_dir = os.path.join(parquet_dir, table)
    if os.path.isfile(single):
        # 单文件表
        p_sql = single.replace("'", "''")
        con = duckdb.connect()
        try:
            row_count = _fast_single_parquet_row_count(single)
            df = con.execute(f"SELECT * FROM read_parquet('{p_sql}') LIMIT {int(sample_limit)}").fetchdf()
            rows = df.to_dict(orient="records")
            return {
                "ok": True,
                "table": table,
                "row_count": row_count,
                "sample": rows,
                "read_mode": "sample",
                "sample_limit": int(sample_limit),
                "sampled": row_count > len(rows) if row_count >= 0 else True,
                "note": "read_table returns sampled rows; execute_python may use full parquet cache loaded by agent.",
            }
        finally:
            con.close()
    if os.path.isdir(part_dir):
        # Hive 分区表：glob + hive_partitioning，可选 partition_filter 做分区裁剪
        glob_path = os.path.join(part_dir, "**", "*.parquet")
        glob_sql = glob_path.replace("'", "''")
        con = duckdb.connect()
        try:
            rp = f"read_parquet('{glob_sql}', hive_partitioning=true)"
            where_clause = ""
            if partition_filter and isinstance(partition_filter, dict):
                conds = [f"{k} = {repr(v)}" for k, v in partition_filter.items()]
                if conds:
                    where_clause = " WHERE " + " AND ".join(conds)
            # Avoid expensive full scans in read_table path for partitioned datasets.
            row_count = -1
            df = con.execute(f"SELECT * FROM {rp}{where_clause} LIMIT {int(sample_limit)}").fetchdf()
            rows = df.to_dict(orient="records")
            return {
                "ok": True,
                "table": table,
                "row_count": row_count,
                "sample": rows,
                "read_mode": "sample",
                "sample_limit": int(sample_limit),
                "sampled": True,
                "note": "read_table returns sampled rows; execute_python may use full parquet cache loaded by agent.",
            }
        finally:
            con.close()
    return {"ok": False, "error": f"parquet not found: {single} or dir {part_dir}"}


def _describe_parquet_table(parquet_dir: str, table: str) -> Dict[str, Any]:
    p = os.path.join(parquet_dir, f"{table}.parquet")
    if not os.path.exists(p):
        return {"ok": False, "error": f"parquet not found: {p}"}
    con = duckdb.connect()
    try:
        p_sql = p.replace("'", "''")
        row_count = _fast_single_parquet_row_count(p)
        info = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{p_sql}')").fetchall()
        cols = [r[0] for r in info]
        null_counts: Dict[str, int] = {}
        if cols:
            null_expr = ", ".join(
                f'SUM(CASE WHEN "{c}" IS NULL THEN 1 ELSE 0 END) AS "{c}"' for c in cols
            )
            null_row = con.execute(f"SELECT {null_expr} FROM read_parquet('{p_sql}')").fetchone()
            if null_row:
                null_counts = {c: int(null_row[i] or 0) for i, c in enumerate(cols)}
        df = con.execute(f"SELECT * FROM read_parquet('{p_sql}') LIMIT 5").fetchdf()
        return {
            "ok": True,
            "table": table,
            "row_count": row_count,
            "null_counts": {k: int(v) for k, v in null_counts.items()},
            "sample": df.to_dict(orient="records"),
        }
    finally:
        con.close()


def tool_suggest_olap(
    workload_summary: str,
    schema: Optional[str] = None,
    db_path: Optional[str] = None,
    model: str = "gpt-5-ca",
) -> Dict[str, Any]:
    """
    基于 schema 和 workload，使用 LLM 生成完整 OLAP 建议：
    - Layout: sort/partition（Python 代码）
    - Index: CREATE INDEX（SQL）或 pandas 优化
    - Materialized Views: 预聚合/预 JOIN（SQL 或 Python）
    """
    if schema is None and db_path:
        res = tool_get_schema(db_path=db_path, spider_tables=None, db_id=None)
        if res.get("ok"):
            schema = res.get("schema", "")
        else:
            schema = "Schema not available."
    if not schema:
        schema = "No schema provided."
    parsed, reasoning = suggest_olap_llm(schema=schema, workload_summary=workload_summary or "", model=model)
    formatted = format_olap_suggestion(parsed)
    return {
        "ok": True,
        "suggestion": formatted,
        "reason": reasoning,
        "parsed": parsed,
    }


# -----------------------------
# get_code_template 工具
# -----------------------------

_CODE_TEMPLATES = {
    "INTERSECT": """\
# Pattern: INTERSECT — items that satisfy BOTH conditions
# "Find X that are both A and B"
a_set = set(df[cond_A]['col'])
b_set = set(df[cond_B]['col'])
result = list(a_set & b_set)

# Multi-table variant (e.g. students who have both a cat and a dog):
has_cat = set(df_hp.merge(df_pets[df_pets.PetType=='cat'], on='PetID')['StuID'])
has_dog = set(df_hp.merge(df_pets[df_pets.PetType=='dog'], on='PetID')['StuID'])
result = df_student[df_student.StuID.isin(has_cat & has_dog)]['Fname'].tolist()
""",
    "NOT_IN": """\
# Pattern: NOT IN — items in X that have no matching row in Y
# "Find X with no Y" / "X that never appear in Y"
ids_in_y = df_y['id_col'].unique()
result = df_x[~df_x['id_col'].isin(ids_in_y)][['name_col']].to_dict('records')
""",
    "EXCEPT": """\
# Pattern: EXCEPT — remove a subset from a base set
# "Find X except those in subset S"
ids_in_subset = df_subset['id_col'].unique()
result = df_x[~df_x['id_col'].isin(ids_in_subset)][['name_col']].to_dict('records')
""",
    "UNION": """\
# Pattern: UNION — combine two sets (distinct)
# "Find X or Y" / "all items that are either A or B"
set_a = set(df[cond_A]['col'])
set_b = set(df[cond_B]['col'])
result = list(set_a | set_b)

# If you need full rows:
result = pd.concat([df_a[cols], df_b[cols]]).drop_duplicates().to_dict('records')
""",
    "GROUP_AGG": """\
# Pattern: GROUP + AGGREGATE — count/sum/avg per group
# "For each X, find the total/max/min/count of Y"
result = df.groupby('group_col').agg(
    count_col=('value_col', 'count'),
    total_col=('value_col', 'sum'),
    max_col=('value_col', 'max'),
).reset_index().to_dict('records')

# With filter after aggregation:
agg = df.groupby('group_col')['value_col'].count().reset_index(name='cnt')
result = agg[agg['cnt'] > threshold].to_dict('records')
""",
    "NESTED_AGG": """\
# Pattern: NESTED AGG — aggregate of aggregates (e.g. max of counts)
# "Find the group with the most/least items"
counts = df.groupby('group_col')['item_col'].count()
result = counts.idxmax()              # group with most items
# or:
top_group = counts.nlargest(1).reset_index()
result = top_group.to_dict('records')
""",
}


def tool_get_code_template(pattern: str) -> Dict[str, Any]:
    """返回指定查询模式的 pandas 代码模板"""
    key = pattern.upper().strip()
    if key in _CODE_TEMPLATES:
        return {"ok": True, "result": _CODE_TEMPLATES[key].strip()}
    valid = ", ".join(_CODE_TEMPLATES.keys())
    return {"ok": False, "error": f"Unknown pattern '{pattern}'. Valid patterns: {valid}"}


# -----------------------------
# OpenAI function calling 格式
# -----------------------------


def get_tool_definitions(sql_free: bool = True) -> List[Dict]:
    """
    返回 OpenAI function calling 格式的工具定义（2Code 精简集）。

    保留：
      - get_schema     : 详细 schema 的 fallback（schema 预注入失败或需要更多细节时）
      - describe_table : 数据质量检查（null counts、行数），edge case 时有用
      - read_table     : 核心：将表加载为 df_<name>
      - execute_python : 核心：pandas 分析

    移除（2Code 执行期间无用）：
      - list_tables    : schema 已预注入 user message，完全冗余
      - get_sample     : read_table 已含样本，重复
      - suggest_olap   : workload 分析工具，与查询执行无关
    """
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_schema",
                "description": (
                    "Get detailed database schema (tables, columns, types). "
                    "Schema is already provided in the user message—only call this if you need "
                    "more detail or the schema hint seems incomplete."
                ),
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_join_path",
                "description": (
                    "Suggest join keys/path using SQLite foreign keys plus column-name matching fallback. "
                    "Use this before writing multi-table JOIN/merge logic."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "tables": {
                            "anyOf": [
                                {"type": "array", "items": {"type": "string"}},
                                {"type": "string"},
                            ],
                            "description": "Optional. Candidate tables (array or comma-separated string).",
                        },
                        "start_table": {"type": "string", "description": "Optional source table for path search."},
                        "end_table": {"type": "string", "description": "Optional target table for path search."},
                        "max_hops": {"type": "integer", "description": "Optional max join hops (default 3)."},
                        "top_k": {"type": "integer", "description": "Optional number of suggestions (default 8)."},
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read_table",
                "description": (
                    "Load a table into memory as df_<tablename>. Returns sampled rows by default. "
                    "Call once per table; the DataFrame persists across all subsequent execute_python calls. "
                    "You can set mode='sample'|'full' (full only for SQLite and blocked for very large tables). "
                    "You can set limit to control sample size. "
                    "For partitioned tables (schema shows 'partitioned by p_<col>_year' etc.), pass partition_filter to load only needed partitions (e.g. {\"p_order_date_year\": 2020}) and save I/O."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "table": {"type": "string", "description": "Exact table name as shown in the schema"},
                        "mode": {
                            "type": "string",
                            "description": "Optional. 'sample' (default) or 'full'. Use 'full' only when exact full-table computation is necessary.",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Optional. Sample row limit when mode='sample' (default 100, max 5000).",
                        },
                        "partition_filter": {
                            "type": "object",
                            "description": "Optional. For Hive-partitioned tables only: load just these partitions, e.g. {\"p_date_year\": 2020}. Keys are partition column names from schema HINT.",
                        },
                    },
                    "required": ["table"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "describe_table",
                "description": (
                    "Get table stats: row count and null counts per column. "
                    "Use only when you need to check data quality (e.g., decide how to handle nulls). "
                    "Skip if you plan to call read_table anyway."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "table": {"type": "string", "description": "Table name"},
                    },
                    "required": ["table"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "execute_python",
                "description": (
                    "Execute Python/pandas code. Available variables: df_<tablename> for each loaded table "
                    "(e.g. df_singer, df_concert). "
                    "NOT stateful between calls—put ALL computation in ONE call. "
                    "Always end with: result = <your_answer>"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {"type": "string", "description": "Python/pandas code. Must set result=<answer>."},
                    },
                    "required": ["code"],
                },
            },
        },
    ]
    tools.append({
        "type": "function",
        "function": {
            "name": "get_code_template",
            "description": (
                "Get a ready-to-use pandas code template for a specific set/aggregation pattern. "
                "Call this BEFORE writing execute_python code when you need to implement: "
                "INTERSECT (both A and B), NOT_IN (X with no Y), EXCEPT (X minus subset), "
                "UNION (A or B), GROUP_AGG (group+aggregate), NESTED_AGG (max of counts, etc.). "
                "Returns a copy-and-adapt template."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "enum": ["INTERSECT", "NOT_IN", "EXCEPT", "UNION", "GROUP_AGG", "NESTED_AGG"],
                        "description": "The query pattern you need a template for",
                    }
                },
                "required": ["pattern"],
            },
        },
    })
    if not sql_free:
        tools.insert(0, {
            "type": "function",
            "function": {
                "name": "query_sql",
                "description": "Execute SQL on the database (fallback).",
                "parameters": {"type": "object", "properties": {"sql": {"type": "string"}}, "required": ["sql"]},
            },
        })
    return tools
