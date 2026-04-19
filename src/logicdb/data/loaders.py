#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据加载与转换：Spider/BIRD 数据集
- 从 SQLite 读取（DuckDB 直接 attach）
- 可选：导出为 Parquet，作为分析友好载体
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import duckdb
import pandas as pd

from logicdb.config import (
    DEFAULT_BIRD_ROOT,
    DEFAULT_SPIDER_ROOT,
    DEFAULT_TABFACT_ROOT,
    DEFAULT_WTQ_ROOT,
)


# 默认路径
DEFAULT_SPIDER_DB = os.path.join(DEFAULT_SPIDER_ROOT, "database")
DEFAULT_BIRD_DB = os.path.join(DEFAULT_BIRD_ROOT, "dev_databases")


@dataclass
class TableSchema:
    """单表 schema"""
    table: str
    columns: List[Tuple[str, str]]  # (col_name, col_type)
    primary_keys: List[int] = field(default_factory=list)
    foreign_keys: List[Tuple[int, int]] = field(default_factory=list)


@dataclass
class DBSchema:
    """数据库 schema"""
    db_id: str
    tables: List[TableSchema]
    table_names: List[str]
    column_names: List[Tuple[int, str]]  # (table_idx, col_name)
    column_types: List[str]


def load_spider_tables(tables_path: str) -> Dict[str, Dict]:
    """加载 Spider tables.json，返回 db_id -> schema_dict"""
    with open(tables_path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    return {r["db_id"]: r for r in rows}


def load_spider_samples(dev_path: str, limit: Optional[int] = None) -> List[Dict]:
    """加载 Spider dev/train samples"""
    with open(dev_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if limit:
        return data[:limit]
    return data


def load_bird_samples(dev_path: str, limit: Optional[int] = None) -> List[Dict]:
    """加载 BIRD dev samples"""
    with open(dev_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if limit:
        return data[:limit]
    return data


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
    """Heuristic delimiter detection for WTQ/TabFact tables."""
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


def _read_table_csv(csv_path: str) -> pd.DataFrame:
    delim = _detect_csv_delimiter(csv_path)
    try:
        df = pd.read_csv(csv_path, sep=delim, engine="python")
    except Exception:
        # Last-resort fallback with permissive tokenizer.
        df = pd.read_csv(csv_path, sep=delim, engine="python", on_bad_lines="skip")
    if df is None:
        df = pd.DataFrame()
    if len(df.columns) == 0:
        return pd.DataFrame({"value": []})
    df.columns = _dedup_columns([str(c) for c in df.columns])
    return df


def _coerce_df_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None:
        return pd.DataFrame({"value": []})
    if len(df.columns) == 0:
        return pd.DataFrame({"value": []})
    out = df.copy()
    out.columns = _dedup_columns([str(c) for c in out.columns])
    return out


def _read_html_tables(html_path: str) -> List[pd.DataFrame]:
    try:
        tables = pd.read_html(html_path)
    except Exception:
        return []
    return [_coerce_df_columns(df) for df in tables if df is not None]


def _read_markdown_table(md_path: str) -> pd.DataFrame:
    lines = []
    with open(md_path, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            s = raw.strip()
            if not s:
                continue
            if "|" in s:
                lines.append(s)
    if len(lines) < 2:
        return pd.DataFrame({"value": []})
    header = [c.strip() for c in lines[0].strip("|").split("|")]
    rows = []
    for ln in lines[2:]:
        vals = [c.strip() for c in ln.strip("|").split("|")]
        if len(vals) < len(header):
            vals += [""] * (len(header) - len(vals))
        rows.append(vals[:len(header)])
    return _coerce_df_columns(pd.DataFrame(rows, columns=header))


def _resolve_document_source(base_path: str, source_format: str) -> str:
    fmt = str(source_format or "html").strip().lower()
    p = Path(base_path)
    candidates = []
    if fmt == "html":
        candidates = [
            str(p) + ".html",
            str(p.with_suffix(".html")),
            str(p.with_suffix(p.suffix + ".html")) if p.suffix else str(p) + ".html",
        ]
    elif fmt == "markdown":
        candidates = [
            str(p) + ".md",
            str(p.with_suffix(".md")),
            str(p.with_suffix(p.suffix + ".md")) if p.suffix else str(p) + ".md",
        ]
    for c in candidates:
        if c and os.path.exists(c):
            return c
    return candidates[0] if candidates else base_path


def _maybe_generate_document_carrier(base_path: str, source_format: str) -> str:
    """
    Best-effort carrier generation for html / markdown from an existing CSV source.
    This closes the ingest path when pre-generated carriers are missing.
    """
    fmt = str(source_format or "html").strip().lower()
    src = Path(base_path)
    if fmt not in {"html", "markdown"}:
        return str(src)
    if not src.exists() or src.suffix.lower() != ".csv":
        return str(src)
    try:
        df = pd.read_csv(src)
    except Exception:
        try:
            df = pd.read_csv(src, sep="#", engine="python")
        except Exception:
            return str(src)
    if fmt == "html":
        out = src.with_suffix(src.suffix + ".html")
        if not out.exists():
            out.write_text(df.to_html(index=False), encoding="utf-8")
        return str(out)
    out = src.with_suffix(src.suffix + ".md")
    if not out.exists():
        cols = [str(c) for c in df.columns]
        lines = []
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
        for _, row in df.iterrows():
            vals = [str("" if pd.isna(v) else v).replace("\n", " ") for v in row.tolist()]
            lines.append("| " + " | ".join(vals) + " |")
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(out)


def ensure_document_sqlite(
    source_path: str,
    sqlite_cache_dir: str,
    db_prefix: str,
    table_name: str = "data",
    source_format: str = "html",
) -> str:
    os.makedirs(sqlite_cache_dir, exist_ok=True)
    source_path = _resolve_document_source(source_path, source_format)
    if not os.path.exists(source_path):
        base_guess = source_path.rsplit(".", 1)[0] if source_path.endswith((".html", ".md")) else source_path
        source_path = _maybe_generate_document_carrier(base_guess, source_format)
    src_abs = os.path.abspath(source_path)
    key = hashlib.sha1((src_abs + "::" + source_format).encode("utf-8")).hexdigest()[:20]
    out_db = os.path.join(sqlite_cache_dir, f"{db_prefix}_{key}.sqlite")
    src_mtime = os.path.getmtime(src_abs) if os.path.exists(src_abs) else -1
    if os.path.exists(out_db):
        try:
            if os.path.getmtime(out_db) >= src_mtime:
                return out_db
        except Exception:
            pass
    if not os.path.exists(src_abs):
        raise FileNotFoundError(f"Document source not found: {src_abs}")
    fmt = str(source_format or "html").strip().lower()
    if fmt == "html":
        tables = _read_html_tables(src_abs)
        df = tables[0] if tables else pd.DataFrame({"value": []})
    elif fmt == "markdown":
        df = _read_markdown_table(src_abs)
    else:
        raise ValueError(f"Unsupported document source format: {source_format}")
    conn = sqlite3.connect(out_db)
    try:
        conn.execute(f'DROP TABLE IF EXISTS "{table_name}"')
        df.to_sql(table_name, conn, if_exists="replace", index=False)
        conn.execute("CREATE TABLE IF NOT EXISTS _meta (k TEXT PRIMARY KEY, v TEXT)")
        conn.execute("INSERT OR REPLACE INTO _meta(k, v) VALUES (?, ?)", ("source_path", src_abs))
        conn.execute("INSERT OR REPLACE INTO _meta(k, v) VALUES (?, ?)", ("source_format", fmt))
        conn.execute("INSERT OR REPLACE INTO _meta(k, v) VALUES (?, ?)", ("table_name", table_name))
        conn.commit()
    finally:
        conn.close()
    return out_db


def ensure_csv_sqlite(
    csv_path: str,
    sqlite_cache_dir: str,
    db_prefix: str,
    table_name: str = "data",
) -> str:
    """
    Convert one CSV table into a SQLite DB file (single table) with cache.
    Rebuild only when source CSV is newer than cached sqlite.
    """
    os.makedirs(sqlite_cache_dir, exist_ok=True)
    csv_abs = os.path.abspath(csv_path)
    key = hashlib.sha1(csv_abs.encode("utf-8")).hexdigest()[:20]
    out_db = os.path.join(sqlite_cache_dir, f"{db_prefix}_{key}.sqlite")
    src_mtime = os.path.getmtime(csv_abs) if os.path.exists(csv_abs) else -1

    if os.path.exists(out_db):
        try:
            if os.path.getmtime(out_db) >= src_mtime:
                return out_db
        except Exception:
            pass

    if not os.path.exists(csv_abs):
        raise FileNotFoundError(f"CSV not found: {csv_abs}")

    df = _read_table_csv(csv_abs)
    conn = sqlite3.connect(out_db)
    try:
        conn.execute(f'DROP TABLE IF EXISTS "{table_name}"')
        df.to_sql(table_name, conn, if_exists="replace", index=False)
        conn.execute("CREATE TABLE IF NOT EXISTS _meta (k TEXT PRIMARY KEY, v TEXT)")
        conn.execute("INSERT OR REPLACE INTO _meta(k, v) VALUES (?, ?)", ("source_csv", csv_abs))
        conn.execute("INSERT OR REPLACE INTO _meta(k, v) VALUES (?, ?)", ("table_name", table_name))
        conn.commit()
    finally:
        conn.close()
    return out_db


def _wtq_unescape(text: str) -> str:
    s = str(text or "")
    # order matters: keep list separator handling outside this function.
    s = s.replace("\\p", "|")
    s = s.replace("\\n", "\n")
    s = s.replace("\\\\", "\\")
    return s


def _parse_wtq_target_value(raw: str) -> List[str]:
    if raw is None:
        return []
    parts = str(raw).split("|")
    out = [_wtq_unescape(x).strip() for x in parts]
    return [x for x in out if x != ""]


def load_wtq_samples(
    wtq_root: str,
    split: str = "pristine-unseen-tables",
    limit: Optional[int] = None,
    sqlite_cache_dir: Optional[str] = None,
    backend: str = "csv",
) -> List[Dict]:
    """
    Load WikiTableQuestions split and convert each table CSV to cached sqlite DB.
    Supported split names include training, pristine-unseen-tables, pristine-seen-tables,
    and random-split-seed-*-train/test/dev (as long as corresponding .tsv exists).
    """
    split_name = split[:-4] if split.endswith(".tsv") else split
    tsv_path = os.path.join(wtq_root, "data", f"{split_name}.tsv")
    if not os.path.exists(tsv_path):
        raise FileNotFoundError(f"WTQ split not found: {tsv_path}")
    storage_backend = str(backend or "csv").strip().lower()
    if storage_backend not in {"csv", "sqlite", "html", "markdown"}:
        raise ValueError(f"Unsupported WTQ backend: {backend}. choose from csv/sqlite/html/markdown")
    cache_dir = sqlite_cache_dir or os.path.join(wtq_root, "_sqlite_cache")

    rows: List[Dict] = []
    with open(tsv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            context = str(r.get("context", "")).strip()
            csv_path = os.path.join(wtq_root, context)
            if storage_backend == "sqlite":
                db_path = ensure_csv_sqlite(
                    csv_path=csv_path,
                    sqlite_cache_dir=cache_dir,
                    db_prefix="wtq",
                    table_name="data",
                )
            elif storage_backend in {"html", "markdown"}:
                doc_src = csv_path
                db_path = ensure_document_sqlite(
                    source_path=doc_src,
                    sqlite_cache_dir=cache_dir,
                    db_prefix=f"wtq_{storage_backend}",
                    table_name="data",
                    source_format=storage_backend,
                )
            else:
                db_path = os.path.abspath(csv_path)
            db_key = hashlib.sha1(os.path.abspath(csv_path).encode("utf-8")).hexdigest()[:16]
            rows.append(
                {
                    "id": r.get("id"),
                    "question": r.get("utterance", ""),
                    "context": context,
                    "db_id": f"wtq_{db_key}",
                    "db_path": db_path,
                    "gold_answer": _parse_wtq_target_value(r.get("targetValue", "")),
                    "source_dataset": "wtq",
                    "storage_backend": storage_backend,
                }
            )
            if limit and len(rows) >= limit:
                break
    return rows


def _tabfact_clean_statement(text: str) -> str:
    s = str(text or "")
    # Convert entity-link markers like #entity;idx1,idx2# back to raw text.
    s = re.sub(r"#([^#;]+);-?\d+,-?\d+#", r"\1", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _tabfact_question_from_statement(stmt: str) -> str:
    return (
        "Fact verification over a table. Decide whether the statement is entailed by the table. "
        "Return ONLY 1 (entailed) or 0 (refuted).\n"
        f"Statement: {stmt}"
    )


def load_tabfact_samples(
    tabfact_root: str,
    split: str = "val",
    limit: Optional[int] = None,
    sqlite_cache_dir: Optional[str] = None,
    backend: str = "csv",
) -> List[Dict]:
    """
    Load TabFact split and flatten to per-statement samples.
    split: train | val | test
    """
    split_map = {
        "train": "train_examples.json",
        "val": "val_examples.json",
        "dev": "val_examples.json",
        "test": "test_examples.json",
    }
    split_key = split.lower().strip()
    if split_key not in split_map:
        raise ValueError(f"Unsupported TabFact split: {split}. choose from train/val/test")
    json_path = os.path.join(tabfact_root, "tokenized_data", split_map[split_key])
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"TabFact split not found: {json_path}")

    storage_backend = str(backend or "csv").strip().lower()
    if storage_backend not in {"csv", "sqlite", "html", "markdown"}:
        raise ValueError(f"Unsupported TabFact backend: {backend}. choose from csv/sqlite/html/markdown")
    cache_dir = sqlite_cache_dir or os.path.join(tabfact_root, "_sqlite_cache")
    blob = json.load(open(json_path, "r", encoding="utf-8"))
    rows: List[Dict] = []

    for table_id, payload in blob.items():
        if not isinstance(payload, list) or len(payload) < 2:
            continue
        statements = payload[0] if isinstance(payload[0], list) else []
        labels = payload[1] if isinstance(payload[1], list) else []
        caption = payload[2] if len(payload) > 2 else ""
        csv_path = os.path.join(tabfact_root, "data", "all_csv", str(table_id))
        if storage_backend == "sqlite":
            db_path = ensure_csv_sqlite(
                csv_path=csv_path,
                sqlite_cache_dir=cache_dir,
                db_prefix="tabfact",
                table_name="data",
            )
        elif storage_backend in {"html", "markdown"}:
            doc_src = csv_path
            db_path = ensure_document_sqlite(
                source_path=doc_src,
                sqlite_cache_dir=cache_dir,
                db_prefix=f"tabfact_{storage_backend}",
                table_name="data",
                source_format=storage_backend,
            )
        else:
            db_path = os.path.abspath(csv_path)
        db_key = hashlib.sha1(os.path.abspath(csv_path).encode("utf-8")).hexdigest()[:16]
        for i, stmt in enumerate(statements):
            if i >= len(labels):
                break
            clean_stmt = _tabfact_clean_statement(stmt)
            try:
                label = int(labels[i])
            except Exception:
                label = 1 if str(labels[i]).strip().lower() in {"1", "true", "entailed"} else 0
            rows.append(
                {
                    "id": f"{table_id}::{i}",
                    "table_id": table_id,
                    "question": _tabfact_question_from_statement(clean_stmt),
                    "statement": clean_stmt,
                    "evidence": f"Table caption: {caption}" if str(caption).strip() else None,
                    "db_id": f"tabfact_{db_key}",
                    "db_path": db_path,
                    "gold_answer": label,
                    "source_dataset": "tabfact",
                    "storage_backend": storage_backend,
                }
            )
            if limit and len(rows) >= limit:
                return rows
    return rows


def build_mschema_from_sqlite(db_path: str, example_num: int = 2) -> str:
    """
    从 SQLite 构建 mschema 格式（与 QBridge SchemaEngine.to_mschema 兼容）。
    格式: # Table: name\n[(col:TYPE, ...), ...]
    用于与 NL2SQL/NL2Code 社区一致，便于论文对齐。
    """
    import sqlite3
    conn = sqlite3.connect(db_path)
    tables = [
        r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
    ]
    lines = []
    fks = []
    for t in tables:
        lines.append(f"# Table: {t}")
        info = conn.execute(f"PRAGMA table_info({t})").fetchall()
        field_lines = []
        for row in info:
            cid, name, dtype, notnull, default, pk = row
            dtype = (dtype or "TEXT").split("(")[0].upper()
            field_line = f"({name}:{dtype}"
            if pk:
                field_line += ", Primary Key"
            try:
                sample = conn.execute(f'SELECT "{name}" FROM "{t}" LIMIT {example_num}').fetchall()
                if sample:
                    ex = [str(s[0])[:30] for s in sample if s[0] is not None]
                    if ex:
                        field_line += f", Examples: [{', '.join(repr(x) for x in ex)}]"
            except Exception:
                pass
            field_line += ")"
            field_lines.append(field_line)
        lines.append("[" + ",\n".join(field_lines) + "]")
        lines.append("")
    if tables:
        try:
            for row in conn.execute(
                "SELECT * FROM pragma_foreign_key_list((SELECT name FROM sqlite_master WHERE type='table' LIMIT 1))"
            ).fetchall():
                pass
        except Exception:
            pass
        # SQLite PRAGMA foreign_key_list(table) returns: id, seq, table, from, to, on_update, on_delete, match
        for t in tables:
            try:
                for row in conn.execute(f"PRAGMA foreign_key_list({t})").fetchall():
                    _, _, ref_table, from_col, to_col = row[:5]
                    fks.append(f"{t}.{from_col}={ref_table}.{to_col}")
            except Exception:
                pass
    if fks:
        lines.append("【Foreign keys】")
        lines.extend(fks)
    conn.close()
    return "\n".join(lines).strip()


def spider_schema_to_str(schema_dict: Dict) -> str:
    """将 Spider tables.json 中单条 schema 转为可读字符串"""
    lines = []
    table_names = schema_dict.get("table_names_original", schema_dict.get("table_names", []))
    column_names_orig = schema_dict.get("column_names_original", [])
    column_types = schema_dict.get("column_types", [])
    for ti, tname in enumerate(table_names):
        col_list = []
        for item, ct in zip(column_names_orig, column_types):
            tidx = item[0]
            cname = item[1]
            if tidx == ti:
                col_list.append(f"{cname} ({ct})")
        if col_list:
            lines.append(f"Table {tname}:\n  " + "\n  ".join(col_list))
    return "\n\n".join(lines) if lines else "No schema"


def get_db_path(db_id: str, dataset: str, spider_root: str, bird_root: str) -> str:
    """获取 SQLite 数据库路径"""
    if dataset == "spider":
        return os.path.join(spider_root, "database", db_id, f"{db_id}.sqlite")
    if dataset == "bird":
        return os.path.join(bird_root, "dev_databases", db_id, f"{db_id}.sqlite")
    if dataset in {"wtq", "tabfact"}:
        raise ValueError(
            f"Dataset '{dataset}' uses per-table CSV->SQLite conversion. "
            "Please provide sample['db_path'] from loader or pass --db_path explicitly."
        )
    raise ValueError(f"Unknown dataset: {dataset}")


def attach_sqlite_to_duckdb(con: duckdb.DuckDBPyConnection, db_path: str, alias: str = "db") -> None:
    """将 SQLite 挂载到 DuckDB。Spider/BIRD 可能有类型不一致，使用 sqlite_all_varchar 兼容"""
    abs_path = os.path.abspath(db_path).replace("\\", "/")
    try:
        con.execute("SET sqlite_all_varchar=true;")  # 兼容 Spider/BIRD 的类型不一致
    except Exception:
        pass
    try:
        con.execute(f"ATTACH '{abs_path}' AS {alias} (TYPE SQLITE);")
    except Exception:
        con.execute(f"ATTACH '{abs_path}' AS {alias};")


def _normalize_empty_strings_for_parquet(df, conn, tname: str):
    """
    SQLite 中数值列可能存空字符串 ''，写 Parquet 时 DuckDB 会报 Could not convert string '' to INT32/DOUBLE。
    将数值列中的 '' 转为 NaN，避免转换失败。
    """
    import pandas as pd
    try:
        info = conn.execute(f'PRAGMA table_info("{tname}")').fetchall()
    except Exception:
        info = []
    # (cid, name, type, notnull, dflt_value, pk)
    numeric_cols = []
    for row in info:
        if not row[2]:
            continue
        t = str(row[2]).lower()
        if any(k in t for k in ("int", "real", "float", "double", "numeric", "decimal")):
            numeric_cols.append(row[1])
    if not numeric_cols:
        numeric_cols = [c for c in df.columns if str(getattr(df[c].dtype, "name", df[c].dtype)) in ("int64", "float64", "Int64", "Float64")]
    for col in numeric_cols:
        if col not in df.columns:
            continue
        if df[col].dtype == object or str(df[col].dtype) == "string":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].replace("", pd.NA)
    return df


def sqlite_to_parquet(
    sqlite_path: str,
    parquet_dir: str,
    db_id: str,
) -> Dict[str, str]:
    """
    将 SQLite 数据库的所有表导出为 Parquet 文件。
    返回: {table_name: parquet_path}
    注意：不要依赖 DuckDB 的 sqlite extension（环境里可能不可用）。
    数值列中的空字符串 '' 会先转为 NULL 再写入，避免 DuckDB 报错。
    """
    import sqlite3
    import pandas as pd

    os.makedirs(parquet_dir, exist_ok=True)
    conn = sqlite3.connect(sqlite_path)
    # Some Spider DBs contain non-UTF8 bytes in TEXT columns (e.g. wta_1.players.last_name).
    # Decode with replacement instead of failing the entire conversion job.
    conn.text_factory = lambda b: b.decode("utf-8", errors="replace")
    try:
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()]
        result: Dict[str, str] = {}
        con = duckdb.connect()
        try:
            for tname in tables:
                out_path = os.path.join(parquet_dir, f"{tname}.parquet")
                try:
                    df = pd.read_sql_query(f'SELECT * FROM \"{tname}\"', conn)
                except Exception:
                    df = pd.read_sql_query(f"SELECT * FROM {tname}", conn)
                df = _normalize_empty_strings_for_parquet(df, conn, tname)
                con.register("tmp_df", df)
                try:
                    out_path_sql = out_path.replace("'", "''")
                    con.execute(f"COPY tmp_df TO '{out_path_sql}' (FORMAT parquet);")
                finally:
                    con.unregister("tmp_df")
                result[tname] = out_path
        finally:
            con.close()
        return result
    finally:
        conn.close()


def get_schema_from_duckdb(con: duckdb.DuckDBPyConnection, schema_or_db: str = "db") -> str:
    """从 DuckDB 连接获取 schema 字符串（已 attach SQLite 或读取 Parquet 后）"""
    try:
        rows = con.execute(f"""
            SELECT table_name, column_name, data_type
            FROM information_schema.columns
            WHERE table_schema = '{schema_or_db}'
            ORDER BY table_name, ordinal_position
        """).fetchall()
    except Exception:
        # 可能 schema 名不同
        rows = con.execute("""
            SELECT table_schema, table_name, column_name, data_type
            FROM information_schema.columns
            WHERE table_schema NOT IN ('information_schema', 'pg_catalog')
            ORDER BY table_name, ordinal_position
        """).fetchall()
    by_table = {}
    for r in rows:
        if len(r) == 3:
            tname, cname, dtype = r
        else:
            _, tname, cname, dtype = r
        by_table.setdefault(tname, []).append(f"  {cname} ({dtype})")
    lines = [f"Table {t}:\n" + "\n".join(cols) for t, cols in sorted(by_table.items())]
    return "\n\n".join(lines)


def get_table_names(con: duckdb.DuckDBPyConnection, schema_or_db: str = "db") -> List[str]:
    """获取已挂载数据库的表名列表"""
    try:
        rows = con.execute(f"""
            SELECT DISTINCT table_name FROM information_schema.tables
            WHERE table_schema = '{schema_or_db}' AND table_type = 'BASE TABLE'
        """).fetchall()
    except Exception:
        rows = con.execute("""
            SELECT DISTINCT table_name FROM information_schema.tables
            WHERE table_schema NOT IN ('information_schema', 'pg_catalog')
        """).fetchall()
    return [r[0] for r in rows]


def run_sql_on_sqlite(con: duckdb.DuckDBPyConnection, sql: str, alias: str = "db") -> Tuple[Optional[List], Optional[str]]:
    """
    在挂载的 SQLite 上执行 SQL。
    返回: (rows, error) 成功时 error=None
    """
    try:
        # SQLite 表在 attach 后通过 alias.table 访问
        # DuckDB 的 ATTACH sqlite 会创建 schema，表在 db.main.table
        rows = con.execute(sql).fetchall()
        col_names = [d[0] for d in con.description] if con.description else []
        result = [dict(zip(col_names, r)) for r in rows] if col_names else rows
        return (result, None)
    except Exception as e:
        return (None, str(e))
