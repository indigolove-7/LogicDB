#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import os
import shutil
from typing import Dict, List, Optional, Any

import duckdb


class PlanValidationError(ValueError):
    pass


def qident(x: str) -> str:
    return '"' + x.replace('"', '""') + '"'


def qpath(p: str) -> str:
    return "'" + p.replace("'", "''") + "'"


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def schema_exists(con: duckdb.DuckDBPyConnection, schema: str) -> bool:
    rows = con.execute(
        "SELECT 1 FROM information_schema.schemata WHERE lower(schema_name)=lower(?) LIMIT 1",
        [schema],
    ).fetchall()
    return bool(rows)


def relation_exists(con: duckdb.DuckDBPyConnection, schema: str, name: str) -> bool:
    rows = con.execute(
        "SELECT 1 FROM information_schema.tables WHERE lower(table_schema)=lower(?) AND lower(table_name)=lower(?) LIMIT 1",
        [schema, name],
    ).fetchall()
    if rows:
        return True
    rows = con.execute(
        "SELECT 1 FROM information_schema.views WHERE lower(table_schema)=lower(?) AND lower(table_name)=lower(?) LIMIT 1",
        [schema, name],
    ).fetchall()
    return bool(rows)


def drop_schema_cascade(con: duckdb.DuckDBPyConnection, schema: str) -> None:
    con.execute(f"DROP SCHEMA IF EXISTS {qident(schema)} CASCADE;")


def bucket_expr(col_expr: str, n: int) -> str:
    """Generate hash bucket expression for partitioning"""
    return f"(hash({col_expr}) % {n})::INT"


def _partition_exprs(pby: List[Dict[str, Any]]) -> tuple[list[str], list[str], list[str]]:
    """
    Return (derived_cols_sql, part_cols_idents, exclude_cols_names)
    matching PARTITION operator behavior.
    """
    derived_cols: list[str] = []
    part_cols: list[str] = []
    exclude_cols: list[str] = []
    for item in pby:
        col = item["col"]
        tf = item["transform"]
        param = item.get("param")

        if tf == "identity":
            pname = f"p_{col}"
            derived_cols.append(f"{qident(col)} AS {qident(pname)}")
        elif tf == "year":
            pname = f"p_{col}_year"
            derived_cols.append(f"year({qident(col)}) AS {qident(pname)}")
        elif tf == "month":
            pname = f"p_{col}_month"
            derived_cols.append(f"month({qident(col)}) AS {qident(pname)}")
        elif tf == "day":
            pname = f"p_{col}_day"
            derived_cols.append(f"day({qident(col)}) AS {qident(pname)}")
        elif tf == "bucket":
            if not param:
                raise PlanValidationError("PARTITION bucket transform requires param=N")
            pname = f"p_{col}_bkt"
            derived_cols.append(f"{bucket_expr(qident(col), int(param))} AS {qident(pname)}")
        elif tf == "truncate":
            if not param:
                raise PlanValidationError("PARTITION truncate transform requires param=width")
            pname = f"p_{col}_trunc"
            derived_cols.append(f"(floor({qident(col)} / {int(param)}) * {int(param)})::BIGINT AS {qident(pname)}")
        else:
            raise PlanValidationError(f"unknown partition transform: {tf}")

        part_cols.append(qident(pname))
        exclude_cols.append(pname)
    return derived_cols, part_cols, exclude_cols


class LayoutContext:
    """
    Context for layout transformation execution.
    Manages connection, backend type, schema, and file paths.
    """
    def __init__(self, con: duckdb.DuckDBPyConnection, plan: Dict[str, Any]):
        self.con = con
        self.plan = plan
        self.variant_name = plan.get("variant_name", "unnamed")
        self.variant_schema = plan.get("variant_schema", "layout_variant")
        self.backend = plan.get("backend", "parquet_hive")
        self.work_dir = plan.get("work_dir", ".")
        self.parquet_compression = plan.get("parquet_compression", "zstd")
        self.parquet_row_group_size = plan.get("parquet_row_group_size", 250000)
        self.baseline_schema = plan.get("baseline_schema", "main")  # Source of truth for full schema
        # If true, ops should always read source tables from baseline_schema (not from variant schema),
        # useful for multi-period/dynamic reorganization to avoid cascading layouts.
        self.always_from_baseline = bool(plan.get("always_from_baseline", False))
        self.created_paths = []
        # Track partitioning specs for tables created by PARTITION so later ops can preserve it.
        # table -> partition_by list
        self.partitioning: Dict[str, List[Dict[str, Any]]] = {}
        
        # Create schema if needed
        if not schema_exists(con, self.variant_schema):
            con.execute(f"CREATE SCHEMA {qident(self.variant_schema)};")
    
    def variant_dir(self) -> str:
        """Get the directory for this variant's parquet files"""
        return os.path.join(self.work_dir, self.variant_name)
    
    def resolve_relation(self, name: str) -> str:
        """
        Resolve a relation name to a fully qualified reference.
        First checks variant schema, then falls back to main schema.
        """
        if (not self.always_from_baseline) and relation_exists(self.con, self.variant_schema, name):
            return f"{qident(self.variant_schema)}.{qident(name)}"
        elif relation_exists(self.con, self.baseline_schema, name):
            return f"{qident(self.baseline_schema)}.{qident(name)}"
        else:
            # Assume it exists in variant schema (will be created)
            return f"{qident(self.variant_schema)}.{qident(name)}"
    
    def create_or_replace_parquet_view(
        self, 
        view_name: str, 
        path: str, 
        hive_partitioning: bool = False,
        exclude_cols: Optional[List[str]] = None
    ) -> None:
        """Create a view over parquet file(s)"""
        if exclude_cols:
            excl = ", ".join(qident(c) for c in exclude_cols)
            select = f"* EXCLUDE ({excl})"
        else:
            select = "*"
        
        hive_opt = "hive_partitioning=true" if hive_partitioning else None
        
        if hive_opt:
            rp = f"read_parquet({qpath(path)}, {hive_opt})"
        else:
            rp = f"read_parquet({qpath(path)})"
        self.con.execute(
            f"CREATE OR REPLACE VIEW {qident(self.variant_schema)}.{qident(view_name)} AS "
            f"SELECT {select} FROM {rp};"
        )


def op_tune_file(ctx: LayoutContext, op: Dict[str, Any]) -> None:
    """
    TUNE_FILE operator: adjust file-level parameters like row_group_size.
    This rewrites the table with new row_group_size setting.
    """
    source = op.get("source") or op.get("table")
    dest = op.get("dest") or source
    row_group_size = op.get("row_group_size")
    
    if not source or not dest:
        raise PlanValidationError("TUNE_FILE requires source/table, dest(optional)")
    
    if row_group_size is not None:
        ctx.parquet_row_group_size = int(row_group_size)
    
    src_ref = ctx.resolve_relation(source)
    
    if ctx.backend == "duckdb_internal":
        # For internal tables, just recreate the table (no row_group tuning)
        ctx.con.execute(
            f"CREATE OR REPLACE TABLE {qident(ctx.variant_schema)}.{qident(dest)} AS "
            f"SELECT * FROM {src_ref};"
        )
        return
    
    # parquet_hive: rewrite with new row_group_size.
    # If dest is partitioned, preserve partitioning layout.
    variant_dir = ctx.variant_dir()
    ensure_dir(variant_dir)

    pby = ctx.partitioning.get(dest)
    if pby:
        # Use temp directory to avoid deleting source data
        out_dir_temp = os.path.join(variant_dir, f"{dest}_temp_tune")
        out_dir_final = os.path.join(variant_dir, dest)
        
        if os.path.exists(out_dir_temp):
            shutil.rmtree(out_dir_temp)
        ensure_dir(out_dir_temp)
        ctx.created_paths.append(out_dir_final)

        derived_cols, part_cols, exclude_cols = _partition_exprs(pby)
        select_sql = "SELECT *"
        if derived_cols:
            select_sql += ", " + ", ".join(derived_cols)
        select_sql += f" FROM {src_ref}"

        ctx.con.execute(
            f"COPY ({select_sql}) TO {qpath(out_dir_temp)} "
            f"(FORMAT parquet, PARTITION_BY ({', '.join(part_cols)}), "
            f"COMPRESSION {ctx.parquet_compression}, ROW_GROUP_SIZE {ctx.parquet_row_group_size});"
        )
        
        # Atomically replace
        if os.path.exists(out_dir_final):
            shutil.rmtree(out_dir_final)
        os.rename(out_dir_temp, out_dir_final)
        
        glob = os.path.join(out_dir_final, "**", "*.parquet")
        ctx.create_or_replace_parquet_view(dest, glob, hive_partitioning=True, exclude_cols=exclude_cols)
        return

    out_file = os.path.join(variant_dir, f"{dest}.parquet")
    ensure_dir(os.path.dirname(out_file))
    if os.path.exists(out_file):
        os.remove(out_file)
    ctx.created_paths.append(out_file)
    
    ctx.con.execute(
        f"COPY (SELECT * FROM {src_ref}) TO {qpath(out_file)} "
        f"(FORMAT parquet, COMPRESSION {ctx.parquet_compression}, ROW_GROUP_SIZE {ctx.parquet_row_group_size});"
    )
    ctx.create_or_replace_parquet_view(dest, out_file, hive_partitioning=False)


# PROJECT operator has been REMOVED
# 
# Reason: Data layout reorganization should NOT delete data.
# 
# Core principle: Layout optimization = changing physical data organization (partitioning, 
# clustering, sorting) to accelerate queries, NOT removing columns.
# 
# PROJECT physically deletes columns, which:
# 1. Causes permanent data loss
# 2. Is incompatible with multi-period reorganization
# 3. Goes against the purpose of "layout" optimization
# 
# Parquet columnar format already provides efficient column pruning at query time,
# skipping unused columns automatically without data loss.
#
# If column pruning is truly needed, it should be done at the application/view level,
# not in the physical layout layer.


def op_cluster_sort(ctx: LayoutContext, op: Dict[str, Any]) -> None:
    source = op.get("source") or op.get("table")
    dest = op.get("dest") or source
    keys = op.get("keys")
    if not source or not dest or not keys:
        raise PlanValidationError("CLUSTER_SORT requires source/table, keys")

    src_ref = ctx.resolve_relation(source)
    order_parts = []
    for k in keys:
        col = qident(k["col"])
        order = (k.get("order") or "asc").upper()
        order_parts.append(f"{col} {order}")
    order_by = ", ".join(order_parts)

    if ctx.backend == "duckdb_internal":
        ctx.con.execute(
            f"CREATE OR REPLACE TABLE {qident(ctx.variant_schema)}.{qident(dest)} AS "
            f"SELECT * FROM {src_ref} ORDER BY {order_by};"
        )
        return

    variant_dir = ctx.variant_dir()
    ensure_dir(variant_dir)

    pby = ctx.partitioning.get(source) or ctx.partitioning.get(dest)
    if pby:
        # Preserve partitioning and sort within each partition by ordering on partition exprs first.
        # NOTE: We write to a TEMPORARY directory first, then rename atomically to avoid
        # deleting the source data if source == dest (same table, chained operations).
        out_dir_temp = os.path.join(variant_dir, f"{dest}_temp_cluster")
        out_dir_final = os.path.join(variant_dir, dest)
        
        if os.path.exists(out_dir_temp):
            shutil.rmtree(out_dir_temp)
        ensure_dir(out_dir_temp)
        ctx.created_paths.append(out_dir_final)

        derived_cols, part_cols, exclude_cols = _partition_exprs(pby)
        select_sql = "SELECT *"
        if derived_cols:
            select_sql += ", " + ", ".join(derived_cols)
        select_sql += f" FROM {src_ref}"
        order_sql = ", ".join(part_cols + [p for p in order_parts])

        # Write to temporary directory
        ctx.con.execute(
            f"COPY ({select_sql} ORDER BY {order_sql}) TO {qpath(out_dir_temp)} "
            f"(FORMAT parquet, PARTITION_BY ({', '.join(part_cols)}), "
            f"COMPRESSION {ctx.parquet_compression}, ROW_GROUP_SIZE {ctx.parquet_row_group_size});"
        )
        
        # Atomically replace the final directory
        if os.path.exists(out_dir_final):
            shutil.rmtree(out_dir_final)
        os.rename(out_dir_temp, out_dir_final)
        
        glob = os.path.join(out_dir_final, "**", "*.parquet")
        ctx.create_or_replace_parquet_view(dest, glob, hive_partitioning=True, exclude_cols=exclude_cols)
        # Carry partitioning forward
        ctx.partitioning[dest] = pby
        return

    out_file = os.path.join(variant_dir, f"{dest}.parquet")
    if os.path.exists(out_file):
        os.remove(out_file)
    ctx.created_paths.append(out_file)

    ctx.con.execute(
        f"COPY (SELECT * FROM {src_ref} ORDER BY {order_by}) TO {qpath(out_file)} "
        f"(FORMAT parquet, COMPRESSION {ctx.parquet_compression}, ROW_GROUP_SIZE {ctx.parquet_row_group_size});"
    )
    ctx.create_or_replace_parquet_view(dest, out_file, hive_partitioning=False)


def op_partition(ctx: LayoutContext, op: Dict[str, Any]) -> None:
    if ctx.backend != "parquet_hive":
        raise PlanValidationError("PARTITION is only supported on parquet_hive backend in this implementation")

    source = op.get("source") or op.get("table")
    dest = op.get("dest") or source
    pby = op.get("partition_by")
    if not source or not dest or not pby:
        raise PlanValidationError("PARTITION requires source/table, partition_by")

    src_ref = ctx.resolve_relation(source)

    ensure_dir(ctx.variant_dir())
    out_dir = os.path.join(ctx.variant_dir(), dest)
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    ensure_dir(out_dir)
    ctx.created_paths.append(out_dir)

    derived_cols, part_cols, exclude_cols = _partition_exprs(pby)

    select_sql = f"SELECT *"
    if derived_cols:
        select_sql += ", " + ", ".join(derived_cols)
    select_sql += f" FROM {src_ref}"

    ctx.con.execute(
        f"COPY ({select_sql}) TO {qpath(out_dir)} "
        f"(FORMAT parquet, PARTITION_BY ({', '.join(part_cols)}), "
        f"COMPRESSION {ctx.parquet_compression}, ROW_GROUP_SIZE {ctx.parquet_row_group_size});"
    )

    glob = os.path.join(out_dir, "**", "*.parquet")
    ctx.create_or_replace_parquet_view(dest, glob, hive_partitioning=True, exclude_cols=exclude_cols)
    # Remember partitioning for later ops so we can preserve it.
    ctx.partitioning[dest] = pby


def op_zorder(ctx: LayoutContext, op: Dict[str, Any]) -> None:
    """
    ZORDER operator: multi-dimensional clustering using z-order curve.
    Approximation: hash the specified columns together, then sort by the hash.
    This co-locates rows that are similar across multiple dimensions.
    """
    source = op.get("source") or op.get("table")
    dest = op.get("dest") or source
    cols = op.get("cols")
    if not source or not dest or not cols:
        raise PlanValidationError("ZORDER requires source/table, dest(optional), cols")
    
    src_ref = ctx.resolve_relation(source)
    cols_expr = ", ".join(qident(c) for c in cols)
    
    if ctx.backend == "duckdb_internal":
        ctx.con.execute(
            f"CREATE OR REPLACE TABLE {qident(ctx.variant_schema)}.{qident(dest)} AS "
            f"SELECT * FROM {src_ref} ORDER BY hash({cols_expr});"
        )
        return
    
    variant_dir = ctx.variant_dir()
    ensure_dir(variant_dir)

    pby = ctx.partitioning.get(source) or ctx.partitioning.get(dest)
    if pby:
        # Use temp directory to avoid deleting source data
        out_dir_temp = os.path.join(variant_dir, f"{dest}_temp_zorder")
        out_dir_final = os.path.join(variant_dir, dest)
        
        if os.path.exists(out_dir_temp):
            shutil.rmtree(out_dir_temp)
        ensure_dir(out_dir_temp)
        ctx.created_paths.append(out_dir_final)

        derived_cols, part_cols, exclude_cols = _partition_exprs(pby)
        select_sql = "SELECT *"
        if derived_cols:
            select_sql += ", " + ", ".join(derived_cols)
        select_sql += f" FROM {src_ref}"
        order_sql = ", ".join(part_cols + [f"hash({cols_expr})"])

        ctx.con.execute(
            f"COPY ({select_sql} ORDER BY {order_sql}) TO {qpath(out_dir_temp)} "
            f"(FORMAT parquet, PARTITION_BY ({', '.join(part_cols)}), "
            f"COMPRESSION {ctx.parquet_compression}, ROW_GROUP_SIZE {ctx.parquet_row_group_size});"
        )
        
        # Atomically replace
        if os.path.exists(out_dir_final):
            shutil.rmtree(out_dir_final)
        os.rename(out_dir_temp, out_dir_final)
        
        glob = os.path.join(out_dir_final, "**", "*.parquet")
        ctx.create_or_replace_parquet_view(dest, glob, hive_partitioning=True, exclude_cols=exclude_cols)
        ctx.partitioning[dest] = pby
        return

    out_file = os.path.join(variant_dir, f"{dest}.parquet")
    if os.path.exists(out_file):
        os.remove(out_file)
    ctx.created_paths.append(out_file)
    
    ctx.con.execute(
        f"COPY (SELECT * FROM {src_ref} ORDER BY hash({cols_expr})) TO {qpath(out_file)} "
        f"(FORMAT parquet, COMPRESSION {ctx.parquet_compression}, ROW_GROUP_SIZE {ctx.parquet_row_group_size});"
    )
    ctx.create_or_replace_parquet_view(dest, out_file, hive_partitioning=False)


def op_mddl(ctx: LayoutContext, op: Dict[str, Any]) -> None:
    """
    MDDL operator: Multidimensional Data Layout (predicate-driven clustering).
    
    Core idea (from Amazon Redshift MDDL):
    - Instead of sorting by column values (Z-Order), sort by predicate truth values
    - Co-locates rows that satisfy the same predicate combinations
    - Optimized for workloads with repetitive filter predicates
    
    Implementation:
    1. For each predicate, generate: CASE WHEN pred THEN 1 ELSE 0 END
    2. Combine with hash() to create mddl_key
    3. ORDER BY mddl_key
    4. Exclude mddl_key from final view
    """
    source = op.get("source") or op.get("table")
    dest = op.get("dest") or source
    predicates = op.get("predicates")
    
    if not source or not dest or not predicates:
        raise PlanValidationError("MDDL requires source/table, dest(optional), predicates (list of predicate SQL strings)")
    
    src_ref = ctx.resolve_relation(source)
    
    # Generate CASE WHEN expressions for each predicate
    case_exprs = []
    for i, pred in enumerate(predicates):
        # Sanitize predicate (basic validation)
        if not pred or len(pred) > 500:
            continue
        case_expr = f"CASE WHEN ({pred}) THEN 1 ELSE 0 END"
        case_exprs.append(case_expr)
    
    if not case_exprs:
        raise PlanValidationError(f"MDDL: No valid predicates provided for table {source}")
    
    # Combine with hash to create mddl_key
    hash_arg = ", ".join(case_exprs)
    mddl_key_expr = f"hash({hash_arg}) AS mddl_key"
    
    if ctx.backend == "duckdb_internal":
        # For internal tables, just sort by mddl_key (don't need to store it)
        ctx.con.execute(
            f"CREATE OR REPLACE TABLE {qident(ctx.variant_schema)}.{qident(dest)} AS "
            f"SELECT * FROM {src_ref} ORDER BY hash({hash_arg});"
        )
        return
    
    variant_dir = ctx.variant_dir()
    ensure_dir(variant_dir)

    pby = ctx.partitioning.get(source) or ctx.partitioning.get(dest)
    if pby:
        # Use temp directory to avoid deleting source data
        out_dir_temp = os.path.join(variant_dir, f"{dest}_temp_mddl")
        out_dir_final = os.path.join(variant_dir, dest)
        
        if os.path.exists(out_dir_temp):
            shutil.rmtree(out_dir_temp)
        ensure_dir(out_dir_temp)
        ctx.created_paths.append(out_dir_final)

        derived_cols, part_cols, exclude_cols = _partition_exprs(pby)
        select_sql = "SELECT *"
        if derived_cols:
            select_sql += ", " + ", ".join(derived_cols)
        select_sql += f", {mddl_key_expr} FROM {src_ref}"
        order_sql = ", ".join(part_cols + ["mddl_key"])

        ctx.con.execute(
            f"COPY ({select_sql} ORDER BY {order_sql}) TO {qpath(out_dir_temp)} "
            f"(FORMAT parquet, PARTITION_BY ({', '.join(part_cols)}), "
            f"COMPRESSION {ctx.parquet_compression}, ROW_GROUP_SIZE {ctx.parquet_row_group_size});"
        )
        
        # Atomically replace
        if os.path.exists(out_dir_final):
            shutil.rmtree(out_dir_final)
        os.rename(out_dir_temp, out_dir_final)
        
        glob = os.path.join(out_dir_final, "**", "*.parquet")
        # exclude derived partition cols and mddl_key
        ctx.create_or_replace_parquet_view(dest, glob, hive_partitioning=True, exclude_cols=(exclude_cols + ["mddl_key"]))
        ctx.partitioning[dest] = pby
        return

    out_file = os.path.join(variant_dir, f"{dest}.parquet")
    if os.path.exists(out_file):
        os.remove(out_file)
    ctx.created_paths.append(out_file)
    
    ctx.con.execute(
        f"COPY (SELECT *, {mddl_key_expr} FROM {src_ref} ORDER BY mddl_key) TO {qpath(out_file)} "
        f"(FORMAT parquet, COMPRESSION {ctx.parquet_compression}, ROW_GROUP_SIZE {ctx.parquet_row_group_size});"
    )
    ctx.con.execute(
        f"CREATE OR REPLACE VIEW {qident(ctx.variant_schema)}.{qident(dest)} AS "
        f"SELECT * EXCLUDE (mddl_key) FROM read_parquet({qpath(out_file)});"
    )


# -----------------------
# Dispatcher to map ops
# -----------------------
OP_DISPATCH = {
    "TUNE_FILE": op_tune_file,
    "CLUSTER_SORT": op_cluster_sort,
    "PARTITION": op_partition,
    "ZORDER": op_zorder,
    "MDDL": op_mddl,
}


def ensure_baseline_source(plan: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure reorganization starts from baseline data, not previous layout.
    
    This is CRITICAL for multi-period reorganization to avoid cascading data loss 
    from destructive operations like PROJECT.
    
    Example problem:
      Cycle 1: PROJECT removes columns D,E,F
      Cycle 2: If we reorganize from Cycle 1's result, D,E,F are permanently lost!
    
    Solution: Always reorganize from baseline (original data).
    
    Args:
        plan: Layout plan
    
    Returns:
        Plan with baseline_schema set (defaults to 'main')
    """
    if "baseline_schema" not in plan:
        plan["baseline_schema"] = "main"
    
    return plan


def apply_layout_plan(
    con: duckdb.DuckDBPyConnection,
    plan: Dict[str, Any],
    base_tables: Optional[List[str]] = None,
) -> List[str]:
    """
    Apply a layout plan: run the ops, return the result.
    
    Note: Automatically ensures reorganization starts from baseline data
    to prevent cascading data loss in multi-period reorganization.
    """
    # Ensure baseline source is set
    plan = ensure_baseline_source(plan)
    
    ctx = LayoutContext(con, plan)

    # Run ops
    for op in plan["ops"]:
        if "op" not in op:
            raise ValueError("Each op in the plan must have an 'op' field")
        op_fn = OP_DISPATCH.get(op["op"])
        if not op_fn:
            raise ValueError(f"Unknown op: {op['op']}")
        op_fn(ctx, op)

    return ctx.created_paths
