#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Layout Reorganization Agent (基于 LLM 的数据布局重组代理)

这个 Agent 使用 LLM 来分析 workload 特征和数据库 schema，
自动生成数据布局重组的逻辑计划，然后通过 layout_compiler 转换为物理迁移计划。
"""

from __future__ import annotations
import os

# NOTE: Do NOT hardcode secrets in source code.
# Configure via env vars or CLI flags:
#   - OPENAI_API_KEY
#   - OPENAI_BASE_URL

# =============================================================================
# LLM Prompts
# =============================================================================

SYSTEM_PROMPT_TEMPLATE = """You are a database physical layout optimization expert for OLAP workloads.
Analyze workload characteristics and schema to generate optimal layout reorganization plans.

## Multi-Dimensional Analysis Framework

**Dimension 1: Access Pattern**
- Scan-heavy workloads: Minimize I/O (compression, column pruning)
- Filter-heavy workloads: Enable pruning (clustering, partitioning)
- Join-heavy workloads: Co-locate join keys (bucketing, sorting)
- Aggregation-heavy workloads: Cluster by group keys

**Dimension 2: Selectivity**
- High selectivity (< {high_selectivity_pct}%): Strong pruning via partition + clustering
- Medium selectivity ({high_selectivity_pct}%-{low_selectivity_pct}%): Clustering or Z-Order
- Low selectivity (> {low_selectivity_pct}%): Focus on scan efficiency

**Dimension 3: Temporal Characteristics**
- Time-range queries (> {temporal_query_ratio_pct}% of workload): Temporal partitioning
  - Start from coarser temporal transforms when tables are large or projected partition counts are high
  - Refine to finer granularity only when expected pruning gain clearly exceeds metadata overhead
  - Avoid plans where partition metadata overhead is likely to dominate scan savings
- High partition pruning potential: PARTITION operator

**Dimension 4: Filter Dimensionality & Predicate Patterns**
- Single-column filter: CLUSTER_SORT on that column
- Two-column filter: CLUSTER_SORT on both (order matters)
- Multi-column ({multi_column_threshold}+ columns) with diverse values: Z-ORDER for balanced pruning
- Multi-column with repetitive predicates (same filters in {repetitive_predicate_threshold}+ queries): MDDL for predicate-driven clustering

**Dimension 5: Table Size**
- Small tables (< {small_table_rows} rows): Often skip optimization (full scan is cheap)
- Medium tables ({small_table_rows}-{large_table_rows} rows): Selective optimization
- Large tables (> {large_table_rows} rows): Primary optimization target
- Very large tables (> {very_large_table_rows} rows): Aggressive optimization required

**Dimension 6: Data Freshness & Growth**
- Static/historical data: More aggressive optimization (data won't change)
- Frequently updated data: Conservative optimization (consider update patterns)
- Append-only data: Optimize for range scans on append key (time-based partitioning)

## Available Operators

**PARTITION**: Physical partitioning for partition pruning
- Syntax: `{{"op": "PARTITION", "partition_by": [{{"col": "date_col", "transform": "year|month|day|identity|bucket|truncate"}}]}}`
- Best for: Time-range queries, high-cardinality equality filters
- Transforms: year, month, day (for dates), bucket (hash, needs param=N), truncate (numeric, needs param=width)

**CLUSTER_SORT**: Single or multi-column sorting for range scans
- Syntax: `{{"op": "CLUSTER_SORT", "keys": [{{"col": "c1", "order": "asc|desc"}}, ...]}}`
- Best for: Single-dimensional filters, ORDER BY queries

**ZORDER**: Multi-dimensional clustering via space-filling curve
- Syntax: `{{"op": "ZORDER", "cols": ["c1", "c2", "c3", ...]}}`
- Best for: Multi-dimensional filters (3+ columns)
- Note: Clusters by column values

**MDDL**: Multidimensional Data Layout (predicate-driven clustering, inspired by Amazon Redshift)
- Syntax: `{{"op": "MDDL", "predicates": ["col1 = 'value'", "col2 >= 100", ...]}}`
- Best for: Repetitive filter predicates (AP workloads with recurring query patterns)
- Note: Clusters by predicate truth values, not column values. Co-locates rows satisfying same predicate combinations.
- Example: `{{"op": "MDDL", "predicates": ["o_orderdate >= DATE '1995-01-01' AND o_orderdate < DATE '1995-04-01'", "o_orderstatus = 'F'"]}}`

**TUNE_FILE**: File-level parameter tuning
- Syntax: `{{"op": "TUNE_FILE", "row_group_size": N}}`
- Best for: Fine-tuning after other operators
- row_group_size guidelines:
  - Smaller values improve pruning granularity but increase metadata overhead
  - Larger values improve compression and scan throughput but reduce pruning precision
  - Choose values from observed workload selectivity and storage/throughput trade-offs

## Composition Rules

1. Pipeline order: PARTITION → ZORDER/CLUSTER_SORT/MDDL → TUNE_FILE
2. PARTITION and ZORDER/CLUSTER_SORT/MDDL must be separate ops
3. Small tables: Often skip optimization (full scan is cheap)
4. Sorting/clustering happens within each partition if both are used
5. MDDL vs Z-ORDER: Use MDDL for repetitive predicates, Z-ORDER for general multi-dimensional filters
6. PARTITION + MDDL: Partition by time, then MDDL within partitions (recommended for AP workloads)
7. Multi-period reorganization: Always reorganize from original baseline data, never from previous layout
8. Core principle: Layout optimization changes data ORGANIZATION (how it's stored), NOT data CONTENT (what is stored)

## Output Format

Return JSON with two fields:

1. **reasoning**: Your analysis explaining decisions
2. **plan**: Reorganization plan with `table_plans` (organized by table)

Example:
```json
{{
  "reasoning": "Analysis:\\n1. lineitem (6M rows): 60% queries filter on l_shipdate, PARTITION by year/month. Repetitive predicates detected (l_returnflag='R', l_discount BETWEEN), MDDL for predicate-driven clustering\\n2. orders: Temporal filters + repetitive status predicates, PARTITION + MDDL\\n3. Dimensions: Skip (< 1M rows)",
  "plan": {{
    "variant_name": "llm_optimized",
    "variant_schema": "v_llm",
    "backend": "parquet_hive",
    "work_dir": "./work",
    "parquet_compression": "zstd",
    "parquet_row_group_size": 250000,
    "table_plans": {{
      "lineitem": [
        {{"op": "PARTITION", "partition_by": [{{"col": "l_shipdate", "transform": "year"}}, {{"col": "l_shipdate", "transform": "month"}}]}},
        {{"op": "MDDL", "predicates": ["l_returnflag = 'R'", "l_discount BETWEEN 0.05 AND 0.07", "l_quantity < 24"]}}
      ],
      "orders": [
        {{"op": "PARTITION", "partition_by": [{{"col": "o_orderdate", "transform": "year"}}]}},
        {{"op": "MDDL", "predicates": ["o_orderstatus = 'F'", "o_orderpriority IN ('1-URGENT','2-HIGH')"]}}
      ]
    }}
  }}
}}
```

Note: Choose between Z-ORDER and MDDL based on workload:
- Z-ORDER: General multi-dimensional filters on various column values
- MDDL: Repetitive predicates (same filters appear in multiple queries)
"""

USER_MESSAGE_TEMPLATE = """# Workload Characteristics
{workload_features}

# Database Schema
{schema_info}

{adaptive_rules}

## Task
Generate optimal layout reorganization plan based on the above workload and schema.

## Output
Return valid JSON with "reasoning" and "plan" fields (use `table_plans` format).
"""

# =============================================================================
# Documentation
# =============================================================================

OPERATORS_DOC = """
## Layout Operators Quick Reference

All operators change data ORGANIZATION, not data CONTENT (no data loss).

1. **PARTITION**: Hive-style partitioning for partition pruning
2. **CLUSTER_SORT**: Single/multi-column sorting for range scans
3. **ZORDER**: Multi-dimensional clustering (column-value based)
4. **MDDL**: Predicate-driven clustering (predicate-truth based, inspired by Amazon Redshift)
5. **TUNE_FILE**: File-level parameter tuning (row_group_size)

## Common Combinations
- PARTITION + ZORDER: Time + multi-dimensional filtering (general)
- PARTITION + MDDL: Time + repetitive predicates (AP workloads, recommended)
- PARTITION + CLUSTER_SORT: Time + single-dimensional filtering

## MDDL vs Z-ORDER
- MDDL: For repetitive filter predicates (e.g., status='F' appears in 10 queries)
- Z-ORDER: For general multi-dimensional filters on various values

Note: Parquet columnar format provides automatic column pruning at query time,
so no need for explicit column selection.
"""

import argparse
import json
import os
import re
from collections import defaultdict
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

import duckdb
try:
    from .operators import apply_layout_plan
except ImportError:
    from logicdb.layout.operators import apply_layout_plan


# ========================================
# 数据库统计信息提取（自适应规则）
# ========================================

@dataclass
class TableStats:
    """单个表的统计信息"""
    name: str
    row_count: int
    column_count: int
    total_size_bytes: int
    columns: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # col_name -> {type, cardinality, null_ratio}
    
    @property
    def size_mb(self) -> float:
        return self.total_size_bytes / (1024 * 1024)


@dataclass
class DatabaseStats:
    """数据库级别的统计信息"""
    total_tables: int
    total_rows: int
    total_size_bytes: int
    tables: Dict[str, TableStats] = field(default_factory=dict)
    
    # 动态计算的阈值（基于百分位数）
    small_table_threshold: int = 0  # 行数阈值
    medium_table_threshold: int = 0
    large_table_threshold: int = 0
    
    high_cardinality_threshold: int = 0  # 基数阈值
    wide_table_threshold: int = 0  # 列数阈值
    
    @property
    def total_size_mb(self) -> float:
        return self.total_size_bytes / (1024 * 1024)
    
    def compute_thresholds(self):
        """根据实际数据分布计算自适应阈值"""
        if not self.tables:
            return
        
        row_counts = sorted([t.row_count for t in self.tables.values()])
        col_counts = sorted([t.column_count for t in self.tables.values()])
        
        # 行数阈值：使用百分位数
        n = len(row_counts)
        if n > 0:
            self.small_table_threshold = row_counts[int(n * 0.3)]  # 30分位
            self.medium_table_threshold = row_counts[int(n * 0.7)]  # 70分位
            self.large_table_threshold = row_counts[int(n * 0.9)]  # 90分位
        
        # 列数阈值：宽表定义
        if len(col_counts) > 0:
            self.wide_table_threshold = col_counts[int(len(col_counts) * 0.7)]
        
        # 基数阈值：用于判断高基数列
        all_cardinalities = []
        for table in self.tables.values():
            for col_info in table.columns.values():
                if 'cardinality' in col_info and col_info['cardinality'] is not None:
                    all_cardinalities.append(col_info['cardinality'])
        
        if all_cardinalities:
            all_cardinalities.sort()
            self.high_cardinality_threshold = all_cardinalities[int(len(all_cardinalities) * 0.7)]


def extract_database_stats(con: duckdb.DuckDBPyConnection, schema: str = "main") -> DatabaseStats:
    """
    提取数据库的详细统计信息（自适应阈值计算）
    
    Args:
        con: DuckDB connection
        schema: Schema name to analyze
    
    Returns:
        DatabaseStats object with adaptive thresholds
    """
    # IMPORTANT: For SF10/SF100 TPC-H, scanning tables to compute COUNT(DISTINCT ...)
    # is extremely expensive. We intentionally use DuckDB's catalog estimates to keep
    # this step cheap enough for iterative experimentation.
    stats = DatabaseStats(total_tables=0, total_rows=0, total_size_bytes=0)
    
    try:
        # duckdb_tables().estimated_size is an estimated row count for regular tables.
        tbl_rows = con.execute(
            """
            SELECT table_name, estimated_size::BIGINT AS row_count, column_count::BIGINT AS column_count
            FROM duckdb_tables()
            WHERE lower(schema_name) = lower(?)
              AND internal = false
              AND temporary = false
            """,
            [schema],
        ).fetchall()
    except Exception as e:
        print(f"Warning: Could not fetch duckdb_tables() for schema '{schema}': {e}")
        return stats
    
    stats.total_tables = len(tbl_rows)
    
    for table_name, row_count, column_count in tbl_rows:
        # crude size estimate (bytes)
        estimated_size = int(row_count) * int(column_count) * 8

        table_stats = TableStats(
            name=str(table_name),
            row_count=int(row_count),
            column_count=int(column_count),
            total_size_bytes=int(estimated_size),
        )
            
        # types only (no expensive per-column scans)
        try:
            columns_info = con.execute(
                """
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE lower(table_schema) = lower(?) AND lower(table_name) = lower(?)
                ORDER BY ordinal_position
                """,
                [schema, table_name],
            ).fetchall()
            for col_name, col_type in columns_info:
                table_stats.columns[str(col_name)] = {"type": str(col_type)}
        except Exception as e:
            raise Exception(f"Error fetching columns for table {table_name}: {e}")

        stats.tables[table_name] = table_stats
        stats.total_rows += int(row_count)
        stats.total_size_bytes += int(estimated_size)

    try:
        stats.compute_thresholds()
    except Exception as e:
        raise Exception(f"Error computing thresholds: {e}")
    return stats

@dataclass
class OptimizationThresholds:
    """所有优化规则的阈值参数（自适应计算）"""
    # Selectivity thresholds
    high_selectivity_pct: float = 1.0      # < 1% is high selectivity
    low_selectivity_pct: float = 10.0      # > 10% is low selectivity
    
    # Filter dimensionality
    multi_column_threshold: int = 3        # 3+ columns for Z-ORDER/MDDL
    repetitive_predicate_threshold: int = 3  # Same predicate in 3+ queries
    
    # Temporal characteristics
    temporal_query_ratio_pct: float = 30.0  # 30%+ queries have time filters
    
    # Table size (dynamically computed)
    small_table_rows: int = 0
    large_table_rows: int = 0
    very_large_table_rows: int = 0
    
    @classmethod
    def from_database_stats(cls, db_stats: DatabaseStats, workload_features: Optional[Dict] = None) -> 'OptimizationThresholds':
        """
        根据数据库统计信息和工作负载特征计算自适应阈值
        
        Args:
            db_stats: Database statistics
            workload_features: Optional workload features for further tuning
        
        Returns:
            OptimizationThresholds with adaptive values
        """
        thresholds = cls()
        
        # 从 db_stats 获取表大小阈值
        thresholds.small_table_rows = db_stats.small_table_threshold
        thresholds.large_table_rows = db_stats.medium_table_threshold
        thresholds.very_large_table_rows = db_stats.large_table_threshold
        
        # 如果有 workload features，进一步调整
        if workload_features:
            # 根据实际 workload 的选择性分布调整
            if 'avg_selectivity' in workload_features:
                avg_sel = workload_features['avg_selectivity']
                # 如果平均选择性很低，降低高选择性阈值
                if avg_sel > 0.5:
                    thresholds.high_selectivity_pct = 5.0
                    thresholds.low_selectivity_pct = 20.0
            
            # 根据实际谓词重复度调整
            if 'max_predicate_frequency' in workload_features:
                max_freq = workload_features['max_predicate_frequency']
                # 如果有非常高频的谓词，降低阈值以启用 MDDL
                if max_freq >= 10:
                    thresholds.repetitive_predicate_threshold = 2
            
            # 根据实际 join 复杂度调整多列阈值
            if 'avg_joins_per_query' in workload_features:
                avg_joins = workload_features['avg_joins_per_query']
                if avg_joins > 3:
                    thresholds.multi_column_threshold = 4  # 更激进的多列聚簇
        
        return thresholds


def generate_adaptive_rules(db_stats: DatabaseStats) -> str:
    """
    根据数据库统计信息生成自适应的规则描述
    
    Args:
        db_stats: Database statistics
    
    Returns:
        Adaptive rules text for LLM prompt
    """
    rules = f"""
## Adaptive Rules (Based on Your Database Statistics)

**Database Overview:**
- Total tables: {db_stats.total_tables}
- Total rows: {db_stats.total_rows:,}
- Total size: {db_stats.total_size_mb:.2f} MB

**Adaptive Thresholds (computed from data distribution):**
- Small table: < {db_stats.small_table_threshold:,} rows (30th percentile)
- Medium table: {db_stats.small_table_threshold:,} - {db_stats.medium_table_threshold:,} rows
- Large table: > {db_stats.medium_table_threshold:,} rows (70th percentile)
- Very large table: > {db_stats.large_table_threshold:,} rows (90th percentile)
- Wide table: > {db_stats.wide_table_threshold} columns

**Table-Specific Statistics:**
"""
    
    # 按表大小排序
    sorted_tables = sorted(db_stats.tables.values(), key=lambda t: t.row_count, reverse=True)
    
    for table in sorted_tables[:10]:  # 只显示前10个最大的表
        size_category = "small"
        if table.row_count > db_stats.large_table_threshold:
            size_category = "very large"
        elif table.row_count > db_stats.medium_table_threshold:
            size_category = "large"
        elif table.row_count > db_stats.small_table_threshold:
            size_category = "medium"
        
        width_category = "wide" if table.column_count > db_stats.wide_table_threshold else "normal"
        
        rules += f"- {table.name}: {table.row_count:,} rows, {table.column_count} cols [{size_category}, {width_category}]\n"
        
        # 找出高基数列（用于分区/聚簇候选）
        high_card_cols = [
            col_name for col_name, info in table.columns.items()
            if info.get('cardinality', 0) > db_stats.high_cardinality_threshold
        ]
        if high_card_cols:
            rules += f"  High-cardinality columns: {', '.join(high_card_cols[:5])}\n"
    
    # Keep this section minimal to avoid duplicating guidance already in SYSTEM_PROMPT_TEMPLATE.
    rules += """
**Notes:**
- Focus on large/very large tables first; small tables are usually not worth physical reorg.
- Prefer PARTITION/CLUSTER_SORT/ZORDER/MDDL/TUNE_FILE only (no destructive ops).
"""
    
    return rules


# ========================================
# LLM 集成部分
# ========================================

def call_llm_for_layout_plan(
    workload_features: str,
    schema_info: str,
    model: str = "deepseek-v3.2",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    db_stats: Optional[DatabaseStats] = None,
    workload_features_dict: Optional[Dict] = None,
    reasoning_effort: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Call LLM to generate data layout reorganization plan
    
    Args:
        workload_features: workload characteristics (text or JSON)
        schema_info: database schema information
        model: LLM model name
        api_key: API key (from env if None)
        base_url: API base URL (auto-detect if None)
        db_stats: Database statistics for adaptive thresholds (if None, use default)
        workload_features_dict: Parsed workload features for threshold tuning
    
    Returns:
        Dict with 'reasoning' and 'plan' fields
    """
    try:
        import openai
    except ImportError:
        raise ImportError("需要安装 openai 库: pip install openai")
    
    client = openai.OpenAI(
        api_key=api_key or os.environ.get("OPENAI_API_KEY"),
        base_url=base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    )
    # Compute adaptive thresholds
    if db_stats is not None:
        thresholds = OptimizationThresholds.from_database_stats(db_stats, workload_features_dict)
        adaptive_rules = generate_adaptive_rules(db_stats)
    else:
        # Use default thresholds if no db_stats
        thresholds = OptimizationThresholds()
        thresholds.small_table_rows = 1_000_000
        thresholds.large_table_rows = 10_000_000
        thresholds.very_large_table_rows = 50_000_000
        adaptive_rules = ""
    
    # Fill system prompt with adaptive thresholds
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        high_selectivity_pct=thresholds.high_selectivity_pct,
        low_selectivity_pct=thresholds.low_selectivity_pct,
        multi_column_threshold=thresholds.multi_column_threshold,
        repetitive_predicate_threshold=thresholds.repetitive_predicate_threshold,
        temporal_query_ratio_pct=thresholds.temporal_query_ratio_pct,
        small_table_rows=f"{thresholds.small_table_rows:,}",
        large_table_rows=f"{thresholds.large_table_rows:,}",
        very_large_table_rows=f"{thresholds.very_large_table_rows:,}"
    )
    
    # Format user message using template
    user_message = USER_MESSAGE_TEMPLATE.format(
        workload_features=workload_features,
        schema_info=schema_info,
        adaptive_rules=adaptive_rules
    )
    
    # Print prompt for debugging
    print("\n" + "="*80)
    print("[LLM PROMPT - SYSTEM]")
    print("="*80)
    print(system_prompt)
    print("\n" + "="*80)
    print("[LLM PROMPT - USER]")
    print("="*80)
    print(user_message)
    print("="*80 + "\n")
    
    # Call LLM
    chat_kwargs: Dict[str, Any] = {}
    if reasoning_effort:
        chat_kwargs["reasoning_effort"] = str(reasoning_effort).strip().lower()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        temperature=0.3,
        max_tokens=4096,
        **chat_kwargs,
    )
    
    def _content_to_text(raw_content: Any) -> str:
        if raw_content is None:
            return ""
        if isinstance(raw_content, str):
            return raw_content
        if isinstance(raw_content, list):
            parts: List[str] = []
            for item in raw_content:
                if isinstance(item, str):
                    parts.append(item)
                    continue
                if isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
                        continue
                    if isinstance(text, list):
                        for seg in text:
                            if isinstance(seg, str):
                                parts.append(seg)
                            elif isinstance(seg, dict):
                                t = seg.get("text") or seg.get("content")
                                if t:
                                    parts.append(str(t))
                        continue
                    content_text = item.get("content")
                    if isinstance(content_text, str):
                        parts.append(content_text)
                        continue
                parts.append(str(item))
            return "\n".join(p for p in parts if p).strip()
        if isinstance(raw_content, dict):
            txt = raw_content.get("text") or raw_content.get("content")
            if txt:
                return str(txt)
            try:
                return json.dumps(raw_content, ensure_ascii=False)
            except Exception:
                return str(raw_content)
        return str(raw_content)

    def _strip_json_comments(s: str) -> str:
        s = re.sub(r"/\*.*?\*/", "", s, flags=re.DOTALL)
        s = re.sub(r"^\s*//.*?$", "", s, flags=re.MULTILINE)
        return s

    def _strip_trailing_commas(s: str) -> str:
        return re.sub(r",\s*([}\]])", r"\1", s)

    def _extract_json_objects(s: str) -> List[str]:
        out: List[str] = []
        if not s:
            return out
        dec = json.JSONDecoder()
        i, n = 0, len(s)
        while i < n:
            if s[i] != "{":
                i += 1
                continue
            try:
                _, end = dec.raw_decode(s, i)
                frag = s[i:end].strip()
                if frag:
                    out.append(frag)
                i = end
            except Exception:
                i += 1
        return out

    def _parse_result_from_text(text: str) -> Tuple[Optional[Dict[str, Any]], List[str]]:
        if not text:
            return None, ["empty_content"]
        candidates: List[str] = []
        for m in re.finditer(r"```json\s*(.*?)\s*```", text, re.IGNORECASE | re.DOTALL):
            blk = (m.group(1) or "").strip()
            if blk:
                candidates.append(blk)
        for m in re.finditer(r"```\s*(.*?)\s*```", text, re.DOTALL):
            blk = (m.group(1) or "").strip()
            if blk and blk not in candidates:
                candidates.append(blk)
        candidates.append(text.strip())
        for obj_text in _extract_json_objects(text):
            if obj_text and obj_text not in candidates:
                candidates.append(obj_text)

        result_obj: Optional[Dict[str, Any]] = None
        parse_errors: List[str] = []
        tried: List[str] = []
        for cand in candidates:
            if not cand:
                continue
            lb = cand.find("{")
            rb = cand.rfind("}")
            sliced = cand[lb:rb + 1] if lb != -1 and rb != -1 and rb > lb else ""
            variants = [
                cand,
                _strip_json_comments(cand),
                _strip_trailing_commas(cand),
                _strip_trailing_commas(_strip_json_comments(cand)),
            ]
            if sliced:
                variants.extend([
                    sliced,
                    _strip_json_comments(sliced),
                    _strip_trailing_commas(sliced),
                    _strip_trailing_commas(_strip_json_comments(sliced)),
                ])
            for v in variants:
                v = (v or "").strip()
                if not v or v in tried:
                    continue
                tried.append(v)
                try:
                    parsed = json.loads(v)
                    if isinstance(parsed, dict):
                        result_obj = parsed
                        break
                    parse_errors.append(f"json_not_object:{type(parsed).__name__}")
                except Exception as e:
                    parse_errors.append(str(e))
            if result_obj is not None:
                break
        return result_obj, parse_errors

    # 解析返回内容（兼容字符串/分段内容）
    msg = response.choices[0].message
    content = _content_to_text(getattr(msg, "content", None))
    if not content:
        content = _content_to_text(getattr(msg, "reasoning", None))

    print("\n" + "="*80)
    print("[LLM REPLY - RAW]")
    print("="*80)
    print(content)
    print("="*80 + "\n")

    result, parse_errors = _parse_result_from_text(content)
    if result is None:
        retry_user_message = (
            user_message
            + "\n\nIMPORTANT: Return ONLY one valid JSON object. "
              "No markdown fence, no prose before/after JSON."
        )
        retry_resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": retry_user_message}
            ],
            temperature=0.0,
            max_tokens=3072,
            **chat_kwargs,
        )
        retry_msg = retry_resp.choices[0].message
        retry_content = _content_to_text(getattr(retry_msg, "content", None))
        if not retry_content:
            retry_content = _content_to_text(getattr(retry_msg, "reasoning", None))
        print("\n" + "="*80)
        print("[LLM REPLY - RAW RETRY]")
        print("="*80)
        print(retry_content)
        print("="*80 + "\n")
        retry_result, retry_errors = _parse_result_from_text(retry_content)
        parse_errors.extend([f"retry:{e}" for e in retry_errors[:5]])
        if retry_result is not None:
            result = retry_result

    if result is None:
        raise ValueError(
            "LLM output is not valid JSON after fallback parsing. "
            f"errors={parse_errors[:3]}"
        )
    
    # 转换 table_plans 格式为 ops 格式（如果需要）
    if "plan" in result and "table_plans" in result["plan"]:
        result["plan"] = convert_table_plans_to_ops(result["plan"])
    
    return result


def heuristic_layout_plan(
    schema_info: str,
    *,
    db_stats: Optional[DatabaseStats] = None,
    workload_stats: Optional[Dict[str, Any]] = None,
    variant_name: str = "logicdb_layout",
    variant_schema: str = "logicdb_layout",
    work_dir: str = "./logicdb_runs/parquet",
) -> Dict[str, Any]:
    """
    Deterministic fallback used when no LLM credentials are available.

    The goal is not to outperform the LLM planner; it is to keep the public demo
    runnable and to emit a conservative plan that mirrors the same OpenOps shape.
    """
    ops: List[Dict[str, Any]] = []
    reasoning: List[str] = [
        "Heuristic fallback selected because no OpenAI credentials were available.",
        "The fallback emits only conservative, non-destructive layout actions.",
    ]
    if db_stats is None or not db_stats.tables:
        return {
            "reasoning": " ".join(reasoning + ["No database statistics were available, so no layout action was proposed."]),
            "plan": {
                "variant_name": variant_name,
                "variant_schema": variant_schema,
                "backend": "parquet_hive",
                "work_dir": work_dir,
                "ops": [],
            },
        }

    filter_freq = dict(workload_stats.get("filter_freq") or []) if workload_stats else {}
    group_freq = dict(workload_stats.get("group_by_freq") or []) if workload_stats else {}

    def _is_temporal(col_name: str) -> bool:
        s = str(col_name or "").lower()
        return any(tok in s for tok in ("date", "time", "timestamp", "year", "month", "day"))

    def _score_col(col_name: str) -> int:
        return int(filter_freq.get(col_name, 0)) + int(group_freq.get(col_name, 0))

    large_tables = sorted(
        db_stats.tables.values(),
        key=lambda t: (t.row_count, t.column_count),
        reverse=True,
    )
    large_tables = [t for t in large_tables if t.row_count >= max(1, db_stats.medium_table_threshold or 0)]
    for table in large_tables[:2]:
        col_names = list(table.columns.keys())
        temporal_cols = [c for c in col_names if _is_temporal(c)]
        ranked_cols = sorted(col_names, key=_score_col, reverse=True)
        if temporal_cols:
            ops.append(
                {
                    "op": "PARTITION",
                    "table": table.name,
                    "partition_by": [{"col": temporal_cols[0], "transform": "year"}],
                }
            )
            reasoning.append(
                f"Table {table.name} is large ({table.row_count:,} rows) and has temporal column {temporal_cols[0]}, so yearly partitioning was selected."
            )
        cluster_cols = [c for c in ranked_cols if c not in temporal_cols][:2]
        if len(cluster_cols) >= 2:
            ops.append(
                {
                    "op": "ZORDER",
                    "table": table.name,
                    "cols": cluster_cols[: min(3, len(cluster_cols))],
                }
            )
            reasoning.append(
                f"Table {table.name} reuses filter/group columns {', '.join(cluster_cols[:2])}, so a multi-column locality action was added."
            )
        elif len(cluster_cols) == 1:
            ops.append(
                {
                    "op": "CLUSTER_SORT",
                    "table": table.name,
                    "keys": [{"col": cluster_cols[0], "order": "asc"}],
                }
            )
            reasoning.append(
                f"Table {table.name} repeatedly touches {cluster_cols[0]}, so a single-column clustering action was added."
            )

    return {
        "reasoning": " ".join(reasoning),
        "plan": {
            "variant_name": variant_name,
            "variant_schema": variant_schema,
            "backend": "parquet_hive",
            "work_dir": work_dir,
            "ops": ops,
        },
    }


def convert_table_plans_to_ops(plan: Dict[str, Any]) -> Dict[str, Any]:
    """
    将按表组织的 table_plans 格式转换为顺序的 ops 格式
    
    table_plans 格式:
    {
      "table_plans": {
        "lineitem": [
          {"op": "PARTITION", "partition_by": [...]},
          {"op": "ZORDER", "cols": [...]}
        ],
        "orders": [...]
      }
    }
    
    转换为 ops 格式:
    {
      "ops": [
        {"op": "PARTITION", "table": "lineitem", "partition_by": [...]},
        {"op": "ZORDER", "table": "lineitem", "cols": [...]},
        {"op": "PARTITION", "table": "orders", ...},
        ...
      ]
    }
    """
    table_plans = plan.pop("table_plans", {})
    ops = []

    def _normalize_action(table_name: str, action: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        act = str(action.get("op") or action.get("action") or action.get("type") or "").strip().upper()
        if not act or act == "NONE":
            return None

        out: Dict[str, Any] = {"table": table_name, "op": act}
        if act == "PARTITION":
            raw = action.get("partition_by") or action.get("columns") or []
            part_cols = []
            for item in raw:
                if isinstance(item, dict):
                    col = item.get("col")
                    if col:
                        part_cols.append({
                            "col": str(col),
                            "transform": str(item.get("transform") or "identity"),
                            **({"param": item["param"]} if item.get("param") is not None else {}),
                        })
                elif item:
                    part_cols.append({"col": str(item), "transform": "identity"})
            if not part_cols:
                return None
            out["partition_by"] = part_cols
            return out

        if act == "CLUSTER_SORT":
            raw = action.get("keys") or action.get("columns") or []
            keys = []
            for item in raw:
                if isinstance(item, dict):
                    col = item.get("col")
                    if col:
                        keys.append({"col": str(col), "order": str(item.get("order") or "asc")})
                elif item:
                    keys.append({"col": str(item), "order": "asc"})
            if not keys:
                return None
            out["keys"] = keys
            return out

        if act == "ZORDER":
            cols = [str(c) for c in (action.get("cols") or action.get("columns") or []) if c]
            if not cols:
                return None
            out["cols"] = cols
            return out

        if act == "MDDL":
            predicates = [str(p) for p in (action.get("predicates") or []) if p]
            if not predicates:
                return None
            out["predicates"] = predicates
            return out

        if act == "TUNE_FILE":
            params = action.get("parameters") or {}
            if isinstance(params, dict):
                if params.get("row_group_size") is not None:
                    out["row_group_size"] = params["row_group_size"]
            return out

        return None

    if isinstance(table_plans, dict):
        # 按表名排序，确保确定性输出
        for table_name in sorted(table_plans.keys()):
            table_entry = table_plans[table_name]
            if not isinstance(table_name, str) or not table_name.strip():
                continue
            if str(table_name).strip().lower() == "cross_table_notes":
                continue
            table_ops = table_entry
            if isinstance(table_entry, dict):
                table_ops = table_entry.get("actions") or table_entry.get("ops") or []
                if not table_ops:
                    synthesized_ops: List[Dict[str, Any]] = []
                    partition = table_entry.get("partition")
                    if isinstance(partition, dict):
                        strategy = str(partition.get("strategy") or "").strip().upper()
                        part_cols = [
                            str(c).strip()
                            for c in list(partition.get("columns") or [])
                            if str(c).strip()
                        ]
                        if strategy not in ("", "NONE") and part_cols:
                            synthesized_ops.append({
                                "type": "PARTITION",
                                "columns": part_cols,
                            })
                    cluster_sort = table_entry.get("cluster_sort")
                    if isinstance(cluster_sort, dict):
                        cluster_cols = [
                            str(c).strip()
                            for c in list(cluster_sort.get("columns") or [])
                            if str(c).strip()
                        ]
                        if cluster_cols:
                            synthesized_ops.append({
                                "type": "CLUSTER_SORT",
                                "columns": cluster_cols,
                            })
                    zorder = table_entry.get("zorder")
                    if isinstance(zorder, dict):
                        z_cols = [
                            str(c).strip()
                            for c in list(zorder.get("columns") or [])
                            if str(c).strip()
                        ]
                        if z_cols:
                            synthesized_ops.append({
                                "type": "ZORDER",
                                "columns": z_cols,
                            })
                    file_tuning = table_entry.get("file_tuning")
                    if isinstance(file_tuning, dict):
                        target_mb = file_tuning.get("target_file_size_mb")
                        if target_mb is not None:
                            synthesized_ops.append({
                                "type": "TUNE_FILE",
                                "parameters": {
                                    "row_group_size": target_mb,
                                },
                            })
                    table_ops = synthesized_ops
            if not isinstance(table_ops, list):
                continue
            for op_dict in table_ops:
                if not isinstance(op_dict, dict):
                    continue
                normalized = _normalize_action(str(table_name).strip(), op_dict)
                if normalized is not None:
                    ops.append(normalized)
    elif isinstance(table_plans, list):
        for entry in table_plans:
            if not isinstance(entry, dict):
                continue
            table_name = str(entry.get("table") or "").strip()
            if not table_name:
                continue
            for action in (entry.get("actions") or []):
                if not isinstance(action, dict):
                    continue
                normalized = _normalize_action(table_name, action)
                if normalized is not None:
                    ops.append(normalized)
    
    plan["ops"] = ops
    return plan


# ========================================
# TPC-H Workload 特征提取
# ========================================

def extract_tpch_workload_features(con: duckdb.DuckDBPyConnection) -> str:
    """
    Extract comprehensive workload features from TPC-H queries using the local WorkloadCharacterizer.
    """
    try:
        from .workload import WorkloadCharacterizer
        
        rows = con.execute("SELECT query_nr, query FROM tpch_queries() ORDER BY query_nr").fetchall()
        queries = [(str(row[1]).strip().rstrip(";"), f"Q{int(row[0]):02d}", 1.0) for row in rows]
        
        # Use WorkloadCharacterizer for accurate feature extraction
        wc = WorkloadCharacterizer()
        for sql, qid, weight in queries:
            wc.add_query(sql, qid, weight)
        
        wf = wc.aggregate()
        
        # Format for LLM consumption
        output = {
            "workload_summary": {
                "total_queries": wf.num_queries,
                "avg_tables_per_query": round(wf.avg_tables_per_query, 2),
                "avg_joins_per_query": round(wf.avg_joins_per_query, 2),
                "avg_predicates_per_query": round(wf.avg_predicates_per_query, 2),
                "time_range_query_ratio": f"{wf.time_range_query_ratio:.1%}",
                "partition_pruning_potential": f"{wf.partition_pruning_potential:.1%}",
            },
            "table_access_patterns": {},
            "join_graph": {f"{t1}<->{t2}": freq for (t1, t2), freq in wf.join_graph.items()},
            "layout_recommendations": {
                "partition_candidates": {t: list(cols) for t, cols in wf.recommended_partition_columns.items()},
                "clustering_candidates": {t: list(cols) for t, cols in wf.recommended_clustering_columns.items()},
                "zorder_candidates": {t: list(cols) for t, cols in wf.recommended_zorder_columns.items()},
            },
        }
        
        # Per-table details
        for table, tap in wf.tables.items():
            output["table_access_patterns"][table] = {
                "query_frequency": tap.query_count,
                "access_ratio": f"{tap.query_count / wf.num_queries:.1%}",
                "filtered_scan_ratio": f"{tap.filtered_scan_count / max(1, tap.query_count):.1%}",
                "has_temporal_predicates": tap.has_temporal_predicates,
                "temporal_columns": list(tap.temporal_columns),
                "join_relationships": tap.join_with_tables,
                "top_filter_columns": sorted(
                    [(col, cap.filter_count) for col, cap in tap.columns.items() if cap.filter_count > 0],
                    key=lambda x: x[1],
                    reverse=True
                )[:5],
                "top_join_columns": sorted(
                    [(col, cap.join_count) for col, cap in tap.columns.items() if cap.join_count > 0],
                    key=lambda x: x[1],
                    reverse=True
                )[:5],
            }
        
        return json.dumps(output, indent=2, ensure_ascii=False)
        
    except ImportError:
        # Fallback to simple features if the optional parser path is unavailable.
        return """TPC-H workload (22 queries):
- Time-range queries dominant (l_shipdate, o_orderdate)
- Multi-table joins (lineitem-orders-customer, part-partsupp-supplier)
- Aggregation-heavy (SUM, AVG, COUNT)
- Selective filtering (dates, status, amount ranges)
"""
    except Exception as e:
        # Fallback on error
        return f"""TPC-H workload (22 queries):
- Complex analytical workload
- Error extracting detailed features: {str(e)}
- Recommend: temporal partitioning + multi-dimensional clustering
"""


def extract_workload_features_from_jsonl(path: str, max_queries: int = 500) -> str:
    """
    Extract workload features from a JSONL file produced by `tpc_qgen.py`.
    Each line is expected to include at least: {"template": int, "sql": str, ...}
    To keep runtime bounded, we sample up to `max_queries` queries, stratified by template.
    """
    try:
        from .workload import WorkloadCharacterizer
    except ImportError:
        # Fallback: just pass first few SQLs as plain text
        lines = []
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= min(max_queries, 50):
                    break
                obj = json.loads(line)
                lines.append(obj.get("sql", "").strip().rstrip(";"))
        return "JSONL workload (fallback):\n" + "\n".join(lines[:20])

    def _template_key(v: Any) -> str:
        if v is None:
            return "__NONE__"
        s = str(v).strip()
        return s if s else "__NONE__"

    # Stratified sampling by template id (adaptive cap by observed template count)
    template_ids = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                template_ids.add(_template_key(obj.get("template")))
            except Exception:
                continue
    num_templates = max(1, len(template_ids))

    per_t = defaultdict(int)
    sampled = []
    per_template_cap = max(1, (max_queries + num_templates - 1) // num_templates)

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if len(sampled) >= max_queries:
                break
            obj = json.loads(line)
            tmpl = _template_key(obj.get("template"))
            if per_t[tmpl] >= per_template_cap:
                continue
            per_t[tmpl] += 1
            sampled.append((obj.get("sql", "").strip().rstrip(";"), f"T{tmpl}_Q{obj.get('qid', 0)}", 1.0))

    wc = WorkloadCharacterizer()
    for sql, qid, weight in sampled:
        if sql:
            wc.add_query(sql, qid, weight)
    wf = wc.aggregate()

    output = {
        "workload_summary": {
            "sampled_queries": wf.num_queries,
            "avg_tables_per_query": round(wf.avg_tables_per_query, 2),
            "avg_joins_per_query": round(wf.avg_joins_per_query, 2),
            "avg_predicates_per_query": round(wf.avg_predicates_per_query, 2),
            "time_range_query_ratio": f"{wf.time_range_query_ratio:.1%}",
            "partition_pruning_potential": f"{wf.partition_pruning_potential:.1%}",
        },
        "layout_recommendations": {
            "partition_candidates": {t: list(cols) for t, cols in wf.recommended_partition_columns.items()},
            "clustering_candidates": {t: list(cols) for t, cols in wf.recommended_clustering_columns.items()},
            "zorder_candidates": {t: list(cols) for t, cols in wf.recommended_zorder_columns.items()},
        },
        "top_predicates": {t: preds[:10] for t, preds in wf.top_predicates.items()},
    }
    return json.dumps(output, indent=2, ensure_ascii=False)


def get_tpch_schema_info(
    con: duckdb.DuckDBPyConnection,
    schema: str = "main",
    output_format: str = "mschema",
) -> str:
    """
    获取 TPC-H schema 信息。
    output_format:
      - "json": machine-readable JSON schema
      - "mschema": text format aligned with bird_gen's mschema style
    """
    tables = [r[0] for r in con.execute(
        """
        SELECT table_name
        FROM information_schema.tables
        WHERE lower(table_schema) = lower(?)
          AND table_type = 'BASE TABLE'
        ORDER BY table_name
        """,
        [schema],
    ).fetchall()]

    # Common TPC-H FK relationships (only emitted when both sides exist)
    tpch_fk = [
        ("customer", "c_nationkey", "nation", "n_nationkey"),
        ("orders", "o_custkey", "customer", "c_custkey"),
        ("lineitem", "l_orderkey", "orders", "o_orderkey"),
        ("lineitem", "l_partkey", "part", "p_partkey"),
        ("lineitem", "l_suppkey", "supplier", "s_suppkey"),
        ("lineitem", "l_partkey,l_suppkey", "partsupp", "ps_partkey,ps_suppkey"),
        ("partsupp", "ps_partkey", "part", "p_partkey"),
        ("partsupp", "ps_suppkey", "supplier", "s_suppkey"),
        ("supplier", "s_nationkey", "nation", "n_nationkey"),
        ("nation", "n_regionkey", "region", "r_regionkey"),
    ]
    
    schema_info = {}
    for table in tables:
        try:
            cols = con.execute(
                """
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE lower(table_schema) = lower(?) AND lower(table_name) = lower(?)
                ORDER BY ordinal_position
                """,
                [schema, table],
            ).fetchall()
            row_count = con.execute(f"SELECT COUNT(*) as cnt FROM {schema}.{table}").fetchone()[0]
            schema_info[table] = {
                "row_count": row_count,
                "columns": [{"column_name": c, "column_type": t} for c, t in cols],
            }
        except Exception as e:
            schema_info[table] = {"error": str(e)}
    
    if output_format == "json":
        return json.dumps(schema_info, indent=2, ensure_ascii=False)

    # mschema-like text output
    lines = []
    for table in tables:
        info = schema_info.get(table, {})
        if "error" in info:
            continue
        lines.append(f"# Table: {table}")
        lines.append("[")
        pk_col = None
        # TPC-H PK naming convention
        if table == "partsupp":
            pk_cols = {"ps_partkey", "ps_suppkey"}
        else:
            pk_cols = {f"{table[0]}_{table}_key"}  # fallback, mostly unused
            # Correct primary keys for canonical TPC-H table names
            known_pk = {
                "customer": {"c_custkey"},
                "lineitem": {"l_orderkey", "l_linenumber"},
                "nation": {"n_nationkey"},
                "orders": {"o_orderkey"},
                "part": {"p_partkey"},
                "partsupp": {"ps_partkey", "ps_suppkey"},
                "region": {"r_regionkey"},
                "supplier": {"s_suppkey"},
            }
            pk_cols = known_pk.get(table, set())

        col_lines = []
        for col in info.get("columns", []):
            cname = str(col["column_name"])
            ctype = str(col["column_type"]).upper()
            if cname in pk_cols:
                col_lines.append(f"({cname}:{ctype}, Primary Key)")
            else:
                col_lines.append(f"({cname}:{ctype})")
        lines.append(",\n".join(col_lines))
        lines.append("]")

    present = set(tables)
    fk_lines = []
    for t1, c1, t2, c2 in tpch_fk:
        if t1 in present and t2 in present:
            fk_lines.append(f"{t1}.{c1}={t2}.{c2}")
    if fk_lines:
        lines.append("【Foreign keys】")
        lines.extend(fk_lines)

    return "\n".join(lines)


# ========================================
# 执行重组计划
# ========================================

def execute_layout_plan(
    con: duckdb.DuckDBPyConnection,
    plan: Dict[str, Any],
    dry_run: bool = False
) -> None:
    """
    执行布局重组计划
    
    Args:
        con: DuckDB 连接
        plan: 重组计划（包含 ops 列表）
        dry_run: 如果为 True，只打印计划不实际执行
    """
    print("\n" + "=" * 80)
    print("执行布局重组计划")
    print("=" * 80)
    print(f"\nVariant: {plan.get('variant_name', 'unnamed')}")
    print(f"Schema: {plan.get('variant_schema', 'unnamed')}")
    print(f"Backend: {plan.get('backend', 'parquet_hive')}")
    print(f"\n共 {len(plan.get('ops', []))} 个算子操作:\n")
    
    for i, op in enumerate(plan.get("ops", []), 1):
        print(f"{i}. {op['op']} - table: {op.get('table', op.get('source', 'N/A'))}")
        for k, v in op.items():
            if k not in ["op", "table", "source"]:
                print(f"   {k}: {v}")
    
    if dry_run:
        print("\n[DRY RUN] 不执行实际迁移")
        return
    
    print("\n开始执行...")
    created_paths = apply_layout_plan(con, plan)
    
    print("\n完成！创建的文件/目录:")
    for path in created_paths:
        print(f"  - {path}")


# ========================================
# 主函数
# ========================================

def main():
    parser = argparse.ArgumentParser(
        description="基于 LLM 的数据布局重组代理",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--db",
        required=True,
        help="DuckDB 数据库文件路径（例如 tpch_sf1.duckdb）"
    )
    
    parser.add_argument(
        "--workload",
        default="tpch",
        help="Workload 类型（tpch 或自定义 JSON 文件路径）"
    )
    
    parser.add_argument(
        "--schema",
        default="auto",
        help="Schema 信息（auto 自动提取，或 JSON 文件路径）"
    )
    
    parser.add_argument(
        "--output",
        default="llm_layout_plan.json",
        help="输出的计划文件路径"
    )
    
    parser.add_argument(
        "--workdir",
        default="./work_llm",
        help="工作目录（存放重组后的文件）"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只生成计划不执行"
    )
    
    parser.add_argument(
        "--execute",
        action="store_true",
        help="生成计划后立即执行"
    )
    
    parser.add_argument(
        "--model",
        default="deepseek-v3.2",
        help="LLM 模型名称"
    )
    parser.add_argument("--api-key", default=None, help="LLM API key (defaults to env OPENAI_API_KEY)")
    parser.add_argument("--base-url", default=None, help="LLM API base url (defaults to env OPENAI_BASE_URL)")
    
    args = parser.parse_args()
    
    # 连接数据库
    print(f"连接数据库: {args.db}")
    con = duckdb.connect(args.db)
    
    # 加载 TPC-H extension（如果需要）
    if args.workload == "tpch":
        try:
            con.execute("INSTALL tpch;")
            con.execute("LOAD tpch;")
        except Exception as e:
            print(f"警告: 无法加载 TPC-H extension: {e}")
    
    # 提取 workload 特征
    print("\n提取 Workload 特征...")
    if args.workload == "tpch":
        workload_features = extract_tpch_workload_features(con)
    elif os.path.exists(args.workload) and args.workload.lower().endswith(".jsonl"):
        workload_features = extract_workload_features_from_jsonl(args.workload)
    elif os.path.exists(args.workload):
        with open(args.workload, "r", encoding="utf-8") as f:
            workload_features = f.read()
    else:
        workload_features = args.workload
    
    print("Workload 特征:")
    print(workload_features[:500] + "..." if len(workload_features) > 500 else workload_features)
    
    # 提取 Schema 信息
    print("\n提取 Schema 信息...")
    if args.schema == "auto":
        schema_info = get_tpch_schema_info(con)
    elif os.path.exists(args.schema):
        with open(args.schema, "r", encoding="utf-8") as f:
            schema_info = f.read()
    else:
        schema_info = args.schema
    
    print("Schema 信息:")
    print(schema_info[:500] + "..." if len(schema_info) > 500 else schema_info)
    
    # 提取数据库统计信息（自适应规则）
    print("\n提取数据库统计信息（用于自适应规则）...")
    db_stats = extract_database_stats(con, schema="main")
    print(f"  ✓ 分析了 {db_stats.total_tables} 个表，共 {db_stats.total_rows:,} 行")
    print(f"  ✓ 自适应阈值: 小表<{db_stats.small_table_threshold:,}行, 大表>{db_stats.medium_table_threshold:,}行")
    
    # 解析 workload features 为字典（用于阈值微调）
    workload_features_dict = None
    try:
        workload_features_dict = json.loads(workload_features)
    except:
        # 如果不是 JSON，保持为 None
        pass
    
    # 调用 LLM 生成计划
    print(f"\n调用 LLM ({args.model}) 生成布局重组计划...")
    print(f"API Base URL: {args.base_url or os.environ.get('OPENAI_BASE_URL') or 'default'}")
    
    # 显示使用的阈值
    if db_stats:
        thresholds = OptimizationThresholds.from_database_stats(db_stats, workload_features_dict)
        print(f"  ✓ 自适应阈值: 小表<{thresholds.small_table_rows:,}行, 大表>{thresholds.large_table_rows:,}行")
        print(f"  ✓ 多列阈值: {thresholds.multi_column_threshold}列, 重复谓词: {thresholds.repetitive_predicate_threshold}次")
        print(f"  ✓ 选择性阈值: 高<{thresholds.high_selectivity_pct}%, 低>{thresholds.low_selectivity_pct}%")
    
    result = call_llm_for_layout_plan(
        workload_features, 
        schema_info, 
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
        db_stats=db_stats,
        workload_features_dict=workload_features_dict
    )
    
    # 打印 LLM 的思考过程
    print("\n" + "=" * 80)
    print("LLM 推理过程 (Chain-of-Thought)")
    print("=" * 80)
    print(result.get("reasoning", "无"))
    
    # 提取计划
    plan = result.get("plan", {})
    
    # 更新计划参数
    plan["work_dir"] = args.workdir
    if "variant_name" not in plan:
        plan["variant_name"] = "llm_optimized"
    if "variant_schema" not in plan:
        plan["variant_schema"] = "v_llm"
    if "backend" not in plan:
        plan["backend"] = "parquet_hive"
    
    # 保存计划
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({"reasoning": result.get("reasoning"), "plan": plan}, f, indent=2, ensure_ascii=False)
    print(f"\n✓ 计划已保存到: {args.output}")
    
    # 打印计划摘要
    print("\n" + "=" * 80)
    print("生成的布局重组计划")
    print("=" * 80)
    
    # 按表分组显示（支持新旧两种格式）
    if "table_plans" in plan:
        # 新格式：已经按表组织
        ops_by_table = {}
        for table, ops in plan["table_plans"].items():
            ops_by_table[table] = [{"table": table, **op} for op in ops]
    else:
        # 旧格式：ops 列表
        ops_by_table = {}
        for op in plan.get("ops", []):
            table = op.get("table") or op.get("source", "unknown")
            if table not in ops_by_table:
                ops_by_table[table] = []
            ops_by_table[table].append(op)
    
    for table, ops in ops_by_table.items():
        ops_desc = []
        for op in ops:
            if op["op"] == "PARTITION":
                pby = op.get("partition_by", [])
                pby_str = ", ".join(f"{p['col']} ({p['transform']})" for p in pby)
                ops_desc.append(f"PARTITION({pby_str})")
            elif op["op"] == "ZORDER":
                cols = op.get("cols", [])
                ops_desc.append(f"ZORDER({', '.join(cols)})")
            elif op["op"] == "CLUSTER_SORT":
                keys = op.get("keys", [])
                keys_str = ", ".join(f"{k['col']}" for k in keys)
                ops_desc.append(f"CLUSTER_SORT({keys_str})")
            elif op["op"] == "PROJECT":
                cols = op.get("cols", [])
                ops_desc.append(f"PROJECT({len(cols)} cols)")
            elif op["op"] == "TUNE_FILE":
                rg = op.get("row_group_size", "N/A")
                ops_desc.append(f"TUNE_FILE(row_group={rg})")
            else:
                ops_desc.append(op["op"])
        
        print(f"  - {table}: {' + '.join(ops_desc)}")
    
    # 如果指定 --execute，执行计划
    if args.execute:
        execute_layout_plan(con, plan, dry_run=args.dry_run)
    else:
        print(f"\n提示: 使用 --execute 参数来执行重组计划")
        print(f"或者稍后手动执行: python {__file__} --db {args.db} --execute --workload {args.output}")
    
    con.close()


if __name__ == "__main__":
    main()
