#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OLAP 优化建议：Layout + Index + Materialized Views
全部使用 LLM 生成，覆盖数据分析场景的三种优化维度
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Optional, Tuple

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


# 与 AIDB OLAP 模块对齐：layout_engine / index_rec / mv_opt 使用的概念
SYSTEM_PROMPT = """You are an OLAP optimization expert. Your suggestions align with AIDB's optimization stack (layout_engine, index_rec, mv_opt).

## 1. Data Layout (物理布局) - 对应 layout_engine
- **CLUSTER_SORT**: Pre-sort by filter/group/order columns (single or multi-column)
- **PARTITION**: Partition by date/time (year, month) for temporal queries
- **ZORDER/MDDL**: For multi-dimensional filters (pandas: sort by multiple columns)
- Output: Python pandas code (df.sort_values, df["year"] = pd.to_datetime(...).dt.year, etc.)

## 2. Index (索引) - 对应 index_rec
- **B-tree**: filter/join/order columns, equality and range predicates
- **Partial**: WHERE clause for frequent predicate combinations (SQLite: CREATE INDEX ... WHERE ...)
- **Covering**: INCLUDE columns for index-only scans
- Output: SQL (CREATE INDEX ...) and/or Python (df.set_index, pre-sort before to_parquet)

## 3. Materialized Views (物化视图) - 对应 mv_opt
- **Pre-join**: Store frequently joined table combinations (CREATE TABLE mv AS SELECT ... JOIN ...)
- **Pre-aggregate**: Store GROUP BY results for repeated aggregations

## Rules
- Be concise: 1-3 suggestions per category, only when beneficial
- For SQLite: output valid SQLite CREATE INDEX / CREATE TABLE AS
- For Python: output valid pandas code, table names match schema
- If workload is empty or trivial, suggest only the most impactful optimization
- Always explain briefly why each suggestion helps

## Output Format (JSON)
{
  "reasoning": "Brief analysis of workload patterns",
  "layout": {
    "suggestions": ["python code 1", "python code 2"],
    "reason": "why"
  },
  "index": {
    "suggestions": ["CREATE INDEX ...", "python code"],
    "reason": "why"
  },
  "materialized_views": {
    "suggestions": ["CREATE TABLE ... AS SELECT ...", "python pre-aggregate code"],
    "reason": "why"
  }
}
"""


def suggest_olap_llm(
    schema: str,
    workload_summary: str,
    model: str = "gpt-5-ca",
) -> Tuple[Dict[str, Any], str]:
    """
    使用 LLM 生成完整的 OLAP 优化建议（Layout + Index + MV）。
    返回: (parsed_result_dict, raw_reasoning)
    """
    try:
        client = OpenAI() if OpenAI else None
    except Exception:
        client = None
    if not client:
        return _fallback_suggest_olap(workload_summary), "OpenAI client not available"

    user_msg = f"""## Database Schema
{schema[:3000]}

## Workload / Analysis Summary
{workload_summary[:2000] if workload_summary else "No workload provided. General analysis context."}

Generate OLAP optimization suggestions. Return valid JSON only."""

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.2,
        )
        content = resp.choices[0].message.content or ""
    except Exception as e:
        return _fallback_suggest_olap(workload_summary), f"LLM error: {e}"

    # Parse JSON from response
    parsed = _extract_json(content)
    if parsed:
        return parsed, parsed.get("reasoning", "")
    return _fallback_suggest_olap(workload_summary), "LLM response invalid, used heuristic fallback"


def _extract_json(text: str) -> Optional[Dict]:
    """从 LLM 返回中提取 JSON"""
    text = text.strip()
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        pass
    # 尝试 ```json 块
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass
    return None


def _fallback_suggest_olap(workload_summary: str) -> Dict[str, Any]:
    """LLM 不可用时的启发式回退"""
    s = workload_summary.lower()
    layout = []
    index = []
    mv = []

    if any(k in s for k in ["date", "time", "year", "month"]):
        layout.append('# Time-based layout\ndf["year"] = pd.to_datetime(df["date_col"]).dt.year\ndf = df.sort_values(["year", "date_col"])')
    if any(k in s for k in ["join", "merge"]):
        index.append("-- Index join columns\nCREATE INDEX IF NOT EXISTS idx_t1_col ON table1(join_col);\nCREATE INDEX IF NOT EXISTS idx_t2_col ON table2(join_col);")
    if any(k in s for k in ["group", "agg", "sum", "count"]):
        mv.append("# Pre-aggregate\n# df_agg = df.groupby(['key_col']).agg({'val': 'sum'})\n# df_agg.to_parquet('agg.parquet')")

    if not layout:
        layout.append("df = df.sort_values(df.columns[0])  # Sort by first column")
    if not index:
        index.append("-- Consider: CREATE INDEX ON table(filter_col) for frequent filters")
    if not mv:
        mv.append("# Consider pre-aggregating repeated GROUP BY queries")

    return {
        "reasoning": "Heuristic fallback (no LLM)",
        "layout": {"suggestions": layout, "reason": "General layout hints"},
        "index": {"suggestions": index, "reason": "General index hints"},
        "materialized_views": {"suggestions": mv, "reason": "General MV hints"},
    }


def suggest_olap_code(workload_summary: str) -> Tuple[str, str]:
    """
    兼容旧接口：仅 workload_summary，无 LLM，返回简单 Python 建议。
    新代码应使用 suggest_olap_llm。
    """
    result = _fallback_suggest_olap(workload_summary)
    parts = []
    for cat in ["layout", "index", "materialized_views"]:
        sugg = result.get(cat, {})
        items = sugg.get("suggestions", [])
        if items:
            parts.append(f"## {cat.upper()}\n" + "\n".join(items))
    code = "\n\n".join(parts) if parts else "df = df.sort_values(df.columns[0])"
    reason = result.get("reasoning", "")
    return code, reason


def layout_sort_to_pandas(cols: list, ascending: bool = True) -> str:
    """CLUSTER_SORT -> pandas sort_values"""
    asc = "True" if ascending else "False"
    cols_str = ", ".join(f'"{c}"' for c in cols)
    return f"df = df.sort_values([{cols_str}], ascending={asc})"


def layout_partition_to_pandas(date_col: str, by: str = "year") -> str:
    """PARTITION (time) -> pandas 分区列"""
    if by == "year":
        return f'df["year"] = pd.to_datetime(df["{date_col}"]).dt.year'
    if by == "month":
        return f'df["month"] = pd.to_datetime(df["{date_col}"]).dt.month'
    return f'df["part"] = pd.to_datetime(df["{date_col}"]).dt.to_period("M")'


def format_olap_suggestion(parsed: Dict[str, Any]) -> str:
    """将 LLM 解析结果格式化为可读字符串"""
    lines = []
    if parsed.get("reasoning"):
        lines.append(f"**Analysis**: {parsed['reasoning']}\n")
    for cat in ["layout", "index", "materialized_views"]:
        block = parsed.get(cat, {})
        suggs = block.get("suggestions", [])
        reason = block.get("reason", "")
        if suggs:
            lines.append(f"### {cat.replace('_', ' ').title()}")
            if reason:
                lines.append(f"*{reason}*")
            for s in suggs:
                lines.append(f"```\n{s}\n```")
            lines.append("")
    return "\n".join(lines).strip()
