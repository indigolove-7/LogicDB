#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ReAct 风格 Agent：多步推理 + 工具调用
目标：像 Cursor 一样边看数据、边反馈、一步步完成分析
"""

from __future__ import annotations

import json
import os
import re
import hashlib
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, List, Optional

from logicdb.data.loaders import (
    get_db_path,
    load_spider_tables,
    spider_schema_to_str,
)
from .tools import (
    get_tool_definitions,
    tool_describe_table,
    tool_execute_python,
    tool_get_code_template,
    tool_get_join_path,
    tool_get_sample,
    tool_get_schema,
    tool_list_tables,
    tool_query_sql,
    tool_read_table,
    tool_suggest_olap,
)
from logicdb.skills.library import build_skill_hint

# OpenAI client
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


@dataclass
class AgentContext:
    """Agent 运行上下文"""
    db_id: str
    db_path: str
    dataset: str  # spider | bird | wtq | tabfact
    question: str
    spider_tables: Optional[Dict[str, Dict]] = None
    evidence: Optional[str] = None  # BIRD 外部知识，若有则注入 prompt
    parquet_dir: Optional[str] = None  # 若提供，则优先从 parquet 读取（含离线优化/MV）
    max_turns: int = 10
    model: str = "gpt-5-ca"
    use_mschema: bool = True  # schema 使用 mschema 格式（与 QBridge 一致）
    # 推理/思考程度：仅对支持 reasoning 的模型有效（如 gpt-5 系）。传 "low"/"none" 可减少 think 时长与 token
    reasoning_effort: Optional[str] = None  # 如 "none", "low", "medium", "high"
    api_base: Optional[str] = None  # 若指定则 OpenAI(base_url=api_base)，用于 DeepSeek 等兼容 API
    # Strict code-only mode: hide SQL-style tools and reject query_sql at runtime.
    sql_free_only: bool = False
    # Optional tool exposure controls for code path.
    code_tool_allowlist: Optional[List[str]] = None
    code_tool_denylist: Optional[List[str]] = None
    # Optional retrieved skills distilled from successful trajectories.
    skill_bank_path: Optional[str] = None
    skill_top_k: int = 0
    skill_max_chars: int = 900
    # Shared controller profile (e.g., OpenOps physical/runtime hints).
    shared_profile: Optional[Dict[str, Any]] = None
    # Inject runtime profile into prompts and update bottleneck feedback online.
    enable_runtime_profile: bool = True
    runtime_profile_max_history: int = 8


def _extract_tool_call(content: Any) -> Optional[Dict]:
    """解析 LLM 返回中的第一个 tool_call（兼容单调用）"""
    if not hasattr(content, "tool_calls") or not content.tool_calls:
        return None
    tc = content.tool_calls[0]
    name = tc.function.name
    try:
        args = json.loads(tc.function.arguments)
    except json.JSONDecodeError:
        args = {}
    return {"name": name, "arguments": args}


def _extract_all_tool_calls(content: Any) -> List[Dict[str, Any]]:
    """解析 LLM 返回中的全部 tool_calls。API 要求每个 tool_call_id 都必须有对应响应"""
    if not hasattr(content, "tool_calls") or not content.tool_calls:
        return []
    out = []
    for tc in content.tool_calls:
        name = tc.function.name
        try:
            args = json.loads(tc.function.arguments)
        except json.JSONDecodeError:
            args = {}
        out.append({"id": tc.id, "name": name, "arguments": args})
    return out


def _extract_tool_call_from_text(text: str) -> Optional[Dict]:
    """从文本中解析 Action: tool_name {json} 格式（ReAct 风格）"""
    m = re.search(r"Action:\s*(\w+)\s*(\{[^}]+\})", text, re.DOTALL)
    if m:
        try:
            args = json.loads(m.group(2))
            return {"name": m.group(1), "arguments": args}
        except json.JSONDecodeError:
            pass
    # 尝试 ```json 块
    m = re.search(r"```json\s*(\{[^`]+)\s*```", text, re.DOTALL)
    if m:
        try:
            args = json.loads(m.group(1))
            name = args.pop("tool", args.pop("action", "query_sql"))
            return {"name": name, "arguments": args}
        except (json.JSONDecodeError, KeyError):
            pass
    return None


_COOPT_FAST_CACHE: Dict[str, bool] = {}
_COOPT_FAST_LOCK = threading.Lock()


def _coopt_fast_enabled(ctx: AgentContext) -> bool:
    """
    Enable fast-path only when parquet manifest shows at least one optimization
    module is effectively enabled (layout/index/mv).
    """
    fast_flag = str(os.environ.get("NL2CODE_COOPT_FAST_PATH", "0") or "").strip().lower()
    if fast_flag not in {"1", "true", "yes", "on"}:
        return False
    force_flag = str(os.environ.get("NL2CODE_COOPT_FAST_FORCE", "0") or "").strip().lower()
    if force_flag in {"1", "true", "yes", "on"}:
        return True
    pdir = str(getattr(ctx, "parquet_dir", "") or "").strip()
    if not pdir:
        return False
    key = os.path.abspath(pdir)
    with _COOPT_FAST_LOCK:
        if key in _COOPT_FAST_CACHE:
            return _COOPT_FAST_CACHE[key]

    candidates = [
        os.path.join(os.path.dirname(key), "manifest.json"),
        os.path.join(key, "manifest.json"),
    ]
    enabled = False
    for mp in candidates:
        if not os.path.exists(mp):
            continue
        try:
            obj = json.load(open(mp, "r", encoding="utf-8"))
            mods = obj.get("enabled_modules") or {}
            enabled = bool(mods.get("layout") or mods.get("index") or mods.get("mv"))
            break
        except Exception:
            continue

    with _COOPT_FAST_LOCK:
        _COOPT_FAST_CACHE[key] = enabled
    return enabled


def run_tool(
    name: str,
    arguments: Dict[str, Any],
    ctx: AgentContext,
    last_read_sample: Optional[List[Dict]] = None,
    samples_dict: Optional[Dict[str, list]] = None,
    parquet_df_cache: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """执行工具并返回结果"""
    if name == "query_sql":
        if bool(getattr(ctx, "sql_free_only", False)):
            return {
                "ok": False,
                "error": "query_sql is disabled in sql_free_only mode; use read_table + execute_python.",
            }
        return tool_query_sql(
            sql=arguments.get("sql", ""),
            db_path=ctx.db_path,
        )
    if name == "get_schema":
        return tool_get_schema(
            db_path=ctx.db_path,
            spider_tables=ctx.spider_tables,
            db_id=ctx.db_id,
            use_mschema=getattr(ctx, "use_mschema", True),
            parquet_dir=getattr(ctx, "parquet_dir", None),
        )
    if name == "get_sample":
        return tool_get_sample(
            db_path=ctx.db_path,
            table=arguments.get("table", ""),
            n=int(arguments.get("n", 5)),
        )
    if name == "list_tables":
        return tool_list_tables(db_path=ctx.db_path, parquet_dir=getattr(ctx, "parquet_dir", None))
    if name == "read_table":
        table = arguments.get("table", "")
        partition_filter = arguments.get("partition_filter")
        read_mode = str(arguments.get("mode", "sample") or "sample").strip().lower()
        if read_mode not in {"sample", "full"}:
            read_mode = "sample"
        default_limit = 40 if str(getattr(ctx, "dataset", "") or "").lower() in {"wtq", "tabfact"} else 100
        try:
            sample_limit = int(arguments.get("limit", default_limit))
        except Exception:
            sample_limit = default_limit
        if sample_limit <= 0:
            sample_limit = default_limit
        # Avoid repeated table reads in one trajectory.
        # For parquet mode, only fast-return when full DataFrame was already cached.
        cached_rows = samples_dict.get(table) if (samples_dict and table) else None
        if (
            table
            and not partition_filter
            and samples_dict
            and table in samples_dict
            and read_mode != "full"
            and isinstance(cached_rows, list)
            and len(cached_rows) >= sample_limit
            and (
                not getattr(ctx, "parquet_dir", None)
                or (parquet_df_cache and table in parquet_df_cache)
            )
        ):
            return {
                "ok": True,
                "table": table,
                "row_count": -1,
                "sample": cached_rows[:sample_limit],
                "cached_sample": True,
                "read_mode": "sample",
                "sample_limit": sample_limit,
                "sampled": True,
                "note": "cached sampled rows; not guaranteed to represent full-table aggregates",
            }
        return tool_read_table(
            db_path=ctx.db_path,
            table=table,
            parquet_dir=getattr(ctx, "parquet_dir", None),
            partition_filter=partition_filter,
            limit=sample_limit,
            mode=read_mode,
        )
    if name == "execute_python":
        return tool_execute_python(
            code=arguments.get("code", ""),
            df_refs=arguments.get("df_refs"),
            sample=last_read_sample,
            samples_dict=samples_dict or {},
            parquet_df_cache=parquet_df_cache or {},
        )
    if name == "suggest_olap":
        return tool_suggest_olap(
            workload_summary=arguments.get("workload_summary", ""),
            schema=arguments.get("schema"),
            db_path=ctx.db_path,
            model=ctx.model,
        )
    if name == "describe_table":
        return tool_describe_table(
            db_path=ctx.db_path,
            table=arguments.get("table", ""),
            parquet_dir=getattr(ctx, "parquet_dir", None),
        )
    if name == "get_join_path":
        return tool_get_join_path(
            db_path=ctx.db_path,
            tables=arguments.get("tables"),
            start_table=arguments.get("start_table"),
            end_table=arguments.get("end_table"),
            max_hops=int(arguments.get("max_hops", 3)),
            top_k=int(arguments.get("top_k", 8)),
        )
    if name == "get_code_template":
        return tool_get_code_template(pattern=arguments.get("pattern", ""))
    return {"ok": False, "error": f"Unknown tool: {name}"}


def _build_openai_client(ctx: AgentContext):
    """Create an OpenAI-compatible client from context settings."""
    if not OpenAI:
        return None
    if getattr(ctx, "api_base", None):
        return OpenAI(base_url=ctx.api_base)
    return OpenAI()


def _to_json_serializable(val: Any) -> Any:
    """Convert nested values to JSON-serializable structures."""
    if val is None:
        return None
    if isinstance(val, set):
        return [_to_json_serializable(x) for x in val]
    if isinstance(val, tuple):
        return [_to_json_serializable(x) for x in val]
    if isinstance(val, list):
        return [_to_json_serializable(x) for x in val]
    if isinstance(val, dict):
        return {str(k): _to_json_serializable(v) for k, v in val.items()}
    if hasattr(val, "tolist"):
        try:
            return val.tolist()
        except Exception:
            pass
    if hasattr(val, "item"):
        try:
            return val.item()
        except Exception:
            pass
    return val


def _normalize_scalar(v: Any) -> str:
    """Normalize a scalar value for stable set/diff comparison."""
    if v is None:
        return "__NULL__"
    try:
        if hasattr(v, "item"):
            v = v.item()
    except Exception:
        pass
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if v.is_integer():
            return str(int(v))
        return str(v)
    if isinstance(v, str):
        return v.strip().lower()
    return str(v).strip().lower()


def _flatten_result_values(result: Any, limit: int = 512) -> List[Any]:
    """Best-effort flatten nested result into scalar-like values."""
    out: List[Any] = []

    def _rec(v: Any) -> None:
        if len(out) >= limit:
            return
        if v is None or isinstance(v, (str, int, float, bool)):
            out.append(v)
            return
        if isinstance(v, dict):
            for vv in v.values():
                _rec(vv)
                if len(out) >= limit:
                    return
            return
        if isinstance(v, (list, tuple, set)):
            for it in v:
                _rec(it)
                if len(out) >= limit:
                    return
            return
        if hasattr(v, "item"):
            try:
                out.append(v.item())
                return
            except Exception:
                pass
        out.append(v)

    _rec(result)
    return out


def _coerce_binary_label(val: Any) -> Optional[int]:
    """Parse common truthy/falsy outputs into TabFact label 1/0."""
    if val is None:
        return None
    if isinstance(val, bool):
        return 1 if val else 0
    if isinstance(val, (int, float)):
        try:
            return 1 if int(val) != 0 else 0
        except Exception:
            return None
    s = str(val).strip().lower()
    if s in {"1", "true", "yes", "entailed", "entailment", "support", "supported"}:
        return 1
    if s in {"0", "false", "no", "refuted", "refute", "contradiction"}:
        return 0
    try:
        x = float(s)
        return 1 if int(x) != 0 else 0
    except Exception:
        return None


def _normalize_wtq_value(v: Any) -> Any:
    """Normalize WTQ value surface form (lightweight, non-destructive)."""
    if v is None:
        return None
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, (int, float)):
        try:
            if isinstance(v, float) and v.is_integer():
                return int(v)
        except Exception:
            pass
        return v
    s = str(v).strip().strip("\"'`")
    s = re.sub(r"\s+", " ", s)
    if not s:
        return None
    num_s = s.replace(",", "")
    try:
        x = float(num_s)
        if x.is_integer():
            return int(x)
        return x
    except Exception:
        return s


def _apply_dataset_result_contract(ctx: AgentContext, result: Any) -> tuple[Any, Optional[str]]:
    """
    Coerce result shape for dataset-specific output contract.
    - TabFact: scalar int 1/0
    - WTQ: scalar or list of scalar answers (no row dict wrappers)
    """
    ds = (ctx.dataset or "").lower()
    if result is None:
        return result, None
    if ds == "tabfact":
        vals = _flatten_result_values(result)
        for v in vals:
            lab = _coerce_binary_label(v)
            if lab is not None:
                return lab, "tabfact_label_coerced"
        return result, None
    if ds == "wtq":
        vals = _flatten_result_values(result)
        norm_vals = [_normalize_wtq_value(v) for v in vals]
        norm_vals = [v for v in norm_vals if v is not None]
        if not norm_vals:
            return [], "wtq_empty_answer_list"
        if len(norm_vals) == 1:
            return norm_vals[0], "wtq_single_value_coerced"
        return norm_vals, "wtq_multi_value_coerced"
    return result, None


def _row_to_tuple(row: Any) -> tuple:
    """Normalize one row into a hashable tuple."""
    if isinstance(row, dict):
        return tuple(sorted(_normalize_scalar(v) for v in row.values()))
    if isinstance(row, (list, tuple)):
        return tuple(_normalize_scalar(x) for x in row)
    return (_normalize_scalar(row),)


def _result_row_count(result: Any) -> int:
    """Best-effort row count across result shapes."""
    if result is None:
        return 0
    if isinstance(result, list):
        return len(result)
    if isinstance(result, dict):
        return 1
    return 1


def _compact_result_preview(result: Any, limit: int = 5) -> List[str]:
    """Compact human-readable preview lines for prompts/logs."""
    rows: List[Any]
    if result is None:
        rows = []
    elif isinstance(result, list):
        rows = result[:limit]
    else:
        rows = [result]
    out: List[str] = []
    for r in rows:
        try:
            out.append(str(_row_to_tuple(r)))
        except Exception:
            out.append(str(r))
    return out


def _result_signature(result: Any, sample_rows: int = 64) -> str:
    """Stable lightweight signature to compare large candidates without full payload."""
    h = hashlib.sha1()
    h.update(str(type(result).__name__).encode("utf-8"))
    h.update(str(_result_row_count(result)).encode("utf-8"))
    rows = result if isinstance(result, list) else [result]
    for r in rows[:sample_rows]:
        h.update(str(_row_to_tuple(r)).encode("utf-8"))
    return h.hexdigest()[:16]


def _result_to_set_limited(result: Any, cap: int = 2000) -> tuple[set, bool]:
    """
    Convert result to a normalized set with optional cap.
    Returns (set, truncated) where truncated=True means comparison is approximate.
    """
    if result is None:
        return set(), False
    if isinstance(result, list):
        truncated = len(result) > cap
        rows = result[:cap]
    else:
        truncated = False
        rows = [result]
    return {_row_to_tuple(r) for r in rows}, truncated


def _summarize_candidate_diff(sql_result: Any, code_result: Any, cap: int = 2000) -> Dict[str, Any]:
    """
    Summarize two candidate results without embedding full payload into prompts.
    This is designed for large outputs: compare normalized sets up to a cap and
    attach signatures + small mismatch samples.
    """
    sql_set, sql_trunc = _result_to_set_limited(sql_result, cap=cap)
    code_set, code_trunc = _result_to_set_limited(code_result, cap=cap)
    inter = sql_set & code_set
    only_sql = list(sql_set - code_set)
    only_code = list(code_set - sql_set)
    exact = not (sql_trunc or code_trunc)
    equal = (sql_set == code_set) if exact else (
        len(only_sql) == 0 and len(only_code) == 0 and _result_row_count(sql_result) == _result_row_count(code_result)
    )
    return {
        "exact_compare": exact,
        "equal": equal,
        "sql_rows": _result_row_count(sql_result),
        "code_rows": _result_row_count(code_result),
        "sql_arity": _result_row_arity(sql_result),
        "code_arity": _result_row_arity(code_result),
        "sql_columns_preview": _result_column_names_preview(sql_result),
        "code_columns_preview": _result_column_names_preview(code_result),
        "sql_signature": _result_signature(sql_result),
        "code_signature": _result_signature(code_result),
        "sql_truncated_for_compare": sql_trunc,
        "code_truncated_for_compare": code_trunc,
        "intersection_count": len(inter),
        "only_sql_count": len(only_sql),
        "only_code_count": len(only_code),
        "only_sql_sample": [str(x) for x in only_sql[:8]],
        "only_code_sample": [str(x) for x in only_code[:8]],
        "sql_preview": _compact_result_preview(sql_result, limit=5),
        "code_preview": _compact_result_preview(code_result, limit=5),
    }


def _validate_candidate_result(result: Any) -> tuple[bool, Optional[str]]:
    """
    Reject obvious debug/intermediate outputs from being treated as final answer.
    Keep conservative to avoid false rejection of legitimate results.
    """
    if result is None:
        return False, "result is None"
    if isinstance(result, str):
        s = result.lower()
        if "dtype:" in s and "name:" in s:
            return False, "looks like pandas Series repr"
        if "columns:" in s and ("index(" in s or "dtype:" in s):
            return False, "looks like debug print output"
        return True, None
    if isinstance(result, dict):
        key_l = [str(k).lower() for k in result.keys()]
        if any("columns" in k for k in key_l):
            return False, "dictionary appears to contain debug columns"
        if any(k in {"error", "exception", "traceback", "stacktrace"} for k in key_l):
            return False, "dictionary appears to represent an error payload"
        val_blob = " ".join(str(v).lower() for v in list(result.values())[:5])
        if any(tok in val_blob for tok in ["error:", "exception", "traceback"]):
            return False, "dictionary value looks like runtime error text"
        return True, None
    if isinstance(result, list) and result:
        # List of debug strings (column dumps / dtype lines).
        if all(isinstance(x, str) for x in result):
            joined = " ".join(x.lower() for x in result[:10])
            if "dtype:" in joined or "columns" in joined:
                return False, "list appears to contain debug text"
        if all(isinstance(x, dict) for x in result[:5]):
            for row in result[:5]:
                row_keys = {str(k).lower() for k in row.keys()}
                if row_keys & {"error", "exception", "traceback", "stacktrace"}:
                    return False, "list appears to contain error payload rows"
    return True, None


def _is_effectively_empty_result(result: Any) -> bool:
    """Detect empty answer payloads that are usually unusable for final fusion."""
    if result is None:
        return True
    if isinstance(result, (list, tuple, dict, set)):
        return len(result) == 0
    if isinstance(result, str):
        return result.strip() == ""
    try:
        import pandas as _pd
        if isinstance(result, (_pd.DataFrame, _pd.Series)):
            return len(result) == 0
    except Exception:
        pass
    return False


def _extract_single_scalar(result: Any) -> Any:
    """Return the scalar when result effectively contains one value; otherwise None."""
    vals = _flatten_result_values(result, limit=4)
    cleaned: List[Any] = []
    for v in vals:
        if v is None:
            continue
        if isinstance(v, str) and not v.strip():
            continue
        cleaned.append(v)
    if len(cleaned) == 1:
        return cleaned[0]
    return None


def _is_numeric_like(v: Any) -> bool:
    if v is None:
        return False
    if isinstance(v, bool):
        return False
    if isinstance(v, (int, float)):
        return True
    s = str(v).strip()
    if not s:
        return False
    s = s.replace(",", "")
    if s.endswith("%"):
        s = s[:-1].strip()
    try:
        float(s)
        return True
    except Exception:
        return False


def _looks_like_error_text(v: Any) -> bool:
    if not isinstance(v, str):
        return False
    s = v.strip().lower()
    if not s:
        return False
    return any(tok in s for tok in ["error", "exception", "traceback"])


def _reverse_deduce_intent(
    ctx: AgentContext,
    artifact_type: str,
    artifact_text: str,
    result: Any,
    client: Any = None,
    timeout_s: float = 45.0,
) -> Dict[str, Any]:
    """
    Ask an LLM to infer what a SQL/code artifact is answering and score
    alignment with the original question.
    """
    default = {
        "alignment_score": 0.5,
        "critical_mismatch": False,
        "mismatch_tags": [],
        "inferred_focus": "",
        "reason": "reverse_deduce_unavailable",
        "confidence": 0.3,
        "prompt_tokens": 0,
        "completion_tokens": 0,
    }
    cli = client or _build_openai_client(ctx)
    if not cli:
        return default

    artifact = str(artifact_text or "").strip()
    if not artifact:
        d = dict(default)
        d["reason"] = "empty_artifact"
        return d

    result_preview = _compact_result_preview(result, limit=6)
    row_count = _result_row_count(result)
    arity = _result_row_arity(result)
    system_prompt = """You evaluate alignment between an original question and an executed SQL/code artifact.
Infer what the artifact is truly answering from:
- artifact text
- observed result preview
Then score alignment with the original question.

Return strict JSON:
{
  "alignment_score": 0.0 to 1.0,
  "critical_mismatch": true/false,
  "mismatch_tags": ["projection_mismatch"|"aggregation_mismatch"|"filter_mismatch"|"ordering_mismatch"|"count_vs_list_mismatch"|"topk_mismatch"|"other"],
  "inferred_focus": "...",
  "reason": "...",
  "confidence": 0.0 to 1.0
}
Set critical_mismatch=true only when mismatch is clear.
"""
    user_prompt = (
        f"Original question:\n{ctx.question}\n\n"
        f"Artifact type: {artifact_type}\n\n"
        f"Artifact text:\n{artifact[:2800]}\n\n"
        f"Observed result preview:\n{json.dumps(result_preview, ensure_ascii=False)}\n"
        f"row_count={row_count}, arity={arity}"
    )
    try:
        resp = cli.chat.completions.create(
            model=ctx.model,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            timeout=timeout_s,
        )
        usage = getattr(resp, "usage", None)
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0) if usage else 0
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0) if usage else 0
        txt = (resp.choices[0].message.content or "").strip()
        m = re.search(r"\{.*\}", txt, re.DOTALL)
        if not m:
            d = dict(default)
            d["reason"] = txt[:200] if txt else "reverse_deduce_no_json"
            d["prompt_tokens"] = prompt_tokens
            d["completion_tokens"] = completion_tokens
            return d
        obj = json.loads(m.group(0))
        try:
            score_f = float(obj.get("alignment_score", 0.5))
        except Exception:
            score_f = 0.5
        try:
            conf_f = float(obj.get("confidence", 0.3))
        except Exception:
            conf_f = 0.3
        tags = obj.get("mismatch_tags", [])
        if not isinstance(tags, list):
            tags = [str(tags)]
        return {
            "alignment_score": max(0.0, min(1.0, score_f)),
            "critical_mismatch": bool(obj.get("critical_mismatch", False)),
            "mismatch_tags": [str(x) for x in tags[:8]],
            "inferred_focus": str(obj.get("inferred_focus", ""))[:220],
            "reason": str(obj.get("reason", ""))[:320],
            "confidence": max(0.0, min(1.0, conf_f)),
            "raw": txt[:800],
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }
    except Exception as e:
        d = dict(default)
        d["reason"] = f"reverse_deduce_error: {e}"
        return d


def _score_sql_candidate(
    reverse_meta: Dict[str, Any],
    shape_reasons: List[str],
    semantic_reasons: List[str],
    sql_errors_so_far: int,
) -> float:
    """Score one SQL candidate for drift-resistant selection."""
    score = float(reverse_meta.get("alignment_score", 0.5) or 0.5)
    if reverse_meta.get("critical_mismatch"):
        score -= 0.45
    if shape_reasons:
        score -= 0.28
    if semantic_reasons:
        score -= 0.22
    score -= min(max(int(sql_errors_so_far or 0), 0), 3) * 0.05
    return max(0.0, min(1.0, score))


def _is_sql_candidate_clean(candidate: Dict[str, Any]) -> bool:
    """Conservative gate before candidate-bank final selection."""
    if not candidate:
        return False
    if candidate.get("shape_reasons"):
        return False
    if candidate.get("semantic_reasons"):
        return False
    rev = candidate.get("reverse_meta") or {}
    if bool(rev.get("critical_mismatch", False)):
        return False
    return True


def _question_expects_single_row(question: str) -> bool:
    """Heuristic: detect singular-intent questions where multi-row outputs are likely wrong."""
    q = (question or "").strip().lower()
    if not q:
        return False
    plural_markers = [
        "what are", "which are", "list ", "show ", "give all", "all ",
        "how many", "number of", "countries where", "names of", "records",
    ]
    if any(m in q for m in plural_markers):
        return False
    singular_markers = [
        "what is the", "who is the", "which is the",
        "youngest", "oldest", "highest", "lowest",
        "maximum", "minimum", "most ", "least ", "top 1", "first ",
    ]
    return any(m in q for m in singular_markers)


def _question_has_multi_output_intent(question: str) -> bool:
    q = (question or "").lower()
    if not q:
        return False
    multi_phrases = [
        "full communication address",
        "name and ",
        "names and ",
        "include the name",
        "showing their",
        "including their",
        "with their",
        "along with",
        "as well as",
        "both the",
    ]
    if any(m in q for m in multi_phrases):
        return True
    # output-field style "list/show/give ... and ... <attribute>"
    return bool(
        re.search(
            r"\b(list|show|give|state|provide)\b[^?.]{0,80}\band\b[^?.]{0,80}\b(name|title|phone|address|city|state|zip|id|code|date|type|website|score|rank|number)\b",
            q,
        )
    )


_NUM_WORD_TO_INT = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}


def _question_has_count_intent(question: str, expr_text: str = "") -> bool:
    q = (question or "").lower()
    t = (expr_text or "").lower()
    return (
        "how many" in q
        or "number of" in q
        or " no. of" in q
        or "count of" in q
        or "count(" in t
    )


def _question_has_aggregate_intent(question: str, expr_text: str = "") -> bool:
    q = (question or "").lower()
    t = (expr_text or "").lower()
    markers = [
        "average", "avg", "mean", "sum", "total", "percentage", "percent", "ratio",
        "rate", "maximum", "minimum", "highest", "lowest", "most", "least",
    ]
    if any(m in q for m in markers):
        return True
    return any(tok in t for tok in ["avg(", "sum(", "max(", "min(", "count("])


def _extract_topk_from_question(question: str) -> Optional[int]:
    q = (question or "").lower()
    # e.g. top 5 / lowest three / first 2 / last one
    m = re.search(
        r"\b(top|lowest|highest|smallest|largest|first|last)\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\b",
        q,
    )
    if not m:
        return None
    tok = m.group(2)
    if tok.isdigit():
        return int(tok)
    return _NUM_WORD_TO_INT.get(tok)


def _question_expects_single_column(question: str, expr_text: str = "") -> bool:
    q = (question or "").lower()
    if not q:
        return False
    # Intents should be inferred from the question only, not from model-generated SQL/code text.
    if _question_has_count_intent(question):
        return True
    if _question_has_multi_output_intent(question):
        return False
    single_markers = [
        "what is",
        "who is",
        "which is",
        "when did",
        "where is",
        "where can",
        "name of",
        "names of",
        "title of",
        "phone number",
        "zip code",
        "website",
        "type of",
    ]
    if any(m in q for m in single_markers):
        return True
    k = _extract_topk_from_question(question)
    if k is not None:
        topk_single_markers = [
            "name", "names", "title", "titles", "code", "codes", "id", "ids",
            "zip", "phone", "website", "city", "country", "race", "driver",
            "rate", "ratio", "score", "average", "amount",
        ]
        return any(m in q for m in topk_single_markers)
    return False


def _result_row_arity(result: Any) -> int:
    if result is None:
        return 0
    row = result[0] if isinstance(result, list) and result else result
    if isinstance(row, dict):
        return len(row)
    if isinstance(row, (list, tuple)):
        return len(row)
    return 1


def _result_column_names_preview(result: Any, limit: int = 6) -> List[str]:
    row = result[0] if isinstance(result, list) and result else result
    if isinstance(row, dict):
        return [str(k) for k in list(row.keys())[:limit]]
    if isinstance(row, (list, tuple)):
        return [f"col{i}" for i in range(min(len(row), limit))]
    return ["value"]


def _extract_single_row_scalar(row: Any) -> Any:
    if isinstance(row, dict):
        # Prefer explicit count-like key when present.
        for k, v in row.items():
            if "count" in str(k).lower():
                return v
        if len(row) == 1:
            return next(iter(row.values()))
        return None
    if isinstance(row, (list, tuple)):
        return row[0] if len(row) == 1 else None
    return row


def _project_result_to_single_column(question: str, result: Any) -> tuple[Any, Optional[str]]:
    helper_col_hints = ("rank", "count", "avg", "sum", "total", "score", "ratio", "rate", "percent", "num", "number")
    q_tokens = set(re.findall(r"[a-z0-9]+", (question or "").lower()))

    def _pick_key(keys: List[str]) -> str:
        best_key = keys[0]
        best_score = -1e9
        for k in keys:
            kl = str(k).lower()
            k_tokens = set(re.findall(r"[a-z0-9]+", kl))
            overlap = len(q_tokens & k_tokens)
            penalty = 0.6 if any(h in kl for h in helper_col_hints) else 0.0
            score = overlap - penalty
            if score > best_score:
                best_score = score
                best_key = k
        return best_key

    if isinstance(result, dict) and len(result) > 1:
        key = _pick_key(list(result.keys()))
        return result.get(key), str(key)

    if isinstance(result, list) and result:
        first = result[0]
        if isinstance(first, dict):
            keys = list(first.keys())
            if len(keys) <= 1:
                return result, None
            key = _pick_key(keys)
            projected = [r.get(key) if isinstance(r, dict) else r for r in result]
            if len(projected) == 1:
                return projected[0], str(key)
            return projected, str(key)
        if isinstance(first, (list, tuple)) and len(first) > 1:
            projected = [r[0] if isinstance(r, (list, tuple)) and r else r for r in result]
            if len(projected) == 1:
                return projected[0], "col0"
            return projected, "col0"
    return result, None


def _apply_result_shape_repair(question: str, result: Any) -> tuple[Any, List[str]]:
    """
    Lightweight post-result repair to fix common shape mistakes:
    - count intent but returning entity rows
    - top-k intent with missing LIMIT
    - singular-intent question returning many rows
    """
    if result is None:
        return result, []

    repaired = result
    notes: List[str] = []
    row_count = _result_row_count(repaired)

    if _question_has_count_intent(question):
        if isinstance(repaired, list):
            if row_count > 1:
                repaired = row_count
                notes.append(f"count_intent: converted {row_count} rows to scalar count")
            elif row_count == 1:
                scalar = _extract_single_row_scalar(repaired[0])
                if scalar is not None and scalar is not repaired[0]:
                    repaired = scalar
                    notes.append("count_intent: reduced one-row structure to scalar")
        elif isinstance(repaired, dict):
            scalar = _extract_single_row_scalar(repaired)
            if scalar is not None and scalar is not repaired:
                repaired = scalar
                notes.append("count_intent: reduced dict to scalar")

    k = _extract_topk_from_question(question)
    if k and isinstance(repaired, list) and len(repaired) > k:
        repaired = repaired[:k]
        notes.append(f"topk_intent: truncated result to first {k} rows")

    if _question_expects_single_row(question) and isinstance(repaired, list) and len(repaired) > 1:
        repaired = repaired[:1]
        notes.append("single_row_intent: truncated to first row")

    if _question_expects_single_column(question) and _result_row_arity(repaired) > 1:
        projected, key = _project_result_to_single_column(question, repaired)
        if projected is not repaired:
            repaired = projected
            notes.append(f"single_column_intent: projected to one column ({key or 'best_guess'})")

    return repaired, notes


def _result_shape_mismatch_reasons(question: str, expr_text: str, result: Any) -> List[str]:
    reasons: List[str] = []
    if result is None:
        return ["result is None"]
    rows = _result_row_count(result)
    arity = _result_row_arity(result)

    # Do not infer intent from expr_text here; otherwise a wrong SQL with COUNT() can self-justify.
    if _question_has_count_intent(question):
        if rows != 1 or arity != 1:
            reasons.append(f"count intent expects scalar (1x1) but got rows={rows}, arity={arity}")

    k = _extract_topk_from_question(question)
    # top-k outputs may legitimately contain fewer than k rows if source rows are insufficient.
    if k is not None and rows > k:
        reasons.append(f"top-k intent expects at most {k} rows but got {rows}")

    if _question_expects_single_row(question) and rows > 1:
        reasons.append(f"singular intent likely expects 1 row but got {rows}")
    if _question_expects_single_column(question) and arity != 1:
        reasons.append(f"single-column intent expects arity=1 but got arity={arity}")
    return reasons


def _semantic_mismatch_reasons(question: str, expr_text: str, result: Any, mode: str) -> List[str]:
    """
    Lightweight semantic guardrails for known high-frequency failure modes.
    mode: "sql" | "code"
    """
    q = (question or "").lower()
    t = (expr_text or "").lower()
    reasons: List[str] = []

    # both A and B -> INTERSECT semantics, not UNION/OR
    if "both" in q:
        if mode == "sql":
            if (" intersect " not in f" {t} ") and ((" union " in f" {t} ") or (" or " in f" {t} ")):
                reasons.append("question says BOTH but SQL uses UNION/OR instead of INTERSECT")
        else:
            if ("&" not in t and ".intersection(" not in t and "intersect" not in t) and ("|" in t or " union " in f" {t} " or " or " in f" {t} "):
                reasons.append("question says BOTH but code uses union/or semantics")

    # youngest/oldest lexical consistency around age
    if "youngest" in q:
        if mode == "sql":
            if "max(age" in t or ("order by age desc" in t and "limit 1" in t):
                reasons.append("question says youngest but SQL uses max/desc age logic")
        else:
            if "idxmax" in t or ".max(" in t:
                reasons.append("question says youngest but code uses max logic")
    if "oldest" in q:
        if mode == "sql":
            if "min(age" in t or ("order by age asc" in t and "limit 1" in t):
                reasons.append("question says oldest but SQL uses min/asc age logic")
        else:
            if "idxmin" in t or ".min(" in t:
                reasons.append("question says oldest but code uses min logic")

    if _question_expects_single_row(question) and _result_row_count(result) > 1:
        reasons.append("question likely expects a single row but result has multiple rows")

    return reasons


def _format_tool_result(result: Dict[str, Any]) -> str:
    """将工具结果格式化为供 LLM 阅读的字符串。
    - schema/tables/suggestion：原样 JSON（有 4000 字符上限）
    - sample（read_table 返回）：最多 5 条完整行（不截断单行），行数多时附说明
    - result（execute_python 返回）：最多 20 条，4000 字符上限
    """
    if result.get("ok"):
        # execute_python result
        if result.get("result") is not None:
            r = result["result"]
            if isinstance(r, list) and len(r) > 20:
                r = r[:20] + [f"... and {len(r) - 20} more rows"]
            return json.dumps(r, ensure_ascii=False, default=str, indent=2)[:4000]

        # read_table sample：5 条完整行，不字符截断
        if result.get("sample") is not None:
            sample = result["sample"]
            row_count = result.get("row_count", -1)
            sampled = bool(result.get("sampled", False))
            read_mode = str(result.get("read_mode", "sample"))
            sample_limit = result.get("sample_limit", len(sample))
            keep = sample[:5]
            body = json.dumps(keep, ensure_ascii=False, default=str, indent=2)
            header = (
                f"[read_table] mode={read_mode}, sample_limit={sample_limit}, "
                f"row_count={row_count}, sampled={sampled}"
            )
            if len(sample) > 5 or (row_count > 5):
                total = row_count if row_count >= 0 else len(sample)
                body += f"\n... (showing 5/{total} rows)"
            note = result.get("note")
            if note:
                return f"{header}\n{note}\n{body}"
            return f"{header}\n{body}"

        # schema/tables/suggestion/join-path
        r = (
            result.get("schema")
            or result.get("tables")
            or (
                {
                    "tables_considered": result.get("tables_considered"),
                    "suggested_paths": result.get("suggested_paths"),
                    "fk_join_edges": result.get("fk_join_edges"),
                    "name_match_edges": result.get("name_match_edges"),
                    "note": result.get("note"),
                }
                if result.get("suggested_paths") is not None
                else None
            )
            or ({"suggestion": result.get("suggestion"), "reason": result.get("reason")} if result.get("suggestion") else None)
        )
        if r is None:
            r = "Done."
        return json.dumps(r, ensure_ascii=False, default=str, indent=2)[:4000]

    return f"Error: {result.get('error', 'Unknown')}"


def _extract_sql_from_text(text: str) -> str:
    """Extract likely SQL payload from free-form model text."""
    s = str(text or "").strip()
    if not s:
        return ""

    # Prefer fenced SQL block when present.
    m = re.search(r"```(?:sql)?\s*(.*?)```", s, flags=re.IGNORECASE | re.DOTALL)
    if m:
        s = m.group(1).strip()

    # Drop common leading labels.
    s = re.sub(r"^\s*(?:final\s+sql|sql|query)\s*[:：]\s*", "", s, flags=re.IGNORECASE)
    s = s.strip("`").strip()
    return s


def _first_sql_statement(sql: str) -> str:
    """Keep only the first SQL statement to avoid executing extra generated text."""
    s = str(sql or "").strip()
    if not s:
        return ""
    in_single = False
    in_double = False
    escape = False
    for i, ch in enumerate(s):
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == "'" and not in_double:
            in_single = not in_single
            continue
        if ch == '"' and not in_single:
            in_double = not in_double
            continue
        if ch == ";" and not in_single and not in_double:
            return s[:i].strip()
    return s


_SQL_IDENT_TRAILING_KEYWORDS = {
    "asc", "desc", "nulls", "first", "last", "is", "not", "in", "like",
    "between", "and", "or", "then", "else", "end", "when", "from", "where",
    "group", "order", "limit", "offset", "having", "join", "on", "as",
}


def _quote_dotted_space_identifiers(sql: str) -> str:
    """
    Fix common malformed identifier patterns like:
      s.School Type  -> s."School Type"
    while avoiding SQL keyword tails (e.g. `t.col desc`).
    """
    pat = re.compile(
        r"\b([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*(?:\s+[A-Za-z_][A-Za-z0-9_]*)+?)\b"
        r"(?=\s*(?:=|<>|!=|<|>|,|\)|$|like\b|in\b|is\b|between\b|order\b|group\b|limit\b|and\b|or\b))",
        flags=re.IGNORECASE,
    )

    def _repl(m: re.Match) -> str:
        alias = m.group(1)
        phrase = m.group(2).strip()
        toks = phrase.split()
        if len(toks) > 4:
            return m.group(0)
        low_toks = [t.lower() for t in toks]
        if any(t in _SQL_IDENT_TRAILING_KEYWORDS for t in low_toks):
            return m.group(0)
        if '"' in phrase:
            return m.group(0)
        return f'{alias}."{phrase}"'

    return pat.sub(_repl, sql)


def _post_process_sql_text(raw_sql: str, db_id: Optional[str] = None) -> tuple[str, Dict[str, Any]]:
    """
    Normalize SQL text before execution.
    This is intentionally lightweight and string-level only.
    """
    original = str(raw_sql or "")
    sql = _extract_sql_from_text(original)
    meta: Dict[str, Any] = {"raw_len": len(original), "changed": False}

    # Normalize smart quotes and invisible chars.
    quote_map = str.maketrans({
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u00a0": " ",
    })
    sql = sql.translate(quote_map).replace("\ufeff", "").replace("\u200b", "")

    # Clean markdown leftovers and collapse whitespace.
    sql = sql.replace("```sql", " ").replace("```", " ")
    sql = sql.replace("\n", " ").replace("\t", " ")
    sql = re.sub(r"\s+", " ", sql).strip()

    # Normalize common malformed operators and duplicated SELECT tokens.
    sql = sql.replace("> =", ">=").replace("< =", "<=").replace("! =", "!=")
    sql = re.sub(r"\bSELECT\s+SELECT\b", "SELECT", sql, flags=re.IGNORECASE)
    sql = re.sub(r"\bSELECT\s+SQL\b", "SELECT", sql, flags=re.IGNORECASE)

    # Keep only SQL body if model prefixed prose before SELECT/WITH.
    m = re.search(r"\b(select|with|pragma)\b", sql, flags=re.IGNORECASE)
    if m and m.start() > 0:
        sql = sql[m.start():].strip()

    # Keep exactly one statement for safety.
    sql = _first_sql_statement(sql).strip()

    # Spider/BIRD benchmark helper from prior pipelines.
    sql = re.sub(r"YEAR\s*\(\s*CURDATE\s*\(\s*\)\s*\)\s*", "2020 ", sql, flags=re.IGNORECASE)
    # SQLite compatibility: YEAR(col) / EXTRACT(YEAR FROM col) -> CAST(strftime('%Y', col) AS INTEGER)
    sql = re.sub(
        r"\bEXTRACT\s*\(\s*YEAR\s+FROM\s+([^)]+?)\s*\)",
        r"CAST(strftime('%Y', \1) AS INTEGER)",
        sql,
        flags=re.IGNORECASE,
    )
    sql = re.sub(
        r"\bYEAR\s*\(\s*([^)]+?)\s*\)",
        r"CAST(strftime('%Y', \1) AS INTEGER)",
        sql,
        flags=re.IGNORECASE,
    )

    # Remove accidental database prefix in SQLite benchmarks (db.table -> table).
    if db_id:
        sql = re.sub(rf"\b{re.escape(db_id)}\s*\.", "", sql, flags=re.IGNORECASE)

    # Repair unquoted alias.column-with-space patterns.
    sql = _quote_dotted_space_identifiers(sql)

    sql = sql.rstrip(";").strip()
    meta["processed_len"] = len(sql)
    meta["changed"] = (sql != original.strip())
    return sql, meta


_SCHEMA_PRUNE_THRESHOLD = 8   # 表数超过此值时才做 schema pruning


def _prune_tables_by_question(
    question: str,
    table_names: List[str],
    table_cols: Optional[Dict[str, List[str]]] = None,
    keep_top: int = 8,
) -> List[str]:
    """
    轻量 schema pruning（零 API 调用）：
    当表数超过 _SCHEMA_PRUNE_THRESHOLD 时，用 keyword 匹配筛出最相关的表。
    - 问题中出现的词 vs 表名 / 列名（忽略大小写、下划线转空格）
    - 始终保留 MV 表（以 mv_ 开头，已是预计算结果）
    - 至少保留 keep_top 张最相关的表

    Args:
        table_names: 所有表名列表
        table_cols : {table_name: [col_name, ...]}，有则用于辅助评分
        keep_top   : 最多保留的普通表数量
    """
    preferred_module = str(os.environ.get("NL2CODE_COOPT_PREFERRED_MODULE", "") or "").strip().lower()
    import re
    # 问题 token 集合（小写、字母数字）
    q_tokens = set(re.findall(r"[a-z0-9]+", question.lower()))
    # 停用词（几乎无辨别力）
    stopwords = {"the", "a", "an", "in", "of", "for", "what", "how", "many",
                 "all", "list", "show", "find", "get", "give", "is", "are",
                 "that", "which", "with", "from", "where", "by", "and", "or",
                 "to", "on", "at", "id", "name"}
    q_tokens -= stopwords

    mv_tables = [t for t in table_names if t.startswith("mv_")]
    regular_tables = [t for t in table_names if not t.startswith("mv_")]

    def _score(tname: str) -> float:
        # 拆解表名为 tokens
        t_tokens = set(re.findall(r"[a-z0-9]+", tname.lower()))
        score = len(q_tokens & t_tokens)
        # 加权列名匹配
        if table_cols and tname in table_cols:
            for col in table_cols[tname]:
                c_tokens = set(re.findall(r"[a-z0-9]+", col.lower()))
                score += 0.3 * len(q_tokens & c_tokens)
        return score

    def _rank(names: List[str]) -> List[str]:
        return sorted(names, key=_score, reverse=True)

    # Co-opt strict-all routing hint:
    # keep all physical modules available, but narrow schema exposure by intent.
    if preferred_module in {"index", "layout"}:
        if len(regular_tables) <= _SCHEMA_PRUNE_THRESHOLD:
            return regular_tables
        return _rank(regular_tables)[:keep_top]

    if preferred_module == "mv":
        mv_keep = max(1, int(_env_float("NL2CODE_COOPT_MV_KEEP_TOP", 3.0)))
        scored_mv = _rank(mv_tables)
        scored_regular = _rank(regular_tables)
        if scored_mv:
            return scored_mv[:mv_keep] + scored_regular[:2]
        if len(regular_tables) <= _SCHEMA_PRUNE_THRESHOLD:
            return regular_tables
        return scored_regular[:keep_top]

    if len(table_names) <= _SCHEMA_PRUNE_THRESHOLD:
        return table_names  # 表少，不裁剪

    scored_regular = _rank(regular_tables)
    if mv_tables:
        # Keep only top-MV tables to avoid prompt bloat on joint optimization.
        mv_keep = max(1, int(_env_float("NL2CODE_COOPT_MV_KEEP_TOP", 3.0)))
        scored_mv = _rank(mv_tables)[:mv_keep]
        return scored_regular[:keep_top] + scored_mv
    return scored_regular[:keep_top]


def _build_fk_hint(db_path: Optional[str], tables: List[str]) -> Optional[str]:
    """
    从 SQLite 提取 FK 关系，生成简洁的提示行，帮助 LLM 正确 JOIN 表。
    格式：-- Foreign Keys:\n  <child_table>.<col> → <parent_table>.<parent_col>
    仅输出 tables 列表中涉及的 FK。
    """
    if not db_path or not os.path.exists(db_path):
        return None
    if str(db_path).lower().endswith(".csv"):
        return None
    try:
        import sqlite3
        conn = sqlite3.connect(db_path)
        fk_lines = []
        tables_set = set(t.lower() for t in tables)
        for t in tables:
            rows = conn.execute(f'PRAGMA foreign_key_list("{t}")').fetchall()
            for row in rows:
                # row: (id, seq, table, from, to, ...)
                _, _, ref_table, from_col, to_col = row[0], row[1], row[2], row[3], row[4]
                # 只输出 parent 也在 pruned tables 里的 FK（避免输出已被裁剪掉的表）
                if ref_table.lower() in tables_set:
                    to_display = to_col if to_col else from_col
                    fk_lines.append(f"  {t}.{from_col} → {ref_table}.{to_display}")
        conn.close()
        if fk_lines:
            return "-- Foreign Keys (use these to JOIN tables):\n" + "\n".join(fk_lines)
    except Exception:
        pass
    return None


def _build_schema_hint(ctx: AgentContext) -> Optional[str]:
    """
    预先获取 schema 并注入到 user message，避免 list_tables/get_schema 热身轮。
    对大型 schema（表数 > 8）自动做 keyword-based pruning，减少 prompt 噪声。
    返回紧凑 schema 字符串（tables + columns + optimization hints），失败返回 None。
    """
    parquet_dir = getattr(ctx, "parquet_dir", None)
    try:
        if parquet_dir and os.path.isdir(parquet_dir):
            from .tools import _list_parquet_tables, _parquet_schema_to_str, _load_manifest, _mv_semantic_notes
            all_tables = _list_parquet_tables(parquet_dir)
            manifest = _load_manifest(parquet_dir)
            mv_notes = _mv_semantic_notes(manifest)

            # ── schema pruning（仅大型 schema）───────────────────────────
            tables = _prune_tables_by_question(ctx.question, all_tables)
            if len(tables) < len(all_tables):
                pruned_note = (f"[Schema pruned: showing {len(tables)}/{len(all_tables)} tables "
                               f"most relevant to your question. Call get_schema() for the full schema.]")
            else:
                pruned_note = None

            # ── index/categorical hints ──────────────────────────────────
            index_hints: Dict[str, Dict] = {}
            if manifest:
                for rec in manifest.get("index_recs", []):
                    tname = rec.get("table")
                    if not tname:
                        continue
                    entry: Dict[str, Any] = {}
                    if rec.get("pandas_index_col"):
                        entry["pandas_index_col"] = rec["pandas_index_col"]
                    if rec.get("categorical_cols"):
                        entry["categorical_cols"] = rec["categorical_cols"]
                    if entry:
                        index_hints[tname] = entry

            schema_str = _parquet_schema_to_str(parquet_dir, tables, mv_notes=mv_notes, index_hints=index_hints)
            if pruned_note:
                schema_str = pruned_note + "\n\n" + schema_str

            # ── FK 信息补充（从 SQLite 原始文件提取） ────────────────────────
            fk_str = _build_fk_hint(ctx.db_path, tables)
            if fk_str:
                schema_str += "\n\n" + fk_str
            return schema_str

        # SQLite fallback
        db_path = ctx.db_path
        if db_path and os.path.exists(db_path):
            if str(db_path).lower().endswith(".csv"):
                schema_res = tool_get_schema(
                    db_path=db_path,
                    spider_tables=None,
                    db_id=None,
                    use_mschema=False,
                    parquet_dir=None,
                )
                if schema_res.get("ok"):
                    return schema_res.get("schema")
                return None
            import sqlite3
            conn = sqlite3.connect(db_path)
            all_tables = [r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            ).fetchall()]
            # 提取列名供 pruning 评分
            table_cols: Dict[str, List[str]] = {}
            for t in all_tables:
                table_cols[t] = [r[1] for r in conn.execute(f"PRAGMA table_info(\"{t}\")").fetchall()]
            conn.close()

            tables = _prune_tables_by_question(ctx.question, all_tables, table_cols=table_cols)
            lines = [f"{t}({', '.join(table_cols[t])})" for t in tables]
            fk_str = _build_fk_hint(db_path, tables)
            if fk_str:
                lines.append(fk_str)
            return "\n".join(lines)
    except Exception:
        pass
    return None


def _compress_old_tool_messages(messages: List[Dict[str, Any]], keep_last_tool: int = 1) -> List[Dict[str, Any]]:
    """
    将历史 tool 消息中已消费（非最近 keep_last_tool 条）的大型结果压缩为摘要，
    避免 read_table 结果永久累积在 context 中，降低后续轮次的 prompt token 数。
    只压缩 role=='tool' 且内容 > 500 chars 的消息；保留 error 消息不压缩。
    """
    tool_indices = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
    # 保留最后 keep_last_tool 条不压缩
    to_compress = tool_indices[:-keep_last_tool] if len(tool_indices) > keep_last_tool else []
    compressed = list(messages)
    for idx in to_compress:
        content = compressed[idx].get("content", "")
        if len(content) > 500 and "Error:" not in content[:50]:
            # 提取前 120 chars 作为摘要（含列名等关键信息）
            summary = content[:120].rstrip()
            compressed[idx] = {**compressed[idx], "content": f"[Compressed] {summary} ..."}
    return compressed


def _dataset_output_contract(ctx: AgentContext) -> str:
    ds = (ctx.dataset or "").lower()
    if ds == "tabfact":
        return (
            "\n\n[Output contract for TabFact]\n"
            "- This is table fact verification.\n"
            "- The dataset exposes a single table named `data`.\n"
            "- Final answer MUST be an integer label: 1 for ENTAILED, 0 for REFUTED.\n"
            "- In execute_python, set `result = 1` or `result = 0` only (no explanation dict/string).\n"
        )
    if ds == "wtq":
        return (
            "\n\n[Output contract for WikiTableQuestions]\n"
            "- The dataset exposes a single table named `data`.\n"
            "- Return answer value(s) only; no explanation text.\n"
            "- Single answer: scalar string/number.\n"
            "- Multi-answer: list of scalar values.\n"
            "- Use exact value surface form for final answer (avoid unexplained abbreviation).\n"
            "- Do not append extra annotation like '(USA)' or '(10)' unless explicitly asked.\n"
            "- For count questions, return one exact scalar count, not a row/list preview.\n"
        )
    return ""


def _build_retrieved_skill_block(ctx: AgentContext, mode: str) -> Dict[str, Any]:
    """
    Build a compact retrieved-skill hint block.
    This is advisory only; tools and execution remain unchanged.
    """
    try:
        top_k = int(getattr(ctx, "skill_top_k", 0) or 0)
    except Exception:
        top_k = 0
    if top_k <= 0:
        return {"text": "", "count": 0, "skill_ids": []}
    bank_path = getattr(ctx, "skill_bank_path", None)
    if not bank_path:
        return {"text": "", "count": 0, "skill_ids": []}
    try:
        max_chars = int(getattr(ctx, "skill_max_chars", 900) or 900)
    except Exception:
        max_chars = 900
    hint = build_skill_hint(
        question=ctx.question,
        dataset=ctx.dataset,
        mode=mode,
        skill_bank_path=bank_path,
        top_k=top_k,
        max_chars=max(200, max_chars),
    )
    return hint if isinstance(hint, dict) else {"text": "", "count": 0, "skill_ids": []}


def _collect_runtime_snapshot() -> Dict[str, Any]:
    """
    Lightweight host snapshot without optional dependencies.
    """
    cpu_load_pct = 0.0
    mem_used_pct = 0.0
    disk_used_pct = 0.0
    cpu_count = max(1, int(os.cpu_count() or 1))
    try:
        load1 = float(os.getloadavg()[0])
        cpu_load_pct = max(0.0, min(100.0, (load1 / float(cpu_count)) * 100.0))
    except Exception:
        cpu_load_pct = 0.0
    try:
        mem_total_kb = 0.0
        mem_avail_kb = 0.0
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for ln in f:
                if ln.startswith("MemTotal:"):
                    mem_total_kb = float((ln.split() or [0, 0])[1])
                elif ln.startswith("MemAvailable:"):
                    mem_avail_kb = float((ln.split() or [0, 0])[1])
        if mem_total_kb > 0.0:
            mem_used_pct = max(0.0, min(100.0, (1.0 - (mem_avail_kb / mem_total_kb)) * 100.0))
    except Exception:
        mem_used_pct = 0.0
    try:
        du = shutil.disk_usage("/")
        if du.total > 0:
            disk_used_pct = max(0.0, min(100.0, (float(du.used) / float(du.total)) * 100.0))
    except Exception:
        disk_used_pct = 0.0
    return {
        "cpu_load_pct": round(float(cpu_load_pct), 2),
        "mem_used_pct": round(float(mem_used_pct), 2),
        "disk_used_pct": round(float(disk_used_pct), 2),
        "cpu_count": int(cpu_count),
        "thread_budget_hint": int(cpu_count),
    }


def _diagnose_runtime_bottleneck(
    *,
    turn: int,
    turn_tool_times: List[Dict[str, Any]],
    read_table_stats: Dict[str, Any],
    execute_error_count: int,
) -> Dict[str, Any]:
    tool_sum: Dict[str, float] = {}
    total_s = 0.0
    for e in list(turn_tool_times or []):
        tool = str(e.get("tool", "") or "")
        dur = float(e.get("duration_s", 0.0) or 0.0)
        if not tool or dur < 0.0:
            continue
        tool_sum[tool] = float(tool_sum.get(tool, 0.0) + dur)
        total_s += dur
    dominant_tool = ""
    dominant_s = 0.0
    for k, v in tool_sum.items():
        if float(v) > float(dominant_s):
            dominant_tool = str(k)
            dominant_s = float(v)
    dominant_pct = (dominant_s / total_s * 100.0) if total_s > 1e-9 else 0.0

    sampled_reads = int(read_table_stats.get("sample_read_calls", 0) or 0)
    full_reads = int(read_table_stats.get("full_read_calls", 0) or 0)

    bottleneck_class = "mixed"
    advice = "Prefer one-pass vectorized execute_python and avoid redundant tool calls."
    if execute_error_count > 0:
        bottleneck_class = "error_repair_bound"
        advice = "Stabilize key dtypes before merge/filter and keep final computation in a single execute_python call."
    elif dominant_tool == "read_table":
        if sampled_reads > full_reads:
            bottleneck_class = "io_sampling_bound"
            advice = "Avoid repeated read_table on the same table; promote to one full read only when exact aggregation is required."
        else:
            bottleneck_class = "io_scan_bound"
            advice = "Push tighter column/filter selection before full scans and reuse loaded DataFrames."
    elif dominant_tool == "execute_python":
        if full_reads > 0:
            bottleneck_class = "compute_fullscan_bound"
            advice = "Use vectorized operations (groupby/merge/agg), avoid Python loops, and reduce intermediate copies."
        else:
            bottleneck_class = "compute_transform_bound"
            advice = "Consolidate transforms into one execute_python and keep operations column-pruned."
    elif dominant_tool == "query_sql":
        bottleneck_class = "sql_engine_bound"
        advice = "Prefer earlier predicate pushdown and narrower projections; keep SQL attempt count minimal."

    return {
        "turn": int(turn),
        "dominant_tool": str(dominant_tool or "none"),
        "dominant_s": round(float(dominant_s), 4),
        "dominant_pct": round(float(dominant_pct), 1),
        "total_tool_s": round(float(total_s), 4),
        "class": str(bottleneck_class),
        "advice": str(advice),
    }


def _build_runtime_profile_prompt(shared_profile: Dict[str, Any]) -> str:
    snap = dict((shared_profile or {}).get("runtime_snapshot") or {})
    phy = dict((shared_profile or {}).get("physical_state") or {})
    policy = dict((shared_profile or {}).get("execution_policy") or {})
    bott = dict((shared_profile or {}).get("last_bottleneck") or {})
    cpu = float(snap.get("cpu_load_pct", 0.0) or 0.0)
    mem = float(snap.get("mem_used_pct", 0.0) or 0.0)
    disk = float(snap.get("disk_used_pct", 0.0) or 0.0)
    threads = int(snap.get("thread_budget_hint", snap.get("cpu_count", 1)) or 1)
    bcls = str(bott.get("class", "cold_start") or "cold_start")
    btool = str(bott.get("dominant_tool", "none") or "none")
    badvice = str(
        bott.get("advice")
        or "Use one-pass vectorized execute_python; avoid redundant reads and repair retries."
    )
    phy_pressure = float(
        policy.get("pressure_score", phy.get("pressure_score", 0.0)) or 0.0
    )
    runtime_mode = str(policy.get("runtime_mode", "balanced") or "balanced")
    worker_target = int(policy.get("worker_target", threads) or threads)
    fast_turn_cap = int(policy.get("fast_turn_cap", 0) or 0)
    maintenance_hold = bool(policy.get("maintenance_hold", False))
    policy_hint = "Prefer one-pass vectorized execution and minimal retries."
    if runtime_mode == "pressure_relief" or phy_pressure >= 0.8:
        policy_hint = (
            "Host is under pressure: avoid redundant query_sql retries, avoid repeated read_table on same table, "
            "and keep exactly one final execute_python block with result=..."
        )
    elif runtime_mode == "throughput":
        policy_hint = (
            "Host has headroom: still keep vectorized one-pass code, but prioritize exact full-read aggregation "
            "when needed for correctness."
        )
    return (
        "[Shared Runtime Profile]\n"
        f"- host_cpu_load_pct={cpu:.1f}, host_mem_used_pct={mem:.1f}, host_disk_used_pct={disk:.1f}\n"
        f"- thread_budget_hint={threads}\n"
        f"- execution_policy: mode={runtime_mode}, worker_target={worker_target}, "
        f"fast_turn_cap={fast_turn_cap}, maintenance_hold={int(maintenance_hold)}, pressure_score={phy_pressure:.3f}\n"
        f"- latest_bottleneck={bcls} (dominant_tool={btool})\n"
        f"- controller_advice={badvice}\n"
        f"- policy_hint={policy_hint}\n"
        "When writing execute_python: prefer vectorized pandas, avoid repeated read_table on same table, "
        "avoid Python loops and repeated SQL retries, and keep final answer in one execution block with `result=...`."
    )


def _code_tool_defs(ctx: AgentContext) -> List[Dict[str, Any]]:
    """
    Dataset-aware tool exposure for code path.
    WTQ/TabFact are single-table by default, so hide join-path tool to reduce noise.
    """
    tools = get_tool_definitions()
    if bool(getattr(ctx, "sql_free_only", False)):
        blocked = {"query_sql", "suggest_olap"}
        tools = [
            t for t in tools
            if t.get("function", {}).get("name") not in blocked
        ]

    allowlist = getattr(ctx, "code_tool_allowlist", None)
    if allowlist:
        allow = {str(x).strip() for x in allowlist if str(x).strip()}
        if allow:
            tools = [
                t for t in tools
                if t.get("function", {}).get("name") in allow
            ]

    denylist = getattr(ctx, "code_tool_denylist", None)
    if denylist:
        deny = {str(x).strip() for x in denylist if str(x).strip()}
        if deny:
            tools = [
                t for t in tools
                if t.get("function", {}).get("name") not in deny
            ]

    ds = (ctx.dataset or "").lower()
    if ds in {"wtq", "tabfact"}:
        return [t for t in tools if t.get("function", {}).get("name") != "get_join_path"]
    return tools


def run_react_agent(
    ctx: AgentContext,
    trajectory_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """
    ReAct 主循环：Think -> Act -> Observe -> Reflect
    trajectory_callback(traj_event): 每步关键决策/调用节点回调，用于 agentic 轨迹记录
    """
    client = _build_openai_client(ctx)
    if not client:
        return {"ok": False, "error": "OpenAI client not available", "trace": [], "trajectory": []}

    trajectory: List[Dict[str, Any]] = []

    def _log(event: Dict[str, Any]) -> None:
        trajectory.append(event)
        if trajectory_callback:
            trajectory_callback(event)

    system_prompt = """You are a SQL-Free data analyst agent. Use **Python/pandas only**—no SQL.

**Workflow (2Code):**
1. The schema is already provided in the user message. Skip list_tables/get_schema unless you need more detail.
2. For multi-table questions, call get_join_path first to obtain reliable join keys/path.
3. read_table: load table(s). For JOINs: read_table("t1"), read_table("t2"), then execute_python with df_t1, df_t2.
   - Default mode is sampled preview rows. You may pass limit (e.g. limit=200) or mode='full' when exact full-table computation is necessary.
4. execute_python: write ALL computation in a SINGLE call. df_<tablename> is available (e.g. df_singer, df_concert). **Always end with `result = <your answer>`**.
5. Done. Do not use tools after you have the result.

**Rules:**
- Do NOT use SQL. All analysis via read_table + execute_python.
- For JOIN/merge across tables, prefer get_join_path suggestions instead of guessing key columns.
- **Skip describe_table**: it is rarely needed since read_table already shows columns and sample data.
- **read_table once per table**: DataFrame df_t persists across all subsequent execute_python calls—do NOT read the same table again.
- **Sampling awareness**: read_table observation includes sampled=True/False and row_count. If sampled=True, do NOT trust it for exact global counts/sums unless you explicitly load full data.
- **df variable naming**: read_table("TableName") → df_TableName AND df_tablename both work.
- **execute_python is NOT persistent between calls**: variables from a previous execute_python call are NOT available. Put ALL computation in ONE call and end with `result = ...`. Never call execute_python with just `result` alone.
- **Repair**: When a tool returns an Error, analyze it and retry with a corrected approach.
- **Dirty data**: use df.dropna(), df.drop_duplicates(), or filter invalid rows before computing.
- **Always set result=... in execute_python** (e.g. result=len(df) or result=df[['col']].to_dict('records')).

**Set semantics & aggregation:**
- **Empty result**: set `result = []` and do not invent rows.
- For INTERSECT / NOT_IN / EXCEPT / UNION / GROUP_AGG / NESTED_AGG patterns: call `get_code_template(pattern=...)` to get a ready-to-adapt pandas template before writing execute_python code.
"""
    system_prompt += _dataset_output_contract(ctx)

    # ── Schema 预注入：避免 list_tables/get_schema 热身轮 ─────────────────────
    schema_hint = _build_schema_hint(ctx)

    user_content = f"Question: {ctx.question}\n\nDatabase: {ctx.db_id}"
    user_content += _dataset_output_contract(ctx)
    if schema_hint:
        user_content += f"\n\n**Schema (use this directly, no need to call list_tables or get_schema):**\n{schema_hint}"
    if getattr(ctx, "evidence", None) and str(ctx.evidence).strip():
        user_content += f"\n\n**External knowledge (use this to interpret the question):**\n{ctx.evidence.strip()}"
        _log({"event": "evidence_provided", "evidence": ctx.evidence[:200]})
    skill_hint = _build_retrieved_skill_block(ctx, mode="code")
    if skill_hint.get("text"):
        user_content += (
            "\n\n[Retrieved skills from successful trajectories - advisory only]\n"
            + str(skill_hint["text"])
        )
        _log(
            {
                "event": "skill_retrieved",
                "mode": "code",
                "count": int(skill_hint.get("count", 0) or 0),
                "skill_ids": list(skill_hint.get("skill_ids") or [])[:8],
            }
        )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    import time as _time

    def _is_empty_result(val: Any) -> bool:
        """是否为「空结果」：仅 []、{}、None、'' 及空 pandas Series/DataFrame 视为空；0/0.0/False 视为有效答案"""
        if val is None:
            return True
        if isinstance(val, (list, tuple, dict)) and len(val) == 0:
            return True
        if isinstance(val, str) and val.strip() == "":
            return True
        # pandas Series / DataFrame：空则视为无效结果
        try:
            import pandas as _pd
            if isinstance(val, (_pd.Series, _pd.DataFrame)) and len(val) == 0:
                return True
        except ImportError:
            pass
        return False

    trace = []
    final_answer = None
    predicted_result = None  # 最后一次 execute_python 的结构化结果，供 eval 对比 gold SQL
    empty_result_retry_used = False    # 空结果自查：仅允许一次「给 LLM 再试一轮」
    error_retry_used = False           # execute_python 报错自查：仅允许一次「请修复后重试」
    shape_retry_used = False           # 结果形状不匹配（count/top-k/单列）仅允许一次重试
    semantic_retry_used = False        # 语义守卫（set-op/top1/youngest-oldest）仅允许一次
    sampled_agg_retry_used = False     # 聚合问题但只在 sampled 数据上计算时，提示一次 full-read 重算
    last_read_sample = None
    samples_dict = {}  # table_name -> sample rows（list of dicts），供多表 merge
    parquet_df_cache: Dict[str, Any] = {}  # table_name -> optimized DataFrame（parquet 模式）
    last_execute_code: Optional[str] = None
    read_table_stats: Dict[str, Any] = {
        "calls": 0,
        "tables": set(),
        "sample_read_calls": 0,
        "full_read_calls": 0,
        "sampled_true_calls": 0,
        # Number of read_table calls where runtime bound a full DataFrame behind sampled preview.
        "full_df_bind_calls": 0,
        "full_df_bound_tables": set(),
    }
    read_table_row_counts: Dict[str, int] = {}
    execute_error_count = 0

    # ── 用量统计 ─────────────────────────────────────────────────────────────
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    # 每次工具调用的耗时：[{"tool": "...", "duration_s": ...}, ...]
    tool_call_times: List[Dict[str, Any]] = []

    _log({"event": "start", "question": ctx.question, "db_id": ctx.db_id, "model": ctx.model,
          "evidence_provided": bool(getattr(ctx, "evidence", None) and str(ctx.evidence or "").strip())})
    coopt_fast = _coopt_fast_enabled(ctx)
    runtime_profile_enabled = bool(getattr(ctx, "enable_runtime_profile", True))
    runtime_profile: Dict[str, Any] = dict(getattr(ctx, "shared_profile", {}) or {})
    if runtime_profile_enabled:
        runtime_profile.setdefault("profile_source", "agent_runtime")
        runtime_profile.setdefault("bottleneck_history", [])
        runtime_profile["runtime_snapshot"] = _collect_runtime_snapshot()
        runtime_profile.setdefault(
            "last_bottleneck",
            {
                "turn": -1,
                "class": "cold_start",
                "dominant_tool": "none",
                "advice": "Use one-pass vectorized execute_python; avoid redundant reads and retries.",
            },
        )

    for turn in range(ctx.max_turns):
        # ── 压缩旧 tool 结果，防止 prompt 无限膨胀 ──────────────────────────
        send_messages = _compress_old_tool_messages(messages, keep_last_tool=(1 if coopt_fast else 2))
        if runtime_profile_enabled:
            runtime_profile["runtime_snapshot"] = _collect_runtime_snapshot()
            send_messages = list(send_messages) + [
                {"role": "system", "content": _build_runtime_profile_prompt(runtime_profile)}
            ]

        _t_llm_start = _time.perf_counter()
        create_kwargs = dict(
            model=ctx.model,
            messages=send_messages,
            tools=_code_tool_defs(ctx),
            tool_choice="auto",
            timeout=120.0,
        )
        # 关闭/减弱“思考”：部分后端（OpenAI 兼容、ChatAnywhere 等）支持 reasoning_effort，可减少延迟与 token
        if getattr(ctx, "reasoning_effort", None):
            create_kwargs["reasoning_effort"] = ctx.reasoning_effort
        try:
            resp = client.chat.completions.create(**create_kwargs)
        except Exception as e:
            _log({"event": "error", "turn": turn, "error": str(e)})
            return {"ok": False, "error": str(e), "trace": trace, "trajectory": trajectory,
                    "total_prompt_tokens": total_prompt_tokens,
                    "total_completion_tokens": total_completion_tokens,
                    "tool_call_times": tool_call_times}
        llm_duration_s = round(_time.perf_counter() - _t_llm_start, 3)

        # 记录本轮 token
        usage = getattr(resp, "usage", None)
        if usage:
            total_prompt_tokens    += getattr(usage, "prompt_tokens", 0) or 0
            total_completion_tokens += getattr(usage, "completion_tokens", 0) or 0

        choice = resp.choices[0]
        msg = choice.message
        assistant_content = msg.content or ""
        messages.append({
            "role": "assistant",
            "content": assistant_content,
            "tool_calls": getattr(msg, "tool_calls", None),
        })

        # 关键决策：LLM 的 reasoning（llm_duration_s 便于排查哪轮 API 慢）
        _log({"event": "think", "turn": turn, "content": assistant_content[:500] if assistant_content else None,
              "prompt_tokens": getattr(usage, "prompt_tokens", None) if usage else None,
              "completion_tokens": getattr(usage, "completion_tokens", None) if usage else None,
              "llm_duration_s": llm_duration_s})

        all_tcs = _extract_all_tool_calls(msg)
        if not all_tcs:
            final_answer = assistant_content or "No answer"
            _log({"event": "finish", "turn": turn, "answer": final_answer[:500]})
            break

        # 对每个 tool_call 执行并添加 tool 响应（API 要求每个 tool_call_id 都必须有响应）
        turn_tool_times: List[Dict[str, Any]] = []
        for tc in all_tcs:
            _log({"event": "act", "turn": turn, "tool": tc["name"], "args": tc["arguments"]})
            _t_tool_start = _time.perf_counter()
            tool_result = run_tool(
                tc["name"], tc["arguments"], ctx,
                last_read_sample=last_read_sample,
                samples_dict=samples_dict,
                parquet_df_cache=parquet_df_cache,
            )
            tool_duration = round(_time.perf_counter() - _t_tool_start, 4)
            tool_call_times.append({"tool": tc["name"], "duration_s": tool_duration,
                                    "ok": tool_result.get("ok", False)})
            turn_tool_times.append({"tool": tc["name"], "duration_s": tool_duration})

            if tc["name"] == "read_table" and tool_result.get("ok"):
                tname = tc["arguments"].get("table", "unknown")
                last_read_sample = tool_result.get("sample")
                if tname:
                    samples_dict[tname] = last_read_sample
                    read_table_stats["tables"].add(str(tname))
                    try:
                        read_table_row_counts[str(tname)] = int(tool_result.get("row_count", -1))
                    except Exception:
                        read_table_row_counts[str(tname)] = -1
                read_table_stats["calls"] += 1
                read_mode = str(tool_result.get("read_mode", "sample")).lower()
                if read_mode == "full":
                    read_table_stats["full_read_calls"] += 1
                else:
                    read_table_stats["sample_read_calls"] += 1
                if bool(tool_result.get("sampled", False)):
                    read_table_stats["sampled_true_calls"] += 1
                # parquet 模式：额外加载带 categorical 优化的 DataFrame（单文件或分区表）
                parquet_dir = getattr(ctx, "parquet_dir", None)
                if parquet_dir and tname:
                    try:
                        from .tools import (
                            _load_parquet_table_to_pandas,
                            _get_index_rec_for_table,
                        )
                        part_filter = tc["arguments"].get("partition_filter")
                        # Avoid reloading the same full parquet table repeatedly.
                        should_load = True
                        if not part_filter and tname in parquet_df_cache and parquet_df_cache.get(tname) is not None:
                            should_load = False
                        if should_load:
                            idx_rec = _get_index_rec_for_table(parquet_dir, tname)
                            cat_hint = idx_rec.get("categorical_cols") if idx_rec else None
                            pd_idx_col = idx_rec.get("pandas_index_col") if idx_rec else None
                            loaded = _load_parquet_table_to_pandas(
                                parquet_dir, tname,
                                partition_filter=part_filter,
                                categorical_hint=cat_hint,
                                pandas_index_col=pd_idx_col,
                            )
                            if loaded is not None:
                                parquet_df_cache[tname] = loaded
                    except Exception:
                        pass
                # CSV TQA: keep prompt observation sampled, but bind full DataFrame for execute_python.
                if (
                    str(getattr(ctx, "dataset", "") or "").lower() in {"wtq", "tabfact"}
                    and str(getattr(ctx, "db_path", "") or "").lower().endswith(".csv")
                    and tname
                ):
                    try:
                        from .tools import _csv_table_aliases, _load_csv_df

                        csv_df = _load_csv_df(ctx.db_path)
                        alias_set = {str(tname), "data"}
                        for alias in _csv_table_aliases(ctx.db_path):
                            alias_set.add(str(alias))
                        for alias in alias_set:
                            safe_alias = alias.replace("-", "_").replace(" ", "_")
                            parquet_df_cache[safe_alias] = csv_df
                        read_table_stats["full_df_bind_calls"] += 1
                        read_table_stats["full_df_bound_tables"].add(str(tname))
                        tool_result["runtime_full_df_bound"] = True
                    except Exception:
                        pass

            result_str = _format_tool_result(tool_result)
            result_preview = result_str[:300] + "..." if len(result_str) > 300 else result_str

            # read_table 成功后追加"状态已就绪"提示，防止 LLM 重复读同一张表
            if tc["name"] == "read_table" and tool_result.get("ok"):
                tname = tc["arguments"].get("table", "unknown")
                # 收集所有已加载的表
                loaded = sorted(set(list(samples_dict.keys()) + list(parquet_df_cache.keys())))
                loaded_hint = ", ".join(f"df_{t}" for t in loaded)
                result_str = result_str + (
                    f"\n\n[State] df_{tname} is now registered. "
                    f"Do NOT call read_table(\"{tname}\") again. "
                    f"Currently loaded DataFrames: {loaded_hint}. "
                    f"Proceed to execute_python to compute the answer."
                )
                if bool(tool_result.get("runtime_full_df_bound", False)):
                    result_str += (
                        "\n[Runtime] execute_python will use a full in-memory DataFrame for this table; "
                        "the sampled rows above are only a preview."
                    )

            # execute_python 成功且有 result：非空则采纳并结束；空结果则触发一次「自查重试」
            if tc["name"] == "execute_python" and tool_result.get("ok") and tool_result.get("result") is not None:
                last_execute_code = str(tc["arguments"].get("code", "") or "")
                res = tool_result["result"]
                if _is_empty_result(res):
                    if not empty_result_retry_used:
                        empty_result_retry_used = True
                        result_str += (
                            "\n\n[Self-check] Result is empty. Common causes:\n"
                            "1. TYPE MISMATCH on join/filter key — e.g. concert.Stadium_ID is VARCHAR '1' but "
                            "stadium.Stadium_ID is BIGINT 1. Fix: cast to same type before comparing, e.g. "
                            "df_concert['Stadium_ID'].astype(int) or df_stadium['Stadium_ID'].astype(str).\n"
                            "2. Over-restrictive filter — check string case, exact spelling, or numeric range.\n"
                            "3. Wrong table or column chosen.\n"
                            "Look at the schema types (VARCHAR vs BIGINT) for each column involved in your "
                            "merge/filter, fix the mismatch, and call execute_python again. (One retry allowed.)"
                        )
                        # 不设置 predicted_result，本轮不 break，给 LLM 再试一轮
                    else:
                        predicted_result = res
                else:
                    valid, invalid_reason = _validate_candidate_result(res)
                    if valid:
                        needs_exact = _question_has_count_intent(ctx.question, last_execute_code or "") or _question_has_aggregate_intent(ctx.question, last_execute_code or "")
                        sample_only_reads = (
                            int(read_table_stats.get("sample_read_calls", 0) or 0) > 0
                            and int(read_table_stats.get("full_read_calls", 0) or 0) == 0
                        )
                        row_counts = [v for v in read_table_row_counts.values() if isinstance(v, int) and v >= 0]
                        manageable = (not row_counts) or all(v <= 250000 for v in row_counts)
                        if needs_exact and sample_only_reads and manageable and not sampled_agg_retry_used:
                            sampled_agg_retry_used = True
                            result_str += (
                                "\n\n[Exactness check] This question likely requires exact aggregation/count, but your current answer "
                                "is computed after sampled read_table calls only. Re-read required tables with mode='full' "
                                "(if row_count is manageable) and recompute once for an exact answer."
                            )
                        else:
                            shape_reasons = _result_shape_mismatch_reasons(
                                ctx.question, last_execute_code or "", res
                            )
                            if shape_reasons and not shape_retry_used:
                                shape_retry_used = True
                                reason_text = "; ".join(shape_reasons[:2])
                                result_str += (
                                    "\n\n[Shape check] Your current output shape mismatches question intent: "
                                    f"{reason_text}. Please fix logic and run execute_python once more."
                                )
                            else:
                                semantic_reasons = _semantic_mismatch_reasons(
                                    ctx.question, last_execute_code or "", res, mode="code"
                                )
                                if semantic_reasons and not semantic_retry_used:
                                    semantic_retry_used = True
                                    reason_text = "; ".join(semantic_reasons[:2])
                                    result_str += (
                                        "\n\n[Semantic check] Your current code output may violate question intent: "
                                        f"{reason_text}. Please fix logic and run execute_python once more."
                                    )
                                else:
                                    predicted_result = res
                    else:
                        result_str += (
                            "\n\n[Validation] The captured result looks like an intermediate/debug artifact "
                            f"({invalid_reason}). Please return the FINAL answer only in `result`, "
                            "with the exact columns requested by the question."
                        )

            # execute_python 没有设置 result 且无 error → 给 LLM 明确提示，避免反复单行调用
            if (tc["name"] == "execute_python" and tool_result.get("ok")
                    and tool_result.get("result") is None
                    and not tool_result.get("error")):
                result_str += (
                    "\n\n[Hint] No 'result' variable was captured. "
                    "Remember: execute_python is NOT stateful between calls—put ALL "
                    "computation in a SINGLE call and end with `result = <your_answer>`. "
                    "Do NOT call execute_python with just `result` alone."
                )

            # execute_python 抛出错误（ok=False）→ 引导 LLM 修复后重试
            if tc["name"] == "execute_python" and not tool_result.get("ok") and tool_result.get("error"):
                execute_error_count += 1
                if not error_retry_used:
                    error_retry_used = True
                    result_str += (
                        "\n\n[Self-check] Your code raised an error (shown above). Common causes: "
                        "type mismatch in merge/join key (e.g. str '2' vs int 2 — cast to the same type first), "
                        "wrong column name, or accessing a non-existent index. "
                        "Fix the code and call execute_python again. (One error-retry allowed.)"
                    )

            trace.append({"turn": turn, "tool": tc["name"], "args": tc["arguments"],
                          "result_ok": tool_result.get("ok"), "duration_s": tool_duration})
            _log({"event": "observe", "turn": turn, "tool": tc["name"],
                  "result_ok": tool_result.get("ok"), "duration_s": tool_duration,
                  "result_preview": result_preview})

            messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": result_str[:(2500 if coopt_fast else 6000)],
            })

        if runtime_profile_enabled:
            bott = _diagnose_runtime_bottleneck(
                turn=turn,
                turn_tool_times=turn_tool_times,
                read_table_stats=read_table_stats,
                execute_error_count=execute_error_count,
            )
            history = list(runtime_profile.get("bottleneck_history") or [])
            history.append(bott)
            max_hist = max(1, int(getattr(ctx, "runtime_profile_max_history", 8) or 8))
            if len(history) > max_hist:
                history = history[-max_hist:]
            runtime_profile["bottleneck_history"] = history
            runtime_profile["last_bottleneck"] = bott
            _log(
                {
                    "event": "bottleneck_update",
                    "turn": turn,
                    "class": bott.get("class"),
                    "dominant_tool": bott.get("dominant_tool"),
                    "dominant_pct": bott.get("dominant_pct"),
                }
            )

        # execute_python 成功且获得 result 后立即结束——无需额外 LLM 调用生成文字 answer
        # result set 就是最终答案，final_answer 设为 result 的字符串表示
        if predicted_result is not None:
            final_answer = str(predicted_result)
            _log({"event": "finish", "turn": turn, "answer": final_answer[:500], "early_exit": True})
            break

    # ── 汇总工具耗时统计 ─────────────────────────────────────────────────────
    tool_time_summary: Dict[str, Any] = {}
    for entry in tool_call_times:
        t = entry["tool"]
        if t not in tool_time_summary:
            tool_time_summary[t] = {"count": 0, "total_s": 0.0}
        tool_time_summary[t]["count"] += 1
        tool_time_summary[t]["total_s"] = round(tool_time_summary[t]["total_s"] + entry["duration_s"], 4)
    total_tool_s = sum(v["total_s"] for v in tool_time_summary.values())
    for v in tool_time_summary.values():
        v["pct"] = round(v["total_s"] / total_tool_s * 100, 1) if total_tool_s > 0 else 0.0

    # ── MV 使用统计 ──────────────────────────────────────────────────────────
    parquet_dir_for_mv = getattr(ctx, "parquet_dir", None)
    mv_tables: List[str] = []
    if parquet_dir_for_mv:
        try:
            manifest_path = os.path.join(os.path.dirname(parquet_dir_for_mv), "manifest.json")
            if os.path.exists(manifest_path):
                import json as _json
                with open(manifest_path) as _mf:
                    _manifest = _json.load(_mf)
                mv_tables = [r["name"] for r in _manifest.get("mv_records", [])
                             if r.get("status") in ("created", "already_exists")]
        except Exception:
            pass
    tables_read = [e["args"].get("table", "") for e in trace if e.get("tool") == "read_table"]
    mv_used = [t for t in tables_read if t in mv_tables]

    if predicted_result is not None:
        predicted_result, _ = _apply_dataset_result_contract(ctx, predicted_result)
        final_answer = str(predicted_result)

    code_ok = predicted_result is not None
    return {
        "ok": code_ok,
        "answer": final_answer,
        "predicted_result": _to_json_serializable(predicted_result),
        "trace": trace,
        "trajectory": trajectory,
        "turns": len(trace),
        "error": None if code_ok else "code_subagent_no_structured_result",
        # ── 新增指标 ──
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "total_tokens": total_prompt_tokens + total_completion_tokens,
        "tool_time_summary": tool_time_summary,   # {tool_name: {count, total_s, pct}}
        "tool_call_times": tool_call_times,        # 逐次列表，供 trajectory 分析
        "mv_tables_available": mv_tables,
        "mv_tables_used": mv_used,
        "mv_used_count": len(mv_used),
        "last_execute_code": last_execute_code,
        "read_table_summary": {
            "calls": int(read_table_stats.get("calls", 0)),
            "tables_read": len(read_table_stats.get("tables", set())),
            "tables": sorted(str(t) for t in read_table_stats.get("tables", set())),
            "sample_read_calls": int(read_table_stats.get("sample_read_calls", 0)),
            "full_read_calls": int(read_table_stats.get("full_read_calls", 0)),
            "sampled_true_calls": int(read_table_stats.get("sampled_true_calls", 0)),
            "full_df_bind_calls": int(read_table_stats.get("full_df_bind_calls", 0)),
            "full_df_bound_tables_count": len(read_table_stats.get("full_df_bound_tables", set())),
        },
        "execute_error_count": int(execute_error_count),
        "runtime_profile": runtime_profile if runtime_profile_enabled else {},
        "bottleneck_summary": dict(runtime_profile.get("last_bottleneck") or {}) if runtime_profile_enabled else {},
    }


def _sql_tool_defs(ctx: Optional[AgentContext] = None) -> List[Dict[str, Any]]:
    """Expose only SQL-focused tools for the SQL sub-agent."""
    allow = {"query_sql", "get_schema", "get_join_path"}
    ds = (ctx.dataset or "").lower() if ctx else ""
    if ds in {"wtq", "tabfact"}:
        allow.discard("get_join_path")
    out: List[Dict[str, Any]] = []
    for t in get_tool_definitions(sql_free=False):
        fn = t.get("function", {}).get("name")
        if fn in allow:
            out.append(t)
    return out


_C3SQL_SQL_GEN_CHAT_PROMPT: List[Dict[str, str]] = [
    {
        "role": "user",
        "content": """Tips 1:
Question: Which A has most number of B?
Gold SQL: select A from B group by A order by count ( * ) desc limit 1;
Notice that the Gold SQL doesn't select COUNT(*) because the question only wants to know the A and the number should be only used in ORDER BY clause, there are many questions asks in this way, and I need you to remember this in the the following questions.""",
    },
    {
        "role": "assistant",
        "content": "Thank you for the tip! I'll keep in mind that when the question only asks for a certain field, I should not include the COUNT(*) in the SELECT statement, but instead use it in the ORDER BY clause to sort the results based on the count of that field.",
    },
    {
        "role": "user",
        "content": """Tips 2:
Don't use "IN", "OR", "LEFT JOIN" as it might cause extra results, use "INTERSECT" or "EXCEPT" instead, and remember to use "DISTINCT" or "LIMIT" when necessary.
For example,
Question: Who are the A who have been nominated for both B award and C award?
Gold SQL should be: select A from X where award = 'B' intersect select A from X where award = 'C';""",
    },
    {
        "role": "assistant",
        "content": 'Thank you for the tip! I\'ll remember to use "INTERSECT" or "EXCEPT" instead of "IN", "OR", or "LEFT JOIN" when I want to find records that match or don\'t match across two tables. Additionally, I\'ll make sure to use "DISTINCT" or "LIMIT" when necessary to avoid repetitive results or limit the number of results returned.',
    },
]


def run_sql_subagent(
    ctx: AgentContext,
    trajectory_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    max_sql_refine: int = 1,
) -> Dict[str, Any]:
    """
    SQL-first sub-agent:
    - Tooling: query_sql + get_schema
    - If SQL execution fails, allow one refine attempt then stop.
    """
    client = _build_openai_client(ctx)
    if not client:
        return {"ok": False, "error": "OpenAI client not available", "trace": [], "trajectory": []}

    trajectory: List[Dict[str, Any]] = []

    def _log(event: Dict[str, Any]) -> None:
        trajectory.append(event)
        if trajectory_callback:
            trajectory_callback(event)

    # SQL path must stay consistent with query_sql backend (sqlite). Do not expose parquet/MV schema here.
    sql_ctx = replace(ctx, parquet_dir=None, use_mschema=True)

    schema_hint = _build_schema_hint(sql_ctx)
    schema_section = schema_hint if schema_hint else "[Schema not preloaded. Call get_schema() if needed.]"
    user_content = (
        "You are now a sqlite data analyst, and you are given a database schema as follows:\n\n"
        f"{schema_section}\n\n"
        f"Database: {ctx.db_id}\n\n"
        "【Question】\n"
        f"{ctx.question}\n\n"
        "Generate one executable SQL with no explanations, then call query_sql with that SQL."
    )
    if (ctx.dataset or "").lower() == "tabfact":
        user_content += (
            "\n\n[Output contract for TabFact]\n"
            "Use table `data` as the main table.\n"
            "Generate SQL that returns exactly one row/one column with integer label: "
            "1 for ENTAILED, 0 for REFUTED."
        )
    elif (ctx.dataset or "").lower() == "wtq":
        user_content += (
            "\n\n[Output contract for WikiTableQuestions]\n"
            "Use table `data` as the main table.\n"
            "Generate SQL that returns only the answer value(s), no explanation text."
        )
    if getattr(ctx, "evidence", None) and str(ctx.evidence).strip():
        user_content += f"\n\nExternal knowledge:\n{ctx.evidence.strip()}"
    skill_hint = _build_retrieved_skill_block(ctx, mode="sql")
    if skill_hint.get("text"):
        user_content += (
            "\n\n[Retrieved skills from successful trajectories - advisory only]\n"
            + str(skill_hint["text"])
        )
        _log(
            {
                "event": "skill_retrieved",
                "mode": "sql",
                "count": int(skill_hint.get("count", 0) or 0),
                "skill_ids": list(skill_hint.get("skill_ids") or [])[:8],
                "subagent": "sql",
            }
        )

    system_prompt = """You are now an excellent SQL writer and a SQL generation sub-agent.
Use query_sql to execute SQL and return the FINAL query result.

Rules:
1. Prefer one correct SQL statement that directly answers the question.
2. You may call get_schema/get_join_path for join-key/path detail; do not use python tools.
3. If query_sql returns an error or obvious shape mismatch, fix SQL and retry within budget.
4. Once query_sql succeeds, stop and do not call more tools.
5. Call query_sql with SQL only (no markdown, no explanations).
"""
    if (ctx.dataset or "").lower() == "tabfact":
        system_prompt += "\n6. For TabFact, SQL must return a single integer label: 1 (ENTAILED) or 0 (REFUTED)."
    elif (ctx.dataset or "").lower() == "wtq":
        system_prompt += "\n6. For WikiTableQuestions, return answer values only."
    messages = [{"role": "system", "content": system_prompt}]
    messages.append({"role": "user", "content": user_content})

    import time as _time
    trace: List[Dict[str, Any]] = []
    predicted_result: Any = None
    final_answer: Optional[str] = None
    total_prompt_tokens = 0
    total_completion_tokens = 0
    tool_call_times: List[Dict[str, Any]] = []
    sql_errors = 0
    last_sql: Optional[str] = None
    last_processed_sql: Optional[str] = None
    raw_sql_attempts: List[str] = []
    processed_sql_attempts: List[str] = []
    sql_postprocess_applied = False
    shape_retry_used = False
    semantic_retry_used = False
    hard_stop = False
    last_successful_sql_result: Any = None
    last_successful_sql: Optional[str] = None
    last_successful_processed_sql: Optional[str] = None
    sql_fallback_used = False
    sql_candidates: List[Dict[str, Any]] = []
    best_sql_candidate: Optional[Dict[str, Any]] = None
    best_sql_candidate_score = -1.0
    sql_reverse_calls = 0

    _log({"event": "start", "question": ctx.question, "db_id": ctx.db_id, "model": ctx.model, "subagent": "sql"})
    coopt_fast = _coopt_fast_enabled(ctx)
    max_turns_eff = min(ctx.max_turns, 4 if coopt_fast else 6)
    max_sql_refine_eff = min(max_sql_refine, 1) if coopt_fast else max_sql_refine
    if not coopt_fast:
        messages.extend(_C3SQL_SQL_GEN_CHAT_PROMPT)
    for turn in range(max_turns_eff):
        _t_llm_start = _time.perf_counter()
        send_messages = _compress_old_tool_messages(messages, keep_last_tool=1) if coopt_fast else messages
        try:
            resp = client.chat.completions.create(
                model=ctx.model,
                messages=send_messages,
                tools=_sql_tool_defs(ctx),
                tool_choice="auto",
                timeout=120.0,
            )
        except Exception as e:
            _log({"event": "error", "turn": turn, "error": str(e), "subagent": "sql"})
            return {"ok": False, "error": str(e), "trace": trace, "trajectory": trajectory}
        llm_duration_s = round(_time.perf_counter() - _t_llm_start, 3)

        usage = getattr(resp, "usage", None)
        if usage:
            total_prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
            total_completion_tokens += getattr(usage, "completion_tokens", 0) or 0

        msg = resp.choices[0].message
        assistant_content = msg.content or ""
        messages.append({"role": "assistant", "content": assistant_content, "tool_calls": getattr(msg, "tool_calls", None)})
        _log({
            "event": "think",
            "turn": turn,
            "content": assistant_content[:500] if assistant_content else None,
            "prompt_tokens": getattr(usage, "prompt_tokens", None) if usage else None,
            "completion_tokens": getattr(usage, "completion_tokens", None) if usage else None,
            "llm_duration_s": llm_duration_s,
            "subagent": "sql",
        })

        all_tcs = _extract_all_tool_calls(msg)
        if not all_tcs:
            # Keep this sub-agent deterministic: require a tool call.
            messages.append({
                "role": "user",
                "content": "Please call query_sql with one executable SQL statement.",
            })
            continue

        for tc in all_tcs:
            _log({"event": "act", "turn": turn, "tool": tc["name"], "args": tc["arguments"], "subagent": "sql"})
            _t_tool_start = _time.perf_counter()
            exec_args = dict(tc["arguments"])
            raw_sql = None
            processed_sql = None
            sql_post_meta: Dict[str, Any] = {}
            query_shape_reasons: List[str] = []
            query_semantic_reasons: List[str] = []
            query_reverse_meta: Optional[Dict[str, Any]] = None
            query_candidate_score: Optional[float] = None
            query_drift_detected = False
            if tc["name"] == "query_sql":
                raw_sql = str(
                    exec_args.get("sql")
                    or exec_args.get("query")
                    or exec_args.get("statement")
                    or ""
                )
                processed_sql, sql_post_meta = _post_process_sql_text(raw_sql, db_id=ctx.db_id)
                raw_sql_attempts.append(raw_sql)
                processed_sql_attempts.append(processed_sql)
                if sql_post_meta.get("changed"):
                    sql_postprocess_applied = True
                exec_args["sql"] = processed_sql
                if not processed_sql:
                    tool_result = {"ok": False, "error": "Empty SQL after post-processing", "result": None}
                else:
                    tool_result = run_tool(tc["name"], exec_args, sql_ctx)
            else:
                tool_result = run_tool(tc["name"], exec_args, sql_ctx)
            tool_duration = round(_time.perf_counter() - _t_tool_start, 4)
            tool_call_times.append({"tool": tc["name"], "duration_s": tool_duration, "ok": tool_result.get("ok", False)})

            if tc["name"] == "query_sql":
                last_sql = raw_sql if raw_sql is not None else str(tc["arguments"].get("sql", "") or "")
                last_processed_sql = processed_sql if processed_sql is not None else last_sql
                if tool_result.get("ok"):
                    last_successful_sql_result = tool_result.get("result")
                    last_successful_sql = last_sql
                    last_successful_processed_sql = last_processed_sql
                    query_shape_reasons = _result_shape_mismatch_reasons(
                        ctx.question, last_processed_sql or "", tool_result.get("result")
                    )
                    query_semantic_reasons = _semantic_mismatch_reasons(
                        ctx.question, last_processed_sql or "", tool_result.get("result"), mode="sql"
                    )
                    if coopt_fast:
                        query_reverse_meta = {
                            "alignment_score": 0.5,
                            "critical_mismatch": False,
                            "mismatch_tags": [],
                            "reason": "skipped_in_coopt_fast_mode",
                            "prompt_tokens": 0,
                            "completion_tokens": 0,
                        }
                    else:
                        query_reverse_meta = _reverse_deduce_intent(
                            ctx=ctx,
                            artifact_type="sql",
                            artifact_text=last_processed_sql or "",
                            result=tool_result.get("result"),
                            client=client,
                        )
                        sql_reverse_calls += 1
                        total_prompt_tokens += int(query_reverse_meta.get("prompt_tokens", 0) or 0)
                        total_completion_tokens += int(query_reverse_meta.get("completion_tokens", 0) or 0)
                    query_candidate_score = _score_sql_candidate(
                        reverse_meta=query_reverse_meta,
                        shape_reasons=query_shape_reasons,
                        semantic_reasons=query_semantic_reasons,
                        sql_errors_so_far=sql_errors,
                    )
                    candidate_item = {
                        "turn": turn,
                        "raw_sql": last_sql,
                        "processed_sql": last_processed_sql,
                        "result": tool_result.get("result"),
                        "shape_reasons": query_shape_reasons,
                        "semantic_reasons": query_semantic_reasons,
                        "reverse_meta": query_reverse_meta,
                        "score": query_candidate_score,
                    }
                    sql_candidates.append(candidate_item)
                    prev_best = best_sql_candidate_score
                    if query_candidate_score > best_sql_candidate_score:
                        best_sql_candidate_score = query_candidate_score
                        best_sql_candidate = candidate_item
                    elif prev_best - query_candidate_score >= 0.12:
                        query_drift_detected = True
                    if query_shape_reasons:
                        sql_errors += 1
                        if not shape_retry_used and sql_errors <= max_sql_refine_eff:
                            shape_retry_used = True
                        else:
                            hard_stop = sql_errors > max_sql_refine_eff
                    elif query_semantic_reasons:
                        sql_errors += 1
                        if not semantic_retry_used and sql_errors <= max_sql_refine_eff:
                            semantic_retry_used = True
                        else:
                            hard_stop = sql_errors > max_sql_refine_eff
                    elif (
                        query_candidate_score >= 0.90
                        and not query_reverse_meta.get("critical_mismatch", False)
                    ):
                        predicted_result = tool_result.get("result")
                        final_answer = str(predicted_result)
                else:
                    sql_errors += 1
                    if sql_errors > max_sql_refine_eff:
                        hard_stop = True

            result_str = _format_tool_result(tool_result)
            if tc["name"] == "query_sql" and raw_sql is not None and processed_sql is not None:
                if raw_sql.strip() != processed_sql:
                    result_str += (
                        "\n\n[SQL Post-process] Normalized SQL before execution:\n"
                        f"{processed_sql}"
                    )
                else:
                    result_str += "\n\n[SQL Post-process] SQL normalized and executed."
            if tc["name"] == "query_sql" and tool_result.get("ok"):
                if query_shape_reasons:
                    result_str += (
                        "\n\n[SQL Shape check] Possible output-shape mismatch: "
                        + "; ".join(query_shape_reasons[:2])
                        + ". Please revise SQL and retry."
                    )
                elif query_semantic_reasons:
                    result_str += (
                        "\n\n[SQL Semantic check] Possible intent mismatch: "
                        + "; ".join(query_semantic_reasons[:2])
                        + ". Please revise SQL and retry."
                    )
                if query_reverse_meta:
                    result_str += (
                        "\n\n[Reverse intent check] alignment_score="
                        f"{query_reverse_meta.get('alignment_score', 0.5):.2f}, "
                        f"critical_mismatch={bool(query_reverse_meta.get('critical_mismatch', False))}, "
                        f"tags={','.join(query_reverse_meta.get('mismatch_tags', [])[:3]) or 'none'}."
                    )
                    reason_txt = str(query_reverse_meta.get("reason", "") or "").strip()
                    if reason_txt:
                        result_str += f"\n[Reverse reason] {reason_txt[:200]}"
                if query_drift_detected and best_sql_candidate is not None:
                    result_str += (
                        "\n\n[Drift guard] This SQL candidate is weaker than an earlier successful one. "
                        "Prefer revising toward prior alignment instead of replacing it."
                    )
            if tc["name"] == "query_sql" and not tool_result.get("ok"):
                if sql_errors <= max_sql_refine_eff:
                    result_str += (
                        "\n\n[SQL Self-check] SQL execution failed. Fix table/column names, join keys, "
                        "or syntax and retry query_sql once."
                    )
                else:
                    result_str += "\n\n[SQL Self-check] Retry budget exhausted. Stop SQL path."
            trace_item = {
                "turn": turn,
                "tool": tc["name"],
                "args": tc["arguments"],
                "result_ok": tool_result.get("ok"),
                "duration_s": tool_duration,
            }
            if tc["name"] == "query_sql":
                trace_item["raw_sql"] = raw_sql
                trace_item["processed_sql"] = processed_sql
                trace_item["sql_postprocess"] = sql_post_meta
                trace_item["shape_reasons"] = query_shape_reasons
                trace_item["semantic_reasons"] = query_semantic_reasons
                trace_item["reverse_meta"] = query_reverse_meta
                trace_item["candidate_score"] = query_candidate_score
            trace.append(trace_item)
            _log({
                "event": "observe",
                "turn": turn,
                "tool": tc["name"],
                "result_ok": tool_result.get("ok"),
                "duration_s": tool_duration,
                "result_preview": (result_str[:300] + "...") if len(result_str) > 300 else result_str,
                "subagent": "sql",
            })
            messages.append({"role": "tool", "tool_call_id": tc["id"], "content": result_str[:(2500 if coopt_fast else 6000)]})

        if predicted_result is not None:
            final_answer = str(predicted_result)
            _log({"event": "finish", "turn": turn, "answer": final_answer[:500], "early_exit": True, "subagent": "sql"})
            break
        if hard_stop:
            _log({"event": "finish", "turn": turn, "answer": "SQL retry exhausted", "subagent": "sql"})
            break

    if predicted_result is None and best_sql_candidate is not None:
        predicted_result = best_sql_candidate.get("result")
        final_answer = str(predicted_result)
        sql_fallback_used = True
        _log(
            {
                "event": "finish",
                "answer": final_answer[:500] if final_answer else None,
                "subagent": "sql",
                "selected_from_candidate_bank": True,
                "selected_score": round(float(best_sql_candidate.get("score", 0.0) or 0.0), 4),
            }
        )
    elif predicted_result is None and last_successful_sql_result is not None:
        # Keep the last executable SQL result instead of dropping SQL path entirely.
        predicted_result = last_successful_sql_result
        final_answer = str(predicted_result)
        sql_fallback_used = True
        _log(
            {
                "event": "finish",
                "answer": final_answer[:500] if final_answer else None,
                "subagent": "sql",
                "fallback_to_last_successful_sql": True,
            }
        )

    if predicted_result is not None:
        predicted_result, _ = _apply_dataset_result_contract(ctx, predicted_result)
        final_answer = str(predicted_result)

    tool_time_summary: Dict[str, Any] = {}
    for entry in tool_call_times:
        t = entry["tool"]
        if t not in tool_time_summary:
            tool_time_summary[t] = {"count": 0, "total_s": 0.0}
        tool_time_summary[t]["count"] += 1
        tool_time_summary[t]["total_s"] = round(tool_time_summary[t]["total_s"] + entry["duration_s"], 4)
    total_tool_s = sum(v["total_s"] for v in tool_time_summary.values())
    for v in tool_time_summary.values():
        v["pct"] = round(v["total_s"] / total_tool_s * 100, 1) if total_tool_s > 0 else 0.0

    return {
        "ok": predicted_result is not None,
        "answer": final_answer,
        "predicted_result": _to_json_serializable(predicted_result),
        "trace": trace,
        "trajectory": trajectory,
        "turns": len(trace),
        "error": None if predicted_result is not None else "sql_subagent_failed",
        "sql_errors": sql_errors,
        "last_sql": last_sql,
        "last_processed_sql": last_processed_sql,
        "raw_sql_attempts": raw_sql_attempts,
        "processed_sql_attempts": processed_sql_attempts,
        "sql_postprocess_applied": sql_postprocess_applied,
        "sql_fallback_used": sql_fallback_used,
        "last_successful_sql": last_successful_sql,
        "last_successful_processed_sql": last_successful_processed_sql,
        "sql_reverse_calls": int(sql_reverse_calls),
        "sql_candidate_count": len(sql_candidates),
        "best_sql_candidate_score": (
            round(float(best_sql_candidate_score), 4) if best_sql_candidate_score >= 0 else None
        ),
        "best_sql_candidate": (
            {
                "turn": best_sql_candidate.get("turn"),
                "processed_sql": best_sql_candidate.get("processed_sql"),
                "score": round(float(best_sql_candidate.get("score", 0.0) or 0.0), 4),
                "shape_reasons": best_sql_candidate.get("shape_reasons", []),
                "semantic_reasons": best_sql_candidate.get("semantic_reasons", []),
                "reverse_meta": best_sql_candidate.get("reverse_meta", {}),
            }
            if best_sql_candidate
            else None
        ),
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "total_tokens": total_prompt_tokens + total_completion_tokens,
        "tool_time_summary": tool_time_summary,
        "tool_call_times": tool_call_times,
        "mv_tables_available": [],
        "mv_tables_used": [],
        "mv_used_count": 0,
    }


def _candidate_risk_flags(
    ctx: AgentContext,
    source: str,
    out_meta: Dict[str, Any],
    shape_issues: List[str],
    semantic_issues: List[str],
) -> List[str]:
    flags: List[str] = []
    q = ctx.question or ""
    if shape_issues:
        flags.append("shape_mismatch_detected")
    if semantic_issues:
        flags.append("semantic_mismatch_detected")
    if source == "code":
        read_stats = out_meta.get("read_table_summary") or {}
        sampled_calls = int(read_stats.get("sample_read_calls", 0) or 0)
        full_calls = int(read_stats.get("full_read_calls", 0) or 0)
        sampled_true_calls = int(read_stats.get("sampled_true_calls", 0) or 0)
        bound_tables = int(read_stats.get("full_df_bound_tables_count", 0) or 0)
        tables_read = int(read_stats.get("tables_read", 0) or 0)
        full_df_bound = bound_tables > 0 and (tables_read == 0 or bound_tables >= tables_read)
        if (
            (_question_has_count_intent(q) or _question_has_aggregate_intent(q))
            and sampled_calls > 0
            and full_calls == 0
            and not full_df_bound
        ):
            flags.append("aggregate_intent_but_code_uses_sample_reads_only")
        if sampled_true_calls > 0 and sampled_calls > 0 and sampled_true_calls == sampled_calls and not full_df_bound:
            flags.append("all_code_table_reads_are_sampled")
        if int(out_meta.get("execute_error_count", 0) or 0) > 0:
            flags.append("code_had_runtime_errors")
    elif source == "sql":
        if int(out_meta.get("sql_errors", 0) or 0) > 0:
            flags.append("sql_had_retry_or_execution_errors")
    return flags


def _severe_risk_flags(source: str, flags: Optional[List[str]]) -> List[str]:
    if not flags:
        return []
    if source == "code":
        # Keep severe list narrow: these two risks are consistently high-impact for code correctness.
        severe = {
            "aggregate_intent_but_code_uses_sample_reads_only",
            "code_had_runtime_errors",
        }
    else:
        # SQL retries are common and often recoverable; do not treat them as severe by default.
        severe = {
            "semantic_mismatch_detected",
        }
    return [f for f in flags if f in severe]


def _judge_hybrid_choice(
    ctx: AgentContext,
    diff_summary: Dict[str, Any],
    sql_meta: Dict[str, Any],
    code_meta: Dict[str, Any],
    sql_risk_flags: Optional[List[str]] = None,
    code_risk_flags: Optional[List[str]] = None,
    default_choice_override: Optional[str] = None,
) -> Dict[str, Any]:
    """Ask one LLM judge call to pick between SQL and Code candidates using compact diff only."""
    client = _build_openai_client(ctx)
    default_choice = "sql" if (ctx.dataset or "").lower() == "spider" else "code"
    if default_choice_override in {"sql", "code"}:
        default_choice = default_choice_override
    if not client:
        return {"choice": default_choice, "reason": "OpenAI client not available", "prompt_tokens": 0, "completion_tokens": 0}

    system_prompt = """You are a result fusion judge for NL2DB.
Choose one final candidate: 'sql' or 'code'.
Use ONLY the compact comparison summary and question intent.
Prefer the candidate that better matches requested projection, grouping, and filtering.
Prefer the candidate with fewer risk flags when one side is clearly riskier.
Do NOT apply any fixed preference toward SQL or Code.
Output strict JSON: {"choice":"sql|code","reason":"..."}.
If uncertain, choose "%s".
"""
    system_prompt = system_prompt % default_choice
    user_prompt = (
        f"Question:\n{ctx.question}\n\n"
        f"SQL candidate meta:\n{json.dumps(sql_meta, ensure_ascii=False)}\n\n"
        f"Code candidate meta:\n{json.dumps(code_meta, ensure_ascii=False)}\n\n"
        f"SQL risk flags:\n{json.dumps(sql_risk_flags or [], ensure_ascii=False)}\n\n"
        f"Code risk flags:\n{json.dumps(code_risk_flags or [], ensure_ascii=False)}\n\n"
        f"Diff summary:\n{json.dumps(diff_summary, ensure_ascii=False)}"
    )
    try:
        resp = client.chat.completions.create(
            model=ctx.model,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            timeout=60.0,
        )
        content = (resp.choices[0].message.content or "").strip()
        usage = getattr(resp, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
        completion_tokens = getattr(usage, "completion_tokens", 0) if usage else 0
        choice = default_choice
        reason = content[:300]
        m = re.search(r"\{.*\}", content, re.DOTALL)
        if m:
            try:
                obj = json.loads(m.group(0))
                c = str(obj.get("choice", "code")).strip().lower()
                if c in {"sql", "code"}:
                    choice = c
                reason = str(obj.get("reason", reason))
            except Exception:
                pass
        return {
            "choice": choice,
            "reason": reason,
            "raw": content,
            "prompt_tokens": prompt_tokens or 0,
            "completion_tokens": completion_tokens or 0,
        }
    except Exception as e:
        return {"choice": default_choice, "reason": f"judge_failed: {e}", "prompt_tokens": 0, "completion_tokens": 0}


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return str(v).strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name)
    if v is None:
        return default
    try:
        return float(v)
    except Exception:
        return default


def _candidate_issue_score(
    source: str,
    result: Any,
    shape_issues: List[str],
    semantic_issues: List[str],
    severe_risk_flags: List[str],
    out_meta: Dict[str, Any],
) -> float:
    score = 0.0
    score += 3.0 * float(len(severe_risk_flags or []))
    score += 1.0 * float(len(shape_issues or []))
    score += 0.8 * float(len(semantic_issues or []))
    if _is_effectively_empty_result(result):
        score += 2.0
    if source == "code":
        if int(out_meta.get("execute_error_count", 0) or 0) > 0:
            score += 1.5
    else:
        if int(out_meta.get("sql_errors", 0) or 0) > 0:
            score += 0.6
    return score


def _should_run_tqa_risk_verify(
    ctx: AgentContext,
    diff_summary: Optional[Dict[str, Any]],
    code_result: Any,
    sql_result: Any,
    code_shape_issues: List[str],
    sql_shape_issues: List[str],
    code_semantic_issues: List[str],
    sql_semantic_issues: List[str],
    code_severe_risk_flags: List[str],
    sql_severe_risk_flags: List[str],
    out_code: Dict[str, Any],
    out_sql: Dict[str, Any],
) -> tuple[bool, List[str]]:
    if str(getattr(ctx, "dataset", "") or "").lower() not in {"wtq", "tabfact"}:
        return False, []
    if not _env_bool("NL2CODE_TQA_RISK_VERIFY", True):
        return False, []
    if not diff_summary or bool(diff_summary.get("equal", False)):
        return False, []

    code_empty = _is_effectively_empty_result(code_result)
    sql_empty = _is_effectively_empty_result(sql_result)
    has_empty_asym = code_empty != sql_empty
    has_severe = bool(code_severe_risk_flags or sql_severe_risk_flags)
    has_shape_asym = bool(code_shape_issues) != bool(sql_shape_issues)
    has_sem_asym = bool(code_semantic_issues) != bool(sql_semantic_issues)
    has_code_err = int(out_code.get("execute_error_count", 0) or 0) > 0
    has_sql_err = int(out_sql.get("sql_errors", 0) or 0) > 0
    has_exec_err = has_code_err or has_sql_err
    has_agg_numeric_asym = False
    if (_question_has_count_intent(ctx.question or "") or _question_has_aggregate_intent(ctx.question or "")):
        code_scalar = _extract_single_scalar(code_result)
        sql_scalar = _extract_single_scalar(sql_result)
        if _is_numeric_like(code_scalar) != _is_numeric_like(sql_scalar):
            has_agg_numeric_asym = True

    # Keep this verifier lightweight:
    # trigger only for clearly risky conflicts instead of all disagreements.
    trigger = (
        has_severe
        or has_shape_asym
        or has_sem_asym
        or (has_empty_asym and (has_exec_err or has_agg_numeric_asym))
        or (has_exec_err and has_agg_numeric_asym)
    )

    reasons: List[str] = []
    if has_empty_asym:
        reasons.append("empty_asymmetry")
    if has_severe:
        reasons.append("severe_risk_flags")
    if has_shape_asym:
        reasons.append("shape_asymmetry")
    if has_sem_asym:
        reasons.append("semantic_asymmetry")
    if has_code_err:
        reasons.append("code_runtime_errors")
    if has_sql_err:
        reasons.append("sql_execution_errors")
    if has_agg_numeric_asym:
        reasons.append("aggregate_numeric_asymmetry")

    return trigger, reasons


def _run_tqa_risk_verify(
    ctx: AgentContext,
    default_choice: str,
    trigger_reasons: List[str],
    diff_summary: Dict[str, Any],
    code_result: Any,
    sql_result: Any,
    code_shape_issues: List[str],
    sql_shape_issues: List[str],
    code_semantic_issues: List[str],
    sql_semantic_issues: List[str],
    code_risk_flags: List[str],
    sql_risk_flags: List[str],
) -> Dict[str, Any]:
    meta: Dict[str, Any] = {
        "enabled": True,
        "triggered": True,
        "trigger_reasons": trigger_reasons[:8],
        "choice": "abstain",
        "confidence": 0.0,
        "reason": "not_run",
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "applied": False,
    }
    cli = _build_openai_client(ctx)
    if not cli:
        meta["reason"] = "no_openai_client"
        return meta

    system_prompt = (
        "You are a cautious verifier for table QA result fusion.\n"
        "Choose one of: code | sql | abstain.\n"
        "Only choose code/sql when evidence is clearly stronger; otherwise abstain.\n"
        "Return strict JSON: "
        "{\"choice\":\"code|sql|abstain\",\"confidence\":0.0,\"reason\":\"...\"}."
    )
    user_prompt = (
        f"Dataset: {ctx.dataset}\n"
        f"Question/Statement:\n{ctx.question}\n\n"
        f"Current default choice: {default_choice}\n"
        f"Trigger reasons: {json.dumps(trigger_reasons, ensure_ascii=False)}\n\n"
        f"Code result summary: rows={_result_row_count(code_result)}, arity={_result_row_arity(code_result)}, "
        f"preview={json.dumps(_compact_result_preview(code_result, 5), ensure_ascii=False)}\n"
        f"Code shape issues: {json.dumps(code_shape_issues[:4], ensure_ascii=False)}\n"
        f"Code semantic issues: {json.dumps(code_semantic_issues[:4], ensure_ascii=False)}\n"
        f"Code risk flags: {json.dumps(code_risk_flags[:6], ensure_ascii=False)}\n\n"
        f"SQL result summary: rows={_result_row_count(sql_result)}, arity={_result_row_arity(sql_result)}, "
        f"preview={json.dumps(_compact_result_preview(sql_result, 5), ensure_ascii=False)}\n"
        f"SQL shape issues: {json.dumps(sql_shape_issues[:4], ensure_ascii=False)}\n"
        f"SQL semantic issues: {json.dumps(sql_semantic_issues[:4], ensure_ascii=False)}\n"
        f"SQL risk flags: {json.dumps(sql_risk_flags[:6], ensure_ascii=False)}\n\n"
        f"Diff summary:\n{json.dumps(diff_summary, ensure_ascii=False)}"
    )
    try:
        resp = cli.chat.completions.create(
            model=ctx.model,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            timeout=45.0,
        )
        usage = getattr(resp, "usage", None)
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0) if usage else 0
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0) if usage else 0
        txt = (resp.choices[0].message.content or "").strip()
        m = re.search(r"\{.*\}", txt, re.DOTALL)
        obj: Dict[str, Any] = {}
        if m:
            try:
                obj = json.loads(m.group(0))
            except Exception:
                obj = {}
        choice = str(obj.get("choice", "abstain")).strip().lower()
        if choice not in {"code", "sql", "abstain"}:
            choice = "abstain"
        try:
            conf = float(obj.get("confidence", 0.0))
        except Exception:
            conf = 0.0
        conf = max(0.0, min(1.0, conf))
        meta.update(
            {
                "choice": choice,
                "confidence": conf,
                "reason": str(obj.get("reason", txt[:220] or "verifier_no_reason"))[:280],
                "raw": txt[:600],
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            }
        )
        return meta
    except Exception as e:
        meta["reason"] = f"verifier_error: {e}"
        return meta


def _merge_prefixed_tool_summary(prefix: str, summary: Dict[str, Any], out: Dict[str, Any]) -> None:
    for k, v in (summary or {}).items():
        out[f"{prefix}.{k}"] = v


def run_hybrid_agent(
    ctx: AgentContext,
    trajectory_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """
    Hybrid mode without a standalone router:
    - run SQL sub-agent and Code sub-agent in parallel
    - if both valid and conflict, run one compact judge call
    - if SQL path fails after one refine, fall back to Code result
    """
    trajectory: List[Dict[str, Any]] = []
    traj_lock = threading.Lock()

    def _log(event: Dict[str, Any]) -> None:
        with traj_lock:
            trajectory.append(event)
        if trajectory_callback:
            trajectory_callback(event)

    _log({"event": "start", "mode": "hybrid", "question": ctx.question, "db_id": ctx.db_id, "model": ctx.model})

    def _code_cb(evt: Dict[str, Any]) -> None:
        _log({"source": "code", **evt})

    def _sql_cb(evt: Dict[str, Any]) -> None:
        _log({"source": "sql", **evt})

    coopt_fast = _coopt_fast_enabled(ctx)
    code_only_fast = coopt_fast and _env_bool("NL2CODE_COOPT_FAST_CODE_ONLY", True)
    if code_only_fast:
        fast_turns = int(_env_float("NL2CODE_COOPT_FAST_CODE_MAX_TURNS", 8.0))
        fast_turns = max(1, fast_turns)
        code_ctx = replace(ctx, max_turns=min(ctx.max_turns, fast_turns))
        out_code = run_react_agent(code_ctx, _code_cb)
        out_sql = {
            "ok": False,
            "predicted_result": None,
            "trace": [],
            "trajectory": [],
            "tool_time_summary": {},
            "tool_call_times": [],
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "sql_errors": 0,
            "error": "skipped_in_coopt_fast_code_only_mode",
        }
        _log({"event": "hybrid_fastpath", "mode": "code_only", "coopt_fast": True})
    else:
        with ThreadPoolExecutor(max_workers=2) as ex:
            fut_code = ex.submit(run_react_agent, ctx, _code_cb)
            fut_sql = ex.submit(run_sql_subagent, ctx, _sql_cb, 3)
            out_code = fut_code.result()
            out_sql = fut_sql.result()
    dataset = str(getattr(ctx, "dataset", "") or "").lower()

    raw_code_result = out_code.get("predicted_result")
    raw_sql_result = out_sql.get("predicted_result")
    code_result, code_shape_repairs = _apply_result_shape_repair(ctx.question, raw_code_result)
    sql_result, sql_shape_repairs = _apply_result_shape_repair(ctx.question, raw_sql_result)
    if code_shape_repairs:
        _log({"event": "postprocess", "source": "code", "notes": code_shape_repairs[:3]})
    if sql_shape_repairs:
        _log({"event": "postprocess", "source": "sql", "notes": sql_shape_repairs[:3]})
    code_valid, code_invalid_reason = _validate_candidate_result(code_result)
    sql_valid, sql_invalid_reason = _validate_candidate_result(sql_result)
    code_usable = out_code.get("ok", False) and code_result is not None and code_valid
    sql_usable = out_sql.get("ok", False) and sql_result is not None and sql_valid

    diff_summary: Optional[Dict[str, Any]] = None
    judge_meta: Dict[str, Any] = {}
    tqa_verify_meta: Dict[str, Any] = {}
    chosen_source = ""
    final_result: Any = None
    final_reason = ""

    code_semantic_issues: List[str] = []
    sql_semantic_issues: List[str] = []
    code_shape_issues: List[str] = []
    sql_shape_issues: List[str] = []
    code_risk_flags: List[str] = []
    sql_risk_flags: List[str] = []
    code_severe_risk_flags: List[str] = []
    sql_severe_risk_flags: List[str] = []

    if code_usable and sql_usable:
        diff_summary = _summarize_candidate_diff(sql_result, code_result, cap=2000)
        sql_semantic_issues = _semantic_mismatch_reasons(
            ctx.question, str(out_sql.get("last_processed_sql") or out_sql.get("last_sql") or ""), sql_result, mode="sql"
        )
        code_semantic_issues = _semantic_mismatch_reasons(
            ctx.question, str(out_code.get("last_execute_code") or ""), code_result, mode="code"
        )
        sql_shape_issues = _result_shape_mismatch_reasons(
            ctx.question, str(out_sql.get("last_processed_sql") or out_sql.get("last_sql") or ""), sql_result
        )
        code_shape_issues = _result_shape_mismatch_reasons(
            ctx.question, str(out_code.get("last_execute_code") or ""), code_result
        )
        code_risk_flags = _candidate_risk_flags(ctx, "code", out_code, code_shape_issues, code_semantic_issues)
        sql_risk_flags = _candidate_risk_flags(ctx, "sql", out_sql, sql_shape_issues, sql_semantic_issues)
        code_severe_risk_flags = _severe_risk_flags("code", code_risk_flags)
        sql_severe_risk_flags = _severe_risk_flags("sql", sql_risk_flags)
        if diff_summary.get("equal", False):
            chosen_source = "code"
            final_result = code_result
            final_reason = "sql/code results agree; prefer code in analytics-first mode"
        elif code_severe_risk_flags and not sql_severe_risk_flags:
            chosen_source = "sql"
            final_result = sql_result
            final_reason = f"code candidate has higher severe risk: {', '.join(code_severe_risk_flags[:2])}"
        elif dataset in {"wtq", "tabfact"}:
            # Base choice for single-table TQA (analytics-first but risk-aware).
            agg_intent = _question_has_count_intent(ctx.question or "") or _question_has_aggregate_intent(ctx.question or "")
            code_scalar = _extract_single_scalar(code_result)
            sql_scalar = _extract_single_scalar(sql_result)
            code_errorlike = _looks_like_error_text(code_scalar)
            sql_numeric = _is_numeric_like(sql_scalar)
            code_numeric = _is_numeric_like(code_scalar)
            if _is_effectively_empty_result(code_result) and not _is_effectively_empty_result(sql_result):
                chosen_source = "sql"
                final_result = sql_result
                final_reason = "tqa_policy: code result empty, fallback to sql non-empty result"
            elif code_errorlike and not _looks_like_error_text(sql_scalar):
                chosen_source = "sql"
                final_result = sql_result
                final_reason = "tqa_policy: code scalar looks like runtime error text"
            elif agg_intent and sql_numeric and not code_numeric:
                chosen_source = "sql"
                final_result = sql_result
                final_reason = "tqa_policy: aggregate/count intent prefers numeric sql scalar over non-numeric code output"
            elif (code_shape_issues and not sql_shape_issues) or (code_semantic_issues and not sql_semantic_issues):
                chosen_source = "sql"
                final_result = sql_result
                final_reason = "tqa_policy: code has shape/semantic risk while sql is cleaner"
            else:
                chosen_source = "code"
                final_result = code_result
                final_reason = "tqa_policy: prefer code candidate unless code has clear risk"

            # For TQA, a single judge call on conflicts is usually a stronger signal than a hard-coded
            # code-first prior. Keep this lightweight and optionally disable via env if needed.
            use_tqa_conflict_judge = _env_bool("NL2CODE_TQA_CONFLICT_JUDGE", True)
            if use_tqa_conflict_judge:
                sql_meta = {
                    "rows": _result_row_count(sql_result),
                    "signature": _result_signature(sql_result),
                    "preview": _compact_result_preview(sql_result, 5),
                    "sql_errors": out_sql.get("sql_errors", 0),
                    "last_sql": out_sql.get("last_sql"),
                    "last_processed_sql": out_sql.get("last_processed_sql"),
                }
                code_meta = {
                    "rows": _result_row_count(code_result),
                    "signature": _result_signature(code_result),
                    "preview": _compact_result_preview(code_result, 5),
                    "turns": out_code.get("turns", 0),
                    "read_table_summary": out_code.get("read_table_summary", {}),
                    "execute_error_count": out_code.get("execute_error_count", 0),
                }
                judge_meta = _judge_hybrid_choice(
                    ctx,
                    diff_summary,
                    sql_meta,
                    code_meta,
                    sql_risk_flags=sql_risk_flags,
                    code_risk_flags=code_risk_flags,
                    default_choice_override=chosen_source,
                )
                judge_choice = str(judge_meta.get("choice", chosen_source)).strip().lower()
                if judge_choice in {"sql", "code"}:
                    chosen_source = judge_choice
                    final_result = sql_result if judge_choice == "sql" else code_result
                    final_reason = f"tqa_conflict_judge: {str(judge_meta.get('reason', ''))[:220]}".strip()
                # Keep severe-risk guardrails even when judge is enabled.
                if chosen_source == "code" and code_severe_risk_flags and not sql_severe_risk_flags:
                    chosen_source = "sql"
                    final_result = sql_result
                    final_reason = (
                        "tqa_conflict_judge override->sql due to severe code risk: "
                        f"{', '.join(code_severe_risk_flags[:2])}"
                    )
                elif chosen_source == "sql" and sql_severe_risk_flags and not code_severe_risk_flags:
                    chosen_source = "code"
                    final_result = code_result
                    final_reason = (
                        "tqa_conflict_judge override->code due to severe sql risk: "
                        f"{', '.join(sql_severe_risk_flags[:2])}"
                    )
            else:
                # Light-weight verifier: only run on high-risk conflicts and allow abstain by default.
                verify_on, trigger_reasons = _should_run_tqa_risk_verify(
                    ctx=ctx,
                    diff_summary=diff_summary,
                    code_result=code_result,
                    sql_result=sql_result,
                    code_shape_issues=code_shape_issues,
                    sql_shape_issues=sql_shape_issues,
                    code_semantic_issues=code_semantic_issues,
                    sql_semantic_issues=sql_semantic_issues,
                    code_severe_risk_flags=code_severe_risk_flags,
                    sql_severe_risk_flags=sql_severe_risk_flags,
                    out_code=out_code,
                    out_sql=out_sql,
                )
                if verify_on:
                    tqa_verify_meta = _run_tqa_risk_verify(
                        ctx=ctx,
                        default_choice=chosen_source,
                        trigger_reasons=trigger_reasons,
                        diff_summary=diff_summary or {},
                        code_result=code_result,
                        sql_result=sql_result,
                        code_shape_issues=code_shape_issues,
                        sql_shape_issues=sql_shape_issues,
                        code_semantic_issues=code_semantic_issues,
                        sql_semantic_issues=sql_semantic_issues,
                        code_risk_flags=code_risk_flags,
                        sql_risk_flags=sql_risk_flags,
                    )
                    verify_choice = str(tqa_verify_meta.get("choice", "abstain")).lower()
                    verify_conf = float(tqa_verify_meta.get("confidence", 0.0) or 0.0)
                    verify_conf_th = _env_float("NL2CODE_TQA_RISK_VERIFY_CONF", 0.82)
                    if verify_choice in {"code", "sql"} and verify_choice != chosen_source and verify_conf >= verify_conf_th:
                        src = "code" if chosen_source == "code" else "sql"
                        dst = verify_choice
                        src_score = _candidate_issue_score(
                            source=src,
                            result=code_result if src == "code" else sql_result,
                            shape_issues=code_shape_issues if src == "code" else sql_shape_issues,
                            semantic_issues=code_semantic_issues if src == "code" else sql_semantic_issues,
                            severe_risk_flags=code_severe_risk_flags if src == "code" else sql_severe_risk_flags,
                            out_meta=out_code if src == "code" else out_sql,
                        )
                        dst_score = _candidate_issue_score(
                            source=dst,
                            result=code_result if dst == "code" else sql_result,
                            shape_issues=code_shape_issues if dst == "code" else sql_shape_issues,
                            semantic_issues=code_semantic_issues if dst == "code" else sql_semantic_issues,
                            severe_risk_flags=code_severe_risk_flags if dst == "code" else sql_severe_risk_flags,
                            out_meta=out_code if dst == "code" else out_sql,
                        )
                        if dst_score <= src_score:
                            chosen_source = dst
                            final_result = code_result if dst == "code" else sql_result
                            final_reason = (
                                f"tqa_risk_verify override: {src}->{dst} "
                                f"(conf={verify_conf:.2f}, src_score={src_score:.2f}, dst_score={dst_score:.2f})"
                            )
                            tqa_verify_meta["applied"] = True
                        else:
                            tqa_verify_meta["reason"] = (
                                f"{tqa_verify_meta.get('reason', '')} | not_applied_due_to_higher_target_risk"
                            )[:280]
                    else:
                        tqa_verify_meta.setdefault("applied", False)
        else:
            sql_meta = {
                "rows": _result_row_count(sql_result),
                "signature": _result_signature(sql_result),
                "preview": _compact_result_preview(sql_result, 5),
                "sql_errors": out_sql.get("sql_errors", 0),
                "last_sql": out_sql.get("last_sql"),
                "last_processed_sql": out_sql.get("last_processed_sql"),
            }
            code_meta = {
                "rows": _result_row_count(code_result),
                "signature": _result_signature(code_result),
                "preview": _compact_result_preview(code_result, 5),
                "turns": out_code.get("turns", 0),
                "read_table_summary": out_code.get("read_table_summary", {}),
                "execute_error_count": out_code.get("execute_error_count", 0),
            }
            judge_meta = _judge_hybrid_choice(
                ctx,
                diff_summary,
                sql_meta,
                code_meta,
                sql_risk_flags=sql_risk_flags,
                code_risk_flags=code_risk_flags,
                default_choice_override=(
                    "sql"
                    if (
                        code_severe_risk_flags
                        or (
                            "all_code_table_reads_are_sampled" in code_risk_flags
                            and _question_has_aggregate_intent(ctx.question or "")
                        )
                        or int(out_code.get("execute_error_count", 0) or 0) > 0
                    )
                    else "code"
                ),
            )
            chosen_source = judge_meta.get("choice", "code")
            if (
                chosen_source == "code"
                and code_severe_risk_flags
                and not sql_severe_risk_flags
            ):
                chosen_source = "sql"
                final_result = sql_result
                final_reason = (
                    "judge picked code but overridden to sql due to severe code risk: "
                    f"{', '.join(code_severe_risk_flags[:2])}"
                )
            elif (
                chosen_source == "code"
                and code_risk_flags
                and not sql_severe_risk_flags
                and not sql_semantic_issues
            ):
                chosen_source = "sql"
                final_result = sql_result
                final_reason = (
                    "judge picked code but overridden to sql due to code risk flags: "
                    f"{', '.join(code_risk_flags[:2])}"
                )
            elif chosen_source == "sql":
                final_result = sql_result
                final_reason = str(judge_meta.get("reason", ""))
            else:
                final_result = code_result
                final_reason = str(judge_meta.get("reason", ""))
    elif code_usable:
        code_shape_issues = _result_shape_mismatch_reasons(
            ctx.question, str(out_code.get("last_execute_code") or ""), code_result
        )
        chosen_source = "code"
        final_result = code_result
        final_reason = (
            "sql path unavailable; use code result"
            + (f" (shape warning: {'; '.join(code_shape_issues[:1])})" if code_shape_issues else "")
        )
    elif sql_usable:
        sql_shape_issues = _result_shape_mismatch_reasons(
            ctx.question, str(out_sql.get("last_processed_sql") or out_sql.get("last_sql") or ""), sql_result
        )
        chosen_source = "sql"
        final_result = sql_result
        final_reason = (
            "code path unavailable; use sql result"
            + (f" (shape warning: {'; '.join(sql_shape_issues[:1])})" if sql_shape_issues else "")
        )
    else:
        chosen_source = "none"
        final_result = None
        final_reason = (
            f"both candidates unusable: code_ok={out_code.get('ok')} code_invalid={code_invalid_reason}; "
            f"sql_ok={out_sql.get('ok')} sql_invalid={sql_invalid_reason}"
        )

    trace: List[Dict[str, Any]] = []
    for t in out_code.get("trace", []) or []:
        trace.append({"source": "code", **t})
    for t in out_sql.get("trace", []) or []:
        trace.append({"source": "sql", **t})

    tool_time_summary: Dict[str, Any] = {}
    _merge_prefixed_tool_summary("code", out_code.get("tool_time_summary", {}), tool_time_summary)
    _merge_prefixed_tool_summary("sql", out_sql.get("tool_time_summary", {}), tool_time_summary)

    judge_prompt_toks = int(judge_meta.get("prompt_tokens", 0) or 0)
    judge_comp_toks = int(judge_meta.get("completion_tokens", 0) or 0)
    verify_prompt_toks = int(tqa_verify_meta.get("prompt_tokens", 0) or 0)
    verify_comp_toks = int(tqa_verify_meta.get("completion_tokens", 0) or 0)
    total_prompt_tokens = (
        int(out_code.get("total_prompt_tokens", 0) or 0)
        + int(out_sql.get("total_prompt_tokens", 0) or 0)
        + judge_prompt_toks
        + verify_prompt_toks
    )
    total_completion_tokens = (
        int(out_code.get("total_completion_tokens", 0) or 0)
        + int(out_sql.get("total_completion_tokens", 0) or 0)
        + judge_comp_toks
        + verify_comp_toks
    )

    if final_result is not None:
        final_result, _ = _apply_dataset_result_contract(ctx, final_result)

    ok = final_result is not None
    final_answer = str(final_result) if ok else None
    _log({"event": "finish", "mode": "hybrid", "chosen_source": chosen_source, "ok": ok, "reason": final_reason[:300]})
    return {
        "ok": ok,
        "answer": final_answer,
        "predicted_result": _to_json_serializable(final_result),
        "trace": trace,
        "trajectory": trajectory,
        "turns": len(trace),
        "error": None if ok else final_reason,
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "total_tokens": total_prompt_tokens + total_completion_tokens,
        "tool_time_summary": tool_time_summary,
        "tool_call_times": (out_code.get("tool_call_times", []) or []) + (out_sql.get("tool_call_times", []) or []),
        "mv_tables_available": out_code.get("mv_tables_available", []),
        "mv_tables_used": out_code.get("mv_tables_used", []),
        "mv_used_count": out_code.get("mv_used_count", 0),
        "runtime_profile": out_code.get("runtime_profile", {}),
        "bottleneck_summary": out_code.get("bottleneck_summary", {}),
        "hybrid": {
            "chosen_source": chosen_source,
            "final_reason": final_reason,
            "coopt_fast": bool(coopt_fast),
            "code_only_fast": bool(code_only_fast),
            "sql_ok": out_sql.get("ok", False),
            "code_ok": out_code.get("ok", False),
            "sql_errors": out_sql.get("sql_errors", 0),
            "code_invalid_reason": code_invalid_reason,
            "sql_invalid_reason": sql_invalid_reason,
            "code_semantic_issues": code_semantic_issues,
            "sql_semantic_issues": sql_semantic_issues,
            "code_shape_issues": code_shape_issues,
            "sql_shape_issues": sql_shape_issues,
            "code_risk_flags": code_risk_flags,
            "sql_risk_flags": sql_risk_flags,
            "code_severe_risk_flags": code_severe_risk_flags,
            "sql_severe_risk_flags": sql_severe_risk_flags,
            "code_shape_repairs": code_shape_repairs,
            "sql_shape_repairs": sql_shape_repairs,
            "diff_summary": diff_summary,
            "judge_meta": judge_meta,
            "tqa_risk_verify": tqa_verify_meta,
        },
        "subagent_outputs": {
            "code": {
                "ok": out_code.get("ok"),
                "error": out_code.get("error"),
                "turns": out_code.get("turns"),
                "predicted_result": _to_json_serializable(out_code.get("predicted_result")),
                "last_execute_code": out_code.get("last_execute_code"),
                "read_table_summary": out_code.get("read_table_summary", {}),
                "execute_error_count": out_code.get("execute_error_count", 0),
                "bottleneck_summary": out_code.get("bottleneck_summary", {}),
            },
            "sql": {
                "ok": out_sql.get("ok"),
                "error": out_sql.get("error"),
                "turns": out_sql.get("turns"),
                "predicted_result": _to_json_serializable(out_sql.get("predicted_result")),
                "last_sql": out_sql.get("last_sql"),
                "last_processed_sql": out_sql.get("last_processed_sql"),
                "last_successful_sql": out_sql.get("last_successful_sql"),
                "last_successful_processed_sql": out_sql.get("last_successful_processed_sql"),
                "raw_sql_attempts": out_sql.get("raw_sql_attempts", []),
                "processed_sql_attempts": out_sql.get("processed_sql_attempts", []),
                "sql_postprocess_applied": out_sql.get("sql_postprocess_applied", False),
                "sql_fallback_used": out_sql.get("sql_fallback_used", False),
                "sql_reverse_calls": out_sql.get("sql_reverse_calls", 0),
                "sql_candidate_count": out_sql.get("sql_candidate_count", 0),
                "best_sql_candidate_score": out_sql.get("best_sql_candidate_score"),
                "best_sql_candidate": out_sql.get("best_sql_candidate"),
            },
        },
    }


def run_react_agent_simple(ctx: AgentContext) -> Dict[str, Any]:
    """
    简化版 ReAct：无 function calling，解析文本中的 Action
    适用于不支援 function calling 的模型
    """
    client = None
    if OpenAI:
        client = OpenAI(base_url=ctx.api_base) if getattr(ctx, "api_base", None) else OpenAI()
    if not client:
        return {"ok": False, "error": "OpenAI client not available", "trace": []}

    prompt = f"""You are a data analyst. Answer this question about the database.

Database: {ctx.db_id}
Question: {ctx.question}

You can use these tools by responding with:
Action: <tool_name> {{"arg": "value"}}

Tools:
- query_sql: {{"sql": "SELECT ..."}}
- get_schema: {{}}
- get_sample: {{"table": "table_name", "n": 5}}
- list_tables: {{}}

First use get_schema or list_tables to understand the database, then use query_sql to get the answer.
When you have the answer, say "Final Answer: <your answer>" and stop.
"""

    messages = [{"role": "user", "content": prompt}]
    trace = []

    for turn in range(ctx.max_turns):
        try:
            resp = client.chat.completions.create(model=ctx.model, messages=messages)
        except Exception as e:
            return {"ok": False, "error": str(e), "trace": trace}

        content = resp.choices[0].message.content or ""
        messages.append({"role": "assistant", "content": content})

        if "Final Answer:" in content:
            final = content.split("Final Answer:")[-1].strip()
            return {"ok": True, "answer": final, "trace": trace, "turns": len(trace)}

        tc = _extract_tool_call_from_text(content)
        if not tc:
            continue

        tool_result = run_tool(tc["name"], tc["arguments"], ctx)
        trace.append({"turn": turn, "tool": tc["name"], "result_ok": tool_result.get("ok")})
        result_str = _format_tool_result(tool_result)
        messages.append({"role": "user", "content": f"Observation:\n{result_str}\n\nContinue or give Final Answer."})

    return {"ok": False, "error": "Max turns reached", "trace": trace}
