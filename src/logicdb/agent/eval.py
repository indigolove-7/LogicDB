from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from typing import Any, Tuple

from logicdb.agent.tools import tool_query_sql


def _normalize_val(v) -> str:
    if v is None:
        return "__NULL__"
    try:
        if hasattr(v, "item"):
            v = v.item()
        if isinstance(v, (int, float)):
            if isinstance(v, float) and v == int(v):
                return str(int(v))
            return str(v).strip()
    except (TypeError, ValueError):
        pass
    if isinstance(v, str):
        return v.strip().lower()
    return str(v).strip().lower()


def _is_scalar(v: Any) -> bool:
    return isinstance(v, (str, int, float, bool)) or v is None


def _unwrap_wrapped_result(v: Any) -> Any:
    cur = v
    for _ in range(3):
        if not isinstance(cur, dict):
            break
        keys = set(str(k).lower() for k in cur.keys())
        for cand in ("predicted_result", "answer", "result", "final_answer", "final", "value", "label"):
            if cand in keys:
                real_key = next((k for k in cur.keys() if str(k).lower() == cand), None)
                if real_key is not None:
                    cur = cur.get(real_key)
                    break
        else:
            break
    return cur


def _normalize_text_token(s: str) -> str:
    s = str(s).strip().strip("\"'`")
    s = re.sub(r"\s+", " ", s)
    return s.lower()


def _normalize_wtq_text(s: str) -> str:
    x = str(s)
    x = "".join(c for c in unicodedata.normalize("NFKD", x) if unicodedata.category(c) != "Mn")
    x = re.sub(r"[‘’´`]", "'", x)
    x = re.sub(r"[“”]", "\"", x)
    x = re.sub(r"[‐‑‒–—−]", "-", x)
    while True:
        old = x
        x = re.sub(r"((?<!^)\[[^\]]*\]|\[\d+\]|[•♦†‡*#+])*$", "", x.strip())
        x = re.sub(r"(?<!^)( \([^)]*\))*$", "", x.strip())
        x = re.sub(r'^"([^"]*)"$', r"\1", x.strip())
        if x == old:
            break
    if x.endswith("."):
        x = x[:-1]
    x = re.sub(r"\s+", " ", x, flags=re.U).lower().strip()
    return x


def _normalize_answer_token(v: Any) -> str:
    if v is None:
        return "__null__"
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, (int, float)):
        if isinstance(v, float):
            try:
                if not math.isfinite(v):
                    return "__null__"
                if v == int(v):
                    return str(int(v))
                return str(v)
            except Exception:
                return "__null__"
        return str(v)
    s = _normalize_wtq_text(v)
    if s in {"", "none", "null", "nan"}:
        return "__null__"
    num_s = s.replace(",", "")
    try:
        x = float(num_s)
        if x == int(x):
            return str(int(x))
        return str(x)
    except Exception:
        pass
    return s


def _flatten_result_values(result: Any) -> list[Any]:
    result = _unwrap_wrapped_result(result)
    if _is_scalar(result):
        return [result]
    if isinstance(result, dict):
        return list(result.values())
    if isinstance(result, (list, tuple, set)):
        vals: list[Any] = []
        for item in result:
            if isinstance(item, dict):
                if len(item) == 1:
                    vals.extend(item.values())
                else:
                    vals.extend(list(item.values()))
            elif isinstance(item, (list, tuple, set)):
                vals.extend(_flatten_result_values(item))
            else:
                vals.append(item)
        return vals
    return [result]


def _parse_binary_label(v: Any) -> int | None:
    if v is None:
        return None
    if isinstance(v, bool):
        return 1 if v else 0
    if isinstance(v, (int, float)):
        return 1 if int(v) != 0 else 0
    s = _normalize_text_token(v)
    if s in {"1", "true", "yes", "entailed", "entailment", "support", "supported"}:
        return 1
    if s in {"0", "false", "no", "refuted", "refute", "contradiction"}:
        return 0
    try:
        x = float(s)
        return 1 if int(x) != 0 else 0
    except Exception:
        return None


def _has_answer_target(rec: dict) -> bool:
    ds = str(rec.get("source_dataset") or "").strip().lower()
    db_id = str(rec.get("db_id") or "").strip().lower()
    if ds in {"wtq", "tabfact"}:
        return True
    if db_id.startswith("wtq_") or db_id.startswith("tabfact_"):
        return True
    return ("gold_answer" in rec) and (rec.get("gold_answer") is not None)


def answer_match(pred_result: Any, gold_answer: Any, rec: dict) -> Tuple[bool, Any, Any, str | None]:
    ds = str(rec.get("source_dataset") or "").strip().lower()
    db_id = str(rec.get("db_id") or "").strip().lower()
    is_tabfact = ds == "tabfact" or db_id.startswith("tabfact_")
    if is_tabfact:
        gold_label = _parse_binary_label(gold_answer)
        if gold_label is None:
            return False, None, None, f"invalid gold label: {gold_answer}"
        pred_vals = _flatten_result_values(pred_result)
        pred_label = None
        for v in pred_vals:
            pred_label = _parse_binary_label(v)
            if pred_label is not None:
                break
        if pred_label is None:
            return False, pred_result, gold_label, "cannot parse predicted label"
        return pred_label == gold_label, pred_label, gold_label, None

    gold_vals_raw = gold_answer if isinstance(gold_answer, list) else [gold_answer]
    pred_vals_raw = _flatten_result_values(pred_result)
    gold_vals = [_normalize_answer_token(x) for x in gold_vals_raw if x is not None]
    pred_vals = [_normalize_answer_token(x) for x in pred_vals_raw if x is not None]
    gold_ctr = Counter(gold_vals)
    pred_ctr = Counter(pred_vals)
    match = pred_ctr == gold_ctr
    return match, list(pred_ctr.elements()), list(gold_ctr.elements()), None


def _row_to_tuple(row) -> tuple:
    if isinstance(row, dict):
        return tuple(sorted(_normalize_val(v) for v in row.values()))
    if isinstance(row, (list, tuple)):
        return tuple(_normalize_val(x) for x in row)
    return (_normalize_val(row),)


def _result_to_set(result) -> set:
    if result is None:
        return set()
    if isinstance(result, (int, float, str)):
        return {(_normalize_val(result),)}
    if isinstance(result, dict):
        return {_row_to_tuple(result)}
    if isinstance(result, list):
        if not result:
            return set()
        return {_row_to_tuple(r) for r in result}
    return {(_normalize_val(result),)}


def _gold_result_to_set(rows: list) -> set:
    return {_row_to_tuple(r) for r in rows or []}


def exec_match(pred_result, gold_sql: str, db_path: str) -> tuple[bool, set, set, str | None]:
    gold_res = tool_query_sql(gold_sql, db_path)
    if not gold_res.get("ok"):
        return False, set(), set(), gold_res.get("error") or "gold sql failed"
    gold_rows = gold_res.get("result") or []
    pred_set = _result_to_set(pred_result)
    gold_set = _gold_result_to_set(gold_rows)
    return pred_set == gold_set, pred_set, gold_set, None
