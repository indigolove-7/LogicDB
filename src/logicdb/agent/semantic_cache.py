#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于 OpenAI text-embedding-3-small + FAISS 的语义缓存。

设计：
- Key   : (db_id, question) 的语义向量（dim=256，余弦相似度）
- 分区  : 按 db_id 独立建 FAISS 索引，查询时只搜同 DB 的候选项
- 存储  : 单个 JSON 文件（entries 含 embedding 列表），无需额外 FAISS 文件
- 阈值  : similarity >= threshold 才视为命中（默认 0.90）

使用流程：
    cache = SemanticCache("logicdb_cache/sem_cache.json", threshold=0.90)
    result, sim = cache.lookup(db_id, question)
    if result:
        ...  # cache hit
    else:
        result = run_agent(...)
        cache.add(db_id, question, result)
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import faiss
    _FAISS_AVAILABLE = True
except ImportError:
    faiss = None  # type: ignore
    _FAISS_AVAILABLE = False


# ─── embedding 调用（带重试，共享全局 client）────────────────────────────────

_openai_client: Any = None


def _get_client() -> Any:
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        _openai_client = OpenAI()
    return _openai_client


def embed_text(
    text: str,
    model: str = "text-embedding-3-small",
    dimensions: int = 256,
    retries: int = 3,
) -> Optional[np.ndarray]:
    """调用 OpenAI embedding API，返回 L2 归一化的 float32 向量，失败返回 None。"""
    for attempt in range(retries):
        try:
            resp = _get_client().embeddings.create(
                model=model,
                input=[text],
                dimensions=dimensions,
            )
            vec = np.array(resp.data[0].embedding, dtype="float32")
            norm = np.linalg.norm(vec)
            if norm > 1e-9:
                vec /= norm
            return vec
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(1.5 ** attempt)
            else:
                print(f"[SemanticCache] embed error: {e}")
    return None


# ─── SemanticCache ──────────────────────────────────────────────────────────

class SemanticCache:
    """
    OpenAI embedding + FAISS 语义缓存。
    - 按 db_id 分区：查询时只在同 DB 内搜索，避免跨 DB 误命中
    - 持久化：JSON 文件，entries 含 embedding（list[float]）
    - 线程安全：单进程写、多进程只读（for batch eval）
    """

    def __init__(
        self,
        cache_path: str,
        threshold: float = 0.90,
        embed_model: str = "text-embedding-3-small",
        embed_dim: int = 256,
    ) -> None:
        self.cache_path = cache_path
        self.threshold = threshold
        self.embed_model = embed_model
        self.embed_dim = embed_dim

        # entries: list of {db_id, question, embedding:[float], result:{...}}
        self.entries: List[Dict[str, Any]] = []
        # per-db_id FAISS index (in-memory)
        self._db_index: Dict[str, Dict[str, Any]] = {}
        # {db_id: {"index": faiss.Index, "positions": [int into self.entries]}}

        self._load()

    # ── 公开接口 ──────────────────────────────────────────────────────────

    def lookup(self, db_id: str, question: str) -> Tuple[Optional[Dict[str, Any]], float]:
        """
        语义检索：返回 (cached_result, similarity)。
        未命中或无 FAISS 时返回 (None, 0.0)。
        """
        if not _FAISS_AVAILABLE:
            return self._lookup_numpy(db_id, question)

        db_id = (db_id or "").strip().lower()
        if db_id not in self._db_index:
            return None, 0.0

        q_vec = embed_text(question, model=self.embed_model, dimensions=self.embed_dim)
        if q_vec is None:
            return None, 0.0

        idx_data = self._db_index[db_id]
        D, I = idx_data["index"].search(q_vec.reshape(1, -1), 1)
        if I[0][0] < 0:
            return None, 0.0

        sim = float(D[0][0])  # inner product = cosine（归一化后）
        if sim < self.threshold:
            return None, sim

        pos = idx_data["positions"][I[0][0]]
        return self.entries[pos]["result"], sim

    def add(self, db_id: str, question: str, result: Dict[str, Any]) -> bool:
        """
        将一条执行结果加入缓存（内存 + 磁盘）。
        成功返回 True，embedding 失败返回 False。
        """
        db_id = (db_id or "").strip().lower()
        vec = embed_text(question, model=self.embed_model, dimensions=self.embed_dim)
        if vec is None:
            return False

        # 精简 result 中的大字段
        result_lite = _slim_result(result)

        entry = {
            "db_id": db_id,
            "question": question,
            "embedding": vec.tolist(),
            "result": result_lite,
        }
        pos = len(self.entries)
        self.entries.append(entry)
        self._add_to_index(db_id, vec, pos)
        self._save()
        return True

    @property
    def size(self) -> int:
        return len(self.entries)

    def stats(self) -> Dict[str, Any]:
        db_counts: Dict[str, int] = {}
        for e in self.entries:
            db_counts[e["db_id"]] = db_counts.get(e["db_id"], 0) + 1
        return {
            "total_entries": self.size,
            "num_dbs": len(db_counts),
            "top_dbs": sorted(db_counts.items(), key=lambda x: -x[1])[:5],
            "faiss_available": _FAISS_AVAILABLE,
            "embed_model": self.embed_model,
            "embed_dim": self.embed_dim,
            "threshold": self.threshold,
        }

    # ── 内部方法 ──────────────────────────────────────────────────────────

    def _add_to_index(self, db_id: str, vec: np.ndarray, pos: int) -> None:
        if not _FAISS_AVAILABLE:
            return
        if db_id not in self._db_index:
            idx = faiss.IndexFlatIP(self.embed_dim)
            self._db_index[db_id] = {"index": idx, "positions": []}
        self._db_index[db_id]["index"].add(vec.reshape(1, -1))
        self._db_index[db_id]["positions"].append(pos)

    def _build_index(self) -> None:
        """从 entries 重建所有 FAISS 索引（load 时调用）。"""
        if not _FAISS_AVAILABLE:
            return
        self._db_index = {}
        for pos, entry in enumerate(self.entries):
            db_id = entry["db_id"]
            vec = np.array(entry["embedding"], dtype="float32")
            self._add_to_index(db_id, vec, pos)

    def _lookup_numpy(self, db_id: str, question: str) -> Tuple[Optional[Dict[str, Any]], float]:
        """FAISS 不可用时的 numpy 兜底（线性扫描）。"""
        db_id = (db_id or "").strip().lower()
        candidates = [(e, np.array(e["embedding"], dtype="float32"))
                      for e in self.entries if e.get("db_id") == db_id]
        if not candidates:
            return None, 0.0
        q_vec = embed_text(question, model=self.embed_model, dimensions=self.embed_dim)
        if q_vec is None:
            return None, 0.0
        best_sim, best_entry = -1.0, None
        for entry, vec in candidates:
            sim = float(np.dot(q_vec, vec))
            if sim > best_sim:
                best_sim = sim
                best_entry = entry
        if best_sim >= self.threshold and best_entry:
            return best_entry["result"], best_sim
        return None, best_sim

    def _save(self) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(self.cache_path)) or ".", exist_ok=True)
        data = {
            "embed_model": self.embed_model,
            "embed_dim": self.embed_dim,
            "threshold": self.threshold,
            "entries": self.entries,
        }
        tmp = self.cache_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        os.replace(tmp, self.cache_path)

    def _load(self) -> None:
        if not os.path.exists(self.cache_path):
            return
        try:
            with open(self.cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.embed_dim = int(data.get("embed_dim", self.embed_dim))
            self.entries = data.get("entries", [])
            self._build_index()
            print(f"[SemanticCache] Loaded {self.size} entries "
                  f"({len(self._db_index)} dbs) from {self.cache_path}")
        except Exception as e:
            print(f"[SemanticCache] Load error: {e}")
            self.entries = []
            self._db_index = {}


# ─── 辅助 ────────────────────────────────────────────────────────────────────

def _to_json_safe(obj: Any) -> Any:
    """递归转为 JSON 可序列化类型，避免 Int64DType / numpy 等导致 dump 报错。"""
    if obj is None:
        return None
    if isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, (int, float)):
        if hasattr(obj, "item"):
            return obj.item()
        return obj
    if isinstance(obj, set):
        return _to_json_safe(list(obj))
    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _to_json_safe(v) for k, v in obj.items()}
    # pandas / numpy 等
    if hasattr(obj, "tolist"):
        return _to_json_safe(obj.tolist())
    if hasattr(obj, "item"):
        return obj.item()
    if hasattr(obj, "to_dict"):
        try:
            d = obj.to_dict(orient="records") if hasattr(obj, "shape") and len(getattr(obj, "shape", ())) == 2 else obj.to_dict()
        except Exception:
            d = obj.to_dict()
        return _to_json_safe(d)
    # dtype 等不可序列化：转成字符串
    if type(obj).__name__ in ("Int64DType", "Float64DType", "object", "bool_", "int64", "float64"):
        return str(obj)
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        return str(obj)


def _slim_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """只保留评估和展示所需的字段，去掉 trajectory/tool_call_times 等大字段；并转为 JSON 安全类型。"""
    keys = ("ok", "answer", "predicted_result", "turns", "error",
            "total_prompt_tokens", "total_completion_tokens", "total_tokens",
            "mv_used_count")
    out = {k: result[k] for k in keys if k in result}
    return _to_json_safe(out)
