"""精确向量检索：分批扫描向量、局部 Top-K，最后仅加载命中片段正文。

内存随 batch_size * dim 增长，而不是整个知识库；计算仍是 O(ND)，并非 ANN。
"""
from __future__ import annotations

import sqlite3

import numpy as np

from kb_agent.retrieval.types import Hit


def search_dense(
    conn: sqlite3.Connection,
    query_vec: np.ndarray,
    limit: int = 20,
    doc_ids: list[str] | None = None,
    owner_id: int | None = None,
    batch_size: int = 512,
) -> list[Hit]:
    if query_vec.size == 0 or doc_ids == [] or limit <= 0:
        return []
    if batch_size <= 0:
        raise ValueError("batch_size 必须大于 0")
    query_vec = np.asarray(query_vec, dtype=np.float32)
    if query_vec.ndim != 1 or not np.isfinite(query_vec).all():
        raise ValueError("查询向量必须是一维有限数值")
    norm = np.linalg.norm(query_vec)
    if not norm:
        return []
    query_vec = query_vec / norm
    sql = ["SELECT c.id, c.vec FROM chunks c JOIN documents d ON d.id=c.doc_id",
           "WHERE c.vec IS NOT NULL AND d.status='ready'"]
    params: list[object] = []
    if owner_id is not None:
        sql.append("AND d.owner_id=?")
        params.append(owner_id)
    if doc_ids is not None:
        sql.append(f"AND c.doc_id IN ({','.join('?' for _ in doc_ids)})")
        params.extend(doc_ids)
    cursor = conn.execute(" ".join(sql), params)
    best: list[tuple[float, int]] = []
    while rows := cursor.fetchmany(batch_size):
        valid = [r for r in rows if len(r["vec"]) == query_vec.size * 4]
        if not valid:
            continue
        matrix = np.frombuffer(b"".join(r["vec"] for r in valid), dtype="<f4").reshape(len(valid), -1)
        scores = matrix @ query_vec
        eligible = np.flatnonzero(np.isfinite(scores) & (scores > 0))
        if eligible.size > limit:
            values = scores[eligible]
            cutoff = np.partition(values, values.size - limit)[-limit]
            # 包含边界并列项，随后按 chunk id 排序，保证结果可复现。
            eligible = eligible[values >= cutoff]
        best.extend((float(scores[i]), int(valid[i]["id"])) for i in eligible)
        best = sorted(best, key=lambda x: (-x[0], x[1]))[:limit]
    if not best:
        return []
    ids = [cid for _, cid in best]
    records = conn.execute(
        "SELECT c.*, d.filename FROM chunks c JOIN documents d ON d.id=c.doc_id "
        f"WHERE c.id IN ({','.join('?' for _ in ids)})", ids,
    ).fetchall()
    by_id = {int(r["id"]): r for r in records}
    hits: list[Hit] = []
    for score, cid in best:
        r = by_id.get(cid)
        if r is not None:
            hits.append(Hit(chunk_id=cid, doc_id=r["doc_id"], filename=r["filename"], text=r["text"],
                            heading=r["heading"], page=r["page"], ordinal=r["ordinal"],
                            score=score, source="dense"))
    return hits
