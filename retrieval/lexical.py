"""字面召回：SQLite FTS5（内部就是 BM25）。

BM25 的两个核心直觉：
    1. TF 饱和：一个词出现 10 次不代表相关度是出现 1 次的 10 倍，所以用 tf*(k1+1)/(tf+k1*...)；
    2. IDF + 长度归一：越罕见的词权重越高；越长的文档要打折，否则长文档天然占便宜。
FTS5 的 bm25() 返回的是**负数**（越小越相关），所以下面取反再排序。
"""

from __future__ import annotations

import sqlite3

from kb_agent.core.text import match_expr
from kb_agent.retrieval.types import Hit


def search_lexical(
    conn: sqlite3.Connection,
    query: str,
    limit: int = 20,
    doc_ids: list[str] | None = None,
    owner_id: int | None = None,
) -> list[Hit]:
    if doc_ids == []:
        return []
    expr = match_expr(query)
    if not expr:
        return []

    sql = [
        "SELECT c.id AS chunk_id, c.doc_id, c.ordinal, c.heading, c.page, c.text,",
        "       d.filename, bm25(chunks_fts) AS rank",
        "FROM chunks_fts",
        "JOIN chunks c ON c.id = chunks_fts.rowid",
        "JOIN documents d ON d.id = c.doc_id",
        "WHERE chunks_fts MATCH ? AND d.status='ready'",
    ]
    params: list[object] = [expr]
    if owner_id is not None:
        sql.append("AND d.owner_id=?")
        params.append(owner_id)
    if doc_ids:
        placeholders = ",".join("?" for _ in doc_ids)
        sql.append(f"  AND c.doc_id IN ({placeholders})")
        params.extend(doc_ids)
    sql.append("ORDER BY rank LIMIT ?")
    params.append(limit)

    rows = conn.execute("\n".join(sql), params).fetchall()
    hits: list[Hit] = []
    for row in rows:
        hits.append(
            Hit(
                chunk_id=row["chunk_id"],
                doc_id=row["doc_id"],
                filename=row["filename"],
                text=row["text"],
                heading=row["heading"],
                page=row["page"],
                ordinal=row["ordinal"],
                score=-float(row["rank"]),  # bm25 是负数，取反后越大越相关
                source="lexical",
            )
        )
    return hits
