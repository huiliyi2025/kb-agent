"""索引构建：解析 → 分块 → 向量 → 落库（一个文档一个事务）。

这里是"接入链路"的入口，值得注意的两个工程点：
    1. 幂等：用 content_hash 判重，同一份文件传两次不会产生两份 chunk；
    2. 状态机：pending → ready / failed。失败必须写 error 落库，否则用户看到的是"永远转圈"。
"""

from __future__ import annotations

import hashlib
import logging
import numpy as np

from kb_agent.core.config import Settings
from kb_agent.core.db import Database
from kb_agent.core.text import to_index_text
from kb_agent.ingest.loaders import Section, UnsupportedFormat, load_document
from kb_agent.ingest.splitter import chunk_sections
from kb_agent.core.embedding import Embedder, EmbeddingError


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build_rows(
    sections: list[Section], embedder: Embedder, chunk_size: int, overlap: int
) -> list[dict]:
    """纯函数：段落 -> 待入库的行。不碰数据库，所以能用单测直接断言。"""
    chunks = chunk_sections(sections, chunk_size=chunk_size, overlap=overlap)
    if not chunks:
        return []
    vectors = embedder.embed([c.embed_text for c in chunks]) if embedder.dim > 0 else None
    if vectors is not None:
        vectors = np.asarray(vectors, dtype="<f4")
        if vectors.shape != (len(chunks), embedder.dim) or not np.isfinite(vectors).all():
            raise EmbeddingError("向量数量、维度或数值不合法，未写入索引")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise EmbeddingError("向量服务返回零向量，未写入索引")
        vectors = (vectors / norms).astype("<f4")
    rows: list[dict] = []
    for i, chunk in enumerate(chunks):
        rows.append(
            {
                "ordinal": chunk.ordinal,
                "text": chunk.text,
                "heading": chunk.heading,
                "page": chunk.page,
                "idx_text": to_index_text(chunk.embed_text),
                "heading_idx": to_index_text(chunk.heading or ""),
                "vec": vectors[i].tobytes() if vectors is not None else None,
            }
        )
    return rows


def ingest_document(
    db: Database,
    settings: Settings,
    embedder: Embedder,
    *,
    owner_id: int,
    filename: str,
    mime: str | None,
    data: bytes,
) -> tuple[str, int, bool]:
    """返回 (doc_id, chunk_count, reused)。reused=True 表示命中了已有文档，直接返回旧结果。"""
    if len(data) > settings.max_upload_mb * 1024 * 1024:
        raise UnsupportedFormat(f"文件超过 {settings.max_upload_mb}MB 限制")
    if not data:
        raise UnsupportedFormat("空文件")
    filename = filename.replace("\\", "/").rsplit("/", 1)[-1][:255] or "unnamed"
    doc_id, created = db.insert_document(
        owner_id=owner_id,
        filename=filename,
        mime=mime or "",
        size_bytes=len(data),
        content_hash=sha256_bytes(data),
    )
    if not created:
        doc = db.get_document(doc_id)
        return doc_id, int(doc["chunk_count"] if doc else 0), True

    try:
        db.update_document_status(doc_id, "indexing")
        sections = load_document(filename, data)
        rows = build_rows(sections, embedder, settings.chunk_size, settings.chunk_overlap)
        if not rows:
            raise UnsupportedFormat("没有解析出任何正文（可能是扫描版 PDF 或空文件）")
        count = db.add_chunks(doc_id, rows)
    except Exception as exc:
        # 失败要把原因写回去，并且让异常继续往上抛给调用方决定 HTTP 状态码
        logging.getLogger("kb_agent").warning("索引失败 doc_id=%s type=%s", doc_id, type(exc).__name__)
        error = str(exc)[:500] if isinstance(exc, UnsupportedFormat) else "索引构建失败，请检查服务配置后重新上传"
        db.update_document_status(doc_id, "failed", error)
        raise
    return doc_id, count, False
