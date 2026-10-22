"""检索编排：改写（可选）→ 双路召回 → RRF 融合 → 轻量重排 → Top-K。

这是整个检索侧的编排，代码量不到 100 行。
读完这个文件就理解了检索的完整流程。
"""

from __future__ import annotations

from kb_agent.core.db import Database
from kb_agent.core.text import tokenize
from kb_agent.retrieval.dense import search_dense
from kb_agent.core.embedding import Embedder
from kb_agent.retrieval.fusion import reciprocal_rank_fusion
from kb_agent.retrieval.lexical import search_lexical
from kb_agent.retrieval.types import Hit


class Retriever:
    def __init__(
        self,
        db: Database,
        embedder: Embedder,
        *,
        rrf_k: int = 60,
        candidates: int = 20,
    ):
        self.db = db
        self.embedder = embedder
        self.rrf_k = rrf_k
        self.candidates = candidates

    def search(
        self, query: str, top_k: int = 5, doc_ids: list[str] | None = None,
        *, owner_id: int | None = None,
    ) -> list[Hit]:
        query = (query or "").strip()
        if not query or doc_ids == []:
            return []
        top_k = max(1, min(top_k, 20))

        conn = self.db.connect()
        try:
            lexical_hits = search_lexical(conn, query, limit=self.candidates, doc_ids=doc_ids, owner_id=owner_id)
            dense_hits: list[Hit] = []
            if self.embedder.dim > 0:
                qvec = self.embedder.embed([query])[0]
                dense_hits = search_dense(conn, qvec, limit=self.candidates, doc_ids=doc_ids, owner_id=owner_id)
        finally:
            conn.close()

        if not lexical_hits and not dense_hits:
            return []

        if dense_hits:
            fused = reciprocal_rank_fusion([lexical_hits, dense_hits], k=self.rrf_k)
        else:
            fused = reciprocal_rank_fusion([lexical_hits], k=self.rrf_k, sources=("lexical",))

        return self._rerank(query, fused)[:top_k]

    # ------------------------------------------------------------------ 重排
    @staticmethod
    def _rerank(query: str, hits: list[Hit]) -> list[Hit]:
        """轻量重排：在 RRF 排名基础上，奖励"查询词覆盖率高"的片段。

        为什么不做 Cross-Encoder 重排？成本高（要再跑一个模型）、延迟大。
        先用规则把"明显不相关但碰巧命中一个常见词"的片段压下去，
        真正的精排留给线上再上 bge-reranker——这是明确的取舍。
        """
        if not hits:
            return []
        q_terms = set(tokenize(query))
        best = max(h.score for h in hits) or 1.0
        for hit in hits:
            rrf_norm = hit.score / best
            doc_terms = set(tokenize(f"{hit.heading or ''} {hit.text}"))
            coverage = len(q_terms & doc_terms) / len(q_terms) if q_terms else 0.0
            hit.components["coverage"] = round(coverage, 4)
            hit.score = round(0.65 * rrf_norm + 0.35 * coverage, 6)
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits
