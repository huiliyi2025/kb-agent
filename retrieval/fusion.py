"""RRF（Reciprocal Rank Fusion）——多路召回怎么合并。

为什么不是"把 BM25 分数和余弦相似度加权平均"？
    因为两者量纲不同：余弦在 [-1,1]，BM25 可以到 20+。归一化之后仍然不稳定，
    换一批数据权重就得重调。RRF 只看**排名**不看分数，天然免疫量纲问题：

        score(d) = Σ_i  1 / (k + rank_i(d))        k 通常取 60

这是"混合检索"的常用融合方式。
"""

from __future__ import annotations

from kb_agent.retrieval.types import Hit


def reciprocal_rank_fusion(
    ranked_lists: list[list[Hit]], k: int = 60, sources: tuple[str, ...] = ("lexical", "dense")
) -> list[Hit]:
    fused: dict[int, Hit] = {}
    scores: dict[int, float] = {}

    for list_index, hits in enumerate(ranked_lists):
        tag = sources[list_index] if list_index < len(sources) else f"list{list_index}"
        for rank, hit in enumerate(hits):
            hits_id = hit.chunk_id
            if hits_id not in fused:
                fused[hits_id] = Hit(
                    chunk_id=hit.chunk_id,
                    doc_id=hit.doc_id,
                    filename=hit.filename,
                    text=hit.text,
                    heading=hit.heading,
                    page=hit.page,
                    ordinal=hit.ordinal,
                )
                scores[hits_id] = 0.0
            scores[hits_id] += 1.0 / (k + rank + 1)
            # 记录"被哪几路召回"，两路都命中的 chunk 通常是真正相关的
            last = fused[hits_id].components.get(tag)
            if last is None:
                fused[hits_id].components[tag] = float(rank + 1)

    out: list[Hit] = []
    for chunk_id, hit in fused.items():
        hit.score = scores[chunk_id]
        hit.source = "hybrid" if len(hit.components) > 1 else next(iter(hit.components), "")
        out.append(hit)
    out.sort(key=lambda h: h.score, reverse=True)
    return out
