"""RAG 编排：用户范围检索 → 可回答性门槛 → 有预算的上下文 → 引用校验 → 原子保存。"""
from __future__ import annotations

import json
import re
import time
from typing import Iterator, Sequence

from kb_agent.core.config import Settings
from kb_agent.core.db import Database
from kb_agent.core.text import content_chars
from kb_agent.gen.llm import LLM
from kb_agent.gen.prompt import INSUFFICIENT_MARK, build_context, build_messages
from kb_agent.retrieval.idf import CharIdf, get_char_idf
from kb_agent.retrieval.service import Retriever
from kb_agent.retrieval.types import Hit
from kb_agent.schemas import AskResponse, Citation, RetrievedChunk

REFUSAL_TEXT = (
    "知识库里没有找到能支撑这个问题的内容，所以我不做猜测。"
    "建议：① 换一个更具体的问法（带上制度名、时间、部门）；② 先上传包含该内容的文档。"
)
_CITE_RE = re.compile(r"\[(\d+)\]")


def assess_answerability(
    question: str, hits: Sequence[Hit], idf: CharIdf | None = None, top_n: int = 3
) -> float:
    """IDF 加权字符覆盖率，是启发式分数，不是模型正确率或概率。"""
    q_chars = content_chars(question)
    if not q_chars or not hits:
        return 0.0
    if idf is not None:
        total_weight = idf.weight_of(q_chars) or 1.0

        def covered_weight(chars: set[str]) -> float:
            return idf.weight_of(chars & q_chars) / total_weight
    else:
        def covered_weight(chars: set[str]) -> float:
            return len(chars & q_chars) / len(q_chars)
    merged = content_chars(" ".join(f"{h.heading or ''} {h.text}" for h in hits[:top_n]))
    top1 = covered_weight(content_chars(f"{hits[0].heading or ''} {hits[0].text}"))
    return round(0.6 * covered_weight(merged) + 0.4 * top1, 4)


def resolve_citations(answer: str, citations: Sequence[Citation]) -> list[Citation]:
    by_index = {c.index: c for c in citations}
    ordered: list[Citation] = []
    seen: set[int] = set()
    for raw in _CITE_RE.findall(answer):
        index = int(raw)
        if index not in seen and index in by_index:
            seen.add(index)
            ordered.append(by_index[index])
    return ordered


def finalize_answer(answer: str, citations: Sequence[Citation]) -> tuple[str, list[Citation], bool]:
    """引用缺失或越界时拒答；合法编号只证明来源存在，不证明陈述正确。"""
    answer = answer.strip()
    indices = {int(raw) for raw in _CITE_RE.findall(answer)}
    available = {c.index for c in citations}
    if (not answer or INSUFFICIENT_MARK in answer or REFUSAL_TEXT in answer
            or not indices or not indices <= available):
        return REFUSAL_TEXT, [], True
    return answer, resolve_citations(answer, citations), False


class RagService:
    def __init__(self, db: Database, retriever: Retriever, llm: LLM, settings: Settings):
        self.db, self.retriever, self.llm, self.settings = db, retriever, llm, settings

    def _retrieve(self, question, top_k, doc_ids, user_id):
        hits = self.retriever.search(question, top_k=top_k or self.settings.top_k,
                                     doc_ids=doc_ids, owner_id=user_id)
        confidence = assess_answerability(question, hits, get_char_idf(self.db, user_id))
        retrieved = [RetrievedChunk(
            chunk_id=h.chunk_id, doc_id=h.doc_id, filename=h.filename, heading=h.heading,
            score=round(h.score, 4), source=h.source,
            components={key: round(value, 4) for key, value in h.components.items()},
        ) for h in hits]
        return hits, confidence, retrieved

    def _run(self, question: str, *, stream: bool, top_k: int | None = None,
             doc_ids: list[str] | None = None, session_id: str | None = None,
             user_id: int | None = None) -> Iterator[dict]:
        started = time.perf_counter()
        history: list[dict] = []
        if session_id:
            if user_id is None:
                raise PermissionError("读取会话必须提供用户身份")
            self.db.assert_session_owner(session_id, user_id)
            history = [dict(row) for row in self.db.recent_messages(session_id, limit=4)]
        hits, confidence, retrieved = self._retrieve(question, top_k, doc_ids, user_id)
        yield {"event": "meta", "data": {"confidence": confidence, "session_id": session_id,
                                           "retrieved": [r.model_dump() for r in retrieved]}}
        answer, used, refused = REFUSAL_TEXT, [], True
        if hits and confidence >= self.settings.min_answerability:
            context, citations = build_context(hits)
            messages = build_messages(question, context, history)
            if stream:
                buffer: list[str] = []
                for token in self.llm.stream(messages):
                    buffer.append(token)
                    yield {"event": "token", "data": {"text": token}}
                raw_answer = "".join(buffer)
            else:
                raw_answer = self.llm.complete(messages)
            answer, used, refused = finalize_answer(raw_answer, citations)
        response = AskResponse(
            answer=answer, citations=used, refused=refused, confidence=confidence,
            latency_ms=int((time.perf_counter() - started) * 1000), session_id=session_id,
            model=self.llm.name, retrieved=retrieved,
        )
        if session_id and user_id is not None:
            self.db.save_exchange(session_id, user_id, question, answer,
                                  json.dumps([c.model_dump() for c in used], ensure_ascii=False))
        yield {"event": "done", "data": response.model_dump()}

    def ask(self, question: str, *, top_k: int | None = None, doc_ids: list[str] | None = None,
            session_id: str | None = None, user_id: int | None = None) -> AskResponse:
        for event in self._run(question, stream=False, top_k=top_k, doc_ids=doc_ids,
                               session_id=session_id, user_id=user_id):
            if event["event"] == "done":
                return AskResponse(**event["data"])
        raise RuntimeError("RAG 未完成")

    def stream(self, question: str, *, top_k: int | None = None, doc_ids: list[str] | None = None,
               session_id: str | None = None, user_id: int | None = None) -> Iterator[dict]:
        yield from self._run(question, stream=True, top_k=top_k, doc_ids=doc_ids,
                             session_id=session_id, user_id=user_id)
