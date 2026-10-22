"""Agent：给模型一个"工具箱"，让它在循环里自己决定查什么、查几次。

和 RAG 的区别：
    RAG  = 固定流水线：一次检索 → 一次生成。可控、便宜、延迟低。
    Agent = 模型自己决定流程：可能"先查制度名，再查具体条款"，多次检索后综合。
            灵活，但贵、慢、不可控，必须有 max_steps 兜底。

工程上三件事必须做，否则线上必炸：
    1. 步数上限（防止死循环烧钱）；
    2. 工具白名单（防止模型调用不存在的工具/传非法参数）；
    3. 每一步留 trace（否则出了问题你根本不知道它干了什么）。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Callable, Sequence

from kb_agent.core.db import Database
from kb_agent.gen.llm import LLM, LLMError
from kb_agent.gen.prompt import AGENT_SYSTEM
from kb_agent.gen.rag import finalize_answer
from kb_agent.retrieval.service import Retriever
from kb_agent.schemas import AgentResponse, AgentStep, Citation


@dataclass
class Tool:
    name: str
    description: str
    fn: Callable[..., str]


def make_tools(retriever: Retriever, db: Database, citations: list[Citation],
               *, owner_id: int | None = None) -> list[Tool]:
    """工具通过闭包共享一个 citations 列表，编号在整轮对话里保持稳定。"""
    by_chunk: dict[int, int] = {}

    def search_kb(query: str, top_k: int = 5) -> str:
        if not isinstance(query, str) or not query.strip() or len(query) > 500:
            raise ValueError("query 必须是 1~500 字符的非空文本")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 10:
            raise ValueError("top_k 必须是 1~10 的整数")
        hits = retriever.search(query, top_k=top_k, owner_id=owner_id)
        if not hits:
            return "（没有检索到相关资料）"
        blocks: list[str] = []
        for hit in hits:
            if hit.chunk_id not in by_chunk:
                index = len(citations) + 1
                by_chunk[hit.chunk_id] = index
                citations.append(
                    Citation(
                        index=index,
                        doc_id=hit.doc_id,
                        filename=hit.filename,
                        heading=hit.heading,
                        page=hit.page,
                        chunk_id=hit.chunk_id,
                        snippet=hit.snippet(),
                    )
                )
            index = by_chunk[hit.chunk_id]
            blocks.append(
                f"[{index}] 来源: {hit.filename} / {hit.heading or '-'}\n内容: {hit.text[:1000]}"
            )
        return "\n\n".join(blocks)

    def doc_outline(doc_id: str) -> str:
        doc = db.get_document(str(doc_id))
        if doc is None or (owner_id is not None and int(doc["owner_id"]) != owner_id):
            return "文档不存在或无权访问"
        headings = db.list_headings(str(doc_id))
        if not headings:
            return f"文档 {doc_id} 没有章节信息（可能是纯文本或解析时没识别出标题）"
        return "该文档的章节：\n" + "\n".join(f"- {h}" for h in headings)

    return [
        Tool(
            name="search_kb",
            description='检索企业知识库，参数：{"query": "检索词", "top_k": 5}。返回带 [n] 编号的片段。',
            fn=search_kb,
        ),
        Tool(
            name="doc_outline",
            description='查看某份文档的章节目录，参数：{"doc_id": "doc_xxx"}。适合先看全貌再定位。',
            fn=doc_outline,
        ),
    ]


class Agent:
    def __init__(self, llm: LLM, tools: Sequence[Tool], max_steps: int = 4):
        self.llm = llm
        self.tools = {tool.name: tool for tool in tools}
        self.max_steps = max(1, min(max_steps, 8))

    def _tool_spec(self) -> str:
        return "\n".join(f"- {tool.name}: {tool.description}" for tool in self.tools.values())

    def run(self, question: str, citations: list[Citation] | None = None) -> AgentResponse:
        started = time.perf_counter()
        citations = citations if citations is not None else []
        messages: list[dict] = [
            {"role": "system", "content": AGENT_SYSTEM.format(tools=self._tool_spec(), max_steps=self.max_steps)},
            {"role": "user", "content": f"问题：{question}"},
        ]
        steps: list[AgentStep] = []
        answer = ""
        used_tools: set[str] = set()

        for step_no in range(1, self.max_steps + 1):
            try:
                decision = self.llm.complete_json(messages)
            except LLMError as exc:
                raise LLMError("Agent 模型调用失败") from exc

            if not isinstance(decision, dict):
                raise LLMError("Agent 决策必须是 JSON 对象")

            if "final" in decision and decision["final"]:
                answer = str(decision["final"]).strip()
                steps.append(
                    AgentStep(
                        step=step_no,
                        thought=str(decision.get("thought", "")),
                        action="final",
                        action_input={},
                        observation=answer[:200],
                    )
                )
                break

            action = str(decision.get("action", ""))
            action_input = decision.get("action_input") or {}
            call_key = json.dumps([action, action_input], sort_keys=True, ensure_ascii=False)
            tool = self.tools.get(action)
            if not isinstance(action_input, dict):
                observation = "action_input 必须是 JSON 对象"
            elif tool is None:
                observation = f"工具 {action} 不存在，可用工具：{', '.join(self.tools)}"
            elif call_key in used_tools:
                observation = "相同工具和参数已经调用，请调整检索词或作答"
            else:
                used_tools.add(call_key)
                try:
                    observation = tool.fn(**action_input)
                except (TypeError, ValueError) as exc:
                    observation = f"参数错误：{exc}"
            steps.append(
                AgentStep(
                    step=step_no,
                    thought=str(decision.get("thought", "")),
                    action=action,
                    action_input=action_input if isinstance(action_input, dict) else {},
                    observation=observation[:600],
                )
            )
            messages.append({"role": "assistant", "content": json.dumps(decision, ensure_ascii=False)})
            messages.append({"role": "user", "content": f"观察结果: {observation}"})
        else:
            # 步数用尽还没收敛：用最后一次观察兜底，绝不返回空
            last_observation = steps[-1].observation if steps else "（没有可用信息）"
            answer = f"（已达到最大步数 {self.max_steps}，以下是已检索到的内容）\n{last_observation}"

        answer, used_citations, _ = finalize_answer(answer, citations)
        return AgentResponse(
            answer=answer,
            steps=steps,
            citations=used_citations,
            elapsed_ms=int((time.perf_counter() - started) * 1000),
        )
