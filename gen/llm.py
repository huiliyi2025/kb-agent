"""LLM 客户端：一个协议 + 两个实现。

这是整个项目最重要的设计决策之一：**业务代码永远不 import openai**。
模型调用收敛到 Protocol 后面，换模型 = 改配置，
并且离线能跑单测（FakeLLM 保证 CI 不花钱、不联网、结果确定）。
"""

from __future__ import annotations

import json
import re
import time
from typing import Iterator, Protocol, Sequence

import httpx

Message = dict[str, str]


class LLMError(RuntimeError):
    """模型调用失败。上层捕获后决定降级还是 502。"""


class LLM(Protocol):
    name: str

    def complete(
        self, messages: Sequence[Message], *, temperature: float = 0.2, max_tokens: int = 800
    ) -> str: ...

    def stream(
        self, messages: Sequence[Message], *, temperature: float = 0.2, max_tokens: int = 800
    ) -> Iterator[str]: ...

    def complete_json(
        self, messages: Sequence[Message], *, temperature: float = 0.0, max_tokens: int = 600
    ) -> dict: ...


# ===================================================================== 离线 Mock
_CONTEXT_BLOCK_RE = re.compile(
    r"\[(\d+)\]\s*来源:\s*(.+?)\n\s*内容:\s*(.*?)(?=\n\s*\[\d+\]\s*来源:|\n【资料结束】|\Z)",
    re.S,
)
_SENTENCE_RE = re.compile(r"[^。！？!?\n]+[。！？!?]?")


def _parse_contexts(text: str) -> list[tuple[int, str, str]]:
    return [(int(m.group(1)), m.group(2).strip(), m.group(3).strip()) for m in _CONTEXT_BLOCK_RE.finditer(text)]


class FakeLLM:
    """确定性的"抽取式"假模型。

    它不生成任何新内容，只从检索到的资料里挑最相关的句子并带上引用编号。
    这不是为了糊弄——它的价值在于：
        * CI / 单测不依赖网络和 API Key，跑得起、跑得快、结果稳定；
        * 能验证"检索 + 引用 + 拒答"整条链路是否正确，而不用管模型好不好。
    接上真模型只需要把 KB_LLM_PROVIDER 改成 openai。
    """

    name = "fake-extractive"

    def __init__(self) -> None:
        self.calls = 0

    # ------------------------------------------------------------------ 工具
    @staticmethod
    def _pick_sentences(question: str, contexts: list[tuple[int, str, str]], limit: int = 2):
        from kb_agent.core.text import tokenize

        q_terms = set(tokenize(question))
        scored: list[tuple[float, int, str]] = []
        # 只在排名最高的 2 个片段里挑句子：跨文档拼答案会让引用看起来毫无逻辑
        allowed = {index for index, _source, _body in contexts[:2]}
        for index, source, body in contexts:
            if index not in allowed:
                continue
            src_terms = set(tokenize(source))  # 章节标题也算上下文（"二、住宿标准"）
            for sentence in _SENTENCE_RE.findall(body):
                sentence = sentence.strip()
                if len(sentence) < 6:
                    continue
                terms = set(tokenize(sentence))
                if not terms:
                    continue
                overlap = len(q_terms & terms)
                if overlap == 0:
                    continue
                # 正文命中为主，标题命中给 40% 的加成。
                # 注意：不要给"句子更长"加权——那会让相邻条款（450 元）盖过正确条款（600 元）。
                score = overlap / max(1, len(q_terms)) + 0.4 * len(q_terms & src_terms) / max(
                    1, len(q_terms)
                )
                scored.append((score, index, sentence))
        scored.sort(key=lambda x: x[0], reverse=True)

        picked: list[tuple[int, str]] = []
        used_index: set[int] = set()
        for _score, index, sentence in scored:
            if len(picked) >= limit:
                break
            if index in used_index:
                continue
            used_index.add(index)
            picked.append((index, sentence))
        return picked

    # ------------------------------------------------------------------ 协议
    def complete(self, messages, *, temperature: float = 0.2, max_tokens: int = 800) -> str:
        self.calls += 1
        joined = messages[-1].get("content", "")
        contexts = _parse_contexts(joined)
        question = _extract_question(joined)
        if not contexts:
            return "根据现有资料无法回答该问题。"
        picked = self._pick_sentences(question, contexts)
        if not picked:
            return "根据现有资料无法回答该问题。"
        body = " ".join(f"{sentence}[{index}]" for index, sentence in picked)
        return f"{body}（Mock 模型输出，用于验证链路；接入真实模型后回答会更自然）"

    def stream(self, messages, *, temperature: float = 0.2, max_tokens: int = 800) -> Iterator[str]:
        text = self.complete(messages, temperature=temperature, max_tokens=max_tokens)
        # 按 6 个字符一片吐，模拟流式 token，方便前端联调打字机效果
        for i in range(0, len(text), 6):
            time.sleep(0.01)
            yield text[i : i + 6]

    def complete_json(self, messages, *, temperature: float = 0.0, max_tokens: int = 600) -> dict:
        joined = "\n".join(m.get("content", "") for m in messages)
        question = _extract_question(joined)
        # 已有观察结果 -> 直接给最终答案（模拟 ReAct 的第二轮）
        if "观察结果" in joined:
            observations = [m["content"].split(":", 1)[-1] for m in messages
                            if m.get("content", "").startswith("观察结果:")]
            last = observations[-1].strip() if observations else ""
            if last:
                blocks = _parse_contexts(last)
                picked = self._pick_sentences(question, blocks) if blocks else []
                if picked:
                    body = " ".join(f"{sentence}[{index}]" for index, sentence in picked)
                    return {
                        "thought": "已有足够的观察结果，可以作答",
                        "final": f"{body}（Mock 模型输出，仅用于验证链路）",
                    }
                return {"thought": "观察结果里没有可用的句子", "final": last[:300]}
        return {
            "thought": "先检索知识库，看有没有相关资料",
            "action": "search_kb",
            "action_input": {"query": question, "top_k": 5},
        }


def _extract_question(text: str) -> str:
    match = re.search(r"问题[:：]\s*(.+)", text)
    if match:
        return match.group(1).strip().splitlines()[0].strip()
    return text.strip().splitlines()[-1][:200] if text.strip() else ""


# ===================================================================== 真实模型
class OpenAICompatLLM:
    """任何 OpenAI 兼容的 /chat/completions：DeepSeek、通义、Kimi、Ollama、vLLM 都能用。"""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 60.0,
    ):
        if not api_key:
            raise LLMError("KB_LLM_API_KEY 为空：把 KB_LLM_PROVIDER 改回 fake，或填上真实 Key")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.name = model

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def _payload(self, messages, temperature, max_tokens) -> dict:
        return {
            "model": self.model,
            "messages": list(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

    def complete(self, messages, *, temperature: float = 0.2, max_tokens: int = 800) -> str:
        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(
                    f"{self.base_url}/chat/completions",
                    headers=self._headers(),
                    json=self._payload(messages, temperature, max_tokens),
                )
        except httpx.HTTPError as exc:
            raise LLMError(f"请求模型失败：{exc}") from exc
        if resp.status_code >= 400:
            raise LLMError(f"模型返回 {resp.status_code}：{resp.text[:300]}")
        try:
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            if not isinstance(content, str):
                raise TypeError("content 必须是字符串")
            return content
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise LLMError("模型响应格式异常") from exc

    def stream(self, messages, *, temperature: float = 0.2, max_tokens: int = 800) -> Iterator[str]:
        try:
            with httpx.Client(timeout=self.timeout) as client:
                with client.stream(
                    "POST",
                    f"{self.base_url}/chat/completions",
                    headers=self._headers(),
                    json={**self._payload(messages, temperature, max_tokens), "stream": True},
                ) as resp:
                    if resp.status_code >= 400:
                        detail = resp.read().decode("utf-8", "replace")
                        raise LLMError(f"模型返回 {resp.status_code}：{detail[:300]}")
                    completed = False
                    for line in resp.iter_lines():
                        if not line or not line.startswith("data:"):
                            continue
                        chunk = line[5:].strip()
                        if chunk == "[DONE]":
                            completed = True
                            break
                        try:
                            payload = json.loads(chunk)
                            choices = payload["choices"]
                            if not choices:  # usage-only 事件
                                continue
                            delta = choices[0]["delta"].get("content")
                            completed = completed or choices[0].get("finish_reason") is not None
                            if delta is not None and not isinstance(delta, str):
                                raise TypeError("delta 必须是字符串")
                        except (ValueError, KeyError, IndexError, TypeError, AttributeError) as exc:
                            raise LLMError("模型流式响应格式异常") from exc
                        if delta:
                            yield delta
                    if not completed:
                        raise LLMError("模型流提前结束，请重试")
        except httpx.HTTPError as exc:
            raise LLMError(f"流式请求失败：{exc}") from exc

    def complete_json(self, messages, *, temperature: float = 0.0, max_tokens: int = 600) -> dict:
        """不依赖 response_format（很多模型不支持），从文本里抠第一段 JSON。"""
        hint = {"role": "user", "content": "只输出一个 JSON 对象，不要 markdown 代码块，不要任何解释。"}
        text = self.complete([*messages, hint], temperature=temperature, max_tokens=max_tokens)
        return _loads_first_json(text)


def _loads_first_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?|```$", "", text, flags=re.S).strip()
    start = text.find("{")
    if start == -1:
        raise LLMError(f"模型没有返回 JSON：{text[:200]}")
    try:
        value, _ = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError as exc:
        raise LLMError("JSON 不完整或格式错误") from exc
    if not isinstance(value, dict):
        raise LLMError("模型决策必须是 JSON 对象")
    return value


def build_llm(settings) -> LLM:
    if settings.llm_provider == "openai":
        return OpenAICompatLLM(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            model=settings.llm_model,
            timeout=settings.llm_timeout,
        )
    if settings.llm_provider == "fake":
        return FakeLLM()
    raise ValueError(f"未知 LLM provider: {settings.llm_provider}")
