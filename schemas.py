"""接口层的数据契约（Pydantic）。

对照 Java：这就是 DTO + Bean Validation。
FastAPI 会自动把 BaseModel 变成：请求体校验 + OpenAPI 文档 + 响应序列化。
好处是"接口即文档"，坏处是模型一多就很啰嗦——所以只在这里放对外暴露的结构。
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


# ----------------------------------------------------------------- 认证
class RegisterRequest(BaseModel):
    username: str = Field(min_length=3, max_length=32)
    password: str = Field(min_length=6, max_length=64)


class LoginRequest(BaseModel):
    username: str = Field(min_length=3, max_length=32)
    password: str = Field(min_length=1, max_length=64)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    username: str
    role: str


# ----------------------------------------------------------------- 文档
class DocumentOut(BaseModel):
    id: str
    filename: str
    status: str
    chunk_count: int
    size_bytes: int
    created_at: str
    error: str | None = None
    reused: bool = False


# ----------------------------------------------------------------- 问答
class Citation(BaseModel):
    index: int                  # 回答里的 [1] 对应这里
    doc_id: str
    filename: str
    heading: str | None = None
    page: int | None = None
    chunk_id: int
    snippet: str


class RetrievedChunk(BaseModel):
    chunk_id: int
    doc_id: str
    filename: str
    heading: str | None = None
    score: float
    source: str
    components: dict[str, float] = Field(default_factory=dict)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)
    session_id: str | None = Field(default=None, min_length=1, max_length=64)
    doc_ids: list[str] | None = Field(default=None, max_length=100)
    top_k: int | None = Field(default=None, ge=1, le=20)

    @field_validator("question")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("问题不能为空")
        return value.strip()


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    refused: bool = False
    confidence: float = 0.0
    latency_ms: int = 0
    session_id: str | None = None
    model: str = ""
    retrieved: list[RetrievedChunk] = Field(default_factory=list)


# ----------------------------------------------------------------- Agent
class AgentStep(BaseModel):
    step: int
    thought: str
    action: str
    action_input: dict = Field(default_factory=dict)
    observation: str = ""


class AgentRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)
    max_steps: int = Field(default=4, ge=1, le=8)

    @field_validator("question")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("问题不能为空")
        return value.strip()


class AgentResponse(BaseModel):
    answer: str
    steps: list[AgentStep] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    elapsed_ms: int = 0


# ----------------------------------------------------------------- 其他
class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    llm: str
    embedder: str
    documents: int
    chunks: int
