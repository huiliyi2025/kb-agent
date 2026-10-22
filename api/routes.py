"""HTTP 路由（对应 Spring 的 @RestController）。

原则：路由层只做三件事——校验入参、调用领域服务、把结果转换成响应。
业务逻辑一行都不写在这里，否则三个月后你会有一个 800 行的 controller。
"""

from __future__ import annotations

import json
import logging
import sqlite3
from typing import Iterator

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from fastapi.responses import StreamingResponse
from starlette.concurrency import run_in_threadpool

from kb_agent.api.deps import AppState, auth_rate_limit, get_current_user, get_state, issue_token, rate_limit
from kb_agent.core.security import Principal, hash_password, verify_password
from kb_agent.gen.agent import Agent, make_tools
from kb_agent.ingest.indexer import ingest_document
from kb_agent.ingest.loaders import UnsupportedFormat
from kb_agent.schemas import (
    AgentRequest,
    AgentResponse,
    AskRequest,
    AskResponse,
    DocumentOut,
    HealthResponse,
    LoginRequest,
    RegisterRequest,
    TokenResponse,
)

router = APIRouter(prefix="/api")
logger = logging.getLogger("kb_agent")


# ===================================================================== 认证
@router.post("/auth/register", response_model=TokenResponse, tags=["auth"], dependencies=[Depends(auth_rate_limit)])
def register(payload: RegisterRequest, state: AppState = Depends(get_state)) -> TokenResponse:
    if state.db.get_user(payload.username) is not None:
        raise HTTPException(status.HTTP_409_CONFLICT, "用户名已被占用")
    try:
        user_id = state.db.create_user(payload.username, hash_password(payload.password))
    except sqlite3.IntegrityError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, "用户名已被占用") from exc
    data = issue_token(state, payload.username, "user", user_id)
    return TokenResponse(**data)


@router.post("/auth/login", response_model=TokenResponse, tags=["auth"], dependencies=[Depends(auth_rate_limit)])
def login(payload: LoginRequest, state: AppState = Depends(get_state)) -> TokenResponse:
    row = state.db.get_user(payload.username)
    if row is None or not verify_password(payload.password, row["password_hash"]):
        # 故意不区分"用户不存在"和"密码错误"，避免撞库时枚举账号
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "用户名或密码错误")
    data = issue_token(state, row["username"], row["role"], int(row["id"]))
    return TokenResponse(**data)


@router.get("/me", tags=["auth"])
def me(user: Principal = Depends(get_current_user)) -> dict:
    return {"user_id": user.user_id, "username": user.username, "role": user.role}


# ===================================================================== 文档
def _to_document_out(row) -> DocumentOut:
    return DocumentOut(
        id=row["id"],
        filename=row["filename"],
        status=row["status"],
        chunk_count=int(row["chunk_count"]),
        size_bytes=int(row["size_bytes"]),
        created_at=row["created_at"],
        error=row["error"],
    )


@router.post("/documents", response_model=DocumentOut, tags=["documents"], dependencies=[Depends(rate_limit("upload"))])
async def upload_document(
    file: UploadFile = File(...),
    state: AppState = Depends(get_state),
    user: Principal = Depends(get_current_user),
) -> DocumentOut:
    limit = state.settings.max_upload_mb * 1024 * 1024
    try:
        data = await file.read(limit + 1)
    finally:
        await file.close()
    if len(data) > limit:
        raise HTTPException(
            413,
            f"文件超过 {state.settings.max_upload_mb}MB 限制",
        )
    if not data:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "空文件")

    try:
        doc_id, count, reused = await run_in_threadpool(
            ingest_document,
            state.db,
            state.settings,
            state.embedder,
            owner_id=user.user_id,
            filename=file.filename or "unnamed",
            mime=file.content_type,
            data=data,
        )
    except UnsupportedFormat as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc

    row = state.db.get_document(doc_id)
    if row is None:
        raise HTTPException(409, "文档已被删除，请重新上传")
    out = _to_document_out(row)
    out.reused = reused
    return out


@router.get("/documents", response_model=list[DocumentOut], tags=["documents"])
def list_documents(
    state: AppState = Depends(get_state), user: Principal = Depends(get_current_user)
) -> list[DocumentOut]:
    return [_to_document_out(row) for row in state.db.list_documents(user.user_id)]


@router.get("/documents/{doc_id}/chunks", tags=["documents"])
def list_chunks(
    doc_id: str,
    limit: int = 20,
    state: AppState = Depends(get_state),
    user: Principal = Depends(get_current_user),
) -> dict:
    """调参利器：分块效果不好时，先看这里切成了什么样，再改 chunk_size。"""
    doc = state.db.get_document(doc_id)
    if doc is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "文档不存在")
    if int(doc["owner_id"]) != user.user_id:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "无权访问该文档")
    conn = state.db.connect()
    try:
        rows = conn.execute(
            "SELECT ordinal, heading, page, text, char_len FROM chunks"
            " WHERE doc_id=? ORDER BY ordinal LIMIT ?",
            (doc_id, max(1, min(limit, 200))),
        ).fetchall()
    finally:
        conn.close()
    return {"doc_id": doc_id, "total": int(doc["chunk_count"]), "chunks": [dict(r) for r in rows]}


@router.delete("/documents/{doc_id}", tags=["documents"])
def delete_document(
    doc_id: str,
    state: AppState = Depends(get_state),
    user: Principal = Depends(get_current_user),
) -> dict:
    doc = state.db.get_document(doc_id)
    if doc is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "文档不存在")
    if int(doc["owner_id"]) != user.user_id:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "无权删除该文档")
    state.db.delete_document(doc_id)
    return {"deleted": doc_id}


# ===================================================================== 问答
@router.post(
    "/chat",
    response_model=AskResponse,
    tags=["chat"],
    dependencies=[Depends(rate_limit("chat"))],
)
def chat(
    payload: AskRequest,
    state: AppState = Depends(get_state),
    user: Principal = Depends(get_current_user),
) -> AskResponse:
    session_id = state.db.ensure_session(user.user_id, payload.session_id, payload.question)
    return state.rag.ask(
        payload.question,
        top_k=payload.top_k,
        doc_ids=payload.doc_ids,
        session_id=session_id,
        user_id=user.user_id,
    )


def _sse(event: dict) -> str:
    return f"event: {event['event']}\ndata: {json.dumps(event['data'], ensure_ascii=False)}\n\n"


@router.post(
    "/chat/stream",
    tags=["chat"],
    dependencies=[Depends(rate_limit("chat"))],
)
def chat_stream(
    payload: AskRequest,
    state: AppState = Depends(get_state),
    user: Principal = Depends(get_current_user),
) -> StreamingResponse:
    session_id = state.db.ensure_session(user.user_id, payload.session_id, payload.question)

    def generator() -> Iterator[str]:
        try:
            for event in state.rag.stream(
                payload.question,
                top_k=payload.top_k,
                doc_ids=payload.doc_ids,
                session_id=session_id,
                user_id=user.user_id,
            ):
                yield _sse(event)
        except Exception:  # 流开始后只能发 error 事件，不能修改 HTTP 状态码。
            logger.exception("流式问答失败")
            yield _sse({"event": "error", "data": {"message": "回答生成失败，请稍后重试"}})

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ===================================================================== Agent
@router.post(
    "/agent",
    response_model=AgentResponse,
    tags=["agent"],
    dependencies=[Depends(rate_limit("agent"))],
)
def run_agent(
    payload: AgentRequest,
    state: AppState = Depends(get_state),
    user: Principal = Depends(get_current_user),
) -> AgentResponse:
    citations: list = []
    tools = make_tools(state.retriever, state.db, citations, owner_id=user.user_id)
    agent = Agent(state.llm, tools, max_steps=payload.max_steps)
    return agent.run(payload.question, citations=citations)


# ===================================================================== 运维
@router.get("/sessions", tags=["chat"])
def list_sessions(state: AppState = Depends(get_state), user: Principal = Depends(get_current_user)) -> list[dict]:
    return state.db.list_sessions(user.user_id)


@router.get("/sessions/{session_id}/messages", tags=["chat"])
def session_messages(session_id: str, state: AppState = Depends(get_state),
                     user: Principal = Depends(get_current_user)) -> list[dict]:
    state.db.assert_session_owner(session_id, user.user_id)
    return [dict(row) for row in state.db.recent_messages(session_id, limit=100)]


@router.get("/health", response_model=HealthResponse, tags=["ops"])
def health(state: AppState = Depends(get_state)) -> HealthResponse:
    conn = state.db.connect()
    try:
        documents = conn.execute("SELECT COUNT(*) AS c FROM documents").fetchone()["c"]
        chunks = conn.execute("SELECT COUNT(*) AS c FROM chunks").fetchone()["c"]
    finally:
        conn.close()
    return HealthResponse(
        llm=state.llm.name,
        embedder=state.embedder.name,
        documents=int(documents),
        chunks=int(chunks),
    )
