"""组装根（Composition Root）+ 依赖注入。

所有对象的创建都集中在这里，路由函数只声明"我要什么"，不关心"怎么造"。
对照 Spring：这里同时扮演 @Configuration + BeanFactory 的角色。
区别是——Python 不用注解魔法，你一眼就能看到整个对象图。
"""

from __future__ import annotations

from dataclasses import dataclass
import sqlite3

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from kb_agent.core.config import Settings
from kb_agent.core.db import Database
from kb_agent.core.ratelimit import SlidingWindowLimiter
from kb_agent.core.security import Principal, create_token, decode_token, hash_password
from kb_agent.gen.llm import LLM, build_llm
from kb_agent.gen.rag import RagService
from kb_agent.core.embedding import Embedder, build_embedder
from kb_agent.retrieval.service import Retriever

_bearer = HTTPBearer(auto_error=False)


@dataclass
class AppState:
    settings: Settings
    db: Database
    embedder: Embedder
    llm: LLM
    retriever: Retriever
    rag: RagService
    limiter: SlidingWindowLimiter


def build_state(settings: Settings) -> AppState:
    db = Database(settings.db_abspath)
    db.init_schema()

    embedder = build_embedder(
        settings.embed_provider,
        base_url=settings.embed_base_url,
        api_key=settings.embed_api_key,
        model=settings.embed_model,
        dim=settings.embed_dim,
    )
    if embedder.dim > 0:
        signature = f"{embedder.name}:{embedder.dim}:{settings.embed_base_url if settings.embed_provider == 'openai' else ''}"
        db.ensure_embedding(signature)
    llm = build_llm(settings)
    retriever = Retriever(db, embedder, rrf_k=settings.rrf_k)
    rag = RagService(db, retriever, llm, settings)
    limiter = SlidingWindowLimiter(settings.rate_limit_per_min)

    _seed_admin(db, settings)
    return AppState(
        settings=settings,
        db=db,
        embedder=embedder,
        llm=llm,
        retriever=retriever,
        rag=rag,
        limiter=limiter,
    )


def _seed_admin(db: Database, settings: Settings) -> None:
    """首次启动建一个管理员账号，方便 CLI / 前端开箱即用。生产环境请改成初始化脚本 + 强制改密。"""
    if db.get_user(settings.admin_user) is None:
        try:
            db.create_user(settings.admin_user, hash_password(settings.admin_password), role="admin")
        except sqlite3.IntegrityError:
            if db.get_user(settings.admin_user) is None:
                raise


def get_state(request: Request) -> AppState:
    return request.app.state.ctx


def get_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> Principal:
    state: AppState = request.app.state.ctx
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="缺少 Bearer Token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    principal = decode_token(state.settings.jwt_secret, credentials.credentials)
    if principal is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token 无效或已过期",
            headers={"WWW-Authenticate": "Bearer"},
        )
    row = state.db.get_user_by_id(principal.user_id)
    if row is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "用户已失效")
    return Principal(user_id=int(row["id"]), username=row["username"], role=row["role"])


def auth_rate_limit(request: Request) -> None:
    state: AppState = request.app.state.ctx
    host = request.client.host if request.client else "unknown"
    allowed, _ = state.limiter.allow(f"auth:{host}")
    if not allowed:
        raise HTTPException(429, "认证请求过于频繁，请稍后再试", headers={"Retry-After": "60"})


def rate_limit(bucket: str):
    """依赖工厂：给某类接口套上限流。"""

    def dependency(
        request: Request, user: Principal = Depends(get_current_user)
    ) -> int:
        state: AppState = request.app.state.ctx
        allowed, remaining = state.limiter.allow(f"{bucket}:{user.user_id}")
        if not allowed:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"请求过于频繁（上限 {state.settings.rate_limit_per_min} 次/分钟），请稍后再试",
                headers={"Retry-After": "60"},
            )
        return remaining

    return dependency


def issue_token(state: AppState, username: str, role: str, user_id: int) -> dict:
    principal = Principal(user_id=user_id, username=username, role=role)
    return {
        "access_token": create_token(state.settings.jwt_secret, principal, state.settings.jwt_ttl_seconds),
        "expires_in": state.settings.jwt_ttl_seconds,
        "username": username,
        "role": role,
    }
