"""FastAPI 应用工厂。

为什么用工厂函数而不是模块级 app = FastAPI()？
    因为测试要能给每个用例一份干净的 settings + 独立的数据库文件。
    模块级单例会让"跑测试时污染开发库"这种事故变成家常便饭。

启动：
    uvicorn kb_agent.api.app:create_app --factory --port 8000
"""

from __future__ import annotations

import logging
import mimetypes
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from kb_agent import __version__
from kb_agent.api.deps import build_state
from kb_agent.api.routes import router
from kb_agent.core.config import Settings
from kb_agent.core.embedding import EmbeddingError
from kb_agent.gen.llm import LLMError

logger = logging.getLogger("kb_agent")
STATIC_DIR = Path(__file__).resolve().parent / "static"

# Windows / 部分发行版的 mimetypes 表不认识 .mjs，会误当 text/plain。
# 浏览器对 <script type="module"> 强制要求 JS MIME，否则整个前端模块不执行。
mimetypes.add_type("text/javascript", ".mjs")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()

    app = FastAPI(
        title="kb-agent",
        version=__version__,
        description="企业知识库 RAG + Agent 问答服务",
    )
    app.state.ctx = build_state(settings)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(LLMError)
    async def _llm_error(_: Request, exc: LLMError) -> JSONResponse:
        # 模型侧故障不能算成 500（那会污染监控），统一 502 + 可读信息
        logger.warning("LLM 调用失败: %s", exc)
        return JSONResponse(status_code=502, content={"detail": "模型服务暂不可用，请稍后重试"})

    @app.exception_handler(EmbeddingError)
    async def _embedding_error(_: Request, exc: EmbeddingError) -> JSONResponse:
        # 向量服务同理：上传/检索时上游挂了是 502，不是我这边有 bug
        logger.warning("向量服务调用失败: %s", exc)
        return JSONResponse(status_code=502, content={"detail": "向量服务暂不可用，请稍后重试"})

    @app.exception_handler(PermissionError)
    async def _permission_error(_: Request, exc: PermissionError) -> JSONResponse:
        return JSONResponse(status_code=403, content={"detail": "资源不存在或无权访问"})

    @app.middleware("http")
    async def request_metadata(request: Request, call_next):
        started = time.perf_counter()
        request_id = uuid.uuid4().hex
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        logger.info("request_id=%s method=%s path=%s status=%s headers_ms=%.1f",
                    request_id, request.method, request.url.path, response.status_code,
                    (time.perf_counter() - started) * 1000)
        return response

    app.include_router(router)

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

        @app.get("/", include_in_schema=False)
        def index() -> FileResponse:
            return FileResponse(STATIC_DIR / "index.html")

    return app
