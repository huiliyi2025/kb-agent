from dataclasses import replace

import httpx
import jwt
import numpy as np
import pytest

from kb_agent.core.config import Settings
from kb_agent.core.embedding import EmbeddingError, OpenAICompatEmbedder
from kb_agent.core.ratelimit import SlidingWindowLimiter
from kb_agent.core.security import decode_token, verify_password
from kb_agent.gen.agent import Agent, make_tools
from kb_agent.gen.llm import LLMError, OpenAICompatLLM, _loads_first_json
from kb_agent.gen.rag import finalize_answer
from kb_agent.retrieval.dense import search_dense
from kb_agent.schemas import Citation
from conftest import register, upload


@pytest.mark.parametrize("changes", [
    {"chunk_overlap": 400}, {"chunk_size": 0}, {"llm_provider": "typo"},
    {"embed_provider": "typo"}, {"min_answerability": float("nan")},
    {"top_k": 21}, {"environment": "production"},
])
def test_invalid_configuration_fails_at_startup(changes):
    with pytest.raises(ValueError):
        Settings(**changes)


def test_production_settings():
    settings = Settings(environment="production", jwt_secret="a" * 32, admin_password="test-password-123")
    assert settings.environment == "production"
    with pytest.raises(ValueError):
        replace(settings, cors_origins=("*",))


@pytest.mark.parametrize("payload", [{"sub": "1"}, {"sub": "NaN", "iat": 1, "exp": 9999999999},
                                     {"sub": "-1", "iat": 1, "exp": 9999999999}])
def test_malformed_signed_token_is_rejected(payload):
    secret = "x" * 32
    assert decode_token(secret, jwt.encode(payload, secret, algorithm="HS256")) is None


def test_password_hash_corruption_and_limiter_cleanup():
    assert not verify_password("password", "pbkdf2$abc$zz$xx")
    limiter = SlidingWindowLimiter(1)
    assert limiter.allow("a", now=0)[0]
    assert not limiter.allow("a", now=59)[0]
    assert limiter.allow("a", now=60)[0]
    limiter.allow("b", now=120)
    assert "a" not in limiter._hits


@pytest.mark.parametrize("answer", ["", "无引用的结论", "错误编号[2]", "混合编号[1][99]", "根据现有资料无法回答该问题。"])
def test_missing_or_invalid_citations_refused(answer):
    citation = Citation(index=1, doc_id="d", filename="f", chunk_id=1, snippet="来源")
    text, cites, refused = finalize_answer(answer, [citation])
    assert refused and not cites and text


def test_json_parser_handles_quoted_braces():
    assert _loads_first_json('前缀 {"final":"字符串中的 } 和 { 不影响解析"} 后缀')["final"].endswith("解析")


def test_static_module_served_as_javascript(client):
    # Windows 的 mimetypes 不认识 .mjs；若被当成 text/plain，
    # 浏览器会拒载 ES 模块导致前端 boot() 不执行（登录/健康状态全部失效）。
    module_resp = client.get("/static/sse.mjs")
    assert module_resp.status_code == 200
    assert "javascript" in module_resp.headers["content-type"]
    entry_resp = client.get("/static/app.js")
    assert entry_resp.status_code == 200
    assert "javascript" in entry_resp.headers["content-type"]


def mock_client(monkeypatch, module, handler):
    real = httpx.Client
    monkeypatch.setattr(module.httpx, "Client", lambda **kwargs: real(transport=httpx.MockTransport(handler), **kwargs))


@pytest.mark.parametrize("body", [
    {"data": []},
    {"data": [{"index": 0, "embedding": [1, 2]}]},
    {"data": [{"index": 0, "embedding": [0] * 16}]},
    {"data": [{"index": 7, "embedding": [1] * 16}]},
    {"data": [{"index": 0, "embedding": ["nan"] * 16}]},
])
def test_bad_embedding_contract(monkeypatch, body):
    import kb_agent.core.embedding as module
    mock_client(monkeypatch, module, lambda req: httpx.Response(200, json=body))
    embedder = OpenAICompatEmbedder("https://example.invalid/v1", "test", "test", 16)
    with pytest.raises(EmbeddingError):
        embedder.embed(["example"])


def test_embedding_response_reordered_and_normalized(monkeypatch):
    import kb_agent.core.embedding as module
    body = {"data": [{"index": 1, "embedding": [0, 2]}, {"index": 0, "embedding": [3, 0]}]}
    mock_client(monkeypatch, module, lambda req: httpx.Response(200, json=body))
    result = OpenAICompatEmbedder("https://example.invalid/v1", "test", "test", 2).embed(["a", "b"])
    np.testing.assert_allclose(result, [[1, 0], [0, 1]])


def test_truncated_upstream_sse_raises(monkeypatch):
    import kb_agent.gen.llm as module
    mock_client(monkeypatch, module, lambda req: httpx.Response(200, text='data: {"choices":[{"delta":{"content":"partial"}}]}\n\n'))
    llm = OpenAICompatLLM("https://example.invalid/v1", "test", "test")
    with pytest.raises(LLMError, match="提前结束"):
        list(llm.stream([]))


def test_agent_invalid_arguments_and_repeat_budget(client):
    state = client.app.state.ctx
    class Decisions:
        def __init__(self):
            self.items = iter([
                {"action": "search_kb", "action_input": {"query": "标准", "top_k": "bad"}},
                {"action": "search_kb", "action_input": {"query": "标准", "top_k": "bad"}},
                {"action": "search_kb", "action_input": [1, 2]},
            ])
        def complete_json(self, messages):
            return next(self.items)
    result = Agent(Decisions(), make_tools(state.retriever, state.db, []), max_steps=3).run("标准")
    assert len(result.steps) == 3
    assert "参数错误" in result.steps[0].observation
    assert "已经调用" in result.steps[1].observation
    assert "JSON 对象" in result.steps[2].observation


def test_batched_dense_matches_reference_and_scopes(client):
    state = client.app.state.ctx
    owner = state.db.get_user("admin")["id"]
    doc_id, _ = state.db.insert_document(owner, "vectors.md", "text/plain", 1, "vectors")
    rng = np.random.default_rng(42)
    matrix = rng.normal(size=(150, 32)).astype("<f4")
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    state.db.add_chunks(doc_id, [{"ordinal": i, "text": str(i), "idx_text": str(i), "vec": v.tobytes()}
                                 for i, v in enumerate(matrix)])
    query = matrix[15]
    expected = np.argsort(-(matrix @ query))[:12]
    conn = state.db.connect()
    try:
        hits = search_dense(conn, query, limit=12, owner_id=owner, batch_size=7)
        assert [h.ordinal for h in hits] == list(expected)
        assert not search_dense(conn, query, owner_id=owner + 100)
        assert not search_dense(conn, np.zeros(32))
    finally:
        conn.close()


def test_failed_stream_sanitized_and_not_persisted(client):
    headers = register(client, "alice")
    upload(client, headers, "# 标准\n住宿标准是每晚 600 元。")
    state = client.app.state.ctx
    def broken(messages):
        yield "partial"
        raise LLMError("api-key=SECRET")
    state.rag.llm.stream = broken
    response = client.post("/api/chat/stream", headers=headers, json={"question": "住宿标准是多少？"})
    assert "event: error" in response.text and "event: done" not in response.text
    assert "SECRET" not in response.text
    sid = client.get("/api/sessions", headers=headers).json()[0]["id"]
    assert client.get(f"/api/sessions/{sid}/messages", headers=headers).json() == []
