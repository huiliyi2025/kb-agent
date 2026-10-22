import asyncio
import threading

import httpx
import pytest

from kb_agent.ingest.indexer import build_rows
from kb_agent.ingest.loaders import Section


def test_failed_replacement_rolls_back_text_and_fts(client):
    state = client.app.state.ctx
    doc, _ = state.db.insert_document(state.db.get_user("admin")["id"], "a.md", "text/plain", 1, "a")
    good = build_rows([Section("原始内容不能丢失。")], state.embedder, 100, 10)
    state.db.add_chunks(doc, good)
    bad = [dict(good[0]), {"ordinal": 1, "text": "破损记录缺少 idx_text"}]
    with pytest.raises(KeyError):
        state.db.add_chunks(doc, bad)
    conn = state.db.connect()
    try:
        assert conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0] == 1
        assert conn.execute("SELECT text FROM chunks").fetchone()[0] == "原始内容不能丢失。"
    finally:
        conn.close()


def test_upload_indexing_does_not_block_event_loop(client, monkeypatch):
    import kb_agent.api.routes as routes
    original = routes.ingest_document
    started, release = threading.Event(), threading.Event()
    def slow(*args, **kwargs):
        started.set()
        if not release.wait(timeout=3):
            raise RuntimeError("事件循环被阻塞，无法释放索引任务")
        return original(*args, **kwargs)
    monkeypatch.setattr(routes, "ingest_document", slow)
    async def scenario():
        transport = httpx.ASGITransport(app=client.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as session:
            login = await session.post("/api/auth/login", json={"username": "admin", "password": "admin123"})
            headers = {"Authorization": "Bearer " + login.json()["access_token"]}
            task = asyncio.create_task(session.post("/api/documents", headers=headers,
                                                    files={"file": ("a.txt", "异步上传测试。".encode())}))
            try:
                assert await asyncio.to_thread(started.wait, 2)
                response = await asyncio.wait_for(session.get("/api/health"), timeout=1)
                assert response.status_code == 200
            finally:
                release.set()
                result = await task
            assert result.status_code == 200
    asyncio.run(scenario())
