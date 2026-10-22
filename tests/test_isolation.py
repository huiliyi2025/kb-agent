import json

import pytest

from kb_agent.gen.agent import make_tools
from conftest import register, upload


@pytest.mark.parametrize("endpoint", ["/api/chat", "/api/chat/stream", "/api/agent"])
def test_no_cross_user_retrieval(client, endpoint):
    alice = register(client, "alice")
    bob = register(client, "bobby")
    doc = upload(client, alice, "# 机密\n猎鹰项目密码是 ALICE-SECRET-9527。")
    response = client.post(endpoint, headers=bob, json={"question": "猎鹰项目密码是什么？"})
    assert response.status_code == 200
    assert "ALICE-SECRET-9527" not in response.text
    assert doc["id"] not in response.text
    own = client.post(endpoint, headers=alice, json={"question": "猎鹰项目密码是什么？"})
    assert own.status_code == 200
    assert "ALICE-SECRET-9527" in own.text


@pytest.mark.parametrize("endpoint", ["/api/chat", "/api/chat/stream"])
def test_foreign_session_denied_before_stream(client, endpoint):
    alice, bob = register(client, "alice"), register(client, "bobby")
    sid = client.post("/api/chat", headers=alice, json={"question": "测试"}).json()["session_id"]
    response = client.post(endpoint, headers=bob, json={"question": "秘密", "session_id": sid})
    assert response.status_code == 403
    assert client.get(f"/api/sessions/{sid}/messages", headers=bob).status_code == 403
    assert len(client.get(f"/api/sessions/{sid}/messages", headers=alice).json()) == 2


def test_document_scope_and_empty_filter(client):
    alice, bob = register(client, "alice"), register(client, "bobby")
    doc = upload(client, alice, "# 工资\n月工资标准是 98765 元。")
    for doc_ids in ([doc["id"]], []):
        result = client.post("/api/chat", headers=bob, json={"question": "月工资标准是多少", "doc_ids": doc_ids}).json()
        assert result["refused"] and result["retrieved"] == []
    result = client.post("/api/chat", headers=alice, json={"question": "月工资标准是多少", "doc_ids": []}).json()
    assert result["retrieved"] == []
    assert client.get(f"/api/documents/{doc['id']}/chunks", headers=bob).status_code == 403
    assert client.delete(f"/api/documents/{doc['id']}", headers=bob).status_code == 403
    state = client.app.state.ctx
    tools = {t.name: t for t in make_tools(state.retriever, state.db, [], owner_id=state.db.get_user("bobby")["id"])}
    assert "无权" in tools["doc_outline"].fn(doc["id"])


def test_history_no_duplicate_question_and_atomic_failure(client):
    alice = register(client, "alice")
    upload(client, alice, "# 住宿\n住宿标准是每晚 600 元。")
    state = client.app.state.ctx
    observed = []
    original = state.rag.llm.complete
    def capture(messages, **kwargs):
        observed.append(messages)
        return original(messages, **kwargs)
    state.rag.llm.complete = capture
    question = "住宿标准是多少？"
    first = client.post("/api/chat", headers=alice, json={"question": question}).json()
    sid = first["session_id"]
    assert sum(question in m["content"] for m in observed[0]) == 1
    second = client.post("/api/chat", headers=alice, json={"question": question, "session_id": sid})
    assert second.status_code == 200
    messages = client.get(f"/api/sessions/{sid}/messages", headers=alice).json()
    assert [m["role"] for m in messages] == ["user", "assistant", "user", "assistant"]
    from kb_agent.gen.llm import LLMError
    def fail(*args, **kwargs):
        raise LLMError("private upstream details")
    state.rag.llm.complete = fail
    response = client.post("/api/chat", headers=alice, json={"question": question, "session_id": sid})
    assert response.status_code == 502
    assert "private upstream" not in response.text
    assert len(client.get(f"/api/sessions/{sid}/messages", headers=alice).json()) == 4


def test_sse_done_and_sync_match(client):
    alice = register(client, "alice")
    upload(client, alice, "# 住宿\n住宿标准是每晚 600 元。")
    body = {"question": "住宿标准是多少？"}
    sync = client.post("/api/chat", headers=alice, json=body).json()
    stream = client.post("/api/chat/stream", headers=alice, json=body)
    events = [json.loads(x.split("data: ", 1)[1]) for x in stream.text.strip().split("\n\n") if x.startswith("event: done")]
    assert len(events) == 1
    for key in ("answer", "citations", "refused", "confidence"):
        assert events[0][key] == sync[key]


def test_upload_limits_and_bad_format(client):
    alice = register(client, "alice")
    assert client.post("/api/documents", headers=alice, files={"file": ("x.txt", b"")}).status_code == 400
    assert client.post("/api/documents", headers=alice, files={"file": ("x.exe", b"x")}).status_code == 400
    from dataclasses import replace
    client.app.state.ctx.settings = replace(client.app.state.ctx.settings, max_upload_mb=1)
    response = client.post("/api/documents", headers=alice, files={"file": ("x.txt", b"x" * (1024 * 1024 + 1))})
    assert response.status_code == 413


def test_auth_and_validation(client):
    assert client.get("/api/documents").status_code == 401
    alice = register(client, "alice")
    assert client.post("/api/chat", headers=alice, json={"question": "   "}).status_code == 422
    assert client.post("/api/agent", headers=alice, json={"question": "\n"}).status_code == 422
    state = client.app.state.ctx
    with state.db.session() as conn:
        conn.execute("DELETE FROM users WHERE username='alice'")
    assert client.get("/api/me", headers=alice).status_code == 401
