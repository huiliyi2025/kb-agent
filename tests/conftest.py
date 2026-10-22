import pytest
from fastapi.testclient import TestClient

from kb_agent.api.app import create_app
from kb_agent.core.config import Settings


@pytest.fixture
def client(tmp_path):
    settings = Settings(db_path=str(tmp_path / "kb.db"), environment="test",
                        jwt_secret="test-secret-32-characters-long-1234", embed_dim=32)
    with TestClient(create_app(settings)) as client:
        yield client


def register(client, name):
    response = client.post("/api/auth/register", json={"username": name, "password": "test-pass-123"})
    assert response.status_code == 200, response.text
    return {"Authorization": "Bearer " + response.json()["access_token"]}


def upload(client, headers, text, filename="policy.md"):
    response = client.post("/api/documents", headers=headers,
                           files={"file": (filename, text.encode(), "text/markdown")})
    assert response.status_code == 200, response.text
    return response.json()
