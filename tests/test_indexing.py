from concurrent.futures import ThreadPoolExecutor
import io
import zipfile

import numpy as np
import pytest

from kb_agent.core.embedding import EmbeddingError
from kb_agent.ingest.indexer import ingest_document
from kb_agent.ingest.loaders import UnsupportedFormat, load_document
from kb_agent.retrieval.idf import get_char_idf
from conftest import register, upload


def test_concurrent_deduplication(client):
    state = client.app.state.ctx
    owner = state.db.get_user("admin")["id"]
    def insert(_):
        return state.db.insert_document(owner, "a.md", "text/plain", 5, "same-hash")
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(insert, range(16)))
    assert len({doc_id for doc_id, _ in results}) == 1
    assert sum(created for _, created in results) == 1


def test_failed_embedding_can_retry_and_no_partial_chunks(client):
    state = client.app.state.ctx
    owner = state.db.get_user("admin")["id"]
    class Broken:
        dim = 32
        name = "broken"
        def embed(self, texts):
            return np.zeros((0, self.dim))
    kwargs = dict(owner_id=owner, filename="retry.md", mime="text/markdown", data="# 标准\n住宿每晚 600 元。".encode())
    with pytest.raises(EmbeddingError):
        ingest_document(state.db, state.settings, Broken(), **kwargs)
    doc = state.db.list_documents(owner)[0]
    assert doc["status"] == "failed" and doc["chunk_count"] == 0
    doc_id, count, reused = ingest_document(state.db, state.settings, state.embedder, **kwargs)
    assert doc_id == doc["id"] and count == 1 and not reused
    assert ingest_document(state.db, state.settings, state.embedder, **kwargs)[2]


def test_fts_cascade_and_revision_invalidation(client):
    headers = register(client, "alice")
    doc = upload(client, headers, "# 条款\n甲乙丙丁戊己庚辛壬癸。")
    state = client.app.state.ctx
    owner = state.db.get_user("alice")["id"]
    old = get_char_idf(state.db, owner)
    with state.db.session() as conn:
        conn.execute("UPDATE chunks SET text='天地玄黄宇宙洪荒日月。' WHERE doc_id=?", (doc["id"],))
    new = get_char_idf(state.db, owner)
    assert new.version != old.version
    assert "天" in new.weights and "甲" not in new.weights
    with state.db.session() as conn:
        conn.execute("DELETE FROM users WHERE id=?", (owner,))
        assert conn.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0] == 0


def test_embedding_signature_rejects_same_dim_other_model(client):
    headers = register(client, "alice")
    upload(client, headers, "测试文档正文")
    with pytest.raises(ValueError, match="向量模型"):
        client.app.state.ctx.db.ensure_embedding("different-model:32")


def docx(xml):
    data = io.BytesIO()
    with zipfile.ZipFile(data, "w") as archive:
        archive.writestr("word/document.xml", xml)
    return data.getvalue()


def test_docx_headings_tables_and_entities():
    xml = '''<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>
    <w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr><w:r><w:t>费用</w:t></w:r></w:p>
    <w:p><w:r><w:t>A &amp; B &#x4E2D;</w:t></w:r></w:p>
    <w:tbl><w:tr><w:tc><w:p><w:r><w:t>住宿</w:t></w:r></w:p></w:tc>
    <w:tc><w:p><w:r><w:t>600</w:t></w:r></w:p></w:tc></w:tr></w:tbl></w:body></w:document>'''
    sections = load_document("a.docx", docx(xml))
    assert sections[0].heading == "费用"
    assert sections[0].text == "A & B 中"
    assert "住宿 | 600" in sections[1].text
    with pytest.raises(UnsupportedFormat):
        load_document("a.docx", docx('<!DOCTYPE x [<!ENTITY e "bad">]><x>&e;</x>'))


def test_bom_heading():
    sections = load_document("a.md", "\ufeff# 标题\n正文内容。".encode())
    assert sections[0].heading == "标题"
