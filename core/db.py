"""SQLite 数据访问层。

为什么是 SQLite 而不是 MySQL + 向量库？
    1. 适合本地小规模知识库；容量边界必须通过真实语料压测确定；
    2. 零运维：克隆仓库即可运行，不依赖任何外部服务；
    3. 代价是可扩展性，瓶颈与升级路径见 README 的「升级路径」。

对照 Java：这个文件 ≈ JdbcTemplate + schema.sql，没有 ORM（JPA 的魔法在这里是负担）。
"""

from __future__ import annotations

import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    username      TEXT    NOT NULL UNIQUE,
    password_hash TEXT    NOT NULL,
    role          TEXT    NOT NULL DEFAULT 'user',
    created_at    TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS documents (
    id           TEXT    PRIMARY KEY,
    owner_id     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    filename     TEXT    NOT NULL,
    mime         TEXT,
    size_bytes   INTEGER NOT NULL DEFAULT 0,
    content_hash TEXT    NOT NULL,
    status       TEXT    NOT NULL DEFAULT 'pending',   -- pending | indexing | ready | failed
    chunk_count  INTEGER NOT NULL DEFAULT 0,
    error        TEXT,
    created_at   TEXT    NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS ux_documents_owner_hash ON documents(owner_id, content_hash);
CREATE INDEX IF NOT EXISTS ix_documents_owner ON documents(owner_id);

CREATE TABLE IF NOT EXISTS chunks (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id     TEXT    NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    ordinal    INTEGER NOT NULL,
    heading    TEXT,
    page       INTEGER,
    text       TEXT    NOT NULL,
    char_len   INTEGER NOT NULL DEFAULT 0,
    vec        BLOB,                                    -- float32 小端，NULL = 没算向量
    created_at TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_chunks_doc ON chunks(doc_id);

-- 字面检索索引：FTS5 内部就是 BM25。rowid 与 chunks.id 一一对应。
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    idx_text,
    heading_idx,
    tokenize = 'unicode61'
);

CREATE TABLE IF NOT EXISTS sessions (
    id         TEXT    PRIMARY KEY,
    user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title      TEXT    NOT NULL DEFAULT '新会话',
    created_at TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT    NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role       TEXT    NOT NULL,
    content    TEXT    NOT NULL,
    citations  TEXT,
    created_at TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_messages_session ON messages(session_id);

CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
INSERT OR IGNORE INTO metadata VALUES ('corpus_revision', '0');
CREATE TRIGGER IF NOT EXISTS chunks_insert_revision AFTER INSERT ON chunks BEGIN
    UPDATE metadata SET value = CAST(value AS INTEGER) + 1 WHERE key='corpus_revision';
END;
CREATE TRIGGER IF NOT EXISTS chunks_update_revision AFTER UPDATE ON chunks BEGIN
    UPDATE metadata SET value = CAST(value AS INTEGER) + 1 WHERE key='corpus_revision';
END;
CREATE TRIGGER IF NOT EXISTS chunks_delete_cleanup AFTER DELETE ON chunks BEGIN
    DELETE FROM chunks_fts WHERE rowid=OLD.id;
    UPDATE metadata SET value = CAST(value AS INTEGER) + 1 WHERE key='corpus_revision';
END;
"""


def utcnow() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def new_id(prefix: str = "") -> str:
    raw = uuid.uuid4().hex[:16]
    return f"{prefix}{raw}" if prefix else raw


class Database:
    """连接工厂 + 事务边界。SQLite 连接很便宜，每次开一个用完就关，天然线程安全。"""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            self.path,
            timeout=15,
            isolation_level=None,  # 关掉隐式事务，事务边界由我自己控制
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")      # 读写不互相阻塞
        conn.execute("PRAGMA foreign_keys=ON")       # 级联删除生效
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    @contextmanager
    def session(self) -> Iterator[sqlite3.Connection]:
        """显式事务：with db.session() as conn: ...  异常自动回滚。"""
        conn = self.connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            yield conn
            conn.execute("COMMIT")
        except Exception:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise
        finally:
            conn.close()

    def init_schema(self) -> None:
        conn = self.connect()
        try:
            conn.executescript(SCHEMA)
        finally:
            conn.close()

    # ------------------------------------------------------------------ 常用查询
    def create_user(self, username: str, password_hash: str, role: str = "user") -> int:
        with self.session() as conn:
            cur = conn.execute(
                "INSERT INTO users(username, password_hash, role, created_at) VALUES (?,?,?,?)",
                (username, password_hash, role, utcnow()),
            )
            return int(cur.lastrowid)

    def get_user(self, username: str) -> sqlite3.Row | None:
        conn = self.connect()
        try:
            return conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        finally:
            conn.close()

    def get_user_by_id(self, user_id: int) -> sqlite3.Row | None:
        conn = self.connect()
        try:
            return conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        finally:
            conn.close()

    def insert_document(
        self,
        owner_id: int,
        filename: str,
        mime: str,
        size_bytes: int,
        content_hash: str,
    ) -> tuple[str, bool]:
        """返回 (doc_id, created)。同一个人重复上传同一份文件时复用已有记录（幂等）。"""
        with self.session() as conn:
            row = conn.execute(
                "SELECT id, status FROM documents WHERE owner_id=? AND content_hash=?",
                (owner_id, content_hash),
            ).fetchone()
            if row:
                if row["status"] == "failed":
                    conn.execute("UPDATE documents SET status='indexing', error=NULL WHERE id=?", (row["id"],))
                    return row["id"], True
                return row["id"], False
            doc_id = new_id("doc_")
            conn.execute(
                "INSERT INTO documents(id, owner_id, filename, mime, size_bytes, content_hash,"
                " status, created_at) VALUES (?,?,?,?,?,?,'indexing',?)",
                (doc_id, owner_id, filename, mime, size_bytes, content_hash, utcnow()),
            )
        return doc_id, True

    def get_document(self, doc_id: str) -> sqlite3.Row | None:
        conn = self.connect()
        try:
            return conn.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()
        finally:
            conn.close()

    def list_documents(self, owner_id: int) -> list[sqlite3.Row]:
        conn = self.connect()
        try:
            return list(
                conn.execute(
                    "SELECT * FROM documents WHERE owner_id = ? ORDER BY created_at DESC",
                    (owner_id,),
                )
            )
        finally:
            conn.close()

    def delete_document(self, doc_id: str) -> None:
        with self.session() as conn:
            conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))

    def add_chunks(self, doc_id: str, chunks: list[dict]) -> int:
        """一批 chunk 落库 + 写 FTS 索引，一个事务内完成（要么全有，要么全无）。"""
        ts = utcnow()
        with self.session() as conn:
            conn.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
            for ch in chunks:
                cur = conn.execute(
                    "INSERT INTO chunks(doc_id, ordinal, heading, page, text, char_len, vec, created_at)"
                    " VALUES (?,?,?,?,?,?,?,?)",
                    (
                        doc_id,
                        ch["ordinal"],
                        ch.get("heading"),
                        ch.get("page"),
                        ch["text"],
                        len(ch["text"]),
                        ch.get("vec"),
                        ts,
                    ),
                )
                conn.execute(
                    "INSERT INTO chunks_fts(rowid, idx_text, heading_idx) VALUES (?,?,?)",
                    (int(cur.lastrowid), ch["idx_text"], ch.get("heading_idx", "")),
                )
            conn.execute(
                "UPDATE documents SET status='ready', chunk_count=?, error=NULL WHERE id=?",
                (len(chunks), doc_id),
            )
        return len(chunks)

    def update_document_status(self, doc_id: str, status: str, error: str | None = None) -> None:
        with self.session() as conn:
            conn.execute(
                "UPDATE documents SET status=?, error=? WHERE id=?", (status, error, doc_id)
            )

    def ensure_session(self, user_id: int, session_id: str | None, title: str) -> str:
        sid = session_id or new_id("ses_")
        with self.session() as conn:
            row = conn.execute("SELECT id, user_id FROM sessions WHERE id=?", (sid,)).fetchone()
            if row and int(row["user_id"]) != user_id:
                raise PermissionError("会话不存在或无权访问")
            if not row:
                conn.execute(
                    "INSERT INTO sessions(id, user_id, title, created_at) VALUES (?,?,?,?)",
                    (sid, user_id, title[:40], utcnow()),
                )
        return sid

    def assert_session_owner(self, session_id: str, user_id: int) -> None:
        conn = self.connect()
        try:
            row = conn.execute("SELECT user_id FROM sessions WHERE id=?", (session_id,)).fetchone()
            if row is None or int(row["user_id"]) != user_id:
                raise PermissionError("会话不存在或无权访问")
        finally:
            conn.close()

    def list_sessions(self, user_id: int) -> list[dict]:
        conn = self.connect()
        try:
            return [dict(r) for r in conn.execute(
                "SELECT id, title, created_at FROM sessions WHERE user_id=? ORDER BY created_at DESC, id LIMIT 100",
                (user_id,),
            )]
        finally:
            conn.close()

    def save_exchange(self, session_id: str, user_id: int, question: str, answer: str, citations: str) -> None:
        """成功的问答成对提交；失败或断流不会留下半轮历史。"""
        with self.session() as conn:
            row = conn.execute("SELECT user_id FROM sessions WHERE id=?", (session_id,)).fetchone()
            if row is None or int(row["user_id"]) != user_id:
                raise PermissionError("会话不存在或无权访问")
            conn.executemany(
                "INSERT INTO messages(session_id, role, content, citations, created_at) VALUES (?,?,?,?,?)",
                [(session_id, "user", question, None, utcnow()),
                 (session_id, "assistant", answer, citations, utcnow())],
            )

    def ensure_embedding(self, signature: str) -> None:
        """同维度也可能来自不同模型，禁止在旧向量上静默切换模型。"""
        with self.session() as conn:
            row = conn.execute("SELECT value FROM metadata WHERE key='embedding_signature'").fetchone()
            has_vectors = conn.execute("SELECT 1 FROM chunks WHERE vec IS NOT NULL LIMIT 1").fetchone()
            if has_vectors and (row is None or row["value"] != signature):
                raise ValueError("向量模型与已有索引不一致或旧索引缺少签名；请使用新数据库并重新导入文档")
            conn.execute("INSERT OR REPLACE INTO metadata VALUES ('embedding_signature', ?)", (signature,))

    def add_message(
        self, session_id: str, role: str, content: str, citations: str | None = None
    ) -> None:
        with self.session() as conn:
            conn.execute(
                "INSERT INTO messages(session_id, role, content, citations, created_at)"
                " VALUES (?,?,?,?,?)",
                (session_id, role, content, citations, utcnow()),
            )

    def recent_messages(self, session_id: str, limit: int = 6) -> list[sqlite3.Row]:
        conn = self.connect()
        try:
            rows = list(
                conn.execute(
                    "SELECT role, content, citations, created_at FROM messages WHERE session_id=?"
                    " ORDER BY id DESC LIMIT ?",
                    (session_id, limit),
                )
            )
            return list(reversed(rows))
        finally:
            conn.close()

    def list_headings(self, doc_id: str) -> list[str]:
        """某份文档的章节目录（按出现顺序去重）。

        注意这个方法存在的意义：Agent 的 doc_outline 工具需要它，
        而 gen/ 层不允许自己写 SQL（见 scripts/check_arch.py 的规则）。
        把查询收进数据访问层，规则才守得住。
        """
        conn = self.connect()
        try:
            rows = conn.execute(
                "SELECT heading FROM chunks WHERE doc_id=? AND heading IS NOT NULL"
                " ORDER BY ordinal",
                (doc_id,),
            ).fetchall()
        finally:
            conn.close()
        seen: set[str] = set()
        ordered: list[str] = []
        for row in rows:
            heading = row["heading"]
            if heading not in seen:
                seen.add(heading)
                ordered.append(heading)
        return ordered
