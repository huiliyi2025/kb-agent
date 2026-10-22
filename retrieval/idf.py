"""字级 IDF 模型——用来给"这个问题能不能答"打分。

为什么需要它？
    直接用"查询里的字有多少出现在文档里"太宽松：
    "公司股票代码是多少"里的 公司/代 在语料里到处都是，覆盖率能到 0.5，
    于是一个知识库里根本没有的问题被误判成"能答"。

IDF 的直觉（和 BM25 同源）：
    越罕见的字，携带的信息越多。
    "股""票"在语料里从没出现过 → 权重最高 → 没被覆盖就是强负信号；
    "公""司"到处都有 → 权重接近 0 → 覆不覆盖都无所谓。

代价是每次查询要扫一遍 chunk 文本算 df，所以我们把结果缓存起来，
用数据库修订号失效，等长替换也能识别；缓存按用户隔离并限制容量。
"""

from __future__ import annotations

import math
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Mapping

from kb_agent.core.db import Database
from kb_agent.core.text import content_chars


@dataclass(frozen=True)
class CharIdf:
    weights: Mapping[str, float]
    default: float          # 语料里从没出现过的字的权重
    n_chunks: int
    version: int

    def weight_of(self, chars: set[str]) -> float:
        return sum(self.weights.get(ch, self.default) for ch in chars)


_cache: OrderedDict[tuple[str, int | None], CharIdf] = OrderedDict()
_lock = threading.Lock()


def _version(conn) -> int:
    return int(conn.execute("SELECT value FROM metadata WHERE key='corpus_revision'").fetchone()["value"])


def get_char_idf(db: Database, owner_id: int | None = None) -> CharIdf:
    conn = db.connect()
    try:
        conn.execute("BEGIN")
        version = _version(conn)
        key = (str(db.path.resolve()), owner_id)
        with _lock:
            cached = _cache.get(key)
        if cached is not None and cached.version == version:
            return cached

        sql = "SELECT c.text FROM chunks c JOIN documents d ON d.id=c.doc_id WHERE d.status='ready'"
        rows = conn.execute(sql + (" AND d.owner_id=?" if owner_id is not None else ""),
                            (owner_id,) if owner_id is not None else ()).fetchall()
    finally:
        conn.close()

    n_chunks = max(1, len(rows))
    df: dict[str, int] = {}
    for row in rows:
        for ch in content_chars(row["text"]):
            df[ch] = df.get(ch, 0) + 1

    # 加 1 平滑，避免除零；未见过的字按 df=0.5 处理，权重略高于任何出现过的字
    weights = {ch: math.log(1.0 + n_chunks / (count + 0.5)) for ch, count in df.items()}
    default = math.log(1.0 + n_chunks / 0.5)
    idf = CharIdf(weights=weights, default=default, n_chunks=n_chunks, version=version)
    with _lock:
        _cache[key] = idf
        _cache.move_to_end(key)
        while len(_cache) > 128:
            _cache.popitem(last=False)
    return idf


def clear_cache() -> None:
    """测试用：清掉进程内缓存。"""
    with _lock:
        _cache.clear()
