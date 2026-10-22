"""两种召回共用的数据结构。放一个文件里，是为了避免两个模块互相 import 造成循环依赖。"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Hit:
    chunk_id: int
    doc_id: str
    filename: str
    text: str
    heading: str | None = None
    page: int | None = None
    ordinal: int = 0
    score: float = 0.0          # 融合后的最终分（0~1）
    source: str = ""            # lexical | dense | hybrid
    components: dict[str, float] = field(default_factory=dict)

    def snippet(self, limit: int = 160) -> str:
        from kb_agent.core.text import truncate

        return truncate(self.text, limit)
