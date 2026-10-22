"""分块（Chunking）——RAG 里性价比最高、也最容易被做错的一步。

三个必须自己回答的问题：
    1. 多大？太大 → 噪声多、贵、稀释相似度；太小 → 语义断裂、答非所问。默认 420 字符。
    2. 重叠？必须重叠，否则一句话被腰斩，两边都不完整。默认 60 字符（约 15%）。
    3. 在哪切？**在句号处切**，不要硬切在第 420 个字符，否则一半的 chunk 以半句话开头。

对照参考项目：Langchain-Chatchat 把分块交给 RecursiveCharacterTextSplitter，
参数藏在配置文件里、中文标点还要自己补。这里 60 行写死逻辑，好处是"每一刀在哪、为什么"
都写在代码里。
"""

from __future__ import annotations

from dataclasses import dataclass

from kb_agent.core.text import normalize
from kb_agent.ingest.loaders import Section

# 句末符号：中英文都算。换行也算，因为列表项本身就是语义单元。
_BREAK_CHARS = "。！？!?；;…\n"
# 回溯窗口：只在最后 40% 里找句号，避免为了凑句子把 chunk 切得太短
_MIN_FILL_RATIO = 0.6


@dataclass
class Chunk:
    ordinal: int
    text: str
    heading: str | None
    page: int | None
    embed_text: str  # 用于向量化/FTS：标题 + 正文，检索时标题能提供额外信号


def split_text(text: str, chunk_size: int = 420, overlap: int = 60) -> list[str]:
    text = normalize(text)
    if not text:
        return []
    if chunk_size < 50:
        raise ValueError("chunk_size 太小，语义会被切碎")
    overlap = max(0, min(overlap, chunk_size // 2))

    pieces: list[str] = []
    start, n = 0, len(text)
    while start < n:
        end = min(start + chunk_size, n)
        if end < n:
            floor = start + int(chunk_size * _MIN_FILL_RATIO)
            cut = -1
            for i in range(end, floor, -1):
                if text[i - 1] in _BREAK_CHARS:
                    cut = i
                    break
            if cut > start:
                end = cut
        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)
        if end >= n:
            break
        # 下一段回退 overlap 个字符；start 必须严格前进，否则死循环
        start = max(end - overlap, start + 1)
    return pieces


def chunk_sections(
    sections: list[Section], chunk_size: int = 420, overlap: int = 60
) -> list[Chunk]:
    chunks: list[Chunk] = []
    ordinal = 0
    for section in sections:
        for piece in split_text(section.text, chunk_size, overlap):
            heading = section.heading
            embed_text = f"{heading}\n{piece}" if heading else piece
            chunks.append(
                Chunk(
                    ordinal=ordinal,
                    text=piece,
                    heading=heading,
                    page=section.page,
                    embed_text=embed_text,
                )
            )
            ordinal += 1
    return chunks
