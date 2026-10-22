"""中文/英文混合文本的分词（检索系统的地基）。

一个残酷的事实：FTS5 / 大多数搜索引擎的 unicode61 分词器把"报销标准"当成一个词，
所以搜"报销"是搜不到的。中文没有空格，必须自己切。

这里用最经典也最省事的方案：CJK 连续片段切成 **bigram（二元组）**，英文数字按词切。
    索引侧：报销标准 -> 报销 | 销标 | 标准
    查询侧：同上
bigram 的好处是召回率接近字级、准确率接近词级，而且不需要维护词典；
缺点是索引会膨胀约 2 倍——这是明确的工程取舍。
"""

from __future__ import annotations

import re

_CJK_RANGES = ((0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF))

_WS_RE = re.compile(r"[ \t\u3000]+")


def is_cjk(ch: str) -> bool:
    code = ord(ch)
    return any(lo <= code <= hi for lo, hi in _CJK_RANGES)


def normalize(text: str) -> str:
    """统一换行、压缩空格、全角字母数字转半角。

    这里刻意**不用** unicodedata.normalize("NFKC")：NFKC 会把中文全角标点
    "，；？（）" 一起转成 ASCII，导致分块时找不到句末标点、引用片段看起来像外文。
    只做我们真正需要的映射：全角数字/字母/空格 → 半角。
    """
    text = (text or "").translate(_FULLWIDTH_MAP)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _WS_RE.sub(" ", text)
    # 干掉 3 个以上的连续空行，保留段落
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _build_fullwidth_map() -> dict[int, str]:
    mapping: dict[int, str] = {0x3000: " "}  # 全角空格
    for code in range(0xFF10, 0xFF1A):  # ０-９
        mapping[code] = chr(code - 0xFF10 + ord("0"))
    for code in range(0xFF21, 0xFF3B):  # Ａ-Ｚ
        mapping[code] = chr(code - 0xFF21 + ord("A"))
    for code in range(0xFF41, 0xFF5B):  # ａ-ｚ
        mapping[code] = chr(code - 0xFF41 + ord("a"))
    return mapping


_FULLWIDTH_MAP = _build_fullwidth_map()


def tokenize(text: str) -> list[str]:
    """切成检索用的 token。写索引和写查询必须调用同一个函数，否则永远对不上。"""
    text = (text or "").lower()
    tokens: list[str] = []
    cjk_run: list[str] = []
    word_run: list[str] = []

    def flush() -> None:
        if cjk_run:
            run = "".join(cjk_run)
            if len(run) == 1:
                tokens.append(run)
            else:
                tokens.extend(run[i : i + 2] for i in range(len(run) - 1))
            cjk_run.clear()
        if word_run:
            tokens.append("".join(word_run))
            word_run.clear()

    for ch in text:
        if is_cjk(ch):
            if word_run:
                flush()
            cjk_run.append(ch)
        elif ch.isalnum():
            if cjk_run:
                flush()
            word_run.append(ch)
        else:
            flush()
    flush()
    return tokens


def to_index_text(text: str) -> str:
    """存进 FTS5 的形态：token 之间用空格隔开，交给 unicode61 当"词"处理。"""
    return " ".join(tokenize(text))


def match_expr(query: str, max_terms: int = 24) -> str:
    """把用户输入变成 FTS5 的 MATCH 表达式。

    用 OR 而不是 AND：只要命中一部分词就能召回，排序交给 BM25。
    全部包成双引号，防止用户输入 `"` `*` `-` 之类的语法字符把查询搞崩。
    """
    terms = list(dict.fromkeys(tokenize(query)))[:max_terms]
    escaped = [t.replace('"', '""') for t in terms if t]
    return " OR ".join(f'"{t}"' for t in escaped)


def content_chars(text: str) -> set[str]:
    """取出"承载信息"的字符：去掉标点、空白，以及高频虚词。

    为什么要去虚词：问句里的"多少""怎么""是否"几乎在任何文档里都能命中，
    它们会让"可回答性"分数虚高，导致该拒答的没拒。
    """
    return {ch for ch in (text or "").lower() if ch.isalnum() and ch not in _STOP_CHARS}


# 汉语里出现频率极高、几乎不携带领域信息的字。只去掉明确的虚词，宁缺毋滥。
_STOP_CHARS = set("的了吗呢吧啊呀嘛是怎什多么几个如何否和与及或在有为对从到把被之其也就还都很更最再将已经并且但而所以如果因为可以可能应该需要")



def truncate(text: str, limit: int = 160) -> str:
    text = (text or "").strip().replace("\n", " ")
    return text if len(text) <= limit else text[: limit - 1] + "…"
