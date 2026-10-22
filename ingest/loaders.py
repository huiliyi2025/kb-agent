"""文档解析：把各种格式变成 (标题, 页码, 正文) 的段落流。

参考项目 Langchain-Chatchat 用 Unstructured + 一堆 loader 库，装完 2GB 依赖、还经常因为
某个 C 扩展版本对不上装不上。原则是：**能用标准库就别引依赖**。
    .md/.txt  -> 标准库
    .docx     -> 标准库（docx 本质是 zip + XML，解压取 w:t 文本即可）
    .pdf      -> pypdf（纯 Python，唯一必须的第三方解析器）
"""

from __future__ import annotations

import io
import re
import zipfile
from defusedxml import ElementTree as ET
from defusedxml.common import DefusedXmlException
from dataclasses import dataclass

from kb_agent.core.text import normalize


@dataclass
class Section:
    """解析后的最小单元。chunk 从这里切出来，保留来源信息用于引用。"""

    text: str
    heading: str | None = None
    page: int | None = None


class UnsupportedFormat(Exception):
    pass


def load_document(filename: str, data: bytes) -> list[Section]:
    suffix = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if suffix in (".md", ".markdown", ".txt"):
        return _load_text(data)
    if suffix == ".docx":
        return _load_docx(data)
    if suffix == ".pdf":
        return _load_pdf(data)
    raise UnsupportedFormat(f"暂不支持 {suffix or '未知'} 格式，请上传 md/txt/docx/pdf")


def _decode(data: bytes) -> str:
    for enc in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


def _load_text(data: bytes) -> list[Section]:
    """按 Markdown 标题切段，标题会一路带进 chunk 的元数据里（引用时很有用）。"""
    text = normalize(_decode(data))
    sections: list[Section] = []
    heading: str | None = None
    buf: list[str] = []

    def flush() -> None:
        body = "\n".join(buf).strip()
        if body:
            sections.append(Section(text=body, heading=heading))
        buf.clear()

    for line in text.split("\n"):
        match = _HEADING_RE.match(line)
        if match:
            flush()
            heading = match.group(2).strip()
            continue
        buf.append(line)
    flush()
    return sections or [Section(text=text)]


def _load_docx(data: bytes) -> list[Section]:
    """解析段落、标题和表格行；限制 XML 解压大小，禁止 DTD/实体展开。"""
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            info = zf.getinfo("word/document.xml")
            if info.file_size > 20 * 1024 * 1024:
                raise UnsupportedFormat("docx 正文解压后超过 20MB 限制")
            root = ET.fromstring(zf.read(info), forbid_dtd=True)
    except (zipfile.BadZipFile, KeyError, ET.ParseError, DefusedXmlException, RuntimeError) as exc:
        raise UnsupportedFormat("docx 结构异常或包含不支持的 XML") from exc
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    w = "{" + ns["w"] + "}"
    body = root.find("w:body", ns)
    if body is None:
        return []
    sections: list[Section] = []
    heading = None
    def text_of(node):
        return "".join((e.text or "") if e.tag == w + "t" else "\n" if e.tag == w + "br" else "\t"
                       for e in node.iter() if e.tag in {w + "t", w + "br", w + "tab"}).strip()
    for node in body:
        if node.tag == w + "p":
            text = text_of(node)
            style = node.find("w:pPr/w:pStyle", ns)
            style_name = style.get(w + "val", "") if style is not None else ""
            if re.match(r"^(heading|标题)\s*\d", style_name, flags=re.I):
                heading = text or heading
            elif text:
                sections.append(Section(text=normalize(text), heading=heading))
        elif node.tag == w + "tbl":
            rows = [" | ".join(text_of(cell) for cell in row.findall("w:tc", ns))
                    for row in node.findall("w:tr", ns)]
            text = normalize("\n".join(rows))
            if text:
                sections.append(Section(text=text, heading=heading))
    return sections


def _load_pdf(data: bytes) -> list[Section]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover
        raise UnsupportedFormat("需要安装 pypdf 才能解析 PDF：pip install pypdf") from exc

    try:
        reader = PdfReader(io.BytesIO(data))
    except Exception as exc:  # pypdf 的异常类型不稳定，统一兜住
        raise UnsupportedFormat(f"PDF 解析失败：{exc}") from exc

    sections: list[Section] = []
    for page_no, page in enumerate(reader.pages, start=1):
        try:
            text = normalize(page.extract_text() or "")
        except Exception:
            text = ""
        if text:
            sections.append(Section(text=text, page=page_no))
    return sections
