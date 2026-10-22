"""向量化（Embedding）实现。

模型可替换是这个设计的第一目标。
答案就在这里——所有实现都藏在同一个 Protocol 后面，换模型只改一行配置。

三个实现：
    HashingEmbedder  —— 离线、零依赖、确定性；demo 和单测用它（质量一般但链路完整）
    OpenAICompatEmbedder —— 任何兼容 /v1/embeddings 的服务（bge-m3 / text-embedding-3 / Ollama）
    NullEmbedder     —— 关掉向量通道，只留 BM25

为什么它不在 retrieval/ 里？
    ingest 层要"存向量"，所以它依赖 Embedder 这个抽象。如果 Embedder 定义在 retrieval/，
    就成了"接入层依赖检索层"——方向反了（scripts/check_arch.py 会直接报错）。
    结论：**被下层依赖的抽象，必须放在依赖链的最底部**。这就是依赖倒置在真实项目里的样子。
"""

from __future__ import annotations

import hashlib
from typing import Protocol, Sequence

import httpx
import numpy as np

from kb_agent.core.text import tokenize


class Embedder(Protocol):
    dim: int
    name: str

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """返回 (n, dim) 的 float32 矩阵，每行已 L2 归一化。"""
        ...


class EmbeddingError(RuntimeError):
    """向量服务侧故障（网络不可达、4xx/5xx）。

    单独定义异常类型，是为了让接口层能精确映射成 502，
    而不是把上游故障和本地 bug 一起塞进 500。
    """


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (mat / norms).astype(np.float32)


class HashingEmbedder:
    """带符号的哈希向量（hashing trick / random projection 的简化版）。

    原理：token -> 哈希 -> 落到 dim 个桶之一，符号由另一个 bit 决定。
    好处是完全无状态、无模型、可复现；坏处是没有语义（"报销"和"报账"不相似）。
    所以它只适合做 pipeline 的冒烟测试，不代表线上检索质量。
    """

    def __init__(self, dim: int = 256):
        if dim <= 8:
            raise ValueError("dim 太小，哈希冲突会吃掉所有信号")
        self.dim = dim
        self.name = f"hashing-{dim}"

    def _bucket_sign(self, token: str) -> tuple[int, float]:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        return value % self.dim, (1.0 if (value >> 63) & 1 else -1.0)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in tokenize(text):
                idx, sign = self._bucket_sign(token)
                out[row, idx] += sign
        return _l2_normalize(out)


class OpenAICompatEmbedder:
    """任何 OpenAI 兼容的 /embeddings 接口。bge-m3 一次能传 32 条左右，这里按 16 条分批。"""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        dim: int,
        timeout: float = 30.0,
        batch_size: int = 16,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.dim = dim
        self.name = f"openai-compat:{model}"
        self.timeout = timeout
        self.batch_size = batch_size

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vectors: list[list[float]] = []
        try:
            with httpx.Client(timeout=self.timeout) as client:
                for i in range(0, len(texts), self.batch_size):
                    batch = list(texts[i : i + self.batch_size])
                    resp = client.post(
                        f"{self.base_url}/embeddings",
                        headers={"Authorization": f"Bearer {self.api_key}"},
                        json={"model": self.model, "input": batch},
                    )
                    if resp.status_code >= 400:
                        raise EmbeddingError(
                            f"向量服务返回 {resp.status_code}：{resp.text[:200]}"
                        )
                    data = resp.json()["data"]
                    if not isinstance(data, list) or len(data) != len(batch):
                        raise EmbeddingError("向量服务返回的数量与输入不一致")
                    indices = [d["index"] for d in data]
                    if sorted(indices) != list(range(len(batch))):
                        raise EmbeddingError("向量服务返回重复或缺失的 index")
                    data.sort(key=lambda d: d["index"])
                    vectors.extend(item["embedding"] for item in data)
        except httpx.HTTPError as exc:
            # 网络层错误必须转成领域异常：否则上传接口会吐 500，
            # 而 500 在监控里意味着"我的服务有 bug"，实际是上游挂了（应该是 502）。
            raise EmbeddingError(f"向量服务不可达：{exc}") from exc
        except (ValueError, KeyError, TypeError) as exc:
            raise EmbeddingError("向量服务响应格式异常") from exc

        try:
            mat = np.asarray(vectors, dtype=np.float32)
        except (ValueError, TypeError) as exc:
            raise EmbeddingError("向量服务返回非数值向量") from exc
        if mat.shape != (len(texts), self.dim) or not np.isfinite(mat).all():
            # 维度不一致是线上最常见的事故：换了模型没改 KB_EMBED_DIM，检索会静默变差
            raise EmbeddingError(
                f"embedding 数量、维度或数值不合法：形状 {mat.shape}，预期 ({len(texts)}, {self.dim})。"
                "请改 KB_EMBED_DIM 并重建索引。"
            )
        if np.any(np.linalg.norm(mat, axis=1) == 0):
            raise EmbeddingError("向量服务返回零向量")
        return _l2_normalize(mat)


class NullEmbedder:
    """关闭向量通道。dim=0 是一个哨兵值，检索层看到它就不走向量召回。"""

    dim = 0
    name = "none"

    def embed(self, texts: Sequence[str]) -> np.ndarray:  # pragma: no cover - 不应被调用
        return np.zeros((len(texts), 0), dtype=np.float32)


def build_embedder(
    provider: str, *, base_url: str = "", api_key: str = "", model: str = "", dim: int = 1024
) -> Embedder:
    if provider == "openai":
        return OpenAICompatEmbedder(base_url=base_url, api_key=api_key, model=model, dim=dim)
    if provider == "none":
        return NullEmbedder()
    if provider == "hashing":
        return HashingEmbedder(dim=dim)
    raise ValueError(f"未知 embedding provider: {provider}")
