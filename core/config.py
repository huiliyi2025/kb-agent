"""全局配置：只从环境变量读（12-Factor），代码里不出现任何硬编码密钥。

和 Spring Boot 的对照：
    @Value("${kb.llm.model}")  ->  settings.llm_model
    application.yml            ->  .env
    @ConfigurationProperties   ->  这个 dataclass
"""

from __future__ import annotations

import os
import math
from dataclasses import dataclass
from pathlib import Path

# 当前目录本身就是发行项目根，不向父目录写数据库或读取 .env。
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(os.environ.get("KB_PROJECT_ROOT", str(
    PACKAGE_ROOT if (PACKAGE_ROOT / "pyproject.toml").exists() else Path.cwd()
))).resolve()


def load_env_file(path: Path | None = None) -> None:
    """极简 .env 解析（不引入 python-dotenv，少一个依赖少一个坑）。

    已存在的环境变量优先（os.environ.setdefault），这样 Docker / CI 里注入的值不会被 .env 覆盖。
    """
    env_path = path or PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError as exc:
        raise ValueError(f"{name} 必须是整数") from exc


def _float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError as exc:
        raise ValueError(f"{name} 必须是数字") from exc


@dataclass(frozen=True)
class Settings:
    # 服务
    app_name: str = "kb-agent"
    environment: str = "development"
    cors_origins: tuple[str, ...] = ()
    db_path: str = "data/kb.db"
    jwt_secret: str = "dev-only-change-me"
    jwt_ttl_seconds: int = 86400
    admin_user: str = "admin"
    admin_password: str = "admin123"

    # LLM
    llm_provider: str = "fake"          # fake | openai
    llm_base_url: str = "https://api.deepseek.com/v1"
    llm_api_key: str = ""
    llm_model: str = "deepseek-chat"
    llm_timeout: int = 60

    # Embedding
    embed_provider: str = "hashing"     # hashing | openai | none
    embed_base_url: str = "https://api.siliconflow.cn/v1"
    embed_api_key: str = ""
    embed_model: str = "BAAI/bge-m3"
    embed_dim: int = 1024

    # 检索
    chunk_size: int = 420
    chunk_overlap: int = 60
    top_k: int = 5
    rrf_k: int = 60
    min_answerability: float = 0.55

    # 保护
    max_upload_mb: int = 20
    rate_limit_per_min: int = 30

    def __post_init__(self) -> None:
        if self.environment not in {"development", "production", "test"}:
            raise ValueError("KB_ENV 必须是 development / production / test")
        if self.llm_provider not in {"fake", "openai"}:
            raise ValueError("未知 KB_LLM_PROVIDER")
        if self.embed_provider not in {"hashing", "openai", "none"}:
            raise ValueError("未知 KB_EMBED_PROVIDER")
        for name in ("jwt_ttl_seconds", "llm_timeout", "max_upload_mb", "rate_limit_per_min", "rrf_k"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} 必须大于 0")
        if not 50 <= self.chunk_size <= 10000 or not 0 <= self.chunk_overlap <= self.chunk_size // 2:
            raise ValueError("chunk_size 必须为 50~10000，overlap 必须为 0~chunk_size/2")
        if not 1 <= self.top_k <= 20:
            raise ValueError("top_k 必须为 1~20")
        if not math.isfinite(self.min_answerability) or not 0 <= self.min_answerability <= 1:
            raise ValueError("min_answerability 必须为 0~1")
        if self.embed_provider != "none" and self.embed_dim <= 8:
            raise ValueError("embed_dim 必须大于 8")
        if self.environment == "production":
            if self.jwt_secret == "dev-only-change-me" or len(self.jwt_secret) < 32:
                raise ValueError("生产环境必须设置至少 32 字符的 KB_JWT_SECRET")
            if self.admin_password == "admin123" or len(self.admin_password) < 12:
                raise ValueError("生产环境必须设置至少 12 字符的 KB_ADMIN_PASSWORD")
            if "*" in self.cors_origins:
                raise ValueError("生产环境不允许通配 CORS 来源")

    @classmethod
    def from_env(cls) -> "Settings":
        load_env_file()
        return cls(
            environment=_str("KB_ENV", cls.environment).lower(),
            cors_origins=tuple(x.strip() for x in _str("KB_CORS_ORIGINS", "").split(",") if x.strip()),
            db_path=_str("KB_DB_PATH", cls.db_path),
            jwt_secret=_str("KB_JWT_SECRET", cls.jwt_secret),
            jwt_ttl_seconds=_int("KB_JWT_TTL_SECONDS", cls.jwt_ttl_seconds),
            admin_user=_str("KB_ADMIN_USER", cls.admin_user),
            admin_password=_str("KB_ADMIN_PASSWORD", cls.admin_password),
            llm_provider=_str("KB_LLM_PROVIDER", cls.llm_provider).lower(),
            llm_base_url=_str("KB_LLM_BASE_URL", cls.llm_base_url),
            llm_api_key=_str("KB_LLM_API_KEY", cls.llm_api_key),
            llm_model=_str("KB_LLM_MODEL", cls.llm_model),
            llm_timeout=_int("KB_LLM_TIMEOUT", cls.llm_timeout),
            embed_provider=_str("KB_EMBED_PROVIDER", cls.embed_provider).lower(),
            embed_base_url=_str("KB_EMBED_BASE_URL", cls.embed_base_url),
            embed_api_key=_str("KB_EMBED_API_KEY", cls.embed_api_key),
            embed_model=_str("KB_EMBED_MODEL", cls.embed_model),
            embed_dim=_int("KB_EMBED_DIM", cls.embed_dim),
            chunk_size=_int("KB_CHUNK_SIZE", cls.chunk_size),
            chunk_overlap=_int("KB_CHUNK_OVERLAP", cls.chunk_overlap),
            top_k=_int("KB_TOP_K", cls.top_k),
            rrf_k=_int("KB_RRF_K", cls.rrf_k),
            min_answerability=_float("KB_MIN_ANSWERABILITY", cls.min_answerability),
            max_upload_mb=_int("KB_MAX_UPLOAD_MB", cls.max_upload_mb),
            rate_limit_per_min=_int("KB_RATE_LIMIT_PER_MIN", cls.rate_limit_per_min),
        )

    @property
    def db_abspath(self) -> Path:
        """相对路径一律相对项目根目录，避免"在哪个目录启动"影响行为。"""
        p = Path(self.db_path)
        return p if p.is_absolute() else PROJECT_ROOT / p
