"""认证与密码哈希。只依赖标准库 + PyJWT。

对照 Java：相当于 Spring Security 的 PasswordEncoder + JwtTokenProvider，
但这里不到 60 行——认证逻辑短小、没有框架黑箱，出问题容易定位。
"""

from __future__ import annotations

import hashlib
import hmac
import os
import time
from dataclasses import dataclass

import jwt

_PBKDF2_ROUNDS = 120_000


def hash_password(password: str, *, salt: bytes | None = None) -> str:
    """PBKDF2-HMAC-SHA256，返回 `pbkdf2$轮数$salt_hex$hash_hex`。

    为什么不用加盐 MD5：GPU 一秒能撞几十亿次。为什么自己拼字符串而不装 bcrypt：
    减少依赖，同时 salt 与迭代次数的作用在代码里一目了然。
    """
    salt = salt or os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, _PBKDF2_ROUNDS)
    return f"pbkdf2${_PBKDF2_ROUNDS}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, rounds, salt_hex, hash_hex = stored.split("$")
        if scheme != "pbkdf2" or not 1 <= int(rounds) <= 2_000_000:
            return False
        digest = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), int(rounds)
        )
    except (ValueError, TypeError):
        return False
    # 用 compare_digest 防时序攻击；== 会因为提前返回而泄露"前几位对了"
    return hmac.compare_digest(digest.hex(), hash_hex)


@dataclass(frozen=True)
class Principal:
    user_id: int
    username: str
    role: str


def create_token(secret: str, principal: Principal, ttl_seconds: int) -> str:
    now = int(time.time())
    payload = {
        "sub": str(principal.user_id),
        "username": principal.username,
        "role": principal.role,
        "iat": now,
        "exp": now + ttl_seconds,
    }
    return jwt.encode(payload, secret, algorithm="HS256")


def decode_token(secret: str, token: str) -> Principal | None:
    """解不出来就返回 None，让上层决定 401，绝不抛出 500。"""
    try:
        payload = jwt.decode(token, secret, algorithms=["HS256"], options={"require": ["sub", "exp", "iat"]})
        user_id = int(payload["sub"])
        if user_id <= 0:
            return None
    except (jwt.PyJWTError, ValueError, TypeError, KeyError):
        return None
    return Principal(
        user_id=user_id,
        username=payload.get("username", ""),
        role=payload.get("role", "user"),
    )
