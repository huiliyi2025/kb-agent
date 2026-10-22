"""滑动窗口限流（单进程内存版）。

生产环境换成 Redis + Lua 或网关限流；这里的价值是让你能回答：
"LLM 接口是按 token 收费的，你怎么防止一个用户把额度刷爆？"
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque


class SlidingWindowLimiter:
    """滑动窗口：每个 key 记录最近 60 秒内的调用时间戳。

    相比固定窗口（每分钟清零），滑动窗口不会在窗口边界被"双倍突发"打穿。
    """

    def __init__(self, limit_per_minute: int, window_seconds: float = 60.0):
        self.limit = max(1, limit_per_minute)
        self.window = window_seconds
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()
        self._next_cleanup = 0.0

    def allow(self, key: str, now: float | None = None) -> tuple[bool, int]:
        """返回 (是否放行, 剩余可用次数)。"""
        now = time.monotonic() if now is None else now
        with self._lock:
            if now >= self._next_cleanup:
                for old_key in list(self._hits):
                    hits = self._hits[old_key]
                    if not hits or now - hits[-1] >= self.window:
                        del self._hits[old_key]
                self._next_cleanup = now + self.window
            q = self._hits[key]
            while q and now - q[0] >= self.window:
                q.popleft()
            if len(q) >= self.limit:
                return False, 0
            q.append(now)
            return True, self.limit - len(q)

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()
