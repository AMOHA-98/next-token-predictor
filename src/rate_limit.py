from __future__ import annotations
from typing import Dict

try:
    from aiolimiter import AsyncLimiter  # type: ignore
except Exception:  # soft fallback until dependency is installed
    class AsyncLimiter:  # type: ignore
        def __init__(self, rate: int, time_period: int) -> None:
            self.rate = rate
            self.time_period = time_period

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False


user_limiters: Dict[str, AsyncLimiter] = {}


def limiter_for(user_id: str) -> AsyncLimiter:
    """Return a per-user AsyncLimiter (3 req/s with burst of 3)."""
    lim = user_limiters.get(user_id)
    if lim is None:
        lim = AsyncLimiter(3, 1)
        user_limiters[user_id] = lim
    return lim


