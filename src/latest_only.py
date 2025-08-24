from __future__ import annotations
import asyncio
from typing import Dict, Tuple, Callable, Awaitable, List


class LatestOnly:
    """
    Ensure at most one task per user runs at a time, always processing the latest
    scheduled (prefix, suffix). Intermediate states are dropped.
    """

    def __init__(self) -> None:
        self._locks: Dict[str, asyncio.Lock] = {}
        self._latest: Dict[str, Tuple[str, str]] = {}
        self._waiters: Dict[str, List[asyncio.Future]] = {}

    async def run(
        self,
        user_id: str,
        prefix: str,
        suffix: str,
        fn: Callable[[str, str], Awaitable[str]],
    ) -> str:
        loop = asyncio.get_event_loop()
        lock = self._locks.setdefault(user_id, asyncio.Lock())
        self._latest[user_id] = (prefix, suffix)

        # Register a waiter for the result of the next completed run
        fut: asyncio.Future = loop.create_future()
        self._waiters.setdefault(user_id, []).append(fut)

        if lock.locked():
            # Another request is running; wait for its completion to deliver latest result
            return await fut

        async with lock:
            result: str = ""
            while True:
                p, s = self._latest.pop(user_id, (None, None))  # type: ignore[assignment]
                if p is None or s is None:
                    break
                result = await fn(p, s)
                # Resolve all waiters with the produced result
                waiters = self._waiters.pop(user_id, [])
                for w in waiters:
                    if not w.done():
                        w.set_result(result)
                if user_id not in self._latest:
                    break
            # Ensure the waiter created for the starter of this run is resolved
            if not fut.done():
                fut.set_result(result)
            return result


