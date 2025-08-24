from __future__ import annotations
import asyncio
from typing import Awaitable, Callable, Dict, Any
import logging


class SingleFlight:
    """
    Collapse identical concurrent requests so work runs only once per key.
    Subsequent awaiters receive the same result or exception.
    """

    def __init__(self) -> None:
        self._futures: Dict[str, asyncio.Future] = {}
        self._logger = logging.getLogger("ntp")

    async def do(self, key: str, coro_factory: Callable[[], Awaitable[Any]]) -> Any:
        fut = self._futures.get(key)
        if fut is not None:
            if self._logger.isEnabledFor(logging.INFO):
                self._logger.info("singleflight join key=%s", key)
            return await fut

        loop = asyncio.get_event_loop()
        fut = loop.create_future()
        self._futures[key] = fut
        if self._logger.isEnabledFor(logging.INFO):
            self._logger.info("singleflight leader key=%s", key)

        try:
            result = await coro_factory()
            fut.set_result(result)
            return result
        except Exception as e:  # pragma: no cover - propagate exceptions too
            fut.set_exception(e)
            raise
        finally:
            self._futures.pop(key, None)


