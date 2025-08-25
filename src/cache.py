from __future__ import annotations
import time
try:
    from cachetools import TTLCache  # type: ignore
    suggest_cache = TTLCache(maxsize=512, ttl=20)
except Exception:
    class _SimpleTTL:
        def __init__(self, ttl: int = 20):
            self.ttl = ttl
            self._d = {}
        def __setitem__(self, k, v):
            self._d[k] = (v, time.time() + self.ttl)
        def get(self, k, default=None):
            v = self._d.get(k, None)
            if not v: return default
            val, exp = v
            if time.time() > exp:
                self._d.pop(k, None)
                return default
            return val
        def __contains__(self, k):
            return self.get(k) is not None
    suggest_cache = _SimpleTTL(20)


