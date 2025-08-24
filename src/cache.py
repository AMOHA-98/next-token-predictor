from __future__ import annotations
try:
    from cachetools import TTLCache  # type: ignore
except Exception:  # soft import; user will install later
    TTLCache = dict  # type: ignore

# Small short-TTL cache for suggestions. If cachetools isn't installed yet,
# a plain dict acts as a no-TTL stub; it's fine for development until deps are added.
suggest_cache = TTLCache(maxsize=512, ttl=20) if hasattr(TTLCache, "__call__") else TTLCache()


