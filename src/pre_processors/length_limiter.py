from __future__ import annotations
from pydantic import BaseModel
from .types_preproc import PrefixAndSuffix


class _Dummy(BaseModel):
    pass


class LengthLimiter:
    def __init__(self, max_prefix: int, max_suffix: int):
        self.max_prefix = max_prefix
        self.max_suffix = max_suffix

    def process(self, prefix: str, suffix: str, context) -> PrefixAndSuffix:
        return PrefixAndSuffix(prefix=prefix[-self.max_prefix:], suffix=suffix[:self.max_suffix])

    def removes_cursor(self, prefix: str, suffix: str) -> bool:
        return False


