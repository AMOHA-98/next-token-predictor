from __future__ import annotations
from pydantic import BaseModel


class PrefixAndSuffix(BaseModel):
    prefix: str
    suffix: str


