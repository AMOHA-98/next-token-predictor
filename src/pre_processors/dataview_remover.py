from __future__ import annotations
import re
from pydantic import BaseModel
from ..context_detection import UNIQUE_CURSOR
from .types_preproc import PrefixAndSuffix

DATA_VIEW_REGEX = re.compile(r"```dataview(js){0,1}(.|\n)*?```", re.M)


class _Dummy(BaseModel):
    pass


class DataViewRemover:
    def process(self, prefix: str, suffix: str, context) -> PrefixAndSuffix:
        text = prefix + UNIQUE_CURSOR + suffix
        text = DATA_VIEW_REGEX.sub("", text)
        prefix_new, suffix_new = text.split(UNIQUE_CURSOR)
        return PrefixAndSuffix(prefix=prefix_new, suffix=suffix_new)

    def removes_cursor(self, prefix: str, suffix: str) -> bool:
        text = prefix + UNIQUE_CURSOR + suffix
        matches = DATA_VIEW_REGEX.findall(text)
        if not matches:
            return False
        for m in DATA_VIEW_REGEX.finditer(text):
            if UNIQUE_CURSOR in text[m.start():m.end()]:
                return True
        return False


