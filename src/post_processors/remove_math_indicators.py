from __future__ import annotations
import re
from ..context_detection import Context


class RemoveMathIndicators:
    def process(self, prefix: str, suffix: str, completion: str, context: Context) -> str:
        if context == Context.MathBlock:
            completion = re.sub(r"\n?\$\$\n?", "", completion)
            completion = completion.replace("$", "")
        return completion


