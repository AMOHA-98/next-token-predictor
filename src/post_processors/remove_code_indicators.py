from __future__ import annotations
import re
from ..context_detection import Context


class RemoveCodeIndicators:
    def process(self, prefix: str, suffix: str, completion: str, context: Context) -> str:
        if context == Context.CodeBlock:
            completion = re.sub(r"```[a-zA-Z]+[ \t]*\n?", "", completion)
            completion = re.sub(r"\n?```[ \t]*\n?", "", completion)
            completion = completion.replace("`", "")
        return completion


