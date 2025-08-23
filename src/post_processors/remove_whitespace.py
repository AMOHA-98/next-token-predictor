from __future__ import annotations
from ..context_detection import Context


class RemoveWhitespace:
    def process(self, prefix: str, suffix: str, completion: str, context: Context) -> str:
        if context in {
            Context.Text,
            Context.Heading,
            Context.MathBlock,
            Context.TaskList,
            Context.NumberedList,
            Context.UnorderedList,
        }:
            if prefix.endswith(" ") or suffix.endswith("\n"):
                completion = completion.lstrip()
            if suffix.startswith(" "):
                completion = completion.rstrip()
        return completion


