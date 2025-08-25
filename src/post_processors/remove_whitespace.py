from __future__ import annotations
from ..context_detection import Context


class RemoveWhitespace:
    def process(self, prefix: str, suffix: str, completion: str, context: Context) -> str:
        if context in {
            Context.Text, Context.Heading, Context.MathBlock, Context.TaskList,
            Context.NumberedList, Context.UnorderedList,
        }:
            # If user already typed a space or a newline boundary, don't start with another
            if prefix.endswith((" ", "\t", "\n")) or suffix.startswith("\n"):
                completion = completion.lstrip()

            # If the next visible char is punctuation, trim any trailing space we added
            if suffix[:1] in {".", ",", ";", ":", "!", "?", ")", "]", "}", "»", "”"}:
                completion = completion.rstrip()

        return completion


