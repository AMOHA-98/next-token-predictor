from __future__ import annotations
import re
from enum import Enum
from .utils import generate_random_string

UNIQUE_CURSOR = generate_random_string(16)
HEADER_REGEX = rf"^#+\s.*{UNIQUE_CURSOR}.*$"
UNORDERED_LIST_REGEX = rf"^\s*(-|\*)\s.*{UNIQUE_CURSOR}.*$"
TASK_LIST_REGEX = rf"^\s*(-|[0-9]+\.) +\[.\]\s.*{UNIQUE_CURSOR}.*$"
BLOCK_QUOTES_REGEX = rf"^\s*>.*{UNIQUE_CURSOR}.*$"
NUMBERED_LIST_REGEX = rf"^\s*\d+\.\s.*{UNIQUE_CURSOR}.*$"
MATH_BLOCK_REGEX = re.compile(r"\$\$[\s\S]*?\$\$", re.M)
INLINE_MATH_BLOCK_REGEX = re.compile(r"\$[\s\S]*?\$", re.M)
CODE_BLOCK_REGEX = re.compile(r"```[\s\S]*?```", re.M)
INLINE_CODE_BLOCK_REGEX = re.compile(r"`.*`", re.M)


class Context(str, Enum):
    Text = "Text"
    Heading = "Heading"
    BlockQuotes = "BlockQuotes"
    UnorderedList = "UnorderedList"
    NumberedList = "NumberedList"
    CodeBlock = "CodeBlock"
    MathBlock = "MathBlock"
    TaskList = "TaskList"


def _is_cursor_in_regex_block(prefix: str, suffix: str, pattern: re.Pattern) -> bool:
    text = prefix + UNIQUE_CURSOR + suffix
    blocks = pattern.findall(text)
    for b in blocks:
        if UNIQUE_CURSOR in (b if isinstance(b, str) else "".join(b)):
            return True
    return False


def get_context(prefix: str, suffix: str) -> Context:
    text = prefix + UNIQUE_CURSOR + suffix
    if re.search(HEADER_REGEX, text, re.M):
        return Context.Heading
    if re.search(BLOCK_QUOTES_REGEX, text, re.M):
        return Context.BlockQuotes
    if re.search(TASK_LIST_REGEX, text, re.M):
        return Context.TaskList
    if _is_cursor_in_regex_block(prefix, suffix, MATH_BLOCK_REGEX) or _is_cursor_in_regex_block(prefix, suffix, INLINE_MATH_BLOCK_REGEX):
        return Context.MathBlock
    if _is_cursor_in_regex_block(prefix, suffix, CODE_BLOCK_REGEX) or _is_cursor_in_regex_block(prefix, suffix, INLINE_CODE_BLOCK_REGEX):
        return Context.CodeBlock
    if re.search(NUMBERED_LIST_REGEX, text, re.M):
        return Context.NumberedList
    if re.search(UNORDERED_LIST_REGEX, text, re.M):
        return Context.UnorderedList
    return Context.Text


