from __future__ import annotations


def _is_ws(ch: str | None) -> bool:
    return ch is not None and (ch.isspace())


def _start_locations(text: str) -> list[int]:
    locs: list[int] = []
    if text and not _is_ws(text[0]):
        locs.append(0)
    for i in range(1, len(text)):
        if _is_ws(text[i - 1]) and not _is_ws(text[i]):
            locs.append(i)
    return locs


def _remove_leading_ws(completion: str) -> str:
    i = 0
    while i < len(completion) and completion[i].isspace():
        i += 1
    return completion[i:]


def _remove_word_overlap_prefix(prefix: str, completion: str) -> str:
    right_trimmed = completion.lstrip()
    starts = _start_locations(prefix)
    while starts:
        idx = starts.pop()
        left_sub = prefix[idx:]
        if right_trimmed.startswith(left_sub):
            return right_trimmed.replace(left_sub, "", 1)
    return completion


def _remove_word_overlap_suffix(completion: str, suffix: str) -> str:
    suffix_trimmed = _remove_leading_ws(suffix)
    starts = _start_locations(completion)
    while starts:
        idx = starts.pop()
        comp_sub = completion[idx:]
        if suffix_trimmed.startswith(comp_sub):
            return completion[:idx]
    return completion


def _remove_ws_overlap_prefix(prefix: str, completion: str) -> str:
    i = len(prefix) - 1
    while completion and i >= 0 and completion[0] == prefix[i]:
        completion = completion[1:]
        i -= 1
    return completion


def _remove_ws_overlap_suffix(completion: str, suffix: str) -> str:
    i = 0
    while completion and i < len(suffix) and completion[-1] == suffix[i]:
        completion = completion[:-1]
        i += 1
    return completion


class RemoveOverlap:
    def process(self, prefix: str, suffix: str, completion: str, context) -> str:
        completion = _remove_word_overlap_prefix(prefix, completion)
        completion = _remove_word_overlap_suffix(completion, suffix)
        completion = _remove_ws_overlap_prefix(prefix, completion)
        completion = _remove_ws_overlap_suffix(completion, suffix)
        return completion


