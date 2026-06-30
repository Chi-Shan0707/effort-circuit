from __future__ import annotations

import re
from dataclasses import dataclass


ANSWER_PATTERNS = [
    re.compile(r"####\s*(-?\d+(?:\.\d+)?)"),
    re.compile(r"(?:final answer|answer|therefore|result)\s*(?:is|=|:)?\s*(-?\d+(?:\.\d+)?)", re.I),
    re.compile(r"=\s*(-?\d+(?:\.\d+)?)\s*(?:$|[.\n])"),
    re.compile(r"(-?\d+(?:\.\d+)?)\s*$"),
]


@dataclass(frozen=True)
class ExtractedAnswer:
    text: str | None
    first_final_position: int | None


def extract_answer(text: str) -> ExtractedAnswer:
    if not text:
        return ExtractedAnswer(None, None)
    marker = re.compile(r"(####|final answer|answer)\s*(?:is|=|:)?", re.I)
    matches = list(marker.finditer(text))
    for match in reversed(matches):
        line = text[match.end() :].splitlines()[0]
        numbers = re.findall(r"-?\d+(?:\.\d+)?", line)
        if numbers:
            return ExtractedAnswer(numbers[-1], match.end() + line.rfind(numbers[-1]))
    for pattern in ANSWER_PATTERNS:
        matches = list(pattern.finditer(text.strip()))
        if matches:
            match = matches[-1]
            return ExtractedAnswer(match.group(1), match.start(1))
    return ExtractedAnswer(None, None)


def normalize_answer(answer: object) -> str | None:
    if answer is None:
        return None
    value = str(answer).strip().replace(",", "")
    if not value:
        return None
    try:
        numeric = float(value)
        if numeric.is_integer():
            return str(int(numeric))
        return f"{numeric:.8g}"
    except ValueError:
        return value.lower()


def is_correct(extracted: object, gold: object) -> bool:
    return normalize_answer(extracted) == normalize_answer(gold)
