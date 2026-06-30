from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable

import numpy as np
import pandas as pd


FINAL_RE = re.compile(r"(final answer|answer|####)", re.I)


def first_final_position(text: str) -> int | None:
    match = FINAL_RE.search(text or "")
    return None if match is None else match.start()


def count_reasoning_tokens(tokenizer, prompt: str, completion: str) -> tuple[int, int]:
    total = len(tokenizer.encode(prompt + completion, add_special_tokens=False))
    completion_tokens = len(tokenizer.encode(completion, add_special_tokens=False))
    return completion_tokens, total


def whitespace_token_count(text: str) -> int:
    return len((text or "").split())


def repetition_rate(text: str, ngram: int = 3) -> float:
    words = re.findall(r"\w+", (text or "").lower())
    if len(words) < ngram:
        return 0.0
    grams = [tuple(words[i : i + ngram]) for i in range(len(words) - ngram + 1)]
    counts = Counter(grams)
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    return repeated / max(1, len(grams))


def malformed_rate(answers: Iterable[object]) -> float:
    values = list(answers)
    if not values:
        return 0.0
    return sum(value is None or value == "" or (isinstance(value, float) and math.isnan(value)) for value in values) / len(values)


def cost_adjusted_score(accuracy: float, reasoning_tokens: float, baseline_tokens: float = 1.0) -> float:
    return float(accuracy) / (1.0 + max(0.0, reasoning_tokens - baseline_tokens) / 256.0)


def summarize_eval(df: pd.DataFrame) -> dict[str, float]:
    if df.empty:
        return {}
    return {
        "accuracy": float(df["correct"].mean()) if "correct" in df else float("nan"),
        "reasoning_tokens": float(df["reasoning_tokens"].mean()) if "reasoning_tokens" in df else float("nan"),
        "total_tokens": float(df["total_tokens"].mean()) if "total_tokens" in df else float("nan"),
        "repetition_rate": float(np.mean([repetition_rate(x) for x in df.get("completion", [])])),
        "malformed_rate": malformed_rate(df.get("extracted_answer", [])),
    }
