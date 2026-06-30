from __future__ import annotations

import random
from dataclasses import dataclass, asdict
from typing import Iterable


@dataclass(frozen=True)
class Problem:
    problem_id: str
    question: str
    gold_answer: str


def _make_problem(index: int, rng: random.Random, split: str) -> Problem:
    kind = rng.choice(["addmul", "twostep", "linear", "compare"])
    a, b, c = rng.randint(2, 30), rng.randint(2, 30), rng.randint(2, 20)
    if kind == "addmul":
        answer = (a + b) * c
        question = f"If x = {a} + {b}, what is x times {c}?"
    elif kind == "twostep":
        answer = a * b - c
        question = f"A box has {a} rows with {b} items in each row, then {c} items are removed. How many remain?"
    elif kind == "linear":
        answer = a + b * c
        question = f"Mia has {a} tokens and gets {b} more tokens for each of {c} rounds. How many tokens does she have?"
    else:
        left = a * c
        right = b * c
        answer = abs(left - right)
        question = f"What is the positive difference between {a} times {c} and {b} times {c}?"
    return Problem(f"{split}-{index:06d}", question, str(answer))


def load_dataset(name: str, n: int, seed: int = 42) -> list[dict[str, str]]:
    if name not in {"synthetic_math", "heldout_synthetic_math"}:
        raise ValueError(f"Unsupported dataset {name!r}; expected synthetic_math or heldout_synthetic_math")
    offset = 10_000 if name.startswith("heldout") else 0
    rng = random.Random(seed + offset)
    split = "heldout" if name.startswith("heldout") else "train"
    return [asdict(_make_problem(i, rng, split)) for i in range(n)]


def iter_dataset(name: str, n: int, seed: int = 42) -> Iterable[dict[str, str]]:
    yield from load_dataset(name, n, seed)
