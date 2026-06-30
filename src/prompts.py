from __future__ import annotations


def build_prompt(question: str, mode: str) -> str:
    mode = mode.strip()
    if mode == "instant":
        return f"Answer with only the final number.\nQuestion: {question}\nAnswer:"
    if mode == "cot":
        return f"Solve step by step, then give the final number as 'Final answer: <number>'.\nQuestion: {question}\nReasoning:"
    if mode == "verify":
        return (
            "Solve the problem, verify the arithmetic once, then give 'Final answer: <number>'.\n"
            f"Question: {question}\nReasoning:"
        )
    if mode == "budget_forcing":
        return (
            "Use careful reasoning. Do not answer until you have checked the plan and arithmetic. "
            "End with 'Final answer: <number>'.\n"
            f"Question: {question}\nReasoning:"
        )
    raise ValueError(f"Unknown prompt mode {mode!r}")


def parse_modes(value: str) -> list[str]:
    modes = [mode.strip() for mode in value.split(",") if mode.strip()]
    if not modes:
        raise ValueError("At least one mode is required")
    return modes
