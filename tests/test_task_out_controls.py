import pytest

from src.task_out_controls import build_control_tasks, build_prompt


def test_control_tasks_cover_reasoning_and_non_reasoning():
    tasks = build_control_tasks(n_per_subtype=2, seed=0)
    families = {task.family for task in tasks}
    subtypes = {task.subtype for task in tasks}
    assert families == {"reasoning", "non_reasoning"}
    assert {"arithmetic", "date_arithmetic", "symbolic"}.issubset(subtypes)
    assert {"capital_lookup", "translation", "sentiment_rewrite"}.issubset(subtypes)


def test_control_prompt_modes_are_distinct():
    question = "What is 2 + 2?"
    assert "Answer:" in build_prompt(question, "low")
    assert "verify" in build_prompt(question, "high_reasoning").lower()
    assert "verbosely" in build_prompt(question, "high_verbosity").lower()
    assert "Final answer" in build_prompt(question, "high_format")


def test_unknown_control_prompt_mode_raises():
    with pytest.raises(ValueError):
        build_prompt("What is 2 + 2?", "missing")
