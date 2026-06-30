from src.stop_after_final_experiment import extract_first_strict_final


def test_strict_final_extraction_ignores_generic_answer_marker():
    assert extract_first_strict_final("Reasoning...\nAnswer: 1") is None


def test_strict_final_extraction_accepts_final_answer_line():
    assert extract_first_strict_final("Reasoning...\nFinal answer: 123") == "123"
