from src.answer_extractors import extract_answer, is_correct, normalize_answer


def test_extract_final_answer_phrase():
    extracted = extract_answer("We compute 2+3=5. Final answer: 5")
    assert extracted.text == "5"
    assert extracted.first_final_position is not None


def test_extract_hash_answer():
    assert extract_answer("reasoning\n#### 42").text == "42"


def test_normalize_numeric_answers():
    assert normalize_answer("5.0") == "5"
    assert is_correct("005", 5)


def test_extract_uses_last_number_on_final_answer_line():
    assert extract_answer("Final answer: 54 - 5 = 49").text == "49"
    assert extract_answer("The final answer is 60 + 45 = 105.").text == "105"


def test_extract_prefers_last_final_answer_marker():
    text = "Final answer: 30\nReasoning again\nFinal answer: 105"
    assert extract_answer(text).text == "105"
