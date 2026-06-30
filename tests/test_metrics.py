from src.metrics import cost_adjusted_score, malformed_rate, repetition_rate


def test_repetition_rate_detects_repeated_ngrams():
    assert repetition_rate("a b c a b c") > 0
    assert repetition_rate("a b c d") == 0


def test_malformed_rate():
    assert malformed_rate([None, "", "3", 4]) == 0.5


def test_cost_adjusted_score_penalizes_extra_tokens():
    assert cost_adjusted_score(1.0, 512, 1) < cost_adjusted_score(1.0, 2, 1)
