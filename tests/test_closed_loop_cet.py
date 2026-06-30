from src.closed_loop_cet import alpha_policy, has_valid_final


def test_has_valid_final_requires_final_marker():
    assert has_valid_final("Reasoning...\nFinal answer: 42")
    assert not has_valid_final("Reasoning...\nAnswer: 42")


def test_alpha_policy_stops_on_repetition():
    alpha = alpha_policy(
        generated_tokens=4,
        continue_mass=0.1,
        final_mass=0.2,
        repetition=0.5,
        alpha_max=1.0,
        min_reasoning_tokens=16,
        repetition_threshold=0.18,
    )
    assert alpha == 0.0


def test_alpha_policy_pushes_away_from_early_final():
    alpha = alpha_policy(
        generated_tokens=4,
        continue_mass=0.1,
        final_mass=0.2,
        repetition=0.0,
        alpha_max=1.0,
        min_reasoning_tokens=16,
        repetition_threshold=0.18,
    )
    assert alpha == 1.0
