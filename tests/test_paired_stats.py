from src.paired_stats import exact_mcnemar_p, paired_bootstrap_ci, wilson_ci


def test_wilson_ci_handles_zero_successes():
    ci = wilson_ci(0, 5)
    assert ci["low"] == 0.0
    assert 0.0 < ci["high"] < 1.0


def test_exact_mcnemar_p_no_discordance_is_one():
    assert exact_mcnemar_p(0, 0) == 1.0


def test_paired_bootstrap_diff_direction():
    ci = paired_bootstrap_ci([(False, True), (False, True), (True, True)], iterations=100, seed=0)
    assert ci["mean_diff"] == 2 / 3
    assert ci["low"] >= 0.0
