from src.data_driven_token_metric import parse_conditions, summarize


def test_parse_conditions_includes_low_high_baselines():
    assert parse_conditions("18:1,27:0.75") == [
        ("low", None, None),
        ("high", None, None),
        ("layer18_t1", 18, 1.0),
        ("layer27_t0.75", 27, 0.75),
    ]


def test_summarize_groups_by_condition():
    rows = [
        {
            "condition": "low",
            "cluster_gap": 1.0,
            "continue_mass": 2.0,
            "final_mass": 1.0,
            "kl_vs_low": 0.0,
            "answer_margin": -1.0,
            "eos_final_margin": -0.5,
        },
        {
            "condition": "low",
            "cluster_gap": 3.0,
            "continue_mass": 4.0,
            "final_mass": 1.0,
            "kl_vs_low": 0.0,
            "answer_margin": -3.0,
            "eos_final_margin": -1.5,
        },
    ]
    summary = summarize(rows)
    assert summary[0]["condition"] == "low"
    assert summary[0]["mean_cluster_gap"] == 2.0
    assert summary[0]["mean_answer_margin"] == -2.0
