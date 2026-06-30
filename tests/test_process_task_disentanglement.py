from src.process_task_disentanglement import summarize


def test_summarize_reports_source_beats_target_rate():
    rows = [
        {
            "condition": "cross",
            "layer": 27,
            "continue_final_gap": 1.0,
            "target_answer_logprob": -3.0,
            "source_answer_logprob": -2.0,
            "source_minus_target_logprob": 1.0,
            "target_answer_rank": 10,
            "source_answer_rank": 5,
        },
        {
            "condition": "cross",
            "layer": 27,
            "continue_final_gap": 3.0,
            "target_answer_logprob": -1.0,
            "source_answer_logprob": -4.0,
            "source_minus_target_logprob": -3.0,
            "target_answer_rank": 2,
            "source_answer_rank": 20,
        },
    ]
    summary = summarize(rows)[0]
    assert summary["mean_continue_final_gap"] == 2.0
    assert summary["source_answer_beats_target_rate"] == 0.5
