# Step 6: Process vs Task Disentanglement

## Requirement

`second_prompt.md` required separating process state from task state. The key test is cross-question patching:

```text
source question A high/verify state → target question B low/instant run
```

A clean process-posture intervention should increase continuation/reasoning posture on target B without leaking source A's answer or suppressing target B's answer content. The prompt also proposed comparing single-sample state patching with a mean process direction:

```text
v_process = mean(h_verify - h_instant)
```

## Implementation

Added `src/process_task_disentanglement.py` and `tests/test_process_task_disentanglement.py`.

Conditions:

| condition | intervention |
| --- | --- |
| `low` | target low/instant prompt, no patch |
| `same_question_high_replace` | target high state replaced into target low prompt |
| `cross_question_high_replace` | source high state from another problem replaced into target low prompt |
| `mean_process_direction_add` | add mean train-set `verify - instant` hook-state direction to target low state |

Metrics:

- continue-final gap;
- target answer logprob/rank;
- source answer logprob/rank;
- source-minus-target logprob;
- source-answer-beats-target rate.

The source-answer metrics are leakage checks. The target-answer metrics are content-preservation checks.

## Command

```bash
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
python -m src.process_task_disentanglement \
  --model ../TinyLoRA-GRPO-Coder/models/Qwen3-0.6B \
  --train-dataset synthetic_math \
  --train-n 8 \
  --eval-dataset heldout_synthetic_math \
  --eval-n 5 \
  --low-mode instant \
  --high-mode verify \
  --layers 18,27 \
  --alpha 1 \
  --out outputs/process_task_disentanglement_n5_layers18_27.json \
  --device cpu \
  --dtype float32 \
  --num-threads 4
```

## Results

Artifacts:

- `outputs/process_task_disentanglement_n5_layers18_27.json`
- `outputs/process_task_disentanglement_n5_layers18_27.md`

Summary:

| condition | layer | n | gap | target rank | source rank | source-target lp | source beats target |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| low | 18 | 5 | `0.000099` | `7.4` | `11.4` | `-0.437551` | `0.200` |
| same_question_high_replace | 18 | 5 | `0.194027` | `325.0` | `386.4` | `-0.425360` | `0.200` |
| cross_question_high_replace | 18 | 5 | `0.207214` | `295.4` | `438.6` | `-0.656561` | `0.200` |
| mean_process_direction_add | 18 | 5 | `0.205474` | `330.6` | `383.8` | `-0.698142` | `0.200` |
| low | 27 | 5 | `0.000099` | `7.4` | `11.4` | `-0.437551` | `0.200` |
| same_question_high_replace | 27 | 5 | `0.264117` | `672.8` | `847.4` | `-0.528995` | `0.200` |
| cross_question_high_replace | 27 | 5 | `0.284528` | `681.4` | `841.6` | `-0.418365` | `0.200` |
| mean_process_direction_add | 27 | 5 | `0.258192` | `440.8` | `947.2` | `-0.861446` | `0.200` |

## Interpretation

This experiment gives a nuanced result.

Supported:

1. Cross-question source states transfer continuation posture. Cross-question replacement raises the gap from approximately `0.0001` to `0.207` at layer 18 and `0.285` at layer 27.
2. Mean process-direction addition also transfers posture, with gaps `0.205` at layer 18 and `0.258` at layer 27.
3. There is no strong source-answer leakage by this next-token metric. The source answer beats the target answer in only `1/5` cases, the same rate as baseline.
4. The mean direction appears safer than raw layer27 cross replacement on source-answer leakage margin: layer27 mean direction has source-minus-target logprob `-0.861`, more source-suppressed than cross replacement `-0.418`.

Not supported:

1. Content preservation. All posture interventions make the target answer rank much worse. Baseline target answer rank is `7.4`; layer27 cross replacement pushes it to `681.4`, and layer27 mean direction to `440.8`.
2. Useful task solving. The interventions transfer process posture but suppress immediate answer-token readiness.
3. A clean process-only vector. Even the mean direction changes answer-token geometry substantially.

The central conclusion is:

> Cross-question and mean-direction interventions can move process posture without obvious source-answer leakage, but they also damage target-answer availability. The current process signal is not yet disentangled enough to be a safe useful-reasoning controller.

## Consequence For The Roadmap

Step 7 should not blindly apply a constant prefix patch. A CET controller must be closed-loop:

- increase reasoning posture only when the model is finalizing too early;
- monitor answer-token readiness or final validity;
- reduce intervention when answer rank/margin collapses;
- stop immediately after valid final answer;
- avoid repeating high-posture steering after repetition begins.

## Verification

```bash
python -m src.process_task_disentanglement --help
python -m pytest -q
```

Result: `21 passed`.
