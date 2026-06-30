# Step 3: Expanded Heldout Statistics

## Requirement

`second_prompt.md` asked for expanded heldout statistics rather than raw small-n accuracy tables. The goal is not to make n=5 look significant; it is to prevent overclaiming by attaching uncertainty and paired evidence to each intervention.

From first principles, the relevant unit is the problem instance. A useful intervention should flip the same problem from wrong to right more often than it flips it from right to wrong. Therefore the correct comparison is paired, not independent-condition accuracy.

## Implementation

Added `src/paired_stats.py`.

The script reads experiment JSON rows and reports:

- Wilson 95% confidence intervals for condition accuracy.
- Paired bootstrap 95% confidence intervals for accuracy difference versus baseline.
- Exact McNemar-style discordant counts:
  - baseline wrong / candidate right;
  - baseline right / candidate wrong;
  - two-sided exact binomial p-value over discordant pairs.
- Auxiliary protocol metrics:
  - strict stop rate;
  - strict malformed rate;
  - mean generated tokens;
  - mean repetition rate.

Regression tests in `tests/test_paired_stats.py` cover zero-success Wilson intervals, no-discordance McNemar behavior, and bootstrap directionality.

## Command

```bash
python -m src.paired_stats \
  --input outputs/stop_after_final_heldout_n5_hook_states.json \
  --out outputs/stop_after_final_stats_n5.json \
  --bootstrap 10000 \
  --seed 123
```

## Results

Artifacts:

- `outputs/stop_after_final_stats_n5.json`
- `outputs/stop_after_final_stats_n5.md`

### First Strict Final Accuracy

All conditions remain 0/5 under the strict first-final metric:

| condition | n | acc | Wilson 95% CI | diff vs baseline | bootstrap 95% CI | discordant pairs | exact p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 5 | 0.000 | [0.000, 0.434] | +0.000 | [+0.000, +0.000] | 0 / 0 | 1.000 |
| layer18_t1 | 5 | 0.000 | [0.000, 0.434] | +0.000 | [+0.000, +0.000] | 0 / 0 | 1.000 |
| layer27_t0.75 | 5 | 0.000 | [0.000, 0.434] | +0.000 | [+0.000, +0.000] | 0 / 0 | 1.000 |
| layer27_t1 | 5 | 0.000 | [0.000, 0.434] | +0.000 | [+0.000, +0.000] | 0 / 0 | 1.000 |

This is the central result. There is no observed strict first-commit utility gain in this run.

### Last Detectable Answer Accuracy

The weaker last-answer metric reproduces the earlier appearance of a possible layer27_t0.75 benefit, but it is not statistically stable:

| condition | n | acc | Wilson 95% CI | diff vs baseline | bootstrap 95% CI | b-wrong/c-right | b-right/c-wrong | exact p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 5 | 0.400 | [0.118, 0.769] | +0.000 | [+0.000, +0.000] | 0 | 0 | 1.000 |
| layer18_t1 | 5 | 0.000 | [0.000, 0.434] | -0.400 | [-0.800, +0.000] | 0 | 2 | 0.500 |
| layer27_t0.75 | 5 | 0.600 | [0.231, 0.882] | +0.200 | [+0.000, +0.600] | 1 | 0 | 1.000 |
| layer27_t1 | 5 | 0.200 | [0.036, 0.624] | -0.200 | [-0.600, +0.000] | 0 | 1 | 1.000 |

The paired evidence for layer27_t0.75 is only one discordant improvement and zero discordant regressions. That is a useful lead, not a reliable conclusion.

### Protocol Quality

| condition | n | strict stop rate | strict malformed rate | mean tokens | repetition |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 5 | 0.200 | 0.800 | 105.40 | 0.0800 |
| layer18_t1 | 5 | 0.600 | 0.400 | 74.00 | 0.0466 |
| layer27_t0.75 | 5 | 0.200 | 0.800 | 105.80 | 0.0251 |
| layer27_t1 | 5 | 0.200 | 0.800 | 105.80 | 0.0861 |

The high malformed rate is as important as the accuracy result. The model often performs arithmetic in free text or emits generic `Answer:` markers, but fails the stricter `Final answer:` commitment protocol.

## Interpretation

Step 3 reinforces Step 2:

1. The clean first-final metric shows no current utility gain.
2. The last-answer metric is weaker because it can reward post-commit drift or post-marker correction.
3. The uncertainty intervals are wide enough that n=5 cannot settle positive or negative effect sizes.
4. The intervention may affect process posture and repetition without reliably producing correct answer commitment.

The most honest conclusion is that the project has a real local causal control signal, but the task-utility claim is not yet established.

## Scaling Plan

The script is designed to scale to n=50+ rows from the same JSON schema. Recommended next command after improving format control:

```bash
python -m src.stop_after_final_experiment \
  --model ../TinyLoRA-GRPO-Coder/models/Qwen3-0.6B \
  --dataset heldout_synthetic_math \
  --n 50 \
  --low-mode instant \
  --high-mode verify \
  --conditions 27:0.75,27:1 \
  --max-new-tokens 128 \
  --out outputs/stop_after_final_heldout_n50_hook_states.json \
  --device cpu \
  --dtype float32 \
  --num-threads 4

python -m src.paired_stats \
  --input outputs/stop_after_final_heldout_n50_hook_states.json \
  --out outputs/stop_after_final_stats_n50.json
```

Given the current 4-core/8GiB cgroup limit, n=50 should be treated as an overnight CPU run, not an interactive quick check.

## Verification

```bash
python -m src.paired_stats --help
python -m src.paired_stats --input outputs/stop_after_final_heldout_n5_hook_states.json --out outputs/stop_after_final_stats_n5.json --bootstrap 10000 --seed 123
python -m pytest -q
```

Result: `15 passed`.
