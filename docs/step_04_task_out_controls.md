# Step 4: Task-Out and Verbosity Controls

## Requirement

`second_prompt.md` asked for controls that test whether the apparent late-layer effort signal is actually useful reasoning, or merely a generic continuation / verbosity / format direction.

The required control families were:

- reasoning: arithmetic, date arithmetic, symbolic tasks;
- non-reasoning: capital lookup, translation, sentiment rewrite;
- verbosity: verbose explanation but no step-by-step solving;
- format: same `Reasoning` / `Final answer` format without explicit verification.

From first principles, a useful-reasoning claim needs task specificity. If the same intervention moves non-reasoning prompts or verbosity prompts similarly, the safe interpretation is generic generation-posture control rather than reasoning-specific effort.

## Implementation

Added `src/task_out_controls.py` and `tests/test_task_out_controls.py`.

The script builds a factorial control set:

| family | subtypes |
| --- | --- |
| reasoning | arithmetic, date arithmetic, symbolic equation solving |
| non_reasoning | capital lookup, translation, sentiment rewrite |

For each task, it compares a concise low prompt with three source prompts:

| source mode | purpose |
| --- | --- |
| `high_reasoning` | careful reasoning + verification |
| `high_verbosity` | verbose explanation without step-by-step solve |
| `high_format` | same `Reasoning` / `Final answer` format, no explicit verification |

For each layer, the script captures source hook states and patches them into the low prompt at the same decoder block hook site. This follows the Step 1 site-audit conclusion: use hook-captured states instead of assuming `outputs.hidden_states` equivalence.

Measured quantities:

- continue-final gap;
- verbosity-token mass delta;
- format-token mass delta;
- reasoning vs non-reasoning family differences.

## Command

```bash
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
python -m src.task_out_controls \
  --model ../TinyLoRA-GRPO-Coder/models/Qwen3-0.6B \
  --n-per-subtype 2 \
  --layers 18,27 \
  --source-modes high_reasoning,high_verbosity,high_format \
  --out outputs/task_out_controls_n2_layers18_27.json \
  --device cpu \
  --dtype float32 \
  --num-threads 4
```

## Results

Artifacts:

- `outputs/task_out_controls_n2_layers18_27.json`
- `outputs/task_out_controls_n2_layers18_27.md`

Key summary:

| family | source mode | layer | n | delta gap | delta verbosity | delta format |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| reasoning | high_reasoning | 27 | 6 | `+0.230902` | `+0.000117` | `+0.000386` |
| reasoning | high_format | 27 | 6 | `+0.142877` | `+0.000036` | `+0.000025` |
| reasoning | high_verbosity | 27 | 6 | `+0.095382` | `+0.000090` | `+0.000009` |
| non_reasoning | high_format | 27 | 6 | `+0.057764` | `+0.000046` | `+0.000016` |
| non_reasoning | high_reasoning | 27 | 6 | `-0.124015` | `+0.000021` | `+0.000132` |
| non_reasoning | high_verbosity | 27 | 6 | `-0.044269` | `+0.000166` | `+0.000008` |

Layer 18 shows a similar but less clean pattern:

| family | source mode | layer | n | delta gap |
| --- | --- | ---: | ---: | ---: |
| reasoning | high_format | 18 | 6 | `+0.177950` |
| reasoning | high_reasoning | 18 | 6 | `+0.152557` |
| reasoning | high_verbosity | 18 | 6 | `+0.077921` |
| non_reasoning | high_format | 18 | 6 | `+0.074346` |
| non_reasoning | high_reasoning | 18 | 6 | `-0.190641` |
| non_reasoning | high_verbosity | 18 | 6 | `-0.052246` |

## Interpretation

The control results are mixed, and that is scientifically useful.

Supported:

1. The strongest positive layer27 effect appears on reasoning-family tasks with `high_reasoning` source states.
2. `high_verbosity` produces a smaller reasoning-family delta than `high_reasoning`, so the effect is not reducible to verbosity alone in this small run.
3. Verbosity-token mass deltas are tiny, so the measured continue-final shift is not simply a large increase in the hand-coded verbosity token set.

Not yet supported:

1. A pure useful-reasoning claim. `high_format` also increases reasoning-family gap, so formatting/posture contributes substantially.
2. A clean task-specificity claim. Non-reasoning `high_format` still increases the gap, showing that the metric is sensitive to generic format posture outside arithmetic reasoning.
3. A layer27 mechanism claim independent of readout proximity. At the final decoder block, replacing the last-position hook output can make patched logits very close to the source prompt logits, so layer27 remains vulnerable to being interpreted as readout-level policy transfer.

The safest conclusion is:

> Step 4 weakens the interpretation that the late-layer signal is purely useful reasoning. It is better described as a generation-posture signal with reasoning, format, and task-family components. Useful-reasoning specificity remains unproven.

## Consequence For The Roadmap

The next step should replace the hand-coded token metric. The current continue-final gap is too sensitive to prompt family and format. Step 5 should learn discriminative token clusters on a train split and evaluate them on heldout tasks, with additional KL, answer-token margin, EOS/Final margin, and format/verbosity controls.

## Verification

```bash
python -m src.task_out_controls --help
python -m pytest -q
```

Result: `18 passed`.
