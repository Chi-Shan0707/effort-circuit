# Step 5: Data-Driven Token Metric

## Requirement

`second_prompt.md` warned that the hand-coded continue/final token set is fragile. In particular, tokens like `The` can reflect generic English continuation rather than useful reasoning. Step 5 therefore replaces the hand-coded metric with a train/heldout split:

```text
score(token) = mean_logprob_high(token) - mean_logprob_low(token)
continue/high cluster = top high > low tokens on train
final/low cluster = top low > high tokens on train
```

The selected clusters must then be evaluated on heldout rows, not on the same rows used to pick tokens. The script also reports KL divergence, answer-token margin, and EOS/Final margin so cluster gap cannot be overinterpreted alone.

## Implementation

Added `src/data_driven_token_metric.py` and `tests/test_data_driven_token_metric.py`.

Important implementation details:

- Token clusters are selected on `synthetic_math` train rows.
- Heldout evaluation uses `heldout_synthetic_math` rows.
- Very low-probability tokens are filtered with `--min-mean-prob` to avoid pure logprob-ratio artifacts.
- Continue cluster requires `score > 0`.
- Final cluster requires `score < 0`.
- Patch conditions use hook-captured low/high states, consistent with the Step 1 site audit.

Reported heldout metrics:

- data-driven continue mass;
- data-driven final mass;
- data-driven cluster gap;
- KL vs low logits;
- answer-token margin;
- EOS/Final margin.

## Command

```bash
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
python -m src.data_driven_token_metric \
  --model ../TinyLoRA-GRPO-Coder/models/Qwen3-0.6B \
  --train-dataset synthetic_math \
  --train-n 8 \
  --eval-dataset heldout_synthetic_math \
  --eval-n 5 \
  --low-mode instant \
  --high-mode verify \
  --top-k 12 \
  --min-mean-prob 0.0001 \
  --conditions 18:1,27:0.75,27:1 \
  --out outputs/data_driven_metric_train8_heldout5.json \
  --device cpu \
  --dtype float32 \
  --num-threads 4
```

## Results

Artifacts:

- `outputs/data_driven_metric_train8_heldout5.json`
- `outputs/data_driven_metric_train8_heldout5.md`

Learned train clusters:

| cluster | tokens |
| --- | --- |
| high/continue | `Diagram`, `Trace`, `Circ`, `Analy`, `Logic`, `Rational`, `Xiao`, `diagram`, `...,`, `Mark`, `[...]`, `Taking` |
| low/final | `\\`, `3`, `2`, `1` |

Heldout summary:

| condition | n | cluster gap | continue mass | final mass | KL vs low | answer margin | EOS/Final margin |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| low | 5 | `-0.013686` | `0.000000` | `0.013686` | `0.000000` | `-7.716664` | `0.000000` |
| high | 5 | `+0.005499` | `0.006327` | `0.000828` | `6.899302` | `-7.095190` | `-0.006323` |
| layer18_t1 | 5 | `+0.000100` | `0.001953` | `0.001853` | `5.099311` | `-7.082969` | `-0.001949` |
| layer27_t0.75 | 5 | `-0.001869` | `0.001072` | `0.002941` | `2.564157` | `-7.428588` | `-0.001069` |
| layer27_t1 | 5 | `+0.005499` | `0.006327` | `0.000828` | `6.899302` | `-7.095190` | `-0.006323` |

## Interpretation

This step is intentionally more skeptical than the earlier hand-coded gap metric.

Key findings:

1. The train-selected clusters are not semantically clean reasoning tokens. Even with a probability floor, high-vs-low logprob contrast selects odd tokens such as `Diagram`, `Trace`, `Circ`, and `Xiao`.
2. The final/low cluster is dominated by numeric first-answer tokens (`3`, `2`, `1`) plus a formatting token. This reveals that `instant` prompts are mostly next-token answer-number prompts, while `verify` prompts are not.
3. Layer27_t1 exactly matches the high condition on this metric, which is expected for a final-layer hook replacement but reinforces the readout-transfer interpretation.
4. Layer27_t0.75 does not reproduce a strong heldout cluster-gap gain.
5. Answer-token margins remain very negative for all conditions, so none of these next-token posture shifts should be interpreted as directly improving answer commitment.

The clean conclusion is:

> Data-driven metric selection does not rescue the useful-effort claim. It instead shows that high-vs-low token contrast is highly sensitive to prompt artifacts, rare-ish continuation tokens, and the fact that low prompts expect immediate numeric answers.

This is valuable because it falsifies a weak measurement strategy. Future metrics should likely use structured behavioral labels or learned probes over hidden states, not raw top token contrasts alone.

## Consequence For The Roadmap

Step 6 should separate process state from task state. The current token-level metrics cannot tell whether the transferred state contains:

- generic format posture;
- source prompt wording;
- task content;
- answer-number bias;
- actual reasoning process posture.

Cross-question patching and mean process-direction patching are therefore the next necessary gate.

## Verification

```bash
python -m src.data_driven_token_metric --help
python -m pytest -q
```

Result: `20 passed`.
