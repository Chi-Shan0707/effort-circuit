# Third Stage: Neuron / SAE Gate

## Requirement

`third_prompt.md` says not to start with neurons or SAEs; only after residual posture is clarified should the project ask which neurons, MLPs, heads, or SAE features contribute to the posture. After Steps 1–7 and the n=50 heldout audit, this stage performs a small smoke gate to verify that the repository's neuron/SAE path is runnable.

This is intentionally **not** a scientific neuron discovery claim. It is a reproducibility and pipeline gate.

## Commands

```bash
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4

python -m src.collect_traces \
  --model ../TinyLoRA-GRPO-Coder/models/Qwen3-0.6B \
  --dataset synthetic_math \
  --n 8 \
  --modes instant,verify \
  --out outputs/third_stage_traces_n8.parquet \
  --no-model \
  --max-new-tokens 32

python -m src.cache_activations \
  --model ../TinyLoRA-GRPO-Coder/models/Qwen3-0.6B \
  --traces outputs/third_stage_traces_n8.parquet \
  --layers 18,27 \
  --activation-sites resid_post,mlp_act \
  --positions prompt_last \
  --out outputs/third_stage_activations_n8 \
  --device cpu \
  --dtype float32 \
  --shard-size 64

python -m src.discover_neurons \
  --activations outputs/third_stage_activations_n8 \
  --traces outputs/third_stage_traces_n8.parquet \
  --site mlp_act \
  --out outputs/third_stage_neurons_n8.json

python -m src.train_sae \
  --activations outputs/third_stage_activations_n8 \
  --site resid_post \
  --position prompt_last \
  --layer 18 \
  --hidden-dim 256 \
  --epochs 5 \
  --batch-size 8 \
  --lr 0.001 \
  --l1 0.001 \
  --out outputs/third_stage_sae_layer18_smoke
```

## Artifacts

- `outputs/third_stage_traces_n8.parquet`
- `outputs/third_stage_activations_n8/manifest.json`
- `outputs/third_stage_activations_n8/shard_00000.pt`
- `outputs/third_stage_neurons_n8.json`
- `outputs/third_stage_sae_layer18_smoke/sae.pt`
- `outputs/third_stage_sae_layer18_smoke/metrics.json`

## Neuron Smoke Result

`src.discover_neurons` produced 100 ranked MLP activation coordinates across layers 18 and 27. The top rows are dominated by layer 27 in this tiny smoke set, e.g. neuron 385 with Cohen's d around `14.97`.

This should not be overinterpreted because:

1. n=8 with synthetic fallback completions is far too small;
2. labels are based on fallback trace metadata, not a clean behavioral process label;
3. the earlier third-stage audits show layer27 is close to readout/posture effects;
4. no ablation or feature steering validation has been run for these neurons.

## SAE Smoke Result

A small ReLU SAE was trained on layer18 `resid_post` prompt-last activations:

| metric | value |
| --- | ---: |
| input_dim | 1024 |
| hidden_dim | 256 |
| examples | 16 |
| final_loss | 4.9134 |
| mse | 4.2656 |
| explained_variance | 0.6525 |
| L0 | 93.125 |
| dead_feature_rate | 0.6094 |

Interpretation:

- The training loop runs and loss decreases.
- Reconstruction is only moderate and dead-feature rate is high, which is expected for a tiny n=16 smoke run.
- This is not an interpretable SAE; it is a proof that the pipeline can cache activations, train, save, and report metrics.

## Code Improvement

`src/train_sae.py` now writes `metrics.json` alongside `sae.pt`, including final loss, MSE, explained variance, L0, dead feature rate, and number of examples.

Added `tests/test_train_sae.py` to lock basic SAE forward shapes and ReLU nonnegativity.

## Conclusion

The neuron/SAE gate is now operational but not scientifically conclusive. The next serious SAE stage would require:

1. behavior-labeled traces rather than fallback completions;
2. hundreds to thousands of prompts;
3. separate train/heldout activation splits;
4. feature ablation/steering validation;
5. comparison against the proven posture metrics and first-final utility metrics.

For now, the correct claim is:

> The repo can execute a lightweight neuron/SAE pipeline, but no neuron or SAE feature should yet be called a reasoning circuit.

## Verification

```bash
python -m pytest -q
```

Result: `25 passed`.
