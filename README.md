# effort-circuit

Research scaffold for identifying and causally intervening on internal activations that increase useful reasoning effort in a local HuggingFace causal LM. The default configuration points at the local Qwen3-0.6B model:

```bash
TinyLoRA-GRPO-Coder/models/Qwen3-0.6B
```

Core charter: 一切从第一性原理出发。See `FIRST_PRINCIPLES.md`.

Comprehensive findings: see `docs/final_analysis_report.md`.

## Setup

```bash
cd effort-circuit
python -m pytest -q
```

Optional editable install is not required; CLIs are module-entry scripts under `src/`.

## Pipeline

Collect baseline traces:

```bash
python -m src.collect_traces \
  --model ../TinyLoRA-GRPO-Coder/models/Qwen3-0.6B \
  --dataset synthetic_math \
  --n 2000 \
  --modes instant,cot,verify,budget_forcing \
  --out outputs/traces.parquet
```

Cache activations:

```bash
python -m src.cache_activations \
  --model ../TinyLoRA-GRPO-Coder/models/Qwen3-0.6B \
  --traces outputs/traces.parquet \
  --layers all \
  --activation-sites resid_post,mlp_act,mlp_out \
  --positions prompt_last,reasoning_all,pre_final \
  --out outputs/activations/
```

Discover vectors, neurons, RPM modes, run sweeps, run CET, and analyze:

```bash
python -m src.discover_vectors --activations outputs/activations/ --traces outputs/traces.parquet --methods paired_diff,probe,svd --labels high_effort,correct,positive_flip --out outputs/vectors/
python -m src.discover_neurons --activations outputs/activations/ --traces outputs/traces.parquet --site mlp_act --out outputs/neurons.json
python -m src.train_sae --activations outputs/activations/ --site mlp_act --position prompt_last --layer 0 --out outputs/sae/
python -m src.sweeps --model ../TinyLoRA-GRPO-Coder/models/Qwen3-0.6B --dataset heldout_synthetic_math --vectors outputs/vectors/ --neurons outputs/neurons.json --alphas -3,-2,-1,-0.5,0,0.5,1,2,3 --layers all --modes residual_vector,neuron_scale,random_direction --out outputs/sweeps.parquet
python -m src.rpm --activations outputs/activations/ --traces outputs/traces.parquet --layers selected_by_probe --pca-dim 128 --out outputs/rpm/
python -m src.cet_controller --model ../TinyLoRA-GRPO-Coder/models/Qwen3-0.6B --dataset heldout_synthetic_math --effort-vector outputs/rpm/best.pt --probe outputs/vectors/best_probe.pt --target-percentile 75 --alpha-max 2.0 --soft-budget 256 --hard-budget 512 --out outputs/cet_eval.parquet
python -m src.analyze --inputs outputs/sweeps.parquet outputs/cet_eval.parquet --out outputs/report.md
```

## Continuous engineering loop

Run a command continuously so the repo keeps testing or executing a pipeline loop:

```bash
python -m src.loop_engineering --command "python -m pytest -q" --interval 30 --iterations 0
```

`--iterations 0` means run forever until interrupted. A shell wrapper is also available:

```bash
scripts/continuous_engineering_loop.sh "python -m pytest -q"
```

## Notes

- Activation capture uses PyTorch forward hooks and writes sharded `.pt` files plus `manifest.json`.
- Discovery commands tolerate small smoke-test trace sets but are intended for larger generated runs.
- The implementation includes lightweight random-direction controls and verbosity/repetition metrics to guard against uncontrolled length increases.
