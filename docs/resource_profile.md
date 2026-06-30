# CPU / Memory / SSD Resource Profile

Date: 2026-06-30 UTC
Project: `effort-circuit`
Model: `../TinyLoRA-GRPO-Coder/models/Qwen3-0.6B`

## Conclusion

The machine reports a large host CPU and memory pool, but this notebook/container is cgroup-limited. For this workload, the practical setting is:

```bash
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
```

Inside Python, set:

```python
import torch
torch.set_num_threads(4)
```

Reason: the container CPU quota is `400000 100000`, i.e. approximately 4 CPU cores. A real Qwen3-0.6B prefill benchmark shows 4 Torch threads are fastest; 8 threads oversubscribe the quota and slow down.

## First-Principles Interpretation

From first principles, local LLM inference on CPU is bounded by four interacting resources:

1. **Compute quota**: how many CPU cycles the process can actually consume, not how many host CPUs exist.
2. **Memory capacity**: whether model weights, activations, tokenizer buffers, and outputs fit without reclaim or OOM.
3. **Memory bandwidth / cache locality**: CPU inference repeatedly streams model weights; oversubscription can reduce cache efficiency.
4. **Storage latency / throughput**: model loading and output writing touch SSD-backed filesystems; repeated model reloads waste time.

Therefore “use all CPU” does **not** mean setting threads to 240. It means matching worker threads to the cgroup quota and avoiding CPU throttling, memory pressure, and IO waits.

## Host Hardware Reported by Linux

Collected with `lscpu`:

- CPU model: Intel(R) Xeon(R) 6759P-C
- Reported logical CPUs: 240
- Topology: 2 sockets × 60 cores/socket × 2 threads/core
- NUMA nodes: 2
- Notable ISA: AVX2, AVX-512, AVX512_BF16, AVX512_VNNI, AMX_BF16, AMX_INT8
- Host L3 cache: 640 MiB across 2 instances

Important caveat: these are host-visible CPUs, not the quota available to this process.

## Container / Cgroup Limits

Collected from `/sys/fs/cgroup`:

- `cpu.max`: `400000 100000`
- Effective CPU quota: about **4 cores**
- `memory.max`: `8589934592` bytes = **8 GiB**
- `memory.current` at sampling: about **6.65 GiB**
- `swap`: 0 bytes
- `cpu.stat` throttling observed: `nr_throttled=115906`, `throttled_usec=872527796300`

Interpretation:

- The process can see CPU IDs `0-239`, but the scheduler throttles it to ~4 cores.
- Memory headroom is limited. Loading Qwen3-0.6B in float32 brings the container close to its 8 GiB limit.
- No swap means memory spikes can fail hard rather than degrade gradually.

## Memory State

Host-level `free -h` reported:

- Total host memory: 1.9 TiB
- Host available memory: 1.6 TiB
- Host swap: 0 B

But cgroup memory is the binding limit:

- Container memory max: 8 GiB
- Container current usage at sampling: ~6.65 GiB

Practical implication:

- Avoid loading multiple model copies.
- Prefer one long-running process that reuses the model.
- Avoid very large activation caches in `/home/jovyan/work`; write shards and clean intermediate files.
- Prefer `float32` only when required; `bfloat16`/`float16` on CPU may not always speed up and should be benchmarked before use.

## Disk / SSD State

Collected with `df -hT` and `lsblk`:

- Project workspace mount: `/home/jovyan/work`
- Filesystem type: loop-backed XFS
- Workspace capacity seen by `df`: 12 GiB
- Workspace used: 7.3 GiB
- Workspace free: 4.8 GiB
- Root overlay free: 36 GiB
- Large read-only public storage: `/home/jovyan/public-ro` on RAID5 over 4 × Intel SSDPF2KX038T1 NVMe devices, total 10.5 TiB

Practical implication:

- The writable workspace is small; do not store full large activation dumps without sharding and pruning.
- Keep durable experiment summaries as JSON/Markdown; move or delete bulky scratch outputs after analysis.
- Model loading reads from local workspace/model path; repeated reloads are wasteful.

## CPU / Memory / IO Wait Snapshot

Collected with Linux PSI and `vmstat 1 5`:

- CPU PSI `some avg10`: 79.60
- Memory PSI `some avg10`: 45.62
- IO PSI `some avg10`: 1.84
- `vmstat` showed high runnable queues (`r` hundreds) while CPU idle stayed around 1–2% during sampling.

Interpretation:

- The host is busy and/or the container is quota-constrained.
- IO wait is not the dominant bottleneck for the Qwen forward benchmark.
- Oversubscribing CPU threads increases scheduler contention and throttling rather than useful throughput.

## Qwen3-0.6B CPU Thread Benchmark

Benchmark artifact: `outputs/cpu_thread_benchmark.json`

Model load time in benchmark process: 9.35 s

Mean by Torch thread count:

| Torch threads | Mean seconds / prompt | Mean prefill tokens/s |
| ---: | ---: | ---: |
| 1 | 1.1041 | 21.89 |
| 2 | 0.3050 | 78.21 |
| 4 | 0.2438 | 96.56 |
| 8 | 0.3923 | 61.82 |

Raw rows:

| Torch threads | Prompt tokens | Seconds | Prefill tokens/s |
| ---: | ---: | ---: | ---: |
| 1 | 9 | 0.9231 | 9.75 |
| 1 | 25 | 1.0308 | 24.25 |
| 1 | 43 | 1.3583 | 31.66 |
| 2 | 9 | 0.2336 | 38.53 |
| 2 | 25 | 0.2992 | 83.57 |
| 2 | 43 | 0.3821 | 112.53 |
| 4 | 9 | 0.1503 | 59.90 |
| 4 | 25 | 0.2513 | 99.50 |
| 4 | 43 | 0.3300 | 130.29 |
| 8 | 9 | 0.2009 | 44.79 |
| 8 | 25 | 0.5037 | 49.63 |
| 8 | 43 | 0.4724 | 91.03 |

Conclusion from benchmark:

- 1 thread underuses the quota.
- 2 threads are much faster than 1.
- **4 threads are best** and align with cgroup CPU quota.
- 8 threads oversubscribe and are slower than 4.

## Prompt Steering Experiment Status

Artifact: `outputs/prompt_steering_verify_n2_layers.json`

This experiment tests the first-principles local causal question:

> Does the hidden-state difference between high-effort and low-effort prompts causally shift the next-token distribution toward continuation tokens rather than final-answer tokens?

Observed summary:

- Baseline continue mass: 0.378631
- Baseline final mass: 0.00183581
- Baseline continue-minus-final gap: 0.376796
- Best intervention: layer 4, alpha -3.0, delta gap 0.165293
- Worst intervention: layer 8, alpha 3.0, delta gap -0.0399231

Caution:

- This is a tiny `n=2` prompt-level causal screening result, not yet a full reasoning-accuracy claim.
- It is useful because it found a measurable next-token distribution shift without expensive autoregressive generation.

## Recommended Runtime Policy

Use these defaults for CPU-bound experiments in this environment:

```bash
cd /home/jovyan/work/yuhanchi_remote/effort-circuit
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
python -m src.prompt_steering_experiment   --model ../TinyLoRA-GRPO-Coder/models/Qwen3-0.6B   --dataset synthetic_math   --n 2   --high-mode verify   --layers 0,4,8,12,16,20,24,27   --alphas=-3,-1,0,1,3   --out outputs/prompt_steering_verify_n2_layers.json   --device cpu   --dtype float32   --num-threads 4
```

For longer experiments:

1. Keep one Python process alive where possible; avoid reloading the model per sweep.
2. Increase `n` only after the prompt-level signal reproduces.
3. Store raw bulky activations as sharded `.pt`; immediately write compact reports.
4. Monitor:

```bash
cat /sys/fs/cgroup/cpu.stat
cat /sys/fs/cgroup/memory.current
cat /proc/pressure/cpu
cat /proc/pressure/memory
cat /proc/pressure/io
vmstat 1 5
```

## Open Risks

- The container is memory-limited to 8 GiB; full activation caching for all layers/sites may exceed writable disk or memory if not sharded aggressively.
- The CPU quota is 4 cores despite 240 visible logical CPUs; using process pools can easily oversubscribe.
- The writable project filesystem has only about 4.8 GiB free at sampling time.
- Full autoregressive generation remains expensive on CPU; prefer forward-only causal probes before generation sweeps.
