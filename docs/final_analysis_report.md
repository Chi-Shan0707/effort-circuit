# Effort-Circuit 综合分析报告

日期：2026-06-30 UTC  
仓库：`effort-circuit`  
模型：`../TinyLoRA-GRPO-Coder/models/Qwen3-0.6B`  
核心宗旨：一切从第一性原理出发。

## 1. 总结结论

本轮研究已经得到一个有价值、可复现、并且跨越多个因果证据层级的发现：

> 在 Qwen3-0.6B 的 synthetic arithmetic prompts 上，late-layer residual states，尤其是 layer 27，包含一个可因果操纵的“继续结构化推理 / 避免低质量 finalization 或 repetition”的内部状态。对该状态进行 activation patching 或 low→high interpolation，不仅能单调控制下一 token 的 continue-vs-final 概率，还能在小规模严格生成协议中提升答案正确率。

最关键的数字如下：

| 证据层级 | 关键实验 | 最强结果 |
| --- | --- | --- |
| 局部 steering | `prompt_steering_verify_n8_all_layers` | layer 4, alpha -3，把 continue-final gap 提高 `+0.115317` |
| 随机方向对照 | `random_control_verify_n8_layer4_alpha-3` | 候选方向超过 32/32 个随机方向，但最强随机方向很接近 |
| 全层 patching | `activation_patching_instant_verify_n8_all_layers` | layer 27 low→high patch 提高 gap `+0.734238`；layer 18 high→low patch 降低 gap `-0.317724` |
| 插值单调性 | `activation_interpolation_instant_verify_n8_late_layers` | layers 18/20/23/27 全部 low→high 单调增加；layer 27 最强 `+0.734243` |
| 严格生成 train-small | `strict_generation_prefix_patch_n3_late_layers_reextracted` | baseline accuracy `0.333`；layer27_t0.75/t1 accuracy `1.000` |
| 严格生成 heldout-small | `strict_generation_prefix_patch_heldout_n5_layer27_reextracted` | baseline accuracy `0.400`；layer27_t0.75 `0.600`；layer27_t1 `0.800` |

因此，本项目目前最可信的机制性结论不是“找到一个思考神经元”，也不是“任意加一个 reasoning vector 会变聪明”，而是：

1. **早层方向可以影响 logits，但特异性不足。** Layer 4 direction addition 能显著提高 continuation-token mass，但随机方向也能接近该效果，说明这个指标容易被一般 logit 扰动影响。
2. **后层 activation state 更像真正的因果控制变量。** Layers 18–27 的 activation patching 出现了清晰的双向因果效应：high state patch 到 low run 会恢复 continuation posture；low state patch 到 high run 会抑制 continuation posture。
3. **Layer 27 是当前最强候选位点。** 它在 patching、interpolation、strict generation 三个阶段都表现突出。
4. **该效果初步进入任务级 utility。** 在小规模 strict generation 中，layer 27 prefix patch 不只改变下一 token 概率，还提高了 arithmetic answer accuracy，并降低 repetition rate。

## 2. 第一性原理框架

本项目避免直接假设“模型有一个思考神经元”。从自回归语言模型的最小事实出发：

1. 模型每一步由当前 hidden state 产生 logits。
2. logits 决定下一 token 分布。
3. “继续推理”和“立即作答”首先表现为下一 token 分布差异。
4. 如果某个内部状态有因果作用，则干预它必须在保持 prompt/run 其余部分固定时改变下游分布。
5. 如果这种干预有用，则它最终还应改变生成轨迹，并提高正确率或 cost-adjusted correctness，而不是只制造 verbosity。

因此，我们把证据门槛分成四层：

| Gate | 问题 | 本项目证据状态 |
| --- | --- | --- |
| Local causal gate | 干预 hidden state 是否改变下一 token 概率？ | 已通过 |
| Specificity gate | 效果是否强于随机方向/非特异扰动？ | 部分通过；方向加法仍需更强对照 |
| Trajectory gate | 干预是否改变真实生成轨迹？ | 已初步通过 |
| Utility gate | 是否提高正确率且不只是啰嗦？ | 小样本通过，需扩大验证 |

这个框架非常重要：它解释了为什么我们没有停在 probe 或 steering vector 上，而是继续做 random controls、activation patching、interpolation、strict generation 和 heldout replication。

## 3. 环境与资源约束

资源画像见 `docs/resource_profile.md`。

虽然 Linux 可见 240 个逻辑 CPU，但容器 cgroup 实际限制约为 4 CPU cores 和 8 GiB memory：

- `cpu.max = 400000 100000`，约等于 4 cores。
- `memory.max = 8589934592`，约等于 8 GiB。
- workspace 可用磁盘空间有限，不适合长期保存大规模全层 activation dump。

实测 Qwen3-0.6B CPU prefill benchmark 显示：

| Torch threads | Mean seconds / prompt | Mean prefill tokens/s |
| ---: | ---: | ---: |
| 1 | 1.1041 | 21.89 |
| 2 | 0.3050 | 78.21 |
| 4 | 0.2438 | 96.56 |
| 8 | 0.3923 | 61.82 |

因此所有正式实验采用：

```bash
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
# and torch.set_num_threads(4)
```

这不是保守设置，而是当前容器 quota 下的最优设置。使用 8 线程会超配并变慢。

## 4. 实验 001：Prompt-Level Effort Steering

### 4.1 目的

先验证最低成本的局部因果问题：

> `verify` prompt 与 `instant` prompt 的 hidden-state 差分方向，是否能在 neutral `cot` prompt 上提高“继续推理 token”相对“final answer token”的下一步概率？

度量定义：

```text
continue-final gap = P(continue token set) - P(final token set)
```

continue tokens 包括：`First`, `Let's`, `We`, `Since`, `Calculate`, `Step`, `Wait`, `The`。  
final tokens 包括：`Final`, `Answer`, `Therefore`, `Thus`, `So`。

### 4.2 结果

`outputs/prompt_steering_verify_n8_all_layers.json`：

- Baseline mean continue mass：`0.379785`
- Baseline mean final mass：`0.00180739`
- Baseline gap：`0.377977`
- 最强正向变化：layer 4, alpha -3.0
  - mean gap：`0.493294`
  - delta vs baseline：`+0.115317`
- 最强负向变化：layer 7, alpha 3.0
  - mean gap：`0.300347`
  - delta vs baseline：`-0.0776305`

### 4.3 解释

这个实验说明：内部方向加法确实可以改变局部 next-token distribution。但它也暴露两个问题：

1. 最强方向是 `alpha=-3`，与朴素 `verify - instant` 正方向相反，说明 prompt contrast 可能混入 formatting 或其他语义。
2. 该实验只证明局部 logits 可控，不证明真实 reasoning accuracy 改善。

所以它是一个有效筛选器，但不是最终机制结论。

## 5. 实验 002：随机方向对照与短生成诊断

### 5.1 随机方向对照

`outputs/random_control_verify_n8_layer4_alpha-3.json`：

| 指标 | 数值 |
| --- | ---: |
| Candidate delta gap | `0.115317` |
| Random mean delta | `0.0383364` |
| Random min delta | `-0.0457047` |
| Random max delta | `0.114225` |
| Candidate percentile among 32 random directions | `1.000` |

候选方向超过所有 32 个随机方向，但最强随机方向几乎追平。这说明：

- 候选方向确实有效。
- 但当前 token-mass metric 的特异性不够强。
- 一般方向扰动也可能增加 continuation mass。

因此不能把 layer 4 direction 称为“effort circuit”。它只是 early-stage steering candidate。

### 5.2 短生成诊断

`outputs/generation_intervention_verify_n3_layer4_alpha-3.json` 和 `outputs/generation_instant_intervention_verify_n5_layer4_alpha-3.json` 显示，自由生成测试不干净：

- 输出常被 token budget 截断。
- 模型会先吐一个数字，再开始解释。
- `instant` prompt 下也会出现 reasoning-like continuation。
- 短生成中的 accuracy / final_rate 很难解释。

这个负结果很有价值：它说明必须从自由生成切换到更严格的 protocol 和更可控的 patching。

## 6. 实验 003：全层 Activation Patching

### 6.1 为什么 patching 更强

Direction addition 可能把 hidden state 推到 off-manifold 区域；随机方向也可能偶然影响 logits。Activation patching 更接近因果机制验证：

- clean/high run：`verify`
- corrupt/low run：`instant`
- 把某一层最后 token 的 high hidden state 替换到 low run。
- 反过来，把 low hidden state 替换到 high run。

如果某层状态因果相关，那么：

```text
low→high patch 应提高 continuation gap
high→low patch 应降低 continuation gap
```

### 6.2 全层结果

`outputs/activation_patching_instant_verify_n8_all_layers.json`：

| 方向 | 最强层 | 效果 |
| --- | ---: | ---: |
| low→high restore | layer 27 | `+0.734238` vs low |
| high→low suppress | layer 18 | `-0.317724` vs high |

原始 gap：

- `instant` low gap：`9.88969e-05`
- `verify` high gap：`0.307173`

Low→high patch 排名显示 layers 18–27 整体很强：

| layer | low→high delta | high→low delta |
| ---: | ---: | ---: |
| 27 | `+0.734238` | `-0.307079` |
| 20 | `+0.273016` | `-0.308342` |
| 23 | `+0.268370` | `-0.307154` |
| 21 | `+0.265580` | `-0.306911` |
| 24 | `+0.261163` | `-0.307258` |
| 19 | `+0.259649` | `-0.311671` |
| 25 | `+0.250527` | `-0.306917` |
| 26 | `+0.249288` | `-0.306968` |
| 22 | `+0.245819` | `-0.307303` |
| 18 | `+0.218337` | `-0.317724` |

### 6.3 解释

这是目前第一条强因果证据链：

1. late-layer high state 能恢复 low run 的 continuation posture。
2. late-layer low state 能压低 high run 的 continuation posture。
3. 效果集中在后层 18–27，而不是早层。

这改变了研究重点：

- 早层 direction addition 是有趣现象。
- 后层 activation state 才是更可信的 causal site。

## 7. 实验 004：Late-Layer Interpolation 与 Strict Generation

### 7.1 插值实验

为了排除“直接替换造成离散 off-manifold artifact”，我们做 low→high interpolation：

```text
h(t) = (1 - t) h_low + t h_high
```

测试 `t ∈ {0, 0.25, 0.5, 0.75, 1}`。

`outputs/activation_interpolation_instant_verify_n8_late_layers.json`：

| layer | delta t=1 - t=0 | 单调递增 | 曲线概览 |
| ---: | ---: | ---: | --- |
| 18 | `+0.218337` | True | `0.0001 → 0.0007 → 0.0114 → 0.1005 → 0.2184` |
| 20 | `+0.273016` | True | `0.0001 → 0.0017 → 0.0258 → 0.1558 → 0.2731` |
| 23 | `+0.268370` | True | `0.0001 → 0.0018 → 0.0240 → 0.1387 → 0.2685` |
| 27 | `+0.734243` | True | `0.0001 → 0.0011 → 0.0198 → 0.2836 → 0.7343` |

这个结果很关键：late-layer 状态不是只有 hard patch 才有效，而是在连续插值下表现为单调控制变量。

特别是 layer 27 呈现强非线性：

- t=0.5 时 gap 约 `0.0198`
- t=0.75 时 gap 跳到 `0.2836`
- t=1.0 时 gap 到 `0.7343`

这提示 layer 27 可能接近 final logits/readout posture，少量状态变化就会显著改变输出分布。

### 7.2 Strict generation：训练小样本

我们改用更严格的 prompt：

```text
You must use exactly this format:
Reasoning: <brief arithmetic steps>
Final answer: <number>

Question: ...
Reasoning:
```

并只做 prefix patch：在 prefill 最后位置 patch late-layer hidden state，然后让模型生成。

修正答案抽取器后，`outputs/strict_generation_prefix_patch_n3_late_layers_reextracted.json`：

| condition | accuracy | answer rate | final rate | repetition |
| --- | ---: | ---: | ---: | ---: |
| baseline | `0.333` | `1.000` | `1.000` | `0.7260` |
| layer18_t1 | `0.000` | `1.000` | `0.667` | `0.6774` |
| layer20_t1 | `0.333` | `1.000` | `0.667` | `0.5763` |
| layer27_t0.75 | `1.000` | `1.000` | `1.000` | `0.3582` |
| layer27_t1 | `1.000` | `1.000` | `1.000` | `0.2200` |

关键点：layer27 patch 同时提高 accuracy 并降低 repetition。这个结果反驳了“它只是让模型更啰嗦”的简单解释。

### 7.3 Strict generation：heldout 小样本复现

`outputs/strict_generation_prefix_patch_heldout_n5_layer27_reextracted.json`：

| condition | accuracy | answer rate | final rate | repetition |
| --- | ---: | ---: | ---: | ---: |
| baseline | `0.400` | `1.000` | `1.000` | `0.2783` |
| layer27_t0.75 | `0.600` | `1.000` | `0.600` | `0.1609` |
| layer27_t1 | `0.800` | `1.000` | `0.800` | `0.2025` |

这仍是小样本，但方向正确：layer27_t1 从 baseline `0.400` 提到 `0.800`。

## 8. 答案抽取器修正

严格生成实验暴露了原始答案抽取器的问题：它会在一行中抽取第一个数字。例如：

```text
Final answer: 54 - 5 = 49
```

旧抽取可能得到 `54`，正确答案应是 `49`。

修正后策略：

1. 优先寻找最后一个 `Final answer` / `Answer` / `####` marker。
2. 在该 marker 所在行取最后一个数字。
3. 如果没有 marker，再 fallback 到已有 pattern，并取最后一次匹配。

新增测试：

- `Final answer: 54 - 5 = 49` → `49`
- `The final answer is 60 + 45 = 105.` → `105`
- 多次 `Final answer` 时优先最后一次。

测试通过：`10 passed`。

## 9. 机制解释：我们现在相信什么？

### 9.1 最可信解释

当前证据最支持：

> 模型 late residual stream 中存在一种“输出姿态 / reasoning posture”状态。它不是一个孤立神经元，而是后层残差状态中的分布式变量。该变量直接影响下一 token 是继续结构化推理还是进入低质量 finalization/repetition。

Layer 27 很可能靠近 logits/readout，因此 patching 它会产生强烈效果。Layers 18–24 则可能是该 posture 逐步形成和稳定的区域。

### 9.2 早层与后层的差异

早层 layer 4 direction addition 显著，但随机方向也接近，说明早层 perturbation 可能影响更泛化的 “text mode / continuation mode”。

后层 patching 与 interpolation 更可信，因为：

- 它使用真实 hidden state 而非任意方向。
- 它可双向 restore/suppress。
- 它在连续插值下单调。
- 它在 generation 中提升正确率。

### 9.3 为什么不是“啰嗦向量”

如果只是 verbosity vector，我们预期：

- reasoning token 增加；
- repetition 增加；
- accuracy 不稳定或下降。

但 strict generation 中 layer27_t1：

- train-small accuracy：`1.000` vs baseline `0.333`
- heldout-small accuracy：`0.800` vs baseline `0.400`
- repetition 低于 baseline

这说明它至少不只是简单 verbosity。当然，仍需要 task-out controls 来进一步证明 useful-effort specificity。

## 10. 局限与风险

必须明确：当前结论仍是 early-stage mechanistic evidence，不是最终论文级结论。

主要局限：

1. **样本量小。** CPU quota 只有 4 cores，generation 成本较高；目前 train n=3、heldout n=5 只能作为强信号，不足以给出稳健统计结论。
2. **任务范围窄。** 主要是 synthetic arithmetic；还没有事实问答、翻译、摘要等 task-out controls。
3. **生成仍有截断和 repetition。** `max_new_tokens=96` 常被用满，说明还需要 stop-after-final 机制。
4. **Layer 27 可能有 off-distribution 风险。** t=1 有时产生过强输出姿态；t=0.75 在一些样本更自然，但 heldout 中 t=1 更准。
5. **未做多 seed。** 目前 deterministic decoding 下复现，但未系统扫 seed / sample variants。
6. **未证明 closed-loop CET。** 当前是 prefix patch，不是 token-by-token controller。

## 11. 下一步研究计划

优先级从高到低：

### 11.1 Stop-after-final generation

实现生成停止机制：一旦出现首个合法 `Final answer: <number>` 就停止。然后 rerun：

- baseline
- layer27_t0.75
- layer27_t1

建议规模：heldout n=20 起步。

目标：降低 repetition 和 truncation，使 accuracy / token cost 更可靠。

### 11.2 Layer 27 局部插值细扫

重点扫非线性区间：

```text
t ∈ {0.6, 0.7, 0.75, 0.8, 0.9, 1.0}
```

目标：找到 accuracy、final_rate、repetition、cost 的 Pareto 最优点。

### 11.3 Task-out controls

加入非推理任务：

- 简单事实问答
- 翻译
- 情绪改写
- 常识问题

目标：验证 layer27 patch 是否只在 reasoning tasks 上有用，而不是普遍改变输出风格。

### 11.4 CET controller

从 prefix patch 进化到闭环控制：

```text
if answer_margin high and effort_state low:
    inject / interpolate toward high-effort late-layer state
if valid final answer appears:
    stop
if repetition rises:
    reduce intervention
```

目标：不是让模型一直多想，而是在需要时维持 reasoning posture，并在答案稳定时停止。

## 12. 可复现命令

### 12.1 资源设置

```bash
cd /home/jovyan/work/yuhanchi_remote/effort-circuit
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
```

### 12.2 全层 patching

```bash
python -m src.activation_patching_experiment \
  --model ../TinyLoRA-GRPO-Coder/models/Qwen3-0.6B \
  --dataset synthetic_math \
  --n 8 \
  --low-mode instant \
  --high-mode verify \
  --layers all \
  --out outputs/activation_patching_instant_verify_n8_all_layers.json \
  --device cpu \
  --dtype float32 \
  --num-threads 4
```

### 12.3 Late-layer interpolation

```bash
python -m src.activation_interpolation_experiment \
  --model ../TinyLoRA-GRPO-Coder/models/Qwen3-0.6B \
  --dataset synthetic_math \
  --n 8 \
  --low-mode instant \
  --high-mode verify \
  --layers 18,20,23,27 \
  --ts 0,0.25,0.5,0.75,1 \
  --out outputs/activation_interpolation_instant_verify_n8_late_layers.json \
  --device cpu \
  --dtype float32 \
  --num-threads 4
```

### 12.4 Strict generation heldout

```bash
python -m src.strict_generation_patch_experiment \
  --model ../TinyLoRA-GRPO-Coder/models/Qwen3-0.6B \
  --dataset heldout_synthetic_math \
  --n 5 \
  --low-mode instant \
  --high-mode verify \
  --conditions 27:0.75,27:1 \
  --max-new-tokens 96 \
  --out outputs/strict_generation_prefix_patch_heldout_n5_layer27.json \
  --device cpu \
  --dtype float32 \
  --num-threads 4
```

## 13. 文件索引

核心报告：

- `docs/experiment_001_prompt_steering.md`
- `docs/experiment_002_causal_controls_and_patching.md`
- `docs/experiment_003_all_layer_activation_patching.md`
- `docs/experiment_004_multistage_late_layer_intervention.md`
- `docs/final_analysis_report.md`
- `docs/resource_profile.md`

核心脚本：

- `src/prompt_steering_experiment.py`
- `src/random_direction_control.py`
- `src/activation_patching_experiment.py`
- `src/activation_interpolation_experiment.py`
- `src/strict_generation_patch_experiment.py`
- `src/answer_extractors.py`

核心输出：

- `outputs/activation_patching_instant_verify_n8_all_layers.json`
- `outputs/activation_interpolation_instant_verify_n8_late_layers.json`
- `outputs/strict_generation_prefix_patch_n3_late_layers_reextracted.json`
- `outputs/strict_generation_prefix_patch_heldout_n5_layer27_reextracted.json`

## 14. 最终判断

本项目已经从“猜测存在 effort direction”推进到“定位 late-layer causal state，并在小样本上展示 task-level utility”。

最值得保留的结论是：

> Layer 27 residual state patch/interpolation 是当前最强的 causal intervention candidate。它在局部 logits、单调插值、严格生成和 heldout-small 复现中均表现为有用信号。下一阶段应围绕 layer 27 做 stop-after-final、扩大 heldout、加入 task-out controls 和 CET controller。

