# Effort-Circuit 当前综合结论报告

日期：2026-06-30 UTC  
仓库：`effort-circuit`  
模型：`../TinyLoRA-GRPO-Coder/models/Qwen3-0.6B`  
核心宗旨：一切从第一性原理出发。  
当前 HEAD 覆盖至：closure audit after third-stage n=50 heldout and neuron/SAE smoke gate。

## 0. 最终结论先行

本项目已经得到一个真实、有价值、但必须谨慎表述的机制线索：

> 在 Qwen3-0.6B 的 synthetic arithmetic 设置中，late decoder-block hook states，尤其 layers 18–27，能够因果改变模型的下一 token continuation/finalization posture；但在更严格的 first-final commitment evaluation 下，目前没有证据表明该干预已经可靠提高最终答题正确率。

因此，当前最准确的结论不是“发现了思考神经元”，也不是“layer 27 patch 能稳定提升 accuracy”。更准确的结论是：

1. **局部因果控制成立。** Activation patching 和 interpolation 显示 late residual/post-block state 能强力改变 continue-vs-final token 分布。
2. **layer 27 旧结果存在 site-mismatch 风险。** Step 1 显示 `outputs.hidden_states[layer+1]` 与 decoder block hook output 在 layer 27 不等价；所有旧的 layer27 hidden-state patch 结论必须降级为“site-mismatch intervention”或用 hook-captured states 重跑。
3. **任务级 utility 尚未成立。** Step 2 的 stop-after-first-final 实验显示 first strict final accuracy 全部为 `0/5`；Step 3 的配对统计确认没有 observed first-final gain。
4. **旧的 heldout accuracy 改善应重新解释。** 早先 `0.400 → 0.800` 的 heldout-small 改善来自 last-answer extraction，可能奖励 post-marker drift 或后续自我修正，不是干净的首次提交正确率。
5. **下一步真正问题是 process posture 到 answer commitment 的断裂。** 干预能影响“继续/解释/重复/收束”的生成姿态，但还没有稳定把正确计算绑定到规范 `Final answer:` 动作。

一句话版：

> 当前发现是一个强 mechanistic lead 和一个有价值的 negative utility audit；它证明了 late-layer process-posture control 的存在，但也证明了原先的 accuracy claim 不能直接成立。

## 1. 第一性原理问题重构

不要从“有没有一个思考神经元”开始。自回归模型的最小事实是：

1. 每一步 hidden state 经过后续层、norm、lm head 产生 logits。
2. logits 决定下一 token 分布。
3. “继续推理”“开始 final”“重复”“格式化提交”首先都是下一 token policy 的选择。
4. 如果某个内部状态是因果变量，固定其他条件并替换/插值该状态应改变下游 policy。
5. 如果该变量有任务价值，它必须最终改善 first committed answer，而不仅是生成更多文本或让最后一次抽取碰巧正确。

因此本项目的证据门槛应分为五个 gate：

| Gate | 核心问题 | 当前状态 |
| --- | --- | --- |
| Local causal gate | 干预 activation 是否改变下一 token distribution？ | 通过 |
| Specificity gate | 是否强于随机方向或非特异 logit 扰动？ | 部分通过；早层方向仍弱 |
| Site identity gate | 被缓存和被 patch 的 activation 是否同一位置？ | layer 18 通过；layer 27 旧 hidden-state patch 未通过 |
| Trajectory gate | 干预是否改变真实生成轨迹？ | 初步通过 |
| First-commit utility gate | 首次规范提交的答案是否更正确？ | 未通过 |

这个 gate 结构是当前报告最重要的整理：它把“发现了可控状态”和“提高了答案正确率”分开，避免把局部机制信号误写成应用结论。

## 2. 资源与最佳配置

资源画像见 `docs/resource_profile.md`。

实际环境不是 240 核满血机器。虽然 Linux 可见 240 logical CPUs，但容器 cgroup 限制约为：

- CPU quota：`cpu.max = 400000 100000`，约等于 4 cores。
- Memory：`memory.max = 8589934592`，约等于 8 GiB。
- Disk：workspace 可用空间有限，不适合保存大规模 activation dumps。

CPU prefill benchmark 表明 Qwen3-0.6B 在当前容器中 4 threads 最优：

| Torch threads | Mean seconds / prompt | Mean prefill tokens/s |
| ---: | ---: | ---: |
| 1 | 1.1041 | 21.89 |
| 2 | 0.3050 | 78.21 |
| 4 | 0.2438 | 96.56 |
| 8 | 0.3923 | 61.82 |

推荐配置：

```bash
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
# plus --num-threads 4 or torch.set_num_threads(4)
```

超过 4 线程会与 cgroup quota 冲突，导致 oversubscription，反而更慢。

## 3. 已完成证据链

### 3.1 Prompt-level direction steering

早期实验使用 `verify - instant` prompt contrast 构造方向，并在 neutral prompt 上测：

```text
continue-final gap = P(continue token set) - P(final token set)
```

结果：

- baseline gap：`0.377977`
- layer 4 alpha -3：gap delta `+0.115317`
- layer 7 alpha +3：gap delta `-0.0776305`

解释：方向加法能改变局部 logits，但 `alpha=-3` 方向和随机方向接近，说明这个阶段只能作为探索，不是机制结论。

### 3.2 Random direction control

Layer 4 candidate 超过 32/32 个随机方向，但最强随机方向几乎追平：

| 指标 | 数值 |
| --- | ---: |
| Candidate delta gap | `0.115317` |
| Random mean delta | `0.0383364` |
| Random max delta | `0.114225` |
| Candidate percentile | `1.000` |

解释：候选方向有效，但特异性不足。不能称为 effort circuit。

### 3.3 Full-layer activation patching

更强的实验是真实 activation patching：

- low/corrupt：`instant`
- high/clean：`verify`
- patch 最后 prompt token 的某层 state

旧实验结果：

| patch 方向 | 最强结果 |
| --- | ---: |
| low→high restore | layer 27 delta `+0.734238` |
| high→low suppress | layer 18 delta `-0.317724` |

后层 18–27 整体强，说明 process posture 更可能在 late residual/post-block states 中形成或被 read out。

### 3.4 Interpolation

Low→high interpolation：

```text
h(t) = (1 - t) h_low + t h_high
```

结果：layers 18/20/23/27 均单调提高 continue-final gap；layer 27 最强，`t=1 - t=0` delta `+0.734243`。

解释：不是只有 hard replacement 有效；late state 具有连续控制变量特征。但 layer 27 的非线性跃迁也提示它可能非常接近 final readout，site identity 必须严查。

## 4. Step 1：Activation-site audit

文件：

- `src/site_audit.py`
- `docs/step_01_site_audit.md`
- `outputs/site_audit_n2_layers18_27.json`
- `outputs/site_audit_n2_layers18_27.md`

关键发现：

- Passive hook 与 self-hook patch 不改变 logits，说明 hook patch machinery 本身有效。
- layer 18：`outputs.hidden_states[layer+1]` 与 decoder block hook output 等价，delta 为 0。
- layer 27：二者严重不等价，hidden-state-to-hook patch 造成巨大 logit/state delta。

结论：

> 之前 layer 27 的 hidden-state patching 结果不能被解释为“把同一 activation site 的 high state patch 到 low run”。它仍是一个有效干预现象，但可能是 site-mismatch/off-manifold/readout-boundary intervention。

这一步显著降低了旧 layer27 claim 的可信度，但提高了整个项目的严谨性。

## 5. Step 2：Stop-after-first-final evaluation

文件：

- `src/stop_after_final_experiment.py`
- `tests/test_stop_after_final_experiment.py`
- `docs/step_02_stop_after_final.md`
- `outputs/stop_after_final_heldout_n5_hook_states.json`
- `outputs/stop_after_final_heldout_n5_hook_states.md`

核心修正：

- 使用 hook-captured states，而不是旧 `outputs.hidden_states`。
- 生成时遇到严格 `Final answer: <number>` 或 `Final: <number>` 即停止。
- 不把 generic `Answer: <number>` 当作 first-final，因为它在该 prompt 下常是格式漂移或编号占位。

heldout n=5 结果：

| condition | first-final acc | last-answer acc | strict stop rate | mean tokens | repetition |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 0.000 | 0.400 | 0.200 | 105.40 | 0.0800 |
| layer18_t1 | 0.000 | 0.000 | 0.600 | 74.00 | 0.0466 |
| layer27_t0.75 | 0.000 | 0.600 | 0.200 | 105.80 | 0.0251 |
| layer27_t1 | 0.000 | 0.200 | 0.200 | 105.80 | 0.0861 |

关键结论：

> 所有条件 first strict final accuracy 都是 `0/5`。

这直接推翻了“当前 intervention 已经可靠提高 answer correctness”的强说法。layer27_t0.75 在 last-answer metric 上仍有 `0.600`，但这个指标弱，因为它允许模型在非规范 answer marker 后继续漂移。

## 6. Step 3：Expanded paired statistics

文件：

- `src/paired_stats.py`
- `tests/test_paired_stats.py`
- `docs/step_03_expanded_statistics.md`
- `outputs/stop_after_final_stats_n5.json`
- `outputs/stop_after_final_stats_n5.md`

统计方法：

- Wilson 95% CI for accuracy。
- Paired bootstrap 95% CI for difference vs baseline。
- Exact McNemar-style paired discordance counts。

First-final metric：

| condition | n | acc | Wilson 95% CI | diff vs baseline | bootstrap 95% CI | McNemar p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 5 | 0.000 | [0.000, 0.434] | +0.000 | [+0.000, +0.000] | 1.000 |
| layer18_t1 | 5 | 0.000 | [0.000, 0.434] | +0.000 | [+0.000, +0.000] | 1.000 |
| layer27_t0.75 | 5 | 0.000 | [0.000, 0.434] | +0.000 | [+0.000, +0.000] | 1.000 |
| layer27_t1 | 5 | 0.000 | [0.000, 0.434] | +0.000 | [+0.000, +0.000] | 1.000 |

Last-answer metric：

| condition | n | acc | diff vs baseline | bootstrap 95% CI | b-wrong/c-right | b-right/c-wrong | McNemar p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| layer18_t1 | 5 | 0.000 | -0.400 | [-0.800, +0.000] | 0 | 2 | 0.500 |
| layer27_t0.75 | 5 | 0.600 | +0.200 | [+0.000, +0.600] | 1 | 0 | 1.000 |
| layer27_t1 | 5 | 0.200 | -0.200 | [-0.600, +0.000] | 0 | 1 | 1.000 |

解释：n=5 太小，不能证明任何稳定 effect size；但它足以证明当前 first-final utility claim 不成立。

## 7. 因果解释边界

当前可以说：

- Late decoder-block states causally control continuation/finalization posture.
- Hook-captured patching is the safer interpretation path after site audit.
- Interventions affect generation dynamics and format/length/repetition behavior.
- Layer 27 remains interesting, but old layer27 hidden-state results require caution.

当前不应该说：

- 找到了“思考神经元”。
- Layer 27 patch 稳定提升 reasoning accuracy。
- Continue-final gap 等价于 useful reasoning。
- Last extracted answer accuracy 等价于 first committed answer correctness。
- n=5 heldout 数字具有统计显著性。

## 8. 最有价值的科学问题

当前最有价值的问题已经从“能不能让模型多想”变成：

> 为什么 process-posture control 没有转化为 correct answer commitment？

可能机制包括：

1. **Format-following failure**：模型会算对，但不用 `Final answer:` 提交。
2. **Task/process entanglement**：patch state 同时携带题目内容和过程姿态，跨题迁移会污染答案。
3. **Readout boundary artifact**：late layer 尤其 layer 27 可能靠近 lm_head，干预更像改变输出模式而非计算过程。
4. **Metric mismatch**：continue tokens 不等于 useful computation tokens。
5. **Prompt mismatch**：严格 prompt 与模型偏好的 `Answer:` 格式冲突，导致 first-final metric 严苛但产品上合理。

## 9. 下一步路线

### Step 4：Task-out and verbosity controls

必须加入非推理控制任务：

- capital lookup；
- translation；
- sentiment rewrite；
- pure formatting；
- verbose-but-no-step-by-step explanation。

目标：证明干预不是一般 verbosity vector，也不是简单提高 `Answer:`/format token 倾向。

### Step 5：Data-driven token metric

替换手工 continue/final token set：

- 在 train split 上学习 high-vs-low discriminative token clusters；
- 在 heldout 上评估；
- 加入 KL、answer-token margin、EOS/Final margin。

目标：避免 `The` 等泛化 token 污染 effort metric。

### Step 6：Process/task disentanglement

做 cross-question patch：

```text
source question A high-process state → target question B low-process run
```

观察：

- 是否只改变 B 的 posture；
- 是否泄漏 A 的数字或答案；
- mean process direction 是否比 single-sample state 更少泄漏 task content。

### Step 7：CET controller

构造 token-level controller：

- 读 effort/final/repetition signals；
- valid final 后停止；
- repetition 上升时降低干预；
- format failure 时切换到 final-answer forcing 或 prompt repair。

目标：把局部 posture control 转化为可用生成策略。

## 10. 当前报告结论

本仓库现在最有价值的资产不是一个漂亮的 accuracy 表，而是一条逐步收紧的因果审计链：

1. 找到了 early steering signal。
2. 用随机方向发现早层方向特异性不足。
3. 用 activation patching 发现 late-layer state 更强。
4. 用 interpolation 证明 late state 有连续控制特征。
5. 用 site audit 发现 layer 27 旧解释存在 site mismatch。
6. 用 stop-after-first-final 推翻了旧的过强 utility claim。
7. 用 paired statistics 把小样本不确定性显式化。

这是一种更可靠的研究进展：它没有把 weak positive 夸大成发现，而是通过因果干预和反证测试把问题推进到更清楚的位置。

最终当前结论：

> Effort-circuit 目前证明了 Qwen3-0.6B 存在可被干预的 late-layer generation-posture state；尚未证明该 state 单独足以产生可靠的正确答案提交。下一阶段应围绕 task/process 解耦、格式控制、data-driven metrics 和 token-level controller 展开。
