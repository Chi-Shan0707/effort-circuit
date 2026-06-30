我读完这个仓库后，判断是：

**它已经有一个挺有趣的早期机制信号，但还没有到“显著发现 / 可写论文结论”的强度。**
最值得保留的发现不是“找到思考神经元”，而是更微妙的东西：

> 在 Qwen3-0.6B 的 synthetic arithmetic 设置里，靠近输出端的 late residual state，尤其是 layer 27，似乎能作为一种“reasoning / continue posture”控制变量：把 verify/high-effort prompt 的后层状态 patch 到 instant/strict prompt 上，会显著改变下一 token 的 continue-vs-final 倾向，并在极小样本生成实验里提高算术答案正确率。

但我会很小心：目前它更像 **strong lead**，不是 **settled claim**。

---

## 1. 仓库实际做了什么？

这个 repo 不是泛泛写了个框架，它确实跑了几轮实验。README 里写得很清楚：目标是“identifying and causally intervening on internal activations that increase useful reasoning effort”，默认模型是本地 `Qwen3-0.6B`，并且把综合结果放在 `docs/final_analysis_report.md`。仓库 pipeline 包括 prompt steering、activation patching、activation interpolation、strict generation prefix patch、CET controller 等模块。([GitHub][1])

它的第一性原理框架也对：不先假设“思考神经元”，而是从自回归模型的基本机制出发：hidden state 产生 logits，logits 决定下一 token 分布；“继续推理”和“立即作答”首先应该体现为下一 token 分布差异；如果某个内部方向有因果作用，沿它干预 hidden state 应该可重复改变分布。([GitHub][2])

实验环境很受限：虽然 Linux 能看到 240 逻辑 CPU，但容器实际 cgroup quota 约 4 cores、8 GiB memory；Qwen3-0.6B CPU prefill benchmark 里 4 threads 最快，8 threads 反而因为 oversubscription 变慢。这个限制解释了为什么目前样本量很小。([GitHub][3])

---

## 2. 它分析出了什么？

第一阶段是 **prompt-level steering**。它用 `verify` prompt 和 `instant` prompt 的 hidden-state 差分，去干预 neutral `cot` prompt 的下一 token 分布。指标是：

```text
continue-final gap = P(continue token set) - P(final token set)
```

continue token set 包括 `First`, `Let's`, `We`, `Since`, `Calculate`, `Step`, `Wait`, `The`；final token set 包括 `Final`, `Answer`, `Therefore`, `Thus`, `So`。实现里就是取这些字符串的 first-token id，算 softmax mass。([GitHub][4])

结果是：n=8 all-layer run 里，baseline gap 是 `0.377977`；最强正向变化是 layer 4、alpha = -3，gap 提高 `+0.115317`；最强负向变化是 layer 7、alpha = +3，gap 降低 `-0.0776305`。文档自己也很谨慎地说，这只是局部 next-token distribution 的因果结果，不是 accuracy 或 reasoning length 结论。([GitHub][5])

第二阶段是 **random direction control + short generation**。结论有点扎心但重要：layer 4 的 alpha=-3 候选方向确实超过 32/32 个随机方向，但最强随机方向非常接近；也就是说，这个 token-mass metric 很容易被一般 logit 扰动影响。仓库自己也承认，这不能叫 effort circuit，只能叫 early-stage steering candidate。([GitHub][6])

第三阶段才是最有价值的：**activation patching**。设置是：

```text
low / corrupt run = instant prompt
high / clean run = verify prompt

low→high:
  把 verify 的某层最后 token hidden state patch 到 instant run

high→low:
  把 instant 的某层最后 token hidden state patch 到 verify run
```

如果某层真的包含“继续推理姿态”，那 low→high 应该提高 continue gap，high→low 应该降低 continue gap。结果很强：全层 patching 中，layer 27 的 low→high patch 把 gap 提高 `+0.734238`，layer 18 的 high→low patch 把 gap 降低 `-0.317724`；后层 18–27 整体都很强。([GitHub][7])

第四阶段是 **late-layer interpolation**。它不是 hard patch，而是做：

```text
h(t) = (1 - t) h_low + t h_high
```

在 layers 18、20、23、27 上，`t = 0 → 1` 都单调提高 continue-final gap。layer 27 最强，delta 是 `+0.734243`；而且曲线高度非线性，t=0.5 时 gap 只有 `0.01977`，t=0.75 到 `0.28358`，t=1.0 到 `0.73434`。这个结果比“某个 alpha 偶然有效”更有机制味，因为它表现为连续可控变量。([GitHub][8])

第五阶段是 **strict generation prefix patch**。它用严格格式：

```text
You must use exactly this format:
Reasoning:
Final answer:

Question: ...
Reasoning:
```

然后只在 prefill 最后位置做一次 prefix patch，再让模型生成。train-small n=3 上，baseline accuracy 是 `0.333`，layer27_t0.75 和 layer27_t1 都是 `1.000`；heldout-small n=5 上，baseline 是 `0.400`，layer27_t0.75 是 `0.600`，layer27_t1 是 `0.800`。同时 repetition rate 下降，这让“只是变啰嗦”这个解释弱了一点。([GitHub][9])

所以仓库目前最可信的内部结论是：**late residual stream 中存在一种“输出姿态 / reasoning posture”状态，它能直接影响下一 token 是继续结构化推理，还是进入 finalization / repetition。** 仓库最终报告也基本这么写：不是孤立神经元，而是后层残差状态里的分布式变量；layer 27 很可能靠近 logits/readout，layers 18–24 可能是这个 posture 形成和稳定的区域。([GitHub][6])

---

## 3. 我认为最有趣的发现是什么？

最有趣的不是 layer 4 direction，也不是“accuracy 0.4 → 0.8”这个小样本数字。真正有意思的是这三点。

第一，**late-layer state 的双向因果性很清楚**。low→high patch 提高 continue posture，high→low patch 压低 continue posture，而且效果集中在后层。这比单纯 probe 或 activation addition 更可信，因为它用了真实 activation state，而不是随便加一个向量。([GitHub][6])

第二，**layer 27 interpolation 有“阈值 / 相变”味道**。t=0.5 很弱，t=0.75 突然明显，t=1 极强。这很像模型在 residual space 里存在一个非线性 readout boundary：跨过某个区域后，下游 logits 从“该答了”切到“继续推理”。这可以发展成一个更有趣的问题：reasoning effort 是不是一个离散-ish 的 policy posture，而不是线性标量？

第三，**它可能同时降低 repetition**。strict generation 里，layer27_t1 不只是更长，而是 repetition rate 比 baseline 低。train-small 里 baseline repetition `0.7260`，layer27_t1 `0.2200`；heldout-small 里 baseline `0.2783`，layer27_t1 `0.2025`。这说明它可能不是普通 verbosity vector，而是更接近“结构化继续 / 避免坏循环”的输出姿态。([GitHub][9])

如果我要给这个发现取个更准确的名字，我不会叫它 “effort neuron”。我会叫它：

**Late-Layer Reasoning Posture Transfer**
或者更抽象一点：

**Final-token policy posture control**

也就是：不是“模型多会算了”，而是“后层 residual state 让模型选择进入哪种生成策略：直接答、重复、结构化推理、验证、finalize”。

---

## 4. 但现在还不够强，甚至有几个关键风险

这里要非常诚实。这个 repo 的结果有趣，但还有明显硬伤。

第一个问题：**样本量太小**。train n=3，heldout n=5。baseline 2/5 correct 到 layer27_t1 4/5 correct，看起来是 0.4→0.8，但 Fisher exact one-sided 也不显著。它是探索信号，不是统计结论。仓库自己也承认目前是 small-n evidence，需要扩大验证。([GitHub][6])

第二个问题：**strict generation 里所有 condition 的 mean_reasoning_tokens 都是 96**，因为 `max_new_tokens=96`，很多生成其实用满预算。这意味着它还没有真正测出“reasoning length 被有效控制”，只是测出“在固定长生成里，输出内容和最终抽取答案发生变化”。heldout summary 里 baseline、layer27_t0.75、layer27_t1 的 mean_reasoning_tokens 全都是 96。([GitHub][10])

第三个问题：**答案抽取方式可能奖励“多次 Final answer 直到最后一次碰对”**。仓库已经修正了抽取器，让它优先取最后一个 Final/Answer marker 后该行最后一个数字；测试里也加入了 `Final answer: 54 - 5 = 49` 应抽 49、多次 Final answer 时取最后一次。([GitHub][11]) 这个修正合理，但也带来一个实验定义问题：真实产品里模型通常一旦输出第一个 `Final answer` 就该停止；如果一个 completion 先输出错答案，后面又继续 reasoning，最后抽取器取最后一个答案，那它测到的是“无限补救轨迹”，不是“第一次交卷正确率”。final report 也已经把 stop-after-final 列为最高优先级下一步。([GitHub][6])

第四个问题：**continue-final gap 指标还太粗**。continue token 里有 `The`，这是非常泛化的英语开头 token；final token set 又很小。layer 4 随机方向几乎追平候选方向，说明这个指标确实容易被非特异性 logit shift 操纵。([GitHub][6])

第五个问题，也是我觉得最重要的：**activation site identity 需要严肃审计**。代码里 `hidden_last_by_layer` 从 `outputs.hidden_states[1:]` 取状态，而 patching 是 hook 到 `get_layers(model)[layer]` 的 module output。([GitHub][4]) 这两者在 HuggingFace 模型里不一定严格是同一个 site；官方文档也提醒，`hidden_states[-1]` 和最终返回的 last hidden state 不一定总是完全匹配，因为有些模型会额外做 normalization 或后处理。([Hugging Face][12])

这点不是小洁癖。layer 27 的 low→high patch gap `0.734` 甚至大于原始 verify high gap `0.307`，这很可疑：如果你在最后一层把 high prompt 的同一 site hidden state patch 到 low prompt，而后面只剩 final norm/lm_head，理论上它应该更接近 high logits，而不是远超 high。这个现象可能意味着：

1. patch site 和 cached hidden state site 不完全一致；
2. final norm / module output 边界有错位；
3. prompt state 被 off-manifold 地注入了；
4. 或者确实存在一个很有趣的 context/site interaction。

无论哪种，都必须让 Codex 先做 **site-equivalence test**。否则 layer 27 的最强结论有可能是“把 final-normalized state 塞进 decoder block output”，这就不是干净的机制发现，而是强力 readout hack。

---

## 5. 前人有没有类似发现？

有，而且不少。所以这个 repo 的结果不能说“前人完全没做过”。但它仍然可以发展出自己的角度。

最直接相关的是 **Activation Control for Efficiently Eliciting Long CoT Ability**。这篇 2025 arXiv 论文说，少量 high-impact activations，尤其在最后几层，很大程度上 governs long-form reasoning attributes，比如输出长度和 self-reflection；放大这些 activations 并加入 “wait” tokens，可以无训练地唤起 long-CoT 能力，提高 self-reflection rate 和 accuracy。它还发现这些 activation 变化有可预测轨迹：特殊 token 后急升，随后指数衰减。([arXiv][13]) 这和 repo 的 late-layer posture / layer 27 非线性非常接近。

另一个很近的是 **AdaRAS / Reasoning-Critical Neurons**。这篇 2026 arXiv 论文声称，一小部分 MLP neurons 的 activation 和 reasoning correctness 强相关，并提出 test-time activation steering 框架，只在轨迹可能错误时干预 reasoning-critical neurons。([arXiv][14]) 这和你最初的“有没有一组神经元控制 effort”非常贴近，但它关注的是 correctness-critical neurons，而 repo 目前关注的是 residual posture 和 continue/final policy。

还有 EMNLP 2025 的 **Enhancing Chain-of-Thought Reasoning via Neuron Activation Differential Analysis**。它通过比较不同质量 reasoning chains 的 FFN neuron activation，识别 reasoning-critical neurons，并直接刺激这些 neuron 来引导高质量 CoT。([ACL Anthology][15]) 所以“通过 neuron activation 提升 CoT”这条路已经有人做，repo 若要有新意，最好不要只复现 neuron ranking，而要做 trajectory/control/persistence 方向。

再往 activation steering 总方法论看，CAA 已经是经典路线：通过 positive/negative contrastive examples 的 residual stream activation 差值构造 steering vector，并在 inference forward pass 中加这个 vector 来控制行为。([arXiv][16]) Anthropic 的 Golden Gate Claude 也公开展示过：他们在 Claude 里找到可解释 feature，可以调高/调低其 activation 强度，并观察对应行为变化；他们强调这不是 prompt，也不是 fine-tuning，而是对内部 activation 的 surgical change。([Anthropic][17])

还有两类论文和 repo 的 patching 思路很近。**When Chain-of-Thought Fails, the Solution Hides in the Hidden States** 会把 CoT prompt 的 residual hidden state patch 到 direct-answer target prompt 上，看是否恢复正确答案；它明确把 direct-answer prompt 当 target，把 CoT reasoning prompt 当 source，然后在指定 layer 和 final token position 做 activation patching。([arXiv][18]) **How does Chain of Thought Think?** 用 SAE + activation patching 研究 CoT faithfulness，也提醒最终 token residual patching 是静态快照，未来需要 token-level/path-based causal analysis。([arXiv][19])

所以我对新意的判断是：

**已知领域：activation steering、CoT neurons、late-layer activation control、CoT-to-direct hidden-state patching，都已有前人。**

**这个 repo 的潜在新意：把 effort 看成“生成动力学里的 posture / attractor / controller”，不是只找一个 vector 或 neuron；并且在小模型、CPU、合成任务上建立一个从 local logits → monotonic interpolation → trajectory → utility 的低成本因果 gate pipeline。**

这条如果走下去，是有意思的。

---

## 6. 第一性原理下，我会怎么重新表述这个问题？

不要问：

```text
模型有没有思考神经元？
```

更好的问题是：

```text
在生成某个 token 前，模型处于哪种 policy state？
这个 state 是倾向继续计算、验证、重复、还是提交答案？
这个 state 能否被低维变量控制？
这个控制是否保留 task content，而只改变 process posture？
```

从 decoder 角度看，最后 token 的 residual state 有两类信息混在一起：

```text
task state:
  题目数字、中间计算、候选答案、语义内容

process state:
  现在该继续、该检查、该输出 Final、还是陷入重复
```

这个 repo 当前可能主要抓到了 **process state**，尤其是“继续结构化推理 vs 低质量 finalize/repeat”的 process posture。真正下一步应该做的是把 task state 和 process state 解耦。

一个很漂亮的实验是：

```text
同一道题：
  instant state + verify process patch → 是否更会继续算？

不同题：
  题 A 的 verify process state patch 到题 B → 是否只改变题 B 的推理姿态，而不泄漏题 A 的答案？

同题不同过程：
  verbose non-reasoning state vs verify reasoning state → 能否区分“啰嗦”和“有效推理”？
```

如果 layer 27 patch 从题 A 迁移到题 B 只改变 posture，不改变具体答案内容，那它更像 process control。
如果它把题 A 的数字/答案也带过去，那它混入 task content。这个区别非常关键。

---

## 7. Codex 下一步最该做什么？

我会让 Codex 先别急着扩大到 SAE/neuron。第一优先级是把实验定义弄干净。

### Step 1：做 activation-site audit

写一个 `src/site_audit.py`，测试三件事：

```text
1. same-run self patch:
   对同一个 prompt，在 layer L 用它自己的 hook-captured output patch 回去。
   logits delta 应接近 0。

2. hidden_states-to-hook patch:
   用 outputs.hidden_states 的 layer L state patch 到 hook site L。
   如果 logits delta 不接近 0，说明 hidden_states index 和 hook site 不一致。

3. high prompt self-equivalence:
   high_logits 应该等于把 high captured state patch 到 high run 后的 logits。
```

尤其要检查 layer 27。现在 layer27_t1 gap 远超原始 high gap，这个 audit 必须先过。过不了，就先修 index/site，不要继续堆实验。

### Step 2：把 strict generation 改成 stop-after-first-final

当前所有 condition 都跑满 96 tokens，这会污染 reasoning length、repetition 和 correctness。实现：

```text
逐 token generate
一旦出现合法行：
  Final answer: <number>
立即停止
记录 first_final_answer
```

同时报告：

```text
first-final accuracy
last-final accuracy
tokens_to_first_final
answer_change_count
post-final continuation rate
repetition before first final
```

如果 layer27_t1 在 **first-final accuracy** 上仍然从 0.4 提到 0.8，那才是真正强很多。

### Step 3：heldout 从 n=5 扩到 n=50 或 n=100

不需要一口气上千。先做：

```text
heldout_synthetic_math n=50
conditions:
  baseline
  layer27_t0.6
  layer27_t0.7
  layer27_t0.75
  layer27_t0.8
  layer27_t0.9
  layer27_t1.0
```

报告 Wilson CI、paired bootstrap、McNemar 或 exact test。现在 2/5 到 4/5 是好苗头，但不是统计结论。

### Step 4：做 task-out 和 verbosity controls

加入四类对照：

```text
reasoning:
  synthetic arithmetic, date arithmetic, simple symbolic tasks

non-reasoning:
  capital lookup, translation, sentiment rewrite

verbosity:
  "Explain verbosely but do not solve step by step"

format:
  同样包含 Reasoning/Final answer 格式，但不要求 verify
```

最重要的对照是：

```text
high_reasoning = verify arithmetic
high_verbosity = verbose explanation
high_format = same format but no reasoning instruction
```

你要证明 layer 27 不是 “English continuation / verbose / Reasoning marker” 方向，而是更接近 useful reasoning posture。

### Step 5：把 hand-coded token set 换成数据驱动 metric

现在 continue set 有 `The`，风险很大。Codex 可以从实际 high/low prompt 的 next-token 分布中自动抽取 discriminative tokens：

```text
score(token) =
  mean_logprob_high(token) - mean_logprob_low(token)

continue_cluster = top tokens high > low
final_cluster = top tokens low > high
```

然后用 heldout token clusters 评估，不要在同一批 n=8 上既选 token 又报告效果。再补充 KL divergence、answer-token margin、EOS/Final margin。

### Step 6：分离 process state 和 task state

做 cross-question patch：

```text
source question A, high verify state
target question B, strict prompt

patch source high process state into target B
看 B 的答案是否改善，是否泄漏 A 的数字/答案
```

为了只转移 process posture，可以把 source/target 的 state 做均值方向：

```text
v_process = mean(h_verify - h_instant over many questions)
```

然后和单样本 high state patch 比较。若 mean direction 有效且不泄漏内容，它更接近泛化控制变量。

### Step 7：从 prefix patch 进化成 CET controller

当前 prefix patch 是一次性外科手术。真正有趣的是闭环控制：

```text
for each generated token:
  read effort_score / final_margin / repetition_score

  if model is trying to Final too early:
      push toward reasoning posture

  if model has valid final answer:
      stop

  if repetition rising:
      reduce steering or push away from repetition mode
```

这和前人 EELo-CoT 的“wait token 后 activation 急升再衰减”可以很好结合：你可以拟合 layer27 effort posture 的 decay curve，然后让 controller 只在 posture 衰减时补一点，而不是全程暴力 patch。EELo-CoT 明确提到 long-CoT 相关 activation 在最后几层，并有特殊 token 后急升、随后指数衰减的模式。([arXiv][13])

---

## 8. 更有野心的新方法论：Posture Algebra

如果你想让这个项目真的有 “new perspective”，我建议不要只做 “find neurons”。可以做一个更抽象也更漂亮的方向：

**Posture Algebra：把生成策略拆成可组合的 latent process states。**

构造一个 factorial contrast dataset：

```text
axis 1: instant vs verify
axis 2: concise vs verbose
axis 3: correct vs incorrect
axis 4: continue vs finalize
axis 5: structured reasoning vs repetition
```

然后做线性分解：

```text
h = task_content + effort_posture + verbosity_posture + finalization_posture + repetition_posture + noise
```

你要找的不是一个 vector，而是一组可组合控制：

```text
+effort_posture
-verbosity_posture
-repetition_posture
+finalize_when_answer_stable
```

这比“让模型多想”高级很多。因为 reasoning 模型真正需要的是：

```text
该继续时继续
该检查时检查
该停止时停止
避免重复
避免先错后圆
```

如果 Codex 能实现这个，论文味会明显更强。前人有 activation steering 和 CoT neuron，但 “process posture algebra + closed-loop controller + first-final correctness” 这个组合会更有新意。

---

## 9. 我的最终判断

这个仓库现在已经做出了一个值得继续追的发现：

**Qwen3-0.6B 的 late-layer residual state，尤其 layer 27，能强力控制 continue-vs-final 的输出姿态，并在极小样本 strict arithmetic generation 中显示出 accuracy 改善和 repetition 降低。**

但它还不能被称为稳健显著发现，原因是：n 太小、first-final 未停止、所有 strict generation 都打满 token budget、continue token metric 较粗、task-out controls 缺失，并且 activation site identity 需要 audit。

我会把下一轮目标定成一句话：

> 证明 layer 27 不是 off-manifold readout hack，也不是 verbosity vector，而是一个可泛化、可闭环控制的 reasoning process posture。

Codex 的最高优先级不是继续加复杂功能，而是：**site audit → stop-after-first-final → n=50 heldout → task-out controls → process/task disentanglement → CET controller。**

如果这六步跑通，这个项目就从“有趣小实验”升级成一个真正有价值的 mechanistic control story。

[1]: https://github.com/Chi-Shan0707/effort-circuit "GitHub - Chi-Shan0707/effort-circuit · GitHub"
[2]: https://github.com/Chi-Shan0707/effort-circuit/blob/main/FIRST_PRINCIPLES.md "effort-circuit/FIRST_PRINCIPLES.md at main · Chi-Shan0707/effort-circuit · GitHub"
[3]: https://raw.githubusercontent.com/Chi-Shan0707/effort-circuit/main/docs/resource_profile.md "raw.githubusercontent.com"
[4]: https://raw.githubusercontent.com/Chi-Shan0707/effort-circuit/main/src/prompt_steering_experiment.py "raw.githubusercontent.com"
[5]: https://raw.githubusercontent.com/Chi-Shan0707/effort-circuit/main/docs/experiment_001_prompt_steering.md "raw.githubusercontent.com"
[6]: https://raw.githubusercontent.com/Chi-Shan0707/effort-circuit/main/docs/final_analysis_report.md "raw.githubusercontent.com"
[7]: https://raw.githubusercontent.com/Chi-Shan0707/effort-circuit/main/outputs/activation_patching_instant_verify_n8_all_layers.md "raw.githubusercontent.com"
[8]: https://raw.githubusercontent.com/Chi-Shan0707/effort-circuit/main/outputs/activation_interpolation_instant_verify_n8_late_layers.md "raw.githubusercontent.com"
[9]: https://raw.githubusercontent.com/Chi-Shan0707/effort-circuit/main/docs/experiment_004_multistage_late_layer_intervention.md "raw.githubusercontent.com"
[10]: https://raw.githubusercontent.com/Chi-Shan0707/effort-circuit/main/outputs/strict_generation_prefix_patch_heldout_n5_layer27_reextracted.json "raw.githubusercontent.com"
[11]: https://raw.githubusercontent.com/Chi-Shan0707/effort-circuit/main/src/answer_extractors.py "raw.githubusercontent.com"
[12]: https://huggingface.co/docs/transformers/en/main_classes/output?utm_source=chatgpt.com "Model outputs"
[13]: https://arxiv.org/html/2505.17697v1 "Activation Control for Efficiently Eliciting Long Chain-of-thought Ability of Language Models"
[14]: https://arxiv.org/html/2601.19847v1 "Identifying and Transferring Reasoning-Critical Neurons: Improving LLM Inference Reliability via Activation Steering"
[15]: https://aclanthology.org/2025.emnlp-main.817/ "Enhancing Chain-of-Thought Reasoning via Neuron Activation Differential Analysis - ACL Anthology"
[16]: https://arxiv.org/abs/2312.06681 "[2312.06681] Steering Llama 2 via Contrastive Activation Addition"
[17]: https://www.anthropic.com/news/golden-gate-claude "Golden Gate Claude \ Anthropic"
[18]: https://arxiv.org/html/2604.23351v1 "When Chain-of-Thought Fails, the Solution Hides in the Hidden States"
[19]: https://arxiv.org/html/2507.22928v1 "How does Chain of Thought Think? Mechanistic Interpretability of Chain-of-Thought Reasoning with Sparse Autoencoding"
