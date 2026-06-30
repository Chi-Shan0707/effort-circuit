可以，而且这个方向真挺有意思。我的建议是：**不要把项目目标写成“找到一个思考神经元”**，而是写成：

> 找到一组可因果操纵的内部状态，使 decoder 在生成过程中更倾向于维持“继续推理/检查/延迟作答”的自激轨道，并且这种延长不是纯啰嗦，而是在一类任务上提高正确率。

这样更科学，也更容易出结果。因为现在的证据更支持“feature / direction / circuit / distributed latent”，而不是一个干净的单神经元按钮。Anthropic 的 Golden Gate Claude 例子就很典型：他们说找到的是 Claude 内部某个概念对应的“特定神经元组合”，并且可以调高/调低这个 feature 的激活来改变行为；他们也强调这不是 system prompt，也不是 fine-tuning，而是对内部 activation 的干预。([Anthropic][1])

我会把你的实验设计成一个完整项目，名字可以叫：

**CET：Causal Effort Thermostat，因果 effort 恒温器。**

核心新视角是：
**reasoning effort 不是静态属性，而是一个生成轨迹中的“持久模式”。我们不只是找高激活神经元，而是找能让模型留在“继续推理 attractor”里的方向/回路，然后用闭环控制维持它，最后再用正确率和 overthinking 指标约束它。**

---

## 0. 先定三个假设

第一条假设：**存在一个“继续推理 vs 现在作答”的内部 latent direction。**
它可能在 residual stream，也可能在 MLP features，也可能分布在若干层。激活它后，模型更倾向生成 “First… / Then… / Wait… / Let’s check…” 这类继续推理 token，而不是马上输出 final answer。

第二条假设：**这个 latent 可以通过 activation steering 操纵。**
这有现成文献基础。ActAdd/CAA 一类方法就是通过对比两类 prompt 或两类行为的 activation 差值，得到 steering vector，然后在 inference 时把这个 vector 加到隐藏状态里。CAA 论文明确说它通过“positive vs negative examples”的 residual activation 差值计算 steering vectors，并在推理时加入这些 vectors 来控制行为。([arXiv][2]) ActAdd 论文也报告了无需优化、无需 finetune 的 activation addition 能改变模型行为。([arXiv][3])

第三条假设：**更长 reasoning 只在“可被多想修正”的问题上有用。**
所以你的目标不能是无脑拉长输出，而是提高 “useful effort”。s1 的 budget forcing 论文就是一个关键参考：它通过在模型想结束时追加 “Wait” 来延长 thinking，常常触发 double-check 并修正错误。([arXiv][4]) 但后续 overthinking 研究也提醒，更多 token 的边际收益会递减，甚至会让模型放弃原本正确答案。([arXiv][5])

所以最终目标不是：

```text
让模型说更多话
```

而是：

```text
让模型在该多想的问题上多想，在容易过想的问题上停下。
```

这就是我建议做 CET 的原因。

---

## 1. 实验总体架构

我会让 Codex 实现四条线，并且每条线都能独立产出结果。

第一条线：**behavior baseline**
先测模型在 instant、CoT prompt、budget forcing、self-consistency 下的长度和正确率。没有这个 baseline，后面干预没法解释。

第二条线：**发现 effort direction**
用 contrastive activation difference、linear probe、SVD/PCA 找出“高 effort / 好 reasoning”方向。

第三条线：**下钻到 neuron / SAE feature / circuit**
用 MLP neuron scoring、SAE feature extraction、activation patching 证明哪些内部组件有因果作用。Anthropic 的 circuit tracing 方法论本质上也是：先用 attribution graph 产生机制假设，再用 perturbation 实验验证；他们明确说 attribution graphs 生成机制假设，并通过后续 perturbation 实验测试和修正。([Transformer Circuits][6])

第四条线：**CET 闭环控制**
不是固定加一个 vector，而是在每个 token 生成时读取 effort score，如果模型快要 premature answer，就推它回 reasoning attractor；如果进入 overthinking 区域，就降 steering 或提前停止。

---

## 2. 数据集：不要一开始用太难的数学

你的模型只有 0.6B，所以最重要的是选 **zone of proximal development**：模型不是全会，也不是全不会。理想 baseline accuracy 在 30% 到 70% 之间。低于 20%，多想也可能只是胡扯；高于 85%，多想容易 overthinking。

我建议三类任务。

第一类是可程序验证的 synthetic reasoning：

```text
两位数/三位数加减乘，尤其带进位
简单方程求解
模运算
日期/星期推理
括号匹配
小型布尔电路求值
短 permutation composition
```

第二类是简单自然语言数学：

```text
小学应用题
价格/折扣/数量问题
年龄问题
距离速度时间问题
```

第三类是对照任务：

```text
事实问答
翻译
情绪改写
简单常识
```

对照任务很重要，因为你要证明你的 steering 不是单纯让模型变啰嗦。如果 reasoning direction 在事实问答上也疯狂拉长，而且正确率不升，那说明它只是 verbosity vector，不是 useful effort vector。

每条样本都强制格式化：

```text
Question: ...
Reasoning:
...
Final: <answer>
```

这样你可以精确统计 `Reasoning:` 到 `Final:` 之间的 token 数，并用程序抽取答案。

---

## 3. 先跑 baseline

让 Codex 实现四种 generation mode：

```text
instant:
  "Answer directly. Final:"

cot:
  "Think step by step. Put reasoning under Reasoning, final answer under Final."

verify:
  "Think step by step, then check your answer before Final."

budget_forcing:
  如果模型提前输出 Final，就追加 "\nWait, check again.\n" 继续生成，最多追加 N 次。
```

记录这些指标：

```text
accuracy
reasoning_tokens
total_tokens
time_to_final
first_final_position
answer_changes
positive_flips: short wrong → long correct
negative_flips: short correct → long wrong
cost_adjusted_score = accuracy - λ * reasoning_tokens
```

这里的 positive/negative flip 特别关键。overthinking 论文就是沿着不同 compute budget 跟踪答案变化，定义 flip events，并把 correct→incorrect 看作 negative flip。([arXiv][5])

这一步的目标是找到一个事实：

```text
在某些任务上，cot / verify / budget_forcing 比 instant 更准。
```

只有存在这个行为差异，后面才值得找内部 circuit。

---

## 4. 发现 effort direction：三种方法并行做

### 方法 A：contrastive activation difference

对同一道题，构造一对 prompt：

```text
negative / low-effort:
  Answer directly. Final:

positive / high-effort:
  Think step by step. Check your work before Final.
```

对每个 layer，在 prompt 最后一个 token 的 residual stream 上取 activation：

```text
h_pos[layer] = high-effort prompt activation
h_neg[layer] = instant prompt activation
v_effort[layer] = mean(h_pos[layer] - h_neg[layer])
```

然后在 neutral prompt 上测试：

```text
h[layer] ← h[layer] + α * normalize(v_effort[layer])
```

α 做 sweep：

```text
α ∈ {-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3}
```

如果正向 α 增加 reasoning length，负向 α 缩短 reasoning length，而且中等 α 提高正确率，你就有第一版 causal evidence。

### 方法 B：probe direction

从 activation 预测三个 label：

```text
label_1: 这条 trace 是否会超过 K 个 reasoning tokens
label_2: 这条 trace 是否最终正确
label_3: 这道题是否属于 positive flip，即多想后修正错误
```

用 logistic regression 或 linear SVM：

```text
score = wᵀh + b
```

如果 probe 在 held-out task 上 AUROC 明显高于随机，`w` 就是候选 direction。然后不要停在 probe，必须把 `w` 拿去 steering。只做 probe 是相关性；steering 才是因果。

### 方法 C：paired-difference SVD / SAE-free steering

把 paired difference 堆成矩阵：

```text
A = [h_pos_1 - h_neg_1,
     h_pos_2 - h_neg_2,
     ...
     h_pos_N - h_neg_N]
```

做 SVD 或 eigen decomposition：

```text
A Aᵀ 的 top eigenvector = 主 steering direction
```

这和 2025 EMNLP 的 SAE-free CoT steering 很接近。那篇论文提出不依赖预训练 SAE，直接从 residual activations 计算 steering directions；它们用 paired activation differences 构造方向，并报告 SAE-based 和 SAE-free steering 都提升 reasoning 能力。

---

## 5. 找 neuron：但把 neuron 当作第二层证据，不要当第一层

如果你的小模型是 LLaMA/Qwen/Gemma 风格 gated MLP，那一个 “MLP neuron activation” 可以定义为：

```text
a = act(gate_proj(x)) * up_proj(x)
mlp_out = down_proj(a)
```

如果是 GPT-2/ReLU 风格：

```text
a = act(W_in x)
mlp_out = W_out a
```

对每个 layer/neuron，统计 high-quality trace 和 low-quality trace 的 activation 差异：

```text
mean_good[l,n]
mean_bad[l,n]
cohen_d[l,n] = (mean_good - mean_bad) / pooled_std
ratio[l,n] = (mean_good + eps) / (mean_bad + eps)
```

选 top-K neuron 时要用 stability selection：

```text
随机 split 数据 10 次
每次选 top neurons
保留出现频率 > 70% 的 neurons
```

然后做 causality test：

```text
增强:
  a[:, top_neurons] *= 1 + β

抑制:
  a[:, top_neurons] *= 1 - β

或 clamp:
  a[:, top_neurons] = percentile_good_75
```

β sweep：

```text
β ∈ {0.1, 0.25, 0.5, 1.0, 2.0}
```

这条线有直接文献参考。2025 EMNLP 的 “Enhancing Chain-of-Thought Reasoning via Neuron Activation Differential Analysis” 就是通过比较高质量和低质量 CoT 的 FFN neuron activation，找 reasoning-critical neurons，再调节这些 neuron 的 activation strength 来提高 CoT 质量；论文报告平均 2.4% relative improvement。

但这里要小心：**单神经元很可能 polysemantic**。所以 neuron 结果最好和 residual direction / SAE feature / ablation 结果互相交叉验证。

---

## 6. 训练 SAE：为了可解释性，不是为了炫技

如果你的 0.6B 模型没有现成 SAE，可以自己训练小 SAE。SAELens 正好是为训练和分析 sparse autoencoders 做的，支持 TransformerLens、HuggingFace、NNsight 等 PyTorch 模型。([GitHub][7])

建议只训练几个 layer，不要全模型铺开。先用 probe 找出最有信号的 2 到 4 层，再训练：

```text
activation source:
  resid_post[layer] 或 mlp_out[layer]

samples:
  1M 到 10M token activations 起步

SAE:
  expansion_factor = 8 或 16
  TopK SAE 或 L1 SAE
  target sparsity: 16 到 64 active features
```

找 feature 的方式：

```text
feature_score[j] =
  E[f_j | good_reasoning] - E[f_j | bad_reasoning]

或

feature_score[j] =
  corr(f_j, reasoning_tokens) + corr(f_j, correctness)
```

然后干预：

```text
h ← h + α * SAE_decoder_vector[j]
```

或者 group steering：

```text
h ← h + α * Σ_j score[j] * normalize(decoder[j])
```

SAE 这条线很 Anthropic 风格：他们的 sparse feature work 目标就是把模型 activation 分解成更可解释的 features；Golden Gate Claude 展示了调节 feature 强度能导致可见行为变化。([Anthropic][1]) 但 SAE 也不能迷信。2025 有论文指出，SAE 在随机初始化 transformer 上也能产生看似可解释的 latents，因此 SAE 结果必须通过 causal intervention 和 null model 对照验证。([arXiv][8])

所以你的论文/实验报告里要写死一句：

```text
We treat SAE features as hypotheses, not explanations, until they pass causal perturbation tests.
```

---

## 7. 真正“新”的部分：Reasoning Persistence Mode

这是我最想推你做的东西。一般 activation steering 只做静态对比：

```text
high effort activation - low effort activation
```

但你的直觉问的是：有没有一种回路让 decoder 的自激更持久？

所以我们可以把它建模成 **轨迹动力系统**。

对每条 generated reasoning trace，记录某一层 residual state：

```text
h_1, h_2, h_3, ..., h_T
```

先 PCA 降维到 64 或 128 维：

```text
z_t = PCA(h_t)
```

然后拟合一个简单动态模型：

```text
z_{t+1} ≈ A z_t
```

找 A 的 eigenmodes。我们要的不是最大 variance mode，而是：

```text
1. eigenvalue magnitude 接近 1：说明它持久，不容易衰减
2. mode activation 和 remaining_reasoning_length 正相关
3. mode activation 和 correct_after_long_reasoning 正相关
4. 这个 mode 被 "Wait / check again" 激活
5. 负向干预这个 mode 会让模型更快 Final
```

这个 mode 我会叫：

```text
RPM: Reasoning Persistence Mode
```

它比普通 steering vector 更贴合你的“自激持久”问题。因为它不是问“哪个方向区分 high vs low”，而是问：

```text
哪个方向在 autoregressive dynamics 里会自我维持？
```

干预时可以这样：

```text
h_t ← h_t + α * RPM
```

并测试三种注入方式：

```text
prompt-only:
  只在 prompt 最后一个 token 注入

early-generation:
  只在前 8 个 generated tokens 注入

continuous:
  每个 generated token 都注入，直到 Final 或 max budget
```

如果 prompt-only 就能让 reasoning length 变长，说明这个 direction 能启动 attractor。
如果必须 continuous 才能维持，说明它更像外部 forcing。
如果 early-generation 足够，说明模型确实会被推入一个自激轨道。

这会很有趣，因为它不是单纯复现 CAA 或 neuron scaling，而是研究 **decoder generation dynamics 的持久模式**。

---

## 8. CET：闭环 effort controller

静态 steering 的问题是：α 太小没用，α 太大 ramble。2025 的 CoT steering 论文也观察到 steering strength 过大时会导致输出混乱。

所以我建议最后做闭环控制：

```text
effort_score_t = w_effortᵀ h_t
answer_margin_t = logP(Final/Answer/EOS) - logP(continue_tokens)
```

控制器逻辑：

```text
如果 reasoning_tokens < min_budget
并且 answer_margin_t 很高
并且 effort_score_t 低于 target:
    注入 +α_t * effort_vector

如果 reasoning_tokens > soft_budget
并且 answer 已经稳定
或 verifier confidence 高:
    降低 α_t，甚至注入 -α_t * effort_vector
```

α_t 不固定，而是：

```text
α_t = clip(kp * (target_effort - effort_score_t), 0, α_max)
```

这就是 “thermostat”：
不是一直开暖气，而是温度低了才加热，温度够了就停。

你可以把它和四个 baseline 比：

```text
instant
CoT prompt
budget forcing: Wait
static activation steering
CET closed-loop steering
```

成功标准不是 CET 最长，而是 CET 在 cost-adjusted score 上最好：

```text
score = accuracy - λ * reasoning_tokens
```

---

## 9. 必做对照实验

这部分决定你的实验是不是可信。

第一，随机方向对照：

```text
随机采 100 个和 v_effort 同 norm 的方向
同 layer，同 α
看你的 direction 是否排在 top 5%
```

第二，反向 steering：

```text
+v_effort 应该 length ↑
-v_effort 应该 length ↓
```

如果正反都 length ↑，那你可能只是破坏了模型。

第三，matched verbosity control：

```text
找一个“啰嗦/长文本”方向
比较它和 effort direction
```

你的 effort direction 必须比 verbosity direction 更能提升 correctness。

第四，任务外对照：

```text
事实问答、翻译、摘要
```

如果这些任务上 length 大涨但 accuracy 不涨，CET 应该自动少干预。

第五，position ablation：

```text
只在 prompt last token 注入
只在 reasoning tokens 注入
只在 final answer 前注入
全程注入
```

第六，layer sweep：

```text
每层单独 steering
画出 layer × α 的 heatmap
```

第七，patching：

```text
clean run = high-effort prompt
corrupt run = instant prompt

把 clean activation patch 到 corrupt run
看是否恢复 longer reasoning / correct answer

反过来：
把 instant activation patch 到 high-effort run
看是否缩短 reasoning / 降低正确率
```

TransformerLens 和 NNsight 都适合做这个。TransformerLens 支持缓存内部 activations，并能在模型运行时编辑、替换 activation。([GitHub][9]) NNsight 也明确支持访问任意层 activation、修改 activation、做 causal effect 研究。([NNSight][10])

---

## 10. 给 Codex 的 repo 任务书

你可以直接把下面这段给 Codex。

```text
Build a research repo named effort-circuit for a local HuggingFace causal LM around 0.6B parameters.

Goal:
Identify and causally intervene on internal activations that increase useful reasoning effort:
longer reasoning when useful, higher accuracy on held-out reasoning tasks, and no uncontrolled verbosity.

Repo structure:
effort-circuit/
  configs/
    default.yaml
  src/
    datasets.py
    prompts.py
    generation.py
    answer_extractors.py
    metrics.py
    hooks.py
    collect_traces.py
    cache_activations.py
    discover_vectors.py
    discover_neurons.py
    train_sae.py
    intervene.py
    rpm.py
    cet_controller.py
    sweeps.py
    analyze.py
  tests/
    test_answer_extractors.py
    test_hooks.py
    test_metrics.py
  outputs/
  README.md

Implement these CLI commands:

1. Collect baseline traces:
python -m src.collect_traces \
  --model MODEL_NAME_OR_PATH \
  --dataset synthetic_math \
  --n 2000 \
  --modes instant,cot,verify,budget_forcing \
  --out outputs/traces.parquet

Each row must include:
problem_id, question, gold_answer, mode, prompt, completion,
extracted_answer, correct, reasoning_tokens, total_tokens,
first_final_position, seed, temperature, max_new_tokens.

2. Cache activations:
python -m src.cache_activations \
  --model MODEL_NAME_OR_PATH \
  --traces outputs/traces.parquet \
  --layers all \
  --activation-sites resid_post,mlp_act,mlp_out \
  --positions prompt_last,reasoning_all,pre_final \
  --out outputs/activations/

Store activations in memory-mapped arrays or sharded torch files.

3. Discover steering vectors:
python -m src.discover_vectors \
  --activations outputs/activations/ \
  --traces outputs/traces.parquet \
  --methods paired_diff,probe,svd \
  --labels high_effort,correct,positive_flip \
  --out outputs/vectors/

For each layer, compute:
- mean paired high-effort minus instant vector
- logistic regression probe vector
- SVD top vector over paired differences
Report AUROC, correlation with reasoning_tokens, and train/dev/test split performance.

4. Discover neurons:
python -m src.discover_neurons \
  --activations outputs/activations/ \
  --traces outputs/traces.parquet \
  --site mlp_act \
  --out outputs/neurons.json

Score neurons by:
- Cohen's d between good_reasoning and bad_reasoning
- ratio score
- mutual information with correctness
- stability over 10 random splits

5. Run intervention sweeps:
python -m src.sweeps \
  --model MODEL_NAME_OR_PATH \
  --dataset heldout_synthetic_math \
  --vectors outputs/vectors/ \
  --neurons outputs/neurons.json \
  --alphas -3,-2,-1,-0.5,0,0.5,1,2,3 \
  --layers all \
  --modes residual_vector,neuron_scale,random_direction \
  --out outputs/sweeps.parquet

Metrics:
accuracy, reasoning_tokens, total_tokens, positive_flips,
negative_flips, cost_adjusted_score, repetition_rate, malformed_rate.

6. Fit Reasoning Persistence Modes:
python -m src.rpm \
  --activations outputs/activations/ \
  --traces outputs/traces.parquet \
  --layers selected_by_probe \
  --pca-dim 128 \
  --out outputs/rpm/

Fit z_{t+1} ≈ A z_t for generated reasoning trajectories.
Rank modes by:
- eigenvalue magnitude close to 1
- correlation with remaining reasoning length
- correlation with correctness
- causal steering performance on dev set.

7. Run CET controller:
python -m src.cet_controller \
  --model MODEL_NAME_OR_PATH \
  --dataset heldout_synthetic_math \
  --effort-vector outputs/rpm/best.pt \
  --probe outputs/vectors/best_probe.pt \
  --target-percentile 75 \
  --alpha-max 2.0 \
  --soft-budget 256 \
  --hard-budget 512 \
  --out outputs/cet_eval.parquet

CET logic:
- read effort_score at each generation step
- if model tries to answer too early and effort_score below target, inject positive steering
- if over soft budget or answer stable, reduce steering or stop
- compare with instant, CoT prompt, budget forcing, static steering.

8. Analyze:
python -m src.analyze \
  --inputs outputs/sweeps.parquet outputs/cet_eval.parquet \
  --out outputs/report.md

Report:
- baseline table
- layer × alpha heatmaps
- length-vs-accuracy Pareto frontier
- positive/negative flip analysis
- random direction percentile
- best static steering vs CET
- top neurons/features with examples
- failure cases.
```

---

## 11. 成功标准

我会设置三个级别。

最低成功：

```text
找到某一层 residual direction。
+α 让 reasoning length 显著增加。
-α 让 reasoning length 显著减少。
随机方向做不到。
```

中等成功：

```text
在 held-out 同类任务上，中等 α 同时提高 reasoning length 和 accuracy。
例如 accuracy +2 到 +5 pp，且 malformed/repetition 不显著增加。
```

高质量成功：

```text
CET 比 static steering 更好。
它不是总拉长，而是对 positive-flip 题增加 thinking，
对 easy / overthinking-prone 题减少干预。
cost-adjusted_score 超过 CoT prompt 和 budget forcing。
```

真正有论文味的成功：

```text
RPM 模式存在：
某个动态 mode 具有高 temporal persistence，
能预测 remaining reasoning length，
正向注入能启动或维持 reasoning attractor，
反向注入能让模型提前 Final，
并且该 mode 和一组 MLP neurons / SAE features / attribution path 对齐。
```

---

## 12. 最重要的坑

第一，**0.6B 可能没有强 reasoning attractor**。
这不代表方法错，而是模型能力/训练分布可能不支持。解决办法是先选简单任务，或者用很小的 LoRA/SFT 让模型学会 `<think>...</think><answer>...</answer>` 格式，再研究这个格式下的 effort circuit。

第二，**长度提升不等于能力提升**。
你需要把 “reasoning_tokens ↑ but accuracy same/down” 判为失败，而不是成功。

第三，**CoT 文本不一定 faithful**。
Anthropic 在 circuit tracing 文章里也专门讨论 CoT faithfulness，指出模型的“think out loud”不一定反映真实内部机制。([Transformer Circuits][6]) 所以你要靠 activation patching、ablation、反向 steering、随机方向对照来证明因果。

第四，**SAE feature 不等于真实机制**。
SAE 很有用，但有论文显示随机 transformer 也能产生看似可解释的 SAE latents，所以 SAE feature 必须被 causal tests 验证。([arXiv][8])

第五，**泛化的是方法，不是具体 neuron。**
不同模型的 neuron index 不会泛化。更可能泛化的是这套流程：contrastive pairs → direction/probe/SAE/RPM → causal intervention → closed-loop controller。

---

## 13. 我建议你建立的核心理论图景

可以把 decoder 生成看成：

```text
当前 hidden state h_t
→ logits
→ 采样/选择 token y_t
→ y_t 写回 context
→ 下一个 hidden state h_{t+1}
```

如果存在 reasoning attractor，那么某些 token 和 hidden features 会形成正反馈：

```text
effort feature ↑
→ 更可能输出 “let’s check / wait / first / then”
→ 这些 token 进入 context
→ 下一步 effort feature 更容易继续激活
→ reasoning 持续
```

你的实验就是要找这个正反馈链条里的控制点。

普通 activation steering 问：

```text
哪个方向能改变输出？
```

你的 CET/RPM 方法问：

```text
哪个方向能改变生成动力学，让模型进入或退出 reasoning attractor？
```

这就是比较新的 perspective。它不是“找一个神经元按钮”，而是把 reasoning effort 当成 **decoder 自回归系统里的可控持久状态**。这个方向如果跑通，会比单纯复现 CAA、SAE steering 或 neuron scaling 更有意思。

[1]: https://www.anthropic.com/news/golden-gate-claude "Golden Gate Claude \ Anthropic"
[2]: https://arxiv.org/abs/2312.06681 "[2312.06681] Steering Llama 2 via Contrastive Activation Addition"
[3]: https://arxiv.org/html/2308.10248v4 "Activation Addition: Steering Language Models Without Optimization"
[4]: https://arxiv.org/abs/2501.19393?utm_source=chatgpt.com "s1: Simple test-time scaling"
[5]: https://arxiv.org/html/2604.10739v1 "When More Thinking Hurts: Overthinking in LLM Test-Time Compute Scaling"
[6]: https://transformer-circuits.pub/2025/attribution-graphs/biology.html "On the Biology of a Large Language Model"
[7]: https://github.com/decoderesearch/SAELens "GitHub - decoderesearch/SAELens: Training Sparse Autoencoders on Language Models · GitHub"
[8]: https://arxiv.org/html/2501.17727v1 "Sparse Autoencoders Can Interpret Randomly Initialized Transformers"
[9]: https://github.com/TransformerLensOrg/TransformerLens "GitHub - TransformerLensOrg/TransformerLens: A library for mechanistic interpretability of GPT-style language models · GitHub"
[10]: https://nnsight.net/ "nnsight"
