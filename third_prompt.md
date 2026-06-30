我再看一遍后，会把这个仓库的结论重新压缩成一句更准确的话：

**它已经找到了一个“可因果操纵的 late-layer generation posture 信号”，但还没有证明“找到了 reasoning circuit / 思考神经元 / 稳健提高推理能力的方法”。**

换句话说：这不是空的。它确实有有趣东西。但现在最好的命名不是 **effort circuit**，而是更保守、更有解释力的：

**late-layer reasoning / continuation posture transfer**
也就是：把高 effort prompt 下的后层 hidden state 转移到低 effort run 上，会让模型更倾向进入“继续结构化推理/避免糟糕重复或 premature finalization”的生成姿态。

---

## 1. 仓库当前真正完成了什么？

仓库本身是一个用于 Qwen3-0.6B 的 mechanistic intervention scaffold，README 写明目标是识别并因果干预那些能增加“useful reasoning effort”的内部激活，默认模型路径是本地 `Qwen3-0.6B`，并且 pipeline 覆盖 trace collection、activation caching、vector/neuron discovery、SAE、sweep、RPM、CET controller 等模块。注意：README 里很多模块是研究框架和 CLI 入口，不代表所有部分都已经跑完得到强结果。([GitHub][1])

它的第一性原理设定是对的：不预设“思考神经元”，而是从自回归模型的最小事实出发：hidden state 产生 logits，logits 决定下一 token；“继续推理”和“立即回答”先表现为下一 token 分布差异；“有用 effort”不能等同于更长输出，还要不降低正确率、不变成无控制 verbosity，并且要过随机方向、verbosity、任务外对照。([GitHub][2])

目前真实跑完并形成结论的主链路大概是四步：

第一，prompt-level steering。用 `verify` prompt 和 `instant` prompt 的 hidden-state 差分方向，去干预 neutral `cot` prompt 的下一 token 分布。结果显示 layer 4、alpha=-3 能把 continue-final gap 提高 `+0.115317`，但这个结果只是局部 logits 层面的，不是正确率或真实 reasoning length 的结论。([GitHub][3])

第二，随机方向对照和短生成。layer 4 候选方向超过 32/32 个随机方向，但最强随机方向几乎追平；短生成实验也很脏，模型经常先吐数字再解释，很多 completion 被 token cap 截断，所以这一阶段的正确率/长度不太能解释。仓库自己也承认，这说明 layer 4 不是 clean effort circuit，只是 early-stage candidate。([GitHub][4])

第三，全层 activation patching。这里才出现了最强信号：把 `verify` 的 hidden state patch 到 `instant` run，layer 27 的 low→high patch 把 gap 提高 `+0.734238`；反向把 `instant` state patch 到 `verify` run，layer 18 的 high→low patch 把 gap 降低 `-0.317724`。效果集中在 layers 18–27，而不是早层。([GitHub][5])

第四，late-layer interpolation 和 strict generation。对 layers 18、20、23、27 做 low→high hidden-state interpolation，continue-final gap 都随 t 单调上升；layer 27 最强，t 从 0 到 1 的 delta 是 `+0.734243`。然后在严格 `Reasoning:` / `Final answer:` 协议下做 prefix patch，train-small n=3 从 baseline accuracy `0.333` 到 layer27_t0.75/t1 的 `1.000`；heldout-small n=5 从 baseline `0.400` 到 layer27_t1 的 `0.800`。([GitHub][6])

所以，仓库作者自己的最终判断是：**late residual stream，尤其 layer 27，包含一种可因果操纵的“继续结构化推理 / 避免低质量 finalization 或 repetition”的内部状态**；但报告也明确列出样本量小、任务范围窄、生成被截断、未做 task-out controls、未证明闭环 CET 等风险。([GitHub][7])

---

## 2. 我认为最可靠的结论是什么？

最可靠的结论不是“layer 27 让模型变聪明”。更精确是：

**在这些 synthetic arithmetic prompts 上，Qwen3-0.6B 的后层 residual state 能显著改变下一 token 的 process choice：继续推理、进入 final、或者陷入重复。**

这个结论有三层证据支撑。

第一层是局部因果。patching 和 interpolation 确实改变了 continue-final token mass。尤其全层 patching 中，后层 18–27 的 low→high 和 high→low 都有方向性效果，这比单纯 probe 或 activation addition 强。([GitHub][8])

第二层是单调性。layers 18、20、23、27 在 low→high 插值时 gap 都单调增加，layer 27 从接近 0 的 gap 到 `0.734337`，而且 t=0.75 到 t=1.0 有一个明显跃迁。这说明它不是一个离散 hard-patch 偶然现象，而像一个连续但非线性的“输出姿态变量”。([GitHub][9])

第三层是小样本生成 utility。strict generation heldout n=5 里，baseline accuracy 是 `0.400`，layer27_t0.75 是 `0.600`，layer27_t1 是 `0.800`；同时 repetition 比 baseline 更低。这个结果不是论文级统计证据，但确实是“值得继续追”的行为信号。([GitHub][10])

我会把当前发现概括为：

> 模型后层存在一个可转移的 **process posture state**。这个 state 不一定包含更多题目知识，但会影响模型是否以更结构化的方式继续算下去，以及是否避免低质量重复/过早交卷。

这个比“思考神经元”更准确。

---

## 3. 最需要警惕的地方：layer 27 可能有 site-mismatch / final-norm artifact

这是我这次重新看后最想强调的点。

代码里 `hidden_last_by_layer` 是从 `outputs.hidden_states[1:]` 取每层最后 token 的 state；patching 时则 hook 到 `get_layers(bundle.model)[layer]` 的 module output，然后替换 `tensor[:, -1, :]`。([GitHub][11])

Qwen3-0.6B 的 config 显示它有 `num_hidden_layers: 28`。([Hugging Face][12]) Qwen3 模型结构是 decoder layers 后面还有一个 final `self.norm`，然后 CausalLM head 用 `outputs.last_hidden_state` 计算 logits。([GitHub][13])

这会引出一个非常关键的审计问题：**`hidden_states[1:][27]` 到底是 layer 27 decoder block output，还是 final norm 之后的 state？**

如果它是 final norm 之后的 state，而 hook 替换的是 layer 27 decoder block 的输出，那么当前最强的 layer 27 patch 就可能是在把“post-norm state”塞进“pre-final-norm site”。这会是一个 off-manifold readout hack，而不是干净的 layer 27 mechanism。

这个怀疑还有一个现象支持：原始 `verify` high gap 是 `0.307173`，但 layer 27 low→high patch 后 gap 居然到 `0.734238`，远超原始 high prompt 本身。([GitHub][8]) 这不一定证明错，但非常值得审计。正常情况下，如果 patch site 完全同位点，把 high state patch 到 low run 的最后层，结果应该更接近 high logits；远超 high 可能说明 patch 位置、norm、context interaction 或 hidden-state index 有错位。

所以我现在会把 layer 27 结论降级为：

**layer 27/final-readout 区域有强可控信号，但必须先做 activation-site audit，才能说它是 layer 27 residual state，而不是 final norm / hook site 错配造成的放大。**

这一步不做，后面继续跑 n=100 也会有点悬。

---

## 4. 统计角度：现在还不能叫“显著发现”

从统计上看，当前结果非常早期。train-small 是 n=3，heldout-small 是 n=5。heldout 里 baseline 2/5 correct，layer27_t1 4/5 correct，看起来翻倍，但这种样本量无法支撑“显著提升”。仓库最终报告也明确承认 CPU quota 约 4 cores、8GiB memory，所以 generation 成本高，样本量目前偏小。([GitHub][7])

另一个统计污染是：strict generation 的 mean reasoning tokens 全部是 96。heldout 的 baseline、layer27_t0.75、layer27_t1 都是 `96.00`，说明 completion 基本打满 `max_new_tokens=96`。([GitHub][10]) 这意味着它还没有真正测出“模型在某个条件下主动想得更久/更短”，而是在固定 token budget 下，输出轨迹和最后抽取答案发生变化。

还有一个 extraction 问题。仓库已经修正了答案抽取器：优先找最后一个 `Final answer` / `Answer` / `####` marker，并在 marker 所在行取最后一个数字。这个修正很合理，因为 `Final answer: 54 - 5 = 49` 应抽 `49` 而不是 `54`。([GitHub][14]) 但它也带来一个评估定义问题：如果模型先输出错答案，后来又继续乱写/修正，取最后一个答案可能不等价于真实交卷场景。

heldout examples 里可以看到这个问题：baseline 有些输出多次 `Final answer` / `Reasoning` 循环，layer27_t0.75/t1 有时也会先给中间数再继续解释，甚至 layer27_t1 在一个样本里抽到 `360` 而 gold 是 `355`。([GitHub][10]) 所以下一步必须做 **stop-after-first-valid-final**，并同时报告 first-final accuracy 和 last-final accuracy。

---

## 5. 方法论角度：这个仓库最好的地方是什么？

它最好的地方是“门槛意识”正确。它没有一上来就宣称“找到了 neuron”，而是分成 local causal gate、specificity gate、trajectory gate、utility gate。最终报告也明确说：局部 causal gate 已通过，specificity gate 只部分通过，trajectory gate 初步通过，utility gate 小样本通过但需要扩大验证。([GitHub][7])

这个研究 taste 很好。因为 activation steering 领域最容易犯的错就是：找到一个方向让输出变长，然后立刻说它是 reasoning direction。这个 repo 至少已经通过随机方向对照发现 layer 4 方向不够特异，又通过 free generation 负结果发现短生成协议不干净，然后转向 patching/interpolation/strict protocol。这个迭代路径是健康的。([GitHub][4])

但它还缺三种关键对照。

第一是 **verbosity control**。现在 continue tokens 包括 `" The"`，这很危险，因为 `"The"` 可能只是一般英语解释开头，不一定是 reasoning。当前 continue-final gap 更像“是否进入解释性文本”的粗指标，而不是 pure effort metric。([GitHub][3])

第二是 **task-out control**。仓库目前主要是 synthetic arithmetic，还没有事实问答、翻译、摘要、情绪改写等任务外对照。最终报告也承认 task-out controls 还没做。([GitHub][7]) 如果 layer27 patch 在翻译任务上也让模型狂写 reasoning，那它就是 general style/verbosity posture；如果只在 arithmetic 上有 positive utility，那才更接近 useful effort。

第三是 **content/posture disentanglement**。把同一道题的 high state patch 到 low run，可能同时转移了题目内容、中间答案、格式状态和 process posture。真正要证明“它是 process posture”，需要跨题 patch：题 A 的 high posture patch 到题 B，看它是否只改变 B 的推理风格而不泄漏 A 的数字/答案。

---

## 6. 与前人相比，这不是全新领域，但可以有新角度

前人已经做过很多相关东西，所以这个仓库不能声称“首次发现 activation 可提升 reasoning”。

CAA/Contrastive Activation Addition 已经提出：通过 positive/negative examples 的 residual activation 差值构造 steering vector，并在 forward pass 中加入该 vector 来控制行为。这个仓库早期的 `verify - instant` direction，本质上就是 CAA 家族的一个小型 reasoning/effort 版本。([arXiv][15])

更接近的是 2025 的 **Activation Control for Efficiently Eliciting Long Chain-of-thought Ability**。这篇论文发现，最后几层的一小组高影响 activation 很大程度上控制 long-form reasoning 的属性，比如输出长度和 self-reflection；放大这些 activation 并插入 “wait” tokens，可以无训练地唤起 long CoT，并提升 self-reflection rate 和 accuracy。([arXiv][16]) 这和本仓库的 late-layer result 非常近，甚至可以说是最直接的外部参照。

还有 EMNLP 2025 的 neuron activation differential analysis，专门从 FFN neurons 角度识别 reasoning-critical neurons，并通过调节 neuron activation 来提升 CoT 质量。([ACL Anthology][17]) 这说明“reasoning-critical neuron/activation”已经是被研究的问题，不是完全空白。

另一个很贴近的是 2026 的 **When Chain-of-Thought Fails, the Solution Hides in the Hidden States**。它用 activation patching 把 CoT generation 的 token-level hidden states 转移到 direct-answer run 上，发现 patch 后能恢复正确答案，而且任务相关信息集中在 mid-to-late layers。([arXiv][18]) 这个和仓库的 “verify state patch 到 instant run” 是同一大类思想。

Anthropic 的 interpretability work 也提供了宏观背景：他们用 features/attribution graphs 研究模型内部机制，并且 Golden Gate Claude 展示过调节内部 feature activation 可以显著改变模型行为。([Anthropic][19])

所以，这个仓库的潜在新意不在于“activation steering 可控制行为”，而在于可以往这个方向推进：

> 把 reasoning effort 看成 decoder 生成动力学里的 **process posture / attractor state**，而不是一个静态 steering vector 或单神经元。

如果后续能证明：某个 late-layer posture 可以跨题泛化、能闭环维持、能避免 verbosity/repetition、能在 first-final accuracy 上提升，那就会有自己的方法论味道。

---

## 7. 我现在会给它的“结论等级”

我会分四档。

**已经比较站得住：**

1. 内部 activation intervention 可以改变 continue-vs-final 的下一 token 分布。
2. 后层 18–27 比早层 direction addition 更像可解释的 causal site。
3. low→high interpolation 在选定后层上单调提高 continuation gap。
4. 小样本 strict generation 中，layer27_t1 的确表现出更好的 last-extracted-answer accuracy 和更低 repetition。([GitHub][10])

**还只是强苗头：**

1. layer 27/final-readout 区域可能是“reasoning posture”控制点。
2. 这个 posture 可能不是普通 verbosity，因为 repetition 下降、一些 heldout examples 变得更结构化。([GitHub][10])
3. 后层可能存在非线性阈值，t=0.75 到 t=1.0 出现明显跃迁。([GitHub][9])

**还没证明：**

1. 没证明它能稳健提高数学能力。
2. 没证明它能泛化到其他任务、其他 prompt、其他 seed、其他模型。
3. 没证明这是 neuron/circuit，而不是 residual posture/readout effect。
4. 没证明是真正 reasoning length control，因为所有 strict generation 都打满 96 tokens。([GitHub][10])
5. 没证明 layer 27 位点干净，因为 hidden-state index 与 hook site 需要审计。([GitHub][11])

**现在不能说：**

1. “找到了思考神经元。”
2. “证明 high effort 是由某组 neuron 控制。”
3. “activation patching 已经稳定提高 correctness。”
4. “这是商业 reasoning/high 模式的内部机制复现。”

---

## 8. 我建议下一步 Codex 按这个顺序做

第一优先级：**activation site audit**。
写 `site_audit.py`，不要先继续扩大实验。测试三件事：同 prompt self-patch 是否 logits delta≈0；`outputs.hidden_states[k]` patch 到 `get_layers()[k]` 是否 logits delta≈0；layer27 的 hidden state 到底是 decoder block output、final norm output，还是 capture_outputs 的某种特殊结果。尤其要解释为什么 layer27 patch 后 gap `0.734` 远超原始 high gap `0.307`。([GitHub][8])

第二优先级：**stop-after-first-final**。
当前 heldout 所有条件 mean reasoning tokens 都是 96，这说明 max token cap 主导了生成。必须改成逐 token generation，一旦出现合法 `Final answer:` 就停止，并同时记录 first-final answer、last-final answer、tokens-to-first-final、post-final continuation。([GitHub][10])

第三优先级：**n=50 heldout + paired stats**。
只测几个条件：baseline、site-audited layer/posture t=0.6/0.75/0.9/1.0。不要全层扫。报告 Wilson CI、paired bootstrap、McNemar/exact test、positive flips、negative flips。

第四优先级：**替换 continue-final token metric**。
现在手写 token set 太粗，尤其 `"The"` 很容易把 metric 变成“解释性英语开头”。可以从 high/low prompt 的 heldout next-token distributions 自动学习 discriminative token cluster，然后在 heldout 上评估 KL、final-margin、continue-margin，而不是固定几个词。

第五优先级：**process/content disentanglement**。
做三种 patch：same-question patch、cross-question patch、mean-direction patch。
如果 cross-question patch 也能改善 B 题的推理姿态而不泄漏 A 题答案，它更像 process posture；如果它把 A 的数字/答案带过去，那说明当前 state 混入 task content。

第六优先级：**task-out 和 verbosity controls**。
加入非推理任务：翻译、事实问答、情绪改写、摘要。再加入 verbose-but-not-reasoning prompt 和 format-only prompt。目标不是证明 patch 总有效，而是证明它只在“需要推理且 baseline 不稳定”的区间有用。

第七优先级：**CET controller，而不是 prefix patch**。
prefix patch 只是一次性启动姿态。更有趣的是 token-by-token 控制：如果模型过早进入 final margin，就推回 effort posture；如果已经出现合法 final answer，就停；如果 repetition 上升，就减弱 steering 或推离 repetition mode。仓库最初的 CET 设想就是这种闭环控制，但目前还没证明。([GitHub][7])

第八优先级：**再做 neuron/SAE**。
不要现在就下钻 neuron。先把 residual posture 站稳，再问这个 posture 来自哪些 MLP neurons、attention heads 或 SAE features。否则很容易在 artifact 上做解释性包装。

---

## 9. 我会把这个项目往哪个“新 perspective”推？

我不建议继续叫它“找思考神经元”。我建议改成：

**Posture Algebra for Autoregressive Reasoning**

核心想法是：decoder 每一步不只是有 task content，还有 process posture：

```text
hidden state ≈ task content + process posture + verbosity posture + finalization posture + repetition posture
```

这个仓库当前可能抓到的是：

```text
+ structured continuation posture
- bad finalization/repetition posture
```

下一步要做的是把它拆开：

```text
+effort, -verbosity
+verification, -repetition
+continue when uncertain, +finalize when stable
```

这比“让模型多想”更高级。真实 reasoning/high 模式也不是无限生成 token，而是把模型维持在一个合适的推理轨道上，并在答案稳定时退出。

---

## 10. 最终评价

我的最终判断是：

**这个仓库已经有足够有趣的早期发现，值得继续投入；但它还没有足够显著、干净、可泛化的发现，不能过度包装成 reasoning circuit。**

最值得保留的发现是：

> Qwen3-0.6B 在 synthetic arithmetic 上的 late-layer/readout 区域存在一个强 process-posture 信号。把 high-effort prompt 的 hidden state 迁移到 low-effort/strict generation 中，会显著改变 continue-vs-final logits，并在极小 heldout 上方向性提高 last-extracted answer accuracy、降低 repetition。

最需要立刻修的风险是：

> layer 27 可能是 final norm / hidden_states indexing / hook-site mismatch 导致的强 readout artifact；strict generation 全部打满 96 tokens；当前 accuracy 是小样本且依赖 last-answer extraction。

所以我的建议很明确：**先做 site audit 和 stop-after-first-final，再扩大样本。**
如果这两步过了，而且 layer27/late-posture 在 n=50 或 n=100 first-final accuracy 上仍然有效，那这个项目就从“好玩的机制线索”升级成“有实质研究价值的发现”。

[1]: https://github.com/Chi-Shan0707/effort-circuit "GitHub - Chi-Shan0707/effort-circuit · GitHub"
[2]: https://raw.githubusercontent.com/Chi-Shan0707/effort-circuit/main/FIRST_PRINCIPLES.md "raw.githubusercontent.com"
[3]: https://raw.githubusercontent.com/Chi-Shan0707/effort-circuit/main/docs/experiment_001_prompt_steering.md "raw.githubusercontent.com"
[4]: https://raw.githubusercontent.com/Chi-Shan0707/effort-circuit/main/docs/experiment_002_causal_controls_and_patching.md "raw.githubusercontent.com"
[5]: https://raw.githubusercontent.com/Chi-Shan0707/effort-circuit/main/docs/experiment_003_all_layer_activation_patching.md "raw.githubusercontent.com"
[6]: https://raw.githubusercontent.com/Chi-Shan0707/effort-circuit/main/docs/experiment_004_multistage_late_layer_intervention.md "raw.githubusercontent.com"
[7]: https://raw.githubusercontent.com/Chi-Shan0707/effort-circuit/main/docs/final_analysis_report.md "raw.githubusercontent.com"
[8]: https://raw.githubusercontent.com/Chi-Shan0707/effort-circuit/main/outputs/activation_patching_instant_verify_n8_all_layers.md "raw.githubusercontent.com"
[9]: https://raw.githubusercontent.com/Chi-Shan0707/effort-circuit/main/outputs/activation_interpolation_instant_verify_n8_late_layers.md "raw.githubusercontent.com"
[10]: https://raw.githubusercontent.com/Chi-Shan0707/effort-circuit/main/outputs/strict_generation_prefix_patch_heldout_n5_layer27_reextracted.md "raw.githubusercontent.com"
[11]: https://raw.githubusercontent.com/Chi-Shan0707/effort-circuit/main/src/prompt_steering_experiment.py "raw.githubusercontent.com"
[12]: https://huggingface.co/Qwen/Qwen3-0.6B/blob/main/config.json "config.json · Qwen/Qwen3-0.6B at main"
[13]: https://raw.githubusercontent.com/huggingface/transformers/v5.12.0/src/transformers/models/qwen3/modeling_qwen3.py "raw.githubusercontent.com"
[14]: https://raw.githubusercontent.com/Chi-Shan0707/effort-circuit/main/src/answer_extractors.py "raw.githubusercontent.com"
[15]: https://arxiv.org/abs/2312.06681?utm_source=chatgpt.com "Steering Llama 2 via Contrastive Activation Addition"
[16]: https://arxiv.org/abs/2505.17697?utm_source=chatgpt.com "Activation Control for Efficiently Eliciting Long Chain-of-thought Ability of Language Models"
[17]: https://aclanthology.org/2025.emnlp-main.817/?utm_source=chatgpt.com "Enhancing Chain-of-Thought Reasoning via Neuron ..."
[18]: https://arxiv.org/abs/2604.23351?utm_source=chatgpt.com "When Chain-of-Thought Fails, the Solution Hides in the Hidden States"
[19]: https://www.anthropic.com/research/tracing-thoughts-language-model?utm_source=chatgpt.com "Tracing the thoughts of a large language model"
