# HANOI HUST 主线故事与证据闭环
日期：2026-07-28

## 1. 一句话主线

在冻结 `source-access` protocol 下，HANOI HUST 可以形成一条可复核、物理上可解释的源域复合故障证据链。`logistic_l2 + envelope_log_power` 是当前最强源域基线，但它的可信边界、噪声敏感性和效率代价都必须同时报告，不能只报单一分数。

## 2. 论文应该讲的故事

这篇论文不是在讲“我们把准确率做到了多少”，而是在讲：

1. 冻结 `source-access` 之后，源域复合故障诊断仍然可以稳定建模。
2. 代表性选择不是任意的，`envelope_log_power` 在冻结比较集里最强。
3. 标准 `classical comparator` 家族也被对齐进来，`Results 2.3 Classical alignment` 中的 `rf + statistics` 和 `extra_trees + envelope_log_power` 说明主模型不是靠窄 null 才显得强。
4. 评估必须按 `physical bearing`、`group` 和 `calibration` 边界来讲，不能只报均值。
5. 方法在 `clean` 条件下有效，但在噪声、最坏组和代价上存在清晰边界。
6. 因此，论文的价值是“边界清楚、证据完整、可复核”，而不是“只靠一个高分”。

## 3. 可以直接进 Results 的证据

### 3.1 Source champion

- `logistic_l2 + envelope_log_power`
- `mean_component_auroc = 0.8138447971781305`
- `mean_component_balanced_accuracy = 0.7427248677248678`
- `exact_set_accuracy = 0.543859649122807`
- `mean_brier_score = 0.16050664093995246`
- `hamming_loss = 0.19883040935672514`

这应该是主结果锚点。

### 3.2 Classical alignment

- `rf + statistics` 的 exact-set 最高，达到 `0.5789473684210527`
- `extra_trees + envelope_log_power` 保持竞争性，达到 `0.543859649122807`
- `linear_svm + envelope_log_power` 和 `rbf_svm + all` 明显落后
- `logistic_l2 + envelope_log_power` 仍然是整个对齐 family 里的最强主线

这部分的作用不是改主冠军，而是把主冠军放进标准 `classical comparator` 家族里加固，正式承载位置就是 `Results 2.3 Classical alignment`。

### 3.3 Trustworthiness

- `outer` calibration error 最大，ECE = `0.2963116308560487`
- worst exact-set groups: `B5`, `B6`
- worst inner-accuracy group: `I7`

这应该进入 Results 的边界段或 Discussion 开头。

### 3.4 Nested/group screen

- `mean_component_auroc = 0.7014109347442682`
- `mean_component_balanced_accuracy = 0.6849206349206348`
- `exact_set_accuracy = 0.5263157894736842`
- 说明选参与外层评估分离后，结果更严格

这应该作为协议修复或补充材料，而不是主冠军。

### 3.5 Multi-seed repeatability

- seeds: `42`, `7`, `21`, `84`, `168`
- `mean_component_auroc = 0.8138447971781305`
- `mean_component_balanced_accuracy = 0.7427248677248678`
- `exact_set_accuracy = 0.543859649122807`
- `mean_brier_score = 0.16050664093995246`
- `hamming_loss = 0.19883040935672514`
- `std = 0.0` for all listed metrics

这说明当前 5 次重复不是“随机波动很小”，而是冻结协议下的完全复现。  
因此它的论文价值是证明主结果对该协议不敏感，而不是证明任何随机训练都自然稳定。

### 3.5 Noise robustness

- `NOACE classical` 在 `5 dB` 下 exact-set 仍为 `0.5789473684210527`
- source champion 在 `-5 dB` 下 mean AUROC 降到 `0.4668430335097002`
- `NOACE deep` clean 最强，但噪声更敏感

这应该放在鲁棒性段，重点讲边界和稳定性。

### 3.6 Efficiency

- source champion 评估中位时间约 `1.8900 s`
- `NOACE classical` 约 `0.1290 s`
- `NOACE deep` 约 `8.7912 s`

这应该放在复杂度/代价段，和鲁棒性一起讲 trade-off。

## 4. 最终写法顺序

### Results 顺序

1. Frozen protocol and source champion
2. Representation ablation
3. Classical alignment
4. Multi-seed repeatability
5. Trustworthiness / worst-group
6. Nested/group protocol check
7. Noise robustness
8. Efficiency / cost

### Discussion 顺序

1. Why the source result is meaningful under a frozen boundary
2. Why the classical alignment in Results 2.3 strengthens the claim
3. Why the outer component is the weakest calibrated part
4. Why noise reveals that clean performance is not robustness
5. Why efficiency changes the practical interpretation
6. Why Paderborn is conditional, not unconditional

## 5. 当前最稳妥的主张边界

可以稳妥宣称：

- 源域冻结证据链已建立
- 当前最强 representation 是 `envelope_log_power`
- 经典对齐已经把主模型放进标准 comparator family
- 证据链包含 repeatability、trustworthiness、noise robustness 和 efficiency
- repeatability 的含义是冻结协议下完全复现，不是把确定性模型伪装成随机稳定性

不能宣称：

- 已经全面解决 compound fault diagnosis
- 已经证明跨域无条件泛化
- 已经在所有噪声和代价条件下最优

## 6. 下一步

把这份主线直接映射回正式正文：

1. 调整 Results 段落顺序。
2. 让 Discussion 从“主结果”转到“经典对齐 + 边界 + trade-off”。
3. 保持所有结果页和入口文件同步，不再新增孤立证据页。
