# HANOI–HUST Applied Sciences G2 匹配协议审计

**日期：** 2026-07-30  
**汇总产物：** `results/g2/protocol_cell_metrics.json`  
**状态：** `implemented_cells_complete_window_random_blocked`

## 已完成单元

| 划分层级 | fixed prespecified | nested selection |
|---|---:|---:|
| record-grouped | 100/100 | 100/100 |
| bearing-grouped | 100/100 | 100/100 |
| window-random | blocked | blocked |

所有正式单元共享：

- I0 source-only 信息预算；
- envelope_log_power 表示；
- 14 候选经典核心池；
- 轴承级概率平均和阈值；
- 轴承级 macro component AUROC 主指标；
- 预生成拆分清单；
- 失败运行和选择结果保留。

## 观察到的协议差异

四个已完成单元的拆分均值为：

| 划分 | 选择 | macro AUROC | exact-set |
|---|---|---:|---:|
| record-grouped | fixed | 0.9585 | 0.7907 |
| record-grouped | nested | 0.9728 | 0.8071 |
| bearing-grouped | fixed | 0.8619 | 0.4900 |
| bearing-grouped | nested | 0.8033 | 0.4060 |

以 bearing-grouped 减去 record-grouped 的拆分配对均值为：

| 模型选择 | macro AUROC 差 | exact-set 差 | Hamming loss 差 |
|---|---:|---:|---:|
| fixed | −0.0965 | −0.3007 | +0.1314 |
| nested | −0.1695 | −0.4011 | +0.1907 |

这说明在当前数据和候选池中，记录级划分会给出更乐观的结果；nested 选择并不能消除物理单元划分带来的落差。该表目前是**拆分敏感性摘要**，不是最终的独立单位 cluster-bootstrap 显著性结论。

## 选择稳定性

- record-grouped nested：归一化选择熵 0.8375；
- bearing-grouped nested：归一化选择熵 0.8887；
- fixed prespecified：熵 0（设计所得，不作为稳定性证据）。

在 bearing-grouped nested 中，logistic、Extra Trees 和 RBF-SVM 均被选中，最高单一候选的选择频率为 21/100；这支持进一步报告模型选择不稳定性，但仍需补充 top-k overlap、Kendall/Spearman 及 regret。

已补充的 100 次内层排序稳定性为：

| 单元 | top-3 Jaccard | Spearman ρ | Kendall τ |
|---|---:|---:|---:|
| record-grouped nested | 0.3322 | 0.6022 | 0.4717 |
| bearing-grouped nested | 0.1945 | 0.2016 | 0.1512 |

这表明 bearing-grouped 下候选排序明显更不稳定。但 outer-test regret 尚未关闭，因为当前运行器只保存最终选中候选的外层预测；后续必须保存所有候选的外层预测或等价的审计结果，才能计算无事后选择偏差的 regret。

已补齐全部候选外层测试指标后，macro-AUROC outer-test regret（事后最优候选减去实际选中候选）为：

| 单元 | 平均 regret | 拆分敏感性区间 |
|---|---:|---:|
| record-grouped fixed | 0.0294 | [0.0233, 0.0359] |
| record-grouped nested | 0.0151 | [0.0113, 0.0192] |
| bearing-grouped fixed | 0.0367 | [0.0261, 0.0478] |
| bearing-grouped nested | 0.0953 | [0.0733, 0.1183] |

因此 bearing-grouped nested 的选择程序不仅排序熵高，实际外层 regret 也更大；该结果比单纯报告“某模型被选了多少次”更有解释力。

已对每个重复拆分内部的物理单位做 100 次 cluster bootstrap（总计约 10,000 次有效重采样/主单元）。由于每个测试拆分只有少量独立轴承，区间很宽；这些区间用于显示单位级不确定性，不能被误读为大样本精确区间。正式论文还应以预设主拆分和配对置换做最终确认。

当前已增加 100 个配对拆分差异的 sign-permutation 与 Holm 校正；六个主比较的校正后 p 值均为约 `0.0006`。这些 p 值描述的是重复拆分层面的协议差异，不替代物理单元级推断，也不单独证明跨数据集泛化。

## 重要边界

1. record-grouped 允许同一物理轴承的不同记录跨训练/测试，因此它只能作为泄漏敏感性对照，不是部署安全协议。
2. window-random 没有运行：现有缓存没有独立窗口行，不能把记录内聚合窗口伪装成独立样本。
3. 当前区间为重复拆分敏感性 bootstrap；正式稿还需完成物理单元 cluster bootstrap、配对置换和 Holm 校正。
4. 结果仅覆盖经典核心池和 HANOI 源域，不代表外部迁移或工业部署。

## window-random 描述性结果

1541 个窗口行的 100 次描述性 fixed/nested 运行均得到 macro AUROC 约 1.000、exact-set 约 1.000。该结果只能作为强烈的泄漏敏感性对照：同一物理轴承的多个窗口同时出现在训练和测试时，模型可以利用设备/记录特征，窗口数量不能作为独立证据。该结果不进入显著性推断，不与 bearing-grouped 结果合并，不称为泛化性能。

## 产物

- `research/HANOI_HUST_AS_G2_MATCHED_PROTOCOL_CONTRACT_20260730.yaml`
- `results/g2/split_manifests/hanoi_hust_g2_manifests.json`
- `src/run_hanoi_hust_g2_protocol.py`
- `results/g2/hanoi_hust_record_grouped_fixed_prespecified.json`
- `results/g2/hanoi_hust_record_grouped_nested_selection.json`
- `results/g2/hanoi_hust_bearing_grouped_fixed_prespecified.json`
- `results/g2/hanoi_hust_bearing_grouped_nested_selection.json`
- `src/analyze_hanoi_hust_g2_protocol.py`
- `results/g2/protocol_cell_metrics.json`
- `src/run_hanoi_hust_g2_primary_paired_split.py`
- `results/g2/primary_paired_split.json`
- `src/audit_hanoi_hust_g2_main_table.py`
- `results/g2/g2_main_table.csv`
- `results/g2/g2_main_table_audit.json`
- `src/build_hanoi_hust_window_features.py`
- `artifacts/hanoi_hust_window/source_window_features.npz`

## G2 验收状态

- [x] 合同冻结；
- [x] 100 个 record-grouped 清单；
- [x] 100 个 bearing-grouped 清单；
- [x] 四个可执行单元完成 100/100；
- [x] 每个单元保存单位级预测；
- [x] 选择频率和熵已计算；
- [x] 协议配对差异已计算；
- [x] window-random 原始窗口缓存和 100/100 描述性运行；
- [x] 物理单元 cluster bootstrap（拆分内）；
- [x] Holm 多重校正与重复拆分配对 sign-permutation；
- [x] 预设单一主拆分的物理单元配对置换（5 个预先指定轴承）；
- [x] top-3 Jaccard、Kendall/Spearman；
- [x] outer-test regret（已保存全部外层候选指标）；
- [x] G2 最终主表一键重建审计。

## G2 Gate 判定

G2 的协议、拆分、四个主单元、窗口描述性单元、选择稳定性、regret、单位级 bootstrap、置换检验和主表重建均已产生可追溯产物。G2 可以关闭，前提是论文将窗口随机结果标为描述性泄漏敏感性，并将 5 个轴承的预设主拆分置换明确标为小样本审计，而不是大样本显著性证明。
