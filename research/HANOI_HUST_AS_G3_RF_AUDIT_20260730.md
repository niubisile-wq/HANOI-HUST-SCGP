# G3 Random Forest 基线审计

**日期：** 2026-07-30  
**方法等级：** R2（根据预注册任务和公开 HUST 设置重实现）  
**信息预算：** I0 source-only  
**统计单位：** physical bearing  
**拆分：** G2 预生成的 100 个 bearing-grouped splits

## 冻结设置

- 输入：`envelope_log_power` 工程特征；
- 三个独立二分类头：inner、outer、ball；
- `n_estimators=500`；
- `min_samples_leaf=1`；
- `max_features=sqrt`；
- `class_weight=balanced`；
- 每个外层训练集单独拟合；
- 概率在轴承内聚合后计算指标；
- 无 HANOI compound 标签、无目标域调参。

## 100 次 bearing-grouped 结果

| 指标 | 均值 | 拆分标准差 |
|---|---:|---:|
| macro component AUROC | 0.7849 | 0.1153 |
| macro AUPR | 0.7503 | 0.1377 |
| macro balanced accuracy | 0.6657 | 0.1568 |
| macro-F1 | 0.6279 | 0.1625 |
| Brier score | 0.1798 | 0.0567 |
| exact-set accuracy | 0.4580 | 0.1996 |
| Hamming loss | 0.2700 | 0.1162 |

与 G2 bearing-grouped fixed logistic 的 macro AUROC 均值 `0.8597` 相比，RF 当前结果较低。该阴性结果保留，不通过增加树数、修改测试拆分或删除不利拆分来修饰。

## 验收

- [x] 100 个 bearing-grouped splits 完成；
- [x] 轴承级预测文件保存；
- [x] 超参数写入结果；
- [x] I0 信息边界符合；
- [x] 阴性结果保留；
- [x] R2 实现和数据输入可追溯；
- [ ] 与公开论文的逐项超参数和输入预处理做最终文献核对；
- [ ] 加入 G3 主表和统一 bootstrap/效率表。
