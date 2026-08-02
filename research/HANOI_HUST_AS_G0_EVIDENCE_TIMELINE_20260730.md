# HANOI–HUST Applied Sciences G0 证据链时间线

**审计日期：** 2026-07-30  
**结论级别：** `historical_freeze_and_retrospective_confirmation`，不是公开事前 sealed lockbox

## 时间线

| 时间（Asia/Shanghai） | 证据 | 解释 |
|---|---|---|
| 2026-07-27 23:13:47 | Git `b244ea7`：Preregister Hanoi HUST compositional study | 研究计划提交；不等于后续所有代码和结果均已冻结 |
| 2026-07-27 23:14:28 | Git `d9a389d`：Use trusted CA for Hanoi HUST freeze | 冻结环境相关提交 |
| 2026-07-27 23:14:39 | `research/HANOI_HUST_V3_PREACCESS_FREEZE.json` | 预访问元数据冻结，SHA-256 为 `c5c0a97d...52d74d6` |
| 2026-07-27 23:15:04 | Git `bd4d50a`：Freeze Hanoi HUST v3 archive metadata | 数据包元数据冻结 |
| 2026-07-27 23:16:49 | Git `ac5c2eb`：Bind opaque Hanoi HUST archive download | 下载绑定 |
| 2026-07-27 23:27:47 | `research/HANOI_HUST_V3_ARCHIVE_FREEZE.json` | 归档字节冻结，SHA-256 为 `d1d0cf91...3539728` |
| 2026-07-27 23:28:03 | Git `e94a0cc`：Freeze opaque Hanoi HUST archive bytes | 归档冻结提交 |
| 2026-07-28 20:07:08 | `src/run_hanoi_hust_compound_confirmation.py` 文件时间 | 决定性确认脚本在上述冻结提交之后产生/更新 |
| 2026-07-28 20:09:52 | `research/HANOI_HUST_SOURCE_ACCESS_FREEZE.json` | 源数据访问冻结文件生成；SHA-256 为 `15af6d15...8c8f60e3` |
| 2026-07-28 20:10:51 | `results/confirmation/hanoi_hust_compound_confirmation.json` | HANOI 复合确认结果生成；该结果属于冻结后的历史确认 |
| 2026-07-29 | Zenodo HANOI experiment package v1.0.4 公布 | 公布时间晚于确认结果，不能反向证明事前注册 |

## 审计结论

1. 源域数据的元数据、归档字节和访问边界有 Git 证据。
2. 决定性复合确认脚本和结果晚于最初冻结提交，且没有公开的、早于数值结果查看的完整代码/配置时间戳证据。
3. 因此论文可写：`historical held-out confirmation under a documented local freeze` 或 `retrospective confirmation`。
4. 论文不得写：`prospectively preregistered sealed test`、`publicly time-stamped sealed lockbox` 或由 Zenodo 时间戳推导出的事前保证。
5. 真正的 prospective/sealed 证据必须由新的外部数据锁箱提供，且数值访问前公开代码、配置、哈希和统计方案。

## 证据保留规则

- 不覆盖旧结果；
- 不删除晚于冻结时间的脚本；
- 不用当前仓库状态替代历史时间线；
- 任何事后分析必须在文件名和论文中标注 `post hoc` 或 `retrospective`；
- 新外部锁箱另建独立 registry，不复用 HANOI 的 seal 术语。
