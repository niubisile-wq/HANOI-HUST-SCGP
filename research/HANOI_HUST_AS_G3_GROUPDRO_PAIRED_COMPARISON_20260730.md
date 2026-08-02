# GroupDRO versus fixed logistic paired comparison

The two methods were evaluated on identical 100 bearing-grouped splits, identical physical-bearing units, and identical labels. The paired split-level mean AUROC difference (GroupDRO minus fixed logistic) is **−0.01722**. A deterministic 200,000-draw paired sign permutation gives two-sided `p = 0.00211`.

This supports the conservative decision: GroupDRO is a useful domain-robustness diagnostic, but it does not improve the prespecified primary classifier and must not replace it in the main table.

Artifacts:

- `results/g3/hanoi_hust_groupdro_vs_logistic_paired.json`
- GroupDRO prediction SHA-256: `b920a28f8237fc55ed9f78fd5811983c0faf64e4636b781926af772605c6bdfa`
- Fixed-logistic prediction SHA-256: `76a8e31f528d1e904678837f3fa2ceccab1a3cef862fb84a9d87f424d01cc305`
