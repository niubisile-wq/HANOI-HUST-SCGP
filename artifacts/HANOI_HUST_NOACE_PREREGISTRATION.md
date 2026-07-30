# Hanoi HUST NOACE Preregistration

Frozen intent date: 2026-07-27

## 1. Scientific question

This experiment asks whether component faults can be recognized in a
previously unseen physical bearing with two simultaneous defects when every
training bearing is healthy or has exactly one defect. The target is not
ordinary multi-label classification. Independent component classifiers are
already a published baseline on this dataset.

The development hypothesis is **Nuisance-Orthogonal Additive Component
Evidence (NOACE)**:

> Removing source-estimated bearing-type and load effects before composing
> singleton fault evidence, then requiring one latent component subset to
> explain all three loads of a physical bearing, should generalize better to
> unseen compound bearings than independent discriminative heads.

NOACE is a research hypothesis, not a novelty claim. A later claim requires
a dedicated prior-art audit against semantic, generative, diffusion,
prototype-composition, and multi-label compound-fault methods.

## 2. Resource and independent unit

- Mendeley Data record: `cbv7jyx4p9`;
- version: 3;
- DOI: `10.17632/cbv7jyx4p9.3`;
- licence: CC BY 4.0;
- sample rate stated by the authors: 51,200 samples/s;
- machine speed stated in the associated study: approximately 1,440 rpm;
- 99 MAT acquisitions;
- 33 physical bearings;
- loads: 0 W, 200 W, and 400 W;
- one acquisition per bearing and load.

The independent unit is the physical bearing, not a window or MAT file. The
three loads are repeated environments from the same bearing. Windows, loads,
frequency bands, and repeated train/test partitions cannot be counted as
independent bearings.

The filename contract is fixed before numeric access:

- `N4`-`N8`: five healthy bearings;
- `I4`-`I8`: five inner-race bearings;
- `O4`-`O8`: five outer-race bearings;
- `B5`-`B8`: four ball-fault bearings;
- `IB5`-`IB8`: four inner-plus-ball bearings;
- `IO4`-`IO8`: five inner-plus-outer bearings;
- `OB4`-`OB8`: five outer-plus-ball bearings.

Each bearing has suffixes `00`, `02`, and `04`, denoting 0 W, 200 W, and
400 W. Healthy and singleton bearings form the 19-bearing source partition.
The 14 compound bearings form the sealed confirmation partition.

## 3. Prior-art and benchmark boundary

The 2026 MSSP study *Towards a more realistic evaluation of machine learning
models for bearing fault diagnosis* already establishes:

1. strict bearing-wise splitting;
2. independent binary outputs for inner, outer, and ball faults;
3. Random Forest, SVM, and WDCNN comparisons;
4. a singleton-only training experiment with compound bearings in test;
5. reported Macro AUROC of `74.80% +/- 6.20%` for that source-only setting.

Therefore this project will not claim multi-label formulation, bearing-wise
evaluation, singleton-to-compound transfer, handcrafted features, or an
additive prototype alone as new. The published result is a mandatory external
reference, not a promotion threshold that can be ignored.

## 4. Access history and exclusion

Before this document was committed, only publisher metadata, the ZIP central
directory, the paper, and source code from the official replication repository
were intentionally inspected. No MAT payload or MAT numeric value was opened.

During source-code review, cloning official commit
`4c9fd8cb3cf9d5aa4ef2b653cb314a0727e12f3d` also acquired its tracked
`data/features/hust_features_segmented.parquet` blob:

- Git blob SHA-1: `6cf9e9179c9a0ce9dc31819fd2e1f341f05e45d4`;
- repository-reported bytes: `246992`;
- dataframe reads or semantic numeric inspection: zero.

This acquisition is recorded rather than concealed. The blob and every
upstream saved feature, logit, or result artifact are permanently forbidden as
experimental input. The raw Mendeley v3 archive will be parsed independently.
If the excluded blob is opened or used, prospective confirmation is invalid.

## 5. Two-stage numeric access

### Stage A: source-only development

After a committed preaccess freeze binds all 101 ZIP members by path, size,
CRC-32, compression method, and local offset, the complete archive may be
downloaded and hashed because Mendeley exposes one aggregate ZIP. Only the 57
healthy/singleton MAT members may then be opened.

Source work may resolve the MAT schema, verify sample rate and finite values,
freeze deterministic windows and features, implement comparators and NOACE,
and run bearing-wise source validation. The 42 compound MAT members, their
headers, arrays, summaries, plots, and derivatives remain forbidden.

### Stage B: compound confirmation

Compound numeric access requires a second committed freeze binding:

1. archive SHA-256 and all source member CRC checks;
2. the exact MAT parser and resolved source schema;
3. window length, offsets, detrending, scaling, and aggregation;
4. every feature and model hyperparameter;
5. 100 deterministic bearing-wise source partitions;
6. source-only model selection and negative controls;
7. registered baselines, endpoints, uncertainty calculation, and gates;
8. hashes of code, tests, source cache, and final configuration;
9. evidence that no compound MAT member has been opened.

After Stage B, all registered methods run once. No parser, feature, target
subset, seed, hyperparameter, or endpoint changes are allowed in the
confirmatory track.

## 6. Frozen representation family

The parser must identify the vibration array without fault-name-dependent
logic, reject ambiguous arrays, and verify finite data and the documented
sample rate. The representation will be selected only within this fixed
family:

- robust time statistics;
- fixed physical-frequency log-power bands;
- envelope log-power bands;
- optional order-normalized bands only if source data provide a validated
  speed channel or the constant-speed assumption passes a source-only check.

No bearing characteristic frequency is used unless bearing geometry is
obtained from an authoritative source and frozen before compound access.
Every scaler, nuisance model, variance estimate, and feature selector is fit
inside the source training bearings only.

Windows from a record are aggregated before bearing-level scoring. They may
measure within-record stability but are never independent test samples.

## 7. NOACE algorithm contract

For each source training partition:

1. aggregate each bearing-load record into a fixed robust feature vector;
2. fit a regularized additive nuisance model for bearing type and load using
   source training bearings only;
3. subtract the fitted nuisance contribution;
4. estimate healthy reference and inner, outer, and ball component effects
   jointly from healthy and singleton source bearings;
5. synthesize all eight component-subset prototypes without a compound
   example;
6. estimate a diagonal or shrinkage covariance from source residuals;
7. compute a source-frozen energy for every subset at each load;
8. sum log evidence across the three loads belonging to the same physical
   bearing;
9. obtain each component score by marginalizing the eight subset
   probabilities.

Regularization, covariance shrinkage, feature block, and temperature are
chosen by nested source-bearing validation. Ties use lexical subset order.
No compound count, label, bearing ID, filename, or target statistic may enter
fitting or hyperparameter selection.

## 8. Registered comparators

All methods receive the same source bearings, windows, and feature blocks:

- independent logistic component heads;
- independent SVM component heads;
- independent Random Forest heads using the published HUST hyperparameters;
- independent ExtraTrees heads;
- nearest healthy/singleton prototype;
- additive subset prototype without nuisance removal;
- NOACE without bearing-type nuisance removal;
- NOACE without joint three-load evidence;
- complete frozen NOACE.

The official upstream processed feature blob is not a comparator input.
WDCNN is a secondary reproduction only if its training and compute budget can
be matched without inspecting compound validation performance.

## 9. Splits and endpoints

The 100 split seeds reproduce the published information budget: two training
bearings per healthy/singleton class and disjoint source test bearings. All
three loads from a bearing remain in one partition.

There are two target views:

- **published-protocol view:** two sealed compound bearings per compound
  category for each frozen split;
- **primary confirmation view:** all 14 sealed compound bearings are scored
  by every source-trained model, then predictions are averaged per physical
  bearing across the 100 source partitions.

Primary endpoint: physical-bearing Macro AUROC across inner, outer, and ball
components on the 14 compounds.

Mandatory secondary endpoints:

- per-component AUROC and average precision;
- exact component-set accuracy using source-selected thresholds;
- Hamming loss and macro-F1;
- source-test Macro AUROC;
- results by compound category, bearing type, and load;
- calibration error and Brier score;
- compute time, peak memory, and parameter count;
- paired deltas against every registered comparator.

Uncertainty uses a label-stratified bootstrap over the 14 unique compound
bearings. Repeated windows, loads, and splits are not bootstrap units.
Repeated-split standard deviations are descriptive only.

## 10. Promotion and stop rules

The confirmatory result is promoted only if every gate passes:

1. NOACE compound Macro AUROC is at least `0.80`;
2. each component AUROC is at least `0.70`;
3. NOACE exceeds the strongest target-label-free comparator by at least
   `0.03` absolute Macro AUROC;
4. the 95% label-stratified bootstrap interval for that paired delta excludes
   zero;
5. exact component-set accuracy is at least `0.50`;
6. source-test Macro AUROC is at least `0.70`;
7. every integrity and forbidden-access counter passes.

If source validation fails, compound access is not authorized. If confirmation
fails, all negative results remain reported and the target cannot be reused to
develop a replacement confirmatory method. Supervised compound training may
be computed only after unblinding as an identifiability ceiling and cannot
rescue failure.
