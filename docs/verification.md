# Verification record

Executed 2026-10-19 on Python 3.13.5 with scikit-learn 1.9.1 and the pinned requirements.

**5 tests passed**: identical/shifted/missing PSI, chronological label eligibility, portable inference equivalence, deterministic reruns, invariance to unavailable future labels, and invalid-input rejection (several assertions per test).

`python -m sentinel demo --output artifacts/demo` completed. The [measured report](evidence/report.json) records the dataset SHA256 and split sizes: **1,411 train / 451 validation / 450 mature test**, with **30 test labels withheld**. Threshold selected on validation: **0.18**.

| Metric | Model test | Prior baseline test |
|---|---:|---:|
| Average precision | 0.4573 | 0.2333 |
| ROC AUC | 0.6885 | 0.5000 |
| Brier score (lower is better) | 0.2743 | 0.1824 |
| F2 | 0.6108 | 0.6034 |
| Precision | 0.2458 | 0.2333 |
| Recall | 0.9714 | 1.0000 |

The model ranks better than the prior baseline, but its Brier score is **worse**, and its validation-selected threshold produces **313 false positives** after the synthetic regime change. This is a monitoring demonstration, not a model ready to deploy. Ranking improvement alone does not justify using its probabilities or its threshold in operations.

PSI identified `latency_ratio` (1.0415) and `queue_depth` (0.5951) for review. The test average-precision IID bootstrap interval was approximately [0.3614, 0.5445]. All these values are from synthetic data and depend on the generator and independence assumptions. Real-data usefulness has not been established.
