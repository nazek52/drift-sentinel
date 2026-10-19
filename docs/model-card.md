# Model card: synthetic incident escalation baseline

**Intended use:** educational evaluation of delayed labels, covariate shift, reproducible training and monitoring. **Not intended:** real incident triage, paging decisions, employee evaluation, or deployment without external validation.

**Data:** 2,400 independently generated snapshots, fixed seed 42, five nonnegative telemetry features. Labels are sampled from a logistic relationship and become available 30 ticks after observation. The final 20% changes latency/queue distributions and the queue relationship. Missingness is injected independently at approximately 1.5%. No personal data or external dataset is used.

**Model:** median imputation → standardization → logistic regression. Threshold maximizes validation F2 on a fixed 0.05–0.95 grid. Baseline predicts the training prior. Training and threshold selection remove labels unavailable at their decision boundary. Test reports use only labels mature by as_of.

**Metrics:** average precision, ROC AUC, precision, recall, F2, Brier, confusion counts. The average-precision interval uses 200 seeded IID resamples; that assumption would be inappropriate for repeated observations from the same real incident. Performance must be interpreted relative to the actual test prevalence and baseline.

**Monitoring:** feature PSI uses training-only quantile bins and a separate missing bucket. PSI >= 0.2 is an illustrative review threshold. No alerts are sent and no retraining is performed.

**Limitations:** synthetic labels contain assumptions from the generator, not independently verified outcomes. Real escalation definitions vary by organization; telemetry availability, class imbalance, correlated incidents, operational changes, and delayed/incorrect labels can dominate model behavior. No fairness, calibration-by-segment, or out-of-organization validation has been established.

**Reproducibility:** use pinned requirements, seed 42, dataset hash, split manifest, and the JSON model artifact. Tests verify the future-label invariance and portable inference. See docs/verification.md for the measured run rather than a hard-coded performance claim.
