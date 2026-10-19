# Evaluation architecture

## Three clocks
Observation timestamp tells when features existed. label_available_at tells when the outcome became knowable. as_of tells when the test report is generated. Each training/selection boundary excludes labels that would have arrived later. This prevents a subtle leakage path that chronological row splitting alone does not fix.

## No preprocessing leakage
Imputer and scaler live in an sklearn Pipeline fitted only on the eligible training rows. Validation selects a probability threshold; test labels never affect fitting or threshold choice. Test feature drift is unsupervised and may include rows whose labels are still unavailable, but predictive metrics cannot include those labels.

## Transparent model
Logistic regression supplies a compact, interpretable starting point and a prior baseline establishes a minimum useful comparison. Model state is exported as JSON coefficients and preprocessing parameters instead of executable pickle. Portable inference is tested against sklearn output.

## Reproducibility
Fixed generator/model/bootstrap seeds, pinned dependencies, dataset SHA256, exact split indices, and sklearn version are included. There is no timestamp in the result other than the explicit as_of data cutoff, so identical inputs in the same environment produce identical report content. Floating-point/library differences across platforms may cause small numerical changes.

## Monitoring limits
PSI uses training quantile boundaries, tail bins, a missing bucket, and Jeffreys-style 0.5-count smoothing. A changed marginal distribution is a reason to investigate, not proof that the conditional label relation changed. Label-based quality and feature drift are reported separately. The synthetic generator deliberately changes both so the experiment demonstrates this distinction.
