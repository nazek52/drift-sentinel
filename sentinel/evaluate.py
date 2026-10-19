import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sentinel.data import FEATURES, read


def psi(reference, current, bins=10):
    """Population stability index; reference-only edges and a separate missing bin."""
    reference, current = np.asarray(reference), np.asarray(current)
    finite = reference[np.isfinite(reference)]
    if not len(finite) or not len(current):
        return None
    edges = np.unique(np.quantile(finite, np.linspace(0, 1, bins + 1)[1:-1]))
    # Tail bins include values outside the entire training support.
    edges = np.r_[-np.inf, edges, np.inf]

    def fractions(values):
        counts = np.r_[
            np.histogram(values[np.isfinite(values)], bins=edges)[0], np.isnan(values).sum()
        ]
        smoothed = counts + 0.5
        return smoothed / smoothed.sum()

    expected, actual = fractions(reference), fractions(current)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def metrics(y, scores, threshold):
    predicted = scores >= threshold
    both = len(np.unique(y)) == 2
    return {
        "rows": len(y),
        "positive_rate": float(np.mean(y)),
        "average_precision": float(average_precision_score(y, scores)) if both else None,
        "roc_auc": float(roc_auc_score(y, scores)) if both else None,
        "brier": float(brier_score_loss(y, scores)),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "f2": float(fbeta_score(y, predicted, beta=2, zero_division=0)),
        "confusion_matrix_tn_fp_fn_tp": confusion_matrix(y, predicted, labels=[0, 1])
        .ravel()
        .tolist(),
    }


def evaluate(path, output, as_of=None):
    t, x, y, available = read(path)
    as_of = float(t[-1] if as_of is None else as_of)
    if not np.isfinite(as_of) or as_of < t[-1]:
        raise ValueError("as_of must be finite and at least the final observation timestamp")
    n = len(t)
    train_end, val_end = int(n * 0.6), int(n * 0.8)
    indices = np.arange(n)
    # At each decision boundary, drop labels not yet available in real time.
    train = indices[(indices < train_end) & (available <= t[train_end])]
    validation = indices[(indices >= train_end) & (indices < val_end) & (available <= t[val_end])]
    test = indices[(indices >= val_end) & (available <= as_of)]
    for name, subset in [("train", train), ("validation", validation), ("test", test)]:
        if len(subset) < 30:
            raise ValueError(f"{name} has fewer than 30 mature labels")
    if len(np.unique(y[train])) < 2 or len(np.unique(y[validation])) < 2:
        raise ValueError("Training and validation need both classes")
    if np.isnan(x[train]).all(axis=0).any():
        raise ValueError("A training feature is entirely missing")
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=42),
    )
    model.fit(x[train], y[train])
    val_scores = model.predict_proba(x[validation])[:, 1]
    candidates = np.linspace(0.05, 0.95, 91)
    threshold = float(
        max(
            candidates,
            key=lambda value: fbeta_score(
                y[validation], val_scores >= value, beta=2, zero_division=0
            ),
        )
    )
    test_scores = model.predict_proba(x[test])[:, 1]
    baseline = DummyClassifier(strategy="prior").fit(x[train], y[train])
    baseline_val = baseline.predict_proba(x[validation])[:, 1]
    baseline_threshold = float(
        max(
            candidates,
            key=lambda value: fbeta_score(
                y[validation], baseline_val >= value, beta=2, zero_division=0
            ),
        )
    )
    evaluation = metrics(y[test], test_scores, threshold)
    # IID bootstrap is appropriate only for this independent synthetic generator;
    # real incidents may need group/block resampling. Record that limitation.
    rng = np.random.default_rng(42)
    bootstraps = []
    for _ in range(200):
        sample = rng.integers(0, len(test), len(test))
        if len(np.unique(y[test][sample])) == 2:
            bootstraps.append(average_precision_score(y[test][sample], test_scores[sample]))
    ci = np.quantile(bootstraps, [0.025, 0.975]).tolist() if bootstraps else None
    drift = {name: psi(x[train, i], x[val_end:, i]) for i, name in enumerate(FEATURES)}
    report = {
        "schema_version": 1,
        "dataset_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        "sklearn_version": sklearn.__version__,
        "as_of": as_of,
        "threshold": threshold,
        "split": {
            "train": len(train),
            "validation": len(validation),
            "test": len(test),
            "train_label_cutoff": float(t[train_end]),
            "validation_label_cutoff": float(t[val_end]),
            "withheld_test_labels": int((available[val_end:] > as_of).sum()),
        },
        "validation": metrics(y[validation], val_scores, threshold),
        "test": evaluation,
        "baseline_test": metrics(
            y[test], baseline.predict_proba(x[test])[:, 1], baseline_threshold
        ),
        "test_average_precision_iid_bootstrap_95pct": ci,
        "feature_psi": drift,
        "drift_review_features": [
            key for key, value in drift.items() if value is not None and value >= 0.2
        ],
        "limitations": [
            "Synthetic observations do not establish real incident prediction quality.",
            "PSI >= 0.2 is a review heuristic, not proof of concept drift or a calibrated test.",
            "A random bootstrap assumes independent observations; group real incidents before evaluation.",
            "Threshold maximizes validation F2; business costs and calibration require separate study.",
        ],
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    imputer, scaler, classifier = model.steps[0][1], model.steps[1][1], model.steps[2][1]
    portable = {
        "features": FEATURES,
        "imputation_medians": imputer.statistics_.tolist(),
        "means": scaler.mean_.tolist(),
        "scales": scaler.scale_.tolist(),
        "coefficients": classifier.coef_[0].tolist(),
        "intercept": float(classifier.intercept_[0]),
        "threshold": threshold,
    }
    (output / "model.json").write_text(
        json.dumps(portable, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    with (output / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["timestamp", "probability", "prediction", "mature_label"])
        scores = model.predict_proba(x[val_end:])[:, 1]
        for index, score in zip(indices[val_end:], scores):
            writer.writerow(
                [
                    t[index],
                    float(score),
                    int(score >= threshold),
                    int(y[index]) if available[index] <= as_of else "",
                ]
            )
    (output / "split-manifest.json").write_text(
        json.dumps(
            {"train": train.tolist(), "validation": validation.tolist(), "test": test.tolist()},
            indent=2,
        ),
        encoding="utf-8",
    )
    return report


def predict_portable(model, x):
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or x.shape[1] != len(model["features"]) or np.isinf(x).any():
        raise ValueError("Feature matrix has invalid shape or infinite values")
    filled = np.where(np.isnan(x), model["imputation_medians"], x)
    normalized = (filled - model["means"]) / model["scales"]
    logits = normalized @ np.asarray(model["coefficients"]) + model["intercept"]
    return 1 / (1 + np.exp(-np.clip(logits, -700, 700)))
