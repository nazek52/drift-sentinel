import csv
from pathlib import Path
import numpy as np

FEATURES = ("error_ratio", "latency_ratio", "queue_depth", "cpu_ratio", "deploy_age_hours")
COLUMNS = ("timestamp", *FEATURES, "escalated", "label_available_at")


def generate(path, n=2400, seed=42):
    """Synthetic incident snapshots, not collected incidents or production evidence."""
    if n < 300:
        raise ValueError("Use at least 300 observations")
    rng = np.random.default_rng(seed)
    timestamp = np.arange(n, dtype=float)
    error = rng.beta(1.2, 18, n)
    latency = rng.lognormal(0, 0.45, n)
    queue = rng.gamma(2, 12, n)
    cpu = rng.beta(5, 3, n)
    age = rng.exponential(30, n)
    changed = timestamp >= n * 0.8
    latency[changed] *= 1.65
    queue[changed] *= 1.8
    score = -4.5 + 12 * error + 1.4 * latency + 0.025 * queue + 1.2 * cpu - 0.008 * age
    # A changed queue implementation weakens the old queue/escalation relationship.
    score[changed] -= 0.035 * queue[changed]
    y = rng.binomial(1, 1 / (1 + np.exp(-score)))
    x = np.column_stack((error, latency, queue, cpu, age))
    x[rng.random(x.shape) < 0.015] = np.nan
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(COLUMNS)
        for i in range(n):
            writer.writerow(
                [i, *("" if np.isnan(v) else float(v) for v in x[i]), int(y[i]), i + 30]
            )
    return target


def read(path):
    with open(path, newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(COLUMNS):
            raise ValueError(f"CSV columns must be exactly {','.join(COLUMNS)}")
        rows = []
        for row in reader:
            if len(rows) >= 1_000_000:
                raise ValueError("Maximum one million observations")
            try:
                rows.append([float(row[c]) if row[c] != "" else np.nan for c in COLUMNS])
            except (ValueError, TypeError) as exc:
                raise ValueError(f"Non-numeric value at row {len(rows) + 2}") from exc
    values = np.asarray(rows, dtype=float)
    if len(values) < 300:
        raise ValueError("At least 300 rows are required")
    t, x, y, available = values[:, 0], values[:, 1:-2], values[:, -2], values[:, -1]
    if not np.isfinite(t).all() or not (np.diff(t) > 0).all():
        raise ValueError("Timestamps must be finite, unique, and strictly increasing")
    if not np.isfinite(available).all() or (available < t).any():
        raise ValueError("Label availability must be finite and no earlier than the observation")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("escalated must be binary (0 or 1)")
    if np.isinf(x).any() or (np.nan_to_num(x, nan=0) < 0).any():
        raise ValueError("Features must be nonnegative finite values or blank")
    if np.nanmax(x[:, [0, 3]]) > 1:
        raise ValueError("error_ratio and cpu_ratio must be in [0,1]")
    return t, x, y.astype(int), available
