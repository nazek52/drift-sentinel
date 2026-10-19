import json
import numpy as np
import pytest
from sentinel.data import generate, read
from sentinel.evaluate import evaluate, predict_portable, psi


def test_psi_identical_shifted_and_missing():
    values = np.linspace(0, 1, 1000)
    assert psi(values, values) == pytest.approx(0)
    assert psi(values, values + 2) > 1
    assert psi(values, np.full(1000, np.nan)) > 1
    assert psi(np.full(10, np.nan), values) is None


def test_chronology_delayed_labels_and_portable_inference(tmp_path):
    path = generate(tmp_path / "data.csv", n=600)
    out = tmp_path / "run"
    report = evaluate(path, out)
    t, x, y, available = read(path)
    splits = json.loads((out / "split-manifest.json").read_text())
    train, validation, test = map(np.array, [splits["train"], splits["validation"], splits["test"]])
    assert t[train].max() < t[validation].min() < t[test].min()
    assert available[train].max() <= t[360]
    assert available[validation].max() <= t[480]
    assert available[test].max() <= t[-1]
    assert report["split"]["withheld_test_labels"] == 30
    model = json.loads((out / "model.json").read_text())
    predictions = np.genfromtxt(out / "predictions.csv", delimiter=",", names=True)
    np.testing.assert_allclose(
        predict_portable(model, x[480:]), predictions["probability"], rtol=1e-12
    )


def test_reproducibility_and_no_future_label_leakage(tmp_path):
    path = generate(tmp_path / "data.csv", n=600)
    a = evaluate(path, tmp_path / "a")
    b = evaluate(path, tmp_path / "b")
    assert a == b
    lines = path.read_text().splitlines()
    # The final 30 labels are unavailable at evaluation time. Changing them must
    # not change the model, threshold or test metrics.
    for i in range(len(lines) - 30, len(lines)):
        fields = lines[i].split(",")
        fields[-2] = str(1 - int(fields[-2]))
        lines[i] = ",".join(fields)
    path.write_text("\n".join(lines) + "\n")
    c = evaluate(path, tmp_path / "c")
    assert a["test"] == c["test"] and a["threshold"] == c["threshold"]
    assert (tmp_path / "a/model.json").read_text() == (tmp_path / "c/model.json").read_text()


def test_invalid_order_rejected(tmp_path):
    path = generate(tmp_path / "data.csv", n=300)
    lines = path.read_text().splitlines()
    lines[1], lines[2] = lines[2], lines[1]
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="increasing"):
        evaluate(path, tmp_path / "run")


def test_asof_and_empty_class_validation(tmp_path):
    path = generate(tmp_path / "data.csv", n=300)
    with pytest.raises(ValueError, match="as_of"):
        evaluate(path, tmp_path / "run", as_of=10)
    lines = path.read_text().splitlines()
    for i in range(1, len(lines)):
        fields = lines[i].split(",")
        fields[-2] = "0"
        lines[i] = ",".join(fields)
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="both classes"):
        evaluate(path, tmp_path / "run")
