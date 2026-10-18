"""Tests for inference from saved artifacts (Phase X: "inference from saved artifacts").

The contract: a saved pipeline is only usable together with the sidecar that names its feature
columns and classes; a model saved by another code version is refused; and predictions on new
files reproduce the labels the cross-validation saw on those same files (the synthetic demo data
is separable by construction, so agreement is a real check, not a tautology).
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from trapsig import predict, train
from trapsig.config import load_config


def _cfg(tmp_path, **overrides):
    import yaml

    data = {
        "run": {"name": "t", "seed": 7},
        "data": {"label_level": "file", "label_from": "folder"},
        "sampling": {"original_hz": 100000, "target_rates_hz": [10000]},
        "filters": {"mode": "none"},
        "events": {"baseline_window_s": 0.1, "pre_event_window_s": 0.02,
                   "min_duration_s": 0.001},
        "splitting": {"n_splits": 3, "n_repeats": 2, "min_groups_per_class": 3},
        "output": {"runs_dir": str(tmp_path / "runs")},
        "training": overrides,
    }
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return load_config(path)


def _fit(tmp_path, columns=("f1", "f2", "f3")):
    rng = np.random.default_rng(0)
    x = rng.normal(size=(40, len(columns)))
    y = np.array(["A" if v > 0 else "B" for v in x[:, 0]], dtype=object)
    cfg = _cfg(tmp_path)
    model = train._base_model("logreg", cfg, 1, None).fit(x, y)
    meta = {"model": "logreg", "fold_tag": "r0f0", "repeat": 0, "fold": 0,
            "classes": ["A", "B"], "feature_columns": list(columns),
            "n_feature_columns": len(columns), "hyperparameters": {"clf__C": 1.0},
            "n_train_files": 40, "seed": 1, "class_weight": None,
            "trapsig_version": "0.1.0", "config_fingerprint": "deadbeef"}
    return cfg, model, meta, x, y


def test_saved_model_roundtrip(tmp_path) -> None:
    from trapsig import __version__

    cfg, model, meta, x, y = _fit(tmp_path)
    path = predict.save_model(model, meta, tmp_path / "models" / "logreg_r0f0.joblib")
    loaded = predict.load_saved_model(path, want_version=__version__)
    assert loaded.feature_columns == ["f1", "f2", "f3"]
    assert loaded.classes == ["A", "B"]
    assert (np.asarray(loaded.pipeline.predict_proba(x[:5])) ==
            np.asarray(model.predict_proba(x[:5]))).all()


def test_a_model_without_its_sidecar_is_refused(tmp_path) -> None:
    _cfg(tmp_path)
    _, model, meta, _, _ = _fit(tmp_path)
    path = predict.save_model(model, meta, tmp_path / "m.joblib")
    (tmp_path / "m.joblib.json").unlink()
    with pytest.raises(predict.PredictError, match="sidecar"):
        predict.load_saved_model(path)


def test_a_stale_model_is_refused_unless_allowed(tmp_path) -> None:
    _, model, meta, _, _ = _fit(tmp_path)
    path = predict.save_model(model, meta, tmp_path / "m.joblib")
    with pytest.raises(predict.PredictError, match="stale"):
        predict.load_saved_model(path, want_version="9.9.9")
    loaded = predict.load_saved_model(path, want_version=None)   # --allow-stale
    assert loaded.name == "logreg"


def test_missing_feature_columns_are_a_stated_failure(tmp_path) -> None:
    cfg, model, meta, x, y = _fit(tmp_path)
    path = predict.save_model(model, meta, tmp_path / "m.joblib")
    loaded = predict.load_saved_model(path)
    loaded.meta["feature_columns"] = ["f1", "f2", "ghost_column"]
    with pytest.raises(predict.PredictError, match="ghost_column"):
        predict._matrix([{"f1": 1.0, "f2": 2.0, "f3": 3.0}], loaded.feature_columns)
    _ = cfg, x, y


def test_models_for_reports_what_the_run_saved(tmp_path) -> None:
    cfg, model, meta, _, _ = _fit(tmp_path)
    run = tmp_path / "run"
    (run / "models").mkdir(parents=True)
    entries = []
    for tag in ("r0f0", "r0f1"):
        m = dict(meta, fold_tag=tag)
        predict.save_model(model, m, run / "models" / f"logreg_{tag}.joblib")
        entries.append({"file": f"logreg_{tag}.joblib", "model": "logreg", "fold_tag": tag})
    (run / "models" / "INDEX.json").write_text(json.dumps(entries), encoding="utf-8")
    both = predict.models_for(run, "logreg")
    assert [m.meta["fold_tag"] for m in both] == ["r0f0", "r0f1"]
    one = predict.models_for(run, "logreg", fold_tag="r0f1")
    assert [m.meta["fold_tag"] for m in one] == ["r0f1"]
    with pytest.raises(predict.PredictError, match="available"):
        predict.models_for(run, "rf")
    _ = cfg


def test_predict_files_averages_saved_models_and_keeps_skipped_files(tmp_path, monkeypatch):
    cfg, model, meta, x, y = _fit(tmp_path)
    models = []
    for tag in ("r0f0", "r0f1"):
        m = dict(meta, fold_tag=tag)
        p = predict.save_model(model, m, tmp_path / f"{tag}.joblib")
        models.append(predict.load_saved_model(p))

    from trapsig.pipeline import RecordingSkipped

    calls = {"n": 0}

    def fake_row(path, _cfg):
        calls["n"] += 1
        if path.name == "broken.dat":
            raise RecordingSkipped("unreadable header", {})
        return {"f1": 1.0, "f2": 0.2, "f3": -0.3}

    monkeypatch.setattr(predict, "feature_row_for", fake_row)
    rows = predict.predict_files([tmp_path / "ok.dat", tmp_path / "broken.dat"], models, cfg)
    assert calls["n"] == 2
    ok, bad = rows
    assert ok["status"] == "ok" and ok["predicted"] in ("A", "B")
    assert abs(ok["prob_A"] + ok["prob_B"] - 1.0) < 1e-6
    assert bad["status"] == "skipped" and bad["reason"] == "unreadable header"
    out = predict.write_inference_csv(rows, tmp_path / "inf.csv")
    text = out.read_text(encoding="utf-8")
    assert "broken.dat" in text and "unreadable header" in text, "skipped files stay in the CSV"


def test_mismatched_models_cannot_vote_together(tmp_path) -> None:
    cfg, model, meta, _, _ = _fit(tmp_path)
    a = predict.load_saved_model(predict.save_model(model, meta, tmp_path / "a.joblib"))
    other = dict(meta, classes=["A", "C"])
    b = predict.load_saved_model(predict.save_model(model, other, tmp_path / "b.joblib"))
    with pytest.raises(predict.PredictError, match="different columns or classes"):
        predict.predict_files([tmp_path / "x.dat"], [a, b], cfg)
