"""Unit tests for the nested hyperparameter search (Phase VI).

The contract these tests pin down:

* the search only ever sees the files of the outer training fold it was handed;
* a metric's direction is respected (log loss is negated, higher is better everywhere);
* a trial that cannot be scored is recorded as failed with a reason, never as score 0;
* the same seed draws the same settings, so a reported search is reproducible;
* successive halving falls back to random search, with the reason recorded, when the fold is too
  small to subsample; and
* a trial is scored with the very pipeline the fold fits, not a lookalike.
"""

from __future__ import annotations

import numpy as np
import pytest

from trapsig import optimize
from trapsig.config import load_config


def _cfg(tmp_path, **training) -> object:
    data = {
        "run": {"name": "t", "seed": 7},
        "data": {"label_level": "file", "label_from": "folder"},
        "sampling": {"original_hz": 100000, "target_rates_hz": [10000]},
        "filters": {"mode": "none"},
        "events": {"baseline_window_s": 0.1, "pre_event_window_s": 0.02,
                   "min_duration_s": 0.001},
        "splitting": {"n_splits": 3, "n_repeats": 2, "min_groups_per_class": 3},
        "output": {"runs_dir": str(tmp_path / "runs")},
        "training": {"hpo": "random", "hpo_n_iter": 4, "inner_splits": 3, **training},
    }
    path = tmp_path / "cfg.yaml"
    import yaml

    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return load_config(path)


def _data(n_per_class: int = 6, n_feat: int = 6, seed: int = 0):
    rng = np.random.default_rng(seed)
    xs, ys, gs = [], [], []
    for ci, c in enumerate(("A", "B", "C")):
        for i in range(n_per_class):
            g = f"{c}{i}"
            n = 4
            xs.append(rng.normal(loc=ci * 1.5, size=(n, n_feat)) + rng.normal(size=(n, n_feat)))
            ys += [c] * n
            gs += [g] * n
    return np.vstack(xs), np.array(ys, dtype=object), np.array(gs, dtype=object)


def test_score_predictions_respects_metric_direction() -> None:
    classes = ["A", "B"]
    y = np.array(["A", "B", "A", "B"], dtype=object)
    perfect = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    wrong = np.array([[0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    assert optimize.score_predictions(y, perfect, classes, "balanced_accuracy") == 1.0
    assert optimize.score_predictions(y, wrong, classes, "balanced_accuracy") == 0.0
    good = optimize.score_predictions(y, perfect, classes, "log_loss")
    bad = optimize.score_predictions(y, wrong, classes, "log_loss")
    assert good is not None and bad is not None
    assert good > bad, "log loss must be negated so that higher is better"


def test_undefined_metric_is_a_failed_trial_not_a_zero() -> None:
    classes = ["A", "B"]
    y = np.array(["A", "A", "A", "A"], dtype=object)      # one class: ROC-AUC is undefined
    p = np.array([[0.5, 0.5]] * 4)
    assert optimize.score_predictions(y, p, classes, "roc_auc_ovr") is None
    assert optimize.score_predictions(y, p, classes, "balanced_accuracy") is not None


def test_inner_splits_keep_only_usable_splits_and_stay_inside_the_fold() -> None:
    x, y, g = _data(n_per_class=6)
    splits = optimize.inner_splits(x, y, g, n_splits=4, seed=3)
    assert splits, "a 18-file fold with 3 classes should yield usable inner splits"
    known = set(g.tolist())
    for itr, iva in splits:
        assert set(g[itr].tolist()) <= known and set(g[iva].tolist()) <= known
        assert len({str(v) for v in y[itr].tolist()}) >= 2
        assert len({str(v) for v in y[iva].tolist()}) >= 2
        assert not (set(g[itr].tolist()) & set(g[iva].tolist())), "file-level splits only"


def test_subsample_never_splits_a_file() -> None:
    x, y, g = _data(n_per_class=8)
    xs, ys, gs = optimize._subsample_by_group(x, y, g, 0.5, seed=1)
    keep = set(gs.tolist())
    mask = np.array([str(v) in keep for v in g.tolist()])
    assert xs.shape[0] == int(mask.sum())
    assert 0 < len(keep) < len(set(g.tolist()))


def test_search_is_reproducible_and_scores_the_fold_pipeline(tmp_path) -> None:
    cfg = _cfg(tmp_path, hpo="random", hpo_n_iter=3, models=["logreg"])
    x, y, g = _data()
    a = optimize.tune_model("logreg", x, y, g, ["A", "B", "C"], cfg, seed=11)
    b = optimize.tune_model("logreg", x, y, g, ["A", "B", "C"], cfg, seed=11)
    assert [t.params for t in a.history] == [t.params for t in b.history]
    assert a.status == "computed"
    assert all(t.status == "computed" for t in a.history)
    assert a.best_params
    assert set(a.space) == {"clf__C"}
    # the trial pipeline is the fitted pipeline: imputer + scaler + estimator
    est = optimize.build_estimator("logreg", cfg, 0, None, a.best_params)
    steps = [n for n, _ in est.steps]
    assert steps == ["impute", "scale", "clf"]


def test_a_fitting_failure_is_a_recorded_failed_trial(tmp_path, monkeypatch) -> None:
    cfg = _cfg(tmp_path, hpo="random", hpo_n_iter=3, inner_splits=2)
    x, y, g = _data(n_per_class=4)

    def boom(*_a: object, **_k: object) -> object:
        raise RuntimeError("synthetic fitting failure")

    monkeypatch.setattr(optimize, "build_estimator", boom)
    res = optimize.tune_model("logreg", x, y, g, ["A", "B", "C"], cfg, seed=5)
    assert res.status == "failed"
    assert res.reason and res.best_score is None and not res.best_params
    assert res.history and all(t.status == "failed" and t.reason for t in res.history)
    assert all(t.score is None for t in res.history), "a failed trial is dropped, never scored 0"
    assert any("synthetic fitting failure" in t.reason for t in res.history)


def test_halving_falls_back_when_the_fold_is_too_small(tmp_path) -> None:
    cfg = _cfg(tmp_path, hpo="halving", hpo_n_iter=6, inner_splits=2)
    rng = np.random.default_rng(4)
    xs, ys, gs = [], [], []
    for ci, (c, n_files) in enumerate((("A", 2), ("B", 2), ("C", 1))):   # 5 files < 2 * eta
        for i in range(n_files):
            xs.append(rng.normal(loc=ci, size=(3, 5)))
            ys += [c] * 3
            gs += [f"{c}{i}"] * 3
    x, y, g = np.vstack(xs), np.array(ys, dtype=object), np.array(gs, dtype=object)
    res = optimize.tune_model("logreg", x, y, g, ["A", "B", "C"], cfg, seed=5)
    assert res.reason.startswith("successive halving needs at least")
    assert all(t.rung == 0 for t in res.history), "the fallback must not promote rungs"


def test_write_hpo_csv_keeps_every_trial(tmp_path) -> None:
    rows = [{"model": "logreg", "repeat": 0, "fold": 0, "fold_tag": "r0f0", "method": "random",
             "metric": "balanced_accuracy", "trial": 0, "rung": 0, "n_train_files": 9,
             "score": None, "n_inner_splits": 3, "status": "failed", "reason": "because",
             "params": "{}"}]
    path = optimize.write_hpo_csv(rows, tmp_path / "hpo_trials.csv")
    text = (tmp_path / "hpo_trials.csv").read_text(encoding="utf-8")
    assert str(path).endswith("hpo_trials.csv")
    assert "failed" in text and "because" in text


@pytest.mark.parametrize("name", ["logreg", "rf", "hgb"])
def test_every_searchable_model_has_a_space(tmp_path, name) -> None:
    cfg = _cfg(tmp_path)
    space = optimize.search_space(name, cfg)
    assert space, name
    with pytest.raises(KeyError):
        optimize.search_space("dummy_majority", cfg)
