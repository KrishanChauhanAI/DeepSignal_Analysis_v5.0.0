"""Model A: features, CV plumbing, positive/negative controls, and the train CLI.

Positive control: classes that differ in event statistics must be learned.
Negative control: shuffled labels on noise must give chance-level performance.
Small sizes and reduced ensembles keep these tests fast; they are not performance claims.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from trapsig.cli import main
from trapsig.config import config_from_dict, validate_config, validate_training
from trapsig.events import Event
from trapsig.features import (
    FEATURE_GROUPS,
    acf_at_lags,
    event_features,
    extract_features,
    group_of_features,
    residual_sigma,
    resolve_groups,
    welch_band_powers,
)
from trapsig.pipeline import process_recording
from trapsig.splits import grouped_stratified_folds
from trapsig.synthetic import SyntheticSpec, write_synthetic_dataset
from trapsig.train import (
    TrainError,
    _base_model,
    balanced_accuracy,
    cv_predictions,
    load_prepared,
)

BANDS = [[1.0, 10.0], [10.0, 100.0], [100.0, 1000.0], [1000.0, 5000.0], [45.0, 55.0]]
LAGS = [0.1, 1.0, 10.0]


def _cfg_dict(source: Path, runs: Path, *, n_repeats: int = 2, n_splits: int = 3,
              training: dict | None = None, fs: float = 100_000.0, duration: float = 0.5) -> dict:
    return {
        "run": {"name": "train_test", "seed": 7},
        "data": {
            "source_dir": str(source), "file_glob": "*.dat", "label_level": "file", "label_from": "folder",
            "allow_file_level_fallback": True, "expected_duration_s": duration, "duration_tolerance_s": 0.001,
            "dat": {"encoding": "binary", "dtype": "float32", "byteorder": "little", "header_bytes": 0,
                    "n_columns": 1, "column": 0, "scale": 1.0, "units": "A"},
        },
        "sampling": {"original_hz": fs, "target_rates_hz": [10_000.0]},
        "filters": {"mode": "none"},
        "events": {"baseline_window_s": 0.1, "pre_event_window_s": 0.02, "min_duration_s": 0.001},
        "splitting": {"n_splits": n_splits, "n_repeats": n_repeats, "min_groups_per_class": n_splits},
        "training": {"analysis_rate_hz": 10_000.0, "models": ["logreg"], "lr_c_grid": [0.1, 1.0],
                     "rf_n_estimators": 10, "hgb_max_iter": 10, "n_permutations": 0,
                     "bootstrap_resamples": 20, **(training or {})},
        "output": {"runs_dir": str(runs)},
    }


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------

def test_event_features_are_nan_when_there_are_no_events():
    f = event_features([], n_samples=10_000, fs=10_000.0, i0_level=1.0)
    assert f["event_n"] == 0.0
    assert f["event_has_any"] == 0.0
    assert f["event_dur_frac_of_time"] == 0.0
    assert math.isnan(f["event_dip_mean"])
    assert math.isnan(f["event_log10_dur_median"])


def test_event_features_summarise_known_events():
    ev = [Event(0, 10, 1000.0, 0.0, 0.01, 0.01, 1.0, "rolling_baseline", -0.2, -0.2, 0.3,
                False, False),
          Event(100, 140, 1000.0, 0.1, 0.14, 0.04, 1.0, "rolling_baseline", -0.4, -0.4, 0.5,
                False, False)]
    f = event_features(ev, n_samples=1000, fs=1000.0, i0_level=2.0)
    assert f["event_n"] == 2.0
    assert f["event_dip_mean"] == pytest.approx(-0.3)
    assert f["event_dip_max"] == pytest.approx(-0.2)
    assert f["event_dur_frac_of_time"] == pytest.approx(0.05)
    assert f["event_amp_rel_max"] == pytest.approx(0.25)  # 0.5 / |I0| = 0.5 / 2


def test_residual_sigma_matches_gaussian_sd():
    x = np.random.default_rng(0).normal(0, 2.0, 200_000)
    assert residual_sigma(x) == pytest.approx(2.0, rel=0.02)


def test_band_power_isolates_a_50_hz_tone():
    fs = 10_000.0
    t = np.arange(int(2 * fs)) / fs
    rng = np.random.default_rng(1)
    x = np.sin(2 * np.pi * 50.0 * t) + 1e-3 * rng.standard_normal(t.size)
    p = welch_band_powers(x, fs, BANDS)
    assert p["spec_log10_band_45_55Hz"] > p["spec_log10_band_100_1000Hz"] + 2.0
    assert set(p) >= {"spec_log10_total", "spec_loglog_slope_1_1000", "spec_log10_band_1_10Hz"}


def test_acf_is_near_zero_for_white_noise_and_near_one_for_slow_signal():
    fs = 10_000.0
    rng = np.random.default_rng(2)
    white = acf_at_lags(rng.standard_normal(100_000), fs, LAGS)
    assert all(abs(v) < 0.02 for v in white.values())
    slow = np.sin(2 * np.pi * 2.0 * np.arange(100_000) / fs)  # 2 Hz: lag 10 ms is ~0.99 correlated
    smooth = acf_at_lags(slow, fs, LAGS)
    assert smooth["acf_10ms"] > 0.9


def _plain_signal(n: int = 20_000, seed: int = 3) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    baseline = np.full(n, 1.0)
    return baseline + 1e-3 * rng.standard_normal(n), baseline


def test_extract_features_covers_every_declared_group_and_no_labels():
    fs = 10_000.0
    signal, baseline = _plain_signal()
    feats = extract_features(signal, baseline, fs, [], raw_std=1e-3, raw_level=1.0,
                             bands_hz=BANDS, acf_lags_ms=LAGS)
    prefixes = {k.split("_", 1)[0] for k in feats}
    # Every declared group contributes columns, and every column carries its group's prefix, so
    # disabling a group removes exactly the columns that group owns and nothing else.
    assert prefixes == set(FEATURE_GROUPS)
    assert set(resolve_groups(None)) == set(FEATURE_GROUPS)
    assert all(not str(k).startswith(("class", "label")) for k in feats)
    # A recording with no events still yields every key: unknown is NaN, never a silent absence.
    assert math.isnan(feats["event_dip_mean"]) and math.isnan(feats["ctx_isi_cv"])
    assert feats["event_n"] == 0.0
    assert math.isnan(feats["ctx_onset_rayleigh_r"]) and math.isnan(feats["ctx_burst_frac"])
    assert math.isfinite(feats["ctx_log10_abs_i0"])      # context features that do not need events


def test_feature_groups_gate_columns_and_group_of_features_inverts_it():
    fs = 10_000.0
    signal, baseline = _plain_signal()
    kw = dict(raw_std=1e-3, raw_level=1.0, bands_hz=BANDS, acf_lags_ms=LAGS)
    full = extract_features(signal, baseline, fs, [], **kw)
    subset = extract_features(signal, baseline, fs, [], groups=["noise", "spec", "ctx"], **kw)
    assert {k.split("_", 1)[0] for k in subset} == {"noise", "spec", "ctx"}
    assert all(full[k] == subset[k] or (math.isnan(full[k]) and math.isnan(subset[k]))
               for k in subset)
    by_group = group_of_features(list(full))
    assert sorted(by_group) == sorted(FEATURE_GROUPS)
    assert sorted(c for cols in by_group.values() for c in cols) == sorted(full)
    with pytest.raises(ValueError):
        resolve_groups(["noise", "not_a_group"])


# ---------------------------------------------------------------------------
# CV plumbing, controls
# ---------------------------------------------------------------------------

def _blobs(n_per_class: int, shift: float, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = np.vstack([rng.normal(0, 1, (n_per_class, 5)), rng.normal(shift, 1, (n_per_class, 5))])
    y = np.array(["A"] * n_per_class + ["B"] * n_per_class, dtype=object)
    groups = np.array([f"f{i}" for i in range(2 * n_per_class)], dtype=object)
    return x, y, groups


def _cfg_for_cv(tmp_path: Path, **kw):
    return config_from_dict(_cfg_dict(tmp_path, tmp_path / "runs", **kw))


def test_positive_control_separable_classes_are_learned(tmp_path):
    cfg = _cfg_for_cv(tmp_path, n_repeats=2, n_splits=3)
    x, y, groups = _blobs(30, shift=3.0, seed=10)
    plan = grouped_stratified_folds(groups, y, 3, 2, seed=7)
    pred, proba, chosen = cv_predictions("logreg", x, y, groups, plan.fold_ids, ["A", "B"], cfg)
    for r in range(2):
        assert balanced_accuracy(y, pred[r], ["A", "B"]) > 0.9
    assert np.all(np.isfinite(chosen))  # C was tuned in every outer fold


def test_negative_control_shuffled_labels_are_at_chance(tmp_path):
    cfg = _cfg_for_cv(tmp_path, n_repeats=2, n_splits=3)
    x, _, groups = _blobs(30, shift=0.0, seed=11)  # pure noise features
    y = np.random.default_rng(12).permutation(np.array(["A"] * 30 + ["B"] * 30, dtype=object))
    plan = grouped_stratified_folds(groups, y, 3, 2, seed=7)
    pred, _, _ = cv_predictions("logreg", x, y, groups, plan.fold_ids, ["A", "B"], cfg)
    scores = [balanced_accuracy(y, pred[r], ["A", "B"]) for r in range(2)]
    assert all(0.3 < s < 0.7 for s in scores), scores


def test_every_file_is_predicted_exactly_once_per_repeat(tmp_path):
    cfg = _cfg_for_cv(tmp_path, n_repeats=3, n_splits=3)
    x, y, groups = _blobs(12, shift=2.0, seed=13)
    plan = grouped_stratified_folds(groups, y, 3, 3, seed=7)
    pred, proba, _ = cv_predictions("logreg", x, y, groups, plan.fold_ids, ["A", "B"], cfg)
    assert pred.shape == (3, 24)
    assert all(p is not None for p in pred.ravel())
    assert np.allclose(proba.sum(axis=2), 1.0)


def test_missing_values_are_imputed_inside_the_model(tmp_path):
    x, y, _ = _blobs(20, shift=2.5, seed=14)
    x = x.copy()
    x[::7, 2] = np.nan  # NaN in a feature column must not crash or leak
    m = _base_model("logreg", _cfg_for_cv(tmp_path), seed=1, c_value=1.0).fit(x, y)
    assert np.isfinite(m.predict_proba(x)).all()


@pytest.mark.parametrize("name", ["rf", "hgb"])
def test_tree_models_fit_with_missing_values(tmp_path, name):
    cfg = _cfg_for_cv(tmp_path)
    x, y, _ = _blobs(20, shift=2.5, seed=15)
    x = x.copy()
    x[::5, 0] = np.nan
    m = _base_model(name, cfg, seed=1).fit(x, y)
    assert m.predict(x).shape == (40,)


# ---------------------------------------------------------------------------
# Inputs and validation
# ---------------------------------------------------------------------------

def test_load_prepared_rejects_a_repeat_count_mismatch(tmp_path):
    prep = tmp_path / "prep"
    (prep / "data" / "manifests").mkdir(parents=True)
    (prep / "data" / "manifests" / "fold_assignments.csv").write_text(
        "sample_id,class,group_id,fold_r0,fold_r1,fold_r2\nabc,A,abc,0,1,0\n", encoding="utf-8")
    cfg = _cfg_for_cv(tmp_path, n_repeats=5)
    with pytest.raises(TrainError, match="3 fold repeats.*n_repeats=5"):
        load_prepared(prep, cfg)


def test_load_prepared_requires_a_prepare_run(tmp_path):
    with pytest.raises(TrainError, match="run `trapsig prepare` first"):
        load_prepared(tmp_path, _cfg_for_cv(tmp_path))


def test_training_validation_is_separate_from_base_validation(tmp_path):
    bad = _cfg_dict(tmp_path, tmp_path / "runs", training={
        "psd_bands_hz": [[1.0, 6000.0]], "models": ["svm"], "inner_splits": 1})
    cfg = config_from_dict(bad)
    # The base configuration (used by events and prepare) is not blocked by training settings.
    assert validate_config(cfg) == []
    errs = validate_training(cfg)
    assert any("psd_bands_hz" in e for e in errs)
    assert any("models" in e for e in errs)
    assert any("inner_splits" in e for e in errs)


def test_shipped_template_has_five_repeats_and_valid_training_section():
    from trapsig.config import load_config
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "configs" / "custom_dat.yaml")
    assert cfg.splitting.n_repeats == 5
    assert validate_training(cfg) == []


# ---------------------------------------------------------------------------
# End to end (small synthetic dataset in the user's folder layout)
# ---------------------------------------------------------------------------

@pytest.fixture
def demo_cfg(tmp_path: Path) -> Path:
    src = tmp_path / "dataset"
    write_synthetic_dataset(
        src,
        SyntheticSpec(fs_hz=100_000.0, duration_s=0.5, replicates_per_class=10, file_format="dat", seed=3),
        class_folders=True,
    )
    cfg_path = tmp_path / "cfg.yaml"
    import yaml
    cfg_path.write_text(yaml.safe_dump(_cfg_dict(src, tmp_path / "runs")), encoding="utf-8")
    return cfg_path


def _latest(runs: Path, kind: str) -> Path:
    found = sorted(p for p in runs.glob(f"{kind}_*") if p.is_dir())
    assert found, f"no {kind} run under {runs}"
    return found[-1]


def _latest_prepared(runs: Path) -> Path:
    """The prepared dataset of the newest run: runs/train_<ts>/training_<ts>."""
    found = sorted(p for p in runs.glob("train_*/training_*") if p.is_dir())
    assert found, f"no prepared dataset under {runs}"
    return found[-1]


def test_end_to_end_positive_control_through_the_cli(demo_cfg, tmp_path):
    import json
    assert main(["prepare", "--config", str(demo_cfg)]) == 0
    prep = _latest_prepared(tmp_path / "runs")
    assert main(["train", "--config", str(demo_cfg), "--prepared-run", str(prep)]) == 0
    run = _latest(tmp_path / "runs", "train")
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    assert summary["n_files_used"] == 20
    assert summary["n_files_skipped_or_failed"] == 0
    assert summary["models"]["logreg"]["per_repeat"]["balanced_accuracy"]["mean"] > 0.8
    assert summary["models"]["logreg"]["aggregated"]["balanced_accuracy_bootstrap_ci95"]["method"].startswith(
        "file-level bootstrap")
    assert summary["design_notes"]["holdout"].startswith("none")
    assert (run / "features.csv").is_file()
    assert (run / "predictions.csv").is_file()
    assert (run / "metrics.csv").is_file()
    assert (run / "confusion_logreg.csv").is_file()
    assert (run / "config_used.yaml").is_file()
    assert (run / "confusion_dummy_majority.csv").is_file()


def test_cli_train_rejects_invalid_training_settings(demo_cfg, tmp_path):
    import yaml
    assert main(["prepare", "--config", str(demo_cfg)]) == 0
    prep = _latest_prepared(tmp_path / "runs")
    data = yaml.safe_load(demo_cfg.read_text(encoding="utf-8"))
    data["training"]["models"] = ["svm"]
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(data), encoding="utf-8")
    assert main(["train", "--config", str(bad), "--prepared-run", str(prep)]) == 2


def test_keep_signal_returns_processed_signal_and_baseline(demo_cfg, tmp_path):
    import yaml
    src = tmp_path / "dataset"
    cfg = config_from_dict(yaml.safe_load(demo_cfg.read_text(encoding="utf-8")))
    f = sorted(src.rglob("*.dat"))[0]
    res = process_recording(f, cfg, 10_000.0, keep_signal=True)
    assert res.signal is not None and res.baseline is not None
    assert res.signal.size == res.baseline.size == 5_000
    assert process_recording(f, cfg, 10_000.0).signal is None


@pytest.mark.parametrize("name", ["logreg", "rf", "hgb"])
def test_class_weight_reaches_every_model(tmp_path, name):
    cfg = _cfg_for_cv(tmp_path)
    model = _base_model(name, cfg, seed=1, c_value=1.0, class_weight="balanced")
    assert model.named_steps["clf"].class_weight == "balanced"
    unweighted = _base_model(name, cfg, seed=1, c_value=1.0)
    assert unweighted.named_steps["clf"].class_weight is None
