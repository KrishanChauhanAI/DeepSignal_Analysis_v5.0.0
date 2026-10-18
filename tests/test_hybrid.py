"""Model B, the hybrid ensemble, explainability, conditional analyses, and reporting.

These tests check the *rules*, not the performance: window construction obeys its edge and
length rules, scaling is fitted on training data only, files a model cannot score are excluded
rather than silently averaged, the comparison is made on a matched subset, Grad-CAM has the
length of the window it explains, and anything that cannot be computed is reported as such
instead of as a number.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from trapsig.cli import main
from trapsig.config import config_from_dict, validate_config, validate_modelling
from trapsig.deep import build_model, count_parameters, fit_temperature, validation_split
from trapsig.events import Event
from trapsig.explain import feature_attributions, gradcam_1d, integrated_gradients_1d, occlusion_1d
from trapsig.hybrid import combine_probabilities, gate_g7, hybrid_predictions
from trapsig.synthetic import SyntheticSpec, write_synthetic_dataset
from trapsig.windows import (
    aggregate_to_file,
    apply_robust_scale,
    cut_windows,
    fit_robust_scale,
    resolve_window_length_ms,
)

torch = pytest.importorskip("torch", reason="Model B tests need PyTorch")


# ---------------------------------------------------------------------------
# Windows
# ---------------------------------------------------------------------------

def _events(onsets: list[int], dur: int = 10) -> list[Event]:
    return [Event(o, o + dur, 1000.0, o / 1000.0, (o + dur) / 1000.0, dur / 1000.0, 1.0,
                  "rolling_baseline", -0.3, -0.3, 0.4, False, False) for o in onsets]


def test_window_length_is_derived_from_event_durations_and_clamped():
    durs = [0.004, 0.006, 0.008, 0.010]                      # median 7 ms
    wms, src, stats = resolve_window_length_ms(durs, "auto", 4.0, 10.0, 500.0)
    assert src == "auto"
    assert wms == pytest.approx(28.0)                        # median 7 ms x 4
    assert stats["median_ms"] == pytest.approx(7.0)
    short, src2, _ = resolve_window_length_ms([0.0005], "auto", 4.0, 10.0, 500.0)
    assert src2 == "auto" and short == 10.0                  # clamped up to the minimum
    fixed, src3, _ = resolve_window_length_ms(durs, 42.0, 4.0, 10.0, 500.0)
    assert src3 == "config" and fixed == 42.0
    none, src4, _ = resolve_window_length_ms([], "auto", 4.0, 10.0, 500.0)
    assert src4 == "auto:no-events-fallback" and math.isfinite(none)


def test_cut_windows_excludes_events_whose_window_would_run_off_the_recording():
    sig = np.zeros(1000, dtype=np.float32)
    sig[50:60] = -1.0
    sig[985:995] = -1.0                                      # too close to the end
    ev = _events([50, 985])
    w, idx, ons, n_edge = cut_windows(sig, np.zeros_like(sig), 1000.0, ev, 40, 0.25, 10)
    assert w.shape == (1, 40)                                # only the first event survives
    assert idx.tolist() == [0]
    assert n_edge == 1
    # the pre-event fraction puts the onset a quarter of the way in
    assert np.count_nonzero(w[0][8:12] < 0) >= 1


def test_windows_are_baseline_subtracted_and_scaled_from_training_data_only():
    x = np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]], dtype=np.float32)
    med, iqr = fit_robust_scale(x[:1])
    assert med == pytest.approx(2.0)
    z = apply_robust_scale(x, med, iqr)
    assert np.all(np.isfinite(z))
    assert np.abs(z).max() <= 8.0                            # clipping keeps outliers bounded


def test_aggregate_to_file_keeps_unscored_files_as_nan_and_counts_windows():
    proba = np.array([[1.0, 0.0], [0.0, 1.0], [np.nan, np.nan]])
    sids = np.array(["f1", "f1", "f2"], dtype=object)
    out, counts = aggregate_to_file(proba, sids, ["f1", "f2"])
    assert out[0] == pytest.approx([0.5, 0.5])
    assert np.all(np.isnan(out[1]))                          # never predicted, never invented
    assert counts.tolist() == [2, 0]


def test_aggregate_to_file_majority_vote_counts_votes():
    proba = np.array([[0.9, 0.1], [0.9, 0.1], [0.2, 0.8]])
    sids = np.array(["f1", "f1", "f1"], dtype=object)
    out, _ = aggregate_to_file(proba, sids, ["f1"], rule="majority_vote")
    assert out[0, 0] == pytest.approx(2 / 3)


def test_combine_probabilities_flags_model_a_fallback_rows():
    a = np.full((1, 3, 2), 0.5)
    b = np.full((1, 3, 2), np.nan)
    b[0, 1] = [0.9, 0.1]
    has = np.array([False, True, False])
    out, src = combine_probabilities(a, b, 0.5, has)
    assert (src[0] == np.array(["model_a_only", "hybrid", "model_a_only"], dtype=object)).all()
    assert out[0, 1] == pytest.approx([0.7, 0.3])
    assert out.sum(axis=2).ravel() == pytest.approx(np.ones(3))
    pred = hybrid_predictions(out, ["A", "B"])
    assert pred[0, 1] == "A" and pred[0, 0] == "A"


# ---------------------------------------------------------------------------
# Architecture and training protocol
# ---------------------------------------------------------------------------

def _deep_cfg(**over: object):
    base = {
        "data": {"source_dir": "/tmp", "label_from": "folder", "label_level": "file"},
        "deep": {"window_ms": 50.0, "channels": 16, "n_blocks": 2, "epochs": 3,
                 "batch_size": 4, **over},
    }
    return config_from_dict(base)


def test_cnn_is_compact_and_cam_shaped():
    cfg = _deep_cfg()
    model = build_model(500, 2, cfg)
    n = count_parameters(model)
    assert 1_000 <= n <= 100_000                             # design 11.3 target order
    out = model(torch.randn(3, 1, 500))
    assert out.shape == (3, 2)
    emb = model.embedding(torch.randn(3, 1, 500))
    assert emb.shape == (3, 16)
    feats = model.features(torch.randn(3, 1, 500))
    assert feats.dim() == 3 and feats.shape[1] == 16         # conv -> GAP -> linear is satisfiable


def test_grad_cam_and_ig_have_the_length_of_the_window_they_explain():
    from trapsig.explain import ExplainerModel

    cfg = _deep_cfg()
    model = build_model(120, 2, cfg)
    em = ExplainerModel(model=model, med=0.0, iqr=1.0, length=120, classes=["A", "B"],
                        n_parameters=count_parameters(model), note="test")
    x = np.sin(np.arange(120) / 6.0).astype(np.float32)
    g = gradcam_1d(em, x, 1)
    ig = integrated_gradients_1d(em, x, 1, steps=4)
    oc = occlusion_1d(em, x, 1, n_segments=6)
    assert g.shape == ig.shape == oc.shape == (120,)
    assert np.all(g >= 0)                                    # Grad-CAM is ReLU'd
    assert np.isfinite(ig).all() and np.isfinite(oc).all()


def test_temperature_scaling_softens_over_confident_logits():
    rng = np.random.default_rng(0)
    logits = np.stack([rng.normal(0, 8, 200), rng.normal(0, 8, 200)], axis=1)
    logits[np.arange(200), rng.integers(0, 2, 200)] += 6.0
    y = rng.integers(0, 2, 200)
    t = fit_temperature(logits, y, 2)
    assert t > 1.0                                           # over-confidence is softened


def test_validation_split_never_splits_one_file_across_both_sides():
    y = np.array([0, 1] * 30)
    g = np.array([f"f{i // 4}" for i in range(60)], dtype=object)
    mask = validation_split(y, g, 0.2, seed=1, k=2)
    assert 0 < mask.sum() < g.size
    assert not (set(g[mask].tolist()) & set(g[~mask].tolist()))


def test_validation_split_holds_every_class_the_fold_can_spare():
    """The selection metric, the temperature fit, the prototypes and the fusion head all need
    both classes on the validation side; a stratified best-effort split can miss one."""
    y = np.array([0, 0, 0, 1, 1, 1])
    g = np.array([f"f{i}" for i in range(6)], dtype=object)
    for seed in range(10):
        m = validation_split(y, g, 0.2, seed=seed, k=2)
        assert not (set(g[m].tolist()) & set(g[~m].tolist()))     # still one file, one side
        assert set(y[m].tolist()) == {0, 1}                       # both classes represented
        assert set(y[~m].tolist()) == {0, 1}                      # and the fold stays trainable

    # A class with a single file cannot spare it: the repair must not empty the training side.
    y2 = np.array([0, 0, 0, 0, 1])
    g2 = np.array(["a", "b", "c", "d", "e"], dtype=object)
    for seed in range(6):
        m2 = validation_split(y2, g2, 0.2, seed=seed, k=2)
        assert not (set(g2[m2].tolist()) & set(g2[~m2].tolist()))
        assert m2.any() and (~m2).any()
        if 1 not in set(y2[m2].tolist()):
            # the missing class is exactly the one that had a single training file
            assert list(y2[~m2]).count(1) == 1
        assert set(y2[~m2].tolist()) >= {0}


# ---------------------------------------------------------------------------
# Comparison and gate G7
# ---------------------------------------------------------------------------

def test_gate_g7_never_claims_a_win_when_intervals_overlap():
    g = gate_g7(0.80, "logreg", (0.60, 0.95), 0.90, "cnn1d", (0.70, 1.00), 40)
    assert g["verdict"] == "indistinguishable at this sample size"
    assert g["ci_overlap"] is True
    g2 = gate_g7(0.60, "logreg", (0.50, 0.70), 0.90, "cnn1d", (0.80, 1.00), 40)
    assert g2["verdict"] == "Model B beats Model A"
    g3 = gate_g7(0.90, "logreg", (0.80, 1.00), 0.60, "cnn1d", (0.50, 0.70), 40)
    assert g3["verdict"] == "Model A beats Model B"
    g4 = gate_g7(None, "logreg", None, 0.9, "cnn1d", None, 40)
    assert g4["verdict"] == "not evaluable"


# ---------------------------------------------------------------------------
# Explainability rules
# ---------------------------------------------------------------------------

def test_binary_linear_shap_expands_the_single_coefficient_row():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(40, 6))
    y = np.array(["A"] * 20 + ["B"] * 20, dtype=object)
    x[:20, 0] += 2.0                                         # one informative feature
    g = np.array([f"f{i}" for i in range(40)], dtype=object)
    out = feature_attributions(x, y, g, [f"c{i}" for i in range(6)], ["A", "B"], _deep_cfg(), 1)
    lr = out["models"]["logreg"]
    assert lr["binary_coefficient_expanded"] is True
    assert set(lr["top_features_by_class"]) == {"A", "B"}
    # the informative feature must dominate both class attributions
    for c in ("A", "B"):
        assert lr["top_features_by_class"][c][0][0] == "c0"


# ---------------------------------------------------------------------------
# Configuration rules
# ---------------------------------------------------------------------------

def test_deep_configuration_rules_are_enforced():
    bad = _deep_cfg(window_ms=0.1)                           # 1 sample at 10 kHz
    assert any("deep.window_ms" in e for e in validate_modelling(bad))
    assert any("deep.window_ms" in e for e in validate_modelling(_deep_cfg(window_ms=-1)))
    assert any("deep.kernel_size" in e for e in validate_modelling(_deep_cfg(kernel_size=4)))
    assert any("deep.model_a_weight" in e for e in validate_modelling(_deep_cfg(model_a_weight=2.0)))
    ok = _deep_cfg(window_ms=50.0)
    assert validate_modelling(ok) == []
    assert validate_config(ok) == []                         # base rules are unaffected


def test_advanced_and_explain_gates_require_the_encoder():
    cfg = config_from_dict({"data": {"source_dir": "/tmp", "label_from": "folder",
                                     "label_level": "file"},
                            "deep": {"enabled": False},
                            "advanced": {"few_shot": True},
                            "explain": {"grad_cam_1d": True}})
    errs = validate_modelling(cfg)
    assert any("few_shot needs the CNN encoder" in e for e in errs)
    assert any("grad_cam_1d" in e for e in errs)


def test_report_dpi_bounds_are_enforced():
    cfg = config_from_dict({"data": {"source_dir": "/tmp", "label_from": "folder",
                                     "label_level": "file"}, "report": {"dpi": 5}})
    assert any("report.dpi" in e for e in validate_modelling(cfg))


# ---------------------------------------------------------------------------
# End to end: one command, hybrid, explain, report
# ---------------------------------------------------------------------------

def _hybrid_cfg_dict(source: Path, runs: Path) -> dict:
    return {
        "run": {"name": "hybrid_test", "seed": 7},
        "data": {"source_dir": str(source), "file_glob": "*.dat", "label_level": "file",
                 "label_from": "folder", "allow_file_level_fallback": True,
                 "expected_duration_s": 0.5, "duration_tolerance_s": 0.001,
                 "dat": {"encoding": "binary", "dtype": "float32", "byteorder": "little",
                         "header_bytes": 0, "n_columns": 1, "column": 0, "scale": 1.0,
                         "units": "A"}},
        "sampling": {"original_hz": 100_000.0, "target_rates_hz": [10_000.0]},
        "filters": {"mode": "none"},
        "events": {"baseline_window_s": 0.1, "pre_event_window_s": 0.02, "min_duration_s": 0.001},
        "splitting": {"n_splits": 3, "n_repeats": 2, "min_groups_per_class": 3},
        "training": {"analysis_rate_hz": 10_000.0, "models": ["logreg"], "lr_c_grid": [0.1, 1.0],
                     "rf_n_estimators": 10, "hgb_max_iter": 10, "n_permutations": 0,
                     "bootstrap_resamples": 20, "ablation": "groups", "ablation_model": "logreg",
                     "redundancy_threshold": 0.98},
        "deep": {"enabled": True, "model_set": "hybrid", "window_ms": "auto", "window_ms_min": 10.0,
                 "window_ms_max": 200.0, "channels": 8, "n_blocks": 2, "stem_stride": 2,
                 "epochs": 3, "batch_size": 4, "early_stopping_patience": 2,
                 "n_permutations": 0, "validation_fraction": 0.3,
                 "metric_loss": "supcon", "metric_loss_weight": 0.2, "metric_temperature": 0.1,
                 "prototypical_head": True, "proto_distance": "cosine", "fusion": "both"},
        "advanced": {"few_shot": True, "few_shot_k": [1, 50], "few_shot_episodes": 4,
                     "novelty": True},
        "explain": {"enabled": True, "shap_background": 4, "integrated_gradients_steps": 4,
                    "occlusion_segments": 4, "n_explained_windows": 2, "shortcut_probes": False},
        "report": {"plots": True, "pdf": True, "csv": True, "dpi": 60, "max_plots_per_class": 1},
        "output": {"runs_dir": str(runs)},
    }


@pytest.fixture(scope="module")
def hybrid_run(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    tmp = tmp_path_factory.mktemp("hybrid")
    src = tmp / "dataset"
    write_synthetic_dataset(src, SyntheticSpec(fs_hz=100_000.0, duration_s=0.5,
                                               replicates_per_class=8, file_format="dat", seed=3),
                            class_folders=True)
    cfg_path = tmp / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(_hybrid_cfg_dict(src, tmp / "runs")), encoding="utf-8")
    assert main(["run", "--config", str(cfg_path)]) == 0
    runs = sorted(p for p in (tmp / "runs").glob("train_*"))
    return runs[-1], tmp


def test_run_command_produces_every_deliverable(hybrid_run: tuple[Path, Path]) -> None:
    run_dir, tmp = hybrid_run
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["model_set"] == "hybrid"
    assert set(summary["models"]) >= {"logreg", "cnn1d", "hybrid", "dummy_majority"}
    # One encoder, four readings of its embedding: the softmax head, the prototype head, the
    # fusion head, and the probability blend must all be reported as models of their own.
    assert set(summary["models"]) >= {"proto_cnn", "fusion"}
    assert summary["models"]["proto_cnn"]["kind"] == "prototype"
    assert summary["models"]["fusion"]["kind"] == "fusion"
    assert summary["models"]["fusion"]["fitted_on"].startswith("inner-validation files")
    assert summary["deep"]["fusion_rule"] == "both"
    assert summary["deep"]["metric_loss"] == "supcon"
    assert summary["deep"]["prototypical_head"] is True
    assert summary["deep"]["n_prototyped_files"] > 0
    assert "embedding" in summary["design_notes"]["representation_flow"]
    # Model B must say how many files it could score, and Model A must be scored on the same set.
    assert summary["models"]["cnn1d"]["n_files_evaluated"] <= summary["n_files_used"]
    matched = summary["comparison"]["matched_subset"]
    assert matched["n_files"] == summary["models"]["cnn1d"]["n_files_evaluated"]
    assert "gate_G7" in matched and "verdict" in matched["gate_G7"]
    for name in ("features.csv", "predictions.csv", "metrics.csv", "windows.csv",
                 "model_comparison.csv", "report_models.csv", "report_limitations.csv",
                 "report_information_gaps.csv", "config_used.yaml", "budget_estimate.json",
                 "ablation.csv", "feature_redundancy.csv"):
        assert (run_dir / name).is_file(), name
    # The ablation refits on the same folds, so every variant is a paired comparison.
    assert summary["ablation"]["status"] == "computed"
    assert summary["ablation"]["model"] == "logreg"
    assert {r["variant"] for r in summary["ablation"]["rows"]} >= {"all_groups"}
    assert any(v.startswith("only:") for v in
               (r["variant"] for r in summary["ablation"]["rows"]))
    assert summary["features"]["n_features"] > 40
    assert set(summary["features"]["computed"]) >= {"noise", "event", "ctx", "spec", "tf"}
    assert summary["feature_redundancy"]["threshold"] == 0.98
    assert "never dropped" in summary["feature_redundancy"]["policy"]
    assert (run_dir / "figures").is_dir() and any((run_dir / "figures").glob("*.png"))
    assert (run_dir / "checkpoints").is_dir() and any((run_dir / "checkpoints").glob("*.pt"))
    assert (run_dir / "logs" / "deep_training_log.jsonl").is_file()
    expl = json.loads((run_dir / "explainability.json").read_text(encoding="utf-8"))
    assert "temporal" in expl["methods"]
    assert (run_dir / "attributions.npz").is_file()
    adv = json.loads((run_dir / "advanced_analyses.json").read_text(encoding="utf-8"))
    # K=1 is feasible, K=50 is not, and the report must say SKIPPED with a reason (D32).
    assert adv["few_shot_results"]["results"]["k=1"]["status"] == "computed"
    assert adv["few_shot_results"]["results"]["k=50"]["status"] == "SKIPPED"
    # Novelty is not evaluable without an unknown class, and says so.
    assert adv["novelty_results"]["status"] == "not evaluable"
    assert adv["novelty_results"]["leave_one_class_out_diagnostic"]
    if (run_dir / "report.pdf").is_file():
        assert (run_dir / "report.pdf").stat().st_size > 1000
    else:
        status = summary["downstream"]["pdf"]
        assert status["status"] == "skipped" and "install" in status["reason"].lower()


def test_predictions_mark_unscored_files_and_hybrid_fallback(hybrid_run: tuple[Path, Path]) -> None:
    run_dir, _ = hybrid_run
    import csv as _csv

    with (run_dir / "predictions.csv").open(encoding="utf-8", newline="") as fh:
        rows = list(_csv.DictReader(fh))
    cnn = [r for r in rows if r["model"] == "cnn1d" and r["repeat"] == "0"]
    assert {r["evaluated"] for r in cnn} <= {"True", "False"}
    hyb = [r for r in rows if r["model"] == "hybrid" and r["repeat"] == "0"]
    sources = {r["hybrid_source"] for r in hyb}
    assert sources <= {"hybrid", "model_a_only"}
    # a hybrid row is never a blank average: every file is either blended or flagged
    assert all(r["pred"] for r in hyb)


def test_run_dry_run_processes_nothing(tmp_path: Path) -> None:
    src = tmp_path / "dataset"
    write_synthetic_dataset(src, SyntheticSpec(fs_hz=100_000.0, duration_s=0.5,
                                               replicates_per_class=4, file_format="dat", seed=5),
                            class_folders=True)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(yaml.safe_dump(_hybrid_cfg_dict(src, tmp_path / "runs")), encoding="utf-8")
    assert main(["run", "--config", str(cfg), "--dry-run"]) == 0
    dry = sorted(p for p in (tmp_path / "runs").glob("dryrun_*"))[-1]
    b = json.loads((dry / "budget_estimate.json").read_text(encoding="utf-8"))
    assert b["status"] == "estimate"
    assert b["n_probe_files"] >= 1
    assert b["seconds_per_file_measured"] > 0
    assert any("measured" in s for s in b["basis"])
    assert not list((tmp_path / "runs").glob("train_*"))       # nothing was trained


def test_quick_overrides_are_reported_and_applied(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    from trapsig.cli import _apply_overrides

    cfg = config_from_dict(_hybrid_cfg_dict(tmp_path, tmp_path / "runs"))
    new, notes = _apply_overrides(cfg, quick=True, model_set=None, no_pdf=False, no_explain=False)
    assert new.splitting.n_repeats == 2
    assert new.training.n_permutations == 0
    assert new.deep.epochs <= 8
    assert new.explain.enabled is False
    assert new.report.pdf is False
    assert notes and "--quick" in notes[0]


def test_missing_torch_degrades_to_model_a_without_failing(monkeypatch: pytest.MonkeyPatch,
                                                            tmp_path: Path) -> None:
    import trapsig.train as T

    monkeypatch.setattr(T, "deep_dependency_status", lambda: {
        "available": False, "reason": "ModuleNotFoundError: No module named 'torch'",
        "install_hint": "pip install torch"})
    src = tmp_path / "dataset"
    write_synthetic_dataset(src, SyntheticSpec(fs_hz=100_000.0, duration_s=0.5,
                                               replicates_per_class=5, file_format="dat", seed=9),
                            class_folders=True)
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(_hybrid_cfg_dict(src, tmp_path / "runs")), encoding="utf-8")
    assert main(["prepare", "--config", str(cfg_path)]) == 0
    prep = sorted(p for p in (tmp_path / "runs").glob("train_*/training_*"))[-1]
    assert main(["train", "--config", str(cfg_path), "--prepared-run", str(prep)]) == 0
    run_dir = prep.parent          # training continues in the run folder the dataset belongs to
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["model_set"] == "features"
    assert summary["deep"]["unavailable"]["action_taken"].startswith("ran Model A only")
    assert "pip install torch" in summary["deep"]["unavailable"]["install_hint"]
