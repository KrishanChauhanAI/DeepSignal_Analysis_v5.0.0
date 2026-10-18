"""Tests for the four-branch architecture: segmentation, features, model, leakage, CV plumbing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from trapsig import quad
from trapsig.config import load_config


def _cfg(write_yaml: Any, base: dict[str, Any], **quad_over: Any) -> Any:
    d = dict(base)
    d["splitting"] = {"n_splits": 2, "n_repeats": 1, "min_groups_per_class": 1}
    d["training"] = {"analysis_rate_hz": 1000.0}
    d["deep"] = {"validation_fraction": 0.34}
    d["quad"] = {"enabled": True, "embed_dim": 4, "t_channels": [4], "t_dilations": [1],
                 "t_kernel": 3, "t_transformer_layers": 1, "t_nhead": 2,
                 "s_nfft": 32, "s_hop": 16, "s_channels": [4],
                 "pretrain_epochs": 1, "epochs": 1, "batch_size": 2, "warmup_epochs": 0,
                 "fusion_layers": 1, "fusion_nhead": 2,
                 "episodes": 4, "shots": [1], "queries_per_class": 1,
                 **quad_over}
    return load_config(write_yaml(d, "quad.yaml"))


def test_segment_whole_file_default(write_yaml: Any, base_config_dict: dict[str, Any]) -> None:
    cfg = _cfg(write_yaml, base_config_dict)
    x = np.sin(np.arange(500) / 10.0)
    w = quad.segment_signal(x, 1000.0, cfg)
    assert w.shape == (1, 500)


def test_segment_overlap_and_stride(write_yaml: Any, base_config_dict: dict[str, Any]) -> None:
    d = dict(base_config_dict)
    d["splitting"] = {"n_splits": 2, "n_repeats": 1, "min_groups_per_class": 1}
    d["training"] = {"analysis_rate_hz": 1000.0}
    d["segmentation"] = {"window_s": 0.1, "overlap": 0.5}
    d["quad"] = {"enabled": True}
    cfg2 = load_config(write_yaml(d, "seg.yaml"))
    x = np.arange(1000, dtype=float)
    w = quad.segment_signal(x, 1000.0, cfg2)
    assert w.shape[1] == 100
    assert w.shape[0] > 1
    # determinism
    w2 = quad.segment_signal(x, 1000.0, cfg2)
    assert np.array_equal(w, w2)


def test_window_features_names_and_finite(write_yaml: Any,
                                          base_config_dict: dict[str, Any]) -> None:
    cfg = _cfg(write_yaml, base_config_dict)
    bands = [list(b) for b in cfg.training.psd_bands_hz]
    names = quad.window_feature_names(bands)
    rng = np.random.default_rng(0)
    feats = quad.window_features(rng.normal(size=256), 1000.0, bands)
    assert set(feats) == set(names)
    finite = [k for k, v in feats.items() if np.isfinite(v)]
    assert len(finite) >= len(names) * 0.8      # most features defined on a normal window
    p = feats.get("perm_entropy")
    if p is not None and np.isfinite(p):
        assert 0.0 <= float(p) <= 1.0


def test_model_gates_sum_to_one_and_ablation(write_yaml: Any,
                                             base_config_dict: dict[str, Any]) -> None:
    pytest.importorskip("torch")
    import torch

    cfg = _cfg(write_yaml, base_config_dict)
    rng = np.random.default_rng(0)
    x = torch.tensor(rng.normal(size=(3, 256)), dtype=torch.float32)
    f = torch.tensor(rng.normal(size=(3, 5)), dtype=torch.float32)
    m = quad.QuadModel(cfg, 2, 5, 256)
    logits, emb, gates = m(x, f)
    assert logits.shape == (3, 2)
    assert emb.shape == (3, cfg.quad.embed_dim)
    assert gates.shape[1] == m.n_branches == 3
    assert np.allclose(gates.sum(dim=1).detach().numpy(), 1.0, atol=1e-5)
    m2 = quad.QuadModel(cfg, 2, 5, 256, ablate=["temporal"])
    assert m2.use_t is False and m2.n_branches == 2
    logits2, emb2, gates2 = m2(x, f)
    assert logits2.shape == (3, 2)
    assert np.allclose(gates2.sum(dim=1).detach().numpy(), 1.0, atol=1e-5)
    with pytest.raises(quad.QuadError):
        quad.QuadModel(cfg, 2, 5, 256, ablate=["temporal", "spectral", "engineered"])


def test_imputer_is_train_only() -> None:
    rng = np.random.default_rng(0)
    tr = rng.normal(size=(20, 3))
    tr[0, 0] = np.nan
    te = tr + 100.0                        # shifted test block: must not move the median
    med, iqr = quad._imputer_state(tr)
    filled = quad._apply_imputer(te, med, iqr)
    assert np.all(np.isfinite(filled))
    assert np.allclose(med, np.nanmedian(tr, axis=0))


def test_cv_quad_end_to_end(tmp_path: Path, write_yaml: Any,
                            base_config_dict: dict[str, Any]) -> None:
    pytest.importorskip("torch")
    cfg = _cfg(write_yaml, base_config_dict)
    cache = tmp_path / "cache"
    cache.mkdir()
    rng = np.random.default_rng(0)
    sids = [f"f{i}" for i in range(12)]
    y = np.array(["A", "B"] * 6, dtype=object)
    ok_rows = []
    for i, sid in enumerate(sids):
        sig = rng.normal(size=256) + (3.0 if y[i] == "A" else -3.0)
        np.save(cache / f"{sid}.npy", sig)
        ok_rows.append({"sample_id": sid, "class": str(y[i]), "status": "ok"})
    tensors = quad.build_quad_tensors(str(cache), ok_rows, cfg)
    assert tensors is not None
    assert tensors["x"].shape[0] == 12
    fold_ids = np.array([[0] * 6 + [1] * 6])
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    res = quad.cv_quad(tensors, sids, y, fold_ids, ["A", "B"], cfg, run_dir)
    assert res.proba.shape == (1, 12, 2)
    assert res.mask.all()
    rows = res.proba.sum(axis=2)
    assert np.allclose(rows[np.isfinite(rows)], 1.0, atol=1e-5)
    assert abs(sum(res.gates_mean.values()) - 1.0) < 1e-4
    assert res.fewshot_rows                       # episodes were drawn
    for r in res.fewshot_rows:
        assert 0.0 <= float(r["accuracy"]) <= 1.0
    # ablated variant runs through the same path
    res2 = quad.cv_quad(tensors, sids, y, fold_ids, ["A", "B"], cfg, run_dir, ablate=["spectral"])
    assert "spectral" not in res2.gates_mean


def test_cv_quad_determinism(tmp_path: Path, write_yaml: Any,
                             base_config_dict: dict[str, Any]) -> None:
    pytest.importorskip("torch")
    import torch

    torch.set_num_threads(1)
    cfg = _cfg(write_yaml, base_config_dict, pretrain_epochs=0)
    cache = tmp_path / "cache"
    cache.mkdir()
    rng = np.random.default_rng(1)
    sids = [f"g{i}" for i in range(4)]
    y = np.array(["A", "B", "A", "B"], dtype=object)
    ok_rows = []
    for i, sid in enumerate(sids):
        np.save(cache / f"{sid}.npy", rng.normal(size=128))
        ok_rows.append({"sample_id": sid, "class": str(y[i]), "status": "ok"})
    tensors = quad.build_quad_tensors(str(cache), ok_rows, cfg)
    assert tensors is not None
    fold_ids = np.array([[0, 1, 0, 1]])
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    a = quad.cv_quad(tensors, sids, y, fold_ids, ["A", "B"], cfg, run_dir)
    b = quad.cv_quad(tensors, sids, y, fold_ids, ["A", "B"], cfg, run_dir)
    assert np.allclose(a.proba, b.proba, atol=1e-6)


def test_validators_reject_bad_quad(write_yaml: Any, base_config_dict: dict[str, Any]) -> None:
    from trapsig.config import validate_quad, validate_sampling_menu, validate_segmentation

    d = dict(base_config_dict)
    d["quad"] = {"enabled": True, "embed_dim": 0}
    cfg = load_config(write_yaml(d, "bad.yaml"))
    errs = validate_quad(cfg)
    assert any("embed_dim" in e for e in errs)

    d2 = dict(base_config_dict)
    d2["segmentation"] = {"window_s": 0.1, "overlap": 1.5}
    cfg2 = load_config(write_yaml(d2, "bad2.yaml"))
    assert validate_segmentation(cfg2)

    d3 = dict(base_config_dict)
    d3["sampling"] = {"original_hz": 100000, "target_rates_hz": [10000],
                      "rate_menu_hz": [], "allow_off_menu_rates": False}
    cfg3 = load_config(write_yaml(d3, "bad3.yaml"))
    errs3 = validate_sampling_menu(cfg3)
    assert isinstance(errs3, list)


# ---------------------------------------------------------------------------
# Round 2: registry tiers, triplet loss, scheduling, early stopping, checkpoints
# ---------------------------------------------------------------------------


def test_triplet_loss_basic() -> None:
    import torch

    from trapsig.quad import triplet_loss

    far = torch.tensor([[0.0, 0.0], [0.1, 0.0], [9.0, 9.0], [9.1, 9.0]])
    y = torch.tensor([0, 0, 1, 1])
    assert float(triplet_loss(far, y)) == 0.0            # margins already satisfied
    near = torch.tensor([[0.0, 0.0], [0.1, 0.0], [0.2, 0.0], [0.3, 0.0]])
    assert float(triplet_loss(near, y)) > 0.0
    assert torch.isfinite(triplet_loss(far[:2], torch.tensor([0, 0])))   # too small -> 0


def test_train_options_triplet_schedule_checkpoint(tmp_path: Path, write_yaml: Any,
                                                   base_config_dict: dict[str, Any]) -> None:
    """Round-2 training extras run through cv_quad: triplet term, cosine schedule, early
    stopping and a per-fold checkpoint file under <run>/checkpoints."""
    pytest.importorskip("torch")
    import torch

    cfg = _cfg(write_yaml, base_config_dict, pretrain_epochs=0, epochs=2,
               triplet_weight=0.3, lr_schedule="cosine", early_stopping_patience=1,
               feature_tiers=[1], feature_select="variance")
    cache = tmp_path / "cache"
    cache.mkdir()
    rng = np.random.default_rng(2)
    sids = [f"h{i}" for i in range(6)]
    y = np.array(["A", "B"] * 3, dtype=object)
    ok_rows = []
    for i, sid in enumerate(sids):
        np.save(cache / f"{sid}.npy", rng.normal(size=128) + (2.0 if y[i] == "A" else -2.0))
        ok_rows.append({"sample_id": sid, "class": str(y[i]), "status": "ok"})
    tensors = quad.build_quad_tensors(str(cache), ok_rows, cfg)
    assert tensors is not None
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    res = quad.cv_quad(tensors, sids, y, np.array([[0, 1, 0, 1, 0, 1]]), ["A", "B"], cfg,
                       run_dir)
    assert res.msp is not None and res.msp.shape == (6,)
    assert np.all((res.msp[np.isfinite(res.msp)] > 0) & (res.msp[np.isfinite(res.msp)] <= 1))
    cks = sorted((run_dir / "checkpoints").glob("quad_*.pt"))
    assert cks, "a checkpoint per trained fold is expected"
    blob = torch.load(cks[0], weights_only=False)
    assert blob["classes"] == ["A", "B"] and "state_dict" in blob
