"""Metric learning, the prototypical head, and the fusion head.

These are the parts that turn "several strategies" into one architecture: a single encoder per
fold, an auxiliary loss that shapes its embedding, and two extra readings of that embedding. The
tests check the shapes, the degenerate cases, and above all the fitting data: prototypes and the
fusion head are built from the fold's inner-validation files, and a test file must never be able
to influence another test file's prediction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest

from trapsig.config import config_from_dict, validate_deep
from trapsig.deep import build_prototypes, metric_loss, prototype_probabilities, train_fold
from trapsig.hybrid import fusion_stack
from trapsig.windows import WindowSet

torch = pytest.importorskip("torch", reason="metric-learning tests need PyTorch")

CLASSES = ["A", "B"]


def _cfg(**over: object):
    return config_from_dict({
        "data": {"source_dir": "/tmp", "label_from": "folder", "label_level": "file"},
        "splitting": {"n_splits": 3, "n_repeats": 1, "min_groups_per_class": 3},
        "training": {"analysis_rate_hz": 10_000.0, "models": ["logreg"], "lr_c_grid": [1.0],
                     "n_permutations": 0, "bootstrap_resamples": 0},
        "deep": {"enabled": True, "model_set": "hybrid", "window_ms": 20.0, "channels": 8,
                 "n_blocks": 1, "stem_stride": 2, "epochs": 2, "batch_size": 4,
                 "early_stopping_patience": 2, "n_permutations": 0, "warmup_epochs": 0,
                 **over},
    })


# ---------------------------------------------------------------------------
# The auxiliary loss
# ---------------------------------------------------------------------------

def _clustered(rng: np.random.Generator, n_per_class: int = 12, dim: int = 8):
    centres = rng.normal(0, 3, (2, dim))
    emb = np.concatenate([centres[c] + rng.normal(0, 0.2, (n_per_class, dim))
                          for c in (0, 1)])
    y = np.array([0] * n_per_class + [1] * n_per_class)
    return torch.as_tensor(emb, dtype=torch.float32), torch.as_tensor(y, dtype=torch.long)


@pytest.mark.parametrize("kind", ["supcon", "prototypical"])
def test_metric_loss_is_lower_when_the_embedding_already_separates_the_classes(kind: str):
    rng = np.random.default_rng(0)
    good, y = _clustered(rng)
    shuffled = good[torch.randperm(good.shape[0])]         # same points, labels no longer match
    l_good = float(metric_loss(good, y, kind, 0.1))
    l_bad = float(metric_loss(shuffled, y, kind, 0.1))
    assert np.isfinite(l_good) and np.isfinite(l_bad)
    assert l_good < l_bad


@pytest.mark.parametrize("kind", ["supcon", "prototypical"])
def test_metric_loss_is_zero_when_it_cannot_be_computed(kind: str):
    assert float(metric_loss(torch.zeros(5, 4), torch.zeros(5, dtype=torch.long), "none", 0.1)) == 0.0
    one = float(metric_loss(torch.randn(1, 4), torch.zeros(1, dtype=torch.long), kind, 0.1))
    assert one == 0.0                                      # a batch of one has nothing to contrast
    single_class = float(metric_loss(torch.randn(6, 4), torch.zeros(6, dtype=torch.long), kind, 0.1))
    if kind == "prototypical":
        assert single_class == 0.0                         # one prototype: nothing to be closest to
    else:
        assert np.isfinite(single_class)                   # an all-positive batch is still defined
    all_singletons = float(metric_loss(torch.randn(6, 4), torch.tensor([0, 1, 2, 3, 4, 5]),
                                       kind, 0.1))
    assert all_singletons == 0.0 if kind == "supcon" else np.isfinite(all_singletons)


def test_a_sample_is_never_its_own_positive():
    # Two identical embeddings of different classes carry no usable contrast for supcon: every
    # sample would be its own only partner, so the term is dropped rather than divided by zero.
    emb = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    y = torch.tensor([0, 1])
    assert float(metric_loss(emb, y, "supcon", 0.1)) == 0.0
    # With no negative in the batch the contrast is trivially satisfied: the term is 0, not NaN.
    assert float(metric_loss(emb, torch.tensor([0, 0]), "supcon", 0.1)) == 0.0
    # Add a negative and the term becomes positive: the same-class pair must sit closer together
    # than the odd one out.
    emb3 = torch.tensor([[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0]])
    assert float(metric_loss(emb3, torch.tensor([0, 0, 1]), "supcon", 1.0)) > 0.0


def test_the_new_deep_settings_are_validated_before_a_run_starts():
    from trapsig.config import ConfigError

    # A weighted loss of zero is a contradiction the user can act on, not a silent no-op.
    assert any("metric_loss_weight is 0" in e
               for e in validate_deep(_cfg(metric_loss="supcon", metric_loss_weight=0.0)))
    assert any("metric_loss_weight must be in" in e
               for e in validate_deep(_cfg(metric_loss_weight=50.0)))
    assert validate_deep(_cfg(metric_loss="none", metric_loss_weight=0.0)) == []
    assert any("metric_temperature" in e
               for e in validate_deep(_cfg(prototypical_head=True, metric_temperature=0.0)))
    # Unknown strategy names are rejected at load time by the declared Literal, not at train time.
    for bad in ({"metric_loss": "triplet"}, {"proto_distance": "manhattan"}, {"fusion": "voting"}):
        with pytest.raises(ConfigError):
            _cfg(**bad)
    assert validate_deep(_cfg(fusion="stack", proto_distance="euclidean",
                              metric_loss="prototypical")) == []
    assert validate_deep(_cfg(fusion="both", prototypical_head=False, metric_loss="none",
                              metric_loss_weight=0.0)) == []


# ---------------------------------------------------------------------------
# The prototypical head
# ---------------------------------------------------------------------------

def test_build_prototypes_averages_per_class_and_skips_classes_with_no_rows():
    emb = np.array([[1.0, 0.0], [3.0, 0.0], [0.0, 2.0]], dtype=float)
    y = np.array(["A", "A", "B"], dtype=object)
    protos, kept = build_prototypes(emb, y, ["A", "B", "C"], "cosine")
    assert kept == ["A", "B"]                              # C has no support row
    assert protos.shape == (2, 2)
    assert np.allclose(np.linalg.norm(protos, axis=1), 1.0)  # cosine prototypes are unit vectors
    eu, kept_eu = build_prototypes(emb, y, ["A", "B"], "euclidean")
    assert kept_eu == ["A", "B"] and eu[0] == pytest.approx([2.0, 0.0])


def test_prototype_probabilities_are_normalised_and_follow_the_declared_class_order():
    emb = np.array([[1.0, 0.0], [0.0, 1.0], [0.9, 0.1]], dtype=float)
    protos = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float)
    p = prototype_probabilities(emb, protos, ["A", "B"], ["B", "A"], "cosine", 0.1)
    assert p.shape == (3, 2)
    assert p.sum(axis=1) == pytest.approx(np.ones(3))
    assert np.argmax(p, axis=1).tolist() == [1, 0, 1]      # columns are in ["B", "A"] order
    assert p[0, 1] > p[0, 0] and p[2, 1] > 0.9             # the closest prototype wins
    assert prototype_probabilities(np.empty((0, 2)), protos, ["A", "B"], ["A", "B"], "cosine",
                                   0.1).shape == (0, 2)
    p_eu = prototype_probabilities(emb, protos, ["A", "B"], ["A", "B"], "euclidean", 0.1)
    assert np.argmax(p_eu, axis=1).tolist() == [0, 1, 0]
    # A prototype set that covers no declared class falls back to a uniform prior, never to NaN.
    p_none = prototype_probabilities(emb, protos, ["X", "Y"], ["A", "B"], "cosine", 0.1)
    assert p_none == pytest.approx(np.full((3, 2), 0.5))


# ---------------------------------------------------------------------------
# The fusion head
# ---------------------------------------------------------------------------

def first_of(y_files: np.ndarray, train_files: list[str], file_order: list[str]) -> str:
    return str(y_files[file_order.index(train_files[0])])


@dataclass
class _FakeCV:
    embeddings: np.ndarray
    extra: dict[str, Any] = field(default_factory=dict)


def _branch(n_files: int = 6, channels: int = 4, w_per_file: int = 2, seed: int = 0,
            n_repeats: int = 1, drop_tags: tuple[str, ...] = ()):
    """A DeepBranch whose validation files are a strict subset of each fold's training files."""
    from trapsig.hybrid import DeepBranch

    rng = np.random.default_rng(seed)
    file_order = [f"f{i}" for i in range(n_files)]
    y_files = np.array([CLASSES[i % 2] for i in range(n_files)], dtype=object)
    fold_ids = np.array([[i % 3 for i in range(n_files)] for _ in range(n_repeats)])
    n_windows = n_files * w_per_file
    sids = np.array([file_order[i // w_per_file] for i in range(n_windows)], dtype=object)
    labels = np.array([y_files[i // w_per_file] for i in range(n_windows)], dtype=object)
    # A class-dependent component, so a head fitted on the validation files can actually learn.
    emb = rng.normal(0, 0.3, (n_repeats, n_windows, channels))
    emb += np.where((labels == "B").reshape(1, -1, 1), 2.0, -2.0)
    ws = WindowSet(x=rng.normal(size=(n_windows, 16)).astype(np.float32), sample_id=sids,
                   class_label=labels, group_id=sids.copy(),
                   event_index=np.zeros(n_windows, dtype=np.int64),
                   onset_s=np.zeros(n_windows), length=16, rate_hz=1000.0, window_ms=16.0,
                   window_ms_source="config")
    x_files = np.column_stack([rng.normal(0, 1, n_files),
                               np.where(y_files == "B", 3.0, -3.0) + rng.normal(0, 0.2, n_files)])
    val_embeddings: dict[str, np.ndarray] = {}
    val_files: dict[str, list[str]] = {}
    for r in range(n_repeats):
        for f in range(3):
            tag = f"r{r}f{f}"
            train_files = [s for i, s in enumerate(file_order) if fold_ids[r][i] != f]
            if tag in drop_tags:
                # one class only: a head cannot be fitted, and the reason must be reported
                first = y_files[file_order.index(train_files[0])]
                val = [s for s in train_files
                       if y_files[file_order.index(s)] == first][:2]
            else:
                other = next(s for s in train_files[1:]
                             if y_files[file_order.index(s)] != first_of(y_files, train_files,
                                                                        file_order))
                val = [train_files[0], other]
            rows, sids_v = [], []
            for s in val:
                w = np.flatnonzero(sids == s)
                rows.append(emb[r][w])
                sids_v.extend([s] * len(w))
            val_embeddings[tag] = np.concatenate(rows) if rows else np.empty((0, channels))
            val_files[tag] = sids_v
    db = DeepBranch(proba=np.full((n_repeats, n_files, 2), np.nan),
                    pred=np.full((n_repeats, n_files), "", dtype=object),
                    counts=np.full(n_files, w_per_file), has_windows=np.ones(n_files, dtype=bool),
                    cv=_FakeCV(embeddings=emb), val_embeddings=val_embeddings,
                    val_files=val_files)
    return db, ws, fold_ids, file_order, x_files, y_files


def test_fusion_head_scores_only_the_outer_test_files_of_each_fold():
    db, ws, fold_ids, file_order, x_files, y_files = _branch()
    fs = fusion_stack(db, ws, fold_ids, file_order, x_files, y_files, CLASSES, _cfg())
    assert fs.proba.shape == (1, 6, 2)
    scored = np.flatnonzero(~np.isnan(fs.proba[0, :, 0]))
    assert scored.tolist() == [0, 1, 2, 3, 4, 5]           # every fold fitted a head
    assert fs.ok.all() and fs.notes == []
    assert fs.n_val_files == 6                             # 3 folds x 2 validation files
    assert fs.proba[0, scored].sum(axis=1) == pytest.approx(np.ones(6))
    assert set(fs.pred[0, scored].tolist()) <= set(CLASSES)


def test_a_test_file_cannot_influence_another_test_file():
    """No test-time fitting: the head, its scaler, and the prototypes come from validation files."""
    db, ws, fold_ids, file_order, x_files, y_files = _branch()
    base = fusion_stack(db, ws, fold_ids, file_order, x_files, y_files, CLASSES, _cfg())
    x2 = x_files.copy()
    x2[0] = [1e6, -1e6]                                    # an extreme outlier in the test fold
    moved = fusion_stack(db, ws, fold_ids, file_order, x2, y_files, CLASSES, _cfg())
    assert moved.proba[0, 3] == pytest.approx(base.proba[0, 3])
    db.cv.embeddings[0][0] = 1e6                           # and in its windows
    moved2 = fusion_stack(db, ws, fold_ids, file_order, x_files, y_files, CLASSES, _cfg())
    assert moved2.proba[0, 3] == pytest.approx(base.proba[0, 3])


def test_fusion_head_says_why_it_did_not_fit():
    db, ws, fold_ids, file_order, x_files, y_files = _branch(drop_tags=("r0f0", "r0f1", "r0f2"))
    fs = fusion_stack(db, ws, fold_ids, file_order, x_files, y_files, CLASSES, _cfg())
    assert not fs.ok.any()
    assert any("one class" in n for n in fs.notes)

    empty, ws2, fold2, order2, x2, y2 = _branch()
    empty.val_embeddings = {}
    empty.val_files = {}
    fs2 = fusion_stack(empty, ws2, fold2, order2, x2, y2, CLASSES, _cfg())
    assert not fs2.ok.any() and any("validation embeddings" in n for n in fs2.notes)


def test_a_file_fused_in_only_some_repeats_is_excluded_and_reported():
    # Fold 0 of repeat 1 cannot fit a head, so its test files are fused in one repeat only.
    db, ws, fold_ids, file_order, x_files, y_files = _branch(n_repeats=2, drop_tags=("r1f0",))
    fs = fusion_stack(db, ws, fold_ids, file_order, x_files, y_files, CLASSES, _cfg())
    scored_any = ~np.isnan(fs.proba[:, :, 0]).all(axis=0)
    partial = np.flatnonzero(scored_any & ~fs.ok)
    assert partial.tolist() == [0, 3]                      # fold 0's files, one repeat only
    assert fs.ok.sum() == 4                                # folds 1 and 2 fused in both repeats
    assert any("only some repeats" in n for n in fs.notes)


# ---------------------------------------------------------------------------
# train_fold: the encoder produces everything the other strategies consume
# ---------------------------------------------------------------------------

def test_train_fold_returns_validation_embeddings_and_prototypes():
    rng = np.random.default_rng(4)
    cfg = _cfg(channels=8, epochs=2, validation_fraction=0.4)
    n_tr, n_va, n_te, length = 24, 12, 8, 64
    xtr = rng.normal(size=(n_tr, length)).astype(np.float32)
    xva = rng.normal(size=(n_va, length)).astype(np.float32)
    xte = rng.normal(size=(n_te, length)).astype(np.float32)
    ytr = np.array([i % 2 for i in range(n_tr)])
    yva = np.array([i % 2 for i in range(n_va)])
    res = train_fold(xtr, ytr, xva, yva, xte, 2, cfg, seed=11, imbalance_ratio=1.0,
                     classes_order=CLASSES)
    assert res.proba_test.shape == (n_te, 2) and res.proba_val.shape == (n_va, 2)
    assert res.embedding_val.shape == (n_va, 8)
    assert res.embedding_test.shape == (n_te, 8)
    assert res.proba_test_proto.shape == (n_te, 2)
    assert res.proba_test_proto.sum(axis=1) == pytest.approx(np.ones(n_te), abs=1e-6)
    assert np.isfinite(res.proba_test_proto).all()
    st = res.state
    assert st["prototypical_head"] is True and st["metric_loss"] == "supcon"
    assert "prototypes from" in st["prototypes"]
    hist = res.history
    assert hist and "train_metric_loss" in hist[0] and np.isfinite(hist[0]["train_metric_loss"])
    assert hist[0]["metric_loss_kind"] == "supcon"


def test_train_fold_without_a_validation_split_reports_no_prototypes_instead_of_guessing():
    rng = np.random.default_rng(5)
    cfg = _cfg(channels=8, epochs=1)
    xtr = rng.normal(size=(12, 48)).astype(np.float32)
    xte = rng.normal(size=(4, 48)).astype(np.float32)
    ytr = np.array([i % 2 for i in range(12)])
    res = train_fold(xtr, ytr, np.empty((0, 48), dtype=np.float32), np.empty(0, dtype=np.int64),
                     xte, 2, cfg, seed=3, imbalance_ratio=1.0, classes_order=CLASSES)
    assert res.embedding_val.shape[0] == 0
    assert res.proba_test_proto.shape == (0, 2)
    assert "empty" in res.state["prototypes"]


def test_prototypical_head_can_be_switched_off():
    rng = np.random.default_rng(6)
    cfg = _cfg(channels=8, epochs=1, prototypical_head=False, metric_loss="none",
               metric_loss_weight=0.0)
    xtr = rng.normal(size=(12, 48)).astype(np.float32)
    xva = rng.normal(size=(6, 48)).astype(np.float32)
    xte = rng.normal(size=(4, 48)).astype(np.float32)
    res = train_fold(xtr, np.array([i % 2 for i in range(12)]), xva,
                     np.array([i % 2 for i in range(6)]), xte, 2, cfg, seed=7,
                     imbalance_ratio=1.0, classes_order=CLASSES)
    assert res.proba_test_proto.shape == (0, 2)
    assert res.state["prototypical_head"] is False
    assert "disabled" in res.state["prototypes"]
    assert res.embedding_val.shape[0] == 6                 # the embedding still exists
