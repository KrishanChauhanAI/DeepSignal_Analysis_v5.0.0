"""Tests for the visualization registry: coverage, honest statuses, CSV twins."""

from __future__ import annotations

import numpy as np

from trapsig import viz_registry as vreg


def _ctx() -> vreg.RegistryCtx:
    rng = np.random.default_rng(0)
    fs = 1000.0
    signals = {
        "A": [rng.normal(size=4096) + np.sin(np.arange(4096) / 8.0), rng.normal(size=4096)],
        "B": [rng.normal(size=4096) * 2.0, rng.normal(size=3000)],
    }
    y = np.array(["A", "A", "B", "B"], dtype=object)
    x = rng.normal(size=(4, 6))
    return vreg.RegistryCtx(
        classes=["A", "B"],
        fs=fs,
        signals=signals,
        x=x,
        feat_names=[f"f{i}" for i in range(6)],
        y=y,
        ok_rows=[{"class": c, "sid": f"s{i}"} for i, c in enumerate(["A", "A", "B", "B"])],
        probas={"m": (y, np.array([[0.7, 0.3], [0.6, 0.4], [0.2, 0.8], [0.3, 0.7]]))},
        preds={"m": (y, y.copy())},
        histories={"m": [{"history": [{"epoch": 1, "train_loss": 0.5, "val_loss": 0.6, "val_score": 0.4}]}]},
        embeddings={"e": (rng.normal(size=(6, 8)), np.array(["A"] * 3 + ["B"] * 3, dtype=object))},
        fewshot=[
            {
                "shot": 1,
                "episode": 0,
                "accuracy": 1.0,
                "n_query": 2,
                "n_correct": 2,
                "mean_margin": 0.3,
                "n_way": 2,
                "support_classes": "A|B",
                "pool": "test",
                "protocol": "p",
            }
        ],
        gates={"temporal": 0.4, "spectral": 0.35, "engineered": 0.25},
        summary={
            "models": {
                "quad": {"aggregated": {"balanced_accuracy": {"value": 0.8}}},
                "quad_minus_temporal": {"aggregated": {"balanced_accuracy": {"value": 0.6}}},
            }
        },
        redundancy=[{"feature_a": "f0", "feature_b": "f1", "spearman": 0.95, "abs_spearman": 0.95}],
    )


def test_registry_has_at_least_50_entries_in_six_categories() -> None:
    assert vreg.registry_count() >= 50
    cats = {e["cat"] for e in vreg.REGISTRY}
    assert {"A", "B", "C", "D", "E", "F"}.issubset(cats)
    assert {"K", "Q", "S", "G"}.issubset(cats)
    ids = [e["id"] for e in vreg.REGISTRY]
    assert len(ids) == len(set(ids))
    for e in vreg.REGISTRY:
        assert e["title"] and callable(e["b"]) and e["prereq"] is not None


def test_render_records_every_entry_honestly(tmp_path) -> None:
    ctx = _ctx()
    statuses = vreg.render(ctx, tmp_path / "png", tmp_path / "csv", dpi=60)
    assert len(statuses) == vreg.registry_count()
    counts = {}
    for st in statuses:
        counts[st["status"]] = counts.get(st["status"], 0) + 1
        assert st["status"] in ("generated", "skipped", "not_available", "failed")
        if st["status"] == "generated":
            assert st["file"]
        else:
            assert st["reason"]
    assert counts.get("failed", 0) == 0, [s for s in statuses if s["status"] == "failed"]
    assert counts.get("generated", 0) >= 40
    na = {s["id"] for s in statuses if s["status"] == "not_available"}
    assert {"emd_components", "vmd_components"}.issubset(na)
    # Round-3 optional-dependency plots are explicitly marked unavailable, never failed.
    assert na.issubset(
        {
            "emd_components",
            "vmd_components",
            "sample_entropy_summary",
            "permutation_entropy",
            "multiscale_entropy",
            "r3_umap",
            "r3_pacmap",
        }
    )


def test_csv_twins_and_index(tmp_path) -> None:
    ctx = _ctx()
    statuses = vreg.render(ctx, tmp_path / "png", tmp_path / "csv", dpi=60)
    twins = [s for s in statuses if s["csv_twin"]]
    assert {s["id"] for s in twins} >= {"class_counts", "signal_quality", "train_val_loss"}
    for s in twins:
        assert (tmp_path / "csv" / f"{s['id']}.csv").is_file()
    idx = vreg.write_registry_index(statuses, tmp_path / "viz_registry_index.csv")
    text = (tmp_path / "viz_registry_index.csv").read_text()
    assert text.splitlines()[0].startswith("id,category,title,status")
    assert idx.endswith("viz_registry_index.csv")


def test_empty_context_skips_instead_of_fabricating(tmp_path) -> None:
    ctx = vreg.RegistryCtx(classes=[], fs=1000.0)
    statuses = vreg.render(ctx, tmp_path / "png", tmp_path / "csv", dpi=60)
    assert all(s["status"] in ("skipped", "not_available") for s in statuses)
    assert all(not s["file"] for s in statuses)
