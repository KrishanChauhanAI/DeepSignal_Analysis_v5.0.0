"""Tests for the few-shot episode protocol and its summary statistics."""

from __future__ import annotations

import numpy as np

from trapsig.fewshot import episodic_eval, summarize_episodes


def _separable(n_per: int = 6, dim: int = 8, seed: int = 0):
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(2, dim)) * 6
    emb = np.vstack([centres[c] + rng.normal(scale=0.05, size=(n_per, dim))
                     for c in range(2)])
    y = np.array(["A"] * n_per + ["B"] * n_per, dtype=object)
    return emb, y


def test_episodic_eval_structure() -> None:
    emb, y = _separable()
    rows = episodic_eval(emb, y, ["A", "B"], shots=[1, 3], episodes=5, queries_per_class=2,
                         seed=7)
    assert rows and len(rows) == 10          # 2 shots x 5 episodes
    for r in rows:
        assert 0.0 <= float(r["accuracy"]) <= 1.0
        assert int(r["shot"]) in (1, 3)
        assert r["protocol"]


def test_episodes_are_deterministic() -> None:
    emb, y = _separable()
    a = episodic_eval(emb, y, ["A", "B"], shots=[1], episodes=4, queries_per_class=2, seed=3)
    b = episodic_eval(emb, y, ["A", "B"], shots=[1], episodes=4, queries_per_class=2, seed=3)
    assert [r["accuracy"] for r in a] == [r["accuracy"] for r in b]


def test_separable_embedding_scores_high() -> None:
    emb, y = _separable(n_per=8)
    rows = episodic_eval(emb, y, ["A", "B"], shots=[5], episodes=10, queries_per_class=2,
                         seed=11)
    acc = float(np.mean([r["accuracy"] for r in rows]))
    assert acc > 0.9        # well-separated classes with 5 support shots


def test_too_few_files_yields_no_rows() -> None:
    emb = np.random.default_rng(0).normal(size=(3, 4))
    y = np.array(["A", "A", "B"], dtype=object)
    rows = episodic_eval(emb, y, ["A", "B"], shots=[1], episodes=3, queries_per_class=1, seed=1)
    assert rows == []


def test_summarize_wilson_bounds() -> None:
    emb, y = _separable()
    rows = episodic_eval(emb, y, ["A", "B"], shots=[1, 2], episodes=6, queries_per_class=2,
                         seed=5)
    summ = summarize_episodes(rows)
    assert len(summ) == 2
    for r in summ:
        lo, hi = float(r["wilson_ci95_lo"]), float(r["wilson_ci95_hi"])
        assert 0.0 <= lo <= hi <= 1.0
        assert 0.0 <= float(r["episode_accuracy_mean"]) <= 1.0
        assert r["n_episodes"] == 6
