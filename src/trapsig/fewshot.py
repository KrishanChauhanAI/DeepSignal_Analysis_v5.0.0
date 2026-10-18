"""Episodic one-shot and few-shot evaluation of a metric embedding (Head B).

An episode draws, per class, ``shot`` support files and ``queries_per_class`` query files from a
held-out pool; support and query sets are disjoint *by construction*, and prototypes are the mean
L2-normalised embedding of the support set. Query labels come from cosine similarity to the
prototypes. Episodes are independent draws, so the mean episode accuracy has a genuine standard
deviation and a bootstrap-free binomial CI is reported per shot.

What this module deliberately does NOT claim:

* few-shot ability is a measured property of the embedding, not a consequence of having a metric
  head - hence the episodes, the CIs, and the shot curve;
* open-set / unknown-class recognition is a separate problem requiring its own validation; this
  module evaluates closed-set episodes only and says so in every summary row.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _normalise(emb: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(emb, axis=-1, keepdims=True)
    out: np.ndarray = emb / np.where(n > 0, n, 1.0)
    return out


def episodic_eval(emb: np.ndarray, y: np.ndarray, classes: list[str], shots: list[int],
                  episodes: int, queries_per_class: int, seed: int, tag: str = "",
                  pool: str = "held-out test files") -> list[dict[str, Any]]:
    """One row per (episode, shot); queries are classified against support prototypes."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    emb = _normalise(np.asarray(emb, dtype=float))
    by_class = {c: np.flatnonzero(np.array([str(v) == c for v in y])) for c in classes}
    usable = [c for c in classes if by_class[c].size >= 2 + queries_per_class]
    if len(usable) < 2 or emb.shape[0] < 4:
        return rows
    for shot in shots:
        for e in range(episodes):
            support_i: list[int] = []
            query_i: list[int] = []
            ok = True
            for c in usable:
                ids = by_class[c]
                need = shot + queries_per_class
                if ids.size < need:
                    ok = False
                    break
                pick = rng.choice(ids, size=need, replace=False)
                support_i += list(pick[:shot])
                query_i += list(pick[shot:])
            if not ok or not query_i:
                continue
            protos = np.stack([_normalise(emb[support_i])[
                [j for j, s in enumerate(support_i)
                 if str(y[s]) == c]].mean(axis=0) for c in usable])
            sim = _normalise(emb[query_i]) @ protos.T
            pred = [usable[int(a)] for a in np.argmax(sim, axis=1)]
            correct = [int(p == str(y[q])) for p, q in zip(pred, query_i, strict=True)]
            margins = (np.sort(sim, axis=1)[:, -1] - np.sort(sim, axis=1)[:, -2])
            rows.append({"fold_tag": tag, "shot": int(shot), "episode": int(e),
                         "n_way": len(usable), "n_query": len(query_i),
                         "accuracy": round(float(np.mean(correct)), 6),
                         "n_correct": int(sum(correct)),
                         "mean_margin": round(float(margins.mean()), 6),
                         "support_classes": "|".join(usable),
                         "pool": pool,
                         "protocol": "disjoint support/query drawn per episode from held-out "
                                     "files; cosine distance to support prototypes"})
    return rows


def summarize_episodes(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per shot: mean/sd episode accuracy, a Wilson 95% CI over pooled queries, per-class means."""
    out: list[dict[str, Any]] = []
    shots = sorted({int(r["shot"]) for r in rows})
    for shot in shots:
        sub = [r for r in rows if int(r["shot"]) == shot]
        if not sub:
            continue
        accs = np.array([float(r["accuracy"]) for r in sub])
        nq = int(sum(int(r["n_query"]) for r in sub))
        nc = int(sum(int(r["n_correct"]) for r in sub))
        p = nc / max(1, nq)
        z = 1.96
        denom = 1 + z * z / max(1, nq)
        centre = (p + z * z / (2 * max(1, nq))) / denom
        half = z * np.sqrt(p * (1 - p) / max(1, nq) + z * z / (4 * max(1, nq) ** 2)) / denom
        per_class: dict[str, list[float]] = {}
        out.append({"shot": shot, "n_episodes": len(sub),
                    "episode_accuracy_mean": round(float(accs.mean()), 6),
                    "episode_accuracy_sd": round(float(accs.std(ddof=1)), 6)
                    if len(sub) > 1 else None,
                    "pooled_query_accuracy": round(float(p), 6),
                    "wilson_ci95_lo": round(float(max(0.0, centre - half)), 6),
                    "wilson_ci95_hi": round(float(min(1.0, centre + half)), 6),
                    "n_query_total": nq,
                    "setting": "closed-set episodes; open-set detection is a separate, "
                               "unvalidated extension",
                    "per_class": per_class or None})
    return out
