"""Per-run metadata: the authoritative record of THIS run's split.

The folds are re-randomised on every run (each repeat draws a new stratified group
allocation from the seed), so a metadata file written once would describe a split that no
longer exists. This module therefore writes ``metadata.csv`` and ``metadata.json`` into the
run directory on **every** run — ``dry-run``, ``prepare``, and ``train`` — so that the exact
train / validation / test membership of the run you are looking at can always be recovered
from that run's own folder, without re-deriving anything.

``metadata.csv`` has one row per source file (every discovered file, including ones that are
not in the CV pool and ones that were skipped), with its class, group, hashes, symlink, pool
membership, and its fold and role (``train`` or ``test``) for every repeat. The inner
validation split used by the CNN is drawn from each outer training fold at train time and is
logged per fold in ``logs/deep_training_log.jsonl``; it is not a property of the file, so it
does not appear here.

``metadata.json`` records the run-level facts: which configuration fingerprint produced the
split, the seed, the repeat and fold counts, the grouping policy, the class counts, and an
explicit statement that the split is re-randomised per run.
"""

from __future__ import annotations

import csv
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .config import TrapsigConfig
from .runs import write_json_exclusive, write_text_exclusive

BASE_COLUMNS = [
    "sample_id", "filename", "original_path", "source_subdir", "name_annotation",
    "class", "label_source", "group_id", "group_level", "replicate_id", "session_id",
    "file_sha256", "size_bytes", "n_samples", "duration_s", "sampling_rate_hz",
    "quality_flags", "in_cv_pool", "split", "symlink_path", "link_status",
    "status", "reason", "n_events", "n_windows", "model_b_scored",
]
# Per repeat N the generated columns are: fold_rN, role_rN, trains_in_folds_rN, cnn_inner_val_rN.
PER_REPEAT_COLUMNS = ("fold_r{N}", "role_r{N}", "trains_in_folds_r{N}", "cnn_inner_val_r{N}")


def _as_list(v: Any) -> str:
    if v is None or v == "":
        return ""
    if isinstance(v, (list, tuple)):
        return "|".join(str(x) for x in v)
    return str(v)


def build_metadata_rows(
    rows: list[dict[str, Any]],
    pool_positions: list[int],
    fold_ids: np.ndarray | None,
    n_repeats: int,
    extras: dict[str, dict[str, Any]] | None = None,
    n_splits: int = 0,
) -> list[dict[str, Any]]:
    """One metadata row per source file, in manifest order.

    ``pool_positions`` are the manifest indices that entered the CV pool, in the same order as
    the columns of ``fold_ids`` (shape ``(n_repeats, n_pool)``). Files outside the pool get
    empty fold columns and ``in_cv_pool = false``. ``extras`` adds per-run facts such as the
    per-file status after processing, the event count, whether Model B could score the file,
    and which outer folds drew the file into the CNN's inner validation split
    (``cnn_inner_val_folds``, a list of ``"r{repeat}f{fold}"`` tags).

    Per repeat ``r`` the columns say exactly how the file was used: ``fold_rN`` is the fold it
    belongs to, ``role_rN`` is ``test`` (it is the held-out fold when that fold is evaluated),
    ``trains_in_folds_rN`` lists the folds for which it is part of the training set, and
    ``cnn_inner_val_rN`` lists the folds whose inner validation split it was drawn into.
    """
    ex = extras or {}
    pos_of = {i: p for p, i in enumerate(pool_positions)}
    out: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        sid = str(row.get("sample_id", ""))
        rec: dict[str, Any] = {c: _as_list(row.get(c, "")) for c in BASE_COLUMNS[:21]}
        e = ex.get(sid, {})
        rec["status"] = _as_list(e.get("status", ""))
        rec["reason"] = _as_list(e.get("reason", ""))
        rec["n_events"] = "" if e.get("n_events") is None else e.get("n_events")
        rec["n_windows"] = "" if e.get("n_windows") is None else e.get("n_windows")
        scored = e.get("model_b_scored")
        rec["model_b_scored"] = "" if scored is None else bool(scored)
        in_pool = i in pos_of
        rec["in_cv_pool"] = in_pool
        # Fold tags look like "r0f3": repeat 0, outer fold 3.
        val_by_rep: dict[int, list[int]] = {}
        for tag in (e.get("cnn_inner_val_folds") or []):
            t = str(tag)
            if t.startswith("r") and "f" in t:
                rr, _, ff = t.partition("f")
                try:
                    val_by_rep.setdefault(int(rr[1:]), []).append(int(ff))
                except ValueError:
                    continue
        for r in range(n_repeats):
            if in_pool and fold_ids is not None:
                f = int(fold_ids[r, pos_of[i]])
                rec[f"fold_r{r}"] = f
                # In repeat r this file is the held-out test file when fold f is evaluated.
                rec[f"role_r{r}"] = "test"
                rec[f"trains_in_folds_r{r}"] = _as_list(
                    [j for j in range(n_splits) if j != f] if n_splits else [])
                rec[f"cnn_inner_val_r{r}"] = _as_list(sorted(set(val_by_rep.get(r, []))))
            else:
                rec[f"fold_r{r}"] = ""
                rec[f"role_r{r}"] = "not_in_pool"
                rec[f"trains_in_folds_r{r}"] = ""
                rec[f"cnn_inner_val_r{r}"] = ""
        out.append(rec)
    return out


def _write_csv(rows: list[dict[str, Any]], path: Path) -> Path:
    if not rows:
        write_text_exclusive(path, "")
        return path
    buf = io.StringIO()
    fields = list(rows[0])
    w = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
    w.writeheader()
    for r in rows:
        w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in fields})
    write_text_exclusive(path, buf.getvalue())
    return path


def run_metadata_block(cfg: TrapsigConfig, run_dir: Path, n_source: int, n_pool: int,
                       n_ignored: int, class_counts: dict[str, int], stage: str,
                       extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """The run-level half of the metadata record."""
    sp = cfg.splitting
    block: dict[str, Any] = {
        "run_dir": str(run_dir),
        "stage": stage,
        "written_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config_fingerprint": cfg.fingerprint(),
        "seed": cfg.run.seed,
        "n_repeats": sp.n_repeats,
        "n_splits": sp.n_splits,
        "n_source_files": n_source,
        "n_cv_pool": n_pool,
        "n_ignored_by_glob": n_ignored,
        "class_counts": class_counts,
        "grouping_level": "file" if cfg.data.allow_file_level_fallback else "replicate_or_session",
        "grouping_note": ("no replicate or session metadata exists, so each file is its own "
                          "group; cross-validated estimates are optimistic under session effects"
                          if cfg.data.allow_file_level_fallback else
                          "groups come from replicate or session metadata"),
        "holdout": "none (no replicate groups exist to hold out)",
        "split_randomisation": ("folds are re-randomised on every run from seed + repeat; this "
                                "file is the authoritative record of THIS run's split and is "
                                "rewritten into every run directory"),
        "role_convention": "fold_rN is the fold the file belongs to in repeat N; role_rN=test "
                           "means the file is the held-out test file when that fold is evaluated, "
                           "and trains_in_folds_rN lists the folds of repeat N whose training set "
                           "it belongs to (K-fold: every pool file is tested exactly once per "
                           "repeat). cnn_inner_val_rN lists the folds of repeat N whose inner "
                           "validation split (carved from the outer training set at train time, "
                           "used for early stopping and temperature scaling) this file was drawn "
                           "into; per-epoch detail is in logs/deep_training_log.jsonl",
    }
    if extra:
        block.update(extra)
    return block


def write_run_metadata(run_dir: Path, cfg: TrapsigConfig, rows: list[dict[str, Any]],
                       pool_positions: list[int], fold_ids: np.ndarray | None,
                       n_ignored: int, stage: str,
                       extras: dict[str, dict[str, Any]] | None = None,
                       extra_block: dict[str, Any] | None = None) -> tuple[Path, Path]:
    """Write ``metadata.csv`` and ``metadata.json`` into ``run_dir``. Called on every run."""
    meta_rows = build_metadata_rows(rows, pool_positions, fold_ids, cfg.splitting.n_repeats,
                                    extras, cfg.splitting.n_splits)
    pool = [rows[i] for i in pool_positions]
    counts: dict[str, int] = {}
    for r in pool:
        c = str(r.get("class", ""))
        counts[c] = counts.get(c, 0) + 1
    block = run_metadata_block(cfg, run_dir, len(rows), len(pool), n_ignored, counts, stage,
                               extra_block)
    csv_path = _write_csv(meta_rows, run_dir / "metadata.csv")
    json_path = write_json_exclusive(run_dir / "metadata.json", block)
    return csv_path, json_path


def read_metadata_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def summary_line(run_dir: Path) -> str:
    """One console line so the user sees where this run's split record went."""
    path = Path(run_dir) / "metadata.json"
    if not path.is_file():
        return f"metadata: no record written in {Path(run_dir).name}"
    block = json.loads(path.read_text(encoding="utf-8"))
    return (f"metadata: {block['n_cv_pool']} of {block['n_source_files']} files in the CV pool; "
            f"{block['n_repeats']} repeats x {block['n_splits']} folds re-randomised for this run; "
            f"record: {Path(run_dir).name}/metadata.csv + metadata.json")


def json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    return str(o)


def dumps(obj: Any) -> str:
    return json.dumps(obj, indent=2, default=json_default, allow_nan=False)
