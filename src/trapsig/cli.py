"""Command-line interface: ``trapsig dry-run`` and ``trapsig prepare``.

Exit codes:
  0  success (warnings may be present)
  1  preflight or preparation found ERROR entries; nothing was prepared
  2  configuration could not be parsed
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np

from . import __version__
from .config import ConfigError, TrapsigConfig, load_config, validate_modelling
from .ingest import IngestError, probe_dat
from .manifest import (
    build_manifest,
    discover_sources,
    list_ignored_files,
    parse_name_annotation,
    parse_source_subdir,
    sample_id_for,
    write_manifest_csv,
)
from .metadata import summary_line as metadata_summary_line
from .metadata import write_run_metadata
from .pipeline import RecordingSkipped, process_recording
from .preflight import PreflightReport, run_preflight
from .runs import (
    capture_environment,
    create_prepared_dir,
    create_run_dir,
    write_json_exclusive,
    write_text_exclusive,
)
from .splits import grouped_stratified_folds
from .symlinks import create_split_links, write_link_report

CV_SPLIT = "cv_pool"

PER_FILE_FIELDS = [
    "sample_id", "filename", "class", "source_subdir", "name_annotation", "status", "reason",
    "fs_hz", "n_samples", "duration_s", "raw_min", "raw_max", "raw_mean", "raw_std",
    "n_nonfinite", "baseline_median", "sigma", "n_events", "events_per_min",
    "n_edge_onsets_excluded", "n_too_short_rejected", "n_merged", "original_path",
]

EVENT_FIELDS = [
    "sample_id", "class", "source", "filename", "fs_hz", "filter", "event_index", "onset_s", "offset_s",
    "duration_s", "i0", "i0_source", "delta_i_mean", "delta_i_over_i0", "amplitude",
    "censored_start", "censored_end", "toff_s_to_next",
]


def cmd_events(config_path: str, rate_hz: float) -> int:
    """Detect events in every source recording at ``rate_hz`` and write an event table."""
    raw = Path(config_path).read_text(encoding="utf-8")
    cfg = load_config(config_path)
    rep = run_preflight(cfg)
    if rep.errors:
        _print_report(rep)
        print("\nevents aborted: resolve the ERROR entries above (no run directory created).")
        return 1
    if rate_hz <= 0 or rate_hz > cfg.sampling.original_hz:
        print(f"CONFIGURATION ERROR: --rate-hz must be in (0, {cfg.sampling.original_hz:g}]", file=sys.stderr)
        return 2
    if cfg.filters.lpf_cutoff_hz is not None and cfg.filters.lpf_cutoff_hz >= rate_hz / 2:
        print(f"CONFIGURATION ERROR: LPF cutoff {cfg.filters.lpf_cutoff_hz:g} Hz must be below "
              f"the event-detection Nyquist {rate_hz / 2:g} Hz", file=sys.stderr)
        return 2

    run_dir = create_run_dir(cfg.output.runs_dir, kind="events")
    _save_preflight(run_dir, rep, cfg, raw)
    records = discover_sources(cfg.data, run_dir.name)

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=EVENT_FIELDS)
    writer.writeheader()
    per_file: list[dict[str, Any]] = []
    for rec in records:
        sid = sample_id_for(rec.path, rec.path.stat().st_size)
        base: dict[str, Any] = {
            "sample_id": sid, "filename": rec.path.name, "class": rec.label or "NA",
            "source_subdir": parse_source_subdir(rec.rel_path),
            "name_annotation": parse_name_annotation(rec.path.stem),
            "original_path": str(rec.path),
        }
        try:
            res = process_recording(rec.path, cfg, rate_hz)
        except RecordingSkipped as exc:
            per_file.append({**base, "status": "skipped", "reason": exc.reason, **exc.stats})
            continue
        n_s = float(res.stats.get("n_samples", 0))
        dur = n_s / res.fs_hz if res.fs_hz else 0.0
        per_file.append({
            **base, "status": "ok", "reason": "", **res.stats,
            "fs_hz": res.fs_hz, "duration_s": dur, "filter": res.filter_label,
            "n_events": len(res.events),
            "events_per_min": (len(res.events) / (dur / 60.0)) if dur > 0 else "",
            "n_edge_onsets_excluded": res.n_edge_excluded,
            "n_too_short_rejected": res.n_too_short,
            "n_merged": res.n_merged,
        })
        for k, ev in enumerate(res.events):
            writer.writerow({
                "sample_id": sid, "class": rec.label or "NA",
                "source": str(rec.path), "filename": rec.path.name, "fs_hz": res.fs_hz,
                "filter": res.filter_label, "event_index": k,
                "onset_s": ev.onset_s, "offset_s": ev.offset_s, "duration_s": ev.duration_s,
                "i0": ev.i0, "i0_source": ev.i0_source, "delta_i_mean": ev.delta_i_mean,
                "delta_i_over_i0": ev.delta_i_over_i0, "amplitude": ev.amplitude,
                "censored_start": ev.censored_start, "censored_end": ev.censored_end,
                "toff_s_to_next": float(res.toff_s[k]) if k < res.toff_s.size else "",
            })
    write_text_exclusive(run_dir / "metrics" / "events.csv", buf.getvalue())
    pf = io.StringIO()
    pw = csv.DictWriter(pf, fieldnames=PER_FILE_FIELDS, extrasaction="ignore")
    pw.writeheader()
    for row in per_file:
        pw.writerow(row)
    write_text_exclusive(run_dir / "metrics" / "per_file_summary.csv", pf.getvalue())
    n_skipped = sum(1 for f in per_file if f["status"] == "skipped")
    summary = {
        "trapsig_version": __version__,
        "run_dir": str(run_dir),
        "config_fingerprint": cfg.fingerprint(),
        "event_detection_rate_hz": rate_hz,
        "n_recordings": len(per_file),
        "n_events_total": sum(int(f.get("n_events") or 0) for f in per_file),
        "n_edge_onsets_excluded_total": sum(int(f.get("n_edge_onsets_excluded", 0)) for f in per_file),
        "n_skipped": n_skipped,
        "grouping": {
            "policy": "file_level_fallback" if cfg.data.allow_file_level_fallback else "replicate_or_session",
            "note": ("No replicate information: each file is its own group. Any cross-validation "
                     "estimate is optimistic under session effects.")
            if cfg.data.allow_file_level_fallback else "Grouped by replicate or session.",
        },
        "per_file_summary": "metrics/per_file_summary.csv",
        "per_recording": per_file,
        "note": "Events are label-blind detections; they are not classified or interpreted.",
    }
    write_json_exclusive(run_dir / "metrics" / "events_summary.json", summary)
    _print_report(rep)
    print(f"\nevents: {summary['n_events_total']} detected in {len(per_file)} recording(s) -> {run_dir}")
    return 0


def _print_report(rep: PreflightReport) -> None:
    width = max(len(c.name) for c in rep.checks) if rep.checks else 10
    print(f"\ntrapsig {__version__} preflight  (config {rep.config_fingerprint[:12]})")
    print("-" * (width + 40))
    for c in rep.checks:
        print(f"[{c.status.value:<14}] {c.check_id} {c.name:<{width}}  {c.message}")
        if c.fix and c.status.value in ("ERROR", "WARNING"):
            print(f"{'':<17}fix: {c.fix}")
    counts = rep.counts()
    summary = "  ".join(f"{k}={v}" for k, v in counts.items() if v)
    print("-" * (width + 40))
    print(f"summary: {summary}")
    print("RESULT: " + ("FAILED (errors must be fixed)" if rep.errors else "PASSED"))


def _save_preflight(run_dir: Path, rep: PreflightReport, cfg: TrapsigConfig, raw_config: str,
                    skip_existing: bool = False) -> bool:
    """Write the configuration, environment, and preflight record of a run.

    ``skip_existing`` is used when training continues in the run folder that ``prepare`` already
    created: the record is already there and must not be duplicated or overwritten.
    """
    if skip_existing and (run_dir / "reports" / "preflight_report.json").is_file():
        return False
    write_text_exclusive(run_dir / "config" / "config_used.yaml", raw_config)
    write_json_exclusive(run_dir / "config" / "config_resolved.json",
                         {"fingerprint": cfg.fingerprint(), "config": cfg.to_dict()})
    write_json_exclusive(run_dir / "environment" / "environment.json", capture_environment(run_dir))
    write_json_exclusive(run_dir / "reports" / "preflight_report.json", rep.to_dict())
    lines = ["check_id,name,status,message,fix"]
    for c in rep.checks:
        cells = [c.check_id, c.name, c.status.value, c.message, c.fix]
        lines.append(",".join('"' + x.replace('"', '""') + '"' for x in cells))
    write_text_exclusive(run_dir / "reports" / "preflight_report.csv", "\n".join(lines) + "\n")
    return True


def cmd_train(config_path: str, prepared_run: str) -> int:
    """Model A: preflight, then features, repeated grouped CV, metrics, and label-shuffle null."""
    from .train import TrainError, run_training

    raw = Path(config_path).read_text(encoding="utf-8")
    cfg = load_config(config_path)
    rep = run_preflight(cfg)
    if rep.errors:
        _print_report(rep)
        print("\ntrain aborted: resolve the ERROR entries above (no run directory created).")
        return 1
    tr_errs = validate_modelling(cfg)
    if tr_errs:
        for e in tr_errs:
            print(f"CONFIGURATION ERROR: {e}", file=sys.stderr)
        return 2
    prepared = Path(prepared_run)
    if not prepared.is_dir():
        print(f"CONFIGURATION ERROR: prepared run not found: {prepared}", file=sys.stderr)
        return 2

    run_dir, why_not = _run_dir_for_prepared(cfg, prepared)
    if run_dir is None:
        print(f"CONFIGURATION ERROR: {why_not}", file=sys.stderr)
        return 2
    _save_preflight(run_dir, rep, cfg, raw, skip_existing=True)
    print(f"training run: {run_dir}")
    print(f"prepared dataset: {prepared}")
    try:
        summary = run_training(cfg, prepared, run_dir, progress=lambda m: print(m, flush=True))
    except TrainError as exc:
        print(f"TRAINING ERROR: {exc}", file=sys.stderr)
        print(f"partial outputs (if any) are in {run_dir}")
        return 1
    _print_train_headline(summary)
    print(f"\ntrained run: {run_dir}")
    return 0


def _print_train_headline(summary: dict[str, Any]) -> None:
    print(f"\nfiles used {summary['n_files_used']} of {summary['n_pool_files']}; "
          f"skipped or failed {summary['n_files_skipped_or_failed']}; "
          f"class counts {summary['class_counts_used']}; imbalance {summary['imbalance_ratio']:.2f}")
    for name, entry in summary["models"].items():
        bacc = entry["per_repeat"].get("balanced_accuracy", {})
        acc = entry["per_repeat"].get("accuracy", {})
        perm = entry.get("permutation", {})
        line = f"  {name:<18}"
        if "mean" in bacc:
            line += f" balanced_acc {bacc['mean']:.3f} +/- {bacc['sd']:.3f}"
        if "mean" in acc:
            line += f"  acc {acc['mean']:.3f}"
        if perm.get("p_value") is not None:
            line += f"  perm p={perm['p_value']:.3f}"
        print(line)


def cmd_dry_run(config_path: str) -> int:
    raw = Path(config_path).read_text(encoding="utf-8")
    cfg = load_config(config_path)
    rep = run_preflight(cfg)
    run_dir = create_run_dir(cfg.output.runs_dir, kind="dryrun")
    _save_preflight(run_dir, rep, cfg, raw)
    _print_report(rep)
    print(f"\nreport written to: {run_dir}")
    return rep.exit_code


def cmd_prepare(config_path: str) -> int:
    """Preflight, then write the manifest, symlinks, and leakage-safe fold assignments."""
    cfg = load_config(config_path)
    raw = Path(config_path).read_text(encoding="utf-8")
    rep = run_preflight(cfg)
    if rep.errors:
        _print_report(rep)
        print("\nprepare aborted: resolve the ERROR entries above (no run directory created).")
        return 1
    _print_report(rep)
    code, _, _ = _do_prepare(cfg, raw, echo=True)
    return code


# ---------------------------------------------------------------------------
# One-command pipeline
# ---------------------------------------------------------------------------

def _apply_overrides(cfg: TrapsigConfig, quick: bool, model_set: str | None,
                     no_pdf: bool, no_explain: bool) -> tuple[TrapsigConfig, list[str]]:
    """Apply command-line overrides and return the changed settings, so nothing is silent."""
    import dataclasses

    notes: list[str] = []
    splitting, training, deep, explain, report = (cfg.splitting, cfg.training, cfg.deep,
                                                  cfg.explain, cfg.report)
    if quick:
        splitting = dataclasses.replace(splitting, n_repeats=min(splitting.n_repeats, 2))
        training = dataclasses.replace(training, n_permutations=0, rf_n_estimators=min(
            training.rf_n_estimators, 100), hgb_max_iter=min(training.hgb_max_iter, 60),
            ablation="none", hpo="none")
        deep = dataclasses.replace(deep, hpo_trials=0)
        deep = dataclasses.replace(deep, epochs=min(deep.epochs, 8), n_permutations=0,
                                   hpo_trials=0)
        explain = dataclasses.replace(explain, enabled=False)
        report = dataclasses.replace(report, pdf=False, plots=False, max_plots_per_class=0)
        notes.append("--quick: n_repeats<=2, no permutations, no feature ablation, no "
                     "hyperparameter search, fewer trees/iterations, deep epochs<=8, no "
                     "explainability, no figures or PDF")
    if model_set:
        from .config import ModelSet

        deep = dataclasses.replace(deep, model_set=cast(ModelSet, model_set),
                                   enabled=(model_set != "features"))
        notes.append(f"--model-set {model_set}")
    if no_pdf:
        report = dataclasses.replace(report, pdf=False)
        notes.append("--no-pdf")
    if no_explain:
        explain = dataclasses.replace(explain, enabled=False)
        notes.append("--no-explain")
    return dataclasses.replace(cfg, splitting=splitting, training=training, deep=deep,
                               explain=explain, report=report), notes


def estimate_budget(cfg: TrapsigConfig, n_probe: int = 2) -> dict[str, Any]:
    """Estimate the compute cost by **measuring** it on a few real files, not by assuming a rate.

    The per-file cost is dominated by loading, notch filtering, decimating from the original rate,
    and event detection, so timing one or two real recordings and scaling by the file count is a
    defensible estimate. The deep cost is estimated by timing one training batch at the window
    length this configuration would actually produce. Every number is reported with the basis it
    was measured on, and the whole block is labelled an estimate.
    """
    import time

    from .manifest import discover_sources
    from .pipeline import process_recording

    out: dict[str, Any] = {"basis": [], "n_probe_files": 0}
    records = discover_sources(cfg.data, "budget")
    n_files = len(records)
    out["n_files_discovered"] = n_files
    if n_files == 0:
        out["status"] = "cannot estimate: no data files were discovered"
        return out
    probes = records[:max(1, min(n_probe, n_files))]
    t0 = time.perf_counter()
    n_events: list[int] = []
    durations: list[float] = []
    for rec in probes:
        try:
            res = process_recording(rec.path, cfg, cfg.training.analysis_rate_hz, keep_signal=False)
        except Exception as exc:
            out.setdefault("probe_errors", []).append(f"{rec.path.name}: {type(exc).__name__}: {exc}")
            continue
        n_events.append(len(res.events))
        durations.extend(ev.offset_s - ev.onset_s for ev in res.events)
    elapsed = time.perf_counter() - t0
    if not n_events:
        out["status"] = "cannot estimate: no probe file could be processed"
        return out
    per_file = elapsed / len(n_events)
    out["n_probe_files"] = len(n_events)
    out["seconds_per_file_measured"] = round(per_file, 3)
    out["events_per_file_measured"] = round(float(np.mean(n_events)), 2)
    out["basis"].append(f"collection cost measured on {len(n_events)} real file(s) on this machine")
    collection = per_file * n_files
    out["estimated_collection_minutes"] = round(collection / 60.0, 1)

    # Model A: time one full CV pass on a feature matrix of the expected shape.
    n_features = 30
    est = collection
    try:
        from .train import cv_predictions

        rng = np.random.default_rng(0)
        n_rep = min(cfg.splitting.n_repeats, 2)
        x = rng.normal(size=(n_files, n_features))
        y = np.array(["A"] * (n_files // 2) + ["B"] * (n_files - n_files // 2), dtype=object)
        g = np.array([f"f{i}" for i in range(n_files)], dtype=object)
        folds = np.zeros((n_rep, n_files), dtype=np.int64)
        for r in range(n_rep):
            folds[r] = np.arange(n_files) % cfg.splitting.n_splits
            folds[r] = (folds[r] + r) % cfg.splitting.n_splits
        t1 = time.perf_counter()
        for name in cfg.training.models:
            cv_predictions(name, x, y, g, folds, ["A", "B"], cfg, None)
        model_a = (time.perf_counter() - t1) / max(n_rep, 1) * cfg.splitting.n_repeats
        perm = model_a * cfg.training.n_permutations * len(cfg.training.models)
        out["estimated_model_a_minutes"] = round(model_a / 60.0, 1)
        out["estimated_permutation_null_minutes"] = round(perm / 60.0, 1)
        out["basis"].append(f"Model A CV timed on a synthetic {n_files}x{n_features} matrix, "
                            "scaled to the configured repeats and permutations")
        est += model_a + perm
    except Exception as exc:  # an estimate must never break the dry run
        out["model_a_estimate_error"] = f"{type(exc).__name__}: {exc}"

    if cfg.deep.enabled and cfg.deep.model_set in ("deep", "hybrid"):
        try:
            est += _estimate_deep_seconds(cfg, n_files, float(np.mean(n_events)), durations, out)
        except Exception as exc:
            out["deep_estimate_error"] = f"{type(exc).__name__}: {exc}"
    out["estimated_total_minutes"] = round(est / 60.0, 1)
    out["status"] = "estimate"
    out["caveat"] = ("measured on this machine and this configuration; actual time varies with "
                     "early stopping, window count, and CPU load")
    return out


def _estimate_deep_seconds(cfg: TrapsigConfig, n_files: int, events_per_file: float,
                           durations: list[float], out: dict[str, Any]) -> float:
    """Estimate the CNN cost from one timed batch at the real window length."""
    import time

    try:
        import torch
    except Exception as exc:
        out["deep_estimate"] = {"status": "not estimable",
                                "reason": f"PyTorch is not importable: {type(exc).__name__}"}
        return 0.0
    from .deep import build_model
    from .windows import resolve_window_length_ms, window_samples

    d = cfg.deep
    wms, wsrc, _ = resolve_window_length_ms(durations, d.window_ms, d.window_ms_auto_multiplier,
                                            d.window_ms_min, d.window_ms_max)
    rate = d.window_rate_hz or cfg.training.analysis_rate_hz
    length = window_samples(wms, rate)
    n_windows = int(min(events_per_file, d.max_windows_per_file) * n_files)
    train_windows = max(8, int(n_windows * (1.0 - 1.0 / cfg.splitting.n_splits)))
    batches = max(1, -(-train_windows // d.batch_size))
    torch.manual_seed(0)
    model = build_model(length, 2, cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=d.learning_rate)
    xb = torch.randn(min(d.batch_size, train_windows), 1, length)
    yb = torch.zeros(xb.shape[0], dtype=torch.long)
    loss_fn = torch.nn.CrossEntropyLoss()
    t0 = time.perf_counter()
    for _ in range(2):
        opt.zero_grad(set_to_none=True)
        loss_fn(model(xb), yb).backward()
        opt.step()
    per_batch = (time.perf_counter() - t0) / 2.0
    folds = cfg.splitting.n_splits * cfg.splitting.n_repeats
    seconds = per_batch * batches * d.epochs * folds * 1.15   # 1.15 covers validation and logging
    from .deep import count_parameters
    out["deep_estimate"] = {
        "status": "estimate",
        "window_ms": round(wms, 3), "window_ms_source": wsrc, "window_rate_hz": rate,
        "window_length_samples": length,
        "estimated_windows": n_windows,
        "estimated_batches_per_epoch": batches,
        "seconds_per_batch_measured": round(per_batch, 4),
        "epochs": d.epochs, "folds": folds,
        "estimated_minutes": round(seconds / 60.0, 1),
        "n_parameters": count_parameters(model),
        "basis": "one timed forward+backward batch at the resolved window length, scaled by "
                 "batches x epochs x folds; early stopping usually makes the real cost lower",
    }
    return seconds


def cmd_run(config_path: str, dry_run: bool, quick: bool, model_set: str | None,
            no_pdf: bool, no_explain: bool, from_prepared: str | None, stages: bool) -> int:
    """The whole pipeline in one command: check, prepare, train, explain, plot, report."""
    from .train import TrainError, run_training

    raw = Path(config_path).read_text(encoding="utf-8")
    try:
        cfg0 = load_config(config_path)
    except ConfigError as exc:
        print(f"CONFIGURATION ERROR: {exc}", file=sys.stderr)
        return 2
    cfg, overrides = _apply_overrides(cfg0, quick, model_set, no_pdf, no_explain)
    if overrides:
        print("overrides applied:")
        for o in overrides:
            print(f"  - {o}")

    from .config import validate_config, validate_modelling

    errs = validate_config(cfg) + validate_modelling(cfg)
    if errs:
        for e in errs:
            print(f"CONFIGURATION ERROR: {e}", file=sys.stderr)
        return 2

    print("\n=== step 1/5: preflight checks ===")
    rep = run_preflight(cfg)
    _print_report(rep)
    budget = estimate_budget(cfg)
    _print_budget(budget)
    if dry_run:
        run_dir = create_run_dir(cfg.output.runs_dir, kind="dryrun")
        _save_preflight(run_dir, rep, cfg, raw)
        write_json_exclusive(run_dir / "budget_estimate.json", budget)
        rows, pool, plan = _discover_rows_and_plan(cfg, run_dir.name)
        write_run_metadata(run_dir, cfg, rows, pool, plan.fold_ids,
                           len(list_ignored_files(cfg.data)), "dry-run")
        print(f"\nDRY RUN: nothing was processed or trained. Report written to {run_dir}")
        print(metadata_summary_line(run_dir))
        print("To run for real, repeat this command without --dry-run.")
        return rep.exit_code
    if rep.errors:
        print("\nrun aborted at preflight: resolve the ERROR entries above "
              "(nothing was prepared or trained).")
        return 1

    print("\n=== step 2/5: prepare dataset (manifest, symlinks, leakage-safe folds) ===")
    if from_prepared:
        prepared = Path(from_prepared)
        if not prepared.is_dir():
            print(f"CONFIGURATION ERROR: prepared dataset not found: {prepared}", file=sys.stderr)
            return 2
        resolved, why_not = _run_dir_for_prepared(cfg, prepared)
        if resolved is None:
            print(f"CONFIGURATION ERROR: {why_not}", file=sys.stderr)
            return 2
        run_dir = resolved
        print(f"reusing the prepared dataset: {prepared}")
    else:
        run_dir = create_run_dir(cfg.output.runs_dir, kind="train")
        code, prepared_opt, prep_summary = _do_prepare(cfg, raw, echo=False, parent=run_dir, rep=rep)
        if code != 0 or prepared_opt is None:
            print("run aborted: preparation failed")
            return code or 1
        prepared = prepared_opt
        print(f"prepared dataset: {prepared}  (inside this run folder)")
        print(metadata_summary_line(prepared))
        print(f"  {prep_summary.get('n_source_files')} source files, "
              f"{prep_summary.get('n_cv_pool')} in the CV pool, "
              f"{prep_summary.get('n_ignored_non_matching')} ignored, "
              f"{prep_summary.get('n_repeats')} repeats x {prep_summary.get('n_splits')} folds")

    print("\n=== step 3/5: train and validate ===")
    _save_preflight(run_dir, rep, cfg, raw, skip_existing=True)
    if not (run_dir / "budget_estimate.json").is_file():
        write_json_exclusive(run_dir / "budget_estimate.json", budget)
    print(f"run folder: {run_dir}")
    try:
        summary = run_training(cfg, prepared, run_dir,
                               progress=lambda m: print(m, flush=True), stages=stages)
    except TrainError as exc:
        print(f"TRAINING ERROR: {exc}", file=sys.stderr)
        print(f"partial outputs (if any) are in {run_dir}")
        return 1
    except Exception as exc:
        print(f"TRAINING FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        if os.environ.get("TRAPSIG_DEBUG"):
            import traceback
            traceback.print_exc()
        print(f"partial outputs (if any) are in {run_dir}")
        return 1

    print("\n=== step 4/5: results ===")
    _print_train_headline(summary)
    _print_gate(summary)

    print("\n=== step 5/5: outputs ===")
    _print_outputs(run_dir, summary)
    return 0


def _print_budget(b: dict[str, Any]) -> None:
    print("\ncompute budget (estimated by measurement, not assumed):")
    if b.get("status") != "estimate":
        print(f"  {b.get('status', 'unavailable')}")
        return
    print(f"  files discovered                {b.get('n_files_discovered')}")
    print(f"  probe files timed               {b.get('n_probe_files')} "
          f"({b.get('seconds_per_file_measured')} s each, "
          f"{b.get('events_per_file_measured')} events each)")
    print(f"  collection (load+filter+events) ~{b.get('estimated_collection_minutes')} min")
    if b.get("estimated_model_a_minutes") is not None:
        print(f"  Model A cross-validation        ~{b.get('estimated_model_a_minutes')} min")
    if b.get("estimated_permutation_null_minutes") is not None:
        print(f"  Model A label-shuffle null      ~{b.get('estimated_permutation_null_minutes')} min")
    d = b.get("deep_estimate") or {}
    if d.get("status") == "estimate":
        print(f"  Model B windows                 {d.get('estimated_windows')} "
              f"at {d.get('window_ms')} ms ({d.get('window_ms_source')})")
        print(f"  Model B architecture            {d.get('n_parameters')} parameters")
        print(f"  Model B cross-validation        ~{d.get('estimated_minutes')} min")
    elif d:
        print(f"  Model B                         {d.get('status')}: {d.get('reason', '')}")
    print(f"  TOTAL                           ~{b.get('estimated_total_minutes')} min")
    print(f"  basis: {'; '.join(b.get('basis', []))}")
    print(f"  caveat: {b.get('caveat')}")
    if b.get("probe_errors"):
        for e in b["probe_errors"][:3]:
            print(f"  probe error: {e}")


def _print_gate(summary: dict[str, Any]) -> None:
    g = ((summary.get("comparison") or {}).get("matched_subset") or {}).get("gate_G7")
    if not g:
        print("\ngate G7 (Model B vs Model A): not evaluated - Model B did not produce results "
              "on a matched subset in this run")
        return
    print(f"\ngate G7 (does Model B beat or match Model A?): {g.get('verdict')}")
    print(f"  {g.get('reason')}")
    print(f"  compared on {g.get('n_files_matched_subset')} files with usable event windows")
    a, bm = g.get("model_a") or {}, g.get("model_b") or {}
    print(f"  Model A best ({a.get('name')}): {_fmt(a.get('value'))}  CI {_fmt_ci(a.get('ci95'))}")
    print(f"  Model B ({bm.get('name')}):     {_fmt(bm.get('value'))}  CI {_fmt_ci(bm.get('ci95'))}")
    print(f"  95% intervals overlap: {g.get('ci_overlap')}")


def _fmt(v: Any) -> str:
    return "n/a" if v is None else (f"{float(v):.3f}" if isinstance(v, (int, float)) else str(v))


def _fmt_ci(ci: Any) -> str:
    if not ci or len(ci) != 2:
        return "n/a"
    return f"[{_fmt(ci[0])}, {_fmt(ci[1])}]"


def _print_outputs(run_dir: Path, summary: dict[str, Any]) -> None:
    print(f"run directory: {run_dir}")
    wanted = [
        ("metadata.csv", "THIS run's split: one row per file, with its fold and role per repeat"),
        ("metadata.json", "run-level record: seed, fingerprint, repeats x folds, class counts"),
        ("summary.json", "every number, with the reason for anything not computed"),
        ("report.pdf", "the readable report: results, figures, limitations, gaps"),
        ("report_data_landscape.csv", "PDF section 1 as CSV"),
        ("report_methods.csv", "PDF section 2 as CSV"),
        ("report_results.csv", "PDF section 3 as CSV: model rows, gate G7, permutation nulls"),
        ("report_figures.csv", "PDF section 4 as CSV"),
        ("report_explainability.csv", "PDF section 5 as CSV"),
        ("report_advanced.csv", "PDF section 6 as CSV"),
        ("report_reproducibility.csv", "PDF section 9 as CSV"),
        ("report_ablation.csv", "PDF section 10 as CSV: feature groups, ablation, redundancy"),
        ("report_models.csv", "one row per model: accuracy, F1, AUC, CI, permutation p"),
        ("model_comparison.csv", "model x metric table"),
        ("metrics.csv", "every metric, per repeat and aggregated"),
        ("predictions.csv", "every out-of-fold prediction, per file"),
        ("features.csv", "per-file features, including skipped files and why"),
        ("windows.csv", "per-window provenance for Model B"),
        ("ablation.csv", "one row per feature subset: accuracy on the same folds"),
        ("feature_redundancy.csv", "near-duplicate feature pairs, reported and never dropped"),
        ("attributions_summary.csv", "temporal attribution by segment"),
        ("explainability.json", "Grad-CAM / integrated gradients / SHAP / shortcut probes"),
        ("advanced_analyses.json", "few-shot, semi-supervised, novelty, clustering"),
        ("report_limitations.csv", "PDF section 7 as CSV: limitations, consequences, safeguards"),
        ("report_information_gaps.csv", "PDF section 8 as CSV: what the data lacks and how to close it"),
        ("metrics_extended.csv", "specificity, MCC, Brier, ECE/MCE, PR-AUC, top-k per model"),
        ("fewshot_episodes.csv", "every few-shot episode: shot, accuracy, margin, protocol"),
        ("report_sampling.csv", "PDF section 11 as CSV: sampling-rate menu and segmentation"),
        ("report_extended_metrics.csv", "PDF section 12 as CSV"),
        ("report_quad.csv", "PDF section 13 as CSV: four-branch architecture, gates, few-shot"),
        ("report_viz_registry.csv", "PDF section 14 as CSV: registry counts per category"),
        ("viz_registry_index.csv", "every registry plot: status, reason, file, CSV twin"),
        ("config_used.yaml", "the exact configuration this run used"),
    ]
    for name, why in wanted:
        p = run_dir / name
        if p.is_file():
            print(f"  [file]  {name:<32} {why}")
    for sub, why in (("figures", "every plot"), ("checkpoints", "saved CNN weights per fold"),
                     ("logs", "per-epoch training log (JSON lines)"),
                     ("viz_registry", "the 60+ registry plots (PNG)"),
                     ("viz_registry_csv", "CSV twins for the numeric registry plots"),
                     ("cache", "analysed signals, reused if you rerun training")):
        d = run_dir / sub
        if d.is_dir():
            n = sum(1 for _ in d.rglob("*") if _.is_file())
            print(f"  [dir]   {sub + '/':<32} {n} file(s): {why}")
    q = summary.get("quad")
    if q:
        print(f"\nquad: four-branch model scored {q.get('n_windows')} windows; "
              f"gates {q.get('gate_mean')}")
        fs = q.get("fewshot_summary") or []
        for r in fs:
            print(f"      few-shot shot={r.get('shot')}: episode accuracy "
                  f"{r.get('episode_accuracy_mean')} +/- {r.get('episode_accuracy_sd')} "
                  f"({r.get('n_episodes')} episodes)")
    deep = summary.get("deep") or {}
    if (deep.get("dependency") or {}).get("available") is False:
        u = deep.get("unavailable") or {}
        print(f"\nNOTE: Model B did not run. {u.get('reason', '')}")
        print(f"      To enable it: {u.get('install_hint', '')}")


def _discover_rows_and_plan(cfg: TrapsigConfig, experiment_id: str
                            ) -> tuple[list[dict[str, Any]], list[int], Any]:
    """Discover files, build the manifest, and draw the fold plan. Shared by every stage.

    Splitting is deliberately part of this so that even ``dry-run`` can write the metadata
    record of the split it would use; drawing folds touches no signal data.
    """
    records = discover_sources(cfg.data, experiment_id)
    rows = build_manifest(
        records,
        default_fs=cfg.sampling.original_hz,
        allow_file_fallback=cfg.data.allow_file_level_fallback,
        label_level=cfg.data.label_level,
        dat=cfg.data.dat,
    )
    pool = [
        i for i, r in enumerate(rows)
        if r["class"] != "NA" and r["group_id"] != "NA" and "unreadable" not in r["quality_flags"]
    ]
    for i in pool:
        rows[i]["split"] = CV_SPLIT
    labels = np.array([rows[i]["class"] for i in pool])
    groups = np.array([rows[i]["group_id"] for i in pool])
    plan = grouped_stratified_folds(groups, labels, cfg.splitting.n_splits,
                                    cfg.splitting.n_repeats, cfg.run.seed)
    return rows, pool, plan


def _run_dir_for_prepared(cfg: TrapsigConfig, prepared: Path) -> tuple[Path | None, str]:
    """The run folder a prepared dataset belongs to, or a fresh one if it belongs to none.

    A dataset prepared by this framework lives at ``runs/train_<ts>/training_<ts>``, so training
    writes into that same ``train_<ts>`` folder and one run stays one directory. A dataset from
    anywhere else (an older layout, or a folder you moved) gets a new run folder, and the pointer
    is recorded in ``metadata.json``.
    """
    parent = prepared.parent
    if parent.name.startswith("train_") and prepared.name.startswith("training_"):
        if (parent / "summary.json").is_file():
            return None, (f"that dataset's run folder already contains training results: {parent}\n"
                          f"  start a new run with `trapsig run --config ...`, or move the dataset "
                          f"out of that folder first")
        return parent, ""
    return create_run_dir(cfg.output.runs_dir, kind="train"), ""


def _do_prepare(cfg: TrapsigConfig, raw: str, echo: bool = True, parent: Path | None = None,
                rep: PreflightReport | None = None) -> tuple[int, Path | None, dict[str, Any]]:
    """Prepare the dataset inside a run folder. Shared by ``prepare`` and ``run``.

    ``parent`` is the run folder to prepare into (``run`` passes the one it already created);
    ``rep`` is a preflight report that was already run and saved, so it is not repeated.
    """
    run_dir = parent if parent is not None else create_run_dir(cfg.output.runs_dir, kind="train")
    prepared = create_prepared_dir(run_dir)
    if rep is None:
        _save_preflight(run_dir, run_preflight(cfg), cfg, raw)
    experiment_id = run_dir.name
    rows, pool, plan = _discover_rows_and_plan(cfg, experiment_id)

    fold_path = prepared / "data" / "manifests" / "fold_assignments.csv"
    fold_lines = ["sample_id,class,group_id," + ",".join(
        f"fold_r{r}" for r in range(cfg.splitting.n_repeats))]
    for pos, i in enumerate(pool):
        folds = ",".join(str(int(plan.fold_ids[r, pos])) for r in range(cfg.splitting.n_repeats))
        fold_lines.append(f"{rows[i]['sample_id']},{rows[i]['class']},{rows[i]['group_id']},{folds}")
    write_text_exclusive(fold_path, "\n".join(fold_lines) + "\n")

    pool_rows = [rows[i] for i in pool]
    links = create_split_links(pool_rows, prepared / "data" / "symlinks")
    by_sid = {r.sample_id: r for r in links}
    for i in pool:
        res = by_sid.get(rows[i]["sample_id"])
        if res is not None:
            rows[i]["symlink_path"] = res.link_path
            rows[i]["link_status"] = res.status
    write_link_report(links, prepared / "data" / "symlinks" / "symlink_report.csv")
    write_manifest_csv(rows, prepared / "data" / "manifests" / "master_manifest.csv")
    ignored = list_ignored_files(cfg.data)
    ig = io.StringIO()
    igw = csv.writer(ig)
    igw.writerow(["relative_path", "extension", "size_bytes", "reason"])
    for rel, ext, size in ignored:
        igw.writerow([rel, ext, size, "not matched by data.file_glob"])
    write_text_exclusive(prepared / "data" / "manifests" / "ignored_files.csv", ig.getvalue())

    summary = {
        "trapsig_version": __version__,
        "run_dir": str(run_dir),
        "prepared_dir": str(prepared),
        "config_fingerprint": cfg.fingerprint(),
        "n_source_files": len(rows),
        "n_cv_pool": len(pool),
        "n_ignored_non_matching": len(ignored),
        "n_unassigned": len(rows) - len(pool),
        "n_repeats": cfg.splitting.n_repeats,
        "n_splits": cfg.splitting.n_splits,
        "link_status_counts": {s: sum(1 for r in links if r.status == s)
                               for s in sorted({r.status for r in links})},
        "holdout": "not created (no replicate or session metadata exists to hold out)",
    }
    write_json_exclusive(prepared / "artifacts" / "run_summary.json", summary)
    ignored_count = len(ignored)
    write_run_metadata(prepared, cfg, rows, pool, plan.fold_ids, ignored_count, "prepare")
    if echo:
        print(f"\nrun folder:       {run_dir}")
        print(f"prepared dataset: {prepared}")
        print(json.dumps(summary, indent=2))
        print(metadata_summary_line(prepared))
    return 0, prepared, summary


def cmd_predict(run: str, model: str, fold_tag: str | None, files: list[str],
                input_dir: str | None, out: str | None, allow_stale: bool) -> int:
    """Score new recordings with the fitted pipelines a finished run saved."""
    from datetime import datetime

    from .predict import PredictError, load_run_config, models_for, predict_files, write_inference_csv

    run_dir = Path(run)
    try:
        cfg = load_run_config(run_dir)
        models = models_for(run_dir, model, fold_tag,
                            None if allow_stale else __version__)
        paths = [Path(f) for f in files]
        if input_dir:
            paths += sorted(Path(input_dir).rglob("*.dat"))
        if not paths:
            print("PREDICT ERROR: give --files and/or --input-dir", file=sys.stderr)
            return 2
        missing = [str(p) for p in paths if not p.is_file()]
        if missing:
            print(f"PREDICT ERROR: not found: {', '.join(missing[:5])}", file=sys.stderr)
            return 2
        rows = predict_files(paths, models, cfg)
    except PredictError as exc:
        print(f"PREDICT ERROR: {exc}", file=sys.stderr)
        return 2
    dest = Path(out) if out else (run_dir / "predictions" /
                                  f"inference_{datetime.now():%Y-%m-%d_%H-%M-%S}.csv")
    dest.parent.mkdir(parents=True, exist_ok=True)
    write_inference_csv(rows, dest)
    ok = [r for r in rows if r["status"] == "ok"]
    bad = [r for r in rows if r["status"] != "ok"]
    print(f"predict: {len(models)} saved {model} model(s) "
          f"({', '.join(m.meta['fold_tag'] for m in models[:4])}"
          f"{' ...' if len(models) > 4 else ''}), {len(paths)} file(s)")
    for r in rows:
        line = f"  {r['filename']:<44} {r.get('predicted') or '-':<12} {r['status']}"
        if r["status"] == "ok":
            line += f"  margin {r['margin_top2']:.3f}"
        else:
            line += f"  ({r['reason'][:60]})"
        print(line)
    if bad:
        print(f"NOTE: {len(bad)} file(s) skipped or failed; they are kept in the CSV with reasons.")
    print(f"inference table: {dest}")
    _ = ok
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="trapsig", description="Trapping/translocation signal framework")
    parser.add_argument("--version", action="version", version=f"trapsig {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_dry = sub.add_parser("dry-run", help="validate configuration and data; no processing")
    p_dry.add_argument("--config", required=True, help="path to YAML configuration")

    p_prep = sub.add_parser("prepare", help="preflight, then write manifest, folds, and symlinks")
    p_prep.add_argument("--config", required=True, help="path to YAML configuration")

    p_ev = sub.add_parser("events", help="label-blind event detection on every recording")
    p_ev.add_argument("--config", required=True, help="path to YAML configuration")
    p_ev.add_argument("--rate-hz", required=True, type=float,
                      help="event-detection sampling rate (must be decided from the data audit)")

    p_tr = sub.add_parser("train", help="Model A: preflight, features, repeated grouped CV (uses a prepare run)")
    p_tr.add_argument("--config", required=True, help="path to YAML configuration")
    p_tr.add_argument("--prepared-run", required=True,
                      help="a run directory created by `trapsig prepare` (its data/manifests are read)")

    p_run = sub.add_parser(
        "run",
        help="ONE COMMAND: preflight, prepare, train, validate, explain, plot, report",
        description="Runs the whole pipeline in order. Use --dry-run first to check the data and "
                    "see the estimated runtime without processing anything.")
    p_run.add_argument("--config", required=True, help="path to YAML configuration")
    p_run.add_argument("--dry-run", action="store_true",
                       help="check configuration and data, estimate the compute budget, then stop")
    p_run.add_argument("--quick", action="store_true",
                       help="fast smoke test: 2 repeats, no permutations, few epochs, no figures")
    p_run.add_argument("--model-set", choices=["features", "deep", "hybrid"], default=None,
                       help="override deep.model_set for this run")
    p_run.add_argument("--no-pdf", action="store_true", help="skip the PDF report")
    p_run.add_argument("--no-explain", action="store_true", help="skip explainability")
    p_run.add_argument("--from-prepared", default=None,
                       help="reuse an existing prepared run instead of preparing again")
    p_run.add_argument("--no-stages", action="store_true",
                       help="train only; skip explainability, figures, and reporting")

    p_pr = sub.add_parser(
        "predict",
        help="inference on new .dat files with the models a finished run saved",
        description="Loads the fitted pipelines of one run's models/ directory (plus their JSON "
                    "sidecars, which name the feature columns and classes) and scores new "
                    "recordings. Skipped or unreadable files are reported, not dropped.")
    p_pr.add_argument("--run", required=True, help="a finished run directory (with models/)")
    p_pr.add_argument("--model", default="logreg",
                      help="saved model name to predict with (default logreg)")
    p_pr.add_argument("--fold-tag", default=None,
                      help="restrict to one saved fold model, e.g. r0f0 (default: average all)")
    p_pr.add_argument("--files", nargs="*", default=[], help="new .dat recordings to score")
    p_pr.add_argument("--input-dir", default=None,
                      help="a directory whose .dat files are scored (class folders are not needed)")
    p_pr.add_argument("--out", default=None,
                      help="output CSV (default <run>/predictions/inference_<timestamp>.csv)")
    p_pr.add_argument("--allow-stale", action="store_true",
                      help="load models saved by another trapsig version (reported, not refused)")

    p_ins = sub.add_parser("inspect-dat", help="list .dat layouts consistent with one file (read-only)")
    p_ins.add_argument("--file", required=True, help="one .dat file to inspect")
    p_ins.add_argument("--rate-hz", required=True, type=float, help="sampling rate from the instrument")
    p_ins.add_argument("--duration-s", type=float, default=None,
                       help="recording length in seconds (e.g. 20); enables header-size matching")
    p_ins.add_argument("--max-header-bytes", type=int, default=4096,
                       help="largest header size to consider (default 4096)")

    args = parser.parse_args(argv)
    try:
        if args.command == "dry-run":
            return cmd_dry_run(args.config)
        if args.command == "prepare":
            return cmd_prepare(args.config)
        if args.command == "train":
            return cmd_train(args.config, args.prepared_run)
        if args.command == "run":
            return cmd_run(args.config, args.dry_run, args.quick, args.model_set, args.no_pdf,
                           args.no_explain, args.from_prepared, not args.no_stages)
        if args.command == "events":
            return cmd_events(args.config, args.rate_hz)
        if args.command == "predict":
            return cmd_predict(args.run, args.model, args.fold_tag, args.files, args.input_dir,
                               args.out, args.allow_stale)
        if args.command == "inspect-dat":
            for line in probe_dat(args.file, args.rate_hz, args.duration_s,
                                  max_header_bytes=args.max_header_bytes):
                print(line)
            return 0
    except ConfigError as exc:
        print(f"CONFIGURATION ERROR: {exc}", file=sys.stderr)
        return 2
    except IngestError as exc:
        print(f"INGEST ERROR: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:  # unreadable config path, or an unreadable file in the config
        print(f"CONFIGURATION ERROR: {exc}", file=sys.stderr)
        return 2
    parser.error("unknown command")
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
