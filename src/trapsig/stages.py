"""Post-training stages: explainability, figures, conditional analyses, and the report.

These run inside the training process because they need the window set and the feature table that
training already built. Rebuilding them would mean a second pass over every recording, which on
this dataset costs minutes for no benefit.

Nothing here changes a performance number. Figures and attributions are produced from the
cross-validated outputs already written, plus one separately fitted explanation model that is
labelled as such wherever it appears.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .config import TrapsigConfig
from .pipeline import process_recording

Say = Any


@dataclass
class TrainContext:
    """Everything the downstream stages need, handed over by :func:`trapsig.train.run_training`."""

    y: np.ndarray
    classes: list[str]
    x: np.ndarray
    feat_cols: list[str]
    groups: np.ndarray
    fold_ids: np.ndarray
    sids: list[str]
    ok_rows: list[dict[str, Any]]
    manifest: dict[str, dict[str, str]]
    entries: dict[str, Any]
    agg_pred: dict[str, np.ndarray]
    imbalance: float
    pool: Any
    deep_branch: Any
    summary: dict[str, Any]
    quad_branch: Any = None
    figures: dict[str, str] = field(default_factory=dict)


def _nanmean_repeats(proba: np.ndarray) -> np.ndarray:
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.asarray(np.nanmean(proba, axis=0), dtype=float)


def run_downstream(cfg: TrapsigConfig, run_dir: Path, ctx: TrainContext, say: Say = None) -> dict[str, Any]:
    """Run the enabled downstream stages and return their status blocks."""
    tell = say or (lambda m: None)
    out: dict[str, Any] = {"stages_requested": [], "stages_run": []}
    figs: dict[str, str] = {}

    expl = _stage_explain(cfg, run_dir, ctx, tell, out)
    if cfg.report.plots:
        out["stages_requested"].append("figures")
        figs, statuses = _stage_figures(cfg, run_dir, ctx, expl, tell)
        out["stages_run"].append("figures")
        out["n_figures"] = len(figs)
        out["figures_index"] = str(run_dir / "figures_index.csv")
        out["figure_status_counts"] = {
            st: sum(1 for s in statuses if s["status"] == st)
            for st in sorted({s["status"] for s in statuses})
        }
    else:
        from . import viz

        statuses = [
            {
                "category": c,
                "figure": "",
                "file": "",
                "status": "not requested",
                "reason": "report.plots is false",
                "csv_twin": "",
                "description": d,
            }
            for c, d in viz.FIGURE_CATEGORIES.items()
        ]
        out["figures_index"] = viz.write_figures_index(statuses, run_dir / "figures_index.csv")
    ctx.figures = figs

    out["stages_requested"].append("viz_registry")
    reg = _stage_registry(cfg, run_dir, ctx, expl, tell)
    out["stages_run"].append("viz_registry")
    out["viz_registry"] = reg
    ctx.summary["viz_registry"] = reg

    adv = _stage_advanced(cfg, run_dir, ctx, tell, out)

    if cfg.report.csv:
        from .report import write_csv_rollups

        out["stages_requested"].append("csv")
        # The section CSVs are written after the analyses so they carry the same content the
        # PDF shows, including sections 5 and 6.
        out["csv"] = write_csv_rollups(ctx.summary, run_dir, expl, adv, figs, cfg.to_dict())
        out["stages_run"].append("csv")
    if cfg.report.pdf:
        from .report import build_pdf

        out["stages_requested"].append("pdf")
        out["pdf"] = build_pdf(ctx.summary, run_dir, figs, expl, adv, cfg.to_dict(), tell)
        out["stages_run"].append("pdf")
    return out


# ---------------------------------------------------------------------------
# Visualization registry
# ---------------------------------------------------------------------------


def _stage_registry(
    cfg: TrapsigConfig, run_dir: Path, ctx: TrainContext, expl: dict[str, Any] | None, tell: Say
) -> dict[str, Any]:
    """Run the 60+ entry visualization registry against what this run actually produced."""
    from . import viz_registry as vreg

    signals: dict[str, list[np.ndarray]] = {}
    sig_sids: dict[str, list[str]] = {}
    cache = getattr(ctx.pool, "cache_dir", None)
    if cache and Path(cache).is_dir():
        for row in ctx.ok_rows:
            c = str(row.get("class"))
            if len(signals.setdefault(c, [])) >= 3:
                continue
            f = Path(cache) / f"{row.get('sample_id')}.npy"
            if f.is_file():
                try:
                    signals[c].append(np.load(f))
                    sig_sids.setdefault(c, []).append(str(row.get("sample_id")))
                except Exception as exc:
                    tell(f"registry: could not load cached signal for {row.get('sample_id')}: {exc}")
    meta_rows: list[dict[str, Any]] = []
    meta_csv = run_dir / "metadata.csv"
    if meta_csv.is_file():
        import csv as _csv

        with meta_csv.open(newline="", encoding="utf-8") as fh:
            meta_rows = [dict(r) for r in _csv.DictReader(fh)]
    probas: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    preds: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, ent in ctx.entries.items():
        m = ent.mask
        if not np.any(m):
            continue
        pm = ent.proba[:, m, :].mean(axis=0)
        if np.all(np.isfinite(pm)):
            probas[name] = (ctx.y[m], pm)
            if name in ctx.agg_pred:
                preds[name] = (ctx.y[m], ctx.agg_pred[name][m])
    histories: dict[str, list[dict[str, Any]]] = {}
    if ctx.deep_branch is not None and getattr(ctx.deep_branch, "cv", None) is not None:
        histories["cnn1d"] = list(getattr(ctx.deep_branch.cv, "logs", []) or [])
    embeddings: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    gates: dict[str, float] = {}
    fewshot: list[dict[str, Any]] = []
    if ctx.quad_branch is not None:
        qb = ctx.quad_branch
        histories["quad"] = list(qb.logs)
        gates = dict(qb.gates_mean)
        fewshot = list(qb.fewshot_rows)
        if qb.file_emb:
            pos = {sid: i for i, sid in enumerate(ctx.sids)}
            order = [sid for sid in ctx.sids if sid in qb.file_emb]
            emb = np.stack([qb.file_emb[sid] for sid in order])
            yq = ctx.y[[pos[sid] for sid in order]]
            embeddings["quad"] = (emb, yq)
    rctx = vreg.RegistryCtx(
        classes=list(ctx.classes),
        fs=float(cfg.training.analysis_rate_hz),
        signals=signals,
        x=ctx.x,
        feat_names=list(ctx.feat_cols),
        y=ctx.y,
        ok_rows=list(ctx.ok_rows),
        probas=probas,
        preds=preds,
        histories=histories,
        embeddings=embeddings,
        fewshot=fewshot,
        gates=gates,
        summary=dict(ctx.summary),
        explain=dict(expl or {}),
        redundancy=list(((ctx.summary.get("feature_redundancy") or {}).get("pairs")) or []),
        events=list(getattr(ctx.pool, "event_table", []) or []),
        fold_ids=ctx.fold_ids,
        groups=ctx.groups,
        sids=sig_sids,
        meta_rows=meta_rows,
        cfg_snapshot={
            "analysis_rate_hz": float(cfg.training.analysis_rate_hz),
            "original_hz": float(cfg.sampling.original_hz),
            "dat_config": dict(cfg.to_dict().get("data", {}).get("dat", {})),
            "rate_menu_hz": [float(v) for v in cfg.sampling.rate_menu_hz],
            "filter_mode": str(cfg.filters.mode),
            "lpf_cutoff_hz": cfg.filters.lpf_cutoff_hz,
            "gaussian_sigma_s": cfg.filters.gaussian_sigma_s,
            "events": {
                "threshold_sigma": float(cfg.events.threshold_sigma),
                "min_duration_s": float(cfg.events.min_duration_s),
                "min_separation_s": float(cfg.events.min_separation_s),
            },
            "psd_bands_hz": [list(b) for b in cfg.training.psd_bands_hz],
            "acf_lags_ms": [float(v) for v in cfg.training.acf_lags_ms],
            "window_length_samples": int(getattr(ctx.pool, "window_length_samples", 0) or 0),
            "window_rate_hz": float(getattr(ctx.pool, "window_rate_hz", float("nan")) or float("nan")),
            "window_ms": float(getattr(ctx.pool, "window_ms", float("nan")) or float("nan")),
            "quad_lr_schedule": str(cfg.quad.lr_schedule),
            "quad_epochs": int(cfg.quad.epochs),
            "quad_pretrain_epochs": int(cfg.quad.pretrain_epochs),
        },
    )
    statuses = vreg.render(
        rctx, run_dir / "viz_registry", run_dir / "viz_registry_csv", int(cfg.report.dpi), tell
    )
    index = vreg.write_registry_index(statuses, run_dir / "viz_registry_index.csv")
    cats: dict[str, dict[str, int]] = {}
    for st in statuses:
        d = cats.setdefault(str(st["category"]), {"entries": 0})
        d["entries"] += 1
        d[str(st["status"])] = d.get(str(st["status"]), 0) + 1
    counts: dict[str, int] = {}
    for st in statuses:
        counts[str(st["status"])] = counts.get(str(st["status"]), 0) + 1
    tell(f"registry: {len(statuses)} entries, {counts}")
    return {
        "entries": len(statuses),
        "counts": counts,
        "by_category": cats,
        "index": index,
        "csv_twins": sum(1 for st in statuses if st.get("csv_twin")),
        "png_dir": str(run_dir / "viz_registry"),
    }


# ---------------------------------------------------------------------------
# Explainability
# ---------------------------------------------------------------------------


def _stage_explain(
    cfg: TrapsigConfig, run_dir: Path, ctx: TrainContext, tell: Say, out: dict[str, Any]
) -> dict[str, Any] | None:
    if not cfg.explain.enabled:
        out["explain"] = {"enabled": False}
        return None
    from .explain import run_explainability

    out["stages_requested"].append("explain")
    tell("stage: explainability")
    ws = ctx.pool.windows if ctx.pool is not None else None
    res = run_explainability(
        cfg,
        run_dir,
        ctx.summary,
        ws,
        ctx.x,
        ctx.y,
        ctx.groups,
        ctx.feat_cols,
        ctx.classes,
        ctx.fold_ids,
        ctx.sids,
        tell,
    )
    out["explain"] = {
        "enabled": True,
        "methods": list((res.get("methods") or {}).keys()),
        "torch_available": res.get("torch_available"),
        "shap_available": res.get("shap_available"),
    }
    out["stages_run"].append("explain")
    return res


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _write_rows_csv(rows: list[dict[str, Any]], path: Path, fields: list[str]) -> str:
    """The machine-readable twin of a figure: the same numbers the PNG draws."""
    import csv
    import io

    from .runs import write_text_exclusive

    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
    w.writeheader()
    for r in rows:
        w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in fields})
    return str(write_text_exclusive(path, buf.getvalue()))


def _read_csv(path: Path) -> list[dict[str, Any]]:
    """Read back an artifact the run already wrote, so a figure cannot disagree with its twin."""
    import csv

    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return [dict(r) for r in csv.DictReader(fh)]


def _stage_figures(
    cfg: TrapsigConfig, run_dir: Path, ctx: TrainContext, expl: dict[str, Any] | None, tell: Say
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    from . import viz

    figs: dict[str, str] = {}
    # One status row per requested category, generated or not: "we did not draw it" must be
    # distinguishable from "we forgot to draw it".
    statuses: list[dict[str, Any]] = [
        {
            "category": c,
            "figure": "",
            "file": "",
            "status": "not requested",
            "reason": "report.plots is true but this category produced nothing on this run",
            "csv_twin": "",
            "description": d,
        }
        for c, d in viz.FIGURE_CATEGORIES.items()
    ]

    def record(
        category: str, figure: str, status: str, reason: str = "", file: str = "", csv_twin: str = ""
    ) -> None:
        for st in statuses:
            if st["category"] == category:
                st.update(
                    {"figure": figure, "status": status, "reason": reason, "file": file, "csv_twin": csv_twin}
                )
                return
        statuses.append(
            {
                "category": category,
                "figure": figure,
                "status": status,
                "reason": reason,
                "file": file,
                "csv_twin": csv_twin,
                "description": viz.FIGURE_CATEGORIES.get(category, ""),
            }
        )

    def attempt(category: str, figure: str, fn: Any, twin: str = "") -> None:
        """Draw one figure; a failure is recorded, never fatal to the run."""
        p = fig_dir / f"{figure}.png"
        try:
            r = fn(p)
        except Exception as exc:
            record(category, figure, "failed", f"{type(exc).__name__}: {exc}", "", twin)
            tell(f"figure {figure} failed: {exc}")
            return
        if r:
            figs[figure] = r
            record(category, figure, "generated", "", r, twin)
        else:
            record(
                category,
                figure,
                "skipped",
                "the data this figure needs is absent or degenerate on this run",
                "",
                twin,
            )

    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    dpi = cfg.report.dpi
    tell("stage: figures")

    # Scientific signal figures, from a few example files.
    n_ex = max(1, cfg.report.max_plots_per_class)
    per_class: dict[str, int] = {}
    for row in ctx.ok_rows:
        c = str(row["class"])
        if per_class.get(c, 0) >= n_ex:
            continue
        per_class[c] = per_class.get(c, 0) + 1
        path = (ctx.manifest.get(row["sample_id"]) or {}).get("original_path", "")
        if not path:
            continue
        try:
            res = process_recording(path, cfg, cfg.training.analysis_rate_hz, keep_signal=True)
        except Exception:
            continue  # a figure is never worth aborting a run for
        if res.signal is None or res.baseline is None:
            continue
        p = fig_dir / f"signal_{row['sample_id']}.png"
        figs[f"signal_{row['sample_id']} (class {c})"] = viz.plot_recording(
            res.signal,
            res.baseline,
            res.fs_hz,
            res.events,
            f"{row['filename']} - class {c}, analysed at {res.fs_hz:g} Hz, {len(res.events)} detected events",
            p,
            dpi,
        )
        record(
            "signal_examples",
            f"signal_{row['sample_id']}",
            "generated",
            "",
            str(figs[f"signal_{row['sample_id']} (class {c})"]),
            "",
        )

    # Additive raw-source figures: one independent full-recording overview per metadata row,
    # organized by actual dataset class labels under figures/files/<class>/.
    try:
        import csv as _csv

        from .viz_files import write_raw_file_plots

        _meta_path = run_dir / "metadata.csv"
        _meta_rows: list[dict[str, Any]] = []
        if _meta_path.is_file():
            with _meta_path.open(newline="", encoding="utf-8") as _fh:
                _meta_rows = [dict(_r) for _r in _csv.DictReader(_fh)]
        _raw_rows, _raw_index = write_raw_file_plots(
            _meta_rows, ctx.manifest, cfg.data.dat, cfg.sampling.original_hz, fig_dir, dpi, tell
        )
        _n_raw = sum(_r.get("status") == "generated" for _r in _raw_rows)
        statuses.append(
            {
                "category": "raw_per_file",
                "figure": "raw_file_waveforms",
                "status": "generated" if _n_raw else "skipped",
                "reason": "" if _n_raw else "no readable source recordings",
                "file": str(fig_dir / "files"),
                "csv_twin": _raw_index,
                "description": f"per-file raw overview figures: {_n_raw}/{len(_raw_rows)}",
            }
        )
    except Exception as exc:
        statuses.append(
            {
                "category": "raw_per_file",
                "figure": "raw_file_waveforms",
                "status": "failed",
                "reason": f"{type(exc).__name__}: {exc}",
                "file": "",
                "csv_twin": "",
                "description": "one raw waveform figure per source file",
            }
        )
        tell(f"per-file raw waveform figures failed: {exc}")

    ws = ctx.pool.windows if ctx.pool is not None else None
    if ws is None:
        record(
            "class_mean_window",
            "class_mean_event_window",
            "not requested",
            "no event windows were built on this configuration (the deep encoder is off), "
            "so there is no class-mean window to draw",
            "",
            "",
        )
    else:
        attempt(
            "class_mean_window",
            "class_mean_event_window",
            lambda p: viz.plot_class_mean_windows(ws, ctx.classes, p, dpi),
        )
    attempt(
        "psd_by_class",
        "psd_by_class",
        lambda p: viz.plot_psd_by_class(ctx.ok_rows, ctx.classes, cfg.training.psd_bands_hz, p, dpi),
    )

    # ---- dataset and engineered-feature figures (stage B/D/F of the data pipeline) ----
    comp_twin = ""
    try:
        comp_twin = _write_rows_csv(
            viz.dataset_composition_rows(ctx.ok_rows, ctx.classes),
            run_dir / "report_dataset_composition.csv",
            [
                "class",
                "n_files",
                "n_files_with_events",
                "n_files_without_events",
                "events_total",
                "events_per_file_mean",
                "events_per_file_sd",
                "events_per_file_min",
                "events_per_file_max",
                "duration_s_total",
                "mean_missing_feature_fraction",
            ],
        )
    except Exception as exc:
        tell(f"dataset composition CSV failed: {exc}")
    attempt(
        "dataset_composition",
        "dataset_composition",
        lambda p: viz.plot_dataset_composition(ctx.ok_rows, ctx.classes, ctx.imbalance, p, dpi),
        comp_twin,
    )

    if ctx.x.size and ctx.feat_cols:
        dist_twin = _write_rows_csv(
            viz.feature_distribution_rows(ctx.x, ctx.feat_cols, ctx.y, ctx.classes),
            run_dir / "report_feature_distributions.csv",
            [
                "feature",
                "group",
                "class",
                "n",
                "n_missing",
                "mean",
                "sd",
                "min",
                "q25",
                "median",
                "q75",
                "max",
            ],
        )
        attempt(
            "feature_distributions",
            "feature_distributions",
            lambda p: viz.plot_feature_distributions(ctx.x, ctx.feat_cols, ctx.y, ctx.classes, p, dpi),
            dist_twin,
        )
        miss_twin = _write_rows_csv(
            viz.missing_value_rows(ctx.x, ctx.feat_cols, ctx.sids),
            run_dir / "report_missing_values.csv",
            ["row_type", "key", "group", "n_rows", "n_missing", "fraction_missing", "all_missing"],
        )
        attempt(
            "missing_values",
            "missing_values",
            lambda p: viz.plot_missing_pattern(ctx.x, ctx.feat_cols, ctx.sids, p, dpi),
            miss_twin,
        )
        corr_twin = ""
        try:
            corr_twin = _write_rows_csv(
                viz.feature_correlation_rows(
                    ctx.x, ctx.feat_cols, threshold=cfg.training.redundancy_threshold, report_at=0.80
                ),
                run_dir / "report_feature_correlation.csv",
                [
                    "feature_a",
                    "feature_b",
                    "group_a",
                    "group_b",
                    "spearman",
                    "abs_spearman",
                    "redundant_at_threshold",
                ],
            )
        except Exception as exc:
            tell(f"feature correlation CSV failed: {exc}")
        attempt(
            "feature_correlation",
            "feature_correlation",
            lambda p: viz.plot_feature_correlation(
                ctx.x, ctx.feat_cols, ctx.y, ctx.classes, p, dpi, threshold=cfg.training.redundancy_threshold
            ),
            corr_twin,
        )

    # Model figures.
    if ctx.deep_branch is not None and ctx.deep_branch.cv is not None:
        attempt(
            "deep_learning_curves",
            "learning_curves",
            lambda p: viz.plot_learning_curves(ctx.deep_branch.cv.logs, p, dpi),
        )
    else:
        record(
            "deep_learning_curves",
            "learning_curves",
            "not requested",
            "the deep encoder did not run on this configuration",
            "",
            "",
        )

    for name in ctx.entries:
        if name not in ctx.agg_pred:
            continue
        m = ctx.entries[name].mask
        figs[f"confusion_{name}"] = viz.plot_confusion(
            ctx.y[m],
            ctx.agg_pred[name][m],
            ctx.classes,
            f"{name} (aggregated over repeats, n={int(m.sum())})",
            fig_dir / f"confusion_{name}.png",
            dpi,
        )
        record("confusion", f"confusion_{name}", "generated", "", figs[f"confusion_{name}"], "")
        proba = _nanmean_repeats(ctx.entries[name].proba)[m]
        finite = np.all(np.isfinite(proba), axis=1)
        if finite.any():
            r = viz.plot_roc(
                ctx.y[m][finite],
                proba[finite],
                ctx.classes,
                f"{name} (out-of-fold, aggregated)",
                fig_dir / f"roc_{name}.png",
                dpi,
            )
            if r:
                figs[f"roc_{name}"] = r
                record("roc", f"roc_{name}", "generated", "", r, "")
            r = viz.plot_calibration(
                ctx.y[m][finite], proba[finite], name, fig_dir / f"calibration_{name}.png", dpi
            )
            if r:
                figs[f"calibration_{name}"] = r
                record("calibration", f"calibration_{name}", "generated", "", r, "")
        else:
            record(
                "roc",
                f"roc_{name}",
                "skipped",
                "no finite out-of-fold probability for this model on its mask",
                "",
                "",
            )
        if cfg.training.pr_curves and finite.any():
            r = viz.plot_pr_curve(
                ctx.y[m][finite],
                proba[finite],
                ctx.classes,
                f"{name} (out-of-fold, aggregated)",
                fig_dir / f"pr_{name}.png",
                dpi,
            )
            if r:
                figs[f"pr_{name}"] = r
                record("pr_curves", f"pr_{name}", "generated", "", r, "")

        # Error analysis: the per-repeat predictions, not the aggregated ones, so a file that is
        # right in two repeats and wrong in one is visible as unstable rather than as correct.
        err_twin = _write_rows_csv(
            viz.error_analysis_rows(
                ctx.y, ctx.entries[name].pred, ctx.entries[name].proba, ctx.sids, ctx.classes, name
            ),
            run_dir / f"report_error_analysis_{name}.csv",
            [
                "row_type",
                "model",
                "sample_id",
                "class",
                "repeat",
                "predicted",
                "correct",
                "n_repeats_scored",
                "n_wrong",
                "error_rate",
                "modal_wrong_class",
                "wrong_class_counts",
                "prob_of_true_class",
                "max_prob",
                "margin_top2",
                "mean_prob_of_true_class_when_correct",
                "mean_prob_of_true_class_when_wrong",
                "failure_case",
            ],
        )
        attempt(
            "error_analysis",
            f"error_analysis_{name}",
            lambda p, nm=name, tw=err_twin: viz.plot_error_analysis(
                ctx.y, ctx.entries[nm].pred, ctx.entries[nm].proba, ctx.sids, ctx.classes, nm, p, dpi
            ),
            err_twin,
        )

    attempt(
        "model_comparison",
        "model_comparison_with_bootstrap_intervals",
        lambda p: viz.plot_model_comparison(ctx.summary, p, dpi),
    )

    # ---- four-branch diagnostics ----
    if ctx.quad_branch is not None:
        qb = ctx.quad_branch
        attempt("quad_gates", "quad_fusion_gates", lambda p: viz.plot_quad_gates(qb.gates_mean, p, dpi))
        fs_rows = (ctx.summary.get("quad") or {}).get("fewshot_summary") or []
        attempt("shot_curve", "fewshot_shot_curve", lambda p: viz.plot_shot_curve(fs_rows, p, dpi))
        attempt(
            "branch_ablation", "quad_branch_ablation", lambda p: viz.plot_branch_ablation(ctx.summary, p, dpi)
        )
        if qb.file_emb:
            pos = {sid: i for i, sid in enumerate(ctx.sids)}
            order = [sid for sid in ctx.sids if sid in qb.file_emb]
            emb = np.stack([qb.file_emb[sid] for sid in order])
            yq = ctx.y[[pos[sid] for sid in order]]
            attempt(
                "quad_embedding",
                "quad_embedding_pca",
                lambda p: viz.plot_embedding_xy(emb, yq, ctx.classes, p, dpi, "pca"),
            )
            attempt(
                "quad_embedding",
                "quad_embedding_tsne",
                lambda p: viz.plot_embedding_xy(emb, yq, ctx.classes, p, dpi, "tsne"),
            )
    else:
        for cat, fig in (
            ("quad_gates", "quad_fusion_gates"),
            ("shot_curve", "fewshot_shot_curve"),
            ("branch_ablation", "quad_branch_ablation"),
            ("quad_embedding", "quad_embedding_pca"),
        ):
            record(
                cat, fig, "not requested", "quad.enabled is false: the four-branch model did not run", "", ""
            )

    if ctx.deep_branch is None or ctx.deep_branch.cv is None:
        record(
            "embedding",
            "out_of_fold_embedding_pca",
            "not requested",
            "the deep encoder did not run, so there is no learned representation to project",
            "",
            "",
        )
    else:
        from .advanced import file_representation

        rep = file_representation(ctx.deep_branch.cv, ws, ctx.sids, repeat=0)
        if rep is None:
            record(
                "embedding",
                "out_of_fold_embedding_pca",
                "skipped",
                "no fold produced an out-of-fold embedding for enough files to project",
                "",
                "",
            )
        if rep is not None:
            r = viz.plot_embedding(rep, ctx.classes, fig_dir / "embedding_pca.png", dpi)
            if r:
                figs["out_of_fold_embedding_pca"] = r
                record("embedding", "out_of_fold_embedding_pca", "generated", "", r, "")

    # Attributions for a few explained windows.
    if not (expl and ws is not None and ws.x.shape[0] and (run_dir / "attributions.npz").is_file()):
        record(
            "attribution",
            "attribution_window_*",
            "not requested",
            "explain.enabled is false or no window attribution was computed",
            "",
            "",
        )
    if expl and ws is not None and ws.x.shape[0] and (run_dir / "attributions.npz").is_file():
        data = np.load(run_dir / "attributions.npz", allow_pickle=True)
        methods = [m for m in ("gradcam", "integrated_gradients", "occlusion") if m in data]
        for i, w_i in enumerate(data["window_index"][: min(4, len(data["window_index"]))]):
            attrs = {m: data[m][i] for m in methods}
            p = fig_dir / f"attribution_window_{int(w_i)}.png"
            r = viz.plot_attribution(ws, ws.x[int(w_i)], attrs, int(w_i), p, dpi)
            if r:
                figs[f"attribution_window_{int(w_i)}"] = r
                record("attribution", f"attribution_window_{int(w_i)}", "generated", "", r, "")

    if expl:
        attempt(
            "feature_importance",
            "model_a_feature_importance",
            lambda p: viz.plot_feature_importance(expl, p, dpi),
        )
    else:
        record(
            "feature_importance",
            "model_a_feature_importance",
            "not requested",
            "explain.enabled is false, so no attribution was computed",
            "",
            "",
        )

    # ---- tuning and ablation figures, drawn from the CSVs the run already wrote ----
    abl_rows = list((ctx.summary.get("ablation") or {}).get("rows") or [])
    attempt(
        "ablation",
        "ablation_feature_domains",
        lambda p: viz.plot_ablation(abl_rows, p, dpi),
        str(run_dir / "ablation.csv") if abl_rows else "",
    )
    hpo_rows = _read_csv(run_dir / "hpo_trials.csv")
    attempt(
        "hpo",
        "hpo_trials",
        lambda p: viz.plot_hpo(hpo_rows, p, dpi),
        str(run_dir / "hpo_trials.csv") if hpo_rows else "",
    )

    index = viz.write_figures_index(statuses, run_dir / "figures_index.csv")
    tell(f"stage: figures done ({len(figs)} written, index {Path(index).name})")
    return figs, statuses


def _advanced_status(v: Any) -> Any:
    """A one-line status per analysis, so the console summary is not full of nulls."""
    if not isinstance(v, dict):
        return v
    if v.get("status"):
        return str(v["status"])
    if "results" in v:  # few-shot: one entry per K
        return {k: str(r.get("status", "")) for k, r in (v["results"] or {}).items()}
    if "summary" in v:  # semi-supervised
        su = v["summary"]
        return {
            "supervised_only": su.get("supervised_only_mean"),
            "with_pseudo_labels": su.get("with_pseudo_labels_mean"),
            "accepted": su.get("n_accepted_pseudo_labels"),
        }
    if "leave_one_class_out_diagnostic" in v:
        return {k: e.get("auroc_distance") for k, e in (v["leave_one_class_out_diagnostic"] or {}).items()}
    return "computed" if v else "empty"


# ---------------------------------------------------------------------------
# Conditional analyses
# ---------------------------------------------------------------------------


def _stage_advanced(
    cfg: TrapsigConfig, run_dir: Path, ctx: TrainContext, tell: Say, out: dict[str, Any]
) -> dict[str, Any] | None:
    a = cfg.advanced
    if not any([a.few_shot, a.semi_supervised, a.novelty, a.clustering]):
        return None
    if ctx.deep_branch is None or ctx.deep_branch.cv is None:
        out["advanced"] = {
            "status": "not applicable",
            "reason": "these analyses use the Model B representation, and Model B "
            "did not produce one in this run",
        }
        return None
    from .advanced import file_representation, run_advanced
    from .runs import write_json_exclusive

    out["stages_requested"].append("advanced")
    tell("stage: conditional analyses (few-shot / semi-supervised / novelty / clustering)")
    ws = ctx.pool.windows
    rep = file_representation(ctx.deep_branch.cv, ws, ctx.sids, repeat=0)
    res = run_advanced(
        cfg, rep, ws, ctx.fold_ids, ctx.sids, ctx.classes, ctx.deep_branch.cv, ctx.imbalance, tell
    )
    write_json_exclusive(run_dir / "advanced_analyses.json", res)
    out["advanced"] = {
        k: _advanced_status(v) for k, v in res.items() if k.endswith("_results") or k == "status"
    }
    out["stages_run"].append("advanced")
    return res
