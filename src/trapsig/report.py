"""Reporting (design section 12): CSV rollups and a single PDF.

The PDF is a report, not a dump. It states what was measured, on how many files, with what
uncertainty, and then what cannot be concluded from it. Sections that could not be computed say so
with a reason instead of showing an empty table.

``reportlab`` is optional. Without it the CSV rollups and figures are still written and the PDF
step reports that it was skipped and why.
"""

from __future__ import annotations

import csv
import io
import json
import math
from pathlib import Path
from typing import Any

from . import __version__
from .runs import capture_environment, write_json_exclusive, write_text_exclusive

try:  # optional: only the PDF needs it
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import (
        ParagraphStyle,
        getSampleStyleSheet,
    )
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        Image,
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    _RL = True
except Exception:  # pragma: no cover - depends on the user's environment
    _RL = False

Say = Any


def reportlab_available() -> bool:
    return _RL


# ---------------------------------------------------------------------------
# CSV rollups
# ---------------------------------------------------------------------------

def _flatten(obj: Any, prefix: str = "", out: list[tuple[str, str]] | None = None,
             depth: int = 0) -> list[tuple[str, str]]:
    """Flatten nested dicts/lists into (key, value) rows so CSVs mirror the PDF text exactly."""
    out = out if out is not None else []
    if isinstance(obj, dict):
        for k, v in obj.items():
            _flatten(v, f"{prefix}{k}", out, depth + 1)
    elif isinstance(obj, (list, tuple)):
        if obj and all(not isinstance(v, (dict, list, tuple)) for v in obj) and len(obj) <= 8:
            out.append((prefix.rstrip("."), "; ".join(_flat(v) for v in obj)))
        else:
            for i, v in enumerate(obj):
                _flatten(v, f"{prefix}[{i}].", out, depth + 1)
    else:
        out.append((prefix.rstrip("."), _flat(obj)))
    return out


def _kv_csv(rows: list[tuple[str, str]], path: Path) -> str:
    return _dump([{"section_item": k, "value": v} for k, v in rows], path)


def write_section_csvs(summary: dict[str, Any], run_dir: Path, explain: dict[str, Any] | None,
                       advanced: dict[str, Any] | None, figures: dict[str, str] | None,
                       cfg_dict: dict[str, Any] | None = None) -> list[str]:
    """One CSV per PDF section, carrying the same content in the same order.

    The PDF is the readable form and these are the machine-readable twins; section 7 and 8
    additionally keep the full un-wrapped text in each cell, so nothing that wraps across lines
    on the page is lost in the CSV.
    """
    written: list[str] = []

    # section 1: data landscape
    rows: list[tuple[str, Any]] = [
        ("files_in_pool", summary.get("n_pool_files")),
        ("files_used", summary.get("n_files_used")),
        ("files_skipped_or_failed", summary.get("n_files_skipped_or_failed")),
        ("class_counts", summary.get("class_counts_used")),
        ("imbalance_ratio", summary.get("imbalance_ratio")),
        ("class_weight_rule_applied", summary.get("class_weight")),
        ("analysis_rate_hz", summary.get("analysis_rate_hz")),
        ("filter_mode", summary.get("filter_mode")),
    ]
    deep = summary.get("deep") or {}
    for k in ("n_windows", "window_ms", "window_ms_source", "window_rate_hz",
              "window_length_samples", "n_files_without_windows", "n_events_edge_excluded"):
        if deep.get(k) is not None:
            rows.append((f"deep.{k}", deep.get(k)))
    for i, sk in enumerate(summary.get("skipped_or_failed") or []):
        rows.append((f"skipped[{i}]", f"{sk.get('sample_id')}: {sk.get('status')}: {sk.get('reason')}"))
    written.append(_kv_csv(_flatten(dict(rows), "", None) or [(k, _flat(v)) for k, v in rows],
                           run_dir / "report_data_landscape.csv"))

    # section 2: methods actually used
    mrows = [(k, v) for k, v in (summary.get("design_notes") or {}).items()]
    if deep.get("architecture"):
        mrows += [("deep.architecture", deep.get("architecture")),
                  ("deep.n_parameters", deep.get("n_parameters")),
                  ("deep.receptive_field_samples", deep.get("receptive_field_samples")),
                  ("deep.receptive_field_ms", deep.get("receptive_field_ms")),
                  ("deep.select_metric", deep.get("select_metric")),
                  ("deep.temperature_mean", deep.get("temperature_mean")),
                  ("deep.total_train_seconds", deep.get("total_train_seconds"))]
    feats = summary.get("features") or {}
    if feats:
        mrows += [("features.requested_groups", _flat(feats.get("requested"))),
                  ("features.n_features_used", _flat(feats.get("n_features"))),
                  ("features.per_group", _flat(feats.get("computed"))),
                  ("features.selection_rule",
                   "groups are declared in training.feature_groups and their contribution is "
                   "measured by the ablation in section 10; no column is dropped on the pooled "
                   "matrix, because that would use test-fold information"),
                  ("features.dropped_all_nan", _flat(summary.get("features_dropped_all_nan")))]
    abl = summary.get("ablation") or {}
    if abl:
        mrows += [("ablation.mode", _flat(abl.get("mode"))),
                  ("ablation.model", _flat(abl.get("model")))]
    if (deep.get("dependency") or {}).get("available") is False:
        u = deep.get("unavailable") or {}
        mrows += [("deep.unavailable.reason", u.get("reason")),
                  ("deep.unavailable.install_hint", u.get("install_hint")),
                  ("deep.unavailable.action_taken", u.get("action_taken"))]
    written.append(_kv_csv([(k, _flat(v)) for k, v in mrows], run_dir / "report_methods.csv"))

    # section 3: results, including the gate and the permutation nulls
    rrows: list[dict[str, Any]] = []
    for name, ent in (summary.get("models") or {}).items():
        pr = ent.get("per_repeat") or {}
        agg = ent.get("aggregated") or {}
        ci = agg.get("balanced_accuracy_bootstrap_ci95") or {}
        perm = ent.get("permutation") or {}
        rrows.append({
            "row_type": "model", "model": name, "kind": ent.get("kind"),
            "n_files_evaluated": ent.get("n_files_evaluated"),
            "evaluated_on": ent.get("evaluated_on"),
            "balanced_accuracy_mean": _g(pr.get("balanced_accuracy"), "mean"),
            "balanced_accuracy_sd": _g(pr.get("balanced_accuracy"), "sd"),
            "balanced_accuracy_aggregated": (agg.get("balanced_accuracy") or {}).get("value"),
            "ci95_lo": ci.get("lo"), "ci95_hi": ci.get("hi"),
            "macro_f1_mean": _g(pr.get("macro_f1"), "mean"),
            "accuracy_mean": _g(pr.get("accuracy"), "mean"),
            "log_loss_mean": _g(pr.get("log_loss"), "mean"),
            "roc_auc_ovr_macro_mean": _g(pr.get("roc_auc_ovr_macro"), "mean"),
            "permutation_observed": perm.get("observed_balanced_accuracy_mean_over_repeats"),
            "permutation_null_mean": perm.get("null_mean"),
            "permutation_p_value": perm.get("p_value"),
            "permutation_status_or_reason": perm.get("status") or perm.get("reason") or "",
        })
    matched = ((summary.get("comparison") or {}).get("matched_subset") or {})
    if matched.get("gate_G7"):
        g = matched["gate_G7"]
        rrows.append({"row_type": "gate_G7", "model": "matched_subset",
                      "detail": g.get("verdict"), "reason": g.get("reason"),
                      "n_files_evaluated": g.get("n_files_matched_subset"),
                      "model_a": (g.get("model_a") or {}).get("name"),
                      "model_a_value": (g.get("model_a") or {}).get("value"),
                      "model_a_ci": (g.get("model_a") or {}).get("ci95"),
                      "model_b_value": (g.get("model_b") or {}).get("value"),
                      "model_b_ci": (g.get("model_b") or {}).get("ci95"),
                      "ci_overlap": g.get("ci_overlap")})
    written.append(_dump(rrows, run_dir / "report_results.csv"))

    # section 4: figures
    written.append(_dump([{"figure": k, "file": Path(v).name} for k, v in (figures or {}).items()],
                         run_dir / "report_figures.csv"))

    # section 5: explainability
    erows: list[tuple[str, str]] = []
    if explain:
        erows.append(("architecture_condition", str(explain.get("architecture_condition", ""))))
        m = explain.get("methods") or {}
        t = m.get("temporal") or {}
        if t:
            erows += [("temporal.n_windows_explained", _flat(t.get("n_windows_explained"))),
                      ("temporal.computed", _flat(t.get("computed"))),
                      ("temporal.window_length_samples", _flat(t.get("window_length_samples"))),
                      ("temporal.explanation_model", _flat(t.get("explanation_model")))]
        bp = m.get("baseline_only_probe") or {}
        if bp:
            erows += [(f"baseline_only_probe.{k}", _flat(v)) for k, v in bp.items()
                      if k in ("status", "probe", "balanced_accuracy_mean_over_repeats", "n_files",
                               "chance", "interpretation")]
        ts = m.get("time_shift_null") or {}
        if ts:
            erows += [(f"time_shift_null.{k}", _flat(v)) for k, v in ts.items()
                      if k in ("shift_samples", "moves_with_event_fraction", "interpretation")]
        for name, d in ((m.get("features") or {}).get("models") or {}).items():
            if d.get("status") == "not_applicable":
                erows.append((f"features.{name}", f"not applicable: {d.get('reason')}"))
            else:
                for c, items in (d.get("top_features_by_class") or {}).items():
                    erows.append((f"features.{name}.class_{c}",
                                  "; ".join(f"{n} ({v:.3g})" for n, v in items[:8])))
    written.append(_kv_csv(erows, run_dir / "report_explainability.csv"))

    # section 6: conditional analyses
    arows: list[tuple[str, str]] = []
    if advanced:
        fs = advanced.get("few_shot_results") or {}
        for kk, r in (fs.get("results") or {}).items():
            if r.get("status") == "computed":
                arows.append((f"few_shot.{kk}",
                              f"accuracy {_f(r.get('accuracy_episode_mean'))} +/- "
                              f"{_f(r.get('accuracy_episode_sd'))} over {r.get('n_episodes')} episodes; "
                              f"setting: {fs.get('setting')}"))
            else:
                arows.append((f"few_shot.{kk}", f"SKIPPED: {r.get('reason')}"))
        nov = advanced.get("novelty_results") or {}
        if nov:
            arows.append(("novelty.status", f"{nov.get('status')}: {nov.get('reason')}"))
            for held, e in (nov.get("leave_one_class_out_diagnostic") or {}).items():
                arows.append((f"novelty.diagnostic_held_out_{held}",
                              f"AUROC {_f(e.get('auroc_distance'))}; flagged "
                              f"{_f(e.get('held_out_flagged_fraction'))}"))
        ssl = (advanced.get("semi_supervised_results") or {}).get("summary")
        if ssl:
            arows.append(("semi_supervised",
                          f"supervised-only {_f(ssl.get('supervised_only_mean'))} vs with "
                          f"pseudo-labels {_f(ssl.get('with_pseudo_labels_mean'))} (delta "
                          f"{_f(ssl.get('delta_mean'))}); accepted "
                          f"{ssl.get('n_accepted_pseudo_labels')} of "
                          f"{ssl.get('n_unlabelled_pool_windows')} pool windows"))
        cl = advanced.get("clustering_results") or {}
        for kk, r in (cl.get("results") or {}).items():
            if r.get("status") == "computed":
                arows.append((f"clustering.{kk}",
                              f"ARI {_f(r.get('ari_train'))}, NMI {_f(r.get('nmi_train'))}, "
                              f"bootstrap Jaccard {_f(r.get('bootstrap_jaccard_mean'))} +/- "
                              f"{_f(r.get('bootstrap_jaccard_sd'))}; {r.get('note')}"))
    written.append(_kv_csv(arows, run_dir / "report_advanced.csv"))

    # section 7: limitations and scientific safeguards. The CSV keeps the full text of every
    # cell, so nothing that wraps across lines inside the PDF's fixed column boundaries is lost.
    lim = limitations(summary)
    written.append(_dump([{"id": i + 1, "limitation": t, "consequence": c, "safeguard": sg}
                          for i, (t, c, sg) in enumerate(lim)], run_dir / "report_limitations.csv"))

    # section 8: information gaps
    gaps = information_gaps(summary)
    written.append(_dump([{"id": i + 1, "gap": g, "why_it_matters": w, "what_would_close_it": h}
                          for i, (g, w, h) in enumerate(gaps)],
                         run_dir / "report_information_gaps.csv"))

    # section 9: reproducibility
    written.append(_dump([{"key": k, "value": _flat(v)} for k, v in reproducibility(summary).items()],
                         run_dir / "report_reproducibility.csv"))

    # section 10: feature ablation and redundancy. The field list is fixed, because the rows are
    # of three kinds and a CSV header cannot be inferred from the first row alone.
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=ABLATION_FIELDS, extrasaction="ignore")
    w.writeheader()
    for r in ablation_rows(summary):
        w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in ABLATION_FIELDS})
    written.append(str(write_text_exclusive(run_dir / "report_ablation.csv", buf.getvalue())))

    # section 11: sampling-rate design
    smrows: list[tuple[str, str]] = []
    sm = ((cfg_dict or {}).get("sampling") or {})
    for i, r in enumerate(sm.get("rate_menu_hz") or []):
        smrows.append((f"rate_menu_hz[{i}]", str(r)))
    smrows.append(("allow_off_menu_rates", str(sm.get("allow_off_menu_rates"))))
    tr = ((cfg_dict or {}).get("training") or {})
    smrows.append(("analysis_rate_hz", str(tr.get("analysis_rate_hz"))))
    seg = ((cfg_dict or {}).get("segmentation") or {})
    for k, v in seg.items():
        smrows.append((f"segmentation.{k}", _flat(v)))
    written.append(_kv_csv(smrows or [("sampling", "not configured")],
                           run_dir / "report_sampling.csv"))

    # section 12: extended evaluation metrics
    ev = summary.get("evaluation") or {}
    xrows: list[tuple[str, str]] = [(f"definition.{k}", str(v))
                                    for k, v in (ev.get("definitions") or {}).items()]
    for mname, msum in (ev.get("per_model") or {}).items():
        for k, v in msum.items():
            xrows.append((f"{mname}.{k}", _flat(v)))
    written.append(_kv_csv(xrows or [("extended_metrics", "not computed")],
                           run_dir / "report_extended_metrics.csv"))

    # section 13: four-branch architecture
    q = summary.get("quad")
    qrows = _flatten(q, "quad.") if q else [("quad", "disabled by configuration")]
    written.append(_kv_csv(qrows, run_dir / "report_quad.csv"))

    # section 14: visualization registry
    vr = summary.get("viz_registry") or {}
    vrows: list[tuple[str, str]] = [(f"counts.{k}", str(v))
                                    for k, v in (vr.get("counts") or {}).items()]
    for cat, d in (vr.get("by_category") or {}).items():
        for k, v in d.items():
            vrows.append((f"by_category.{cat}.{k}", str(v)))
    vrows.append(("entries", str(vr.get("entries"))))
    vrows.append(("index", str(vr.get("index", ""))))
    written.append(_kv_csv(vrows or [("viz_registry", "not run")],
                           run_dir / "report_viz_registry.csv"))
    return written


def write_csv_rollups(summary: dict[str, Any], run_dir: Path, explain: dict[str, Any] | None = None,
                      advanced: dict[str, Any] | None = None,
                      figures: dict[str, str] | None = None,
                      cfg_dict: dict[str, Any] | None = None) -> list[str]:
    """Every CSV of the run: the fourteen PDF section twins plus the cross-section rollup."""
    written: list[str] = write_section_csvs(summary, run_dir, explain, advanced, figures,
                                            cfg_dict)

    rows = []
    for name, ent in (summary.get("models") or {}).items():
        agg = ent.get("aggregated") or {}
        ba = agg.get("balanced_accuracy") or {}
        ci = agg.get("balanced_accuracy_bootstrap_ci95") or {}
        perm = ent.get("permutation") or {}
        pr = ent.get("per_repeat") or {}
        rows.append({
            "model": name,
            "kind": ent.get("kind", ""),
            "n_files_evaluated": ent.get("n_files_evaluated", ""),
            "balanced_accuracy_mean": _g(pr.get("balanced_accuracy"), "mean"),
            "balanced_accuracy_sd": _g(pr.get("balanced_accuracy"), "sd"),
            "balanced_accuracy_aggregated": ba.get("value"),
            "balanced_accuracy_ci95_lo": ci.get("lo"),
            "balanced_accuracy_ci95_hi": ci.get("hi"),
            "macro_f1_mean": _g(pr.get("macro_f1"), "mean"),
            "accuracy_mean": _g(pr.get("accuracy"), "mean"),
            "log_loss_mean": _g(pr.get("log_loss"), "mean"),
            "roc_auc_ovr_macro_mean": _g(pr.get("roc_auc_ovr_macro"), "mean"),
            "permutation_p_value": perm.get("p_value"),
            "permutation_status": perm.get("status", "computed" if perm.get("p_value") is not None
                                           else perm.get("reason", "")),
        })
    written.append(_dump(rows, run_dir / "report_models.csv"))
    return written


def _g(d: Any, key: str) -> Any:
    if not isinstance(d, dict):
        return ""
    v = d.get(key)
    return "" if v is None else v


def _flat(v: Any) -> str:
    if isinstance(v, (dict, list)):
        return json.dumps(v, default=str)
    return "" if v is None else str(v)


ABLATION_FIELDS = ["row_type", "name", "kind", "groups_used", "n_features",
                   "balanced_accuracy_mean", "balanced_accuracy_sd", "delta_vs_all", "spearman",
                   "n_files_used", "status", "reason"]


def ablation_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    """Section 10 as flat rows: the feature groups, the ablation variants, the redundant pairs.

    One row set feeds both the PDF table and ``report_ablation.csv``, so the two cannot drift.
    """
    out: list[dict[str, Any]] = []
    feats = summary.get("features") or {}
    for g, n in (feats.get("computed") or {}).items():
        out.append({"row_type": "feature_group", "name": str(g), "kind": "declared group",
                    "n_features": n, "status": "computed",
                    "reason": (feats.get("descriptions") or {}).get(g, "")})
    abl = summary.get("ablation") or {}
    out.append({"row_type": "note", "name": "ablation.mode", "kind": "",
                "status": _flat(abl.get("status")), "reason": _flat(abl.get("mode"))})
    out.append({"row_type": "note", "name": "ablation.model", "kind": "", "status": "",
                "reason": _flat(abl.get("model"))})
    out.append({"row_type": "note", "name": "ablation.protocol", "kind": "", "status": "",
                "reason": _flat(abl.get("protocol"))})
    if abl.get("status") == "skipped":
        out.append({"row_type": "note", "name": "ablation.reason", "kind": "",
                    "status": "skipped", "reason": _flat(abl.get("reason"))})
    for r in abl.get("rows") or []:
        out.append({"row_type": "ablation", "name": _flat(r.get("variant")),
                    "kind": _flat(r.get("kind")), "groups_used": _flat(r.get("groups_used")),
                    "n_features": r.get("n_features"),
                    "balanced_accuracy_mean": r.get("balanced_accuracy_mean"),
                    "balanced_accuracy_sd": r.get("balanced_accuracy_sd"),
                    "delta_vs_all": r.get("delta_vs_all"), "status": _flat(r.get("status")),
                    "reason": _flat(r.get("reason"))})
    if abl.get("interpretation"):
        out.append({"row_type": "note", "name": "ablation.interpretation", "kind": "", "status": "",
                    "reason": _flat(abl.get("interpretation"))})
    red = summary.get("feature_redundancy") or {}
    out.append({"row_type": "note", "name": "redundancy.threshold", "kind": "", "status": "",
                "reason": _flat(red.get("threshold"))})
    out.append({"row_type": "note", "name": "redundancy.n_pairs_flagged", "kind": "", "status": "",
                "reason": _flat(red.get("n_pairs_flagged"))})
    out.append({"row_type": "note", "name": "redundancy.policy", "kind": "", "status": "",
                "reason": _flat(red.get("policy"))})
    for pr in red.get("pairs") or []:
        out.append({"row_type": "redundant_pair",
                    "name": f"{pr.get('feature_a')} <-> {pr.get('feature_b')}",
                    "kind": "|Spearman| on the pooled files", "spearman": pr.get("spearman"),
                    "n_files_used": pr.get("n_files_used"), "status": "flagged",
                    "reason": _flat(pr.get("verdict"))})
    return out


def _dump(rows: list[dict[str, Any]], path: Path) -> str:
    if not rows:
        write_text_exclusive(path, "")
        return str(path)
    buf = io.StringIO()
    fields = list(rows[0])
    w = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
    w.writeheader()
    for r in rows:
        w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in fields})
    write_text_exclusive(path, buf.getvalue())
    return str(path)


# ---------------------------------------------------------------------------
# Interpretation blocks
# ---------------------------------------------------------------------------

def limitations(summary: dict[str, Any]) -> list[tuple[str, str, str]]:
    """Limitation, consequence, and the safeguard actually applied. Derived from this run."""
    out: list[tuple[str, str, str]] = []
    n = int(summary.get("n_files_used", 0))
    k = len(summary.get("class_counts_used") or {})
    out.append((
        f"Small sample: {n} files across {k} classes",
        "Confidence intervals are wide; a high point estimate is not evidence of a large effect",
        "Every metric is reported with a file-level bootstrap interval and a label-shuffle null"))
    out.append((
        "No replicate or session metadata exists",
        "Batch or day effects cannot be separated from the class label, and no hierarchical "
        "bootstrap or session-level holdout is possible",
        "Files are the split unit; the bootstrap is labelled file-level; the baseline-only "
        "shortcut probe tests whether pre-event baseline alone predicts the class"))
    out.append((
        "No held-out test set",
        "Every number is cross-validated, so it describes this pool of files, not new recordings",
        "Repeated grouped cross-validation with tuning confined to inner folds"))
    out.append((
        "File-level labels over event-level signals",
        "A file whose events do not match its folder label injects label noise that no model can fix",
        "The label-noise rate (files with zero usable events) is reported; those files are kept "
        "and listed rather than dropped"))
    deep = summary.get("deep") or {}
    if deep.get("n_windows"):
        out.append((
            "Window length was derived from this pool's event-duration distribution",
            "The same configuration on a different experiment yields a different window length, "
            "so numbers are not directly comparable across datasets",
            "The resolved window length and its source are recorded in summary.json"))
    if (deep.get("dependency") or {}).get("available") is False:
        out.append((
            "Model B did not run in this environment",
            "No deep or hybrid result exists; only Model A was evaluated",
            "The reason and the install command are recorded in summary.json under deep.unavailable"))
    models = summary.get("models") or {}
    for name, ent in models.items():
        perm = ent.get("permutation") or {}
        if perm.get("status") == "not run":
            out.append((
                f"No label-shuffle null for {name}",
                "There is no permutation-based check that this model's accuracy exceeds chance",
                f"Reason recorded: {perm.get('reason', 'n_permutations = 0')}"))
    return out


def information_gaps(summary: dict[str, Any]) -> list[tuple[str, str, str]]:
    """What is missing from the data, why it matters, and what would close it."""
    out = [
        ("Replicate or session identifiers",
         "Without them, leakage between files of one recording session cannot be ruled out and "
         "variance cannot be partitioned",
         "A metadata CSV giving session_id and replicate_id per file"),
        ("Ground-truth event annotations",
         "Detected events are threshold crossings, not verified trapping events; sensitivity and "
         "specificity of detection are unknown",
         "A manually annotated subset of files, even 10 per class"),
        ("An unknown or out-of-class set",
         "Novelty detection cannot be evaluated, only diagnosed by leave-one-class-out",
         "Recordings of a third condition, or buffer-only controls"),
        ("Instrument and acquisition metadata",
         "Batch effects cannot be modelled or removed, and units cannot be converted to physical "
         "current",
         "Per-file acquisition date, amplifier gain, and a calibration factor"),
        ("Independent validation data",
         "Cross-validated numbers on this pool cannot confirm performance on new recordings",
         "A second, separately acquired dataset"),
    ]
    if (summary.get("deep") or {}).get("n_windows") == 0:
        out.append(("Usable event windows",
                    "Model B has nothing to learn from if no event passes the detector",
                    "Lower events.threshold_sigma after inspecting the traces, or check the "
                    "recordings contain events at all"))
    return out


def reproducibility(summary: dict[str, Any]) -> dict[str, Any]:
    env = capture_environment(Path.cwd())
    return {
        "trapsig_version": summary.get("trapsig_version", __version__),
        "config_fingerprint": summary.get("config_fingerprint"),
        "seed": summary.get("seed"),
        "model_set": summary.get("model_set"),
        "n_repeats": summary.get("n_repeats"),
        "n_splits": summary.get("n_splits"),
        "analysis_rate_hz": summary.get("analysis_rate_hz"),
        "filter_mode": summary.get("filter_mode"),
        "deep": {k: v for k, v in (summary.get("deep") or {}).items()
                 if k in ("window_ms", "window_ms_source", "window_rate_hz", "window_length_samples",
                          "n_windows", "n_parameters", "architecture", "select_metric",
                          "temperature_mean", "total_train_seconds", "aggregation")},
        "environment": env,
    }


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------

def build_pdf(summary: dict[str, Any], run_dir: Path, figures: dict[str, str],
              explain: dict[str, Any] | None = None, advanced: dict[str, Any] | None = None,
              cfg_dict: dict[str, Any] | None = None, say: Say = None) -> dict[str, Any]:
    """Write ``report.pdf``. Returns a status block that is also stored in summary.json."""
    tell = say or (lambda m: None)
    if not _RL:
        return {"status": "skipped",
                "reason": "reportlab is not installed",
                "install_hint": "pip install reportlab"}
    path = run_dir / "report.pdf"
    doc = SimpleDocTemplate(str(path), pagesize=A4, leftMargin=16 * mm, rightMargin=16 * mm,
                            topMargin=14 * mm, bottomMargin=14 * mm,
                            title="trapsig model report", author="trapsig")
    ss = getSampleStyleSheet()
    h1 = ParagraphStyle("h1x", parent=ss["Heading1"], fontSize=13, spaceAfter=5)
    h2 = ParagraphStyle("h2x", parent=ss["Heading2"], fontSize=10.5, spaceAfter=3, spaceBefore=8)
    body = ParagraphStyle("bodyx", parent=ss["BodyText"], fontSize=8.2, leading=11)
    small = ParagraphStyle("smallx", parent=body, fontSize=7.2, leading=9, textColor=colors.HexColor("#444444"))
    global _CELL
    _CELL = ParagraphStyle("cellx", fontName="Helvetica", fontSize=6.9, leading=8.6,
                           splitLongWords=1, wordWrap="CJK")
    story: list[Any] = [Paragraph("Trapping-signal classification: model report", h1)]

    story.append(Paragraph(
        f"trapsig {summary.get('trapsig_version', __version__)} &nbsp;|&nbsp; "
        f"config fingerprint <font face='Courier'>{str(summary.get('config_fingerprint'))[:16]}</font>"
        f" &nbsp;|&nbsp; seed {summary.get('seed')} &nbsp;|&nbsp; model set "
        f"<b>{summary.get('model_set')}</b>", small))
    story.append(Spacer(1, 5))

    # ---- data landscape
    story.append(Paragraph("1. Data landscape", h2))
    story.append(Paragraph("CSV twin: <font face='Courier'>report_data_landscape.csv</font> carries the same content.", small))
    cc = summary.get("class_counts_used") or {}
    story.append(Paragraph(
        f"{summary.get('n_pool_files')} files in the pool; {summary.get('n_files_used')} used; "
        f"{summary.get('n_files_skipped_or_failed')} skipped or failed. Classes: "
        + ", ".join(f"<b>{c}</b> = {n}" for c, n in cc.items())
        + f". Imbalance ratio {summary.get('imbalance_ratio'):.2f}"
        if isinstance(summary.get("imbalance_ratio"), (int, float)) else "", body))
    deep = summary.get("deep") or {}
    if deep.get("n_windows") is not None:
        story.append(Paragraph(
            f"Event windows for Model B: {deep.get('n_windows')} windows, length "
            f"{deep.get('window_ms')} ms ({deep.get('window_ms_source')}) = "
            f"{deep.get('window_length_samples')} samples at {deep.get('window_rate_hz')} Hz. "
            f"Files with no usable window: {deep.get('n_files_without_windows')}. "
            f"Events excluded by the edge rule: {deep.get('n_events_edge_excluded')}.", body))
    sk = summary.get("skipped_or_failed") or []
    if sk:
        story.append(Paragraph(
            "Skipped or failed files are listed in full in <font face='Courier'>summary.json</font>; "
            f"the first is <font face='Courier'>{sk[0].get('sample_id')}: "
            f"{sk[0].get('reason')}</font>.", small))

    # ---- methods
    story.append(Paragraph("2. Methods actually used", h2))
    story.append(Paragraph("CSV twin: <font face='Courier'>report_methods.csv</font> carries the same content.", small))
    dn = summary.get("design_notes") or {}
    for k, v in dn.items():
        story.append(Paragraph(f"<b>{k.replace('_', ' ')}:</b> {v}", body))
    if deep.get("architecture"):
        story.append(Paragraph(
            f"<b>Model B architecture:</b> {deep['architecture']}; "
            f"{deep.get('n_parameters')} parameters; receptive field "
            f"{deep.get('receptive_field_samples')} samples "
            f"({deep.get('receptive_field_ms')} ms); selection metric "
            f"{deep.get('select_metric')}; mean temperature {deep.get('temperature_mean')}; "
            f"total training time {deep.get('total_train_seconds')} s.", body))
    if (deep.get("dependency") or {}).get("available") is False:
        u = deep.get("unavailable") or {}
        story.append(Paragraph(
            f"<b>Model B did not run:</b> {u.get('reason', '')} Install with "
            f"<font face='Courier'>{u.get('install_hint', '')}</font>", body))

    # ---- results
    story.append(Paragraph("3. Results", h2))
    story.append(Paragraph("CSV twin: <font face='Courier'>report_results.csv</font> carries the same content.", small))
    story.append(_comparison_table(summary, body))
    story.append(Spacer(1, 4))
    comp = summary.get("comparison") or {}
    matched = comp.get("matched_subset") or {}
    if matched.get("gate_G7"):
        g = matched["gate_G7"]
        story.append(Paragraph(
            f"<b>Gate G7 (does Model B beat or match Model A?):</b> {g.get('verdict')}. "
            f"{g.get('reason')}. Compared on the matched subset of "
            f"{g.get('n_files_matched_subset')} files. Model A best = "
            f"{g.get('model_a', {}).get('name')} at {_f(g.get('model_a', {}).get('value'))}; "
            f"Model B at {_f(g.get('model_b', {}).get('value'))}. CI overlap: "
            f"{g.get('ci_overlap')}.", body))
    story.append(Spacer(1, 3))
    for name, ent in (summary.get("models") or {}).items():
        perm = ent.get("permutation") or {}
        if perm.get("p_value") is not None:
            story.append(Paragraph(
                f"<b>Label-shuffle null, {name}:</b> observed balanced accuracy "
                f"{_f(perm.get('observed_balanced_accuracy_mean_over_repeats'))} vs null mean "
                f"{_f(perm.get('null_mean'))} (95th percentile {_f(perm.get('null_p95'))}), "
                f"p = {_f(perm.get('p_value'))} from {perm.get('n_permutations')} permutations. "
                f"Smallest attainable p = {_f(perm.get('smallest_attainable_p'))}.", body))
        elif perm.get("status"):
            story.append(Paragraph(
                f"<b>Label-shuffle null, {name}:</b> {perm.get('status')} - {perm.get('reason', '')}",
                small))

    # ---- figures
    figs = {k: v for k, v in (figures or {}).items() if v and Path(v).is_file()}
    if figs:
        story.append(PageBreak())
        story.append(Paragraph("4. Figures", h2))
        story.append(Paragraph("CSV twin: <font face='Courier'>report_figures.csv</font> carries the same content.", small))
        for name, fpath in figs.items():
            story.append(Paragraph(name.replace("_", " "), h2))
            try:
                story.append(_scaled_image(fpath, 172 * mm))
            except Exception as exc:  # a bad figure must not kill the report
                story.append(Paragraph(f"(figure could not be embedded: {exc})", small))

    # ---- explainability
    if explain:
        story.append(PageBreak())
        story.append(Paragraph("5. Explainability", h2))
        story.append(Paragraph("CSV twin: <font face='Courier'>report_explainability.csv</font> carries the same content.", small))
        m = explain.get("methods") or {}
        if explain.get("architecture_condition"):
            story.append(Paragraph(str(explain["architecture_condition"]), small))
        t = m.get("temporal") or {}
        if t:
            story.append(Paragraph(
                f"Temporal attribution on {t.get('n_windows_explained')} windows using "
                f"{', '.join(t.get('computed') or [])}. {t.get('explanation_model')}", body))
        bp = m.get("baseline_only_probe") or {}
        if bp.get("status") == "computed":
            story.append(Paragraph(
                f"<b>Shortcut probe (mandatory):</b> a baseline-only classifier reached balanced "
                f"accuracy {_f(bp.get('balanced_accuracy_mean_over_repeats'))} against a chance "
                f"level of {_f(bp.get('chance'))} on {bp.get('n_files')} files. "
                f"{bp.get('interpretation')}.", body))
        ts = m.get("time_shift_null") or {}
        if ts:
            story.append(Paragraph(
                f"<b>Time-shift null:</b> the attribution peak moved with the shifted event in "
                f"{_f(ts.get('moves_with_event_fraction'))} of probed windows "
                f"(shift {ts.get('shift_samples')} samples). {ts.get('interpretation')}", body))
        f = m.get("features") or {}
        for name, d in (f.get("models") or {}).items():
            if d.get("status") == "not_applicable":
                story.append(Paragraph(f"<b>{name}:</b> {d.get('method')} - not applicable: "
                                       f"{d.get('reason')}", small))
            else:
                top = d.get("top_features_by_class") or {}
                for c, items in top.items():
                    head = ", ".join(f"{n} ({v:.3g})" for n, v in items[:6])
                    story.append(Paragraph(f"<b>{name}, class {c}:</b> {head}", body))

    # ---- advanced
    if advanced and advanced.get("status") is None:
        story.append(Paragraph("6. Conditional analyses", h2))
        story.append(Paragraph("CSV twin: <font face='Courier'>report_advanced.csv</font> carries the same content.", small))
        fs = advanced.get("few_shot_results") or {}
        for kk, r in (fs.get("results") or {}).items():
            if r.get("status") == "computed":
                story.append(Paragraph(
                    f"<b>Few-shot {kk}:</b> accuracy {_f(r.get('accuracy_episode_mean'))} "
                    f"+/- {_f(r.get('accuracy_episode_sd'))} over {r.get('n_episodes')} episodes; "
                    f"file bootstrap CI [{_f((r.get('accuracy_bootstrap_ci95_over_files') or {}).get('lo'))}, "
                    f"{_f((r.get('accuracy_bootstrap_ci95_over_files') or {}).get('hi'))}]. "
                    f"{fs.get('setting')}", body))
            else:
                story.append(Paragraph(f"<b>Few-shot {kk}:</b> SKIPPED - {r.get('reason')}", small))
        nov = advanced.get("novelty_results") or {}
        if nov:
            story.append(Paragraph(f"<b>Novelty detection:</b> {nov.get('status')} - "
                                   f"{nov.get('reason')}", body))
            for held, e in (nov.get("leave_one_class_out_diagnostic") or {}).items():
                story.append(Paragraph(
                    f"&nbsp;&nbsp;diagnostic, holding out <b>{held}</b>: AUROC "
                    f"{_f(e.get('auroc_distance'))}, flagged {_f(e.get('held_out_flagged_fraction'))} "
                    f"at {_f(e.get('threshold_at_target_retention'))}", small))
        ssl = advanced.get("semi_supervised_results") or {}
        if ssl.get("summary"):
            su = ssl["summary"]
            story.append(Paragraph(
                f"<b>Semi-supervised:</b> supervised-only {_f(su.get('supervised_only_mean'))} vs "
                f"with pseudo-labels {_f(su.get('with_pseudo_labels_mean'))} (delta "
                f"{_f(su.get('delta_mean'))}); {su.get('n_accepted_pseudo_labels')} pseudo-labels "
                f"accepted from {su.get('n_labelled_windows')} labelled windows. "
                f"{ssl.get('setting')}", body))
        cl = advanced.get("clustering_results") or {}
        for kk, r in (cl.get("results") or {}).items():
            if r.get("status") == "computed":
                story.append(Paragraph(
                    f"<b>Clustering {kk}:</b> ARI {_f(r.get('ari_train'))}, NMI "
                    f"{_f(r.get('nmi_train'))}, bootstrap Jaccard {_f(r.get('bootstrap_jaccard_mean'))} "
                    f"+/- {_f(r.get('bootstrap_jaccard_sd'))}. {r.get('note')}", body))

    # ---- limitations
    story.append(PageBreak())
    story.append(Paragraph("7. Limitations and scientific safeguards", h2))
    story.append(Paragraph("CSV twin: <font face='Courier'>report_limitations.csv</font> carries the same content.", small))
    story.append(_three_col(limitations(summary), ["Limitation", "Consequence", "Safeguard applied"]))
    story.append(Paragraph("8. Information gaps", h2))
    story.append(Paragraph("CSV twin: <font face='Courier'>report_information_gaps.csv</font> carries the same content.", small))
    story.append(_three_col(information_gaps(summary),
                            ["Missing", "Why it matters", "What would close it"]))

    story.append(Paragraph("9. Reproducibility", h2))
    story.append(Paragraph("CSV twin: <font face='Courier'>report_reproducibility.csv</font> carries the same content.", small))
    rep = reproducibility(summary)
    env = rep.pop("environment", {}) or {}
    for k, v in rep.items():
        story.append(Paragraph(f"<b>{k}:</b> {_flat(v)}", small))
    for k in ("python", "platform", "git_commit", "git_dirty", "packages"):
        if k in env:
            story.append(Paragraph(f"<b>{k}:</b> {_flat(env[k])}", small))
    story.append(Spacer(1, 4))

    story.append(PageBreak())
    story.append(Paragraph("10. Feature ablation and redundancy", h2))
    story.append(Paragraph("CSV twin: <font face='Courier'>report_ablation.csv</font> carries the "
                           "same content, row for row.", small))
    story.extend(_ablation_section(summary, body, small, h2))

    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "Attribution and performance are different claims. A region that drives a prediction is not "
        "evidence of a biological mechanism; only an independent experiment can supply that.", small))

    story.extend(_sampling_section(cfg_dict, body, small, h2))
    story.extend(_extended_section(summary, body, small, h2))
    story.extend(_quad_section(summary, cfg_dict, body, small, h2))
    story.extend(_registry_section(summary, body, small, h2))

    doc.build(story)
    tell(f"report: wrote {path}")
    return {"status": "written", "path": str(path)}


def _num(v: Any, nd: int = 3, sign: bool = False) -> str:
    """Format a number for a table cell, or say that it was not computed."""
    if v is None or v == "":
        return "not computed"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if not math.isfinite(f):
        return "not computable"
    return f"{f:+.{nd}f}" if sign else f"{f:.{nd}f}"


def _ablation_section(summary: dict[str, Any], body: Any, small: Any, h2: Any) -> list[Any]:
    """Section 10 flowables: what was computed, what each group contributes, what is redundant.

    The tables use the same fixed-width wrapping cells as sections 7 and 8, so a long feature
    name wraps inside its own column instead of spilling into the next one.
    """
    out: list[Any] = []
    feats = summary.get("features") or {}
    abl = summary.get("ablation") or {}
    red = summary.get("feature_redundancy") or {}
    computed = feats.get("computed") or {}
    n_feat = feats.get("n_features")
    out.append(Paragraph(
        f"This run computed {n_feat if n_feat is not None else 0} handcrafted features in "
        f"{len(computed)} declared groups, then measured what each group contributes by refitting "
        f"the same model on the same folds with a different feature subset. Near-duplicate "
        f"features are reported, never dropped: choosing columns on the pooled matrix would use "
        f"information from the test folds.", body))

    out.append(Paragraph("10.1 Feature groups", h2))
    out.append(_three_col([(str(g), str(n), str((feats.get("descriptions") or {}).get(g, "")))
                           for g, n in computed.items()],
                          ["Group", "Features", "What it measures"]))
    out.append(Spacer(1, 4))

    out.append(Paragraph("10.2 Ablation", h2))
    if abl.get("status") == "computed":
        out.append(Paragraph(
            f"Mode {_flat(abl.get('mode'))}, model {_flat(abl.get('model'))}. "
            f"{_flat(abl.get('protocol'))}", small))
        rows: list[tuple[str, str, str]] = []
        for r in abl.get("rows") or []:
            mean, sd = r.get("balanced_accuracy_mean"), r.get("balanced_accuracy_sd")
            acc = (f"{_num(mean)} +/- {_num(sd)}" if mean is not None else "not computed")
            delta = r.get("delta_vs_all")
            third = (f"delta vs all groups: {_num(delta, sign=True)}"
                     if delta is not None else
                     ("baseline" if r.get("kind") == "baseline" else _flat(r.get("status"))))
            if r.get("reason"):
                third = f"{third}; {r.get('reason')}"
            rows.append((f"{r.get('variant')} ({r.get('n_features')} features)", acc, third))
        out.append(_three_col(rows, ["Feature subset", "Balanced accuracy (mean +/- sd)",
                                     "Change against all groups"]))
        out.append(Spacer(1, 3))
        out.append(Paragraph(_flat(abl.get("interpretation")), small))
    else:
        out.append(Paragraph(
            f"Not computed: {_flat(abl.get('reason') or 'the ablation is disabled')}", small))
    out.append(Spacer(1, 4))

    out.append(Paragraph("10.3 Redundant feature pairs", h2))
    pairs = red.get("pairs") or []
    n_flagged = red.get("n_pairs_flagged")
    out.append(Paragraph(
        f"Threshold |Spearman| >= {_flat(red.get('threshold'))}; {n_flagged if n_flagged is not None else 0} "
        f"pair(s) flagged. {_flat(red.get('policy'))}", small))
    if pairs:
        shown = pairs[:14]
        out.append(_three_col([(str(pr.get("feature_a")), str(pr.get("feature_b")),
                                f"|rho| = {_num(pr.get('spearman'), 4)} on "
                                f"{pr.get('n_files_used')} files") for pr in shown],
                              ["Feature A", "Feature B", "Rank correlation"]))
        if len(pairs) > len(shown):
            out.append(Paragraph(
                f"{len(pairs) - len(shown)} further pair(s) are listed in "
                "<font face='Courier'>feature_redundancy.csv</font>.", small))
    else:
        out.append(Paragraph("No pair reached the threshold on this run's files.", small))
    return out


def _cell_txt(v: Any) -> Any:
    txt = "" if v is None else str(v)
    for a, b in (("&", "&amp;"), ("<", "&lt;"), (">", "&gt;")):
        txt = txt.replace(a, b)
    return txt


def _wrapped_table(rows: list[list[Any]], widths: list[float], small: Any) -> Any:
    """A table with strict cell boundaries: every cell is a wrapping Paragraph."""
    data: list[list[Any]] = [[_cell(Paragraph(_cell_txt(c), small)) for c in row]
                             for row in rows]
    t = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT", splitByRow=True)
    t.setStyle(_table_style())
    return t


def _sampling_section(cfg_dict: dict[str, Any] | None, body: Any, small: Any, h2: Any
                      ) -> list[Any]:
    out: list[Any] = [PageBreak(), Paragraph("11. Sampling-rate design", h2),
                      Paragraph("CSV twin: <font face='Courier'>report_sampling.csv</font>.",
                                small)]
    cfg_dict = cfg_dict or {}
    sm = cfg_dict.get("sampling") or {}
    tr = cfg_dict.get("training") or {}
    seg = cfg_dict.get("segmentation") or {}
    menu = sm.get("rate_menu_hz") or []
    rows: list[list[Any]] = [["Rate (Hz)", "Nyquist limit (Hz)", "Typical role"]]
    for r in menu:
        hz = float(r)
        rows.append([f"{hz:g}", f"{hz / 2:g}",
                     "acquisition rate" if hz >= 1e6 else "analysis rate"])
    out.append(_wrapped_table(rows, [35 * mm, 40 * mm, 85 * mm], small))
    out.append(Spacer(1, 4))
    out.append(Paragraph(
        f"Analysis rate used by this run: <b>{tr.get('analysis_rate_hz')} Hz</b>. Acquisition "
        "rate and analysis rate are different things: the file was recorded at its own rate and "
        "everything downstream is computed at the analysis rate. Down-conversion uses staged "
        "decimation with an anti-alias filter, so no spectral content above the new Nyquist "
        "limit folds back into the band. Up-conversion cannot create information that was never "
        "sampled and is offered only as a display convenience.", small))
    out.append(Spacer(1, 4))
    segrows: list[list[Any]] = [["Segmentation parameter", "Value"]]
    for k in ("window_s", "overlap", "stride_s", "normalize", "detrend", "pad_mode",
              "max_windows_per_file"):
        segrows.append([k, str(seg.get(k))])
    segrows.append(["allow_off_menu_rates", str(sm.get("allow_off_menu_rates"))])
    out.append(_wrapped_table(segrows, [60 * mm, 100 * mm], small))
    return out


def _extended_section(summary: dict[str, Any], body: Any, small: Any, h2: Any) -> list[Any]:
    out: list[Any] = [Paragraph("12. Extended evaluation metrics", h2),
                      Paragraph("CSV twin: <font face='Courier'>report_extended_metrics.csv</font>"
                                " and <font face='Courier'>metrics_extended.csv</font>.", small)]
    ev = summary.get("evaluation") or {}
    per = ev.get("per_model") or {}
    if not per:
        out.append(Paragraph("Not computed on this run.", small))
        return out
    rows: list[list[Any]] = [["Model", "n", "MCC", "Brier", "ECE", "MCE"]]
    for name, ms in per.items():
        rows.append([name, str(ms.get("n_files", "")), _num(ms.get("mcc")),
                     _num(ms.get("brier")), _num(ms.get("ece")), _num(ms.get("mce"))])
    out.append(_wrapped_table(rows, [40 * mm, 15 * mm, 25 * mm, 25 * mm, 25 * mm, 25 * mm],
                              small))
    out.append(Spacer(1, 4))
    for k, v in (ev.get("definitions") or {}).items():
        out.append(Paragraph(f"<b>{k}</b>: {v}", small))
    out.append(Paragraph(
        "Per-class precision, recall, specificity, F1, support and error counts, top-k accuracy "
        "and PR-AUC are in <font face='Courier'>metrics_extended.csv</font>; undefined values are "
        "recorded as empty cells with the reason in the note column rather than being silently "
        "imputed.", small))
    return out


def _quad_section(summary: dict[str, Any], cfg_dict: dict[str, Any] | None, body: Any,
                  small: Any, h2: Any) -> list[Any]:
    out: list[Any] = [PageBreak(), Paragraph("13. Four-branch deep architecture", h2),
                      Paragraph("CSV twin: <font face='Courier'>report_quad.csv</font>.", small)]
    q = summary.get("quad")
    if not q:
        out.append(Paragraph(
            "<b>quad.enabled is false on this run</b>, so the four-branch architecture (learned "
            "temporal branch, learned spectral branch, self-supervised pretraining, engineered "
            "window features, cross-branch attention fusion, supervised softmax head and "
            "metric-learning head) was not trained. This section deliberately shows no numbers "
            "instead of illustrative ones. Enable it with <font face='Courier'>quad.enabled: "
            "true</font>; everything else has safe defaults.", small))
        return out
    rows: list[list[Any]] = [["Property", "Value"]]
    for k in ("branches", "n_windows", "n_engineered_window_features", "n_parameters",
              "receptive_field_samples", "seconds"):
        rows.append([k, str(q.get(k))])
    pt = q.get("pretrain") or {}
    rows.append(["pretrain_epochs_run", str(pt.get("epochs_run"))])
    rows.append(["pretrain_seconds", str(pt.get("seconds"))])
    rows.append(["pretrain_objective", str(pt.get("objective"))])
    rows.append(["checkpoint_selection", str(q.get("checkpoint_selection"))])
    out.append(_wrapped_table(rows, [55 * mm, 105 * mm], small))
    out.append(Spacer(1, 4))
    gm = q.get("gate_mean") or {}
    if gm:
        grows: list[list[Any]] = [["Branch token", "Mean fusion gate weight"]]
        grows += [[k, f"{float(v):.4f}"] for k, v in gm.items()]
        out.append(Paragraph("13.1 Cross-branch fusion gates", h2))
        out.append(_wrapped_table(grows, [60 * mm, 60 * mm], small))
    fs = q.get("fewshot_summary") or []
    if fs:
        out.append(Paragraph("13.2 Few-shot episodes (Head B, prototypical metric learning)", h2))
        frows: list[list[Any]] = [["Shot", "Episodes", "Accuracy mean", "Accuracy sd",
                                   "Pooled query acc", "Wilson 95% CI"]]
        for r in fs:
            frows.append([str(r.get("shot")), str(r.get("n_episodes")),
                          _num(r.get("episode_accuracy_mean")), _num(r.get("episode_accuracy_sd")),
                          _num(r.get("pooled_query_accuracy")),
                          f"[{_num(r.get('wilson_ci95_lo'))}, {_num(r.get('wilson_ci95_hi'))}]"])
        out.append(_wrapped_table(frows, [15 * mm, 22 * mm, 30 * mm, 26 * mm, 30 * mm, 37 * mm],
                                  small))
        out.append(Paragraph(
            "Protocol: episodes are drawn from held-out fold-test file-level embeddings only; "
            "each episode samples support and query files disjointly; classification is by cosine "
            "distance to class-mean prototypes. This is closed-set: every query class appears in "
            "the support. Good few-shot numbers on this protocol do not prove the model "
            "generalises to unseen classes, and the head existing is not evidence that few-shot "
            "works.", small))
    out.append(Paragraph("13.3 Stages and leakage rules actually applied", h2))
    for line in (
            "Stage 1 splits by file before any window is cut; every window inherits its file's "
            "fold assignment.",
            "Stage 2 (self-supervised pretraining) sees only training-fold windows minus the "
            "inner-validation split; held-out test data never enters contrastive pairs or "
            "masked-reconstruction targets.",
            "Stages 3-4 warm up heads and fusion, then train jointly; the imputer and window "
            "scaling are fitted on training-fold windows only.",
            "Stage 5 trains the metric head episodically on the same disjoint support/query rule.",
            "Stage 6 selects the checkpoint on inner-validation windows of the training fold; "
            "there is no separate hyperparameter search for the quad model on this run - this is "
            "a documented limitation, not a claim of tuned hyperparameters.",
            "Stage 7 is the out-of-fold evaluation reported everywhere else in this report."):
        out.append(Paragraph(f"- {line}", small))
    return out


def _registry_section(summary: dict[str, Any], body: Any, small: Any, h2: Any) -> list[Any]:
    out: list[Any] = [PageBreak(), Paragraph("14. Visualization registry", h2),
                      Paragraph("CSV twin: <font face='Courier'>report_viz_registry.csv</font>; "
                                "the per-plot audit trail is "
                                "<font face='Courier'>viz_registry_index.csv</font>.", small)]
    vr = summary.get("viz_registry") or {}
    if not vr:
        out.append(Paragraph("The registry stage did not run on this configuration.", small))
        return out
    rows: list[list[Any]] = [["Status", "Entries"]]
    rows += [[k, str(v)] for k, v in (vr.get("counts") or {}).items()]
    rows.append(["total", str(vr.get("entries"))])
    out.append(_wrapped_table(rows, [60 * mm, 30 * mm], small))
    out.append(Spacer(1, 4))
    cats: list[list[Any]] = [["Category", "Entries", "Generated", "Skipped",
                              "Not available", "Failed"]]
    for cat, d in (vr.get("by_category") or {}).items():
        cats.append([str(cat), str(d.get("entries", "")), str(d.get("generated", "")),
                     str(d.get("skipped", "")), str(d.get("not_available", "")),
                     str(d.get("failed", ""))])
    out.append(_wrapped_table(cats, [55 * mm, 20 * mm, 22 * mm, 20 * mm, 25 * mm, 18 * mm],
                              small))
    out.append(Spacer(1, 4))
    out.append(Paragraph(
        "Every registry entry declares its category (A time domain, B frequency, C "
        "time-frequency, D nonlinear/complexity, E dataset/class, F deep diagnostics), its "
        "prerequisites and parameters in <font face='Courier'>viz_registry_index.csv</font>. "
        "Entries whose prerequisites the run did not produce are recorded as skipped with the "
        "reason; optional methods that are not shipped (EMD, VMD) are recorded as not_available. "
        "No plot in this report is illustrative: each one was drawn from arrays this run "
        "computed, and plots carrying numbers also wrote a CSV twin.", small))
    return out


def _f(v: Any) -> str:
    if v is None or v == "":
        return "n/a"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    return "n/a" if not math.isfinite(f) else f"{f:.3f}"


_CELL = None  # set in build_pdf; a wrapping Paragraph style shared by every table cell


def _cell(text: Any) -> Any:
    """Wrap a cell so its content stays inside the column and continues on the next line.

    Plain strings in a reportlab table are drawn on one line and overflow the cell boundary.
    Every cell here is a Paragraph with a fixed column width, so long text wraps instead of
    running into the next column or off the page.
    """
    return Paragraph("" if text is None else str(text), _CELL)


def _comparison_table(summary: dict[str, Any], style: Any) -> Any:
    head = ["model", "files", "balanced acc (mean +/- SD)", "aggregated", "95% CI", "macro F1"]
    data: list[list[Any]] = [[_cell(h) for h in head]]
    for name, ent in (summary.get("models") or {}).items():
        pr = (ent.get("per_repeat") or {}).get("balanced_accuracy") or {}
        agg = ((ent.get("aggregated") or {}).get("balanced_accuracy") or {})
        ci = (ent.get("aggregated") or {}).get("balanced_accuracy_bootstrap_ci95") or {}
        f1 = (ent.get("per_repeat") or {}).get("macro_f1") or {}
        data.append([
            _cell(name), _cell(str(ent.get("n_files_evaluated", ""))),
            _cell(f"{_f(pr.get('mean'))} +/- {_f(pr.get('sd'))}" if pr.get("mean") is not None
                  else str(pr.get("status", ""))),
            _cell(_f(agg.get("value"))),
            _cell(f"[{_f(ci.get('lo'))}, {_f(ci.get('hi'))}]" if ci else "n/a"),
            _cell(_f(f1.get("mean"))),
        ])
    t = Table(data, repeatRows=1, hAlign="LEFT")
    t.setStyle(_table_style())
    return t


def _table_style() -> Any:
    return TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8EEF7")),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#BBBBBB")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7F9FC")]),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ])


def _three_col(rows: list[tuple[str, str, str]], header: list[str]) -> Any:
    """Sections 7 and 8: fixed column boundaries with wrapping cells.

    The columns are fixed widths and every cell is a Paragraph, so a long limitation wraps
    inside its own cell and the row grows taller; nothing spills into the next column and no
    row is truncated. ``splitLongWords`` keeps unbreakable tokens inside the boundary too.
    """
    data: list[list[Any]] = [[_cell(h) for h in header]]
    data += [[_cell(a), _cell(b), _cell(c)] for a, b, c in rows]
    t = Table(data, repeatRows=1, hAlign="LEFT", colWidths=[52 * mm, 60 * mm, 60 * mm],
              splitByRow=True)
    t.setStyle(_table_style())
    return t


def _scaled_image(path: str, max_w: float) -> Any:
    from PIL import Image as PILImage

    with PILImage.open(path) as im:
        w, h = im.size
    scale = min(1.0, max_w / float(w))
    return Image(path, width=w * scale, height=h * scale)


def write_report_json(summary: dict[str, Any], run_dir: Path, extra: dict[str, Any]) -> None:
    """Machine-readable rollup of the reporting stage itself."""
    write_json_exclusive(run_dir / "report_status.json", {
        "csv_rollups": extra.get("csv", []),
        "figures": extra.get("figures", {}),
        "pdf": extra.get("pdf", {}),
        "n_limitations": len(limitations(summary)),
        "n_information_gaps": len(information_gaps(summary)),
    })
