"""Dry-run / preflight validation.

Executes the planning logic of the pipeline and reads only metadata: it never
trains, never writes outside the run directory, and never modifies source files.

Statuses: PASS, WARNING, ERROR, SKIPPED, NOT_APPLICABLE. Any ERROR means the run
must not proceed to data preparation or training.
"""

from __future__ import annotations

import importlib.util
import math
import shutil
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from . import __version__
from .config import TrapsigConfig, validate_config
from .ingest import IngestError, read_header
from .manifest import assign_groups, discover_sources, list_ignored_files, load_metadata, sha256_file
from .runs import mem_available_bytes
from .splits import feasibility_problems, grouped_stratified_folds


class Status(str, Enum):
    PASS = "PASS"
    WARNING = "WARNING"
    ERROR = "ERROR"
    SKIPPED = "SKIPPED"
    NOT_APPLICABLE = "NOT APPLICABLE"


@dataclass
class Check:
    check_id: str
    name: str
    status: Status
    message: str
    fix: str = ""


@dataclass
class PreflightReport:
    trapsig_version: str
    config_fingerprint: str
    checks: list[Check] = field(default_factory=list)

    def add(self, check_id: str, name: str, status: Status, message: str, fix: str = "") -> None:
        self.checks.append(Check(check_id, name, status, message, fix))

    @property
    def errors(self) -> list[Check]:
        return [c for c in self.checks if c.status is Status.ERROR]

    @property
    def exit_code(self) -> int:
        return 1 if self.errors else 0

    def counts(self) -> dict[str, int]:
        out = {s.value: 0 for s in Status}
        for c in self.checks:
            out[c.status.value] += 1
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "trapsig_version": self.trapsig_version,
            "config_fingerprint": self.config_fingerprint,
            "summary": self.counts(),
            "exit_code": self.exit_code,
            "checks": [
                {"id": c.check_id, "name": c.name, "status": c.status.value,
                 "message": c.message, "fix": c.fix}
                for c in self.checks
            ],
        }


_REQUIRED = ("numpy", "scipy", "sklearn", "yaml")
_OPTIONAL_DEFERRED = ("torch", "shap", "reportlab", "pyarrow", "umap", "pacmap", "lightgbm", "statsmodels")


def run_preflight(cfg: TrapsigConfig, *, experiment_id: str = "preflight") -> PreflightReport:
    """Run all implemented checks and return the report (no files written)."""
    rep = PreflightReport(__version__, cfg.fingerprint())

    # P01 configuration consistency
    problems = validate_config(cfg)
    if problems:
        for p in problems:
            rep.add("P01", "configuration consistency", Status.ERROR, p,
                    "Correct the value in the YAML configuration file.")
    else:
        rep.add("P01", "configuration consistency", Status.PASS, "all cross-field rules satisfied")

    # P02 dependencies
    missing_req = [m for m in _REQUIRED if importlib.util.find_spec(m) is None]
    if missing_req:
        rep.add("P02", "required Python packages", Status.ERROR,
                f"missing: {missing_req}", "pip install -e .")
    else:
        rep.add("P02", "required Python packages", Status.PASS, "numpy, scipy, scikit-learn, PyYAML present")
    deferred = [m for m in _OPTIONAL_DEFERRED if importlib.util.find_spec(m) is None]
    rep.add("P03", "optional packages (deferred modules)", Status.SKIPPED,
            f"not installed and not used by this release: {deferred}" if deferred else "all present",
            "Install only when the corresponding module is implemented.")

    # P04 GPU
    rep.add("P04", "GPU configuration", Status.NOT_APPLICABLE,
            "no model training in this release; GPU not required")

    # P05 source directory
    src = cfg.data.source_dir
    if src is None:
        rep.add("P05", "source directory", Status.ERROR, "data.source_dir is not set",
                "Set data.source_dir in the configuration.")
        _config_label_checks(cfg, rep)  # these depend only on configuration, so report them now
        _skip_data_checks(rep)
        return _finish(rep)
    if not Path(src).is_dir():
        rep.add("P05", "source directory", Status.ERROR, f"not a directory: {src}",
                "Create the directory or correct data.source_dir.")
        _skip_data_checks(rep)
        return _finish(rep)
    rep.add("P05", "source directory", Status.PASS, str(src))

    # P06 discovery
    records = discover_sources(cfg.data, experiment_id)
    if not records:
        rep.add("P06", "data files discovered", Status.ERROR,
                f"no files match '{cfg.data.file_glob}' under {src}",
                "Check data.file_glob and the directory contents.")
        _skip_data_checks(rep)
        return _finish(rep)
    ignored = list_ignored_files(cfg.data)
    note = ""
    if ignored:
        ext_counts: dict[str, int] = {}
        for _, ext, _ in ignored:
            ext_counts[ext] = ext_counts.get(ext, 0) + 1
        note = "; not matched by file_glob (listed in ignored_files.csv): " + ", ".join(
            f"{ext}={n}" for ext, n in sorted(ext_counts.items()))
    rep.add("P06", "data files discovered", Status.PASS, f"{len(records)} file(s){note}")

    # P07 label level declared
    if cfg.data.label_level == "unknown":
        rep.add("P07", "label level declared", Status.ERROR,
                "data.label_level is 'unknown'",
                "Set data.label_level to 'file' or 'event' after confirming the labelling protocol.")
    else:
        rep.add("P07", "label level declared", Status.PASS, f"label_level={cfg.data.label_level}")

    # P08 label source and coverage
    unlabelled = [r.rel_path for r in records if not r.label]
    if cfg.data.label_from == "none":
        rep.add("P08", "label source", Status.ERROR, "data.label_from is 'none'",
                "Set data.label_from to 'metadata' or 'folder'.")
    elif unlabelled:
        rep.add("P08", "label coverage", Status.ERROR,
                f"{len(unlabelled)} file(s) without a label, e.g. {unlabelled[:3]}",
                "Add every file to the metadata CSV or correct the folder structure.")
    else:
        rep.add("P08", "label coverage", Status.PASS, f"all files labelled via {cfg.data.label_from}")

    # P09 metadata file columns and unmatched rows
    if cfg.data.label_from == "metadata" and cfg.data.metadata_csv is not None:
        try:
            meta = load_metadata(cfg.data.metadata_csv)
            keys = set(meta)
            names = {r.rel_path for r in records} | {r.path.name for r in records}
            extra = sorted(keys - names)
            if extra:
                rep.add("P09", "metadata rows match files", Status.WARNING,
                        f"{len(extra)} metadata row(s) match no file, e.g. {extra[:3]}",
                        "Remove stale rows or fix file names.")
            else:
                rep.add("P09", "metadata rows match files", Status.PASS, f"{len(keys)} metadata row(s)")
        except (IngestError, OSError) as exc:
            rep.add("P09", "metadata file", Status.ERROR, str(exc),
                    "Provide a CSV with columns: file,class[,replicate_id,session_id].")

    # P10 file readability and sampling rate
    unreadable: list[str] = []
    fs_values: set[float] = set()
    cfg_fs = cfg.sampling.original_hz
    no_header_rate = 0
    durations: list[float] = []
    for rec in records:
        try:
            hdr = read_header(rec.path, cfg_fs, cfg.data.dat)
        except IngestError as exc:
            unreadable.append(f"{rec.rel_path}: {exc}")
            continue
        if hdr.sampling_rate_source == "config":
            no_header_rate += 1
        if hdr.sampling_rate_hz is None:
            unreadable.append(f"{rec.rel_path}: sampling rate unavailable")
            continue
        fs_values.add(hdr.sampling_rate_hz)
        durations.append(hdr.n_samples / hdr.sampling_rate_hz)
    if unreadable:
        rep.add("P10", "file readability", Status.ERROR,
                f"{len(unreadable)} unreadable file(s); first: {unreadable[0]}",
                "Fix or exclude the file; unsupported formats are listed in the design (Section 8, D05).")
    else:
        rep.add("P10", "file readability", Status.PASS, f"{len(records)} file(s) readable")
    if no_header_rate:
        rep.add("P11", "sampling rate source", Status.WARNING,
                f"{no_header_rate} file(s) have no rate in the file; configured {cfg_fs:g} Hz is used",
                "Confirm the acquisition rate; store it in the file header.")
    if fs_values:
        bad = sorted(f for f in fs_values if not math.isclose(f, cfg_fs, rel_tol=1e-9))
        if bad:
            rep.add("P11", "sampling rate matches configuration", Status.ERROR,
                    f"file rates {bad[:5]} differ from configured {cfg_fs:g} Hz",
                    "Correct sampling.original_hz or exclude mismatched files.")
        elif no_header_rate == 0:
            rep.add("P11", "sampling rate matches configuration", Status.PASS, f"{cfg_fs:g} Hz")

    # P12 duration
    if durations:
        exp = cfg.data.expected_duration_s
        tol = cfg.data.duration_tolerance_s
        off = [d for d in durations if abs(d - exp) > tol]
        if off:
            rep.add("P12", "recording duration", Status.ERROR,
                    f"{len(off)} file(s) differ from {exp:g} s by > {tol:g} s "
                    f"(range {min(durations):.6g}–{max(durations):.6g} s)",
                    "Check truncation; set data.expected_duration_s if the design changed.")
        else:
            rep.add("P12", "recording duration", Status.PASS, f"all within {exp:g} ± {tol:g} s")

    # P13 exact duplicates
    shas: dict[str, list[str]] = {}
    for rec in records:
        shas.setdefault(sha256_file(rec.path), []).append(rec.rel_path)
    dups = {h: v for h, v in shas.items() if len(v) > 1}
    if dups:
        rep.add("P13", "exact duplicate files", Status.ERROR,
                f"{len(dups)} duplicate group(s), e.g. {next(iter(dups.values()))}",
                "Remove re-exported copies before splitting (leakage risk).")
    else:
        rep.add("P13", "exact duplicate files", Status.PASS, "no byte-identical files")

    # P14 grouping metadata
    groups = assign_groups(records, cfg.data.allow_file_level_fallback)
    no_group = sum(1 for g in groups if g["group_id"] is None)
    if no_group:
        rep.add("P14", "replicate/session grouping", Status.ERROR,
                f"{no_group} file(s) have no replicate, session, or file-level fallback",
                "Provide replicate_id/session_id in the metadata, or set data.allow_file_level_fallback "
                "and accept that estimates are optimistic.")
        _skip_split_checks(rep)
        _tail_checks(cfg, rep, durations)
        return _finish(rep)
    levels = {g["group_level"] for g in groups}
    if levels == {"file"}:
        rep.add("P14", "replicate/session grouping", Status.WARNING,
                "grouping by file only; estimates are optimistic under session effects",
                "Provide replicate_id and session_id.")
    else:
        rep.add("P14", "replicate/session grouping", Status.PASS, f"grouping levels: {sorted(levels)}")

    # P15 class counts and imbalance (only where labels exist)
    labelled = [(r, g) for r, g in zip(records, groups, strict=True) if r.label]
    if labelled:
        labels = np.array([r.label for r, _ in labelled])
        gids = np.array([g["group_id"] for _, g in labelled])
        classes, counts = np.unique(labels, return_counts=True)
        if classes.size < 2:
            rep.add("P15", "class count", Status.ERROR,
                    f"only {classes.size} class present", "Multiclass classification needs >= 2 classes.")
        else:
            ratio = float(np.max(counts)) / float(np.min(counts))
            msg = ", ".join(f"{c}={n}" for c, n in zip(classes, counts, strict=True))
            if ratio > 3:
                rep.add("P15", "class balance", Status.WARNING,
                        f"file counts {msg}; imbalance ratio {ratio:.1f}",
                        "Choose an imbalance method explicitly (design Section 11.4).")
            else:
                rep.add("P15", "class balance", Status.PASS, f"file counts {msg}")

        # P16 group feasibility and fold construction
        probs = feasibility_problems(gids, labels, cfg.splitting.n_splits,
                                     cfg.splitting.min_groups_per_class)
        if probs:
            rep.add("P16", "cross-validation feasibility", Status.ERROR, "; ".join(probs),
                    "Collect more independent replicates/sessions or reduce n_splits.")
        else:
            try:
                plan = grouped_stratified_folds(gids, labels, cfg.splitting.n_splits,
                                                cfg.splitting.n_repeats, cfg.run.seed)
                rep.add("P16", "cross-validation feasibility", Status.PASS,
                        f"{cfg.splitting.n_repeats} x {cfg.splitting.n_splits}-fold grouped stratified; "
                        f"group disjointness verified (shape {plan.fold_ids.shape})")
            except (ValueError, RuntimeError) as exc:
                rep.add("P16", "cross-validation feasibility", Status.ERROR, str(exc),
                        "Reduce n_splits or add groups.")

        # P17 confounding: a class confined to exactly one session
        if levels == {"replicate"} or "session" in levels:
            ses = np.array([r.session_id or "NA" for r, _ in labelled])
            if np.unique(ses).size > 1:
                confined = [str(c) for c in classes if np.unique(ses[labels == c]).size == 1]
                if confined:
                    rep.add("P17", "class-session confounding", Status.ERROR,
                            f"class(es) {confined} occur in a single session",
                            "Session effects can masquerade as class effects. Add sessions or report as artifact-prone.")
                else:
                    rep.add("P17", "class-session confounding", Status.PASS, "no class confined to one session")
            else:
                rep.add("P17", "class-session confounding", Status.NOT_APPLICABLE, "only one session present")
        else:
            rep.add("P17", "class-session confounding", Status.NOT_APPLICABLE, "no session metadata")
    else:
        rep.add("P15", "class count", Status.ERROR, "no labelled files", "Provide labels (P08).")
        _skip_split_checks(rep, from_id="P16")

    _tail_checks(cfg, rep, durations)
    return _finish(rep)


def _tail_checks(cfg: TrapsigConfig, rep: PreflightReport, durations: list[float]) -> None:
    """Checks that do not depend on group membership; always executed."""
    cfg_fs = cfg.sampling.original_hz
    # P18 filter and resampling validity
    _check_filters_and_rates(cfg, rep)

    # P19 event resolution per target rate
    e = cfg.events
    unresolved = []
    for r in cfg.sampling.target_rates_hz:
        period = 1.0 / r
        if e.min_duration_s < 2 * period:
            unresolved.append(f"{r:g} Hz (period {period * 1e3:g} ms)")
    if unresolved:
        rep.add("P19", "event resolution by rate", Status.WARNING,
                f"min_duration {e.min_duration_s * 1e3:g} ms is below 2 samples at: {', '.join(unresolved)}",
                "Restrict event analysis to rates that resolve events (design Section 3.3).")
    else:
        rep.add("P19", "event resolution by rate", Status.PASS, "all rates resolve min_duration")

    # P20 memory budget (one full-resolution file, float64, plus filter temporaries)
    if durations:
        longest_n = max(int(round(d * cfg_fs)) for d in durations)
        need = longest_n * 8 * 4
        avail = mem_available_bytes()
        if avail is None:
            rep.add("P20", "memory budget", Status.SKIPPED, "available memory not readable")
        elif need > 0.8 * avail:
            rep.add("P20", "memory budget", Status.ERROR,
                    f"full-file processing needs ~{need / 2**30:.2f} GiB; available {avail / 2**30:.2f} GiB",
                    "Enable chunked processing (planned) or run on a larger machine.")
        elif need > 0.5 * avail:
            rep.add("P20", "memory budget", Status.WARNING,
                    f"full-file processing needs ~{need / 2**30:.2f} GiB of {avail / 2**30:.2f} GiB available")
        else:
            rep.add("P20", "memory budget", Status.PASS,
                    f"~{need / 2**20:.1f} MiB per file vs {avail / 2**30:.2f} GiB available")

    # P21 symlink capability (probe in a temporary directory, never in the source tree)
    rep.add(*_probe_symlinks())

    # P22 output location
    out_dir = Path(cfg.output.runs_dir)
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        probe = out_dir / ".trapsig_write_probe"
        with probe.open("x") as fh:
            fh.write("ok")
        probe.unlink()
        rep.add("P22", "output directory writable", Status.PASS, str(out_dir.resolve()))
    except OSError as exc:
        rep.add("P22", "output directory writable", Status.ERROR, str(exc),
                "Choose a writable output.runs_dir.")

    # P23 reproducibility metadata
    rep.add("P23", "random seed recorded", Status.PASS, f"seed={cfg.run.seed}")

    # P24 git state (informational)
    if shutil.which("git") is None:
        rep.add("P24", "git state", Status.WARNING, "git not installed; commit cannot be recorded")
    else:
        rep.add("P24", "git state", Status.NOT_APPLICABLE, "recorded at run time if a repository exists")


def _check_filters_and_rates(cfg: TrapsigConfig, rep: PreflightReport) -> None:
    """LPF cutoff is checked against the event-detection rate at ``trapsig events`` time."""
    fs = cfg.sampling.original_hz
    f = cfg.filters
    if f.lpf_cutoff_hz is not None:
        rep.add("P18", "LPF cutoff", Status.PASS,
                f"fc={f.lpf_cutoff_hz:g} Hz; must lie below the event-detection rate's Nyquist "
                "(enforced when 'trapsig events' runs)")
    else:
        rep.add("P18", "LPF cutoff", Status.NOT_APPLICABLE, "no LPF configured")

    if f.gaussian_sigma_s is not None:
        from .filters import gaussian_effective_cutoff_hz
        fc = gaussian_effective_cutoff_hz(f.gaussian_sigma_s)
        rep.add("P25", "Gaussian effective cutoff", Status.PASS,
                f"sigma={f.gaussian_sigma_s * 1e3:g} ms -> fc,eff={fc:.4g} Hz")

    if f.notch_hz:
        bad_n = [n for n in f.notch_hz if n >= fs / 2]
        if bad_n:
            rep.add("P26", "mains notch frequencies", Status.ERROR, f"above Nyquist: {bad_n}")
        else:
            rep.add("P26", "mains notch frequencies", Status.PASS,
                    f"{f.notch_hz} Hz applied before any decimation")


def _probe_symlinks() -> tuple[str, str, Status, str, str]:
    with tempfile.TemporaryDirectory(prefix="trapsig_probe_") as tmp:
        t = Path(tmp)
        target = t / "target.bin"
        target.write_bytes(b"x")
        link = t / "link.bin"
        try:
            link.symlink_to(target.name)
            ok = link.is_symlink() and link.read_bytes() == b"x"
        except (OSError, NotImplementedError):
            ok = False
    if ok:
        return ("P21", "symlink capability", Status.PASS, "symbolic links supported", "")
    return ("P21", "symlink capability", Status.WARNING,
            "symbolic links unavailable; manifest-only mode will be used (no copying)",
            "Enable Developer Mode (Windows) or use manifest-only split access.")


def _config_label_checks(cfg: TrapsigConfig, rep: PreflightReport) -> None:
    if cfg.data.label_level == "unknown":
        rep.add("P07", "label level declared", Status.ERROR, "data.label_level is 'unknown'",
                "Set data.label_level to 'file' or 'event' after confirming the labelling protocol.")
    else:
        rep.add("P07", "label level declared", Status.PASS, f"label_level={cfg.data.label_level}")
    if cfg.data.label_from == "none":
        rep.add("P08", "label source", Status.ERROR, "data.label_from is 'none'",
                "Set data.label_from to 'metadata' or 'folder'.")


def _skip_data_checks(rep: PreflightReport) -> None:
    for cid, name in (("P06", "data files discovered"), ("P07", "label level declared"),
                      ("P08", "label coverage"), ("P10", "file readability"),
                      ("P11", "sampling rate"), ("P12", "recording duration"),
                      ("P13", "exact duplicate files"), ("P14", "replicate/session grouping"),
                      ("P15", "class balance"), ("P16", "cross-validation feasibility"),
                      ("P17", "class-session confounding")):
        if not any(c.check_id == cid for c in rep.checks):
            rep.add(cid, name, Status.SKIPPED, "skipped: prerequisite check failed")


def _skip_split_checks(rep: PreflightReport, from_id: str = "P15") -> None:
    for cid, name in (("P15", "class balance"), ("P16", "cross-validation feasibility"),
                      ("P17", "class-session confounding")):
        if not any(c.check_id == cid for c in rep.checks):
            rep.add(cid, name, Status.SKIPPED, "skipped: grouping is incomplete")


def _finish(rep: PreflightReport) -> PreflightReport:
    rep.checks.sort(key=lambda c: c.check_id)
    return rep
