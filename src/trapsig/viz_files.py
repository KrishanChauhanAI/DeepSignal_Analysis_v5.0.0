"""Per-file raw-signal overview figures (additive to the shared registry).

For each source recording, write one independent PNG under
``figures/files/<safe-class>/<sample_id>_<safe-stem>_raw.png``. The plot reads the original
source file through the configured ingestion contract, never an analysis cache. Long traces are
rendered with a min/max envelope per time bucket (not a mean or low-pass), preserving extrema
while bounding plotting memory and point count. Every input receives a row in
``figures/files/raw_file_figures_index.csv`` including transparent skip/failure reasons.
"""

from __future__ import annotations

import csv
import io
import re
from pathlib import Path
from typing import Any

import numpy as np

from . import viz
from .config import DatFormat
from .ingest import load_signal
from .runs import write_text_exclusive


def _safe_name(value: Any, fallback: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "").strip()).strip("._-")
    return s[:100] or fallback


def _minmax_envelope(x: np.ndarray, max_points: int = 120_000) -> tuple[np.ndarray, np.ndarray]:
    """Return original-sample indices and alternating min/max values per bucket.

    For short signals return all samples. For long signals this is a display-only reduction:
    each bucket's extrema are kept in temporal order, so spikes and clipping rails remain visible.
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    if x.size <= max_points:
        return np.arange(x.size, dtype=np.int64), x
    buckets = max(1, max_points // 2)
    edges = np.linspace(0, x.size, buckets + 1, dtype=np.int64)
    inds: list[int] = []
    vals: list[float] = []
    for a, b in zip(edges[:-1], edges[1:], strict=True):
        if b <= a:
            continue
        part = x[a:b]
        finite = np.flatnonzero(np.isfinite(part))
        if not finite.size:
            continue
        q = part[finite]
        lo = int(finite[int(np.argmin(q))]) + int(a)
        hi = int(finite[int(np.argmax(q))]) + int(a)
        for idx in sorted({lo, hi}):
            inds.append(idx)
            vals.append(float(x[idx]))
    return np.asarray(inds, dtype=np.int64), np.asarray(vals, dtype=float)


def _write_index(rows: list[dict[str, Any]], path: Path) -> str:
    fields = [
        "sample_id",
        "class",
        "source_file",
        "status",
        "reason",
        "figure",
        "n_samples",
        "sampling_rate_hz",
        "duration_s",
        "units",
        "rendered_points",
        "render_method",
        "n_nan",
        "n_inf",
    ]
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
    w.writeheader()
    for row in rows:
        w.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in fields})
    return str(write_text_exclusive(path, buf.getvalue()))


def write_raw_file_plots(
    metadata_rows: list[dict[str, Any]],
    manifest: dict[str, dict[str, str]],
    dat: DatFormat,
    default_fs: float,
    figures_dir: Path,
    dpi: int,
    say: Any = None,
) -> tuple[list[dict[str, Any]], str]:
    """Make a raw waveform plot for every metadata/manifest row with a readable source.

    Class directory names are derived from actual labels (never a hardcoded class list); unsafe
    path characters are replaced. Files are processed sequentially to keep memory bounded.
    """
    tell = say or (lambda _msg: None)
    root = figures_dir / "files"
    root.mkdir(parents=True, exist_ok=True)
    status_rows: list[dict[str, Any]] = []

    # The metadata table is authoritative for all discovered files. The manifest supplies paths
    # for eligible records; fallback to metadata.original_path also covers excluded/skipped rows.
    seen: set[str] = set()
    for meta in metadata_rows:
        sid = str(meta.get("sample_id") or "").strip()
        if not sid or sid in seen:
            continue
        seen.add(sid)
        label = str(meta.get("class") or "unlabeled")
        src = str((manifest.get(sid) or {}).get("original_path") or meta.get("original_path") or "")
        row: dict[str, Any] = {
            "sample_id": sid,
            "class": label,
            "source_file": Path(src).name if src else "",
            "status": "",
            "reason": "",
            "figure": "",
            "n_samples": "",
            "sampling_rate_hz": "",
            "duration_s": "",
            "units": dat.units,
            "rendered_points": "",
            "render_method": "",
            "n_nan": "",
            "n_inf": "",
        }
        if not src or not Path(src).is_file():
            row.update(status="skipped", reason="source path unavailable or no longer exists")
            status_rows.append(row)
            continue
        try:
            x, fs = load_signal(src, default_fs=float(default_fs), dat=dat)
            x = np.asarray(x, dtype=float).reshape(-1)
            if not x.size:
                row.update(status="skipped", reason="empty signal")
                status_rows.append(row)
                continue
            n_nan = int(np.isnan(x).sum())
            n_inf = int(np.isinf(x).sum())
            idx, y = _minmax_envelope(x)
            class_dir = root / _safe_name(label, "unlabeled")
            class_dir.mkdir(parents=True, exist_ok=True)
            stem = _safe_name(Path(src).stem, "recording")
            dest = class_dir / f"{_safe_name(sid, 'sample')}_{stem}_raw.png"
            t = idx.astype(float) / float(fs)
            fig, ax = viz.plt.subplots(figsize=(11, 3.6))
            ax.plot(t, y, lw=0.45, color="#0072B2", rasterized=True)
            ax.set_xlabel("time (s)")
            ax.set_ylabel(f"current ({dat.units or 'a.u.'})")
            ax.set_xlim(0.0, max(float(x.size - 1) / float(fs), 1.0 / float(fs)))
            finite = x[np.isfinite(x)]
            if finite.size:
                lo, hi = np.percentile(finite, [0.1, 99.9])
                if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                    ax.set_ylim(float(lo - 0.05 * (hi - lo)), float(hi + 0.05 * (hi - lo)))
            duration = float(x.size / float(fs))
            ax.set_title(
                f"raw recording - class {label} | {Path(src).name} | {x.size:,} samples, "
                f"{float(fs):g} Hz, {duration:.6g} s | display: min/max envelope "
                f"({y.size:,}/{x.size:,} points) | NaN={n_nan:,}, Inf={n_inf:,}",
                fontsize=9,
            )
            ax.grid(True, alpha=0.2, lw=0.4)
            fig.tight_layout()
            viz._save(fig, dest, dpi)
            row.update(
                status="generated",
                figure=str(dest),
                n_samples=int(x.size),
                sampling_rate_hz=float(fs),
                duration_s=duration,
                units=dat.units,
                rendered_points=int(y.size),
                render_method="min/max envelope; display only",
                n_nan=n_nan,
                n_inf=n_inf,
            )
            status_rows.append(row)
        except Exception as exc:
            row.update(status="failed", reason=f"{type(exc).__name__}: {exc}")
            status_rows.append(row)
        if len(status_rows) % 25 == 0:
            tell(f"raw per-file figures: {len(status_rows)}/{len(metadata_rows)}")

    index = _write_index(status_rows, root / "raw_file_figures_index.csv")
    n_ok = sum(r["status"] == "generated" for r in status_rows)
    tell(f"raw per-file figures: {n_ok}/{len(status_rows)} generated under {root}")
    return status_rows, index
