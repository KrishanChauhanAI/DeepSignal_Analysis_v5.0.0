"""Signal-processing pipeline for one recording and event-table generation.

Order of operations (fixed, documented in docs/METHODS.md):

1. Notch filters at the **original** rate (mains removal before any decimation).
2. Anti-aliased decimation to the requested event-detection rate.
3. Mode-specific filter (LPF, Gaussian, or LPF + Gaussian), if configured.
4. Rolling-median baseline, robust noise estimate, hysteresis event detection.

Every step is per recording. No statistic is computed across files here, so
nothing from one file can leak into another.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .config import TrapsigConfig
from .events import Event, detect_events_detailed, off_intervals, robust_sigma, rolling_median_baseline
from .filters import FilterSpec
from .ingest import load_signal
from .resample import decimate_staged


@dataclass(frozen=True)
class RecordingResult:
    source: str
    fs_hz: float
    filter_label: str
    sigma: float
    events: list[Event]
    toff_s: np.ndarray
    n_edge_excluded: int
    n_too_short: int
    n_merged: int
    stats: dict[str, Any] = field(default_factory=dict)
    signal: np.ndarray | None = None      # processed signal, only when keep_signal=True
    baseline: np.ndarray | None = None


class RecordingSkipped(ValueError):
    """A recording is listed in the per-file summary but not analysed. ``reason`` says why."""

    def __init__(self, reason: str, stats: dict[str, Any]) -> None:
        super().__init__(reason)
        self.reason = reason
        self.stats = stats


def raw_stats(x: np.ndarray) -> dict[str, Any]:
    """Raw-signal statistics for the per-file assessment (computed before any filtering)."""
    finite = np.isfinite(x)
    n_bad = int(x.size - np.count_nonzero(finite))
    xf = x[finite] if n_bad else x
    out: dict[str, Any] = {"n_samples": int(x.size), "n_nonfinite": n_bad}
    if xf.size:
        out.update({"raw_min": float(xf.min()), "raw_max": float(xf.max()),
                    "raw_mean": float(xf.mean()), "raw_std": float(xf.std())})
    return out

def make_filter_specs(cfg: TrapsigConfig) -> tuple[FilterSpec, FilterSpec]:
    """Return ``(pre_decimation, post_decimation)`` specs.

    The pre-decimation spec only carries notches (applied at the original rate).
    The post-decimation spec carries the configured mode and nothing else.
    """
    f = cfg.filters
    pre = FilterSpec(mode="none", notches_hz=tuple(f.notch_hz), notch_q=f.notch_q)
    post = FilterSpec(
        mode=f.mode,
        lpf_cutoff_hz=f.lpf_cutoff_hz if f.mode in ("lpf", "lpf+gaussian") else None,
        lpf_order=f.lpf_order,
        gaussian_sigma_s=f.gaussian_sigma_s if f.mode in ("gaussian", "lpf+gaussian") else None,
        gaussian_truncate=f.gaussian_truncate,
    )
    return pre, post


def process_recording(
    path: str | Path,
    cfg: TrapsigConfig,
    rate_hz: float,
    *,
    keep_signal: bool = False,
) -> RecordingResult:
    """Run the full per-recording pipeline and return events and the filter label."""
    path = Path(path)
    x, fs0 = load_signal(path, default_fs=cfg.sampling.original_hz, dat=cfg.data.dat)
    if abs(fs0 - cfg.sampling.original_hz) > 1e-9 * fs0:
        raise ValueError(f"{path.name}: rate {fs0:g} Hz differs from configured {cfg.sampling.original_hz:g} Hz")
    stats = raw_stats(x)
    if stats["n_nonfinite"]:
        raise RecordingSkipped(f"{stats['n_nonfinite']} non-finite sample(s); not analysed", stats)

    pre, post = make_filter_specs(cfg)
    x = pre.apply(x, fs0)
    if rate_hz < fs0:
        x, fs = decimate_staged(x, fs0, rate_hz)
    else:
        fs = fs0
    y = post.apply(x, fs)

    e = cfg.events
    baseline = rolling_median_baseline(y, fs, e.baseline_window_s)
    sigma = robust_sigma(y - baseline)
    stats["sigma"] = float(sigma)
    stats["baseline_median"] = float(np.median(baseline))
    if not np.isfinite(sigma) or sigma <= 0:
        raise RecordingSkipped("noise estimate is zero or undefined (flat or invalid signal)", stats)
    # The start guard must cover the baseline window's half-length as well as the
    # configured edge guard, because the rolling median is least reliable there.
    guard = max(e.edge_guard_s, e.baseline_window_s / 2.0)
    det = detect_events_detailed(
        y,
        fs,
        baseline=baseline,
        threshold_sigma=e.threshold_sigma,
        exit_sigma=e.exit_sigma,
        min_duration_s=e.min_duration_s,
        min_separation_s=e.min_separation_s,
        pre_event_window_s=e.pre_event_window_s,
        sigma=sigma,
        edge_guard_s=guard,
    )
    return RecordingResult(
        source=str(path),
        fs_hz=fs,
        filter_label=post.describe(),
        sigma=det.sigma,
        events=det.events,
        toff_s=off_intervals(det.events),
        n_edge_excluded=det.n_edge_start_excluded,
        n_too_short=det.n_too_short,
        n_merged=det.n_merged,
        stats=stats,
        signal=y if keep_signal else None,
        baseline=baseline if keep_signal else None,
    )
