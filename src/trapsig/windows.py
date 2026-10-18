"""Event-window extraction for Model B (design section 11.3).

A window is a fixed-length slice of the analysed signal centred on a detected
event onset, baseline-subtracted. Windows are the input to the 1D CNN.

Leakage rules enforced here:
  * Windows are cut per file. No statistic is shared across files at this stage.
  * The robust scale (median and IQR) is fitted by :func:`fit_robust_scale` on the
    **training-fold** windows only, and applied unchanged elsewhere. Nothing in this
    module looks at labels.

Files with no detected event yield no windows. They are excluded from Model B and
reported as such, rather than being silently dropped or padded with a guess.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class EventLike(Protocol):
    """Anything with an onset index and onset time. Avoids importing the full ``Event`` type."""
    onset_idx: int
    onset_s: float


@dataclass(frozen=True)
class WindowSet:
    """All windows of a run, with the per-window provenance needed to aggregate to file level."""
    x: np.ndarray                    # (n_windows, length) float32, baseline-subtracted, unscaled
    sample_id: np.ndarray            # (n_windows,) parent file, object dtype
    class_label: np.ndarray          # (n_windows,) label inherited from the parent file, object
    group_id: np.ndarray             # (n_windows,) group inherited from the parent file, object
    event_index: np.ndarray          # (n_windows,) index of the event inside its file
    onset_s: np.ndarray              # (n_windows,) event onset in seconds
    length: int                      # window length in samples
    rate_hz: float                   # sampling rate of the windows
    window_ms: float                 # resolved window length in milliseconds
    window_ms_source: str            # "auto" or "config"
    n_edge_excluded: int = 0         # events whose window would run off the recording
    per_file_counts: dict[str, int] = field(default_factory=dict)
    n_files_without_windows: int = 0
    duration_stats: dict[str, float] = field(default_factory=dict)


def resolve_window_length_ms(
    events_durations_s: list[float],
    window_ms: float | str,
    multiplier: float,
    lo_ms: float,
    hi_ms: float,
) -> tuple[float, str, dict[str, float]]:
    """Return ``(window_ms, source, duration_stats)``.

    ``"auto"`` derives the length from the event-duration distribution of the pool:
    the median duration times ``multiplier``, clamped to ``[lo_ms, hi_ms]``. The
    multiplier exists so the window carries context around the event, not just the
    event itself. Deriving it from the data is what design 11.3 asks for; a fixed
    guess would not adapt to a different experiment.
    """
    d = np.asarray([v for v in events_durations_s if np.isfinite(v) and v > 0], dtype=float)
    stats: dict[str, float] = {}
    if d.size:
        stats = {
            "n_events": float(d.size),
            "median_ms": float(np.median(d) * 1000.0),
            "p25_ms": float(np.percentile(d, 25) * 1000.0),
            "p75_ms": float(np.percentile(d, 75) * 1000.0),
            "p95_ms": float(np.percentile(d, 95) * 1000.0),
            "max_ms": float(d.max() * 1000.0),
        }
    if window_ms != "auto":
        return float(window_ms), "config", stats
    if not d.size:
        # No event durations exist; fall back to the clamp midpoint and say so.
        return float(np.sqrt(lo_ms * hi_ms)), "auto:no-events-fallback", stats
    raw = float(np.median(d) * 1000.0 * multiplier)
    return float(min(max(raw, lo_ms), hi_ms)), "auto", stats


def window_samples(window_ms: float, rate_hz: float) -> int:
    """Number of samples in a window of ``window_ms`` at ``rate_hz``."""
    return max(1, int(round(window_ms / 1000.0 * rate_hz)))


def cut_windows(
    signal: np.ndarray,
    baseline: np.ndarray,
    fs_hz: float,
    events: Sequence[EventLike],
    length: int,
    pre_fraction: float,
    max_windows: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Cut baseline-subtracted windows from one recording.

    The window starts ``pre_fraction * length`` samples before the onset, so the
    CNN sees pre-event baseline as well as the event. Events whose window would run
    past either end of the recording are excluded and counted, not padded: padding
    would inject a constant segment the network could latch onto.

    Returns ``(windows, event_indices, onsets_s, n_edge_excluded)``.
    """
    n = signal.size
    pre = int(round(pre_fraction * length))
    xs: list[np.ndarray] = []
    idxs: list[int] = []
    onsets: list[float] = []
    n_edge = 0
    resid = signal - baseline
    for i, ev in enumerate(events):
        if len(xs) >= max_windows:
            break
        start = ev.onset_idx - pre
        stop = start + length
        if start < 0 or stop > n:
            n_edge += 1
            continue
        seg = np.asarray(resid[start:stop], dtype=np.float32)
        if not np.all(np.isfinite(seg)):
            n_edge += 1
            continue
        xs.append(seg)
        idxs.append(i)
        onsets.append(ev.onset_s)
    if not xs:
        return (np.empty((0, length), dtype=np.float32), np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.float64), n_edge)
    return np.stack(xs), np.asarray(idxs, dtype=np.int64), np.asarray(onsets, dtype=np.float64), n_edge


def build_window_set(
    per_file: list[dict[str, Any]],
    length: int,
    rate_hz: float,
    window_ms: float,
    window_ms_source: str,
    duration_stats: dict[str, float],
) -> WindowSet:
    """Assemble per-file window arrays into one :class:`WindowSet`.

    ``per_file`` items need ``sample_id``, ``class``, ``group_id``, ``windows``,
    ``event_indices``, ``onsets_s``, and ``n_edge_excluded``.
    """
    xs, sids, labs, grps, eidx, ons = [], [], [], [], [], []
    n_edge = 0
    counts: dict[str, int] = {}
    n_no_win = 0
    for rec in per_file:
        w = rec["windows"]
        n_edge += int(rec.get("n_edge_excluded", 0))
        counts[rec["sample_id"]] = int(w.shape[0])
        if w.shape[0] == 0:
            n_no_win += 1
            continue
        xs.append(w)
        sids.append(np.full(w.shape[0], rec["sample_id"], dtype=object))
        labs.append(np.full(w.shape[0], rec["class"], dtype=object))
        grps.append(np.full(w.shape[0], rec["group_id"], dtype=object))
        eidx.append(rec["event_indices"])
        ons.append(rec["onsets_s"])
    if not xs:
        return WindowSet(
            x=np.empty((0, length), dtype=np.float32),
            sample_id=np.empty(0, dtype=object), class_label=np.empty(0, dtype=object),
            group_id=np.empty(0, dtype=object), event_index=np.empty(0, dtype=np.int64),
            onset_s=np.empty(0, dtype=np.float64),
            length=length, rate_hz=rate_hz, window_ms=window_ms, window_ms_source=window_ms_source,
            n_edge_excluded=n_edge, per_file_counts=counts, n_files_without_windows=n_no_win,
            duration_stats=duration_stats,
        )
    return WindowSet(
        x=np.concatenate(xs).astype(np.float32, copy=False),
        sample_id=np.concatenate(sids), class_label=np.concatenate(labs),
        group_id=np.concatenate(grps), event_index=np.concatenate(eidx),
        onset_s=np.concatenate(ons),
        length=length, rate_hz=rate_hz, window_ms=window_ms, window_ms_source=window_ms_source,
        n_edge_excluded=n_edge, per_file_counts=counts, n_files_without_windows=n_no_win,
        duration_stats=duration_stats,
    )


def fit_robust_scale(x: np.ndarray) -> tuple[float, float]:
    """Median and IQR over the pooled training windows. Fitted on training folds only."""
    if x.size == 0:
        return 0.0, 1.0
    flat = x.reshape(-1)
    med = float(np.median(flat))
    q1, q3 = np.percentile(flat, [25.0, 75.0])
    iqr = float(q3 - q1)
    if not np.isfinite(iqr) or iqr <= 1e-12:
        # Degenerate (near-constant) training windows: fall back to a std scale so the
        # division is defined. Recorded by the caller via the returned value.
        sd = float(np.std(flat))
        iqr = sd if sd > 1e-12 else 1.0
    return med, iqr


def apply_robust_scale(x: np.ndarray, med: float, iqr: float) -> np.ndarray:
    """Apply a scale fitted elsewhere. Values are clipped so one outlier cannot dominate."""
    z = (x - med) / iqr
    return np.clip(z, -8.0, 8.0).astype(np.float32, copy=False)


def aggregate_to_file(
    proba: np.ndarray,
    sample_ids: np.ndarray,
    order: list[str],
    rule: str = "mean_proba",
) -> tuple[np.ndarray, np.ndarray]:
    """Collapse window probabilities to one row per file.

    ``proba`` is (n_windows, n_classes). ``order`` is the list of file ids to emit, in
    that order. Returns ``(file_proba, file_counts)``; a file with no window gets NaN
    rows and a count of 0, so the caller can decide the fallback explicitly instead of
    inheriting a silent uniform guess.
    """
    k = proba.shape[1] if proba.ndim == 2 else 0
    out = np.full((len(order), k), np.nan, dtype=float)
    counts = np.zeros(len(order), dtype=np.int64)
    pos = {s: i for i, s in enumerate(order)}
    if proba.size == 0:
        return out, counts
    for i, sid in enumerate(sample_ids):
        j = pos.get(sid)
        if j is None:
            continue
        row = proba[i]
        if np.any(~np.isfinite(row)):
            # This window was never predicted (its fold produced no test windows, or the deep
            # branch skipped it). Leaving it out keeps the file-level vector a real average of
            # predictions instead of a number diluted by an implicit zero.
            continue
        counts[j] += 1
        if rule == "majority_vote":
            onehot = np.zeros(k, dtype=float)
            onehot[int(np.argmax(row))] = 1.0
            row = onehot
        cur = out[j]
        out[j] = row if np.any(np.isnan(cur)) else cur + row
    with np.errstate(invalid="ignore"):
        out = np.where(counts[:, None] > 0, out / np.maximum(counts, 1)[:, None], np.nan)
    return out, counts
