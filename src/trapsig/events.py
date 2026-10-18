"""Label-blind event detection with hysteresis, plus normalised current change.

Conventions (see ``docs/METHODS.md``):

* Current ``I`` is compared with a baseline ``I0_track`` (rolling median).
* Residual ``r = I - baseline``. Blockades give ``r < 0`` (assumption A2).
* Noise ``sigma`` is a robust MAD estimate of the residual.
* Hysteresis: an event segment opens when ``r < -threshold_sigma * sigma`` and
  extends while ``r < -exit_sigma * sigma``. Segments that never reach the
  enter threshold are rejected.
* ``I0`` for ΔI and ΔI/I0 is the median of the raw current over a pre-event
  window. If that window is too short the rolling baseline is used and the
  choice is recorded in ``i0_source``.

Outputs are called *detected events*. They are not labelled as molecular
association or dissociation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import median_filter


@dataclass(frozen=True)
class Event:
    onset_idx: int
    offset_idx: int          # exclusive
    fs: float
    onset_s: float
    offset_s: float
    duration_s: float
    i0: float                # reference current (A or the file's unit)
    i0_source: str           # "pre_event_median" | "rolling_baseline"
    delta_i_mean: float      # mean(I over event) - I0
    delta_i_over_i0: float   # delta_i_mean / I0; NaN if I0 == 0
    amplitude: float         # max |I - I0| inside event
    censored_start: bool     # event touches the first sample of the recording
    censored_end: bool       # event touches the last sample of the recording


def robust_sigma(x: np.ndarray) -> float:
    """Noise estimate ``1.4826 * median(|x - median(x)|)``; returns 0 for constant input."""
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        raise ValueError("empty input")
    med = np.median(x)
    return float(1.4826 * np.median(np.abs(x - med)))


def rolling_median_baseline(x: np.ndarray, fs: float, window_s: float) -> np.ndarray:
    """Rolling median with an odd sample window of about ``window_s`` seconds."""
    x = np.asarray(x, dtype=float)
    n = max(3, int(round(window_s * fs)))
    if n % 2 == 0:
        n += 1
    n = min(n, x.size if x.size % 2 == 1 else x.size - 1)
    if n < 1:
        raise ValueError("signal shorter than baseline window")
    return np.asarray(median_filter(x, size=n, mode="reflect"), dtype=float)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return ``[start, end)`` index pairs for contiguous True runs."""
    if not mask.any():
        return []
    padded = np.concatenate(([False], mask, [False]))
    edges = np.flatnonzero(np.diff(padded.astype(np.int8)))
    return [(int(s), int(e)) for s, e in zip(edges[::2], edges[1::2], strict=True)]


@dataclass(frozen=True)
class DetectionResult:
    events: list[Event]
    sigma: float
    n_edge_start_excluded: int   # onsets inside the start guard (filter transients)
    n_merged: int                # segments merged by minimum separation
    n_too_short: int             # segments rejected by minimum duration


def detect_events(
    current: np.ndarray,
    fs: float,
    *,
    baseline: np.ndarray,
    threshold_sigma: float = 5.0,
    exit_sigma: float = 3.0,
    min_duration_s: float = 1e-3,
    min_separation_s: float = 1e-3,
    pre_event_window_s: float = 0.1,
    sigma: float | None = None,
    edge_guard_s: float = 0.0,
) -> list[Event]:
    """Convenience wrapper around :func:`detect_events_detailed` returning only events."""
    return detect_events_detailed(
        current, fs, baseline=baseline, threshold_sigma=threshold_sigma,
        exit_sigma=exit_sigma, min_duration_s=min_duration_s,
        min_separation_s=min_separation_s, pre_event_window_s=pre_event_window_s,
        sigma=sigma, edge_guard_s=edge_guard_s,
    ).events


def detect_events_detailed(
    current: np.ndarray,
    fs: float,
    *,
    baseline: np.ndarray,
    threshold_sigma: float = 5.0,
    exit_sigma: float = 3.0,
    min_duration_s: float = 1e-3,
    min_separation_s: float = 1e-3,
    pre_event_window_s: float = 0.1,
    sigma: float | None = None,
    edge_guard_s: float = 0.0,
) -> DetectionResult:
    """Detect blockade-type events by hysteresis thresholding of the residual.

    ``edge_guard_s``: events whose onset lies within this many seconds of the
    first sample are excluded and counted. Zero-phase filtering and baseline
    estimation produce transients at the start of a recording that cross the
    threshold; without a valid pre-event baseline these cannot be told apart from
    events. The guard is applied to onsets only. An event cut off by the recording
    end is kept and flagged ``censored_end``.
    """
    current = np.asarray(current, dtype=float)
    baseline = np.asarray(baseline, dtype=float)
    if current.shape != baseline.shape:
        raise ValueError("current and baseline must have the same shape")
    if exit_sigma >= threshold_sigma:
        raise ValueError("exit_sigma must be < threshold_sigma")
    n = current.size
    if n == 0:
        return DetectionResult([], 0.0, 0, 0, 0)

    resid = current - baseline
    sig = robust_sigma(resid) if sigma is None else float(sigma)
    if sig <= 0:
        return DetectionResult([], sig, 0, 0, 0)  # no measurable noise: thresholds undefined

    enter = resid < -threshold_sigma * sig
    stay = resid < -exit_sigma * sig

    segments = [(s, e) for s, e in _runs(stay) if enter[s:e].any()]

    # Merge segments separated by less than min_separation.
    min_sep = int(round(min_separation_s * fs))
    merged: list[tuple[int, int]] = []
    n_merged = 0
    for s, e in segments:
        if merged and s - merged[-1][1] < min_sep:
            merged[-1] = (merged[-1][0], e)
            n_merged += 1
        else:
            merged.append((s, e))

    # Enforce minimum duration.
    min_len = int(round(min_duration_s * fs))
    long_enough = [(s, e) for s, e in merged if (e - s) >= max(1, min_len)]
    n_too_short = len(merged) - len(long_enough)

    # Exclude onsets inside the start guard (edge transients, see docstring).
    guard_n = int(round(edge_guard_s * fs))
    kept = [(s, e) for s, e in long_enough if s >= guard_n]
    n_edge_excluded = len(long_enough) - len(kept)

    n_pre = max(1, int(round(pre_event_window_s * fs)))
    events: list[Event] = []
    for s, e in kept:
        pre_lo = max(0, s - n_pre)
        pre = current[pre_lo:s]
        if pre.size >= max(3, n_pre // 2):
            i0 = float(np.median(pre))
            src = "pre_event_median"
        else:
            i0 = float(np.median(baseline[s:e]))
            src = "rolling_baseline"
        seg = current[s:e]
        dI = seg - i0
        # ΔI/I0 is undefined when I0 == 0; keep the event and report NaN instead of aborting the run.
        rel = float(np.mean(dI) / i0) if i0 != 0 else float("nan")
        events.append(
            Event(
                onset_idx=s,
                offset_idx=e,
                fs=fs,
                onset_s=s / fs,
                offset_s=e / fs,
                duration_s=(e - s) / fs,
                i0=i0,
                i0_source=src,
                delta_i_mean=float(np.mean(dI)),
                delta_i_over_i0=rel,
                amplitude=float(np.max(np.abs(dI))),
                censored_start=bool(s == 0),
                censored_end=bool(e == n),
            )
        )
    return DetectionResult(events, sig, n_edge_excluded, n_merged, n_too_short)


def off_intervals(events: list[Event]) -> np.ndarray:
    """Toff: time from the end of each event to the onset of the next one (seconds).

    The last event has no following onset in the recording, so there are
    ``len(events) - 1`` values.
    """
    if len(events) < 2:
        return np.empty(0)
    return np.array([
        events[i + 1].onset_s - events[i].offset_s for i in range(len(events) - 1)
    ])
