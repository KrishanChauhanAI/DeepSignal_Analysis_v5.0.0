from __future__ import annotations

import numpy as np
import pytest

from trapsig.events import detect_events, off_intervals, robust_sigma, rolling_median_baseline

FS = 10_000.0


def _trace(n: int, base: float = 1.0, noise: float = 0.01, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return base + noise * rng.standard_normal(n)


def test_robust_sigma_is_close_to_standard_deviation_for_gaussian_noise():
    x = np.random.default_rng(1).standard_normal(200_000) * 0.5
    assert robust_sigma(x) == pytest.approx(0.5, rel=0.02)


def test_robust_sigma_is_insensitive_to_a_large_outlier():
    x = np.random.default_rng(2).standard_normal(10_000)
    x[0] = 1e6
    assert robust_sigma(x) == pytest.approx(1.0, rel=0.05)


def test_detects_rectangular_blockades_with_correct_timing_and_depth():
    n = int(2 * FS)
    x = _trace(n, seed=3)
    s1, d1 = int(0.5 * FS), int(0.020 * FS)
    s2, d2 = int(1.2 * FS), int(0.030 * FS)
    x[s1:s1 + d1] *= 0.7
    x[s2:s2 + d2] *= 0.7
    base = np.full(n, 1.0)
    ev = detect_events(x, FS, baseline=base, sigma=0.01)
    assert len(ev) == 2
    assert ev[0].onset_idx == pytest.approx(s1, abs=2)
    assert ev[0].duration_s == pytest.approx(d1 / FS, abs=2 / FS)
    assert ev[0].delta_i_over_i0 == pytest.approx(-0.3, abs=0.02)
    assert ev[0].i0_source == "pre_event_median"
    assert not ev[0].censored_start and not ev[0].censored_end


def test_onset_inside_start_guard_is_excluded_and_counted():
    """Filter/baseline transients at the first samples must not become events."""
    x = _trace(int(FS), seed=4)
    x[:3] *= 0.5  # a 3-sample transient at the recording start
    from trapsig.events import detect_events_detailed
    det = detect_events_detailed(x, FS, baseline=np.full(x.size, 1.0), sigma=0.01,
                                 min_duration_s=1e-4, edge_guard_s=0.01)
    assert det.events == []
    assert det.n_edge_start_excluded == 1


def test_start_event_without_guard_is_flagged_censored():
    x = _trace(int(FS), seed=4)
    x[:int(0.01 * FS)] *= 0.6
    ev = detect_events(x, FS, baseline=np.full(x.size, 1.0), sigma=0.01, pre_event_window_s=0.001,
                       edge_guard_s=0.0)
    assert ev and ev[0].censored_start


def test_event_touching_recording_end_is_flagged_censored():
    x = _trace(int(FS), seed=5)
    x[-int(0.01 * FS):] *= 0.6
    ev = detect_events(x, FS, baseline=np.full(x.size, 1.0), sigma=0.01)
    assert ev and ev[-1].censored_end


def test_single_sample_spike_is_rejected_by_minimum_duration():
    x = _trace(int(FS), seed=6)
    x[5000] = 0.5
    ev = detect_events(x, FS, baseline=np.full(x.size, 1.0), sigma=0.01, min_duration_s=1e-3)
    assert ev == []


def test_hysteresis_keeps_an_event_open_between_thresholds():
    """Residual sits between exit (3 sigma) and enter (5 sigma): one long event, not none."""
    n = 2000
    sigma = 1.0
    r = np.zeros(n)
    r[500:520] = -6.0          # crosses enter threshold briefly
    r[520:1000] = -3.5         # stays below exit threshold but above enter threshold
    ev = detect_events(1.0 + r, FS, baseline=np.ones(n), sigma=sigma, min_duration_s=1e-4)
    assert len(ev) == 1
    assert ev[0].onset_idx == 500
    assert ev[0].offset_idx >= 1000


def test_drift_below_exit_threshold_is_not_an_event():
    n = 2000
    r = np.zeros(n)
    r[300:800] = -3.5          # never reaches enter threshold of 5 sigma
    assert detect_events(1.0 + r, FS, baseline=np.ones(n), sigma=1.0, min_duration_s=1e-4) == []


def test_close_events_are_merged_by_minimum_separation():
    n = 4000
    r = np.zeros(n)
    r[1000:1010] = -6.0
    r[1012:1020] = -6.0        # gap of 2 samples = 0.2 ms
    ev = detect_events(1.0 + r, FS, baseline=np.ones(n), sigma=1.0,
                       min_separation_s=1e-3, min_duration_s=1e-4)
    assert len(ev) == 1


def test_off_intervals_are_positive_and_one_fewer_than_events():
    n = int(2 * FS)
    x = _trace(n, seed=7)
    for s in (2000, 6000, 12000):
        x[s:s + 200] *= 0.7
    ev = detect_events(x, FS, baseline=np.full(n, 1.0), sigma=0.01)
    toff = off_intervals(ev)
    assert toff.size == len(ev) - 1
    assert np.all(toff > 0)


def test_detector_rejects_mismatched_shapes_and_bad_hysteresis():
    with pytest.raises(ValueError):
        detect_events(np.zeros(10), FS, baseline=np.zeros(9))
    with pytest.raises(ValueError):
        detect_events(np.zeros(10), FS, baseline=np.zeros(10), threshold_sigma=3.0, exit_sigma=3.0)


def test_rolling_median_baseline_tracks_slow_drift():
    t = np.arange(int(5 * FS)) / FS
    drift = 0.05 * np.sin(2 * np.pi * 0.1 * t)
    x = 1.0 + drift
    b = rolling_median_baseline(x, FS, window_s=0.5)
    core = slice(len(x) // 10, 9 * len(x) // 10)
    assert np.max(np.abs(b[core] - x[core])) < 0.01
