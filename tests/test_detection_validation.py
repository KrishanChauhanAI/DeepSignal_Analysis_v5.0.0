"""Validate event detection against the injected ground truth of SYNTHETIC recordings.

The synthetic truth is generated alongside the data and is never read by the
pipeline itself; it is used only here to measure detector recall and false
positives. These results describe the synthetic signal model, not real data.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from trapsig.config import config_from_dict
from trapsig.pipeline import process_recording
from trapsig.synthetic import SyntheticSpec, write_synthetic_dataset


def _overlaps(onset: float, dur: float, d_onset: float, d_offset: float, tol: float = 0.005) -> bool:
    return d_onset < onset + dur + tol and d_offset > onset - tol


@pytest.fixture(scope="module")
def validated(tmp_path_factory):
    root = tmp_path_factory.mktemp("valid") / "SYN"
    meta = write_synthetic_dataset(root, SyntheticSpec(replicates_per_class=6, duration_s=0.5, seed=11))
    truth_path = root.parent / f"{root.name}_truth.csv"
    truth = list(csv.DictReader(truth_path.open(encoding="utf-8")))
    cfg = config_from_dict({
        "sampling": {"original_hz": 100000, "target_rates_hz": [10000, 1000]},
        "filters": {"mode": "lpf", "lpf_cutoff_hz": 200.0, "notch_hz": [50]},
        "events": {"baseline_window_s": 0.1, "pre_event_window_s": 0.02, "min_duration_s": 0.001},
    })
    results = {}
    for f in sorted(root.glob("*.csv")):
        results[f.name] = process_recording(f, cfg, rate_hz=1000.0)
    return {"meta": meta, "truth": truth, "results": results}


def _recall_and_fp(validated, cls: str) -> tuple[float, int]:
    truth = [t for t in validated["truth"] if t["class"] == cls]
    files = {t["file"] for t in truth}
    hits = 0
    for t in truth:
        res = validated["results"][t["file"]]
        if any(_overlaps(float(t["onset_s"]), float(t["duration_s"]), ev.onset_s, ev.offset_s)
               for ev in res.events):
            hits += 1
    fp = 0
    for name in files:
        for ev in validated["results"][name].events:
            tt = [t for t in truth if t["file"] == name]
            if not any(_overlaps(float(t["onset_s"]), float(t["duration_s"]), ev.onset_s, ev.offset_s)
                       for t in tt):
                fp += 1
    return hits / len(truth), fp


@pytest.mark.parametrize("cls", ["A", "B"])
def test_recall_on_injected_events(validated, cls):
    recall, _ = _recall_and_fp(validated, cls)
    assert recall >= 0.9, f"class {cls} recall {recall:.2f}"


@pytest.mark.parametrize("cls", ["A", "B"])
def test_no_unmatched_detections_after_edge_guard(validated, cls):
    _, fp = _recall_and_fp(validated, cls)
    assert fp == 0


def test_start_transients_were_present_and_excluded(validated):
    """Regression: without the guard these transients produced false events at onset 0."""
    excluded = sum(r.n_edge_excluded for r in validated["results"].values())
    assert excluded >= 1
    for r in validated["results"].values():
        assert all(ev.onset_s >= 0.01 for ev in r.events)


def test_amplitude_is_recovered_within_filter_distortion(validated):
    """Class B: nominal depth 0.30. The LPF can reduce peak depth; allow a 30% relative band."""
    depths = []
    for t in validated["truth"]:
        if t["class"] != "B":
            continue
        for ev in validated["results"][t["file"]].events:
            if _overlaps(float(t["onset_s"]), float(t["duration_s"]), ev.onset_s, ev.offset_s):
                depths.append(-ev.delta_i_over_i0)
                break
    assert depths, "no class B events matched"
    median_depth = sorted(depths)[len(depths) // 2]
    assert 0.3 * 0.7 <= median_depth <= 0.3 * 1.3, median_depth


def test_metadata_and_truth_files_are_separate_from_recordings(validated):
    root = Path(validated["meta"]).parent / "SYN"
    assert not any("truth" in p.name or "metadata" in p.name for p in root.iterdir())
