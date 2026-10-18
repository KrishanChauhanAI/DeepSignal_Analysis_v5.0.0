from __future__ import annotations

import csv

import numpy as np

from trapsig.config import DatFormat
from trapsig.viz_files import _minmax_envelope, write_raw_file_plots
from trapsig.viz_registry import REGISTRY, registry_count
from trapsig.viz_registry_ext import EXT_ENTRIES


def test_raw_plots_are_per_file_in_actual_class_directories(tmp_path):
    src = tmp_path / "source" / "Class1"
    src.mkdir(parents=True)
    x = np.sin(np.arange(8000, dtype=np.float32) / 17).astype("<f4")
    x[123] = 9.0
    path = src / "recording_01.dat"
    x.tofile(path)
    rows = [
        {"sample_id": "sid-1", "class": "Class1", "original_path": str(path)},
        {"sample_id": "sid-missing", "class": "Class2", "original_path": str(tmp_path / "gone.dat")},
    ]
    statuses, index = write_raw_file_plots(
        rows,
        {"sid-1": {"original_path": str(path)}},
        DatFormat(),
        1000.0,
        tmp_path / "figures",
        60,
    )
    assert statuses[0]["status"] == "generated"
    fig = tmp_path / "figures" / "files" / "Class1" / "sid-1_recording_01_raw.png"
    assert fig.is_file()
    assert statuses[0]["n_samples"] == len(x)
    assert statuses[0]["sampling_rate_hz"] == 1000.0
    assert statuses[1]["status"] == "skipped"
    assert (tmp_path / "figures" / "files" / "raw_file_figures_index.csv").is_file()
    assert index.endswith("raw_file_figures_index.csv")
    with open(index, newline="", encoding="utf-8") as fh:
        index_rows = list(csv.DictReader(fh))
    assert len(index_rows) == 2


def test_minmax_render_reduction_keeps_spike_and_bounds_points():
    x = np.zeros(500_000)
    x[123_456] = 100.0
    indices, values = _minmax_envelope(x, max_points=20_000)
    assert len(indices) <= 20_000
    assert len(indices) == len(values)
    assert 123_456 in indices
    assert values[np.where(indices == 123_456)[0][0]] == 100.0


def test_round3_registry_extension_is_additive_and_has_unique_ids():
    old_ids = {row["id"] for row in REGISTRY if "spec" not in row}
    ext_ids = [row["id"] for row in EXT_ENTRIES]
    assert len(EXT_ENTRIES) >= 80
    assert len(ext_ids) == len(set(ext_ids))
    assert old_ids.issubset({row["id"] for row in REGISTRY})
    assert registry_count() == len(REGISTRY)
    assert all(callable(row["b"]) for row in EXT_ENTRIES)
    assert all(isinstance(row["spec"], int) and 1 <= row["spec"] <= 168 for row in EXT_ENTRIES)
