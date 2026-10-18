from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml


@pytest.fixture
def base_config_dict() -> dict[str, Any]:
    """A minimal, internally consistent configuration (1 MHz values are not used here)."""
    return {
        "run": {"name": "test", "seed": 1},
        "data": {"label_level": "file", "label_from": "folder"},
        "sampling": {"original_hz": 100000, "target_rates_hz": [10000, 1000, 100, 10]},
        "filters": {"mode": "none"},
        "events": {"baseline_window_s": 0.1, "pre_event_window_s": 0.02, "min_duration_s": 0.001},
        "splitting": {"n_splits": 3, "n_repeats": 2, "min_groups_per_class": 3},
        "output": {},
    }


@pytest.fixture
def write_yaml(tmp_path: Path):
    def _write(data: dict[str, Any], name: str = "cfg.yaml") -> Path:
        path = tmp_path / name
        path.write_text(yaml.safe_dump(data), encoding="utf-8")
        return path

    return _write
