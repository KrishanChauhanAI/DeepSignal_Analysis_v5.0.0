"""Run directories, environment capture, and never-overwrite file writes."""

from __future__ import annotations

import datetime as _dt
import importlib.metadata as _md
import importlib.util
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

# One folder per run. The dataset prepared for that run lives INSIDE it, so a run never has a
# sibling folder that belongs to it: runs/train_<ts>/training_<ts>/{data,...}.
RUN_SUBDIRS = (
    "config", "models", "checkpoints", "logs", "metrics", "plots", "predictions",
    "explainability", "reports", "artifacts", "environment",
)

# Only what the prepared dataset actually needs; no empty parallel skeleton.
PREPARED_SUBDIRS = ("data/manifests", "data/symlinks", "artifacts")


class OverwriteError(FileExistsError):
    """Refused to overwrite an existing artifact."""


def create_run_dir(
    runs_dir: str | Path,
    kind: str = "training",
    now: _dt.datetime | None = None,
) -> Path:
    """Create ``runs/<kind>_YYYY-MM-DD_HH-MM-SS[_NNN]/`` with the standard layout.

    If the timestamped name already exists (two runs in the same second) a
    numeric suffix is appended. Existing directories are never reused.
    """
    now = now or _dt.datetime.now()
    stamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    base = Path(runs_dir) / f"{kind}_{stamp}"
    candidate = base
    n = 1
    while candidate.exists():
        n += 1
        candidate = Path(f"{base}_{n:03d}")
    candidate.mkdir(parents=True, exist_ok=False)
    for sub in RUN_SUBDIRS:
        (candidate / sub).mkdir(parents=True, exist_ok=True)
    return candidate


_STAMP = re.compile(r"^[A-Za-z-]+_(.+?)(?:_\d{3})?$")


def run_stamp(name: str | Path) -> str:
    """The timestamp part of a run folder name.

    ``train_2026-10-09_06-10-08`` -> ``2026-10-09_06-10-08``; a collision suffix (``_001``) is
    dropped so the prepared folder inside a run always carries the same stamp as its parent.
    """
    text = Path(name).name
    m = _STAMP.match(text)
    return m.group(1) if m else text


def create_prepared_dir(run_dir: str | Path) -> Path:
    """Create ``<run_dir>/training_<same stamp>/`` - the dataset folder of that run.

    This is the only place a prepared dataset is written, so ``prepare`` and ``run`` produce the
    same layout and a run folder is self-contained: everything that run used and everything it
    produced is under one directory.
    """
    run_dir = Path(run_dir)
    prepared = run_dir / f"training_{run_stamp(run_dir)}"
    if prepared.exists():
        raise OverwriteError(f"refusing to reuse an existing prepared dataset folder: {prepared}")
    prepared.mkdir(parents=True)
    for sub in PREPARED_SUBDIRS:
        (prepared / sub).mkdir(parents=True, exist_ok=True)
    return prepared


def write_text_exclusive(path: str | Path, text: str) -> Path:
    """Write text, refusing to overwrite an existing file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as fh:
            fh.write(text)
    except FileExistsError as exc:
        raise OverwriteError(f"refusing to overwrite existing file: {path}") from exc
    return path


def write_json_exclusive(path: str | Path, obj: Any) -> Path:
    return write_text_exclusive(path, json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n")


def _package_version(name: str) -> str | None:
    try:
        return _md.version(name)
    except _md.PackageNotFoundError:
        return None


def _git_state(cwd: Path) -> dict[str, Any]:
    if shutil.which("git") is None:
        return {"available": False, "reason": "git not installed"}
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True, timeout=10
        )
        if commit.returncode != 0:
            return {"available": False, "reason": "not a git repository"}
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=cwd, capture_output=True, text=True, timeout=10
        )
        return {
            "available": True,
            "commit": commit.stdout.strip(),
            "dirty": bool(dirty.stdout.strip()),
        }
    except (OSError, subprocess.SubprocessError):
        return {"available": False, "reason": "git query failed"}


def mem_total_bytes() -> int | None:
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        return None
    return None


def mem_available_bytes() -> int | None:
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        return None
    return None


def capture_environment(cwd: str | Path = ".") -> dict[str, Any]:
    """Record interpreter, OS, hardware, package versions, git state, and GPU availability."""
    packages = {
        name: _package_version(name)
        for name in ("numpy", "scipy", "scikit-learn", "pandas", "PyYAML", "matplotlib", "trapsig")
    }
    gpu: dict[str, Any] = {"nvidia_smi": shutil.which("nvidia-smi") is not None}
    gpu["torch_installed"] = importlib.util.find_spec("torch") is not None
    return {
        "timestamp_local": _dt.datetime.now().isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "memory_total_bytes": mem_total_bytes(),
        "memory_available_bytes": mem_available_bytes(),
        "packages": packages,
        "gpu": gpu,
        "git": _git_state(Path(cwd)),
    }
