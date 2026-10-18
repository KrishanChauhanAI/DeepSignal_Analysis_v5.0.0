"""Signal ingestion with a format registry.

Supported formats:

* ``.npy``  one-dimensional array. The sampling rate is not in the file and must come
  from configuration; the preflight records this as a WARNING.
* ``.csv`` / ``.txt``  optional ``# key: value`` header lines (``sampling_rate_hz``,
  ``units``), then a single column of current values, or two columns
  ``time_s,current``. Two-column files must have uniform timestamps.
* ``.dat``  laboratory files described by :class:`trapsig.config.DatFormat`. Either raw
  binary samples (dtype, byte order, optional header, optional interleaved columns) or
  text rows. The layout is configuration, not guessed; :func:`probe_dat` lists the layouts
  that a file's size makes possible.

Any other extension raises :class:`IngestError`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .config import DatFormat


class IngestError(RuntimeError):
    """A file cannot be read or does not satisfy the format contract."""


@dataclass(frozen=True)
class SignalHeader:
    path: Path
    fmt: str
    n_samples: int
    sampling_rate_hz: float | None
    sampling_rate_source: str   # "header" | "config" | "timestamps" | "missing"
    units: str | None


_SUPPORTED = {".npy", ".csv", ".txt", ".dat"}


def _parse_csv_header(lines: list[str]) -> dict[str, str]:
    meta: dict[str, str] = {}
    for line in lines:
        if not line.startswith("#"):
            break
        body = line[1:].strip()
        if ":" in body:
            k, v = body.split(":", 1)
            meta[k.strip().lower()] = v.strip()
    return meta


def _np_dtype(fmt: DatFormat) -> np.dtype[Any]:
    base: np.dtype[Any] = np.dtype(fmt.dtype)
    if base.itemsize == 1:
        return base
    order: Literal["<", ">"] = "<" if fmt.byteorder == "little" else ">"
    out: np.dtype[Any] = base.newbyteorder(order)
    return out


def _split_values(line: str) -> list[str]:
    return line.replace(";", ",").replace(",", " ").split()


def _read_dat_header(path: Path, fmt: DatFormat, default_fs: float | None) -> SignalHeader:
    rate = float(default_fs) if default_fs is not None else None
    source = "config" if rate is not None else "missing"
    if fmt.encoding == "binary":
        dt = _np_dtype(fmt)
        size = path.stat().st_size
        payload = size - fmt.header_bytes
        frame = dt.itemsize * fmt.n_columns
        if payload < 0:
            raise IngestError(f"{path.name}: file is smaller than data.dat.header_bytes "
                              f"({fmt.header_bytes})")
        if payload == 0 or payload % frame != 0:
            raise IngestError(
                f"{path.name}: {payload} data bytes is not a whole number of {frame}-byte frames "
                f"(dtype={fmt.dtype}, n_columns={fmt.n_columns}, header_bytes={fmt.header_bytes}); "
                f"check data.dat or run 'trapsig inspect-dat'")
        return SignalHeader(path, "dat", payload // frame, rate, source, fmt.units)

    # text: the whole file is scanned for row count (acceptable for text, which is rarely huge)
    n = 0
    first: str | None = None
    last: str | None = None
    with path.open("r", encoding="utf-8", errors="strict") as fh:
        for i, line in enumerate(fh):
            if i < fmt.header_lines or not line.strip() or line.startswith("#"):
                continue
            n += 1
            if first is None:
                first = line
            last = line
    if n == 0 or first is None or last is None:
        raise IngestError(f"{path.name}: no data rows after {fmt.header_lines} header line(s)")
    for probe in (first, last):
        vals = _split_values(probe)
        if len(vals) <= fmt.column:
            raise IngestError(f"{path.name}: row has {len(vals)} value(s); column {fmt.column} needed")
        try:
            [float(v) for v in vals[: fmt.n_columns]]
        except ValueError as exc:
            raise IngestError(f"{path.name}: non-numeric data row {probe.strip()[:40]!r}") from exc
    return SignalHeader(path, "dat", n, rate, source, fmt.units)


def read_header(
    path: str | Path,
    default_fs: float | None = None,
    dat: DatFormat | None = None,
) -> SignalHeader:
    """Read metadata without loading the signal where the format allows it."""
    path = Path(path)
    ext = path.suffix.lower()
    if ext not in _SUPPORTED:
        raise IngestError(f"unsupported format '{ext}' for {path.name}")
    if not path.is_file():
        raise IngestError(f"not a regular file: {path}")

    if ext == ".dat":
        return _read_dat_header(path, dat or DatFormat(), default_fs)

    if ext == ".npy":
        try:
            arr = np.load(path, mmap_mode="r")
        except (ValueError, OSError) as exc:
            raise IngestError(f"cannot read {path.name}: {exc}") from exc
        if arr.ndim != 1:
            raise IngestError(f"{path.name}: expected 1-D array, got shape {arr.shape}")
        if default_fs is None:
            return SignalHeader(path, "npy", int(arr.size), None, "missing", None)
        return SignalHeader(path, "npy", int(arr.size), float(default_fs), "config", None)

    # csv / txt
    with path.open("r", encoding="utf-8", errors="strict") as fh:
        head_lines: list[str] = []
        for _ in range(64):
            pos = fh.tell()
            line = fh.readline()
            if not line:
                break
            if line.startswith("#"):
                head_lines.append(line.rstrip("\n"))
            else:
                fh.seek(pos)
                break
    meta = _parse_csv_header(head_lines)
    fs_header = meta.get("sampling_rate_hz")
    units = meta.get("units")
    n = 0
    first: str | None = None
    last: str | None = None
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            n += 1
            if first is None:
                first = line
            last = line
    if n == 0 or first is None or last is None:
        raise IngestError(f"{path.name}: no data rows")
    # Cheap content check: first and last data rows must be numeric. Full validation
    # happens in load_signal; this keeps header reads fast on very large files.
    for probe in (first, last):
        try:
            [float(v) for v in probe.replace(";", ",").split(",")[:2]]
        except ValueError as exc:
            raise IngestError(f"{path.name}: non-numeric data row {probe.strip()[:40]!r}") from exc
    if fs_header is not None:
        try:
            fs = float(fs_header)
        except ValueError as exc:
            raise IngestError(f"{path.name}: sampling_rate_hz '{fs_header}' is not a number") from exc
        return SignalHeader(path, ext[1:], n, fs, "header", units)
    if default_fs is not None:
        return SignalHeader(path, ext[1:], n, float(default_fs), "config", units)
    return SignalHeader(path, ext[1:], n, None, "missing", units)


def load_signal(
    path: str | Path,
    default_fs: float | None = None,
    dat: DatFormat | None = None,
) -> tuple[np.ndarray, float]:
    """Load the full signal and return ``(samples, sampling_rate_hz)``."""
    path = Path(path)
    hdr = read_header(path, default_fs, dat)
    if hdr.fmt == "dat":
        fmt = dat or DatFormat()
        if hdr.sampling_rate_hz is None:
            raise IngestError(f"{path.name}: sampling rate missing from configuration")
        if fmt.encoding == "binary":
            dt = _np_dtype(fmt)
            mm = np.memmap(path, dtype=dt, mode="r", offset=fmt.header_bytes,
                           shape=(hdr.n_samples, fmt.n_columns))
            x = np.asarray(mm[:, fmt.column], dtype=float) * fmt.scale
        else:
            values: list[float] = []
            with path.open("r", encoding="utf-8") as fh:
                for i, line in enumerate(fh, start=1):
                    if i <= fmt.header_lines or not line.strip() or line.startswith("#"):
                        continue
                    vals = _split_values(line)[: fmt.n_columns]
                    try:
                        values.append(float(vals[fmt.column]))
                    except (ValueError, IndexError) as exc:
                        raise IngestError(f"{path.name}: line {i} is not a numeric value in "
                                          f"column {fmt.column}: {line.strip()[:40]!r}") from exc
            x = np.asarray(values, dtype=float) * fmt.scale
        return x, hdr.sampling_rate_hz

    if hdr.fmt == "npy":
        if hdr.sampling_rate_hz is None:
            raise IngestError(f"{path.name}: sampling rate missing from file and configuration")
        x = np.asarray(np.load(path), dtype=float)
        return x, hdr.sampling_rate_hz

    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            rows.append([float(v) for v in line.replace(";", ",").split(",")[:2]])
    arr = np.asarray(rows, dtype=float)
    if arr.ndim != 2 or arr.shape[1] not in (1, 2):
        raise IngestError(f"{path.name}: expected 1 or 2 numeric columns")
    if arr.shape[1] == 1:
        if hdr.sampling_rate_hz is None:
            raise IngestError(f"{path.name}: sampling rate missing from file and configuration")
        return arr[:, 0], hdr.sampling_rate_hz
    t, x = arr[:, 0], arr[:, 1]
    dt_t = np.diff(t)
    if dt_t.size == 0 or not np.all(dt_t > 0):
        raise IngestError(f"{path.name}: timestamps must be strictly increasing")
    if (np.max(dt_t) - np.min(dt_t)) > 1e-6 * np.median(dt_t):
        raise IngestError(f"{path.name}: timestamps are not uniform")
    fs_ts = 1.0 / float(np.median(dt_t))
    if hdr.sampling_rate_hz is not None and abs(fs_ts - hdr.sampling_rate_hz) > 1e-6 * fs_ts:
        raise IngestError(f"{path.name}: timestamps imply {fs_ts:g} Hz but header says "
                          f"{hdr.sampling_rate_hz:g} Hz")
    return x, fs_ts


_PROBE_DTYPES = ("int8", "uint8", "int16", "uint16", "int32", "uint32", "float32", "float64")


def probe_dat(
    path: str | Path,
    rate_hz: float,
    duration_s: float | None = None,
    max_columns: int = 4,
    max_header_bytes: int = 4096,
) -> list[str]:
    """Describe the binary layouts that are consistent with a ``.dat`` file's size.

    If ``duration_s`` is given, the expected sample count is ``rate_hz * duration_s`` and a
    layout matches when the file holds that many frames after a header of at most
    ``max_header_bytes``.
    Without a duration, every layout whose frame size divides the file is listed with its
    sample count and zero header. Read-only: nothing is written. The output is for choosing
    ``data.dat`` settings; it does not decide them.
    """
    path = Path(path)
    if path.is_dir():
        raise IngestError(f"{path} is a directory; pass one .dat file "
                          f"(list the folder with 'ls' and copy one full file name)")
    if not path.is_file():
        raise IngestError(f"not a regular file: {path}")
    size = path.stat().st_size
    lines = [f"file: {path.name}   size: {size:,} bytes   rate: {rate_hz:g} Hz"]
    expected: int | None = None
    if duration_s is not None:
        expected = int(round(rate_hz * duration_s))
        lines.append(f"expected samples per channel for {duration_s:g} s: {expected:,}")
    matches = 0
    for name in _PROBE_DTYPES:
        base = np.dtype(name)
        orders = ["little"] if base.itemsize == 1 else ["little", "big"]
        for order in orders:
            dt = base if base.itemsize == 1 else base.newbyteorder("<" if order == "little" else ">")
            for ncols in range(1, max_columns + 1):
                frame = dt.itemsize * ncols
                if expected is not None:
                    header = size - expected * frame
                    if header < 0 or header > max_header_bytes:
                        continue
                    n = expected
                    detail = f"header_bytes = {header:,}"
                else:
                    if size % frame != 0:
                        continue
                    header = 0
                    n = size // frame
                    detail = f"header_bytes = 0 ({n:,} samples = {n / rate_hz:.6g} s at this rate)"
                matches += 1
                head = np.memmap(path, dtype=dt, mode="r", offset=header,
                                 shape=(min(n, 4), ncols))
                first = np.asarray(head[:, 0], dtype=float)
                vals = " ".join(f"{v:.4g}" for v in first)
                lines.append(f"  {name:8s} {order:6s} columns={ncols}  {detail}  first col 0: {vals}")
    if matches == 0:
        lines.append("  no layout matches. The file may be text, compressed, or have a larger header.")
    lines.append("Pick the layout whose first values look physically plausible and whose header "
                 "size is plausible, then set data.dat accordingly.")
    return lines
