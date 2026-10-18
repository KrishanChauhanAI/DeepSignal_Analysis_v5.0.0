"""``python -m trapsig.synthetic_cli --out data/SYNTHETIC_demo`` writes demo data."""

from __future__ import annotations

import argparse

from .synthetic import SyntheticSpec, write_synthetic_dataset


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Write SYNTHETIC demonstration recordings (not real data).")
    p.add_argument("--out", required=True, help="directory for the synthetic recordings")
    p.add_argument("--replicates", type=int, default=6, help="replicates per class")
    p.add_argument("--duration-s", type=float, default=0.5)
    p.add_argument("--fs-hz", type=float, default=100_000.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--format", choices=["csv", "dat"], default="csv",
                   help="csv (with banner) or dat (raw float32 little-endian, no header)")
    p.add_argument("--class-folders", action="store_true",
                   help="write each class into its own subfolder, as a lab would arrange it")
    args = p.parse_args(argv)
    spec = SyntheticSpec(
        fs_hz=args.fs_hz,
        duration_s=args.duration_s,
        replicates_per_class=args.replicates,
        seed=args.seed,
        file_format=args.format,
    )
    meta = write_synthetic_dataset(args.out, spec, class_folders=args.class_folders)
    print(f"SYNTHETIC recordings written to {args.out}; metadata: {meta}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
