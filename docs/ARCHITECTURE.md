# Architecture

## 1. Layers

```
                 ┌────────────────────────── CLI (cli.py) ──────────────────────────┐
                 │  dry-run            prepare                  events               │
                 └──────┬──────────────────┬────────────────────────┬────────────────┘
                        │                  │                        │
   config.py ───────────┼──────────────────┼────────────────────────┤  (typed, validated)
                        │                  │                        │
   preflight.py ◄───────┘                  │                        │  (read-only checks)
        │                                 │                        │
        │   ingest.py ◄── manifest.py ◄───┤                        │
        │       │             │           │                        │
        │       │        symlinks.py ◄────┤                        │
        │       │             │           │                        │
        │       │        splits.py ◄──────┤                        │
        │       │                         │                        │
        │   pipeline.py ◄── filters.py, resample.py, events.py ◄───┘
        │                                                      
   metrics.py  (classification metric engine; used by the evaluation layer, not yet wired)
   runs.py     (run directories, environment capture, exclusive writes; used by every command)
   synthetic.py (demonstration data with ground truth; test and demo only)
```

Dependencies point downward. Pure-signal modules (`filters`, `resample`, `events`) know
nothing about files, configuration, or CLI. This keeps them testable with synthetic arrays.

## 2. Processing order for one recording

1. **Ingest** (`ingest.load_signal`). Validates format and timestamps. Returns samples and the
   sampling rate, taken from the file header, the timestamps, or configuration, in that order.
2. **Notch** at the original rate, for each configured mains frequency. Done before any
   decimation (design R12).
3. **Anti-aliased decimation** (`resample.decimate_staged`) to the event-detection rate.
   Integer ratios only. Factors are split into stages of at most 10, each with a Butterworth
   low-pass whose passband edge is 0.8 × output Nyquist and whose stopband edge is at the
   output Nyquist. Zero-phase, offline.
4. **Mode-specific filter** (`FilterSpec.apply`): LPF, Gaussian, or LPF + Gaussian, or none.
5. **Baseline**: rolling median, odd window of about `baseline_window_s`.
6. **Noise**: robust σ = 1.4826 × MAD of the residual.
7. **Detection** (`events.detect_events_detailed`): hysteresis, minimum separation, minimum
   duration, start-edge guard. Each event records onset, offset, duration, I₀ and its source,
   ΔI, ΔI/I₀, amplitude, and censoring flags. Toff is the interval to the next onset.

## 3. Preparation for a training run

1. Preflight runs first. Any ERROR stops the run before a directory is created.
2. `runs/training_<timestamp>/` is created. Configuration, fingerprint, preflight report, and
   environment are written.
3. `discover_sources` finds files and attaches labels and groups from metadata or folders.
4. `build_manifest` hashes each file, reads headers, flags duplicates and unreadable files.
5. Labelled, readable rows form the **CV pool**. `grouped_stratified_folds` assigns folds
   for each repeat. `assert_plan_valid` checks that no group is split.
6. Symlinks are created under `data/symlinks/cv_pool/<class>/`. Each one is verified. The
   manifest and the symlink report are written with exclusive-create semantics.

## 4. Run directory

```
runs/<kind>_YYYY-MM-DD_HH-MM-SS[_NNN]/
├── config/          config_used.yaml (exact copy), config_resolved.json (+ fingerprint)
├── data/
│   ├── manifests/   master_manifest.csv, fold_assignments.csv
│   ├── symlinks/    cv_pool/<class>/…, symlink_report.csv
│   └── metadata/    (reserved)
├── environment/     environment.json
├── reports/         preflight_report.{json,csv}
├── metrics/         events.csv, events_summary.json   (events command)
├── artifacts/       run_summary.json                  (prepare command)
└── logs/, models/, checkpoints/, plots/, predictions/, explainability/  (reserved)
```

`<kind>` is `dryrun`, `training`, or `events`. A second run in the same second receives a
`_002` suffix. Nothing is reused.

## 5. Design decisions

| Decision | Reason |
|---|---|
| Strict config parsing (unknown keys are errors) | A typo must not silently change an experiment |
| Cross-field validation separate from parsing | Errors can be listed together, as preflight findings |
| Preflight reads metadata only | Dry run must be cheap and must never modify data |
| Symlinks verified, copies forbidden | Raw data are large and must have one authoritative location |
| Folds stored as manifest columns, not per-fold directories | Repeated CV would otherwise multiply links |
| Exclusive-create writes everywhere | Silent overwrites break provenance |
| Notch before decimation | Mains would alias to DC at low rates |
| Start-edge guard instead of silent trimming | Excluded events are counted, so the exclusion is visible |
| Synthetic truth in a separate file, never read by the pipeline | Prevents accidental use of ground truth |
| Metric applicability declared per metric | Prevents misleading values for single-label multiclass |

## 6. Memory and performance

- A 20 s recording at 1 MHz has 2 × 10⁷ samples: about 152.6 MiB as float64, before filter
  temporaries. `pipeline.process_recording` loads the whole recording, so peak memory is a
  small multiple of that. Preflight check P20 estimates this and fails if it exceeds 80% of
  available memory.
- The CSV reader counts and parses rows in Python, which takes seconds per million rows. The
  binary `.dat` reader avoids this: it memory-maps the file and needs no parsing. Use binary
  `.dat` for production-scale 1 MHz data.
- Chunked (overlap-save) processing is the planned fix. It is not implemented yet.

## 7. Known limitations

- The holdout split is not implemented. The CV pool is the only split.
- The start-edge guard is a heuristic. It excludes events whose onset falls within
  `max(edge_guard_s, baseline_window_s / 2)` of the recording start. Validation of that
  choice on real data is required.
- Event detection is threshold-based and assumes blockades are negative (assumption A2).
- The sign convention, units, and Ton/Toff definitions are assumptions until confirmed.
