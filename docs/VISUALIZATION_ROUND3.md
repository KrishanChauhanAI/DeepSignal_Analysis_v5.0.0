# Round 3 — Raw-file figures and expanded scientific visualization registry

This is additive. The original registry builders and the existing run artifacts/schema remain in place. The registry now appends **106** spec-tagged builders, for **173 total registry entries**. The run-level index remains `viz_registry_index.csv`; its additive `spec` column maps new entries to the user's 1–168 visualization checklist. Every builder reports `generated`, `skipped`, `not_available`, or `failed`, with a reason for non-generated outputs. Optional dependencies are explicitly `not_available`; they are not silently replaced.

## Raw waveform figure for every input file

When the figures stage is enabled, the pipeline reads each discovered recording from its original path using the configured ingestion contract (`data.dat`, original sampling rate, scaling, channel, and units). It writes one independent, full-duration overview PNG per metadata row to:

```text
runs/<task>/figures/files/<safe actual class label>/<sample_id>_<safe filename stem>_raw.png
```

The class directory is derived from each real class label; no class names are hard-coded. For example, actual top-level labels `Class1` and `Class2` yield `figures/files/Class1/` and `figures/files/Class2/`. Files are processed sequentially. For long traces, plotting uses a **display-only min/max envelope** (both extrema per time bucket, in temporal order) to preserve spikes/clipping while bounding rendering memory. It does not filter, smooth, normalize, or overwrite the signal. Axis uses the configured units; when the current is uncalibrated, labels remain arbitrary/configured units rather than being called pA/nA/µA.

All discovered files—including skipped/failed inputs—receive a line in:

```text
runs/<task>/figures/files/raw_file_figures_index.csv
```

This makes per-file omissions auditable. The index carries sample id, class, source basename, status/reason, original sample count/rate/duration, display-point count, and render method. Missing or unreadable source files are reported, not dropped.

The full-recording raw figures are separate from the older representative processed-signal plots. In particular, the new raw-vs-filter comparison builder reads the original source file and labels its units and configured filter parameters. Other registry entries that explicitly say “analysed” use the cached post-pipeline representation; the plot title identifies that distinction.

## Added event audit artifact

The one-pass file analysis now also retains the detector's event table and writes `events_table.csv` when events are available. Columns include sample/class/group/file, onset/offset/duration, per-event `I0` and its source, `DeltaI`, `DeltaI/I0`, amplitude, polarity, and edge-censor flags. The event plots consume these actual per-event measurements. Event detections are not presented as biological ground truth; empirical detector precision/recall requires independently annotated events.

## New plot families

Round-3 entries cover raw zoom/baseline/filter overlays; filter response and edge checks; sampling/filter QC; baseline-method comparisons; per-file signal QC, nonfinite/outlier indicators, drift/noise/SNR proxies, file/window/event counts; split/group/hash audits; spectral and ACF diagnostics; event overview/annotation/kinetics; empirical distributions and file-level feature comparisons; PCA/t-SNE; training metrics, calibration, ROC/PR, errors, ablations, feature importance; and lineage/KPI diagrams. They are all driven by run artifacts and data actually present.

Plot-specific limits are disclosed in the title/prerequisite or CSV twin: e.g., observed sample extrema are only a clipping proxy without instrument rail metadata; a dynamic-range/MAD ratio is an SNR proxy rather than event-specific SNR; between-file SD is not a confidence interval; a per-run score is not an external-validation result; and open-set claims require held-out unknown classes. The temporal split, concentration, molecular volume, synthetic truth, and several device/session plots are skipped when their data are absent.

## Coverage interpretation for the 168-item checklist

The 168 entries are a broad scientific wish-list, not 168 plots that are defensible on every dataset/run. The registry directly tags **106** items in its `spec` column; the pre-existing 67-entry registry continues to cover overlapping core plots (waveforms, PSD, scalograms, ACF, class counts, feature correlation, loss/metric curves, confusion/ROC/PR/calibration, embeddings, few-shot, ablation, and explainability). The machine-readable, 168-row crosswalk is [`VISUALIZATION_168_COVERAGE.csv`](VISUALIZATION_168_COVERAGE.csv). The remainder is conditional or requires a distinct experiment/artifact not created by a normal classification run. In particular:

- 1 MHz raw per-file plots are now generated from source files; analysis-rate plots remain separate.
- 3-D embeddings, PaCMAP/UMAP, and optional decomposition/entropy plots depend on dimensions and installed packages.
- Device/session confounding probes, group bootstraps, label-permutation nulls, stricter external holdouts, multi-seed studies, and real-corpus revalidation require dedicated split/experiment protocols; they are not reverse-engineered from one run.
- Synthetic event-boundary validation needs true simulated onset/offset labels. Real events are never treated as their own ground truth.
- Pseudo-labeling, open-set unknown-class evaluation, foundation-model/TabPFN benchmarking, VAE/FDTD/physics/structure/free-energy plots, and streaming deployment curves are generated only when those corresponding models, labels, calibrations, and telemetry are actually supplied.

The registry index and the CSV twins are the run-specific source of truth. `skipped` means a prerequisite was not satisfied, `not_available` means a declared optional dependency is absent, and `failed` means a genuine builder error that should be investigated.

## Output compatibility and performance

The original per-class/feature/model figures remain unchanged. The 106 new builders are appended via `viz_registry_ext.py`; `RegistryCtx` receives optional event, metadata, split, and settings fields with defaults so existing callers remain compatible. Raw full-recording plots stream one source file at a time and use min/max display reduction; feature/embedding plots operate on file-level data rather than multiplying windows into pseudo-replicates. CSV twins accompany numeric registry outputs. The raw figures have an explicit index instead of one enormous multi-file CSV.
