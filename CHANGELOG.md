# Changelog

All notable changes are recorded here. Versions follow [Semantic Versioning](https://semver.org/).

## Round 3 - raw per-file figures and expanded visualization coverage (2026-10-16)

Additive visualization work; existing registry entries and plot/report paths remain intact.

- NEW `src/trapsig/viz_files.py`: independent original-source full-recording PNGs under
  `figures/files/<actual-class-label>/`, with sequential file loading and a display-only min/max
  envelope for long traces; every discovered input is recorded in `raw_file_figures_index.csv`.
- `train.py`: one-pass event rows exported to `events_table.csv` (per-file/class/group, onset,
  offset, dwell, I0 source, DeltaI, normalized DeltaI, amplitude, polarity and censor flags).
- NEW `src/trapsig/viz_registry_ext.py`: 106 spec-tagged builders appended to the prior 67-entry
  registry (173 total). The run index records each generated/skipped/unavailable/failed status;
  absent conditional evidence is never fabricated.
- `viz_registry.py` / `stages.py`: optional event, raw-source settings, metadata and leakage context
  passed additively; original run paths and file/class discovery are retained.
- NEW `docs/VISUALIZATION_ROUND3.md` and the 168-row `docs/VISUALIZATION_168_COVERAGE.csv`;
  README documents the new output tree.
- NEW `tests/test_viz_round3.py`: per-file output, min/max envelope spike retention and additive
  registry checks.
- Verification: end-to-end synthetic demo exited 0; 12/12 per-file raw figures generated; registry
  129 generated / 40 prerequisite-skipped / 4 optional-dependency unavailable / 0 failed. Full suite:
  306 passed; Ruff and mypy clean. The user-requested `runs/` cleanup leaves that directory empty.

## Round 2 - configurable feature registry and robustness add-ons (2026-10-16)

Additive only; every earlier module, artifact and default behaviour is untouched.

- NEW `src/trapsig/featreg.py`: registry of 154 label-blind features across 11 column families
  (time_stats, energy, envelope, dynamics, peaks_events, quality, spectral, time_frequency,
  entropy_nonlinear, ar_cepstral, decomposition; "learned representations" is the 12th family
  and lives in branches A-C) in three tiers (57 mandatory / 76 recommended / 21 experimental),
  each with description, unit and parameters; sampling-rate-aware band/mel/packet ids; NaN
  policy documented per feature; quadratic estimators capped (`_CAP = 2048`).
- `quad.window_features` / `window_feature_names` now delegate to the registry; the legacy 38
  engineered ids remain an exact subset at tiers [1, 2], so saved sidecars stay comparable.
- `build_quad_tensors` honours `quad.feature_tiers` / `feature_families` and produces
  recording-level summaries (`recording_summary`: rec_n_windows + per-feature _rec_mean/_rec_sd).
- `cv_quad`: train-fold-only variance feature selection (`feature_select: variance`, default on;
  test folds never vote), per-file max-softmax-probability vector (`QuadCVResult.msp`).
- `train_quad_fold`: optional batch-hard triplet loss (`triplet_weight`), LR schedules
  (`lr_schedule: none|cosine|step`), early stopping (`early_stopping_patience`), and a
  per-fold checkpoint under `<run>/checkpoints/quad_<tag>.pt`.
- `train.py` writes `feature_registry.csv`, `recording_feature_summary.csv`,
  `quad_open_set_flags.csv` (when `open_set_msp_threshold` is set) and adds
  `summary.quad.feature_registry` / `summary.quad.open_set` blocks.
- `evalmetrics.py`: `cohens_kappa` plus micro/macro/weighted precision/recall/F1 rows.
- `config.py`: 7 new `quad` keys with validators; `configs/custom_dat.yaml` documents each with
  options, defaults, examples and RULE lines (181 RULE lines now).
- Tests: NEW `tests/test_featreg.py` (8) + 2 quad round-2 tests + kappa coverage; suite 303
  passed at the time of writing. ruff + mypy clean (36 source files).
- Docs: `docs/QUAD_ARCHITECTURE.md` section 11 (registry tables, STFT-vs-wavelet trade-off,
  fixed-transforms-vs-learned-encoders, honesty rules).

## [Unreleased]

### Added
- **Four-branch deep architecture (`quad`, additive, default OFF)** — `quad.enabled: true` adds an
  integrated model without touching any existing behaviour: Branch A learned temporal encoder
  (dilated residual multiscale CNN + temporal attention + transformer, receptive field computed
  from the config), Branch B learned spectral encoder on a FIXED Hann-STFT front end with
  frequency attention, Branch C self-supervised pretraining (NT-Xent + masked-span reconstruction
  on training-fold-minus-inner-validation windows only, semantics-preserving augmentations),
  Branch D 38 engineered window features (median/IQR imputed and robust-scaled from the training
  fold only). Cross-branch attention fusion with learned softmax gate weights (reported per run),
  Head A supervised softmax (focal/class weights optional), Head B prototypical metric head with
  SupCon auxiliary, episodic fine-tuning and closed-set few-shot episode evaluation on held-out
  fold-test file embeddings (Wilson 95% CIs). Paired branch ablations via `quad.branch_ablation`
  (`quad_minus_<branch>` models on identical folds). New modules: `quad.py`, `fewshot.py`,
  `evalmetrics.py`, `viz_registry.py`; new config blocks `segmentation`, `quad`,
  `sampling.rate_menu_hz`/`allow_off_menu_rates`, `training.report_top_k`/`calibration_bins`/
  `pr_curves` — every key documented with options, examples and RULEs in `configs/custom_dat.yaml`.
- **Extended evaluation surface** for every scored model: per-class specificity, support and error
  counts, multiclass MCC, Brier, ECE/MCE with stored formulae, one-vs-rest PR-AUC, top-k accuracy
  → `metrics_extended.csv`; undefined values written as empty cells with the reason.
- **Visualization registry**: 67 documented plot types in six categories (time / frequency /
  time-frequency / nonlinear-complexity / dataset-class / DL diagnostics) with prerequisites,
  PNG + CSV twins, and an honest audit trail (`viz_registry_index.csv`: generated | skipped |
  not_available | failed with reasons). EMD/VMD are recorded `not_available` (optional dependency
  not shipped) instead of being faked.
- **PDF sections 11–14** (sampling-rate design; extended metrics; four-branch architecture with
  gates, SSL config, few-shot protocol/results and limitations; visualization registry) each with
  a CSV twin (`report_sampling.csv`, `report_extended_metrics.csv`, `report_quad.csv`,
  `report_viz_registry.csv`); new figures `quad_fusion_gates`, `fewshot_shot_curve`,
  `quad_branch_ablation`, `quad_embedding_pca/tsne`, `pr_<model>`; `fewshot_episodes.csv`.
- **Docs**: `docs/QUAD_ARCHITECTURE.md` (specification, diagram, paradigm fit, feature inventory,
  stages 1–7, leakage protocol, evaluation design, registry, structure delta, open unknowns).
- **Model-specific nested hyperparameter optimization** (`trapsig.optimize`, `deep.search_fold_params`)
  - `training.hpo` (`none` | `random` | `halving`), `training.hpo_metric`
    (`balanced_accuracy` | `macro_f1` | `roc_auc_ovr` | `log_loss`, always oriented
    higher-is-better, log loss negated), `training.hpo_n_iter`, `training.hpo_seed_offset`.
    Per-model spaces: logreg `C`; rf trees/depth/leaf/features; hgb iterations/learning
    rate/leaves/leaf/l2. Every search runs *inside* one outer training fold, scored on grouped
    inner splits of that fold's files, so the test fold never chooses a hyperparameter.
  - Successive halving (eta=3) with a recorded fallback to random search when a fold holds fewer
    than six files; failed trials are written with `status=failed` and a reason, never scored 0.
  - One row per trial in `hpo_trials.csv` (model, repeat, fold, rung, n_train_files, score,
    n_inner_splits, status, reason, params JSON); `summary["hpo"]` aggregates per model; the
    permutation nulls inherit the same search so the p-value is not biased by tuning.
  - Model B: `deep.hpo_trials` full-`train_fold` trials per fold scored on the fold's
    inner-validation files, winner retrained once (exactly one checkpoint per fold); the embedding
    width is deliberately not searchable and every run says so
    (`summary["hpo"].per_model.cnn1d.not_searched`).
  - `--quick` now forces `training.hpo=none` and `deep.hpo_trials=0` and says so.
- **Inference from saved artifacts**: `training.save_models` writes every fitted Model A fold
  pipeline to `models/` with a JSON sidecar (feature columns, classes, hyperparameters, version,
  config fingerprint) plus `models/INDEX.json`; new subcommand
  `trapsig predict --run RUN --model NAME [--fold-tag TAG] (--files ... | --input-dir DIR)`
  scores new `.dat` recordings with the run's own configuration, averages the saved fold models'
  probabilities, keeps skipped files in the output CSV, and refuses models whose sidecar is
  missing or whose version differs (`--allow-stale` to override, reported).
- **Data/EDA, ablation, HPO and error-analysis figures with CSV twins** (`trapsig.viz`):
  `dataset_composition`, `feature_distributions`, `missing_values`, `feature_correlation`,
  `ablation_feature_domains`, `hpo_trials`, `error_analysis_<model>`; twins
  `report_dataset_composition.csv`, `report_feature_distributions.csv`,
  `report_missing_values.csv`, `report_feature_correlation.csv`,
  `report_error_analysis_<model>.csv`. `figures_index.csv` records one row per requested category
  with status generated/skipped/failed/not-requested and the reason, so a missing figure is a
  statement, not a silence. Tall tables are capped for readability; the CSVs keep every row.
- **Fixed**: `plot_psd_by_class` looked for `spec_band_<i>` while `features.py` writes
  `spec_log10_band_<lo>_<hi>Hz`, so the PSD figure was silently never drawn; it now builds names
  with the same helper and the index would have reported the skip either way.
- **Dependency manifests**: `requirements.txt` (core pinned to the executed environment) and
  `environment.yml` (conda + pip, optional reportlab/torch/shap in the pip section).
- Tests: `tests/test_optimize.py`, `tests/test_predict.py`, `tests/test_viz_figures.py`
  (25 tests) on top of the previous 245; suite now 270 passed.
- `docs/ARCHITECTURE_AUDIT.md`: the full Phase I-XII audit with the inventory, the Mermaid
  execution flow, the integration matrix, the HPO report, the verification commands and the
  remaining limitations.
- **Metric learning, a prototypical head, and a leakage-safe fusion head** — the learning
  strategies are now one architecture instead of models bolted on side by side
  (`docs/LEARNING_STRATEGIES.md` is the audit: what is implemented, what flows between the
  components, what contributes to the final prediction, and what is deliberately absent).
  - `deep.metric_loss` (`supcon` | `prototypical` | `none`), `deep.metric_loss_weight`,
    `deep.metric_temperature`: an auxiliary term on the GAP embedding the head already consumes,
    so one objective shapes the representation every downstream strategy reads. The per-epoch
    value is logged as `train_metric_loss`; a batch with no usable contrastive pair contributes
    zero rather than NaN.
  - `deep.prototypical_head` and `deep.proto_distance`: a new scored model `proto_cnn`, the
    nearest-class-prototype reading of the same frozen embedding. Prototypes are the mean
    L2-normalised embedding per class over the fold's **inner-validation files**, which the
    encoder was early-stopped on but never trained on, so the test fold contributes nothing.
  - `deep.fusion` (`none` | `blend` | `stack` | `both`): a new scored model `fusion`, a logistic
    regression per (repeat, fold) on the robust-standardised handcrafted file features
    concatenated with the mean file embedding, fitted on the same inner-validation files and
    applied to that fold's test files. `blend` keeps the pre-declared convex blend (`hybrid`);
    `both` reports the two side by side. A learned combiner no longer needs a nested CV of both
    models, and the obsolete "deliberately not offered" note is gone.
  - A file counts as prototyped or fused only where **every** repeat scored it; partially scored
    files are excluded from that model's metrics and the exclusion is reported.
- **Ten handcrafted feature groups, 90 columns per 20 s file** (`trapsig.features`): `noise`,
  `time`, `amp`, `event`, `ctx`, `spec`, `tf`, `acf`, `shape`, `morph` — time domain (Hjorth,
  line length, Teager energy, permutation entropy, drift), amplitude domain (RMS, peak, crest
  factor, percentiles, dynamic range), frequency domain (Welch bands, slope, centroid, spread,
  entropy, roll-off, harmonicity), time-frequency (STFT flux, per-band drift, frame entropy, db2
  wavelet levels), distribution shape (skew, excess kurtosis, tail fraction), morphology (turning
  points, rise/fall asymmetry, area-to-peak, sharpness, self-similarity to the file's mean event
  waveform), and event context (inter-event intervals, burstiness, onset distribution and Rayleigh
  R, rate drift, amplitude-duration coupling, baseline stability). Every column carries its group's
  prefix, and a value that cannot be computed is NaN with a reason rather than a silent zero or an
  exception. Removed an exact duplicate (`time_mean_abs_diff_norm` was `time_line_length_norm`),
  renamed the event-context features to the `ctx_` prefix so group gating matches column names,
  and clamped the Welch segment length for short signals.
- **Feature ablation and redundancy reporting** (`trapsig.ablation`, PDF section 10 and its twin
  `report_ablation.csv`): `training.feature_groups` declares which groups are computed,
  `training.ablation` (`none` | `groups` | `leave_one_out` | `both`) refits
  `training.ablation_model` on each group alone and on everything minus one group **on the same
  folds, files and settings** as the main cross-validation, and `training.redundancy_threshold`
  flags near-duplicate feature pairs by |Spearman| on the pooled matrix. The ablation never changes
  the headline result, and flagged pairs are reported in `feature_redundancy.csv` and **never
  dropped**, because dropping columns on the pooled matrix would use test-fold information.
  `--quick` sets `training.ablation: none`. The default is `both`.
- `report_ablation.csv` (§10) joins the CSV twins; §2 now also records the feature groups used,
  the per-group counts, the selection rule, and the all-NaN columns that were dropped.
- 25 new tests: `tests/test_ablation.py` (variants, masks, skip reasons, duplicate column names,
  redundancy thresholds and its file-count rule, CSV writers, configuration errors) and
  `tests/test_metric_learning.py` (both auxiliary losses and their degenerate batches, prototype
  construction and probability normalisation, the fusion head's strict per-fold file set, the
  proof that one test file cannot influence another, `train_fold` returning validation embeddings
  and prototypes, and the new configuration rules).
- **Per-run metadata record** (`trapsig.metadata`): `metadata.csv` and `metadata.json` are written
  into *every* run directory — `dry-run`, `prepare`, and `train` — because the folds are
  re-randomised on each run, so the split belongs to the run and not to the configuration.
  `metadata.csv` has one row per source file (pool, non-pool, and skipped) with its manifest
  identity, this run's outcome (`status`, `reason`, `n_events`, `n_windows`, `model_b_scored`),
  and per repeat its `fold_rN`, `role_rN`, `trains_in_folds_rN`, and `cnn_inner_val_rN` (which
  outer folds drew the file into the CNN's inner validation split). `metadata.json` records the
  seed, config fingerprint, repeats x folds, class counts, grouping policy, an explicit
  split-randomisation statement, and the role convention. Both are written with exclusive-create
  semantics, and the console prints where the record went.
- **CSV twins of every PDF section**: `report_data_landscape.csv` (§1), `report_methods.csv` (§2),
  `report_results.csv` (§3, model rows plus the gate G7 row and permutation columns),
  `report_figures.csv` (§4), `report_explainability.csv` (§5), `report_advanced.csv` (§6),
  `report_limitations.csv` (§7), `report_information_gaps.csv` (§8),
  `report_reproducibility.csv` (§9). Each PDF section heading names its twin, and the CSV keeps
  the full un-wrapped text of every cell.
- `TRAPSIG_DEBUG=1` prints the full traceback when a run fails, alongside the one-line reason.
- 22 new tests (`tests/test_metadata.py`) for the metadata column layout, the
  `(n_repeats, n_pool)` fold orientation, role and training-fold semantics, non-pool files,
  reading the prepared split back in the training run, missing-manifest tolerance, the nine
  section CSVs, and the fixed-width wrapping cells of PDF sections 7 and 8.

- **`trapsig run`**: the whole pipeline in one command (preflight → prepare → train → explain →
  advanced analyses → figures → CSV rollups → PDF). Flags: `--dry-run`, `--quick`,
  `--model-set`, `--no-pdf`, `--no-explain`, `--from-prepared`, `--no-stages`. The older
  individual commands are unchanged and still work.
- **Compute budget by measurement.** `run --dry-run` times one or two real recordings for the
  collection cost and one CNN training batch at the resolved window length for the deep cost,
  and reports each component with the basis it was measured on (`budget_estimate.json`).
- **Model B**: compact 1D residual CNN with dilated convolutions, global average pooling, and a
  linear head (`trapsig.deep`): AdamW with warmup and cosine decay, gradient clipping, early
  stopping and checkpoint selection on a validation split carved from the outer training fold by
  file group, class-weighted cross-entropy only at or above the design imbalance threshold, and
  temperature scaling fitted on the validation split only. Per-epoch training log
  (`logs/deep_training_log.jsonl`) and per-fold checkpoints under `checkpoints/`.
- **Event windows** (`trapsig.windows`): fixed-length windows centred on events with a
  pre-event baseline fraction; window length derived from the event-duration distribution
  (`deep.window_ms: auto`); edge-excluded events counted, not padded; robust median/IQR scale
  fitted on training windows only.
- **The hybrid** (`trapsig.hybrid`): window→file aggregation by a pre-declared rule, a fixed
  convex blend of Model A and Model B with per-file fallback flagging in `predictions.csv`, and
  a comparison framework that scores Model A on the Model B subset so the head-to-head is like
  for like. Gate G7 (Model B beats or matches Model A) is decided there with CI overlap assessed.
- **Explainability** (`trapsig.explain`): exact linear SHAP for logistic regression (with the
  binary single-coefficient row expanded exactly), TreeSHAP for the tree models when the optional
  `shap` package is present, Grad-CAM (valid for conv → GAP → linear), integrated gradients,
  occlusion, and the mandatory shortcut probes (baseline-only classifier, time-shift null).
  Attributions are written as `attributions.npz` plus a segment-level CSV.
- **Conditional analyses** (`trapsig.advanced`, all gated, off by default): prototypical few-shot
  over out-of-fold embeddings (setting (b) of design 11.7, SKIPPED with a reason when K is
  infeasible), inductive confidence-aware pseudo-labelling with skew rejection and retraining
  from scratch, novelty detection (reported "not evaluable" without an unknown class, with a
  labelled leave-one-class-out diagnostic), and clustering with bootstrap Jaccard stability.
- **Figures** (`trapsig.viz`): analysed trace with detected events, class-mean event window
  (files averaged first), band power by class, confusion matrices, ROC, calibration, learning
  curves, model comparison with bootstrap intervals, embedding PCA, and attribution panels.
- **Reporting** (`trapsig.report`): `report_models.csv`, `report_limitations.csv`,
  `report_information_gaps.csv`, `report_reproducibility.csv`, and `report.pdf` with results,
  figures, gate G7, limitations with safeguards, information gaps, and reproducibility metadata.
- Optional-dependency extras: `.[deep]` (torch), `.[explain-report]` (shap, reportlab),
  `.[full]`. Without them the pipeline runs Model A and records exactly what to install.
- `validate_modelling`, so `events` and `prepare` are still never blocked by modelling settings.
- 20 new tests (`tests/test_hybrid.py`) covering window rules, leakage-safe scaling, matched
  subset comparison, gate verdicts, Grad-CAM shape, binary SHAP expansion, configuration gates,
  the one-command run, dry-run behaviour, quick overrides, and PyTorch-missing degradation.

### Changed
- `training.ablation` defaults to `both` (was `groups`): the two directions answer different
  questions, and the cost is one cheap refit per variant. `--quick` sets it to `none` and says so
  in the override note.
- `deep.fusion` accepts `none` as well as `blend | stack | both`, and `deep.model_set: hybrid`
  with `deep.fusion: none` is now a configuration ERROR with the two ways out spelled out.
- The `deep.combine` documentation no longer claims a stacked combiner is deliberately absent;
  it governs the blend weight only, and points at `deep.fusion: stack` for the learned combination.
- PDF tables now have hard cell boundaries in sections 7, 8 **and 10** (previously 7 and 8).
- **PDF tables now have hard cell boundaries.** Every cell is a reportlab `Paragraph` in a table
  with fixed column widths, so content wraps inside its own cell instead of running into the next
  column or off the page. Sections 7 and 8 use three fixed columns (52 + 60 + 60 mm, exactly the
  text frame) with `splitByRow=True` and `splitLongWords=1`; nothing is truncated.
- `configs/custom_dat.yaml` documents every key with its type, all allowed options, the default,
  and an example, plus explicit `RULE:` lines for each cross-key dependency (filter mode vs
  `lpf_cutoff_hz` / `gaussian_sigma_s`, `label_from` vs `metadata_csv`, `dat.encoding` vs
  `header_bytes` / `header_lines`, `deep.enabled: false` vs `model_set` and the explainability and
  advanced gates, and which changes invalidate an existing prepared run).
- The step-5 output listing names the metadata files and every section CSV with what it contains.
- `run_training(..., stages=True)` runs the downstream stages inside the training process,
  because they need the window set and feature table already in memory.
- `configs/custom_dat.yaml` and `configs/demo_custom_dat.yaml` gained `deep`, `advanced`,
  `explain`, and `report` sections; the templates now lead with `trapsig run`.
- `matplotlib` is a core dependency; `torch`, `shap`, and `reportlab` are optional extras.

### Fixed
- The supervised-contrastive loss masked its own diagonal with `-inf` and then multiplied by the
  positive-pair mask; `-inf * False` is NaN, so the term was NaN for every batch. It now selects
  with `torch.where`, and a test pins the degenerate batches (one window, one class, all
  singletons, no negatives) to zero rather than NaN.
- Prototypes were built by comparing integer class indices against class **names**, so every fold
  reported "0 prototypes" and the head silently produced nothing. The labels are mapped to names
  first, and `train_fold` is now tested to return a prototype probability per test window.
- `run_ablation` mapped feature names to column indices, so two columns sharing a name made a
  group keep a column it owns (`without:<group>` still contained the informative feature). Variants
  are now built from column positions, with a test for the collision.
- Grad-CAM reduced over the batch axis instead of the channel axis (a broadcasting error), so
  the attribution map had the wrong shape. Caught by the end-to-end run, now shape-tested.
- Linear SHAP indexed the coefficient row by class position, which is out of bounds for binary
  logistic regression (one stored row). The row is now expanded by negation, which is exact.
- File-level aggregation inverted its first-seen test, turning never-predicted windows into
  implicit zero probabilities; unpredicted windows are now excluded and stay NaN.
- Probability rows that summed to 1 only within 1e-6 triggered a spurious sklearn warning in
  `roc_auc_score`; the metric engine now renormalises exactly after validating.
- Binary ROC-AUC: the metric engine passes the positive-class probability column with a 0/1
  truth for two-class runs (the shape sklearn requires) and the two-class path is covered by the
  end-to-end tests. (Reworded: the current behaviour is verified and tested; the original
  failure could not be reproduced because no pre-fix history exists in this workspace.)
- `load_prepared` checks the repeat count before the manifest, so a stale run reports the real
  cause.

### Not yet implemented
Holdout test set and access ledger, self-supervised pretraining, generative augmentation,
automatic feature selection on the pooled matrix, baseline/detrending candidate comparison,
censoring-aware dwell estimators, and chunked 1 MHz processing — each with its reason in
`docs/METHODS.md` section 9 and `docs/LEARNING_STRATEGIES.md` section 5. The learned (stacked)
combiner is no longer on this list: `deep.fusion: stack` implements it leakage-safely.

## [0.1.0] - 2026-10-08

### Added
- Typed, strict YAML configuration with cross-field validation (`trapsig.config`).
- Preflight / dry-run with checks P01–P26 and PASS / WARNING / ERROR / SKIPPED /
  NOT APPLICABLE statuses (`trapsig.preflight`).
- Format registry for `.npy` and `.csv` / `.txt` recordings with header and content checks
  (`trapsig.ingest`).
- Source discovery, metadata join, grouping, SHA-256 hashing, duplicate detection, and the
  master manifest (`trapsig.manifest`).
- Verified symlink split layout with manifest-only fallback; no copying (`trapsig.symlinks`).
- Grouped, stratified, repeated K-fold with assertions against group leakage
  (`trapsig.splits`).
- Staged, anti-aliased integer decimation with measured alias attenuation (`trapsig.resample`).
- LPF, Gaussian, notch filters and an explicit `FilterSpec` for the three modes (`trapsig.filters`).
- Rolling-median baseline, robust noise, hysteresis event detection with start-edge guard,
  Toff, and censoring flags (`trapsig.events`, `trapsig.pipeline`).
- Classification metric engine with applicability and documented exclusions (`trapsig.metrics`).
- Run directories, environment capture, and exclusive-create writes (`trapsig.runs`).
- CLI commands `dry-run`, `prepare`, `events` (`trapsig.cli`).
- SYNTHETIC demonstration data with separate ground truth (`trapsig.synthetic`,
  `trapsig.synthetic_cli`). Not measurements.
- Laboratory `.dat` format: configurable binary (dtype, byte order, header, interleaved columns,
  scale) or text layout under `data.dat`, with size-consistency checks (`trapsig.ingest`).
- Folder-based class labels end to end (`data.label_from: folder`).
- `trapsig inspect-dat`: read-only listing of layouts consistent with one file's size.
- Synthetic `.dat` generator with class folders (`synthetic_cli --format dat --class-folders`).
- Configs `configs/custom_dat.yaml` (template) and `configs/demo_custom_dat.yaml` (synthetic).
- Per-file assessment: `metrics/per_file_summary.csv` with one row per recording, including skipped
  recordings and their reasons (non-finite samples, flat signal).
- Manifest gains `source_subdir` (nested folder) and `name_annotation` (raw text after ` - `).
- `data/manifests/ignored_files.csv` lists every file the glob did not match; preflight P06 counts them.
- Exact duplicates share a group and are flagged when they span classes (no leakage across folds).
- `events_summary.json` states the grouping policy and the skipped count.
- Test suite (147 tests) including detection-recall validation on synthetic ground truth,
  exit-code contract tests (0/1/2), preflight contract tests, and exclusive-write tests.
- `make demo` generates the synthetic data only when missing, so it can be re-run.
- Balanced accuracy defined explicitly: classes with no true support are excluded and named in
  the result, instead of relying on sklearn's silent drop.
- Preflight reports config-only label errors (P07, P08) even when `source_dir` is unset.
- PEP 561 `py.typed` marker; `CONTRIBUTING.md`, `SECURITY.md`, `.editorconfig`,
  `.pre-commit-config.yaml`.
- Documentation: README, `docs/ARCHITECTURE.md`, `docs/METHODS.md`.
- CI workflow (lint, type check, tests, demo) and Makefile.

### Fixed during development
- Start-of-recording filter and baseline transients were detected as events (5 false
  positives in the synthetic validation). They are now excluded and counted.
- Header-only checks accepted non-numeric CSV content as readable. Now the first and last data
  rows must be numeric.
- Two-column CSV files required a header sampling rate even when timestamps supply it.
- sklearn `UserWarning` from balanced accuracy when a predicted class has no true support
  (see the metric engine change above).
- Missing config file raised an uncaught error (exit 1). Now exit 2 as specified.
- Text `.dat` row errors surfaced as raw ValueError at load; now IngestError with the line number.
- Manifest and per-file summary: a new column was missing from the fixed manifest field list; fixed.
- `inspect-dat` on a directory now explains what to pass.
- numpy 2.5 stub incompatibility in preflight P15 (bare `.max()` / `.min()` on counts).

### Not yet implemented
Holdout split, RMS / PSD / ACF / dwell estimators, baseline comparison, feature and deep models,
representation, few-shot, semi-supervised and novelty methods, explainability, figures, PDF
reports, chunked processing. See `README.md`.
