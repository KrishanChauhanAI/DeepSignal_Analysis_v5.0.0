# Methods and conventions

This document states the definitions the code implements. Anything marked **assumption**
must be confirmed with the experimental team before results are interpreted.

## 1. Quantities

| Symbol | Definition | Implemented in |
|---|---|---|
| I(t) | Recorded current (assumed A; assumption A1) | `ingest.load_signal` |
| baseline(t) | Rolling median over `baseline_window_s` | `events.rolling_median_baseline` |
| r(t) | I(t) − baseline(t) | `events.detect_events_detailed` |
| σ | 1.4826 × median(\|r − median(r)\|) (robust, MAD-based) | `events.robust_sigma` |
| I₀ | Median of I over the `pre_event_window_s` before onset; if that window is too short, the rolling baseline at onset (recorded as `i0_source`) | `events.detect_events_detailed` |
| ΔI | mean(I over event) − I₀ | `Event.delta_i_mean` |
| ΔI/I₀ | ΔI / I₀. NaN when I₀ = 0 | `Event.delta_i_over_i0` |
| Amplitude | max\|I − I₀\| inside the event | `Event.amplitude` |
| Duration (Ton, working definition) | Offset − onset, in seconds (assumption A2) | `Event.duration_s` |
| Toff (working definition) | Onset of next event − offset of this event (assumption A2) | `events.off_intervals` |

Only one I₀ definition is used per run. Other definitions from the design (local, recording
median, fixed) are not implemented yet. When they are, they must be reported as sensitivity
analyses, not mixed.

## 2. Event detection

An event is a contiguous segment where r(t) < −`exit_sigma`·σ, that contains at least one
sample where r(t) < −`threshold_sigma`·σ. This is hysteresis: the exit threshold is looser
than the entry threshold, so an event does not split on noise.

Rules applied in this order:

1. Segments closer than `min_separation_s` are merged.
2. Segments shorter than `min_duration_s` are rejected.
3. Segments whose onset is before `max(edge_guard_s, baseline_window_s/2)` are excluded and
   counted. Reason: zero-phase filtering and the baseline produce transients at the start of
   a recording, and synthetic validation shows these cross the entry threshold.
4. Segments that end at the last sample are kept and flagged `censored_end`, because the
   measured synthetic end transient stayed below the entry threshold.

The detector assumes blockades are negative (assumption A2). Positive excursions are not
detected.

## 3. Resampling

Decimation from fs_in to fs_out requires fs_in / fs_out to be an integer. The ratio is
factorized into stages, each at most 10. Each stage applies a zero-phase Butterworth
low-pass, designed with `buttord` so that:

- passband edge = 0.8 × output Nyquist (passband ripple ≤ 0.5 dB)
- stopband edge = output Nyquist (attenuation ≥ 60 dB)

Verification: `alias_suppression_db` measures the gain at 0.25·fs_out (passband) and the
attenuation of a tone at 0.75·fs_out (which would fold into the passband without
anti-aliasing). Tests require ≥ 40 dB of alias attenuation and < 0.5 dB passband error on
the 100 kHz → 1 kHz path.

Mains is removed before decimation (notch at the original rate). A 50 Hz component is
attenuated by more than 60 dB at 10 Hz output in the tested cascade.

## 4. Filter modes

| Mode | Transfer function | Parameters | Label |
|---|---|---|---|
| LPF | Butterworth SOS, zero-phase (`sosfiltfilt`) by default | order, fc | `Filter: LPF (…)` |
| Gaussian | Gaussian kernel, σ in seconds converted to samples | σ_t | `Filter: Gaussian (…)` |
| LPF + Gaussian | LPF followed by Gaussian | all of the above | `Filter: LPF + Gaussian (…)` |

Gaussian effective cutoff: |H(f)| = exp(−2π²σ_t²f²). The −3 dB amplitude point is
f_c = √(ln√2 / 2π²) / σ_t ≈ **0.13251 / σ_t**. Verified numerically in
`tests/test_filters.py`.

Zero-phase filtering squares the magnitude response, so the effective order is twice the
nominal order. It is non-causal and is for offline analysis only.

**Combined mode is not a new information channel.** Both stages are linear and time-invariant,
so the cascade equals one filter with response |H_LPF · H_G|. It is an ablation.

## 5. Splitting

Folds are StratifiedGroupKFold (scikit-learn) with `shuffle=True`, repeated with
`random_state = seed + repeat`. Groups are the strongest available level: replicate, then
session, then file (file only with explicit opt-in). Every repeat is checked by
`assert_plan_valid`: each row is assigned exactly once, and no group appears in more than one
fold.

Because `shuffle=True` re-draws the allocation on every run, the split is a property of the run
and not of the configuration: two runs of the same YAML can put the same file in different folds.
Every run therefore writes its own authoritative record of the split it used (§16), into that
run's directory, and the console prints where it went. Nothing downstream re-derives the split
from the seed; the training stage reads the fold assignment back from the prepared run.

Feasibility is checked before any fold is built:

- at least `min_groups_per_class` distinct groups per class (default 5; a heuristic, see §7)
- at least `n_splits` groups per class, so every test fold can contain every class
- at least 2 classes

## 6. Metrics

Implemented: accuracy, balanced accuracy, macro-F1, micro-F1 (reported as an identity check),
MCC, Cohen's κ, log loss (needs probabilities), ROC-AUC one-vs-rest macro (needs probabilities
and all classes present in the truth).

Excluded, with reasons: Hamming loss and Jaccard (equivalent to accuracy for single-label
data), Gini (not a performance metric here), ranking metrics (only for retrieval evaluation),
regression metrics (only in a named regression sub-analysis).

A metric that cannot be defined returns `status="undefined"` with a reason, never a
misleading number.

## 7. Heuristics and open choices

| Parameter | Value | Status |
|---|---|---|
| `min_groups_per_class` | 5 | Heuristic. Below about five independent groups per class, fold-level variance tends to dominate. Must be justified with the data. |
| `threshold_sigma` | 5.0 | Conventional default. Must be set from the noise audit. |
| `exit_sigma` | 3.0 | Conventional default. |
| `baseline_window_s` | 1.0 (default) | Must be set from the event-duration distribution. |
| `edge_guard_s` | 0.01 | Chosen from the synthetic start transient. Must be checked on real data. |
| Anti-alias passband / stopband | 0.8 / 1.0 of output Nyquist | Standard starting point. Must be verified on real PSDs. |

## 8. Output language

Detections are "events". The words "trapping" and "translocation" are used only where the
experimental design validates them. Model attributions, when implemented, will describe
regions "contributing to the model prediction", not causes.

## 9. Not implemented (see design specification)

ACF decay fits, PDF/CDF and censoring-aware dwell estimators, baseline/detrending candidate
comparison with synthetic-event recovery scoring, a holdout test set and its access ledger
(no replicate groups exist to hold out), self-supervised pretraining (its gate is not met by this
dataset), generative augmentation, automatic feature selection on the pooled matrix, chunked
processing of 1 MHz recordings, and the FDTD/deconvolution/free-energy analyses of design section
13. Model A on 90 features in 10 groups, Model B, metric learning, the prototype head, the stacked
fusion head and the blend, explainability, few-shot, semi-supervised pseudo-labelling, novelty,
clustering, the feature ablation and redundancy report, figures, and the PDF report are implemented
and documented below. `docs/LEARNING_STRATEGIES.md` records each absence with its reason.

## 10. Model A (feature-based classifiers)

**Features** (`trapsig.features`, label-blind; computed from the processed signal and its
rolling-median baseline at the analysis rate). One 20 s file yields **90 columns in 10 groups** at
the shipped settings (5 PSD bands, 3 ACF lags); `spec`, `tf` and `acf` grow with the number of
declared bands and lags. Every column carries its group's prefix, so group ownership is
unambiguous and `training.feature_groups` removes exactly the columns of the groups it omits.

| Group | Columns | Content |
|---|---|---|
| `noise` | 3 | robust residual sigma (MAD-based), log10 raw standard deviation, raw mean level |
| `time` | 9 | zero-crossing rate, Hjorth activity/mobility/complexity, line length, Teager energy, permutation entropy, segment variability, drift |
| `amp` | 9 | RMS, peak, peak-to-peak, crest factor, percentiles, dynamic range in dB, relative to raw level and to residual sigma |
| `event` | 15 | count, rate per minute, log10 duration (median, mean, max), duty fraction, dip depth `delta_I/I0` statistics, relative amplitude statistics, censoring flags |
| `ctx` | 16 | context and temporal dependence: inter-event interval median and CV, burst fraction and longest burst, first/last onset position, onset spread, Rayleigh R of onset phases, rate drift across the recording, amplitude-duration Spearman coupling, baseline stability, log10 reference level, noise-over-I0 |
| `spec` | 13 | log10 Welch band power per configured band, log10 total power, log-log slope, peak frequency, spectral centroid, spread, entropy, roll-off, harmonicity |
| `tf` | 12 | STFT spectral flux, per-band energy drift across frames, frame entropy, db2 wavelet energy per level and its drift |
| `acf` | 3 | normalised residual autocorrelation at 0.1, 1, and 10 ms |
| `shape` | 3 | skewness, excess kurtosis, fraction of residual beyond 3 sigma |
| `morph` | 7 | turning-point density, event rise and fall asymmetry, area-to-peak ratio, sharpness, self-similarity to this recording's mean event waveform |

A feature that cannot be computed — a recording with no events, a signal too short for a Welch
segment — is **NaN with a reason**, never a silent zero and never an exception. A column that is NaN
for every file in the pool is dropped before modelling and listed in
`summary.json:features_dropped_all_nan`.

**Feature selection, ablation and redundancy.** Groups are a declared choice
(`training.feature_groups`, default `all`), made from the signal's characteristics; nothing selects
features by test score. Their contribution is *measured* instead, by `trapsig.ablation`:

- `training.ablation` refits `training.ablation_model` on the all-groups baseline, on each group
  alone (`only:<group>`), and on everything minus one group (`without:<group>`), using **the same
  folds, the same files and the same model settings** as the main cross-validation, so the
  differences are paired comparisons. `only:<g>` asks whether a group carries the signal on its own;
  `without:<g>` asks whether it adds anything to the rest. `--quick` sets `ablation: none`.
- The ablation never changes the headline result: the models are scored on the full declared feature
  set, and no column is dropped on the strength of an ablation score, because selecting on the
  pooled matrix would use test-fold information. No p-value is claimed.
- `training.redundancy_threshold` (default 0.98) flags pairs of features whose |Spearman| on the
  pooled per-file matrix reaches the threshold. A pair is scored only on files where both features
  are finite, and a pair with fewer than 5 such files is left out rather than reported as
  uncorrelated. Flagged pairs are **reported and never dropped**; the decision is left to the
  reader with the evidence in front of them.
- Cost: about 115 ms per 20 s file at the 10 kHz analysis rate for all 90 columns, against roughly
  2.8 s per file to load and decimate a 1 MHz recording. Feature extraction is not the bottleneck,
  so no group is dropped for speed.

Results are written to `ablation.csv`, `feature_redundancy.csv`, `summary.json`
(`features`, `ablation`, `feature_redundancy`), and PDF section 10 with its twin
`report_ablation.csv`.

**Analysis rate and filtering.** Recordings are analysed at 10 kHz (decimated from 1 MHz with
the anti-alias stage). The filter mode is `none`, so no extra low-pass or Gaussian smoothing is
applied. The 50 Hz notch configured for mains is applied at the original rate, as in the event
pipeline.

**Splits.** Each file is its own group, because no replicate or session metadata exist. The
repeated grouped stratified folds come from `prepare` (five repeats of five folds in the shipped
configuration). There is no holdout, because no replicate groups exist to hold out.

**Models.** Three models are trained, as listed in design section 11: logistic regression (L2),
random forest, and histogram gradient boosting (scikit-learn; LightGBM is not installed). Logistic regression tunes C over
`{0.01, 0.1, 1, 10}` by inner grouped CV, scored by log loss, inside each outer training fold
only. Median imputation and standardisation are fitted within each training fold. Two baselines
are reported: majority class and stratified random guessing.

**Class imbalance.** The ratio of the largest to the smallest class is reported. Design section
11.4 applies no correction below a ratio of 1.5 and class weights for linear and tree models
above it. The current dataset has a ratio of 1.41, so no class weights are used.

**Metrics.** All metrics come from the metric engine (section 6). Per-repeat values are reported
as mean and SD across repeats. The aggregated file-level prediction is the argmax of the mean
probability over repeats; every file is out of fold in every repeat, so this is still
out-of-fold.

**Uncertainty.** The bootstrap is file-level: files are resampled with replacement from the
aggregated predictions. A hierarchical bootstrap would need replicate groups, which do not exist,
so the summary labels the interval as file-level.

**Null model.** For each real model, class labels are shuffled across files and the full CV is
rerun, `n_permutations` times. The p-value is `(1 + #{null >= observed}) / (1 + n)`. With 20
permutations, the smallest attainable p-value is 1/21 ≈ 0.048.

**Controls (synthetic data only; these test the code, not the biology).** Both were run through
`trapsig prepare` and `trapsig train` with 60 files (30 per class), 0.5 s at 100 kHz, five repeats
of three folds.

- Positive control (classes differ in event depth and duration): logistic regression 0.983,
  random forest 0.990, histogram boosting 1.000 balanced accuracy; majority baseline 0.500.
- Negative control (class labels shuffled across files, features unchanged): balanced accuracy
  0.37 to 0.46 for the three models, with permutation p-values 0.67 to 0.83; stratified baseline
  0.497. The majority baseline reads 0.407 rather than 0.5 because the majority class changes
  between training folds, so the constant prediction changes per fold.

**Outputs.** `runs/train_<timestamp>/`: `features.csv` (one row per file, with status and reason
for skipped or failed files), `predictions.csv`, `metrics.csv`, `summary.json`,
`confusion_<model>.csv`, `ablation.csv`, `feature_redundancy.csv`, `config_used.yaml`, and the
preflight report under `reports/`.

**Limits.** Results are optimistic under session or batch effects that file-level grouping
cannot see. The raw level feature (`noise_raw_level`) can carry batch information; compare
results with and without it before interpreting a model as capturing biology.


## 11. Model B (deep 1D CNN), metric learning, and fusion

**Windows.** Each detected event yields one fixed-length window of the analysed signal, cut so
that `pre_event_fraction` of it is pre-event baseline. The length is `deep.window_ms`. With
`auto` it is derived from the event-duration distribution of the pool (median duration times
`window_ms_auto_multiplier`, clamped to `[window_ms_min, window_ms_max]`), because a length
invented by hand would not transfer between experiments. Windows whose extent would leave the
recording are excluded and counted; they are never padded, because a padded constant segment is
something a network can latch onto.

**Scaling.** Windows are baseline-subtracted at cut time. The robust scale (median and IQR over
the pooled training windows) is fitted inside each outer training fold and applied unchanged to
the validation and test windows. Values are clipped to +/-8 so one outlier cannot dominate a fold.

**Architecture.** `stem (conv, stride) -> n_blocks residual dilated conv blocks (dilations
1, 2, 4) -> global average pooling -> linear`. The parameter count and the receptive field in
samples and milliseconds are logged per fold. The head being a single linear layer on the GAP of
the final convolutional map is the condition under which Grad-CAM is mathematically valid, which
is why section 12 labels its output `gradcam` and not something looser.

**Training protocol.** AdamW, linear warmup then cosine decay to zero, gradient clipping. Early
stopping and checkpoint selection use a validation split carved from the outer **training** fold
by file group, so all windows of one file land on the same side and the test fold is never
consulted. The split is class-stratified, and when a fold is too small for
`StratifiedGroupKFold` the fallback takes one file per class before filling its budget at random;
if a class is still missing and has at least two files on the training side, one file is moved
across. Four things need both classes on the validation side — the selection metric (macro F1 and
balanced accuracy are degenerate on one class), the temperature fit, the class prototypes, and the
fusion head — and the repair never removes a class from the data the encoder trains on. The selection metric is fixed in configuration before running. Class-weighted
cross-entropy is used only at or above the imbalance threshold of design 11.4. Temperature
scaling is fitted on the validation split only, then applied to the test fold. Seeds are set for
NumPy and PyTorch and deterministic algorithms are enabled where available; the note on what was
and was not deterministic is recorded in the summary.

**Metric learning.** The cross-entropy can be accompanied by an auxiliary term on the same GAP
embedding that the head consumes (`deep.metric_loss`):

- `supcon` — supervised contrastive: every embedding is pulled towards the other embeddings of its
  own class in the batch and pushed away from the rest, on the L2-normalised embedding at
  `deep.metric_temperature`. A sample is never its own positive, and a batch with no usable pair
  (one window, or no same-class partner) contributes zero rather than NaN.
- `prototypical` — batch prototypes: one mean embedding per class in the batch, and each sample is
  asked to be closest to its own.
- `none` — cross-entropy only.

The weight is `deep.metric_loss_weight` (default 0.2) and the per-epoch value is logged as
`train_metric_loss` in `logs/deep_training_log.jsonl`. Because the term acts on the same vector the
prototype head, the fusion head, few-shot, novelty detection and clustering consume, one objective
shapes the representation for every strategy instead of each of them re-learning one.

**Window to file.** Predictions are aggregated to the file level by the rule fixed in
configuration (`mean_proba` or `majority_vote`). A file with no usable window has no Model B
prediction: its row is NaN, it is excluded from Model B metrics, the count is reported, and the
hybrid falls back to Model A for that file with `hybrid_source = model_a_only` in
`predictions.csv`.

**One encoder, four readings.** The fold's frozen embedding is used by four scored models, which
is what makes the learning strategies one architecture rather than four independent ones:

| Model | Head | Fitted on | Config |
|---|---|---|---|
| `cnn1d` | linear on GAP, temperature-scaled | outer-train windows minus the inner-validation files | `deep.enabled` |
| `proto_cnn` | nearest class prototype on the L2-normalised embedding | prototypes = mean embedding per class over the fold's **inner-validation files** | `deep.prototypical_head`, `deep.proto_distance` |
| `fusion` | logistic regression on [robust-standardised handcrafted file features, mean file embedding] | the fold's **inner-validation files** only; the test fold contributes to neither the head nor the standardisation | `deep.fusion: stack` or `both` |
| `hybrid` | fixed convex blend `model_a_weight * Model A + (1 - model_a_weight) * cnn1d` | nothing: the weight is declared before evaluation | `deep.fusion: blend` or `both`, `deep.model_a_weight` |

The inner-validation files are the right fitting data for both extra heads: the encoder was
early-stopped on them but never trained on them, so their embeddings are out-of-sample for it, and
the outer test fold contributes nothing. The stacked head costs one logistic regression per
(repeat, fold) because the embeddings already exist — no second network and no nested CV of both
models. `deep.fusion: both` reports the blend and the stacked head side by side, pre-declared, so
the reader sees which combination this dataset supports; the comparison framework and gate G7 are
unchanged by the choice.

A file counts as prototyped or fused only where **every** repeat scored it. Files scored in some
repeats only are excluded from that model's metrics, and the exclusion is reported
(`hybrid.run_deep_branch`, `hybrid.fusion_stack`), because averaging a file in over the repeats that
could score it would mix two different evaluation sets.

The same embedding, averaged to one vector per file, is also what the conditional analyses of
section 13 consume: few-shot prototypes, confidence-aware pseudo-labelling, novelty distances, and
clustering. None of them trains a second encoder.

**Comparison and gate G7.** Because Model B scores a subset of files, Model A is *also* scored on
exactly that subset, and the gate is decided there. With overlapping 95% bootstrap intervals the
verdict is "indistinguishable at this sample size", and the design's preference for the cheaper,
interpretable Model A applies. The subset size is always reported with the verdict.

**Cost.** Each permutation of the labels costs one full repeated CNN cross-validation, so
`deep.n_permutations` defaults to 0 and the reason is recorded rather than the null being
silently absent. The prototype and fusion heads are effectively free: they reuse the embeddings the
folds already produced. The ablation of section 10 refits only `training.ablation_model`, so its
cost is that of one cheap model per variant.

`docs/LEARNING_STRATEGIES.md` is the audit of every learning strategy in the codebase: what is
implemented, what flows between the components, which of them contribute to the final prediction,
and what is deliberately absent with the reason.

## 12. Explainability

Explanations run after the model is frozen and are never used to select a model.

- **Model A.** Logistic regression gets exact linear SHAP (coefficients times centred,
  standardised features). For the binary case scikit-learn stores one coefficient row; it is
  expanded to two per-class rows by negation, which is exact. Tree models get TreeSHAP with an
  interventional background drawn as a spread over pool rows, and only when the optional `shap`
  package is installed; otherwise the entry states `not_applicable` with that reason.
- **Model B.** Grad-CAM (valid here, see section 11), integrated gradients over a straight path
  from a zero baseline, and occlusion with a flat background. Attributions are written as an
  `.npz` plus a segment-level CSV, because per-sample CSVs at window length are unreadable.
- **Shortcut probes (mandatory).** (i) A baseline-only probe replaces everything after the
  pre-event segment with that window's own pre-event mean and reruns the whole CNN CV; accuracy
  above chance means the label is carried by acquisition structure, not by events. (ii) A
  time-shift null checks whether the attribution peak moves when the window is shifted. Both are
  reported with an interpretation string.

## 13. Conditional analyses (few-shot, semi-supervised, novelty, clustering)

All gated in configuration and off by default. All consume the **out-of-fold** GAP embeddings
that Model B already produced during cross-validation, so no file is represented by a network
that trained on it. That embedding is the same one the metric-learning objective of section 11
shaped and the same one the prototype and fusion heads read, which is why these analyses are
statements about the deployed representation rather than about a separate model trained for the
occasion. They are reported as analyses, not folded into the headline prediction: with every
recording labelled, pseudo-labelling and clustering have no unlabelled corpus to add, and using
the labels twice would inflate the result they are meant to qualify.

- **Few-shot** uses setting (b) of design 11.7: classes are known and the few labelled examples
  come from files the encoder did not train on. Support and query files are disjoint. Episodes
  overlap in their examples, so episode spread is not independent variance; the interval is a
  file-level bootstrap. A K that a class cannot supply is reported as SKIPPED with the reason.
- **Semi-supervised** is inductive exactly as design 11.8 specifies: the unlabelled pool is
  windows from the *same training files* with their labels hidden. Acceptance requires max
  probability >= tau and top-1 minus top-2 margin >= delta; an accepted set skewed beyond 95%
  towards one class is rejected; each round retrains from scratch. The reference is
  supervised-only training on the identical labelled subset and folds.
- **Novelty** cannot be evaluated on a dataset with no unknown class, and says so.
  Leave-one-class-out is provided and labelled a diagnostic only.
- **Clustering** is fitted on training groups of repeat 0 only, with bootstrap Jaccard stability;
  ARI and NMI against the labels are diagnostics, never a selection criterion.

## 14. Reporting

Figures are written with a headless backend at the configured DPI. Scientific figures (trace with
detected events, class-mean event window, band power by class) are kept separate from model
figures (confusion, ROC, calibration, learning curves, comparison with bootstrap intervals,
embedding PCA, attributions). Class means are taken over files first, so one recording with many
events cannot dominate its class.

The PDF states what was measured, on how many files, with what uncertainty, then what cannot be
concluded: a limitations table pairing each limitation with its consequence and the safeguard
actually applied, an information-gap table pairing each missing input with what would close it,
and a reproducibility block with versions, fingerprints, seeds, and environment. Sections that
could not be computed carry their reason instead of an empty table. `reportlab` is optional;
without it the PDF step reports `skipped` with the install command and the CSV rollups and
figures are still written.

**Table cells have hard boundaries.** Every cell in the PDF is a `Paragraph` in a table with
fixed column widths, so long text wraps inside its own cell and the row grows taller instead of
running into the next column or off the page. Sections 7, 8 and 10
Sections 7, 8 and 10 (limitations, information gaps, feature ablation and redundancy)
use three fixed columns of 52 + 60 + 60 mm, which is exactly the width of the text frame, with
`splitByRow=True` and `splitLongWords=1`: a row too tall for the remainder of a page continues on
the next page as whole rows, and an unbreakable token is broken at the boundary rather than
overlapping it. No cell content is truncated anywhere.

**Every PDF section has a CSV twin with the same content.** One CSV per section is written into
the run directory, in the same order as the page: `report_data_landscape.csv` (§1),
`report_methods.csv` (§2), `report_results.csv` (§3, model rows plus the gate G7 row and the
permutation-null columns), `report_figures.csv` (§4), `report_explainability.csv` (§5),
`report_advanced.csv` (§6), `report_limitations.csv` (§7), `report_information_gaps.csv` (§8),
`report_reproducibility.csv` (§9), `report_ablation.csv` (§10: one row per feature group, per
ablation variant, and per flagged redundant pair, plus the protocol and policy notes). Each section
heading in the PDF names its twin. The CSV keeps the full un-wrapped text of every cell, so anything
that spans several lines on the page is present in one field in the CSV; `report_models.csv` remains
the cross-section model rollup, and it now carries the `proto_cnn` and `fusion` rows alongside
Model A, `cnn1d`, `hybrid`, and the baselines.

## 15. The single-command pipeline

`trapsig run` executes preflight, prepare, train, explain, advanced analyses, figures, and
reporting in one process. The downstream stages run inside the training process because they need
the window set and feature table that training already built; rebuilding them would mean reading
every recording twice. `--dry-run` stops after preflight and prints a compute budget that is
**measured** on this machine: one or two real recordings are timed for the collection cost and one
CNN training batch is timed at the resolved window length for the deep cost, each reported with
its basis and a caveat. `--quick` reduces repeats, permutations, trees, iterations, and epochs and
disables explainability and reporting, and says so on the console.

## 16. The per-run metadata record

`metadata.csv` and `metadata.json` are written into **every** run directory by `dry-run`,
`prepare`, and `train` (`trapsig.metadata`). They exist because the split is re-randomised on each
run: a metadata file written once would describe a split that no longer exists, so the record has
to travel with the run it describes. A run folder is therefore self-contained — the split, the
per-file outcome, and the run-level provenance can all be read back from it without re-running
anything.

`metadata.csv` has one row per **source** file (every discovered file, including files outside the
CV pool and files that were skipped), with the manifest identity fields (`sample_id`, `filename`,
`original_path`, `source_subdir`, `name_annotation`, `class`, `label_source`, `group_id`,
`group_level`, `replicate_id`, `session_id`, `file_sha256`, `size_bytes`, `n_samples`,
`duration_s`, `sampling_rate_hz`, `quality_flags`, `split`, `symlink_path`, `link_status`), then:

| Column | Meaning |
|---|---|
| `in_cv_pool` | whether the file entered the cross-validation pool at all |
| `status`, `reason` | this run's per-file outcome (`ok`, `skipped`, `failed`) and why |
| `n_events`, `n_windows` | detected events, and event windows usable by Model B |
| `model_b_scored` | whether the CNN produced a prediction for this file |
| `fold_rN` | the fold the file belongs to in repeat N |
| `role_rN` | `test` — the file is the held-out file when that fold is evaluated |
| `trains_in_folds_rN` | the folds of repeat N whose training set this file belongs to |
| `cnn_inner_val_rN` | the folds of repeat N whose inner validation split drew this file |

Under K-fold cross-validation every pool file is tested exactly once per repeat, which is what
`role_rN = test` states; `trains_in_folds_rN` is its complement and is what makes the train/test
membership of each of the `n_repeats x n_splits` models explicit. Files outside the pool get empty
fold columns and `role_rN = not_in_pool`.

The inner validation split of Model B (early stopping and temperature scaling) is carved from each
outer training fold at train time by file group, so it is not a property of the file alone; it is
recorded per outer fold in `cnn_inner_val_rN` and per epoch in `logs/deep_training_log.jsonl`.
`write_train_metadata` rebuilds the record from the prepared run's `master_manifest.csv` and
`fold_assignments.csv`, so the training run's metadata always agrees with the split it trained on.

`metadata.json` records the run-level facts: `run_dir`, `stage`, `written_at_utc`,
`config_fingerprint`, `seed`, `n_repeats`, `n_splits`, `n_source_files`, `n_cv_pool`,
`n_ignored_by_glob`, `class_counts`, the grouping policy (`grouping_level`, `grouping_note`),
`holdout`, an explicit `split_randomisation` statement, the `role_convention` above, and for a
training run the `prepared_run` it read the split from. Metadata files are written with
exclusive-create semantics: an existing record is never silently overwritten.


## 17. Hyperparameter optimization (nested, model-specific)

Each Model A search (`trapsig.optimize.tune_model`) runs inside one outer training fold: the
candidate settings are scored on grouped, class-stratified inner splits of *that fold's files
only*, so no hyperparameter that predicts a test file was chosen with knowledge of it. Spaces are
per model (logreg: `C`; rf: `n_estimators`, `max_depth`, `min_samples_leaf`, `max_features`; hgb:
`max_iter`, `learning_rate`, `max_leaf_nodes`, `min_samples_leaf`, `l2_regularization`) and their
upper bounds follow the configuration, so the budget remains the user's decision. `random` draws
without replacement; `halving` (eta=3) evaluates every trial on a file-level subsample and
promotes the best third, falling back to random with a recorded reason when the fold holds fewer
than six files. The objective (`training.hpo_metric`) is always oriented higher-is-better; log
loss is negated and `roc_auc_ovr` is undefined on one-class splits, where the trial is recorded as
failed with a reason instead of being scored 0.5. A trial scores the same pipeline the fold fits
(imputer + robust scaler + estimator), never a lookalike. Every trial is one row of
`hpo_trials.csv`; `summary["hpo"]` and `hpo_trials.png` report the spread, the failures and the
per-fold choice. Permutation nulls inherit the identical search, so tuning cannot inflate the
p-value; the ablation refits with `tune=False` and states that its variants are therefore not
confounded by tuning noise.

Model B's search (`deep.search_fold_params`) treats one full `train_fold` as one trial, scored by
`deep.select_metric` on the same inner-validation files that early stopping uses; the winner is
retrained once, so each fold keeps exactly one checkpoint and one log. Learning rate, dropout and
weight decay are searchable; the embedding width, depth, kernel and window length are not, because
the width is the representation contract shared by the prototype head, the fusion head and the
conditional analyses.

## 18. Inference from saved artifacts

With `training.save_models` (default true), every fitted Model A fold pipeline is written to
`models/<name>_<foldtag>.joblib` beside a JSON sidecar naming its feature columns, classes,
hyperparameters, trapsig version and the config fingerprint of the run, plus `models/INDEX.json`.
`trapsig predict --run RUN --model NAME [--fold-tag TAG] (--files ... | --input-dir DIR)` loads
the run's own `config_used.yaml` (so features are built exactly as in training), refuses a model
without a sidecar or from another version (`--allow-stale` overrides, and is reported), builds the
feature vector of each new recording, and averages the saved fold models' probabilities. Skipped
or unreadable files stay in the output CSV with their reason, exactly as in training.

## 19. Visualization subsystem and the figure index

`trapsig.viz` draws eighteen categories (`viz.FIGURE_CATEGORIES`): dataset composition, signal
examples, class-mean window, PSD by class, feature distributions, missing values, feature
correlation, deep learning curves, confusion, ROC, calibration, model comparison, embedding PCA,
attributions, feature importance, ablation, HPO trials and error analysis. Every category ends in
one row of `figures_index.csv` with status `generated`, `skipped` (+reason), `failed`
(+exception) or `not requested` (+reason), so a figure that was not drawn is a statement in the
run record rather than a silence. The data-oriented figures write CSV twins carrying the exact
numbers the PNG shows; tall tables are capped in the figure and complete in the CSV. A figure
failure is caught, recorded and never aborts training.
