# Learning strategies: audit, architecture, and how they combine

This document answers three questions in order:

1. **What is implemented?** An audit of every learning strategy in the codebase, with the file and
   function that implements it and the artifact that shows its result.
2. **Is it one architecture?** How the strategies interact, what flows between them, and which of
   them contribute to the final prediction — as opposed to being reported next to it.
3. **What is deliberately absent?** Strategies that were considered and rejected, with the reason,
   so the absence is a decision on the record rather than an oversight.

It is the companion to `docs/METHODS.md` (the measurement protocol) and to PDF section 10 of every
run (feature ablation and redundancy).

---

## 1. Audit: what the codebase implements

Status key: **SCORED** = produces a row in the model comparison and therefore a candidate final
prediction. **CONDITIONAL** = runs only when its precondition holds, and is reported as an analysis
rather than as a prediction. **ABSENT** = not implemented, with the reason in section 4.

| # | Strategy | Status | Where | What it produces |
|---|----------|--------|-------|------------------|
| 1 | Conventional ML on handcrafted features (Model A) | SCORED | `train.py` (`cv_predictions`, `_base_model`), `features.py` | `logreg`, `rf`, `hgb` rows: balanced accuracy, macro F1, ROC AUC, bootstrap CI, permutation p |
| 2 | Deep learning / ConvNets (Model B) | SCORED | `deep.py` (`CNN1D`, `train_fold`, `cv_windows`) | `cnn1d` row; per-fold checkpoints; per-epoch log |
| 3 | Representation learning | SCORED (shared) | `deep.py` (`CNN1D.embedding`), `advanced.py` (`file_representation`) | one GAP embedding per window per fold, averaged to one vector per file; consumed by 4–10 |
| 4 | Metric learning — training objective | SCORED (shapes 3) | `deep.py` (`metric_loss`) | supervised-contrastive or batch-prototype auxiliary loss added to the cross-entropy |
| 5 | Metric learning — inference head | SCORED | `deep.py` (`build_prototypes`, `prototype_probabilities`), `train.py` | `proto_cnn` row: nearest-prototype classification of the same frozen embedding |
| 6 | Feature + representation fusion | SCORED | `hybrid.py` (`fusion_stack`, `combine_probabilities`, `hybrid_predictions`) | `fusion` row (stacked head) and `hybrid` row (probability blend) |
| 7 | Few-shot learning | CONDITIONAL | `advanced.py` (`few_shot_prototypical`) | K-shot nearest-prototype accuracy with a file-level bootstrap CI, per K |
| 8 | Semi-supervised learning + confidence-aware pseudo-labelling | CONDITIONAL | `advanced.py` (`semi_supervised`) | teacher vs pseudo-labelled student, accepted-label count, τ/δ thresholds used |
| 9 | Clustering (optional) | CONDITIONAL | `advanced.py` (`clustering_analysis`, `_cluster_stability`) | ARI/NMI vs labels, bootstrap Jaccard stability, per k |
| 10 | Novelty / out-of-distribution detection | CONDITIONAL | `advanced.py` (`novelty_detection`) | known-class distance scores; leave-one-class-out AUROC as an explicit diagnostic |
| 11 | Baselines and negative controls | SCORED | `train.py` (`BASELINE_MODELS`, permutation nulls) | `dummy_majority`, `dummy_stratified`, label-shuffle p-values |
| 12 | Explainability | reported | `explain.py` (`gradcam_1d`, `integrated_gradients_1d`, `occlusion_1d`, `feature_attributions`), `advanced.py` shortcut probes | temporal attributions per window, per-class top features, baseline-only probe, time-shift null |

Every row is reproducible from one command (`trapsig run --config configs/custom_dat.yaml`) and
every number that could not be computed is written as a status plus a reason, never as a blank.

---

## 2. One architecture, not ten models

The design rule is: **one encoder per outer fold, one representation, many readings of it.** Nothing
downstream re-learns a representation, and nothing downstream is fitted on a file the encoder was
trained on.

```
                                   20 s .dat recording, 1 MHz
                                            |
                       ingest -> decimate -> filter -> event detection
                                            |
                        +-------------------+--------------------+
                        |                                        |
              analysed signal, per file                   detected events
                        |                                        |
        handcrafted features (10 groups,               windows cut around events
        ~84 columns per file, features.py)             (windows.py, baseline-subtracted,
                        |                               robust-scaled on TRAIN windows only)
                        |                                        |
                        |                              1D residual dilated ConvNet
                        |                              (deep.py: stem -> blocks -> GAP -> head)
                        |                               trained per (repeat, outer fold)
                        |                               cross-entropy  +  metric loss
                        |                               (supcon | prototypical, weight w)
                        |                               early stop + temperature scale
                        |                               on the INNER-VALIDATION files
                        |                                        |
                        |                          GAP embedding E  (one vector per window)
                        |                                        |
                        |            +--------------+------------+------------+--------------+
                        |            |              |                         |              |
                        |      softmax head   mean per file            inner-validation   frozen E
                        |            |              |                   embeddings E_va       |
                        |        cnn1d (3)     proto_cnn (5)                |              few-shot (7)
                        |            |              |            prototypes = mean         pseudo-labels (8)
                        |            |              |            L2-normalised E_va        novelty (10)
                        |            |              |                     |                clustering (9)
                        |            |              |              fusion head (6)          explainability (12)
                        |            |              |        LogisticRegression on
                        +------------+--------------+----- [robust-standardised handcrafted
                                     |              |        features, mean E_va]  fitted on
                                     |              |        the inner-validation FILES only,
                                     |              |        applied to the outer test files
                                     |              |                    |
                                     v              v                    v
                                  hybrid (6) = w * P(Model A) + (1 - w) * P(cnn1d)
                                     |
                     model comparison on a matched subset + gate G7 (hybrid.py)
                                     |
              metrics.csv, predictions.csv, model_comparison.csv, report.pdf sections 3 and 10
```

### What flows between the components

| From | To | Payload | Frozen? |
|------|----|---------|---------|
| `features.py` | Model A, fusion head, explainability | per-file handcrafted features (10 groups) | label-blind, computed once per file |
| `windows.py` | `deep.py` | per-event windows, baseline-subtracted | scaler fitted on train windows of the fold only |
| `deep.py` encoder | everything below | GAP embedding `E` per window | frozen after the fold's early stopping |
| metric loss | encoder weights | gradient that shapes `E` (same-class pull, other-class push) | active during training only |
| `E` on inner-validation files | `proto_cnn`, `fusion` | prototypes; fusion-head training rows | the encoder never trained on these files |
| `E` averaged per file | few-shot, pseudo-labelling, novelty, clustering | `Representation` (`advanced.file_representation`) | out-of-fold by construction |
| Model A probabilities + `cnn1d` probabilities | `hybrid` | fixed convex blend, weight declared in the config | no free parameters, so nothing can be fitted on test data |
| handcrafted features + `E` | `fusion` | one logistic regression per (repeat, fold) | fitted on inner-validation files only |

### Which strategies contribute to the final prediction

* **Contribute (SCORED rows):** Model A (`logreg`, `rf`, `hgb`), `cnn1d`, `proto_cnn`, `fusion`,
  `hybrid`. All five families are compared on the same files in `model_comparison.csv`, and gate G7
  decides on a matched subset whether the deep branch is distinguishable from the feature branch at
  this sample size. The reported "final" model is whatever this comparison supports — the framework
  does not silently prefer the most complex option.
* **Do not contribute, by design (CONDITIONAL rows):** few-shot, semi-supervised pseudo-labelling,
  novelty detection and clustering. They answer questions about the representation ("would K labels
  per class be enough?", "would unlabeled windows help?", "is a class separable from a held-out
  one?", "does the embedding have label-independent structure?"). Folding their outputs into the
  headline prediction would either need unlabeled data this dataset does not have, or would use the
  labels twice. They are reported in PDF section 6 with their preconditions, and each is switched on
  only when its precondition holds.

That distinction is the honest answer to "how do their outputs contribute to final classification":
four of them do, through the fusion and comparison framework; the rest qualify the representation
those four are built on.

---

## 3. Leakage-safety contract

Each component is fitted on data that is out-of-sample for the thing it is asked to score. This is
the property that makes the whole architecture one pipeline rather than a set of optimistic numbers.

| Component | Fitted on | Never sees | Evidence in the run folder |
|-----------|-----------|------------|----------------------------|
| Robust window scaler | train windows of the outer fold | validation and test windows | `design_notes.window_scaling` in `summary.json` |
| Encoder (per fold) | outer-train windows minus inner-validation windows | inner-validation and outer-test files | `logs/deep_training_log.jsonl`, `checkpoints/cnn1d_r*f*.pt` |
| Early stopping + temperature | inner-validation files, drawn class-complete where the fold allows it (`deep.validation_split`) | outer-test files | `temperature_mean`, `select_metric`, `best_epoch` in the fold log |
| Prototypes (`proto_cnn`) | inner-validation file embeddings | outer-test files | `models.proto_cnn.prototypes_from` in `summary.json` |
| Fusion head (`fusion`) | inner-validation files: handcrafted features + mean embedding | outer-test files (also excluded from its standardisation) | `models.fusion.fitted_on`, `n_val_file_fits` |
| Blend (`hybrid`) | nothing (weight declared in the config) | — | `design_notes.hybrid_combination` |
| Model A imputer / C grid | inner folds of the outer-train files | outer-test files | `models.logreg.chosen_C`, `design_notes.tuning` |
| Feature ablation | the same outer folds as the headline CV | no extra data at all | `ablation.csv`, PDF section 10.2 |
| Redundancy report | pooled features, **labels never used** | — | `feature_redundancy.csv`, PDF section 10.3 |
| Few-shot episodes | support files disjoint from query files, both out-of-fold for the encoder | — | `advanced_analyses.json:few_shot_results` |
| Pseudo-labels | teacher trained on a labelled subset of repeat-0 train windows; accepted labels come from the same train files | outer-test files | `advanced_analyses.json:semi_supervised_results` |
| Clustering | train-fold embeddings; other files assigned, never fitted | test-fold structure | `advanced_analyses.json:clustering_results` |

Two rules follow from this and are enforced in code, not in prose:

* A file is scored by `proto_cnn` or `fusion` only where **every** repeat produced a score for it.
  Partially scored files are excluded from the metrics and the exclusion is reported
  (`hybrid.fusion_stack`, `hybrid.run_deep_branch`). Nothing is averaged over files a model never saw.
* Redundant features are **reported and never dropped**. Dropping a column on the pooled matrix
  would use information from every file, including the test folds. The evidence goes to
  `feature_redundancy.csv` and PDF section 10.3; the decision is left to the reader.

---

## 4. Feature architecture

`features.py` computes 10 groups from the analysed signal. Groups are selected by signal
characteristics through `training.feature_groups`, and their contribution is measured by
`training.ablation` on the same folds as the headline cross-validation.

Counts are what one 20 s file produces with `configs/custom_dat.yaml` (5 PSD bands, 3 ACF lags):
**90 columns in 10 groups**. `spec`, `tf` and `acf` grow with the number of declared bands and lags.

| Group | Columns | What it captures |
|-------|---------|------------------|
| `noise` | 3 | robust residual σ, raw level and spread — the recording's noise floor |
| `time` | 9 | zero crossings, Hjorth parameters, line length, Teager energy, permutation entropy, segment variability, drift |
| `amp` | 9 | RMS, peak, peak-to-peak, crest factor, percentiles, dynamic range |
| `event` | 15 | count, rate, duration distribution, dip depth, amplitude, ΔI/I₀ |
| `ctx` | 16 | inter-event intervals, burstiness, onset distribution (Rayleigh R), rate drift across the recording, amplitude–duration coupling, baseline stability, reference level |
| `spec` | 13 | Welch band powers, log-log slope, peak, centroid, spread, entropy, roll-off, harmonicity |
| `tf` | 12 | STFT flux, per-band energy drift, frame entropy, db2 wavelet level energies |
| `acf` | 3 | normalised autocorrelation of the residual at `training.acf_lags_ms` |
| `shape` | 3 | skewness, excess kurtosis, tail fraction |
| `morph` | 7 | turning points, event rise/fall asymmetry, area-to-peak, sharpness, self-similarity to the file's mean event waveform |

Rules that keep this honest:

* Every column carries its group's prefix, so "which columns does this group own" is unambiguous and
  disabling a group removes exactly its columns (`features.resolve_groups`, `features.feature_group`).
* A feature that cannot be computed (no events in a recording, a signal too short for a Welch
  segment or for two STFT frames) is **NaN with a reason**, never a silent zero and never an
  exception. A column that is NaN for every file in the pool is dropped before modelling and listed
  in `summary.json:features_dropped_all_nan` and PDF §2 — on the 0.5 s synthetic demo that removes
  seven `tf` columns, because a 5000-sample signal at 10 kHz yields one STFT frame at
  `nperseg=4096`; a 20 s recording yields dozens and keeps all of them.
* An exact duplicate was removed during the audit (`time_mean_abs_diff_norm` was numerically
  identical to `time_line_length_norm`). Remaining near-duplicates are reported per run.
* Cost is measured, not assumed: ~115 ms per 20 s file at the 10 kHz analysis rate for all 90
  columns, against ~2.8 s per file to load and decimate a 1 MHz recording. Feature extraction is not
  the bottleneck, so no group is dropped for speed.

**Fusion of handcrafted and learned features.** `deep.fusion: stack` concatenates the
robust-standardised per-file handcrafted features with the mean per-file embedding and fits one
logistic regression per (repeat, fold) on the inner-validation files. On the 12-file synthetic demo
that is 83 handcrafted + 16 learned inputs fitted on 38 validation-file draws, scoring 11 of 12
files; a fold whose validation files hold one class cannot fit a logistic regression and says so
instead of guessing. `deep.fusion: blend` combines
the file-level probabilities of Model A and `cnn1d` with a declared weight. `both` reports the two
side by side, pre-declared, so the reader sees which combination this dataset supports.

---

## 5. Considered and deliberately absent

| Strategy | Why it is not here |
|----------|--------------------|
| Self-supervised pre-training (SimCLR / masked-signal) on unlabeled windows | There is no unlabeled corpus: all 135 recordings are labelled, and the encoder is refitted per fold, so a pre-trained backbone would have to be built from the same files it is later evaluated on. If unlabeled recordings arrive, the entry point is `deep.train_fold` (initialise from a checkpoint) and the leakage rule is that pre-training must use files outside the evaluation pool. |
| Triplet / N-pair metric learning with hard mining | `metric_loss: supcon` generalises it (all positives and all negatives per anchor, no mining heuristic to tune), and `prototypical` covers the prototype-distance view. Adding a third objective would add a hyperparameter, not information. |
| Generative augmentation (GAN / diffusion / SMOTE on windows) | Fabricating trapping events risks inventing the very morphology the study measures. Window-level jitter and amplitude scaling are the honest alternatives, and they are not enabled by default because with ~135 files the validation signal is too small to tell augmentation from overfitting. |
| Transformer / attention encoder | The ConvNet is ~10⁴ parameters at the default settings, the order the design asks for (§11.3) given 20 s windows and ~135 files. An attention stack would be untestable at this sample size, and Grad-CAM on a dilated conv stack localises an event in time in a way attention maps do not. |
| Multi-task heads (e.g. predicting event count alongside class) | The auxiliary targets are derived from the same detection step that produced the features, so the task leaks the label rather than adding supervision. |
| Stacked combiner fitted by nested CV of both models | Superseded: `deep.fusion: stack` fits a fusion head on the fold's inner-validation files, which is leakage-safe at the cost of one logistic regression per fold instead of a nested CV of both models (~3× CPU). `deep.combine` keeps the declared-weight blend for comparison. |
| Automatic feature selection (RFE / Boruta / L1 pruning on the pooled matrix) | Any selection performed on the pooled feature matrix uses test-fold information. The ablation reports the evidence per group; selection stays a declared choice (`training.feature_groups`). |

---

## 6. Configuration surface

| Goal | Keys |
|------|------|
| Choose which handcrafted features are computed | `training.feature_groups` |
| Measure what each group contributes | `training.ablation` (`none｜groups｜leave_one_out｜both`), `training.ablation_model` |
| See near-duplicate features | `training.redundancy_threshold` |
| Turn the deep branch on/off | `deep.enabled`, `deep.model_set` (`features｜deep｜hybrid`) |
| Shape the representation | `deep.metric_loss` (`none｜supcon｜prototypical`), `deep.metric_loss_weight`, `deep.metric_temperature` |
| Nearest-prototype classifier | `deep.prototypical_head`, `deep.proto_distance` (`cosine｜euclidean`) |
| Combine handcrafted and learned | `deep.fusion` (`none｜blend｜stack｜both`), `deep.combine`, `deep.model_a_weight` |
| Conditional analyses | `advanced.few_shot`, `advanced.few_shot_k`, `advanced.few_shot_episodes`, `advanced.semi_supervised`, `advanced.pseudo_confidence_tau`, `advanced.pseudo_margin_delta`, `advanced.pseudo_rounds`, `advanced.novelty`, `advanced.clustering`, `advanced.clustering_k` |

Every key above is documented in `configs/custom_dat.yaml` with its type, all allowed values, the
default, an example, and the rules that tie it to other keys (including which settings make another
setting an error). `trapsig run --dry-run` validates the whole configuration before a single
recording is read.

---

## 7. Extending the architecture

The contracts below are what keep a new component leakage-safe. A new piece that respects them slots
in without touching the evaluation protocol.

* **A new feature group.** Add the computation to `features.py`, register its name in
  `FEATURE_GROUPS` and its one-line meaning in `GROUP_DESCRIPTIONS`, and prefix every column with
  the group name. It is then automatically selectable (`training.feature_groups`), ablated
  (`training.ablation`), redundancy-checked, and listed in PDF section 10.1. No other file changes.
* **A new reading of the embedding** (a new head on `E`). Compute it inside `deep.train_fold` from
  `embedding_val` (fitting data) and `embedding_test` (scoring data), return it on `FoldResult`,
  aggregate it in `hybrid.run_deep_branch`, and add it as a `ModelEntry` in `train.py`. It inherits
  the comparison framework, the bootstrap CI, the permutation null and the CSV/PDF rows for free.
* **A new fusion rule.** Add the value to `config.FusionRule`, validate it in `config._deep_errors`,
  implement it next to `hybrid.fusion_stack`, and gate it in `train.py` on `cfg.deep.fusion`. The
  rule must state which files it is fitted on, and those files must be out-of-sample for the encoder.
* **A new conditional analysis.** Add it to `advanced.py` with an explicit precondition: if the
  precondition fails, return `{"status": ..., "reason": ...}` rather than a number. Register the flag
  in `AdvancedConfig` and its gate in `config._advanced_errors`, and it will be reported in PDF
  section 6 and `advanced_analyses.json`.
* **Anything that must never happen.** Fitting on the pooled matrix, selecting features by test
  score, averaging over files a model could not score, or reporting a metric without the number of
  files it was computed on.

---

## 8. Where to look in a run

| Question | Artifact |
|----------|----------|
| Which models were scored, on how many files? | `summary.json:models`, `model_comparison.csv`, PDF §3 |
| Did the deep branch beat the feature branch? | `summary.json:comparison.matched_subset.gate_G7`, PDF §3 |
| Which feature groups carry the signal? | `ablation.csv`, `summary.json:ablation`, PDF §10.2 |
| Which features are near-duplicates? | `feature_redundancy.csv`, `summary.json:feature_redundancy`, PDF §10.3 |
| How was the representation shaped? | `summary.json:deep.metric_loss`, `logs/deep_training_log.jsonl` (`train_metric_loss` per epoch) |
| What did the prototype head do? | `summary.json:models.proto_cnn`, `predictions.csv` rows with `model=proto_cnn` |
| What did the fusion head use? | `summary.json:models.fusion` (`n_handcrafted`, `n_embedding`, `fitted_on`, `n_val_file_fits`) |
| Would fewer labels suffice? | `advanced_analyses.json:few_shot_results`, PDF §6 |
| Would unlabeled windows help? | `advanced_analyses.json:semi_supervised_results`, PDF §6 |
| Which part of a window drove a prediction? | `attributions.npz`, `attributions_summary.csv`, `explainability.json`, PDF §5 |
| Which file was in which fold, in this run? | `metadata.csv` (`fold_rN`, `role_rN`, `trains_in_folds_rN`, `cnn_inner_val_rN`) |
