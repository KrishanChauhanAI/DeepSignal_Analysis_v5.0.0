# The four-branch deep architecture (`quad`): design, contract, and honest scope

Status: **implemented, wired into `trapsig train` / `trapsig run`, default OFF** (`quad.enabled:
false`). Everything in this document is additive: with the flag off, every existing module,
artifact, number, and behaviour is bit-identical to before. With the flag on, the run gains extra
scored models, extra artifacts, and four new PDF sections — it never replaces Model A, Model B,
the hybrid, or any existing output.

---

## 1. Specification (what was asked, what was built)

| Requirement | Where it lives |
| --- | --- |
| A. Learned temporal branch: dilated/residual/multiscale CNN + temporal attention + transformer encoder | `quad.TemporalEncoder` (stem k=`t_kernel`, per-channel residual blocks with dilations `t_dilations`, stride-2 downsampling, multiscale head with dilations (1,4,8), `TemporalAttention`, positional encoding, `nn.TransformerEncoderLayer`, mean+max+attention pooling → 3·C) |
| B. Learned spectral branch with a FIXED transform front end | `quad.SpectralEncoder` (fixed Hann STFT `s_nfft`/`s_hop` → `log1p(|·|)` → stride-2 Conv2d stack → frequency attention → mean+max+attention pooling). The STFT is **not learned**; the conv stack and attention **are**. Wavelets are offered as engineered features/registry plots, not as a second learned front end |
| C. Self-supervised pathway: contrastive + masked reconstruction, semantics-preserving augmentations, no held-out data, transfer to supervised | `quad.pretrain_selfsup` (NT-Xent over two augmented views + masked-span MSE reconstruction; trains temporal+spectral params only; runs on training-fold windows minus the inner-validation split; stage 2 of 7) |
| D. Engineered features, window-level and recording-level, registry with names/units/validation, ablation + redundancy, training-only selection | `quad.window_features` / `window_feature_names` (38 columns incl. band powers); recording level = one window per file (`segmentation.window_s: null`); redundancy pairs are reported by the existing `feature_redundancy.csv` machinery and never dropped; quad branch ablation covers the engineered branch (`branch_ablation: ["engineered"]`) |
| Cross-branch fusion with learned gates, safe handling of disabled branches | `quad.CrossBranchFusion` (LayerNorm per token + branch position + CLS token + transformer; softmax gate over branch tokens; disabled/ablated branches simply have no token — gates always sum to 1 over the branches that exist) |
| Head A supervised softmax (CE/focal/class weights, classwise errors) | `quad.QuadModel.head_a`, `quad.focal_ce` (`focal_gamma`, `class_weight: balanced` from training fold only); classwise errors in `metrics_extended.csv` |
| Head B metric learning (prototypical, SupCon, cosine, episodic, few-shot inference) | prototype logits `proto_scale · (emb @ protosᵀ)` on L2-normalised embeddings; `quad.supcon_loss` auxiliary; `quad._episodic_finetune` (stage 5, disjoint support/query); `fewshot.episodic_eval` inference on held-out fold-test FILE embeddings |
| Staged training 1–7 | see §5 |
| Leakage rules (file grouping, split before windowing, train-only transforms, test excluded from SSL) | see §6 |
| Full metric set, few-shot CIs, CSV persistence, undefined metrics explained | `evalmetrics.extended_metrics` → `metrics_extended.csv`; `fewshot.summarize_episodes` (Wilson 95% CI over pooled queries) → `fewshot_episodes.csv`, PDF §12–13 |
| ≥50 registry plot types in 6 categories with prerequisites, PNG + CSV twins, no fabrication | `viz_registry.py` (67 entries, categories A–F), `viz_registry_index.csv` audit trail |
| Sampling-frequency menu, acquisition vs analysis, anti-alias | `sampling.rate_menu_hz`, `allow_off_menu_rates`; PDF §11 |
| Configurable window/overlap/stride/normalize/detrend/filter | `segmentation` block (documented per key in `configs/custom_dat.yaml`) |

## 2. Data-flow diagram

```
                 one .dat recording (folder = class), analysed at training.analysis_rate_hz
                                          │
              split FIRST by file (grouped repeated K-fold) ──► fold assignment per file
                                          │
                    segment_signal (window_s / overlap / stride / pad / detrend / normalize)
                                          │            windows inherit their file's fold
             ┌────────────────┬───────────┴────────────┬────────────────────┐
             │ Branch A        │ Branch B               │ Branch C (stage 2) │ Branch D
             │ TemporalEncoder │ SpectralEncoder        │ pretrain_selfsup   │ window_features
             │ dilated res CNN │ FIXED Hann STFT →      │ NT-Xent + masked   │ 38 engineered
             │ + multiscale    │ learned conv stack +   │ reconstruction on  │ columns →
             │ + attention +   │ frequency attention    │ TRAIN-minus-inner- │ median/IQR impute
             │ transformer     │ (log1p magnitude)      │ val windows ONLY   │ + robust scale,
             │ → 3·C pooling   │ → 3·C pooling          │ (weights transfer) │ TRAIN fold only
             └───────┬────────┴───────────┬────────────┴─────────┬──────────┴───────┬────────
                     └── per-branch linear projections to embed_dim ──► one token per branch ──┘
                                          │
                    CrossBranchFusion (LayerNorm + branch position + CLS + transformer)
                     learned softmax gate over branch tokens (reported: quad.gate_mean)
                                          │
                                  shared embedding (embed_dim)
                     ┌────────────────────┴────────────────────┐
              Head A: softmax (CE/focal,               Head B: prototype head
              optional balanced weights)               (cosine, learned scale; SupCon aux;
              → proba per window                       episodic fine-tune stage 5)
                     │                                         │
        file proba = mean over the file's test windows   few-shot episodes on held-out
                     │                                   fold-test FILE embeddings
        out-of-fold metrics, extended metrics, PDF, CSVs, registry plots
```

Receptive field: `TemporalEncoder.receptive_field_samples()` computes Branch A's receptive field
from `t_kernel`, `t_dilations`, `t_channels` (stride-2 stages) and the multiscale head; the value
is stored per fold in `summary.json → quad.receptive_field_samples`. It is measured from the
architecture, not assumed.

## 3. Why these four paradigms fit nanopore trapping signals

- **Temporal (A):** trapping/translocation events are short, shape-discriminative deflections
  (depth, width, asymmetry, step structure). Dilated residual stacks give a large receptive field
  at low cost; attention and the transformer stage let the model weigh non-local shape context
  (e.g. a blockage preceded by a capture spike).
- **Spectral (B):** class differences also live in band energy and spectral texture (mains residue
  near 45–55 Hz, broadband noise vs resonant structure). A fixed STFT keeps the transform
  interpretable and cheap; the learned stack adapts the readout.
- **Self-supervised (C):** labels are folder-level and files are few (135 in the real dataset).
  Contrastive + masked reconstruction exploits the unlabelled portion of each training fold to
  learn event-invariant representations before any label is used.
- **Engineered (D):** ΔI/I₀-style amplitude statistics, crest/shape/impulse factors, Hjorth
  parameters, band powers and entropy features are the established vocabulary of this signal
  family; they give the fusion a strong, interpretable branch and keep the deep branches honest
  through paired ablation.

The branches are **one architecture**, not four models: one fusion, one embedding, two heads, one
set of folds, one leakage contract.

## 4. Branch D feature inventory (window-level; recording level = `window_s: null`)

> Round 2 update: the inventory below is now generated from the **configurable feature
> registry** (`src/trapsig/featreg.py`) - 154 registered candidates across 11 column families
> (plus "learned representations", the 12th family, which lives in branches A-C instead of
> columns) in three tiers, all documented with name, description, unit and parameters and
> exported per run to `feature_registry.csv`. See section 11 for the full specification.

All names are deterministic (`window_feature_names(bands)`); values are `float` and `NaN` when
undefined for a window (e.g. no peak) — imputation is median/IQR **from the training fold only**.
Units: amplitude features are in the configured signal unit (raw ADC or arbitrary); time features
in seconds; frequency features in Hz; entropies dimensionless.

| Name | Description | Unit | Parameters |
| --- | --- | --- | --- |
| `w_mean`, `w_sd`, `w_median`, `w_mad`, `w_skew`, `w_kurt` | central tendency, spread, shape | signal unit / – | – |
| `w_rms`, `w_p2p`, `w_energy` | RMS, peak-to-peak, sum of squares | signal unit / – | – |
| `w_crest`, `w_shape`, `w_impulse` | crest = peak/RMS; shape = RMS/mean\|x\|; impulse = peak/mean\|x\| | – | – |
| `w_zcr` | zero-crossing rate (mean-removed) | 1/s | – |
| `w_d1_mean`, `w_d1_sd`, `w_d2_sd` | first/second difference statistics | unit/sample | – |
| `w_slope`, `w_turning_rate` | least-squares slope; sign-change rate of the diff | unit/s, 1/s | – |
| `w_hjorth_mobility`, `w_hjorth_complexity` | Hjorth parameters | 1/s, – | – |
| `w_spec_centroid`, `w_spec_bandwidth`, `w_spec_rolloff85`, `w_spec_flatness`, `w_spec_entropy` | spectral shape from the window FFT | Hz / – / nats | – |
| `w_band_<lo>_<hi>` | log10 band power per configured band | log10 unit² | `training.psd_bands_hz` |
| `w_env_mean`, `w_env_sd`, `w_inst_freq_mean` | Hilbert envelope and instantaneous frequency | unit, Hz | – |
| `w_n_peaks`, `w_peak_interval_mean`, `w_prominence_mean` | peak structure | count, s, unit | scipy `find_peaks` defaults |
| `w_perm_entropy3` | normalised permutation entropy | – | order 3 |
| `w_sampen_approx` | approximate sample entropy | – | m=2, r=0.2·sd |

Validation: every column is finite-or-NaN by construction; `validate_quad` checks the config; the
ablation model `quad_minus_engineered` answers "does this branch help **here**" empirically — the
framework does **not** claim every feature helps.

## 5. Training strategy (stages 1–7, what actually runs)

1. **Inspect and split** — existing pipeline: per-file assessment, grouped repeated K-fold by
   file; segmentation happens after splitting.
2. **SSL pretraining** (`pretrain_epochs`, `use_selfsup`) — NT-Xent + masked reconstruction on
   training-fold windows minus the inner-validation split.
3. **Heads/fusion warmup** (`warmup_epochs`) — encoders frozen, Head A + fusion + Head B scale
   train on labelled training windows.
4. **Joint fine-tuning** (`epochs` − warmup) — everything trains; loss = focal/CE + `supcon_weight`·SupCon.
5. **Episodic metric fine-tuning** (`episodes`//4 steps) — disjoint support/query prototype-CE on
   training windows.
6. **Validation-only model selection** — the best checkpoint is chosen by `select_metric` on the
   inner-validation windows of the training fold. *Documented limitation: there is no separate
   hyperparameter search for quad on this run; hyperparameters are fixed in config before
   evaluation.*
7. **Held-out evaluation** — test-fold windows are predicted, averaged per file, and scored
   out-of-fold with the full metric set; few-shot episodes are drawn from test-fold file
   embeddings only.

## 6. Leakage protocol (enforced in code, not in prose)

- Grouping unit = **file**; `fold_ids` are computed before `segment_signal` runs, and every window
  carries its file's fold.
- Median/IQR imputation and the robust window scale are fitted on **training-fold windows minus
  the inner-validation split** (`_imputer_state`, `_apply_imputer` in `cv_quad`).
- SSL pretraining never sees inner-validation or test windows.
- Checkpoint selection uses inner-validation windows only.
- Few-shot episodes sample support and query **disjointly** from held-out test files; the protocol
  string is stored per episode row.
- Class weights (`balanced`) come from the training fold only.
- Ablation variants run on the same folds and files → paired comparison, no re-splitting.

## 7. Evaluation design

`evalmetrics.extended_metrics` adds, per model, from the **same** out-of-fold arrays used by the
headline metrics: per-class precision/recall/**specificity**/F1/support/error counts, multiclass
**MCC**, **Brier**, **ECE/MCE** (`calibration_bins`, formulae stored under
`summary.json → evaluation.definitions`), one-vs-rest **PR-AUC** (`training.pr_curves`),
**top-k accuracy** (`training.report_top_k`), ROC-AUC OvR. Undefined values (e.g. PR-AUC for an
absent class, specificity with no negatives) are written as empty cells **with the reason** — they
are never imputed. Few-shot results carry episode accuracy mean±sd **and** a Wilson 95% CI over
pooled queries, per shot, with the closed-set caveat printed in PDF §13.2.

## 8. Visualization registry (67 entries, six categories)

`viz_registry.REGISTRY` — A time domain (15), B frequency (12), C time-frequency (10, incl. EMD/VMD
recorded `not_available` because the optional dependency is not shipped), D nonlinear/complexity
(9), E dataset/class (9), F deep diagnostics (12). Each entry declares category, title,
prerequisites, parameters and whether it writes a CSV twin; `render` records
`generated | skipped | not_available | failed` with reasons into `viz_registry_index.csv`. PNGs go
to `runs/.../viz_registry/`, CSV twins to `runs/.../viz_registry_csv/`. Nothing is fabricated: a
builder that cannot meet its prerequisite returns `None` and the index says *skipped: <prereq>*.

## 9. Project structure delta (all additive)

```
src/trapsig/quad.py           Branches A–D, fusion, heads, losses, augmentations, SSL, fold CV
src/trapsig/fewshot.py        episodic_eval + summarize_episodes (Wilson CI)
src/trapsig/evalmetrics.py    extended metric surface + metrics_extended.csv writer
src/trapsig/viz_registry.py   67-entry registry, render(), index writer
config.py                     SamplingConfig.rate_menu_hz, SegmentationConfig, QuadConfig,
                              validate_sampling_menu / validate_segmentation / validate_quad
train.py                      quad block after the deep branch: entries["quad"],
                              entries["quad_minus_<branch>"], summary["quad"], fewshot CSV,
                              extended metrics for every scored model
stages.py                     TrainContext.quad_branch, quad figures, _stage_registry
report.py                     PDF sections 11–14 + their CSV twins
cli.py                        _print_outputs lists the new artifacts
tests/                        test_quad.py, test_fewshot.py, test_evalmetrics.py,
                              test_viz_registry.py
configs/custom_dat.yaml       every new key documented with options, examples, RULEs
```

New run artifacts (only when produced): `metrics_extended.csv`, `fewshot_episodes.csv`,
`report_sampling.csv`, `report_extended_metrics.csv`, `report_quad.csv`,
`report_viz_registry.csv`, `viz_registry_index.csv`, `viz_registry/*.png`,
`viz_registry_csv/*.csv`, plus figures `quad_fusion_gates.png`, `fewshot_shot_curve.png`,
`quad_branch_ablation.png`, `quad_embedding_pca.png`, `quad_embedding_tsne.png`, `pr_<model>.png`.

## 10. Unknowns, stated as configurable options (not guessed)

- Whether few-shot transfer helps at all on the real 135-file dataset → run with `quad.enabled:
  true` and read PDF §13.2; the protocol is closed-set and the head existing is not evidence.
- Which branches actually carry the signal → `branch_ablation: [temporal, spectral, engineered]`
  gives paired `quad_minus_*` models on identical folds.
- Window granularity for 20 s files → `segmentation.window_s: null` (whole file) vs 0.5–2 s.
- Analysis rate → `sampling.rate_menu_hz` + `training.analysis_rate_hz`; strictness via
  `allow_off_menu_rates: false`.
- Hyperparameters of quad itself are fixed pre-registered values; a dedicated HPO for quad is a
  documented open extension, not something this run claims to have done.

## 11. Round 2: the configurable feature registry (`featreg.py`) and robustness add-ons

### 11.1 Twelve families, three tiers, 154 registered candidates

Tier 1 = mandatory core, tier 2 = recommended, tier 3 = experimental (costlier, more fragile,
or dependency-gated). Counts are generated live from the module, not typed by hand:

| family | tier 1 | tier 2 | tier 3 | total | what it covers |
|---|---|---|---|---|---|
| time_stats | 23 | 0 | 0 | 23 | mean/sd/median/MAD/IQR/skew/kurtosis/quantiles/shape/crest/impulse/clearance/ZCR |
| energy | 4 | 3 | 0 | 7 | energy, RMS, mean-abs, p2p, log-energy, Teager-Kaiser energy, Shannon energy entropy (signal-energy indicators in a.u. - joules only after calibration) |
| envelope | 0 | 5 | 0 | 5 | Hilbert envelope mean/sd/kurtosis, instantaneous frequency mean/sd |
| dynamics | 10 | 8 | 0 | 18 | d1/d2 moments, slope, turning rate, Hjorth activity/mobility/complexity, ACF decay/lag-1/first-zero, run lengths |
| peaks_events | 0 | 12 | 0 | 12 | peak count/interval/prominence, burst rate/duration, CUSUM change-point count and max jump |
| quality | 5 | 0 | 0 | 5 | NaN/inf fraction, clipped-sample fraction, DC offset, flatline fraction |
| spectral | 13 | 16 | 0 | 29 | dominant/2nd/3rd peak frequencies, centroid, bandwidth, rolloff85, flatness, entropy, per-band powers + relative band ratios (sampling-rate aware: every band is clamped to Nyquist) |
| time_frequency | 2 | 17 | 2 | 21 | STFT flux/high-frequency ratio, Morlet wavelet energies at 4/8/16/32 widths, wavelet-packet node energies, band-energy time profiles, spectral-entropy time profile |
| entropy_nonlinear | 0 | 2 | 14 | 16 | permutation entropy (orders 2-4), sample/approximate entropy, multiscale entropy (s2/s4/s8), DFA alpha, Higuchi/Katz/Petrosian FD, Hurst R/S, RQA (rec/det/lmax/entropy), LZ complexity, largest-Lyapunov-style slope |
| ar_cepstral | 0 | 13 | 1 | 14 | Yule-Walker AR(1-4) + residual variance, MFCC1-4 (20 mel bands), log-filter-bank energies, 5-taper multitaper dominant frequency (NW=3, K=5) |
| decomposition | 0 | 0 | 4 | 4 | SSA leading-singular-value energy fraction + SV-spectrum entropy (numpy); EMD/VMD first-mode energy fractions (NaN unless optional `emd`/`vmdpy` installed) |
| **total** | **57** | **76** | **21** | **154** | |

The 12th family - **learned representations** - is deliberately NOT a registry column family:
it is branches A (dilated-TCN + transformer on raw windows), B (learnable filterbank + spectral
attention) and C (self-supervised masked-prediction embeddings). Fixed transforms (STFT,
Morlet, mel banks, DPSS multitaper) are reproducible label-blind descriptors; learned encoders
are trained per fold and can overfit - that distinction is preserved in the architecture rather
than blurred into one vocabulary.

Missing-value handling (uniform policy): every undefined feature is NaN with a documented
reason (e.g. no peak in the window); NaNs are median-imputed with train-fold medians, and the
`feature_select: variance` filter (below) drops all-NaN or constant columns using training
windows only. Optional-dependency features (EMD/VMD) are NaN when the package is absent, so a
run without them is identical to a run with `feature_families` excluding decomposition.

### 11.2 Window level AND recording level

`featreg.recording_summary` turns per-window feature dicts into recording-level rows:
`rec_n_windows` plus `<feature>_rec_mean` / `<feature>_rec_sd` for every column (301 columns at
the default tiers). With `segmentation.window_s: null` a recording IS one window, so the
recording level degenerates gracefully to the window level. Output:
`recording_feature_summary.csv` (sample_id, class, all summary columns).

### 11.3 Leakage protocol additions (train-fold-only selection)

- Files are split BEFORE windowing (unchanged, enforced by `cv_quad` fold indices).
- NEW: `quad.feature_select: variance` keeps only columns that are finite and non-constant
  **across the training windows of that fold/repeat**. The test fold never votes. The kept/dropped
  counts are written into the fold notes and the run log.
- Imputer medians/IQRs were already fit on training windows only; that is unchanged.

### 11.4 What we do NOT claim

- A larger registry does not mean better accuracy. Which features help is an empirical question
  answered by the paired branch ablations (`quad_minus_engineered`), the redundancy report
  (Spearman pairs, CSV twin) and the train-only selection logs - never asserted a priori.
- Raw-signal-only baselines remain in the run: Model A (handcrafted + gradient boosting) and
  Model B (deep window model) are scored on identical folds, so "did the registry help?" is
  directly comparable in `report_results.csv`.
- Few-shot/one-shot results stay a SEPARATE evaluation (`fewshot_episodes.csv`): episodes draw
  independent support and query files from the fold's test side only, pooled accuracy with
  Wilson CIs, never mixed into the headline metrics.

### 11.5 STFT vs wavelet trade-off (why the registry carries both)

| | STFT (branch B + `w_stft_*`) | Morlet wavelets (`w_wav_energy_*`) | wavelet packet (`w_wp_energy_*`) |
|---|---|---|---|
| resolution | fixed time-frequency tiling (nfft=256, hop=64) | constant-Q: fine time / coarse frequency at high f, the reverse at low f | adaptive-ish binary splitting, still fixed filters |
| best for | stationary-ish spectra, flux/change detection | transients and bursts (trapping events) | per-band energy without re-transforming |
| cost | one transform per window | 4 convolutions (widths 4-32, capped at 8192 samples) | 4-level packet tree |
| failure mode | smears short events; hop choice is a bias | edge effects on short windows (centre-cropped kernels) | coarse frequency labels per node |

All three are sampling-rate aware: band edges, mel banks and packet nodes are clamped to
Nyquist of `training.analysis_rate_hz`, so the same registry works at 1 MHz raw or any menu rate.

### 11.6 Robustness add-ons (all opt-in, defaults preserve earlier behaviour)

- `quad.triplet_weight` (default 0): batch-hard triplet loss added to CE + SupCon in stages 3-4.
- `quad.lr_schedule` (default none): cosine or step decay across supervised epochs.
- `quad.early_stopping_patience` (default 0 = off): stops on inner-validation plateau; the
  best-checkpoint rule itself always applied.
- Checkpointing: every trained fold now writes `<run>/checkpoints/quad_<tag>.pt`
  (state_dict + classes + best inner-validation score) so a fold can be inspected or reloaded.
- Open-set diagnostic: `quad.open_set_msp_threshold` flags test files whose max-softmax
  probability (mean over windows/repeats) falls below the threshold
  (`quad_open_set_flags.csv` + `summary.json.quad.open_set`). Diagnostic only - true open-set
  recognition needs held-out unknown classes, and the run says so.
- Extended metrics gain `cohens_kappa` and micro/macro/weighted precision/recall/F1 rows per
  model in `metrics_extended.csv` (micro averages equal accuracy in multiclass).

### 11.7 Per-run artifacts added by round 2

| artifact | content |
|---|---|
| `feature_registry.csv` | every selected feature: id, family, family description, tier, description, unit, parameters |
| `recording_feature_summary.csv` | recording-level mean/sd per feature + window count, keyed by sample_id/class |
| `quad_open_set_flags.csv` | per-file MSP + unknown flag (only when the threshold is set) |
| `checkpoints/quad_*.pt` | per-fold model checkpoints |
| fold notes in `summary.json` | train-only selection kept/dropped counts, early-stop events |
