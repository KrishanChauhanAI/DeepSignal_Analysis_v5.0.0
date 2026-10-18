"""Typed, validated configuration loaded from YAML.

Unknown keys are rejected so that a typo can never silently change an
experiment. Cross-field rules live in :func:`validate_config` and are reported
by the preflight stage as ERROR entries.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints

import yaml

FilterMode = Literal["none", "lpf", "gaussian", "lpf+gaussian"]
LabelLevel = Literal["file", "event", "unknown"]
LabelSource = Literal["metadata", "folder", "none"]
DatEncoding = Literal["binary", "text"]
DatDtype = Literal["int8", "uint8", "int16", "uint16", "int32", "uint32", "float32", "float64"]
ModelSet = Literal["features", "deep", "hybrid"]
CombineRule = Literal["equal_weight"]
WindowLength = Literal["auto"]
MetricLoss = Literal["none", "supcon", "prototypical"]
ProtoDistance = Literal["cosine", "euclidean"]
FusionRule = Literal["none", "blend", "stack", "both"]


class ConfigError(ValueError):
    """Raised when a configuration file cannot be parsed or has unknown keys."""


@dataclass(frozen=True)
class RunConfig:
    name: str = "trapsig"
    seed: int = 42


@dataclass(frozen=True)
class DatFormat:
    """Layout of the laboratory ``.dat`` files. Every value here must be confirmed from a real file.

    ``encoding="binary"``: raw samples of ``dtype``, optionally interleaved ``n_columns`` wide,
    after ``header_bytes`` of file header. The sampling rate is ``sampling.original_hz``.
    ``encoding="text"``: one row per sample, values separated by commas, semicolons, or
    whitespace, after ``header_lines`` lines of header.

    ``trapsig inspect-dat`` lists the layouts consistent with a file's size. Use it to choose
    these values; the defaults are a starting guess, not a fact about the instrument.
    """
    encoding: DatEncoding = "binary"
    dtype: DatDtype = "float32"
    byteorder: Literal["little", "big"] = "little"
    header_bytes: int = 0
    header_lines: int = 0
    n_columns: int = 1
    column: int = 0
    scale: float = 1.0
    units: str = "raw"


@dataclass(frozen=True)
class DataConfig:
    source_dir: Path | None = None
    file_glob: str = "*.csv"
    dat: DatFormat = field(default_factory=DatFormat)
    label_level: LabelLevel = "unknown"
    label_from: LabelSource = "none"
    metadata_csv: Path | None = None
    allow_file_level_fallback: bool = False
    expected_duration_s: float = 20.0
    duration_tolerance_s: float = 0.01


@dataclass(frozen=True)
class SamplingConfig:
    original_hz: float = 1_000_000.0
    target_rates_hz: list[float] = field(
        default_factory=lambda: [50_000.0, 10_000.0, 1_000.0, 100.0, 10.0, 2.0]
    )
    # The supported analysis-rate menu (design: the acquisition rate and the analysis rate are
    # different things; downsampling anti-aliases first, upsampling interpolates and adds no
    # information). ``allow_off_menu_rates`` keeps backwards compatibility: the menu is advisory
    # unless the user opts into strict enforcement.
    rate_menu_hz: list[float] = field(
        default_factory=lambda: [1_000_000.0, 50_000.0, 25_000.0, 10_000.0, 1_000.0, 100.0,
                                 10.0, 2.0])
    allow_off_menu_rates: bool = True


@dataclass(frozen=True)
class FilterConfig:
    mode: FilterMode = "none"
    lpf_cutoff_hz: float | None = None
    lpf_order: int = 4
    gaussian_sigma_s: float | None = None
    gaussian_truncate: float = 4.0
    notch_hz: list[float] = field(default_factory=list)
    notch_q: float = 30.0


@dataclass(frozen=True)
class EventConfig:
    baseline_window_s: float = 1.0
    threshold_sigma: float = 5.0
    exit_sigma: float = 3.0
    min_duration_s: float = 1e-3
    min_separation_s: float = 1e-3
    pre_event_window_s: float = 0.1
    edge_guard_s: float = 0.01       # onsets earlier than this are excluded (filter edge transients)


@dataclass(frozen=True)
class SplitConfig:
    n_splits: int = 5
    n_repeats: int = 5
    min_groups_per_class: int = 5


@dataclass(frozen=True)
class TrainingConfig:
    """Model A (feature-based) settings. All values are fixed before evaluation; none is tuned on test folds."""
    analysis_rate_hz: float = 10_000.0
    # Pre-specified spectral bands in Hz (lo, hi). The 45-55 Hz band tracks mains residue.
    psd_bands_hz: list[list[float]] = field(default_factory=lambda: [
        [1.0, 10.0], [10.0, 100.0], [100.0, 1000.0], [1000.0, 5000.0], [45.0, 55.0]])
    # Autocorrelation lags in milliseconds.
    acf_lags_ms: list[float] = field(default_factory=lambda: [0.1, 1.0, 10.0])
    models: list[str] = field(default_factory=lambda: ["logreg", "rf", "hgb"])
    lr_c_grid: list[float] = field(default_factory=lambda: [0.01, 0.1, 1.0, 10.0])
    inner_splits: int = 3
    rf_n_estimators: int = 500
    hgb_max_iter: int = 200
    n_permutations: int = 20
    bootstrap_resamples: int = 1000
    # Which handcrafted feature groups are computed and modelled. "all" = every group in
    # features.FEATURE_GROUPS (noise, time, amp, event, ctx, spec, tf, acf, shape, morph).
    feature_groups: list[str] = field(default_factory=lambda: ["all"])
    # Feature ablation: refit the declared ablation model on each group alone ("groups"), on
    # everything minus one group ("leave_one_out"), on both ("both"), or not at all ("none").
    # Runs on the same folds as the main cross-validation, so the comparison is like for like.
    # "both" is the default because the two directions answer different questions: "only:<g>"
    # asks whether a group carries the signal on its own, "without:<g>" asks whether it adds
    # anything to the rest. --quick sets "none"; the cost is one refit per variant.
    ablation: str = "both"
    ablation_model: str = "logreg"
    # |Spearman| above this between two features is reported as redundant (reported, never
    # silently dropped: dropping on the pooled data would use test-fold information).
    redundancy_threshold: float = 0.98

    # ---- model-specific hyperparameter search, nested inside each outer training fold ----
    #   none     no search: logreg keeps its exact grid over lr_c_grid, rf and hgb keep the
    #            declared n_estimators / max_iter. This is what the shipped defaults use, because
    #            a nested search costs trials x inner_splits extra fits per model per fold.
    #   random   uniform sampling without replacement over the model's declared space
    #   halving  successive halving: every trial starts on a fraction of the training FILES and
    #            the best third is promoted; falls back to random on folds too small to subsample
    # The search is per model, per repeat, per outer fold, and it only ever sees that fold's
    # training files, so no hyperparameter is chosen with knowledge of the test fold.
    # Persist every fitted Model A pipeline (one per repeat x outer fold) into ``models/`` with a
    # JSON sidecar naming its feature columns, classes and hyperparameters, so ``trapsig predict``
    # can reuse the run. The sidecar, not the pickle, is the contract: a model whose columns do not
    # match this code's feature table is refused instead of producing confident nonsense.
    save_models: bool = True
    # Extended evaluation surface (Phase: comprehensive classification evaluation): top-k
    # accuracy (0 disables), Brier score, expected calibration error bins, and OvR precision-
    # recall curves for every scored model. All are computed from the same out-of-fold arrays.
    report_top_k: int = 0
    calibration_bins: int = 10
    pr_curves: bool = True
    hpo: str = "none"
    # Objective of the search. Log loss is negated internally so that every metric is maximised.
    # roc_auc_ovr needs probabilities for every class and is refused (trial marked failed) on an
    # inner split that holds fewer than two classes, rather than being reported as 0.5.
    hpo_metric: str = "balanced_accuracy"
    hpo_n_iter: int = 12              # trials per model per outer fold
    hpo_seed_offset: int = 991        # keeps the search draws independent of the fold draws


@dataclass(frozen=True)
class DeepConfig:
    """Model B (deep 1D CNN) and the hybrid combination (design sections 11.3 and 11.5).

    Every value is fixed before evaluation. The validation split inside each outer
    training fold is used for early stopping, checkpoint selection, and temperature
    scaling; the outer test fold is never used for any of those.
    """
    enabled: bool = True
    # Which models to run: "features" (Model A only), "deep" (Model B only), "hybrid" (both + ensemble).
    model_set: ModelSet = "hybrid"
    # Window sampling rate. Defaults to training.analysis_rate_hz when null.
    window_rate_hz: float | None = None
    # Window length. "auto" derives it from the event-duration distribution of the pool
    # (median duration x window_ms_auto_multiplier, clamped to the min/max below).
    window_ms: float | WindowLength = "auto"
    window_ms_auto_multiplier: float = 4.0
    window_ms_min: float = 10.0
    window_ms_max: float = 500.0
    # Fraction of the window placed before the event onset (pre-event baseline context).
    pre_event_fraction: float = 0.25
    max_windows_per_file: int = 40          # caps files with very many events; chosen before fitting
    # Aggregation from window predictions to the file-level label. Fixed before evaluation.
    aggregation: Literal["mean_proba", "majority_vote"] = "mean_proba"

    # Architecture (compact by design; target 1e4-1e5 parameters).
    channels: int = 32
    n_blocks: int = 3                        # dilated residual blocks: dilations 1, 2, 4
    kernel_size: int = 5
    stem_stride: int = 4                     # downsamples the raw window first (cheap on CPU)
    dropout: float = 0.2

    # Training protocol.
    epochs: int = 40
    batch_size: int = 32
    learning_rate: float = 3e-3
    weight_decay: float = 1e-2
    warmup_epochs: int = 2
    early_stopping_patience: int = 8
    early_stopping_min_delta: float = 1e-4
    # Checkpoint selection metric, fixed before running.
    select_metric: Literal["macro_f1", "balanced_accuracy", "log_loss"] = "macro_f1"
    gradient_clip: float = 1.0
    # Inner validation split (carved from the outer training fold by file group).
    validation_fraction: float = 0.2
    # Label-shuffle permutations for Model B. Expensive on CPU; 0 means "not run", reported as such.
    n_permutations: int = 0
    # Temperature scaling on the validation fold only.
    temperature_scaling: bool = True
    threads: int = 0                         # 0 = leave PyTorch to choose

    # ---- representation and metric learning ----
    # The encoder is shared: its GAP embedding is what the prototypical head, few-shot, novelty,
    # clustering, and the fusion model all consume, so the auxiliary loss below shapes one
    # representation for every downstream strategy instead of each of them re-learning one.
    #   none          classification loss only
    #   supcon        supervised contrastive: same-class embeddings pull together, other classes
    #                 push apart (Khosla et al. 2020), which is what a prototype/distance
    #                 classifier needs
    #   prototypical  batch-prototype loss: each embedding must be closest to its own class
    #                 prototype computed from the same batch
    metric_loss: MetricLoss = "supcon"
    metric_loss_weight: float = 0.2        # weight of the auxiliary term; 0 disables it
    metric_temperature: float = 0.1        # contrastive/prototype softmax temperature
    # Nearest-prototype classifier on the frozen embedding (metric learning at inference time).
    # Prototypes are built from the inner-validation files of each outer fold, so the test fold
    # never contributes to them.
    prototypical_head: bool = True
    proto_distance: ProtoDistance = "cosine"

    # ---- hyperparameter search for Model B ----
    # 0 disables the search and the reason is recorded: each trial is a full train_fold on the
    # outer training windows, so the budget is the user's decision, exactly like n_permutations.
    # Trials are scored on the SAME inner-validation files that early stopping uses, so the search
    # is nested in the outer fold and the test fold never selects a setting. The reported fold is
    # then retrained once with the winning setting, which is what its checkpoint contains.
    # The embedding WIDTH (channels) is deliberately NOT searched: it is part of the
    # representation contract shared by the prototype head, the fusion head, few-shot, novelty
    # detection and clustering, so it must be declared before the run rather than vary per fold.
    hpo_trials: int = 0
    hpo_space_learning_rate: list[float] = field(
        default_factory=lambda: [0.001, 0.003, 0.01])
    hpo_space_dropout: list[float] = field(default_factory=lambda: [0.1, 0.2, 0.3])
    hpo_space_weight_decay: list[float] = field(default_factory=lambda: [0.0, 0.01, 0.03])

    # ---- how Model A (handcrafted features) and Model B (learned representation) combine ----
    #   blend = fixed convex blend of file-level probabilities; no free parameters, so nothing can
    #           be fitted on test data (weight below)
    #   stack = a fusion head on [robust-standardised handcrafted features, file-level embedding]
    #           fitted on the inner-validation FILES of each outer fold: the encoder never saw
    #           those files, and the outer test fold contributes nothing, so it is leakage-safe
    #           and costs one logistic regression per fold
    #   both  = report both, pre-declared; the comparison framework and gate G7 stay on the
    #           matched subset and the reader sees which combination the data supports
    fusion: FusionRule = "both"
    combine: CombineRule = "equal_weight"
    model_a_weight: float = 0.5


@dataclass(frozen=True)
class AdvancedConfig:
    """Conditional analyses (design sections 11.6-11.10). Each is gated; off by default."""
    # Few-shot: prototypical episodes over the frozen CNN encoder. Setting (b) of design 11.7,
    # i.e. new sessions/replicates. With file-level groups only, shots are distinct files.
    few_shot: bool = False
    few_shot_k: list[int] = field(default_factory=lambda: [1, 2, 5, 10])
    few_shot_episodes: int = 200
    few_shot_distance: Literal["cosine", "euclidean"] = "cosine"

    # Semi-supervised: confidence-aware pseudo-labelling on unlabelled windows drawn from
    # training groups only (design 11.8). Requires unlabelled files in the pool.
    semi_supervised: bool = False
    pseudo_confidence_tau: float = 0.95
    pseudo_margin_delta: float = 0.20
    pseudo_rounds: int = 2

    # Novelty: prototype distance, max softmax, and energy score. Thresholds are set on
    # validation known-class data only. Needs an unknown set to be evaluable (design 11.9).
    novelty: bool = False
    novelty_known_retention: float = 0.95

    # Clustering: exploratory only, fitted on training groups (design 11.10).
    clustering: bool = False
    clustering_k: list[int] = field(default_factory=lambda: [2, 3, 4])


@dataclass(frozen=True)
class ExplainConfig:
    """Explainability (design 11.12). Runs after the model is frozen; never used to select it."""
    enabled: bool = True
    # Model A: exact linear SHAP for logreg, TreeSHAP for the tree models. Feature level only.
    shap_features: bool = True
    shap_background: int = 50
    # Model B: Grad-CAM is CAM-valid here (conv -> global average pooling -> linear).
    grad_cam_1d: bool = True
    integrated_gradients: bool = True
    integrated_gradients_steps: int = 32
    occlusion: bool = True
    occlusion_segments: int = 20
    # Mandatory shortcut probes.
    shortcut_probes: bool = True
    n_explained_windows: int = 20


@dataclass(frozen=True)
class ReportConfig:
    """Visualization and reporting (design section 12)."""
    plots: bool = True
    pdf: bool = True
    csv: bool = True
    dpi: int = 150
    max_plots_per_class: int = 3


@dataclass(frozen=True)
class OutputConfig:
    runs_dir: Path = Path("runs")


@dataclass(frozen=True)
class SegmentationConfig:
    """Windowing for models that consume fixed-length segments (the quad-branch encoder).

    Deliberately separate from ``deep.window_ms``: Model B's event windows keep their semantics,
    and this block only drives the new four-branch encoder. ``window_s: null`` means one window
    per recording (the whole file), which is the right default for 20 s recordings at 10 kHz.
    """

    window_s: float | None = None
    overlap: float = 0.5                # fraction of the window; stride = window * (1 - overlap)
    stride_s: float | None = None       # explicit stride wins over overlap when set
    normalize: str = "robust"           # none | zscore | robust (per window)
    detrend: str = "none"               # none | constant | linear
    pad_mode: str = "reflect"           # reflect | zero | edge (short tails only)
    max_windows_per_file: int = 64


@dataclass(frozen=True)
class QuadConfig:
    """The four-branch deep architecture: temporal, spectral, self-supervised, engineered.

    ``enabled: false`` keeps every existing behaviour bit-identical; the branch model is an
    additional scored model, never a replacement for Model A or Model B.
    """

    enabled: bool = False
    embed_dim: int = 64
    # branch switches (an ablation can also zero a branch at fusion time, see branch_ablation)
    use_temporal: bool = True
    use_spectral: bool = True
    use_engineered: bool = True
    use_selfsup: bool = True
    branch_ablation: list[str] = field(default_factory=list)   # e.g. ["temporal", "spectral"]
    # branch A
    t_channels: list[int] = field(default_factory=lambda: [16, 32, 64])
    t_dilations: list[int] = field(default_factory=lambda: [1, 4, 8])
    t_kernel: int = 7
    t_transformer_layers: int = 1
    t_nhead: int = 4
    # branch B (fixed STFT front end + learned spectral CNN + frequency attention)
    s_nfft: int = 256
    s_hop: int = 64
    s_channels: list[int] = field(default_factory=lambda: [16, 32])
    # branch C: self-supervised pretraining on the training fold only
    pretrain_epochs: int = 6
    pretrain_contrastive_weight: float = 1.0
    pretrain_recon_weight: float = 0.5
    pretrain_mask_fraction: float = 0.15
    pretrain_temperature: float = 0.1
    # stage 3/4 supervised fine-tuning
    epochs: int = 12
    learning_rate: float = 0.003
    weight_decay: float = 0.01
    dropout: float = 0.1
    batch_size: int = 32
    warmup_epochs: int = 1
    focal_gamma: float = 0.0            # 0 = plain cross-entropy
    class_weight: str = "none"          # none | balanced
    fusion_layers: int = 1
    fusion_nhead: int = 4
    select_metric: str = "balanced_accuracy"
    # stage 5: episodic metric learning / few-shot evaluation
    episodes: int = 40
    shots: list[int] = field(default_factory=lambda: [1, 5])
    queries_per_class: int = 3
    metric_temperature: float = 0.1
    supcon_weight: float = 0.2
    amp: bool = False                   # mixed precision (CPU runs ignore it safely)
    seed_offset: int = 4242
    # Branch D registry selection (see featreg.FAMILIES / docs/QUAD_ARCHITECTURE.md section 4)
    feature_tiers: list[int] = field(default_factory=lambda: [1, 2])
    feature_families: list[str] = field(default_factory=list)   # empty = all families
    feature_select: str = "variance"    # none | variance (training-fold-only column filter)
    # Head B / optimisation extras (all default OFF to preserve earlier behaviour)
    triplet_weight: float = 0.0
    lr_schedule: str = "none"           # none | cosine | step
    early_stopping_patience: int = 0    # 0 = off; checkpoint selection still applies
    open_set_msp_threshold: float | None = None   # max-softmax-probability rejection flag


@dataclass(frozen=True)
class TrapsigConfig:
    run: RunConfig = field(default_factory=RunConfig)
    data: DataConfig = field(default_factory=DataConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    quad: QuadConfig = field(default_factory=QuadConfig)
    filters: FilterConfig = field(default_factory=FilterConfig)
    events: EventConfig = field(default_factory=EventConfig)
    splitting: SplitConfig = field(default_factory=SplitConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    deep: DeepConfig = field(default_factory=DeepConfig)
    advanced: AdvancedConfig = field(default_factory=AdvancedConfig)
    explain: ExplainConfig = field(default_factory=ExplainConfig)
    report: ReportConfig = field(default_factory=ReportConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary (Paths become strings)."""
        out: dict[str, Any] = _to_jsonable(dataclasses.asdict(self))
        return out

    def fingerprint(self) -> str:
        """SHA-256 of the canonical JSON form. Identifies the exact configuration."""
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Generic dict -> dataclass construction with strict key checking
# ---------------------------------------------------------------------------

def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (bool, int, float, str)) or obj is None:
        return obj
    return str(obj)


def _coerce(value: Any, tp: Any, path: str) -> Any:
    """Coerce ``value`` to annotation ``tp``, rejecting anything that does not fit."""
    origin = get_origin(tp)
    if origin is Union or origin is types.UnionType:
        args = [a for a in get_args(tp) if a is not type(None)]
        if value is None and len(args) != len(get_args(tp)):
            return None
        for a in args:
            try:
                return _coerce(value, a, path)
            except ConfigError:
                continue
        raise ConfigError(f"{path}: value {value!r} does not match any of {args}")
    if origin is Literal:
        allowed = get_args(tp)
        if value not in allowed:
            raise ConfigError(f"{path}: {value!r} not in {list(allowed)}")
        return value
    if origin in (list, tuple):
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"{path}: expected a list, got {type(value).__name__}")
        targs = get_args(tp)
        inner = targs[0] if targs else Any
        return [_coerce(v, inner, f"{path}[{i}]") for i, v in enumerate(value)]
    if tp is Path:
        if not isinstance(value, (str, Path)):
            raise ConfigError(f"{path}: expected a path string, got {type(value).__name__}")
        return Path(value)
    if tp is Any:
        return value
    if isinstance(tp, type):
        if tp is bool:
            if not isinstance(value, bool):
                raise ConfigError(f"{path}: expected bool, got {value!r}")
            return value
        if tp is int:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ConfigError(f"{path}: expected int, got {value!r}")
            return int(value)
        if tp is float:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ConfigError(f"{path}: expected float, got {value!r}")
            return float(value)
        if tp is str:
            if not isinstance(value, str):
                raise ConfigError(f"{path}: expected str, got {value!r}")
            return value
        if dataclasses.is_dataclass(tp):
            return _build(tp, value, path)
    raise ConfigError(f"{path}: unsupported type {tp}")


def _build(cls: type, data: Any, path: str) -> Any:
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: expected a mapping for {cls.__name__}, got {type(data).__name__}")
    hints = get_type_hints(cls, include_extras=True)
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = [k for k in data if k not in known]
    if unknown:
        raise ConfigError(f"{path}: unknown key(s) {unknown}. Allowed: {sorted(known)}")
    kwargs: dict[str, Any] = {}
    for name, tp in hints.items():
        if name not in data:
            continue
        kwargs[name] = _coerce(data[name], tp, f"{path}.{name}")
    return cls(**kwargs)


def config_from_dict(data: dict[str, Any]) -> TrapsigConfig:
    """Build a :class:`TrapsigConfig` from a plain dictionary with strict key checking."""
    cfg = _build(TrapsigConfig, data, "config")
    assert isinstance(cfg, TrapsigConfig)
    return cfg


def validate_segmentation(cfg: TrapsigConfig) -> list[str]:
    g = cfg.segmentation
    errs: list[str] = []
    if g.window_s is not None and g.window_s <= 0:
        errs.append("segmentation.window_s must be > 0 when set")
    if not 0.0 <= g.overlap < 1.0:
        errs.append("segmentation.overlap must lie in [0, 1)")
    if g.stride_s is not None and g.stride_s <= 0:
        errs.append("segmentation.stride_s must be > 0 when set")
    if g.normalize not in ("none", "zscore", "robust"):
        errs.append("segmentation.normalize must be none | zscore | robust")
    if g.detrend not in ("none", "constant", "linear"):
        errs.append("segmentation.detrend must be none | constant | linear")
    if g.pad_mode not in ("reflect", "zero", "edge"):
        errs.append("segmentation.pad_mode must be reflect | zero | edge")
    if g.max_windows_per_file < 1:
        errs.append("segmentation.max_windows_per_file must be >= 1")
    return errs


def validate_quad(cfg: TrapsigConfig) -> list[str]:
    q = cfg.quad
    errs: list[str] = []
    if not q.enabled:
        return errs
    if q.embed_dim < 8:
        errs.append("quad.embed_dim must be >= 8")
    if not (q.use_temporal or q.use_spectral or q.use_engineered):
        errs.append("quad: at least one representation branch must be enabled")
    if q.use_selfsup and not (q.use_temporal or q.use_spectral):
        errs.append("quad.use_selfsup needs the temporal or spectral encoder to pretrain")
    if q.pretrain_epochs < 0 or q.epochs < 1:
        errs.append("quad: pretrain_epochs >= 0 and epochs >= 1")
    if q.focal_gamma < 0:
        errs.append("quad.focal_gamma must be >= 0")
    if q.class_weight not in ("none", "balanced"):
        errs.append("quad.class_weight must be none | balanced")
    if q.batch_size < 2:
        errs.append("quad.batch_size must be >= 2")
    if q.episodes < 1 or not q.shots or any(sh < 1 for sh in q.shots):
        errs.append("quad: episodes >= 1 and shots >= 1 required")
    if q.queries_per_class < 1:
        errs.append("quad.queries_per_class must be >= 1")
    unknown = [b for b in q.branch_ablation
               if b not in ("temporal", "spectral", "engineered", "selfsup")]
    if unknown:
        errs.append(f"quad.branch_ablation unknown branch(es): {', '.join(unknown)}")
    if q.s_nfft < 16 or q.s_hop < 1 or q.s_hop > q.s_nfft:
        errs.append("quad: require s_nfft >= 16 and 1 <= s_hop <= s_nfft")
    if cfg.splitting.n_splits < 2:
        errs.append("quad.enabled requires splitting.n_splits >= 2 (held-out folds)")
    if not set(q.feature_tiers) <= {1, 2, 3} or not q.feature_tiers:
        errs.append("quad.feature_tiers must be a non-empty subset of [1, 2, 3]")
    from . import featreg

    unknown_fams = [f for f in q.feature_families if f not in featreg.FAMILIES]
    if unknown_fams:
        errs.append(f"quad.feature_families unknown: {unknown_fams}")
    if q.feature_select not in ("none", "variance"):
        errs.append("quad.feature_select must be 'none' or 'variance'")
    if q.triplet_weight < 0:
        errs.append("quad.triplet_weight must be >= 0")
    if q.lr_schedule not in ("none", "cosine", "step"):
        errs.append("quad.lr_schedule must be 'none', 'cosine' or 'step'")
    if q.early_stopping_patience < 0:
        errs.append("quad.early_stopping_patience must be >= 0")
    if q.open_set_msp_threshold is not None and not 0 < q.open_set_msp_threshold <= 1:
        errs.append("quad.open_set_msp_threshold must be null or in (0, 1]")
    return errs


def validate_sampling_menu(cfg: TrapsigConfig) -> list[str]:
    sm = cfg.sampling
    errs: list[str] = []
    if not sm.rate_menu_hz or any(v <= 0 for v in sm.rate_menu_hz):
        errs.append("sampling.rate_menu_hz must list positive rates")
    if not sm.allow_off_menu_rates:
        off = [v for v in (cfg.training.analysis_rate_hz, cfg.deep.window_rate_hz)
               if v and v not in sm.rate_menu_hz]
        if off:
            errs.append(f"sampling.allow_off_menu_rates is false but these rates are off-menu: "
                        f"{sorted(set(off))} (menu: {sm.rate_menu_hz})")
    return errs


def validate_training(cfg: TrapsigConfig) -> list[str]:
    """Public entry point for the Model A rules (used by the CLI and the preflight stage)."""
    return _training_errors(cfg)


def validate_deep(cfg: TrapsigConfig) -> list[str]:
    """Public entry point for the Model B and hybrid rules."""
    return _deep_errors(cfg)


def load_config(path: str | Path) -> TrapsigConfig:
    """Load and validate a YAML configuration file."""
    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"configuration file not found: {p}")
    try:
        with p.open(encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        raise ConfigError(f"{p}: invalid YAML ({exc})") from exc
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{p}: top level must be a mapping")
    return config_from_dict(raw)


# ---------------------------------------------------------------------------
# Cross-field validation
# ---------------------------------------------------------------------------

def validate_config(cfg: TrapsigConfig) -> list[str]:
    """Return a list of configuration problems. An empty list means the configuration is usable.

    This is the **base** contract shared by ``dry-run``, ``events``, and ``prepare``. It
    deliberately does not touch the filesystem beyond nothing at all: whether ``source_dir``
    exists is preflight check P05, and modelling rules live in :func:`validate_modelling`.
    Splitting them keeps a training-only mistake from blocking data inspection.
    """
    errs: list[str] = []
    d, s, f, e, sp = cfg.data, cfg.sampling, cfg.filters, cfg.events, cfg.splitting

    if s.original_hz <= 0:
        errs.append("sampling.original_hz must be positive")
    else:
        for r in s.target_rates_hz:
            if r <= 0:
                errs.append(f"sampling.target_rates_hz contains a non-positive rate: {r}")
            elif r > s.original_hz:
                errs.append(f"sampling.target_rates_hz rate {r:g} exceeds original_hz {s.original_hz:g}")
            else:
                ratio = s.original_hz / r
                if abs(ratio - round(ratio)) > 1e-9:
                    errs.append(
                        f"sampling.target_rates_hz rate {r:g} is not an integer divisor of "
                        f"original_hz {s.original_hz:g} (ratio {ratio:.4f}); staged decimation "
                        "needs an integer ratio at every stage"
                    )

    if f.mode in ("lpf", "lpf+gaussian") and f.lpf_cutoff_hz is None:
        errs.append(f"filters.mode {f.mode!r} requires filters.lpf_cutoff_hz")
    if f.mode in ("gaussian", "lpf+gaussian") and f.gaussian_sigma_s is None:
        errs.append(f"filters.mode {f.mode!r} requires filters.gaussian_sigma_s")
    if f.lpf_cutoff_hz is not None and f.mode in ("none", "gaussian"):
        errs.append(f"filters.mode {f.mode!r} does not use it: filters.lpf_cutoff_hz is set")
    if f.gaussian_sigma_s is not None and f.mode in ("none", "lpf"):
        errs.append(f"filters.mode {f.mode!r} does not use it: filters.gaussian_sigma_s is set")
    if f.lpf_cutoff_hz is not None and not (0 < f.lpf_cutoff_hz < s.original_hz / 2):
        errs.append("filters.lpf_cutoff_hz must be in (0, Nyquist of the original rate)")
    if f.gaussian_sigma_s is not None and f.gaussian_sigma_s <= 0:
        errs.append("filters.gaussian_sigma_s must be positive")
    if f.notch_hz:
        for hz in f.notch_hz:
            if hz <= 0 or hz >= s.original_hz / 2:
                errs.append(f"filters.notch_hz value {hz:g} must be in (0, Nyquist of the original rate)")
        if f.notch_q <= 0:
            errs.append("filters.notch_q must be positive")

    if e.baseline_window_s <= 0:
        errs.append("events.baseline_window_s must be positive")
    if e.threshold_sigma <= 0:
        errs.append("events.threshold_sigma must be positive")
    if e.exit_sigma >= e.threshold_sigma:
        errs.append("events.exit_sigma must be below events.threshold_sigma (hysteresis)")
    if e.min_duration_s < 0:
        errs.append("events.min_duration_s must be non-negative")
    if e.min_separation_s < 0:
        errs.append("events.min_separation_s must be non-negative")
    if e.pre_event_window_s < 0:
        errs.append("events.pre_event_window_s must be non-negative")
    if e.edge_guard_s < 0:
        errs.append("events.edge_guard_s must be non-negative")

    if sp.n_repeats < 1:
        errs.append("splitting.n_repeats must be at least 1")
    if sp.n_splits < 2:
        errs.append("splitting.n_splits must be at least 2")
    if sp.min_groups_per_class < sp.n_splits:
        errs.append(
            "splitting.min_groups_per_class must be >= n_splits so every test fold "
            "can contain at least one group per class"
        )

    if d.label_from == "metadata" and d.metadata_csv is None:
        errs.append("data.label_from 'metadata' requires data.metadata_csv")
    if d.label_level == "file" and d.label_from == "none":
        errs.append("data.label_level is 'file' but data.label_from is 'none'; no labels would exist")
    if d.expected_duration_s <= 0:
        errs.append("data.expected_duration_s must be positive")
    if d.duration_tolerance_s < 0:
        errs.append("data.duration_tolerance_s must be non-negative")

    errs.extend(_dat_errors(d.dat))
    errs.extend(validate_sampling_menu(cfg))
    return errs


def validate_modelling(cfg: TrapsigConfig) -> list[str]:
    """Rules for the modelling stages only (Model A, Model B, explainability, reporting).

    Kept separate from :func:`validate_config` on purpose: ``events`` and ``prepare`` must not be
    blocked by a training setting they do not use. ``train`` and ``run`` call both.
    """
    errs = _training_errors(cfg)
    errs.extend(_deep_errors(cfg))
    errs.extend(_advanced_errors(cfg))
    errs.extend(_explain_errors(cfg))
    errs.extend(_report_errors(cfg))
    errs.extend(validate_segmentation(cfg))
    errs.extend(validate_quad(cfg))
    return errs


def _training_errors(cfg: TrapsigConfig) -> list[str]:
    """Rules for Model A. Kept separate so a features-only config is not judged by split rules."""
    errs: list[str] = []
    t = cfg.training
    if t.analysis_rate_hz <= 0:
        errs.append("training.analysis_rate_hz must be positive")
    elif t.analysis_rate_hz > cfg.sampling.original_hz:
        errs.append(
            f"training.analysis_rate_hz {t.analysis_rate_hz:g} exceeds sampling.original_hz "
            f"{cfg.sampling.original_hz:g}"
        )
    if not t.models:
        errs.append("training.models must list at least one model")
    known = {"logreg", "rf", "hgb"}
    for m in t.models:
        if m not in known:
            errs.append(f"training.models contains unknown model {m!r}; allowed: {sorted(known)}")
    if len(set(t.models)) != len(t.models):
        errs.append("training.models contains duplicates")
    if not t.lr_c_grid:
        errs.append("training.lr_c_grid must not be empty")
    if any(c <= 0 for c in t.lr_c_grid):
        errs.append("training.lr_c_grid values must be positive")
    if t.inner_splits < 2:
        errs.append("training.inner_splits must be at least 2")
    if t.n_permutations < 0:
        errs.append("training.n_permutations must be non-negative")
    if t.bootstrap_resamples < 0:
        errs.append("training.bootstrap_resamples must be non-negative")
    if len(t.psd_bands_hz) == 0:
        errs.append("training.psd_bands_hz must list at least one band")
    for i, band in enumerate(t.psd_bands_hz):
        if len(band) != 2 or band[0] >= band[1] or band[0] < 0:
            errs.append(f"training.psd_bands_hz[{i}] must be a pair [lo, hi] with 0 <= lo < hi")
        elif band[1] > t.analysis_rate_hz / 2:
            errs.append(
                f"training.psd_bands_hz[{i}] upper edge {band[1]:g} Hz exceeds the Nyquist frequency "
                f"{t.analysis_rate_hz / 2:g} Hz of training.analysis_rate_hz"
            )
    if any(lag <= 0 for lag in t.acf_lags_ms):
        errs.append("training.acf_lags_ms values must be positive")

    from .features import FEATURE_GROUPS

    if len(t.feature_groups) == 0:
        errs.append("training.feature_groups must list at least one group (or 'all')")
    unknown = [g for g in t.feature_groups if str(g).strip().lower() not in (*FEATURE_GROUPS, "all")]
    if unknown:
        errs.append(f"training.feature_groups has unknown group(s) {unknown}; "
                    f"known groups: {', '.join(FEATURE_GROUPS)} or 'all'")
    if t.ablation not in ("none", "groups", "leave_one_out", "both"):
        errs.append("training.ablation must be one of none | groups | leave_one_out | both")
    if t.ablation != "none" and t.ablation_model not in t.models:
        errs.append(f"training.ablation_model {t.ablation_model!r} must be one of "
                    f"training.models {t.models}, so the ablation uses a model this run fits")
    if not 0.0 < t.redundancy_threshold <= 1.0:
        errs.append("training.redundancy_threshold must be in (0, 1]")
    if t.hpo not in ("none", "random", "halving"):
        errs.append("training.hpo must be one of none | random | halving")
    if t.hpo_metric not in ("balanced_accuracy", "macro_f1", "roc_auc_ovr", "log_loss"):
        errs.append("training.hpo_metric must be one of balanced_accuracy | macro_f1 | "
                    "roc_auc_ovr | log_loss")
    if t.hpo_n_iter < 1:
        errs.append("training.hpo_n_iter must be at least 1")
    if t.hpo_seed_offset < 0:
        errs.append("training.hpo_seed_offset must be non-negative")
    if t.hpo != "none" and t.inner_splits < 2:
        errs.append("training.hpo needs training.inner_splits >= 2 to score a trial")
    return errs


def _deep_errors(cfg: TrapsigConfig) -> list[str]:
    """Rules for Model B and the hybrid combination."""
    errs: list[str] = []
    d = cfg.deep
    if not d.enabled:
        return errs
    if d.window_rate_hz is None:
        rate = cfg.training.analysis_rate_hz
    else:
        rate = d.window_rate_hz
        if rate <= 0:
            errs.append("deep.window_rate_hz must be positive")
        elif rate > cfg.training.analysis_rate_hz:
            errs.append(
                f"deep.window_rate_hz {rate:g} exceeds training.analysis_rate_hz "
                f"{cfg.training.analysis_rate_hz:g}; windows are cut from the analysed signal"
            )
    if d.window_ms != "auto" and d.window_ms <= 0:
        errs.append("deep.window_ms must be positive or 'auto'")
    elif d.window_ms != "auto":
        n = int(round(float(d.window_ms) / 1000.0 * rate))
        if n < 16:
            errs.append(
                f"deep.window_ms {d.window_ms:g} gives only {n} samples at {rate:g} Hz; "
                "a window that short cannot carry a conv stack"
            )
    if d.window_ms_min <= 0 or d.window_ms_max <= 0 or d.window_ms_min > d.window_ms_max:
        errs.append("deep.window_ms_min and deep.window_ms_max must be positive with min <= max")
    if d.window_ms_auto_multiplier <= 0:
        errs.append("deep.window_ms_auto_multiplier must be positive")
    if not 0.0 <= d.pre_event_fraction < 1.0:
        errs.append("deep.pre_event_fraction must be in [0, 1)")
    if d.max_windows_per_file < 1:
        errs.append("deep.max_windows_per_file must be at least 1")
    if d.channels < 1 or d.n_blocks < 1:
        errs.append("deep.channels and deep.n_blocks must be at least 1")
    if d.kernel_size < 3 or d.kernel_size % 2 == 0:
        errs.append("deep.kernel_size must be an odd integer >= 3")
    if d.stem_stride < 1:
        errs.append("deep.stem_stride must be at least 1")
    if not 0.0 <= d.dropout < 1.0:
        errs.append("deep.dropout must be in [0, 1)")
    if d.epochs < 1:
        errs.append("deep.epochs must be at least 1")
    if d.batch_size < 1:
        errs.append("deep.batch_size must be at least 1")
    if d.learning_rate <= 0:
        errs.append("deep.learning_rate must be positive")
    if d.weight_decay < 0:
        errs.append("deep.weight_decay must be non-negative")
    if d.warmup_epochs < 0 or d.warmup_epochs >= d.epochs:
        errs.append("deep.warmup_epochs must be in [0, epochs)")
    if d.early_stopping_patience < 1:
        errs.append("deep.early_stopping_patience must be at least 1")
    if d.validation_fraction <= 0 or d.validation_fraction >= 1:
        errs.append("deep.validation_fraction must be in (0, 1)")
    if d.metric_loss not in ("none", "supcon", "prototypical"):
        errs.append("deep.metric_loss must be one of none | supcon | prototypical")
    if not 0.0 <= d.metric_loss_weight <= 10.0:
        errs.append("deep.metric_loss_weight must be in [0, 10]")
    if d.metric_loss_weight == 0 and d.metric_loss != "none":
        errs.append("deep.metric_loss_weight is 0 but deep.metric_loss is "
                    f"{d.metric_loss!r}; set metric_loss: none to disable it")
    if d.metric_loss != "none" and d.metric_temperature <= 0:
        errs.append("deep.metric_temperature must be positive")
    if d.proto_distance not in ("cosine", "euclidean"):
        errs.append("deep.proto_distance must be cosine or euclidean")
    if d.prototypical_head and d.metric_temperature <= 0:
        errs.append("deep.metric_temperature must be positive when deep.prototypical_head is true: "
                    "the prototype head is a softmax over negative distances at that temperature")
    if d.fusion not in ("none", "blend", "stack", "both"):
        errs.append("deep.fusion must be one of none | blend | stack | both")
    if d.hpo_trials < 0:
        errs.append("deep.hpo_trials must be non-negative")
    if d.hpo_trials > 0:
        if not d.hpo_space_learning_rate or any(v <= 0 for v in d.hpo_space_learning_rate):
            errs.append("deep.hpo_space_learning_rate must list positive values")
        if not d.hpo_space_dropout or any(not 0.0 <= v < 1.0 for v in d.hpo_space_dropout):
            errs.append("deep.hpo_space_dropout must list values in [0, 1)")
        if not d.hpo_space_weight_decay or any(v < 0 for v in d.hpo_space_weight_decay):
            errs.append("deep.hpo_space_weight_decay must list non-negative values")
    if d.fusion == "none" and d.model_set == "hybrid":
        errs.append("deep.model_set is 'hybrid' but deep.fusion is 'none', so nothing would be "
                    "combined; set deep.fusion to blend | stack | both, or deep.model_set to "
                    "'deep' to report the two models separately")
    if d.n_permutations < 0:
        errs.append("deep.n_permutations must be non-negative")
    if d.combine == "equal_weight" and not 0.0 <= d.model_a_weight <= 1.0:
        errs.append("deep.model_a_weight must be in [0, 1]")
    if cfg.deep.model_set == "features":
        errs.append("deep.model_set is 'features' but deep.enabled is true; set deep.enabled: false")
    return errs


def _advanced_errors(cfg: TrapsigConfig) -> list[str]:
    errs: list[str] = []
    a = cfg.advanced
    if a.few_shot:
        if not cfg.deep.enabled:
            errs.append("advanced.few_shot needs the CNN encoder; set deep.enabled: true")
        if not a.few_shot_k or any(k < 1 for k in a.few_shot_k):
            errs.append("advanced.few_shot_k must list positive integers")
        if a.few_shot_episodes < 1:
            errs.append("advanced.few_shot_episodes must be at least 1")
    if a.semi_supervised:
        if not cfg.deep.enabled:
            errs.append("advanced.semi_supervised needs the CNN encoder; set deep.enabled: true")
        if not 0.0 < a.pseudo_confidence_tau <= 1.0:
            errs.append("advanced.pseudo_confidence_tau must be in (0, 1]")
        if not 0.0 <= a.pseudo_margin_delta < 1.0:
            errs.append("advanced.pseudo_margin_delta must be in [0, 1)")
        if a.pseudo_rounds < 1:
            errs.append("advanced.pseudo_rounds must be at least 1")
    if a.novelty:
        if not cfg.deep.enabled:
            errs.append("advanced.novelty needs the CNN encoder; set deep.enabled: true")
        if not 0.0 < a.novelty_known_retention < 1.0:
            errs.append("advanced.novelty_known_retention must be in (0, 1)")
    if a.clustering and (not a.clustering_k or any(k < 2 for k in a.clustering_k)):
        errs.append("advanced.clustering_k must list integers >= 2")
    return errs


def _explain_errors(cfg: TrapsigConfig) -> list[str]:
    errs: list[str] = []
    x = cfg.explain
    if not x.enabled:
        return errs
    if x.shap_background < 1:
        errs.append("explain.shap_background must be at least 1")
    if x.integrated_gradients_steps < 1:
        errs.append("explain.integrated_gradients_steps must be at least 1")
    if x.occlusion_segments < 2:
        errs.append("explain.occlusion_segments must be at least 2")
    if x.n_explained_windows < 1:
        errs.append("explain.n_explained_windows must be at least 1")
    if (x.grad_cam_1d or x.integrated_gradients or x.occlusion) and not cfg.deep.enabled:
        errs.append("explain.grad_cam_1d / integrated_gradients / occlusion need deep.enabled: true")
    return errs


def _report_errors(cfg: TrapsigConfig) -> list[str]:
    errs: list[str] = []
    r = cfg.report
    if r.dpi < 50 or r.dpi > 1200:
        errs.append("report.dpi must be between 50 and 1200")
    if r.max_plots_per_class < 0:
        errs.append("report.max_plots_per_class must be non-negative")
    return errs


def _dat_errors(f: DatFormat) -> list[str]:
    errs: list[str] = []
    if f.header_bytes < 0:
        errs.append("data.dat.header_bytes must be non-negative")
    if f.header_lines < 0:
        errs.append("data.dat.header_lines must be non-negative")
    if f.n_columns < 1:
        errs.append("data.dat.n_columns must be at least 1")
    if not 0 <= f.column < f.n_columns:
        errs.append("data.dat.column must be in [0, n_columns)")
    if f.scale == 0:
        errs.append("data.dat.scale must not be zero")
    if not f.units.strip():
        errs.append("data.dat.units must not be blank")
    if f.encoding == "binary" and f.header_lines:
        errs.append("data.dat.header_lines is ignored for binary encoding")
    if f.encoding == "text" and f.header_bytes:
        errs.append("data.dat.header_bytes is ignored for text encoding")
    return errs
