"""The four-branch deep architecture (Branches A-D, attention fusion, two heads).

Branch contract (all shapes at ``batch = B``, window length ``L``, analysis rate ``fs``):

* **A temporal**  ``[B, 1, L] -> [B, D]``  dilated residual 1D CNN with a multiscale head
  (dilations 1/4/8 in parallel), temporal attention over time steps, then one transformer
  encoder layer over the subsampled sequence and mean+max pooling. The receptive field is
  computed, not assumed: ``receptive_field_samples()`` walks kernels, strides and dilations.
* **B spectral**  ``[B, 1, L] -> [B, D]``  a *fixed* STFT front end (``torch.stft``, Hann window,
  magnitude, ``log1p``) feeding a *learned* 2D spectral CNN with frequency attention. The STFT is
  a signal transform, not a trainable layer; only the encoder on top of it learns. STFT gives
  uniform time/frequency resolution set by ``(nfft, hop)``; a wavelet scalogram would trade that
  for log-spaced resolution at higher cost - the wavelet view is therefore offered as engineered
  features and visualizations instead of a second learned front end.
* **C self-supervised**  not a branch of its own but a *training stage* for A and B: NT-Xent
  contrastive learning over augmentation views plus masked-span reconstruction of the raw signal.
  Pretraining sees only the outer fold's training files (never the test fold, never the inner
  validation files used for early stopping), so the held-out evaluation stays strict.
* **D engineered**  ``[B, F] -> [B, D]``  window-level engineered features (time, amplitude,
  dynamics, spectrum, entropy, envelope - see :func:`window_features`) through an MLP. NaNs are
  imputed with medians fitted on the training fold only; the imputer state travels with the fold.

Fusion: each enabled branch embedding is layer-normalised, becomes one token (plus a CLS token),
and a small transformer encoder mixes them; the CLS output is the shared discriminative embedding.
A per-sample softmax gate over the branch tokens reports *which* branch carried each window, which
is what the fusion-contribution figure draws. Disabled or ablated branches simply have no token.

Heads: **A** a linear softmax classifier (cross-entropy, optional focal gamma and balanced class
weights); **B** prototypes in the same embedding (cosine distance, temperature-scaled) plus a
supervised-contrastive term during fine-tuning, and episodic one-shot/few-shot evaluation in
:mod:`trapsig.fewshot`.

Stages (design section 7): 2 pretrain -> 3 fine-tune heads+fusion with branches frozen
(``warmup_epochs``) -> 4 joint optimisation of all branches -> 5 episodic metric-learning passes.
Checkpoint selection uses only the fold's inner-validation windows.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn.functional as Fn
from torch import nn

from . import featreg
from .config import TrapsigConfig

BRANCHES = ("temporal", "spectral", "engineered")


class QuadError(RuntimeError):
    """Raised when the four-branch model cannot be built or trained for a stated reason."""


# ---------------------------------------------------------------------------
# Segmentation and window-level engineered features (Branch D inputs)
# ---------------------------------------------------------------------------

def segment_signal(x: np.ndarray, fs: float, cfg: TrapsigConfig) -> np.ndarray:
    """Cut one recording into fixed-length windows: ``(n_windows, L)``.

    ``segmentation.window_s is None`` means one window per recording (the whole analysed file),
    which is the honest default for 20 s recordings: no artificial segment boundaries are
    invented. Overlapping windows use ``stride = L * (1 - overlap)`` unless ``stride_s`` is set.
    Short tails are padded (reflect by default) rather than dropped, and the padding is recorded
    in the window feature table through the finite-length statistics.
    """
    g = cfg.segmentation
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        raise QuadError("cannot segment an empty recording")
    L = int(round(g.window_s * fs)) if g.window_s else int(x.size)
    L = max(16, L)
    stride = max(1, int(round(g.stride_s * fs))) if g.stride_s else \
        max(1, int(round(L * (1.0 - g.overlap))))
    if x.size < L:
        pad = L - x.size
        if g.pad_mode == "zero":
            x = np.pad(x, (0, pad))
        elif g.pad_mode == "edge":
            x = np.pad(x, (0, pad), mode="edge")
        else:
            x = np.pad(x, (0, pad), mode="reflect")
    starts = list(range(0, max(1, x.size - L + 1), stride))
    if not starts:
        starts = [0]
    if len(starts) > g.max_windows_per_file:
        idx = np.linspace(0, len(starts) - 1, g.max_windows_per_file).round().astype(int)
        starts = [starts[i] for i in idx]
    out = np.stack([x[s:s + L] for s in starts])
    if g.detrend == "constant":
        out = out - out.mean(axis=1, keepdims=True)
    elif g.detrend == "linear":
        t = np.arange(out.shape[1], dtype=float)
        for i in range(out.shape[0]):
            slope, icept = np.polyfit(t, out[i], 1)
            out[i] -= slope * t + icept
    if g.normalize == "zscore":
        sd = out.std(axis=1, keepdims=True)
        out = (out - out.mean(axis=1, keepdims=True)) / np.where(sd > 0, sd, 1.0)
    elif g.normalize == "robust":
        med = np.median(out, axis=1, keepdims=True)
        iqr = np.percentile(out, 75, axis=1, keepdims=True) - np.percentile(out, 25, axis=1,
                                                                           keepdims=True)
        out = (out - med) / np.where(iqr > 0, iqr, 1.0)
    return out


def _perm_entropy(x: np.ndarray, order: int = 3) -> float:
    """Backwards-compatible alias: permutation entropy lives in the feature registry now."""
    return featreg.perm_entropy(x, order)


def _sample_entropy(x: np.ndarray, m: int = 2, r_frac: float = 0.2) -> float:
    """Backwards-compatible alias: sample entropy lives in the feature registry now."""
    return featreg.sample_entropy(x, m, r_frac)


def window_features(w: np.ndarray, fs: float, bands: list[list[float]],
                    tiers: tuple[int, ...] | list[int] = (1, 2),
                    families: list[str] | None = None) -> dict[str, float]:
    """Window-level engineered features: the Branch D vocabulary from the feature registry.

    Delegates to :mod:`trapsig.featreg`, which holds the registry of 150+ candidates in three
    tiers and twelve families with names, descriptions, units and parameters. Every entry is
    label-blind and NaN when undefined for this window (e.g. no peak). The names are
    deterministic so the sidecar of a saved model and the CSV twins stay comparable. The
    recording level is the single-window case (``segmentation.window_s: null``); recording-level
    summaries across windows are produced by :func:`trapsig.featreg.recording_summary`.
    """
    from . import featreg

    return featreg.compute_features(w, fs, bands, tiers, families)


def window_feature_names(bands: list[list[float]], tiers: tuple[int, ...] | list[int] = (1, 2),
                         families: list[str] | None = None) -> list[str]:
    """The deterministic column order of :func:`window_features`."""
    from . import featreg

    return featreg.feature_names(bands, tiers, families)


# ---------------------------------------------------------------------------
# Branch encoders
# ---------------------------------------------------------------------------

class _ResBlock1d(nn.Module):
    def __init__(self, ch: int, kernel: int, dilation: int, dropout: float):
        super().__init__()
        pad = dilation * (kernel - 1) // 2
        self.c1 = nn.Conv1d(ch, ch, kernel, padding=pad, dilation=dilation)
        self.c2 = nn.Conv1d(ch, ch, kernel, padding=pad, dilation=dilation)
        self.gn = nn.GroupNorm(1, ch)
        self.gn2 = nn.GroupNorm(1, ch)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.drop(Fn.gelu(self.gn(self.c1(x))))
        h = self.gn2(self.c2(h))
        return Fn.gelu(h + x)


class TemporalAttention(nn.Module):
    """Global temporal attention: a 1x1 conv scores time steps, softmax, weighted sum."""

    def __init__(self, ch: int):
        super().__init__()
        self.score = nn.Conv1d(ch, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = torch.softmax(self.score(x), dim=-1)
        return (x * w).sum(dim=-1)


class TemporalEncoder(nn.Module):
    """Branch A: dilated residual CNN + multiscale head + temporal attention + transformer."""

    def __init__(self, q: Any, dropout: float):
        super().__init__()
        chs = list(q.t_channels)
        self.stem = nn.Conv1d(1, chs[0], q.t_kernel, padding=q.t_kernel // 2)
        self.stem_norm = nn.GroupNorm(1, chs[0])
        blocks, downs = [], []
        for i, ch in enumerate(chs):
            blocks.append(_ResBlock1d(ch, q.t_kernel, q.t_dilations[i % len(q.t_dilations)],
                                      dropout))
            downs.append(nn.Identity() if i == len(chs) - 1 else nn.Conv1d(ch, chs[i + 1], 3,
                                                                          stride=2, padding=1))
        self.blocks = nn.ModuleList(blocks)
        self.downs = nn.ModuleList(downs)
        ms = chs[-1] // 3 if chs[-1] % 3 == 0 else max(4, chs[-1] // 3)
        self.ms = nn.ModuleList([nn.Conv1d(chs[-1], ms, q.t_kernel, padding=d * (q.t_kernel - 1) // 2,
                                           dilation=d) for d in (1, 4, 8)])
        self.ms_mix = nn.Conv1d(3 * ms, chs[-1], 1)
        self.attn = TemporalAttention(chs[-1])
        self.pos = nn.Parameter(torch.zeros(1, 128, chs[-1]))
        self.tr = nn.TransformerEncoderLayer(chs[-1], q.t_nhead, dim_feedforward=2 * chs[-1],
                                             dropout=dropout, batch_first=True,
                                             norm_first=True)
        self.n_tr = q.t_transformer_layers
        self.out_ch = chs[-1]

    def features(self, x: torch.Tensor) -> torch.Tensor:
        h: torch.Tensor = Fn.gelu(self.stem_norm(self.stem(x)))
        for blk, dn in zip(self.blocks, self.downs, strict=True):
            h = dn(blk(h))
        h = self.ms_mix(torch.cat([m(h) for m in self.ms], dim=1))
        return h

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.features(x)                       # [B, C, T']
        att = self.attn(h)                         # [B, C]
        seq = h.permute(0, 2, 1)                   # [B, T', C]
        if seq.shape[1] > 128:
            idx = torch.linspace(0, seq.shape[1] - 1, 128).long()
            seq = seq[:, idx]
        seq = seq + self.pos[:, :seq.shape[1]]
        for _ in range(self.n_tr):
            seq = self.tr(seq)
        pooled = torch.cat([seq.mean(dim=1), seq.max(dim=1).values, att], dim=1)
        return pooled, h

    def receptive_field_samples(self, q: Any) -> int:
        rf, stride = 1, 1
        for i in range(len(q.t_channels)):
            d = q.t_dilations[i % len(q.t_dilations)]
            rf += 2 * (d * (q.t_kernel - 1) // 2) * stride
            rf += (q.t_kernel - 1) * stride
            if i < len(q.t_channels) - 1:
                stride *= 2
        return rf


class SpectralEncoder(nn.Module):
    """Branch B: fixed STFT front end, learned spectral CNN, frequency attention."""

    def __init__(self, q: Any, length: int, dropout: float):
        super().__init__()
        self.nfft, self.hop = int(q.s_nfft), int(q.s_hop)
        self.register_buffer("win", torch.hann_window(self.nfft), persistent=False)
        nf = self.nfft // 2 + 1
        chs = list(q.s_channels)
        layers: list[nn.Module] = []
        cin = 1
        for ch in chs:
            layers += [nn.Conv2d(cin, ch, 3, stride=2, padding=1), nn.GroupNorm(1, ch), nn.GELU(),
                       nn.Dropout2d(dropout)]
            cin = ch
        self.conv = nn.Sequential(*layers)
        # frequency attention scores each surviving frequency row from its channel profile
        # [B, F', C] -> [B, F', 1], so the linear layer reads C, the final channel count
        self.freq_attn = nn.Linear(cin, 1)
        self.out_ch = cin
        self._nf = nf

    def spectrogram(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x[:, 0]                 # (B, 1, L) -> (B, L): the STFT wants mono rows
        n = x.shape[-1]
        if n < self.nfft:
            x = Fn.pad(x, (0, self.nfft - n), mode="reflect")
        win: torch.Tensor = self.win if isinstance(self.win, torch.Tensor) \
            else torch.hann_window(self.nfft)
        spec = torch.stft(x, self.nfft, hop_length=self.hop, window=win, return_complex=True,
                          center=True)
        return torch.log1p(spec.abs().unsqueeze(1))          # [B, 1, F, T]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.spectrogram(x)
        h = self.conv(s)                                     # [B, C, F', T']
        # frequency attention: score each remaining frequency row, softmax, weighted sum over F'
        prof = h.mean(dim=-1).permute(0, 2, 1)               # [B, F', C]
        w = torch.softmax(self.freq_attn(prof), dim=1)       # [B, F', 1]
        att = (prof * w).sum(dim=1)                          # [B, C]
        pooled: torch.Tensor = torch.cat([h.mean(dim=(2, 3)), h.amax(dim=(2, 3)), att], dim=1)
        return pooled


class FeatureEncoder(nn.Module):
    """Branch D: MLP over the engineered window features (NaNs imputed by the caller)."""

    def __init__(self, n_feat: int, dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_feat, 2 * dim), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(2 * dim, dim))

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.net(f)
        return out


class CrossBranchFusion(nn.Module):
    """Cross-branch attention: branches as tokens + CLS, transformer mixing, softmax gate."""

    def __init__(self, q: Any, n_branches: int):
        super().__init__()
        d = q.embed_dim
        self.norms = nn.ModuleList([nn.LayerNorm(d) for _ in range(n_branches)])
        self.branch_pos = nn.Parameter(torch.randn(1, n_branches, d) * 0.02)
        self.cls = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.tr = nn.TransformerEncoderLayer(d, q.fusion_nhead, dim_feedforward=2 * d,
                                             dropout=q.dropout, batch_first=True, norm_first=True)
        self.n_layers = q.fusion_layers
        self.gate = nn.Linear(d, n_branches)

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``tokens``: [B, n_branches, D] in the enabled-branch order."""
        toks = torch.stack([n(t) for n, t in zip(self.norms, tokens.unbind(dim=1), strict=True)],
                           dim=1) + self.branch_pos
        b = toks.shape[0]
        seq = torch.cat([self.cls.expand(b, -1, -1), toks], dim=1)
        for _ in range(self.n_layers):
            seq = self.tr(seq)
        emb = seq[:, 0]
        gates = torch.softmax(self.gate(emb), dim=-1)
        return emb, gates


class _ReconHead(nn.Module):
    """Masked-span reconstruction decoder over the temporal feature map (Branch C objective)."""

    def __init__(self, ch: int, up: int):
        super().__init__()
        layers: list[nn.Module] = []
        for _ in range(up):
            layers += [nn.ConvTranspose1d(ch, ch, 4, stride=2, padding=1), nn.GELU()]
        layers += [nn.Conv1d(ch, 1, 3, padding=1)]
        self.net = nn.Sequential(*layers)

    def forward(self, h: torch.Tensor, length: int) -> torch.Tensor:
        y: torch.Tensor = self.net(h)
        if y.shape[-1] >= length:
            return y[..., :length]
        return Fn.pad(y, (0, length - y.shape[-1]))


class QuadModel(nn.Module):
    """The integrated four-branch model with its two heads."""

    def __init__(self, cfg: TrapsigConfig, n_classes: int, n_feat: int, length: int,
                 ablate: list[str] | None = None):
        super().__init__()
        q = cfg.quad
        ab = set(ablate or [])
        self.use_t = q.use_temporal and "temporal" not in ab
        self.use_s = q.use_spectral and "spectral" not in ab
        self.use_f = q.use_engineered and "engineered" not in ab
        if not (self.use_t or self.use_s or self.use_f):
            raise QuadError("every representation branch is disabled or ablated")
        d = q.embed_dim
        drop = q.dropout
        if self.use_t:
            self.temporal = TemporalEncoder(q, drop)
            self.proj_t = nn.Linear(3 * self.temporal.out_ch, d)
        if self.use_s:
            self.spectral = SpectralEncoder(q, length, drop)
            self.proj_s = nn.Linear(3 * self.spectral.out_ch, d)
        if self.use_f:
            self.feature_enc = FeatureEncoder(n_feat, d, drop)
        n_br = int(self.use_t) + int(self.use_s) + int(self.use_f)
        self.fusion = CrossBranchFusion(q, n_br)
        self.head_a = nn.Linear(d, n_classes)
        self.proto_scale = nn.Parameter(torch.tensor(1.0 / max(0.05, q.metric_temperature)))
        self.length = length
        self.n_branches = n_br
        if self.use_t:
            n_down = max(0, len(q.t_channels) - 1)
            self.recon = _ReconHead(q.t_channels[-1], n_down) if n_down else None
            self.up = n_down

    def branch_tokens(self, x: torch.Tensor, feats: torch.Tensor
                      ) -> tuple[list[torch.Tensor], torch.Tensor | None]:
        toks: list[torch.Tensor] = []
        hmap = None
        if x.dim() == 2:
            x = x.unsqueeze(1)          # callers may pass (B, L); encoders want (B, 1, L)
        if self.use_t:
            pooled, hmap = self.temporal(x)
            toks.append(self.proj_t(pooled))
        if self.use_s:
            toks.append(self.proj_s(self.spectral(x)))
        if self.use_f:
            toks.append(self.feature_enc(feats))
        return toks, hmap

    def embed(self, x: torch.Tensor, feats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        toks, _ = self.branch_tokens(x, feats)
        emb, gates = self.fusion(torch.stack(toks, dim=1))
        return emb, gates

    def forward(self, x: torch.Tensor, feats: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        emb, gates = self.embed(x, feats)
        return self.head_a(emb), emb, gates

    def proto_logits(self, emb: torch.Tensor, protos: torch.Tensor) -> torch.Tensor:
        emb = Fn.normalize(emb, dim=-1)
        protos = Fn.normalize(protos, dim=-1)
        return self.proto_scale * (emb @ protos.T)


# ---------------------------------------------------------------------------
# Losses, augmentations, self-supervised pretraining
# ---------------------------------------------------------------------------

def focal_ce(logits: torch.Tensor, y: torch.Tensor, gamma: float,
             weight: torch.Tensor | None) -> torch.Tensor:
    if gamma <= 0 and weight is None:
        return Fn.cross_entropy(logits, y)
    ce = Fn.cross_entropy(logits, y, weight=weight, reduction="none")
    p = torch.gather(torch.softmax(logits, dim=1), 1, y[:, None]).squeeze(1)
    return ((1 - p) ** gamma * ce).mean()


def triplet_loss(emb: torch.Tensor, y: torch.Tensor, margin: float = 0.2) -> torch.Tensor:
    """Batch-hard triplet loss on the shared embedding (Head B, optional third objective).

    For every anchor the hardest positive (same class) and hardest negative (other class) in the
    batch are used; anchors without a second same-class example contribute nothing. Default
    weight is 0 (off), so existing runs are unchanged unless the config opts in.
    """
    if emb.shape[0] < 3:
        return torch.zeros((), device=emb.device)
    d = torch.cdist(emb, emb)
    same = (y[:, None] == y[None, :]).float()
    eye = torch.eye(emb.shape[0], device=emb.device)
    hardest_pos = (d * same).max(dim=1).values
    hardest_neg = (d + (same + eye) * 1e6).min(dim=1).values
    loss = Fn.relu(hardest_pos - hardest_neg + margin)
    valid = same.sum(dim=1) > 1
    if not bool(valid.any()):
        return torch.zeros((), device=emb.device)
    return loss[valid].mean()


def supcon_loss(emb: torch.Tensor, y: torch.Tensor, temperature: float) -> torch.Tensor:
    """Supervised contrastive (Khosla et al. 2020) over the shared embedding."""
    z = Fn.normalize(emb, dim=-1)
    sim = z @ z.T / temperature
    n = y.shape[0]
    same = (y[:, None] == y[None, :])
    mask = same & ~torch.eye(n, dtype=torch.bool, device=y.device)
    if mask.sum() == 0:
        return torch.zeros((), device=y.device)
    logits = sim - torch.eye(n, device=y.device) * 1e9
    logp = Fn.log_softmax(logits, dim=1)
    return -(logp * mask).sum(dim=1)[mask.any(dim=1)].mean()


def augment(x: torch.Tensor, rng: np.random.Generator, mask_frac: float = 0.0
            ) -> tuple[torch.Tensor, torch.Tensor]:
    """Semantics-preserving views: amplitude scale, small additive noise, span masking.

    No time reversal or heavy warping: for trapping/translocation signals those change the
    physical event structure and would manufacture false contrastive pairs.
    """
    v = x * torch.tensor([float(rng.uniform(0.8, 1.2))], device=x.device)
    v = v + 0.02 * torch.randn_like(v) * v.std(dim=(1, 2), keepdim=True)
    mask = torch.zeros_like(v[:, 0, :], dtype=torch.bool)
    if mask_frac > 0:
        L = v.shape[-1]
        span = max(1, int(mask_frac * L))
        for i in range(v.shape[0]):
            s = int(rng.integers(0, max(1, L - span)))
            mask[i, s:s + span] = True
            v[i, 0, s:s + span] = 0.0
    return v, mask


def nt_xent(z1: torch.Tensor, z2: torch.Tensor, temperature: float) -> torch.Tensor:
    z = Fn.normalize(torch.cat([z1, z2], dim=0), dim=-1)
    n = z1.shape[0]
    sim = z @ z.T
    eye = torch.eye(2 * n, dtype=torch.bool, device=z.device)
    pos = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)
    sim = sim.masked_fill(eye, -1e9)
    return Fn.cross_entropy(sim / temperature, pos)


@dataclass
class PretrainRecord:
    epochs: int
    history: list[dict[str, float]] = field(default_factory=list)
    seconds: float = 0.0
    n_windows: int = 0
    objective: str = "nt_xent(views) + masked-span reconstruction"


def pretrain_selfsup(model: QuadModel, xw: torch.Tensor, q: Any, seed: int,
                     say: Any = None) -> PretrainRecord:
    """Stage 2: contrastive + masked reconstruction on the training fold's windows only."""
    tell = say or (lambda m: None)
    rec = PretrainRecord(epochs=int(q.pretrain_epochs), n_windows=int(xw.shape[0]))
    if rec.epochs <= 0 or xw.shape[0] < 4 or not model.use_t:
        rec.objective = rec.objective if model.use_t else "skipped: temporal branch disabled"
        return rec
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed + 11)
    opt = torch.optim.Adam([p for p in model.temporal.parameters()] +
                           [p for p in model.proj_t.parameters()] +
                           ([p for p in model.spectral.parameters()] +
                            [p for p in model.proj_s.parameters()] if model.use_s else []),
                           lr=q.learning_rate, weight_decay=q.weight_decay)
    n = xw.shape[0]
    bs = max(4, q.batch_size)
    for ep in range(rec.epochs):
        perm = torch.from_numpy(rng.permutation(n))
        losses = []
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            if idx.shape[0] < 4:
                continue
            xb = xw[idx]
            v1, m1 = augment(xb, rng, q.pretrain_mask_fraction)
            v2, _ = augment(xb, rng, 0.0)
            opt.zero_grad()
            loss = torch.zeros(())
            _, h1 = model.temporal(v1)
            _, h2 = model.temporal(v2)
            z1 = model.proj_t(_pool3(h1))
            z2 = model.proj_t(_pool3(h2))
            c = nt_xent(z1, z2, q.pretrain_temperature)
            loss = q.pretrain_contrastive_weight * c
            if model.recon is not None:
                rec_out = model.recon(h1, xb.shape[-1])
                denom = m1.sum().clamp_min(1)
                r = ((rec_out - xb) ** 2 * m1.unsqueeze(1)).sum() / denom
                loss = loss + q.pretrain_recon_weight * r
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        rec.history.append({"epoch": ep, "pretrain_loss": float(np.mean(losses)) if losses
                            else float("nan")})
        tell(f"quad pretrain epoch {ep + 1}/{rec.epochs}: loss "
             f"{rec.history[-1]['pretrain_loss']:.4f}")
    rec.seconds = round(time.perf_counter() - t0, 2)
    return rec


def _pool3(h: torch.Tensor) -> torch.Tensor:
    return torch.cat([h.mean(-1), h.max(-1).values, h.std(-1)], dim=1)


# ---------------------------------------------------------------------------
# Staged fold training and repeated CV
# ---------------------------------------------------------------------------

@dataclass
class QuadFoldResult:
    proba_test: np.ndarray
    proba_proto: np.ndarray
    emb_test: np.ndarray
    gates: np.ndarray
    history: list[dict[str, Any]]
    state: dict[str, Any]
    seconds: float
    n_parameters: int


@dataclass
class QuadCVResult:
    proba: np.ndarray                 # (n_repeats, n_files, k), NaN outside the mask
    proba_proto: np.ndarray
    mask: np.ndarray
    gates_mean: dict[str, float]
    logs: list[dict[str, Any]]
    fewshot_rows: list[dict[str, Any]]
    branch_names: list[str]
    seconds: float
    file_emb: dict[str, np.ndarray] = field(default_factory=dict)
    msp: np.ndarray | None = None
    notes: list[str] = field(default_factory=list)


def _imputer_state(feats: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    med = np.nanmedian(feats, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    iqr = np.nanpercentile(feats, 75, axis=0) - np.nanpercentile(feats, 25, axis=0)
    iqr = np.where(np.isfinite(iqr) & (iqr > 0), iqr, 1.0)
    return med, iqr


def _apply_imputer(feats: np.ndarray, med: np.ndarray, iqr: np.ndarray) -> np.ndarray:
    filled: np.ndarray = np.where(np.isfinite(feats), feats, med)
    scaled: np.ndarray = (filled - med) / iqr
    return scaled


def train_quad_fold(xtr: np.ndarray, ftr: np.ndarray, ytr: np.ndarray,
                    xva: np.ndarray, fva: np.ndarray, yva: np.ndarray,
                    xte: np.ndarray, fte: np.ndarray, k: int, classes: list[str],
                    cfg: TrapsigConfig, seed: int, run_dir: Any, tag: str, say: Any = None,
                    ablate: list[str] | None = None) -> QuadFoldResult:
    """Stages 2-5 for one outer fold. Validation windows drive checkpoint selection only."""
    from .deep import select_value

    tell = say or (lambda m: None)
    q = cfg.quad
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    no_improve = 0
    length = xtr.shape[1]
    model = QuadModel(cfg, k, ftr.shape[1], length, ablate)
    n_par = sum(p.numel() for p in model.parameters())
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(dev)
    Xtr = torch.tensor(xtr, dtype=torch.float32, device=dev).unsqueeze(1)
    Xva = torch.tensor(xva, dtype=torch.float32, device=dev).unsqueeze(1)
    Xte = torch.tensor(xte, dtype=torch.float32, device=dev).unsqueeze(1)
    Ftr = torch.tensor(ftr, dtype=torch.float32, device=dev)
    Fva = torch.tensor(fva, dtype=torch.float32, device=dev)
    Fte = torch.tensor(fte, dtype=torch.float32, device=dev)
    idx = {c: i for i, c in enumerate(classes)}
    ytr_t = torch.tensor([idx[str(v)] for v in ytr], dtype=torch.long, device=dev)

    history: list[dict[str, Any]] = []
    pre = PretrainRecord(epochs=0)
    if q.use_selfsup and q.pretrain_epochs > 0:
        pre = pretrain_selfsup(model, Xtr, q, seed, tell)
        history.append({"stage": "pretrain", "epochs": pre.epochs, "seconds": pre.seconds,
                        "history": pre.history})

    cw: torch.Tensor | None = None
    if q.class_weight == "balanced":
        counts = np.array([max(1, int((ytr == c).sum())) for c in classes], dtype=float)
        w = counts.sum() / (len(classes) * counts)
        cw = torch.tensor(w, dtype=torch.float32, device=dev)

    opt = torch.optim.AdamW(model.parameters(), lr=q.learning_rate, weight_decay=q.weight_decay)
    sched: Any = None
    if str(q.lr_schedule) == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, q.epochs))
    elif str(q.lr_schedule) == "step":
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, q.epochs // 3), gamma=0.5)
    n_tr = Xtr.shape[0]
    bs = max(4, min(q.batch_size, n_tr))
    best_score, best_state, best_ep = -math.inf, None, -1
    for ep in range(q.epochs):
        model.train()
        freeze = ep < q.warmup_epochs          # stage 3: heads + fusion; stage 4: joint
        if freeze:
            for mod in ([model.temporal, model.proj_t] if model.use_t else []) + \
                       ([model.spectral, model.proj_s] if model.use_s else []) + \
                       ([model.feature_enc] if model.use_f else []):
                for p in mod.parameters():
                    p.requires_grad_(False)
        else:
            for p in model.parameters():
                p.requires_grad_(True)
        for g in opt.param_groups:
            g["lr"] = q.learning_rate * (0.5 ** (ep // max(1, q.epochs)))
        perm = torch.from_numpy(rng.permutation(n_tr))
        tot, nloss = 0.0, 0
        for i in range(0, n_tr, bs):
            b = perm[i:i + bs]
            if b.shape[0] < 2:
                continue
            opt.zero_grad()
            logits, emb, _ = model(Xtr[b], Ftr[b])
            loss = focal_ce(logits, ytr_t[b], q.focal_gamma, cw)
            if q.supcon_weight > 0 and b.shape[0] >= 4:
                loss = loss + q.supcon_weight * supcon_loss(emb, ytr_t[b], q.metric_temperature)
            if q.triplet_weight > 0 and b.shape[0] >= 4:
                loss = loss + q.triplet_weight * triplet_loss(emb, ytr_t[b])
            loss.backward()
            opt.step()
            tot += float(loss.detach())
            nloss += 1
        model.eval()
        with torch.no_grad():
            lv, embv, _ = model(Xva, Fva)
            pv = torch.softmax(lv, dim=1).cpu().numpy()
            cidx = {c: i for i, c in enumerate(classes)}
            yva_i = np.array([cidx[str(v)] for v in np.asarray(yva).tolist()], dtype=int)
            va_pred_i = np.argmax(pv, axis=1)
            score = select_value(q.select_metric, yva_i, va_pred_i, pv, k)
        row = {"stage": "finetune", "epoch": ep, "joint": not freeze,
               "train_loss": tot / max(1, nloss), "val_score": score}
        history.append(row)
        if math.isfinite(score) and score > best_score:
            best_score, best_ep = score, ep
            best_state = {n: p.detach().cpu().clone() for n, p in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if sched is not None:
            sched.step()
        if q.early_stopping_patience > 0 and no_improve >= int(q.early_stopping_patience):
            history.append({"stage": "early_stop", "epoch": ep,
                            "reason": f"no validation improvement for "
                                      f"{q.early_stopping_patience} epochs"})
            break
        tell(f"quad {tag} epoch {ep + 1}/{q.epochs} ({'joint' if not freeze else 'heads'}): "
             f"loss {row['train_loss']:.4f}, val {q.select_metric} {score:.4f}")
    if best_state is not None:
        model.load_state_dict(best_state)
    ckpt_path = ""
    try:
        from pathlib import Path as _P

        ck_dir = _P(str(run_dir)) / "checkpoints"
        ck_dir.mkdir(parents=True, exist_ok=True)
        ckpt = ck_dir / f"quad_{tag}.pt"
        torch.save({"state_dict": {n: t.detach().cpu() for n, t in model.state_dict().items()},
                    "classes": list(classes), "tag": tag, "best_val_score": best_score,
                    "best_epoch": best_ep, "n_parameters": n_par}, ckpt)
        ckpt_path = str(ckpt)
    except Exception as exc:                                  # checkpointing must never kill a run
        tell(f"quad {tag}: checkpoint not written ({type(exc).__name__}: {exc})")

    # stage 5: episodic metric-learning passes on the training windows (support/query disjoint)
    if q.episodes > 0 and n_tr >= 8:
        ep_loss = _episodic_finetune(model, Xtr, Ftr, ytr, classes, q, rng, opt, tell, tag)
        history.append({"stage": "episodic", "mean_episode_loss": ep_loss})

    model.eval()
    with torch.no_grad():
        lt, embt, gt = model(Xte, Fte)
        pt = torch.softmax(lt, dim=1).cpu().numpy()
        emb_np = embt.cpu().numpy()
        gates = gt.cpu().numpy()
        protos = _prototypes(model, Xtr, Ftr, ytr, classes)
        lp = model.proto_logits(embt, protos.to(dev))
        pp = torch.softmax(lp, dim=1).cpu().numpy()
    names = [n for n, on in zip(BRANCHES, (model.use_t, model.use_s, model.use_f), strict=True)
             if on]
    state = {"receptive_field_samples": (model.temporal.receptive_field_samples(q)
                                         if model.use_t else None),
             "n_parameters": n_par, "best_epoch": best_ep, "best_val_score": best_score,
             "checkpoint": ckpt_path,
             "pretrain_seconds": pre.seconds, "pretrain_epochs_run": pre.epochs,
             "window_length_samples": length, "branches": names,
             "gate_mean": {n: float(gates[:, i].mean()) for i, n in enumerate(names)},
             "stages": "2 pretrain -> 3 heads/fusion -> 4 joint -> 5 episodic",
             "select_metric": q.select_metric}
    return QuadFoldResult(proba_test=pt, proba_proto=pp, emb_test=emb_np, gates=gates,
                          history=history, state=state,
                          seconds=round(time.perf_counter() - t0, 2), n_parameters=n_par)


def _prototypes(model: QuadModel, X: torch.Tensor, F: torch.Tensor, y: np.ndarray,
                classes: list[str]) -> torch.Tensor:
    with torch.no_grad():
        emb, _ = model.embed(X, F)
        emb = Fn.normalize(emb, dim=-1)
        out = []
        for c in classes:
            m = torch.tensor([str(v) == c for v in y], dtype=torch.bool, device=emb.device)
            out.append(emb[m].mean(dim=0) if m.any() else torch.zeros_like(emb[0]))
        return torch.stack(out)


def _episodic_finetune(model: QuadModel, X: torch.Tensor, F: torch.Tensor, y: np.ndarray,
                       classes: list[str], q: Any, rng: np.random.Generator,
                       opt: torch.optim.Optimizer, tell: Any, tag: str) -> float:
    """Episodic N-way K-shot passes: prototypes from support, cross-entropy on disjoint queries."""
    losses = []
    n_ep = max(1, q.episodes // 4)
    for _e in range(n_ep):
        support_i, query_i = [], []
        for c in classes:
            ids = np.flatnonzero(np.array([str(v) == c for v in y]))
            if ids.size < 3:
                continue
            pick = rng.choice(ids, size=min(2 + q.queries_per_class, ids.size), replace=False)
            support_i += list(pick[:2])
            query_i += list(pick[2:])
        if len(query_i) < 2 or len(set(y[support_i].tolist())) < 2:
            continue
        si = torch.tensor(support_i, dtype=torch.long, device=X.device)
        qi = torch.tensor(query_i, dtype=torch.long, device=X.device)
        model.train()
        opt.zero_grad()
        emb_s, _ = model.embed(X[si], F[si])
        emb_q, _ = model.embed(X[qi], F[qi])
        proto_list = []
        for c in sorted(set(y[support_i].tolist())):
            m = torch.tensor([str(v) == c for v in y[support_i]], dtype=torch.bool,
                             device=X.device)
            proto_list.append(Fn.normalize(emb_s, dim=-1)[m].mean(dim=0))
        protos = torch.stack(proto_list)
        names = sorted(set(y[support_i].tolist()))
        yq = torch.tensor([names.index(str(v)) for v in y[qi]], dtype=torch.long,
                          device=X.device)
        loss = Fn.cross_entropy(model.proto_logits(emb_q, protos), yq)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    if losses:
        tell(f"quad {tag}: episodic metric-learning {len(losses)} episodes, mean loss "
             f"{float(np.mean(losses)):.4f}")
    return float(np.mean(losses)) if losses else float("nan")


# ---------------------------------------------------------------------------
# Repeated grouped cross-validation over segmented windows
# ---------------------------------------------------------------------------

def build_quad_tensors(cache_dir: str, ok_rows: list[dict[str, Any]], cfg: TrapsigConfig,
                       say: Any = None) -> dict[str, Any] | None:
    """Segment every cached analysed recording and compute window-level engineered features.

    Returns None when the cache is missing or no file produced a window: the run then reports the
    quad model as not evaluated instead of failing.
    """
    from pathlib import Path

    tell = say or (lambda m: None)
    bands = [list(b) for b in cfg.training.psd_bands_hz]
    tiers = list(cfg.quad.feature_tiers) or [1, 2]
    fams = list(cfg.quad.feature_families) or None
    names: list[str] | None = None
    X, FE, YW, GW, SW = [], [], [], [], []
    rec_rows: list[dict[str, Any]] = []
    for row in ok_rows:
        path = Path(cache_dir) / f"{row['sample_id']}.npy"
        if not path.is_file():
            continue
        x = np.load(path)
        try:
            wins = segment_signal(x, float(cfg.training.analysis_rate_hz), cfg)
        except QuadError:
            continue
        feats = [window_features(w, float(cfg.training.analysis_rate_hz), bands, tiers, fams)
                 for w in wins]
        if feats:
            rec = featreg.recording_summary(feats)
            rec_rows.append({"sample_id": row["sample_id"], "class": row.get("class"),
                             **rec})
        if names is None:
            names = list(feats[0]) if feats else window_feature_names(bands)
        mat = np.full((len(feats), len(names)), np.nan)
        for i, f in enumerate(feats):
            for j, nm in enumerate(names):
                mat[i, j] = float(f[nm]) if nm in f else np.nan
        X.append(wins)
        FE.append(mat)
        YW += [row["class"]] * len(wins)
        GW += [row["sample_id"]] * len(wins)
        SW += [row["sample_id"]] * len(wins)
    if not X or names is None:
        return None
    out = {"x": np.concatenate(X), "feats": np.concatenate(FE),
           "y_win": np.array(YW, dtype=object), "groups": np.array(GW, dtype=object),
           "sids": np.array(SW, dtype=object), "feat_names": names,
           "rec_rows": rec_rows,
           "n_files": len({str(v) for v in SW})}
    tell(f"quad: {out['x'].shape[0]} windows from {out['n_files']} files, "
         f"{len(names)} engineered window features")
    return out


def cv_quad(tensors: dict[str, Any], sids: list[str], y_files: np.ndarray,
            fold_ids: np.ndarray, classes: list[str], cfg: TrapsigConfig, run_dir: Any,
            say: Any = None, ablate: list[str] | None = None) -> QuadCVResult:
    """File-level out-of-fold probabilities for the four-branch model.

    Leakage rules honoured here: segmentation and engineered features are label-blind; the
    imputer and the robust scale are fitted on the fold's training windows minus the inner
    validation windows; self-supervised pretraining sees the same set; the test fold contributes
    windows only at prediction time. Few-shot episodes draw support and query from *disjoint*
    test files of the same fold.
    """
    from . import fewshot
    from .deep import validation_split

    tell = say or (lambda m: None)
    q = cfg.quad
    t0 = time.perf_counter()
    x, FE = tensors["x"], tensors["feats"]
    yw, gw, sw = tensors["y_win"], tensors["groups"], tensors["sids"]
    k = len(classes)
    n_rep, n_files = fold_ids.shape
    proba = np.full((n_rep, n_files, k), np.nan)
    proba_proto = np.full((n_rep, n_files, k), np.nan)
    embs: dict[tuple[int, int], dict[str, np.ndarray]] = {}
    file_acc: dict[str, list[np.ndarray]] = {}
    gates_acc: dict[str, list[float]] = {}
    logs: list[dict[str, Any]] = []
    few_rows: list[dict[str, Any]] = []
    notes: list[str] = []
    sid_pos = {s: i for i, s in enumerate(sids)}
    for r in range(n_rep):
        for f in range(cfg.splitting.n_splits):
            te_f = fold_ids[r] == f
            te_w = np.array([bool(te_f[sid_pos[str(s)]]) for s in sw.tolist()])
            tr_w = ~te_w
            if te_w.sum() == 0 or tr_w.sum() == 0:
                notes.append(f"repeat {r} fold {f}: empty side, skipped")
                continue
            ytr_w = yw[tr_w]
            gtr_w = gw[tr_w]
            cidx = {c: i for i, c in enumerate(classes)}
            ytr_i = np.array([cidx[str(v)] for v in ytr_w.tolist()], dtype=int)
            va = validation_split(ytr_i, gtr_w, cfg.deep.validation_fraction,
                                  cfg.run.seed + 1000 * r + f, k)
            if va.sum() == 0 or (~va).sum() == 0:
                va = np.zeros(ytr_w.shape, dtype=bool)
                notes.append(f"repeat {r} fold {f}: no usable inner validation windows")
            keep_cols = np.arange(FE.shape[1])
            if str(q.feature_select) == "variance":
                tr_fe = FE[tr_w][~va]
                keep_cols = np.array([j for j in range(FE.shape[1])
                                      if np.any(np.isfinite(tr_fe[:, j]))
                                      and np.nanstd(tr_fe[:, j]) > 0], dtype=int)
                notes.append(f"repeat {r} fold {f}: train-only variance selection kept "
                             f"{keep_cols.size} of {FE.shape[1]} engineered columns")
                FE = FE[:, keep_cols]
            med, iqr = _imputer_state(FE[tr_w][~va])
            tag = f"r{r}f{f}"
            tell(f"quad CV: repeat {r + 1}/{n_rep} fold {f + 1}/{cfg.splitting.n_splits} "
                 f"({int((~va).sum())} train, {int(va.sum())} val, {int(te_w.sum())} test windows)")
            res = train_quad_fold(
                x[tr_w][~va], _apply_imputer(FE[tr_w][~va], med, iqr), ytr_w[~va],
                x[tr_w][va], _apply_imputer(FE[tr_w][va], med, iqr), ytr_w[va],
                x[te_w], _apply_imputer(FE[te_w], med, iqr), k, classes, cfg,
                cfg.run.seed + 1000 * r + f + q.seed_offset, run_dir, tag, tell, ablate)
            te_sids = [str(s) for s in sw[te_w].tolist()]
            for sid in set(te_sids):
                m = np.array([s == sid for s in te_sids])
                i = sid_pos[sid]
                proba[r, i] = res.proba_test[m].mean(axis=0)
                proba_proto[r, i] = res.proba_proto[m].mean(axis=0)
            embs[(r, f)] = {sid: res.emb_test[np.array([s == sid for s in te_sids])].mean(axis=0)
                            for sid in set(te_sids)}
            for sid, e in embs[(r, f)].items():
                file_acc.setdefault(sid, []).append(e)
            for nm, v in res.state["gate_mean"].items():
                gates_acc.setdefault(nm, []).append(v)
            logs.append({"repeat": r, "fold": f, "tag": tag, **res.state,
                         "history": res.history, "seconds": res.seconds})
            # few-shot / one-shot episodes on this fold's held-out files
            emb_files = np.stack([embs[(r, f)][s] for s in te_sids])
            y_files_fold = np.array([str(y_files[sid_pos[s]]) for s in te_sids], dtype=object)
            few_rows.extend(fewshot.episodic_eval(
                emb_files, y_files_fold, classes, q.shots, q.episodes, q.queries_per_class,
                cfg.run.seed + 31 * (r * cfg.splitting.n_splits + f), tag=tag))
    msp = np.nanmax(np.nanmean(proba, axis=0), axis=1)
    if not few_rows:
        notes.append(
            "no few-shot episodes were drawn: every episode needs at least "
            f"{2 + int(q.queries_per_class)} test files per class in a fold (2 to form a "
            "support+query pair plus the configured queries); reduce quad.queries_per_class, "
            "reduce splitting.n_splits, or add files")
    mask = np.zeros(n_files, dtype=bool)
    for i in range(n_files):
        mask[i] = all(np.isfinite(proba[r, i]).all() for r in range(n_rep)) \
            and any(np.isfinite(proba[r, i]).all() for r in range(n_rep))
    return QuadCVResult(proba=proba, proba_proto=proba_proto, mask=mask,
                        gates_mean={nm: float(np.mean(v)) for nm, v in gates_acc.items()},
                        logs=logs, fewshot_rows=few_rows,
                        file_emb={sid: np.mean(v, axis=0) for sid, v in file_acc.items()},
                        msp=msp, branch_names=[b for b in BRANCHES],
                        seconds=round(time.perf_counter() - t0, 2), notes=notes)
