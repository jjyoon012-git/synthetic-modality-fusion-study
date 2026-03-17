"""
suppress_rebalance_study.py

2026-03-17
Study implementation inspired by:
"Suppress and Rebalance: Towards Generalized Multi-Modal Face Anti-Spoofing" (CVPR 2024)

[This code is adapted for the user's task]
- Doppler image: relatively reliable modality
- synthetic B-mode: partially unreliable modality (RePaint / copy-paste / inpaint artifacts)
- repaint_mask: external prior for low-reliability regions

Main ideas implemented:
(1) Doppler image -> Doppler patch tokens
(2) B-mode image -> B-mode patch tokens
(3) repaint_mask -> patch-level repaint prior
(4) B-mode token -> MC Dropout based token uncertainty (UEM-like)
(5) B-mode token + repaint prior + uncertainty -> reliability prediction
(6) Doppler query, B-mode key/value, reliability-gated cross-attention
(7) B-mode global structure bypass
(8) classifier -> logits
(9) classification + reliability regularization + prototype loss
(10) simplified ReGrad step for modality imbalance/conflict

This is NOT an official reproduction.
This is a study-oriented prototype with heavy comments.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F



# Utilities: mask processing, TV loss, gradient operations, etc.
def patch_mask_from_pixel_mask(mask: torch.Tensor, patch_size: int) -> torch.Tensor:
    """
    Convert pixel-level mask [B,1,H,W] to patch-level mask [B,N].

    mask value:
        1 = repaint / copy-paste / unreliable region
        0 = clean / normal region
    """
    B, C, H, W = mask.shape
    assert C == 1, "mask channel must be 1"
    assert H % patch_size == 0 and W % patch_size == 0, "H/W must be divisible by patch_size"

    pooled = F.avg_pool2d(mask.float(), kernel_size=patch_size, stride=patch_size)  # [B,1,H/P,W/P]
    return pooled.flatten(2).squeeze(1)  # [B,N]


def token_grid_shape(h: int, w: int, patch_size: int) -> Tuple[int, int]:
    """
    Utility to recover token grid shape.
    """
    assert h % patch_size == 0 and w % patch_size == 0
    return h // patch_size, w // patch_size


def total_variation_2d_from_tokens(x: torch.Tensor, gh: int, gw: int) -> torch.Tensor:
    """
    Smoothness loss on patch reliability map.
    x: [B,N], where N = gh * gw
    """
    B, N = x.shape
    assert N == gh * gw, f"Token count {N} != {gh}*{gw}"

    x2d = x.view(B, gh, gw)
    tv_h = torch.abs(x2d[:, 1:, :] - x2d[:, :-1, :]).mean()
    tv_w = torch.abs(x2d[:, :, 1:] - x2d[:, :, :-1]).mean()
    return tv_h + tv_w


def grad_list_dot(g1: List[Optional[torch.Tensor]], g2: List[Optional[torch.Tensor]]) -> torch.Tensor:
    """
    Dot product between two gradient lists.
    """
    dot = None
    for a, b in zip(g1, g2):
        if a is None or b is None:
            continue
        cur = (a * b).sum()
        dot = cur if dot is None else dot + cur
    if dot is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dot = torch.tensor(0.0, device=device)
    return dot


def grad_list_norm_sq(g: List[Optional[torch.Tensor]]) -> torch.Tensor:
    """
    Squared norm of gradient list.
    """
    val = None
    for x in g:
        if x is None:
            continue
        cur = (x * x).sum()
        val = cur if val is None else val + cur
    if val is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        val = torch.tensor(0.0, device=device)
    return val


def project_conflicting_gradient(
    g_fast: List[Optional[torch.Tensor]],
    g_slow: List[Optional[torch.Tensor]],
) -> List[Optional[torch.Tensor]]:
    """
    Project fast modality gradient to reduce conflict with slow modality gradient.

    If dot(g_fast, g_slow) < 0:
        g_fast' = g_fast - proj_{g_slow}(g_fast)

    This is a simplified study implementation of the "rebalance" idea.
    """
    dot = grad_list_dot(g_fast, g_slow)
    if dot >= 0:
        return g_fast

    denom = grad_list_norm_sq(g_slow).clamp(min=1e-12)
    coeff = dot / denom

    g_proj = []
    for gf, gs in zip(g_fast, g_slow):
        if gf is None:
            g_proj.append(None)
        elif gs is None:
            g_proj.append(gf)
        else:
            g_proj.append(gf - coeff * gs)
    return g_proj


# Central Difference Convolution (CDC)
# 논문에서 local subtle trace를 더 잘 보기 위한 핵심 중 하나
class CentralDifferenceConv2d(nn.Module):
    """
    Study version of CDC (Central Difference Convolution).

    Standard conv output - theta * central response
    This tends to emphasize local difference / edge-like subtle artifacts.

    Useful for:
    - repaint seam
    - copy-paste boundary
    - texture mismatch
    - local artifact cue
    """
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        bias: bool = True,
        theta: float = 0.7,
    ):
        super().__init__()
        self.theta = theta
        self.conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # standard conv
        out_normal = self.conv(x)

        # central difference term
        # sum over spatial kernel weights -> [out_ch, in_ch, 1, 1]
        if abs(self.theta) < 1e-8:
            return out_normal

        kernel_diff = self.conv.weight.sum(dim=(2, 3), keepdim=True)
        out_diff = F.conv2d(
            x,
            kernel_diff,
            bias=None,
            stride=self.conv.stride,
            padding=0,
            groups=1,
        )

        return out_normal - self.theta * out_diff

# Simple Patch Encoder
class SimplePatchEncoder(nn.Module):
    """
    Simple CNN -> patch tokens

    Input:
        [B,C,H,W]
    Output:
        tokens: [B,N,D]
        feat_map: [B,D,H/P,W/P]

    Notes:
    - For B-mode, CDC branch is optionally added to preserve local artifact cue
    - For study purpose, we keep it simple and readable
    """
    def __init__(
        self,
        in_ch: int,
        dim: int = 128,
        patch_size: int = 16,
        use_cdc: bool = False,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.use_cdc = use_cdc

        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, dim // 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dim // 2),
            nn.GELU(),
        )

        self.patch_proj = nn.Conv2d(dim // 2, dim, kernel_size=patch_size, stride=patch_size)

        if use_cdc:
            self.cdc = CentralDifferenceConv2d(
                in_ch=in_ch,
                out_ch=dim // 2,
                kernel_size=3,
                stride=1,
                padding=1,
                theta=0.7,
            )
            self.cdc_bn = nn.BatchNorm2d(dim // 2)
            self.cdc_act = nn.GELU()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        base = self.stem(x)  # [B,dim/2,H,W]

        if self.use_cdc:
            local = self.cdc_act(self.cdc_bn(self.cdc(x)))  # [B,dim/2,H,W]
            base = base + local

        feat = self.patch_proj(base)  # [B,dim,H/P,W/P]
        tokens = feat.flatten(2).transpose(1, 2)  # [B,N,dim]
        return tokens, feat

# UEM-like token uncertainty estimator (MC Dropout)
# 논문 핵심: 단순 prior만이 아니라 token-wise uncertainty 추정
class TokenUncertaintyEstimator(nn.Module):
    """
    MC Dropout based token uncertainty estimator.

    Input:
        tokens: [B,N,D]

    Output:
        uncertainty: [B,N]   # variance-based token uncertainty
        proj_mean:  [B,N,D]  # optional averaged projected token

    Idea:
    - multiple stochastic forward passes
    - compute token-wise variance
    - larger variance => less reliable token
    """
    def __init__(
        self,
        dim: int,
        hidden_dim: int = 128,
        dropout_p: float = 0.2,
        mc_samples: int = 5,
    ):
        super().__init__()
        self.mc_samples = mc_samples
        self.dropout_p = dropout_p

        self.proj1 = nn.Linear(dim, hidden_dim)
        self.proj2 = nn.Linear(hidden_dim, dim)
        self.act = nn.GELU()

    def single_pass(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.proj1(tokens)
        x = self.act(x)
        # inference 시에도 stochastic하게 보기 위해 training=True 고정
        x = F.dropout(x, p=self.dropout_p, training=True)
        x = self.proj2(x)
        return x

    def forward(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        samples = []
        for _ in range(self.mc_samples):
            samples.append(self.single_pass(tokens))  # [B,N,D]

        stacked = torch.stack(samples, dim=0)  # [T,B,N,D]
        proj_mean = stacked.mean(dim=0)        # [B,N,D]
        proj_var = stacked.var(dim=0, unbiased=False)  # [B,N,D]

        # token-wise variance -> scalar uncertainty
        uncertainty = proj_var.mean(dim=-1)  # [B,N]
        return uncertainty, proj_mean

# Reliability head
# 입력: B-mode token + repaint prior + uncertainty
# 출력: patch reliability r_i in [0,1]
class ReliabilityHead(nn.Module):
    """
    Predict patch reliability in [0,1].

    Input:
        b_tokens:        [B,N,D]
        repaint_prior:   [B,N]   # external prior from repaint mask
        uncertainty:     [B,N]   # internal uncertainty from MC Dropout

    Output:
        reliability:     [B,N]

    Interpretation:
    - repaint_prior: "외부적으로 이 패치가 좀 수상함"
    - uncertainty:   "모델 내부적으로도 이 패치가 불안정함"
    - reliability:   "최종적으로 얼마나 믿을지"
    """
    def __init__(self, dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        b_tokens: torch.Tensor,
        repaint_prior: torch.Tensor,
        uncertainty: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat(
            [
                b_tokens,
                repaint_prior.unsqueeze(-1),
                uncertainty.unsqueeze(-1),
            ],
            dim=-1,
        )
        reliability = torch.sigmoid(self.net(x)).squeeze(-1)
        return reliability


# Reliability-gated cross-attention
# Query: Doppler
# Key/Value: synthetic B-mode
# reliability가 낮은 B-mode token은 attention에서 덜 기여하도록 gating
class ReliabilityGatedCrossAttention(nn.Module):
    """
    Reliability-gated cross-attention.

    Query: Doppler
    Key/Value: synthetic B-mode

    Gating strategy:
    - reliability가 낮은 B-mode token은 K/V magnitude를 줄임
    - 추가로 log(reliability)를 attention bias로 넣어 soft suppression
    - optional query modulation도 가능하게 열어둠
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        gate_kv: bool = True,
        gate_query: bool = False,
    ):
        super().__init__()
        assert dim % num_heads == 0

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.gate_kv = gate_kv
        self.gate_query = gate_query

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.out_proj = nn.Linear(dim, dim)
        self.out_drop = nn.Dropout(proj_drop)

    def _reshape_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        x = x.view(B, N, self.num_heads, self.head_dim)
        x = x.transpose(1, 2)  # [B,H,N,Dh]
        return x

    def forward(
        self,
        query_feat: torch.Tensor,                # [B,Nq,D]
        key_value_feat: torch.Tensor,            # [B,Nk,D]
        reliability_kv: torch.Tensor,            # [B,Nk]
        reliability_q: Optional[torch.Tensor] = None,  # [B,Nq] if needed
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q = self._reshape_heads(self.q_proj(query_feat))
        k = self._reshape_heads(self.k_proj(key_value_feat))
        v = self._reshape_heads(self.v_proj(key_value_feat))

        # optional query gating
        if self.gate_query and reliability_q is not None:
            rq = reliability_q.unsqueeze(1).unsqueeze(-1)  # [B,1,Nq,1]
            q = q * rq

        # key/value gating
        if self.gate_kv:
            rk = reliability_kv.unsqueeze(1).unsqueeze(-1)  # [B,1,Nk,1]
            k = k * rk
            v = v * rk

        # standard attention logits
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B,H,Nq,Nk]

        # reliability bias: unreliable token은 softmax에서 더 불리하게
        log_r = torch.log(reliability_kv.clamp(min=1e-6)).unsqueeze(1).unsqueeze(2)  # [B,1,1,Nk]
        attn = attn + log_r

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v  # [B,H,Nq,Dh]
        out = out.transpose(1, 2).contiguous().view(query_feat.size(0), query_feat.size(1), self.dim)
        out = self.out_proj(out)
        out = self.out_drop(out)

        # 평균 attention map도 같이 리턴 (시각화/디버깅용)
        attn_mean = attn.mean(dim=1)  # [B,Nq,Nk]
        return out, attn_mean

# Prototype head (SSP-like study version)
# 논문 SSP를 그대로 재현하진 않고,
# modality feature compactness / convergence speed 측정용으로 단순화
class PrototypeHead(nn.Module):
    """
    Learnable class prototypes for a given branch.

    This is a simplified study version of prototype-based monitoring.
    Used for:
    - feature compactness
    - rough modality convergence speed measurement
    """
    def __init__(self, num_classes: int, dim: int):
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(num_classes, dim) * 0.02)

    def forward(self, feat: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        feat:   [B,D]
        target: [B]
        """
        proto = self.prototypes[target]  # [B,D]
        return F.mse_loss(feat, proto)


# main: suppress + rebalance fusion model
class SuppressRebalanceFusionModel(nn.Module):
    """
    Main study model:
    - Doppler encoder
    - synthetic B-mode encoder (+ CDC)
    - repaint prior from repaint mask
    - MC Dropout token uncertainty (UEM-like)
    - learned reliability refinement
    - reliability-gated cross-attention
    - structure bypass
    - fused classifier + modality auxiliary classifiers
    - prototype heads for simplified SSP-like study
    """
    def __init__(
        self,
        num_classes: int = 3,
        dim: int = 128,
        patch_size: int = 16,
        num_heads: int = 4,
        alpha_prior: float = 0.6,
        structure_bypass_weight: float = 0.2,
        uncertainty_temperature: float = 5.0,
        mc_samples: int = 5,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.alpha_prior = alpha_prior
        self.structure_bypass_weight = structure_bypass_weight
        self.uncertainty_temperature = uncertainty_temperature

        # Doppler: relatively reliable modality
        self.doppler_encoder = SimplePatchEncoder(
            in_ch=3,
            dim=dim,
            patch_size=patch_size,
            use_cdc=False,
        )

        # B-mode: potentially unreliable modality -> CDC 추가
        self.bmode_encoder = SimplePatchEncoder(
            in_ch=1,
            dim=dim,
            patch_size=patch_size,
            use_cdc=True,
        )

        # uncertainty estimation module
        self.uem = TokenUncertaintyEstimator(
            dim=dim,
            hidden_dim=dim,
            dropout_p=0.2,
            mc_samples=mc_samples,
        )

        # reliability prediction
        self.rel_head = ReliabilityHead(dim=dim, hidden_dim=dim)

        # cross-modal fusion
        self.cross_attn = ReliabilityGatedCrossAttention(
            dim=dim,
            num_heads=num_heads,
            attn_drop=0.0,
            proj_drop=0.0,
            gate_kv=True,
            gate_query=False,
        )

        self.norm_d = nn.LayerNorm(dim)
        self.norm_b = nn.LayerNorm(dim)

        # fusion feed-forward
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

        # fused classifier
        self.cls_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, num_classes),
        )

        # modality auxiliary classifiers
        self.doppler_aux_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, num_classes),
        )

        self.bmode_aux_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, num_classes),
        )

        # prototype heads: modality convergence monitoring
        self.proto_fused = PrototypeHead(num_classes=num_classes, dim=dim)
        self.proto_d = PrototypeHead(num_classes=num_classes, dim=dim)
        self.proto_b = PrototypeHead(num_classes=num_classes, dim=dim)

    def get_regrad_parameters(self) -> List[nn.Parameter]:
        """
        Parameters to which simplified ReGrad will be applied.

        Study choice:
        - reliability head
        - cross attention
        - fusion FFN

        These are the layers where modality interaction actually happens.
        """
        params = []
        params += list(self.rel_head.parameters())
        params += list(self.cross_attn.parameters())
        params += list(self.ffn.parameters())
        return [p for p in params if p.requires_grad]

    def forward(
        self,
        x_doppler: torch.Tensor,     # [B,3,H,W]
        x_bmode: torch.Tensor,       # [B,1,H,W]
        repaint_mask: torch.Tensor,  # [B,1,H,W]
    ) -> Dict[str, torch.Tensor]:
        B, _, H, W = x_bmode.shape
        gh, gw = token_grid_shape(H, W, self.patch_size)

        # (1) modality encoders
        d_tokens, d_feat_map = self.doppler_encoder(x_doppler)  # [B,Nd,D], [B,D,H/P,W/P]
        b_tokens, b_feat_map = self.bmode_encoder(x_bmode)      # [B,Nb,D], [B,D,H/P,W/P]

        # (2) repaint prior from mask
        # patch_mask: 값이 1에 가까울수록 repaint 비율이 높은 patch
        patch_mask = patch_mask_from_pixel_mask(repaint_mask, self.patch_size)  # [B,N]
        repaint_prior = 1.0 - self.alpha_prior * patch_mask
        repaint_prior = repaint_prior.clamp(0.05, 1.0)

        # (3) token uncertainty (UEM-like)
        # variance가 클수록 uncertainty 큼
        # uncertainty -> reliability와 반비례
        b_norm = self.norm_b(b_tokens)
        token_uncertainty_raw, _ = self.uem(b_norm)  # [B,N]

        # 안정적으로 0~1 범위의 "uncertainty score"로 압축
        # 값이 클수록 더 불확실
        uncertainty_score = 1.0 - torch.exp(-self.uncertainty_temperature * token_uncertainty_raw)
        uncertainty_score = uncertainty_score.clamp(0.0, 1.0)

        # (4) reliability prediction
        # repaint prior + internal uncertainty + token feature
        reliability = self.rel_head(
            b_tokens=b_norm,
            repaint_prior=repaint_prior,
            uncertainty=uncertainty_score,
        )  # [B,N]

        # (5) cross attention
        # Doppler query / B-mode key-value
        # unreliable B-mode token은 suppression
        d_norm = self.norm_d(d_tokens)
        cross, attn_map = self.cross_attn(
            query_feat=d_norm,
            key_value_feat=b_norm,
            reliability_kv=reliability,
            reliability_q=None,
        )

        # residual fusion
        fused = d_tokens + cross
        fused = fused + self.ffn(fused)

        # (6) pooling
        # fused global + B-mode structure bypass
        # hard masking 대신 global structure는 살려둠
        d_global = d_tokens.mean(dim=1)    # [B,D]
        b_global = b_tokens.mean(dim=1)    # [B,D]
        fused_global = fused.mean(dim=1) + self.structure_bypass_weight * b_global

        # (7) logits
        logits = self.cls_head(fused_global)
        logits_d_aux = self.doppler_aux_head(d_global)
        logits_b_aux = self.bmode_aux_head(b_global)

        return {
            "logits": logits,
            "logits_d_aux": logits_d_aux,
            "logits_b_aux": logits_b_aux,
            "reliability": reliability,
            "uncertainty_score": uncertainty_score,
            "repaint_prior": repaint_prior,
            "patch_mask": patch_mask,
            "doppler_tokens": d_tokens,
            "bmode_tokens": b_tokens,
            "fused_tokens": fused,
            "doppler_global": d_global,
            "bmode_global": b_global,
            "fused_global": fused_global,
            "attn_map": attn_map,
            "grid_h": torch.tensor(gh, device=x_bmode.device),
            "grid_w": torch.tensor(gw, device=x_bmode.device),
            "d_feat_map": d_feat_map,
            "b_feat_map": b_feat_map,
        }

# Loss
@dataclass
class LossWeights:
    cls: float = 1.0
    aux_d: float = 0.3
    aux_b: float = 0.3
    prior: float = 0.4
    uncertainty_align: float = 0.3
    smooth: float = 0.05
    repaint_sep: float = 0.2
    proto_fused: float = 0.1
    proto_d: float = 0.05
    proto_b: float = 0.05

def suppress_rebalance_loss(
    model: SuppressRebalanceFusionModel,
    outputs: Dict[str, torch.Tensor],
    targets: torch.Tensor,
    weights: LossWeights = LossWeights(),
) -> Dict[str, torch.Tensor]:
    """
    Main loss terms.

    Included:
    - fused classification
    - modality auxiliary classification
    - reliability prior matching
    - reliability should be inversely aligned with uncertainty
    - reliability smoothness over patch grid
    - repaint region reliability separation
    - simplified prototype loss

    Note:
    simplified ReGrad itself is NOT in this function;
    it is applied in train_step() as an extra gradient operation.
    """
    logits = outputs["logits"]
    logits_d_aux = outputs["logits_d_aux"]
    logits_b_aux = outputs["logits_b_aux"]

    reliability = outputs["reliability"]
    uncertainty_score = outputs["uncertainty_score"]
    repaint_prior = outputs["repaint_prior"]
    patch_mask = outputs["patch_mask"]

    fused_global = outputs["fused_global"]
    d_global = outputs["doppler_global"]
    b_global = outputs["bmode_global"]

    gh = int(outputs["grid_h"].item())
    gw = int(outputs["grid_w"].item())

    # classification losses
    loss_cls = F.cross_entropy(logits, targets)
    loss_aux_d = F.cross_entropy(logits_d_aux, targets)
    loss_aux_b = F.cross_entropy(logits_b_aux, targets)

    # reliability should roughly follow repaint prior
    # early-stage study setting에서는 prior를 약한 teacher처럼 사용
    loss_prior = F.l1_loss(reliability, repaint_prior)

    target_rel_from_uncertainty = 1.0 - uncertainty_score
    loss_uncertainty_align = F.l1_loss(reliability, target_rel_from_uncertainty)

    loss_smooth = total_variation_2d_from_tokens(reliability, gh, gw)

    repaint_mean = (reliability * patch_mask).sum() / (patch_mask.sum() + 1e-6)
    clean_mask = 1.0 - patch_mask
    clean_mean = (reliability * clean_mask).sum() / (clean_mask.sum() + 1e-6)
    loss_repaint_sep = F.relu(repaint_mean - clean_mean + 0.1)

    loss_proto_fused = model.proto_fused(fused_global, targets)
    loss_proto_d = model.proto_d(d_global, targets)
    loss_proto_b = model.proto_b(b_global, targets)

    # main total loss
    total = (
        weights.cls * loss_cls
        + weights.aux_d * loss_aux_d
        + weights.aux_b * loss_aux_b
        + weights.prior * loss_prior
        + weights.uncertainty_align * loss_uncertainty_align
        + weights.smooth * loss_smooth
        + weights.repaint_sep * loss_repaint_sep
        + weights.proto_fused * loss_proto_fused
        + weights.proto_d * loss_proto_d
        + weights.proto_b * loss_proto_b
    )

    return {
        "loss": total,
        "loss_cls": loss_cls,
        "loss_aux_d": loss_aux_d,
        "loss_aux_b": loss_aux_b,
        "loss_prior": loss_prior,
        "loss_uncertainty_align": loss_uncertainty_align,
        "loss_smooth": loss_smooth,
        "loss_repaint_sep": loss_repaint_sep,
        "loss_proto_fused": loss_proto_fused,
        "loss_proto_d": loss_proto_d,
        "loss_proto_b": loss_proto_b,
    }

# Simplified ReGrad training step
def train_step_with_regrad(
    model: SuppressRebalanceFusionModel,
    optimizer: torch.optim.Optimizer,
    x_doppler: torch.Tensor,
    x_bmode: torch.Tensor,
    repaint_mask: torch.Tensor,
    targets: torch.Tensor,
    loss_weights: LossWeights = LossWeights(),
    regrad_strength_fast: float = 0.2,
    regrad_strength_slow: float = 0.2,
) -> Dict[str, float]:
    """
    Simplified ReGrad-style training step.

    Procedure:
    1) compute normal full loss and backward
    2) compute auxiliary modality gradients on interaction layers
    3) choose faster/slower modality using prototype loss (lower proto loss => faster)
    4) if gradients conflict, project fast gradient against slow gradient
    5) inject projected modality gradients into selected interaction parameters

    Notes:
    - This is a study approximation of the paper's rebalance idea
    - Not an exact official ReGrad reproduction
    """
    model.train()
    optimizer.zero_grad()

    outputs = model(x_doppler, x_bmode, repaint_mask)
    loss_dict = suppress_rebalance_loss(model, outputs, targets, weights=loss_weights)

    total_loss = loss_dict["loss"]

    # (1) normal backward for all losses
    total_loss.backward(retain_graph=True)

    # (2) choose parameters where modality interaction matters
    regrad_params = model.get_regrad_parameters()

    # auxiliary modality losses
    loss_d_aux = loss_dict["loss_aux_d"]
    loss_b_aux = loss_dict["loss_aux_b"]

    # prototype losses as rough convergence-speed indicators
    # lower prototype loss -> feature more compact -> faster modality
    proto_d = loss_dict["loss_proto_d"].detach()
    proto_b = loss_dict["loss_proto_b"].detach()

    # (3) compute grads of modality auxiliary losses
    # on the interaction layers
    grads_d = torch.autograd.grad(
        loss_d_aux,
        regrad_params,
        retain_graph=True,
        allow_unused=True,
    )
    grads_b = torch.autograd.grad(
        loss_b_aux,
        regrad_params,
        retain_graph=True,
        allow_unused=True,
    )

    # (4) determine fast / slow modality
    # Doppler가 더 compact하면 fast modality로 간주
    if proto_d < proto_b:
        fast_name = "doppler"
        slow_name = "bmode"
        g_fast = list(grads_d)
        g_slow = list(grads_b)
    else:
        fast_name = "bmode"
        slow_name = "doppler"
        g_fast = list(grads_b)
        g_slow = list(grads_d)

    # gradient conflict 완화
    g_fast_proj = project_conflicting_gradient(g_fast, g_slow)

    # (5) inject rebalanced gradients
    # 기존 total_loss.backward()로 쌓인 grad 위에 추가
    for p, gf, gs in zip(regrad_params, g_fast_proj, g_slow):
        if p.grad is None:
            p.grad = torch.zeros_like(p)

        if gf is not None:
            p.grad = p.grad + regrad_strength_fast * gf
        if gs is not None:
            p.grad = p.grad + regrad_strength_slow * gs

    optimizer.step()

    # logging
    logs = {
        "loss": float(total_loss.detach().item()),
        "loss_cls": float(loss_dict["loss_cls"].detach().item()),
        "loss_aux_d": float(loss_dict["loss_aux_d"].detach().item()),
        "loss_aux_b": float(loss_dict["loss_aux_b"].detach().item()),
        "loss_prior": float(loss_dict["loss_prior"].detach().item()),
        "loss_uncertainty_align": float(loss_dict["loss_uncertainty_align"].detach().item()),
        "loss_smooth": float(loss_dict["loss_smooth"].detach().item()),
        "loss_repaint_sep": float(loss_dict["loss_repaint_sep"].detach().item()),
        "loss_proto_fused": float(loss_dict["loss_proto_fused"].detach().item()),
        "loss_proto_d": float(loss_dict["loss_proto_d"].detach().item()),
        "loss_proto_b": float(loss_dict["loss_proto_b"].detach().item()),
        "fast_modality": 0.0 if fast_name == "doppler" else 1.0,  # log-friendly placeholder
    }
    return logs


@torch.no_grad()
def infer(
    model: SuppressRebalanceFusionModel,
    x_doppler: torch.Tensor,
    x_bmode: torch.Tensor,
    repaint_mask: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    model.eval()
    outputs = model(x_doppler, x_bmode, repaint_mask)
    probs = F.softmax(outputs["logits"], dim=-1)
    preds = probs.argmax(dim=-1)

    outputs["probs"] = probs
    outputs["preds"] = preds
    return outputs

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_classes = 3
    dim = 128
    patch_size = 16
    num_heads = 4
    batch_size = 4
    H = W = 224

    # model
    model = SuppressRebalanceFusionModel(
        num_classes=num_classes,
        dim=dim,
        patch_size=patch_size,
        num_heads=num_heads,
        alpha_prior=0.6,
        structure_bypass_weight=0.2,
        uncertainty_temperature=5.0,
        mc_samples=5,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

    # -------------------------------------------------
    # dummy batch
    # -------------------------------------------------
    x_d = torch.randn(batch_size, 3, H, W, device=device)
    x_b = torch.randn(batch_size, 1, H, W, device=device)

    # repaint 영역을 일부 랜덤 생성
    repaint_mask = (torch.rand(batch_size, 1, H, W, device=device) > 0.8).float()

    y = torch.randint(0, num_classes, (batch_size,), device=device)
    logs = train_step_with_regrad(
        model=model,
        optimizer=optimizer,
        x_doppler=x_d,
        x_bmode=x_b,
        repaint_mask=repaint_mask,
        targets=y,
        loss_weights=LossWeights(),
        regrad_strength_fast=0.2,
        regrad_strength_slow=0.2,
    )

    print("=== Suppress & Rebalance Study Demo ===")
    for k, v in logs.items():
        print(f"{k}: {v:.6f}")
    outputs = infer(model, x_d, x_b, repaint_mask)

    print("logits shape:", tuple(outputs["logits"].shape))
    print("probs shape:", tuple(outputs["probs"].shape))
    print("preds shape:", tuple(outputs["preds"].shape))
    print("reliability shape:", tuple(outputs["reliability"].shape))
    print("uncertainty_score shape:", tuple(outputs["uncertainty_score"].shape))
    print("patch_mask shape:", tuple(outputs["patch_mask"].shape))
    print("attn_map shape:", tuple(outputs["attn_map"].shape))


if __name__ == "__main__":
    main()