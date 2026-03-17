'''
2026-03-17
(1) Doppler image -> Doppler patch tokens
(2) B-mode image -> B-mode patch tokens
(3) repaint_mask -> patch-level repaint prior
(4) B-mode token + prior -> reliability 예측
(5) Doppler query, B-mode key/value, reliability 반영 cross-attention
(6) fused token 생성
(7) B-mode global structure bypass 추가
(8) classifier -> logits
(9) classification + reliability regularization loss 계산
'''

from __future__ import annotations
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# Utilities: 픽셀 마스크 -> 패치 마스크(avg_pool2d)
def patch_mask_from_pixel_mask(mask: torch.Tensor, patch_size: int) -> torch.Tensor:
    """
    Convert pixel-level mask [B,1,H,W] to patch-level mask [B,N].
    mask value: 1 = repaint/copypaste region, 0 = normal region
    """
    B, C, H, W = mask.shape
    assert C == 1
    assert H % patch_size == 0 and W % patch_size == 0

    pooled = F.avg_pool2d(mask.float(), kernel_size=patch_size, stride=patch_size)  # [B,1,H/P,W/P]
    return pooled.flatten(2).squeeze(1)  # [B,N]

# Utilities: total variation for 1D signal (reliability along token sequence)
def total_variation_1d(x: torch.Tensor) -> torch.Tensor:
    """
    Compute total variation for 1D signal.
    x: [B,N]
    """
    return torch.mean(torch.abs(x[:, 1:] - x[:, :-1])) 

# Encoder: 이미지를 패치 토큰으로 변경
class SimplePatchEncoder(nn.Module):
    """
    Simple CNN -> 토큰 인코더
    Input: [B,C,H,W]
    Output: [B,N,D]
    """
    def __init__(self, in_ch: int, dim: int = 128, patch_size: int = 16):
        super().__init__()
        self.patch_size = patch_size
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, dim //2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dim //2),
            nn.GELU(),
            nn.Conv2d(dim // 2, dim, kernel_size=patch_size, stride=patch_size),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.conv(x) # [B,dim,H/P,W/P]
        tokens = feat.flatten(2).transpose(1, 2) # [B,N,dim]
        return tokens

# Reliability head: 패치별 신뢰도 r_i ∈ [0,1] 예측 방법 => B-mode 토큰 + repaint prior -> MLP -> sigmoid
class ReliabilityHead(nn.Module):
    """
    Predict patch reliability in [0,1].
    Input:
        b_tokens: [B,N,D]
        repaint_prior: [B,N]
    Output:
        reliability: [B,N]
    """
    def __init__(self, dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, b_tokens: torch.Tensor, repaint_prior: torch.Tensor) -> torch.Tensor:
        x = torch.cat([b_tokens, repaint_prior.unsqueeze(-1)], dim=-1)
        reliability = torch.sigmoid(self.net(x)).squeeze(-1)
        return reliability

# Reliability-gated cross-attention
class ReliabilityGatedCrossAttention(nn.Module):
    """
    Query: Doppler
    Key/Value: synthetic B-mode
    단, B-mode token마다 reliability가 있음
    reliability가 낮은 B-mode token은 attention에서 덜 기여하게 만든다
    """
    def __init__(self, dim: int, num_heads: int = 4, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # 초초초기화
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.out_proj = nn.Linear(dim, dim)
        self.out_drop = nn.Dropout(proj_drop)
    
    # mha 준비
    def _reshape_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        x = x.view(B, N, self.num_heads, self.head_dim)
        x = x.transpose(1, 2)  # [B,H,N,Dh]
        return x

    def forward(
        self,
        query_feat: torch.Tensor,      # [B,Nq,D]
        key_value_feat: torch.Tensor,  # [B,Nk,D]
        reliability: torch.Tensor,     # [B,Nk]
    ) -> torch.Tensor:
        # QKV 생성 + 헤드 분할
        q = self._reshape_heads(self.q_proj(query_feat))
        k = self._reshape_heads(self.k_proj(key_value_feat))
        v = self._reshape_heads(self.v_proj(key_value_feat))

        # reliability를 attention에 적용: k,v에 reliability 곱하기: reliability가 낮은 토큰은 k,v가 작아져서 attention에서 덜 기여하게 됨
        r = reliability.unsqueeze(1).unsqueeze(-1)  # [B,1,Nk,1]
        k = k * r
        v = v * r
        # attn 계산: QK^T / sqrt(dh) + log(reliability) as bias
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B,H,Nq,Nk]

        # add log reliability as attention bias
        log_r = torch.log(reliability.clamp(min=1e-6)).unsqueeze(1).unsqueeze(2)  # [B,1,1,Nk]
        attn = attn + log_r

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v  # [B,H,Nq,Dh]
        out = out.transpose(1, 2).contiguous().view(query_feat.size(0), query_feat.size(1), self.dim)
        out = self.out_proj(out)
        out = self.out_drop(out)
        return out

# Fusion model
class SuppressRebalanceFusionModel(nn.Module):
    """
    Main study model:
    - Doppler encoder
    - synthetic B-mode encoder
    - reliability prior from repaint mask
    - learned reliability refinement
    - reliability-gated cross-attention
    - structure bypass
    """
    def __init__(
        self,
        num_classes: int = 3,
        dim: int = 128,
        patch_size: int = 16,
        num_heads: int = 4,
        alpha_prior: float = 0.5,
        structure_bypass_weight: float = 0.2,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.alpha_prior = alpha_prior
        self.structure_bypass_weight = structure_bypass_weight

        self.doppler_encoder = SimplePatchEncoder(in_ch=3, dim=dim, patch_size=patch_size)
        self.bmode_encoder = SimplePatchEncoder(in_ch=1, dim=dim, patch_size=patch_size)

        self.rel_head = ReliabilityHead(dim=dim)
        self.cross_attn = ReliabilityGatedCrossAttention(dim=dim, num_heads=num_heads)

        self.norm_d = nn.LayerNorm(dim)
        self.norm_b = nn.LayerNorm(dim)

        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

        self.cls_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, num_classes),
        )

    def forward(
        self,
        x_doppler: torch.Tensor,     # [B,3,H,W]
        x_bmode: torch.Tensor,       # [B,1,H,W]
        repaint_mask: torch.Tensor,  # [B,1,H,W], 1 = repaint/copypaste region
    ):
        d_tokens = self.doppler_encoder(x_doppler)  # [B,Nd,D]
        b_tokens = self.bmode_encoder(x_bmode)      # [B,Nb,D]

        patch_mask = patch_mask_from_pixel_mask(repaint_mask, self.patch_size)  # [B,Nb]

        # hard masking is NOT used
        repaint_prior = 1.0 - self.alpha_prior * patch_mask
        repaint_prior = repaint_prior.clamp(0.05, 1.0)

        d_norm = self.norm_d(d_tokens)
        b_norm = self.norm_b(b_tokens)

        reliability = self.rel_head(b_norm, repaint_prior)  # [B,Nb]

        cross = self.cross_attn(d_norm, b_norm, reliability)
        fused = d_tokens + cross
        fused = fused + self.ffn(fused)

        # structure bypass: preserve global B-mode structure instead of zeroing it out
        b_global = b_tokens.mean(dim=1)
        fused_global = fused.mean(dim=1) + self.structure_bypass_weight * b_global

        logits = self.cls_head(fused_global)

        return {
            "logits": logits,
            "reliability": reliability,
            "repaint_prior": repaint_prior,
            "patch_mask": patch_mask,
            "doppler_tokens": d_tokens,
            "bmode_tokens": b_tokens,
            "fused_tokens": fused,
        }



# Loss
@dataclass
class LossWeights:
    cls: float = 1.0
    prior: float = 0.5
    smooth: float = 0.05
    repaint_sep: float = 0.2


def suppress_rebalance_loss(
    outputs: dict,
    targets: torch.Tensor,
    weights: LossWeights = LossWeights(),
) -> dict:
    logits = outputs["logits"]
    reliability = outputs["reliability"]
    repaint_prior = outputs["repaint_prior"]
    patch_mask = outputs["patch_mask"]

    loss_cls = F.cross_entropy(logits, targets)

    # reliability should roughly follow prior in early-stage study setting
    loss_prior = F.l1_loss(reliability, repaint_prior)

    # smooth reliability along token sequence
    loss_smooth = total_variation_1d(reliability)

    # repaint region should have lower reliability than clean region
    repaint_mean = (reliability * patch_mask).sum() / (patch_mask.sum() + 1e-6)
    clean_mask = 1.0 - patch_mask
    clean_mean = (reliability * clean_mask).sum() / (clean_mask.sum() + 1e-6)
    loss_repaint_sep = F.relu(repaint_mean - clean_mean + 0.1)

    # L=λcls​Lcls​+λprior​Lprior​+λsmooth​Lsmooth​+λsep​Lsep​
    total = (
        weights.cls * loss_cls
        + weights.prior * loss_prior
        + weights.smooth * loss_smooth
        + weights.repaint_sep * loss_repaint_sep
    )

    return {
        "loss": total,
        "loss_cls": loss_cls.detach(),
        "loss_prior": loss_prior.detach(),
        "loss_smooth": loss_smooth.detach(),
        "loss_repaint_sep": loss_repaint_sep.detach(),
    }

from __future__ import annotations
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# Utilities: 픽셀 마스크 -> 패치 마스크(avg_pool2d)
def patch_mask_from_pixel_mask(mask: torch.Tensor, patch_size: int) -> torch.Tensor:
    """
    Convert pixel-level mask [B,1,H,W] to patch-level mask [B,N].
    mask value: 1 = repaint/copypaste region, 0 = normal region
    """
    B, C, H, W = mask.shape
    assert C == 1
    assert H % patch_size == 0 and W % patch_size == 0

    pooled = F.avg_pool2d(mask.float(), kernel_size=patch_size, stride=patch_size)  # [B,1,H/P,W/P]
    return pooled.flatten(2).squeeze(1)  # [B,N]

# Utilities: total variation for 1D signal (reliability along token sequence)
def total_variation_1d(x: torch.Tensor) -> torch.Tensor:
    """
    Compute total variation for 1D signal.
    x: [B,N]
    """
    return torch.mean(torch.abs(x[:, 1:] - x[:, :-1])) 

# Encoder: 이미지를 패치 토큰으로 변경
class SimplePatchEncoder(nn.Module):
    """
    Simple CNN -> 토큰 인코더
    Input: [B,C,H,W]
    Output: [B,N,D]
    """
    def __init__(self, in_ch: int, dim: int = 128, patch_size: int = 16):
        super().__init__()
        self.patch_size = patch_size
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, dim //2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dim //2),
            nn.GELU(),
            nn.Conv2d(dim // 2, dim, kernel_size=patch_size, stride=patch_size),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.conv(x) # [B,dim,H/P,W/P]
        tokens = feat.flatten(2).transpose(1, 2) # [B,N,dim]
        return tokens

# Reliability head: 패치별 신뢰도 r_i ∈ [0,1] 예측 방법 => B-mode 토큰 + repaint prior -> MLP -> sigmoid
class ReliabilityHead(nn.Module):
    """
    Predict patch reliability in [0,1].
    Input:
        b_tokens: [B,N,D]
        repaint_prior: [B,N]
    Output:
        reliability: [B,N]
    """
    def __init__(self, dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, b_tokens: torch.Tensor, repaint_prior: torch.Tensor) -> torch.Tensor:
        x = torch.cat([b_tokens, repaint_prior.unsqueeze(-1)], dim=-1)
        reliability = torch.sigmoid(self.net(x)).squeeze(-1)
        return reliability

# Reliability-gated cross-attention
class ReliabilityGatedCrossAttention(nn.Module):
    """
    Query: Doppler
    Key/Value: synthetic B-mode
    단, B-mode token마다 reliability가 있음
    reliability가 낮은 B-mode token은 attention에서 덜 기여하게 만든다
    """
    def __init__(self, dim: int, num_heads: int = 4, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # 초초초기화
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.out_proj = nn.Linear(dim, dim)
        self.out_drop = nn.Dropout(proj_drop)
    
    # mha 준비
    def _reshape_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        x = x.view(B, N, self.num_heads, self.head_dim)
        x = x.transpose(1, 2)  # [B,H,N,Dh]
        return x

    def forward(
        self,
        query_feat: torch.Tensor,      # [B,Nq,D]
        key_value_feat: torch.Tensor,  # [B,Nk,D]
        reliability: torch.Tensor,     # [B,Nk]
    ) -> torch.Tensor:
        # QKV 생성 + 헤드 분할
        q = self._reshape_heads(self.q_proj(query_feat))
        k = self._reshape_heads(self.k_proj(key_value_feat))
        v = self._reshape_heads(self.v_proj(key_value_feat))

        # reliability를 attention에 적용: k,v에 reliability 곱하기: reliability가 낮은 토큰은 k,v가 작아져서 attention에서 덜 기여하게 됨
        r = reliability.unsqueeze(1).unsqueeze(-1)  # [B,1,Nk,1]
        k = k * r
        v = v * r
        # attn 계산: QK^T / sqrt(dh) + log(reliability) as bias
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B,H,Nq,Nk]

        # add log reliability as attention bias
        log_r = torch.log(reliability.clamp(min=1e-6)).unsqueeze(1).unsqueeze(2)  # [B,1,1,Nk]
        attn = attn + log_r

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v  # [B,H,Nq,Dh]
        out = out.transpose(1, 2).contiguous().view(query_feat.size(0), query_feat.size(1), self.dim)
        out = self.out_proj(out)
        out = self.out_drop(out)
        return out

# Fusion model
class SuppressRebalanceFusionModel(nn.Module):
    """
    Main study model:
    - Doppler encoder
    - synthetic B-mode encoder
    - reliability prior from repaint mask
    - learned reliability refinement
    - reliability-gated cross-attention
    - structure bypass
    """
    def __init__(
        self,
        num_classes: int = 3,
        dim: int = 128,
        patch_size: int = 16,
        num_heads: int = 4,
        alpha_prior: float = 0.5,
        structure_bypass_weight: float = 0.2,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.alpha_prior = alpha_prior
        self.structure_bypass_weight = structure_bypass_weight

        self.doppler_encoder = SimplePatchEncoder(in_ch=3, dim=dim, patch_size=patch_size)
        self.bmode_encoder = SimplePatchEncoder(in_ch=1, dim=dim, patch_size=patch_size)

        self.rel_head = ReliabilityHead(dim=dim)
        self.cross_attn = ReliabilityGatedCrossAttention(dim=dim, num_heads=num_heads)

        self.norm_d = nn.LayerNorm(dim)
        self.norm_b = nn.LayerNorm(dim)

        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

        self.cls_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, num_classes),
        )

    def forward(
        self,
        x_doppler: torch.Tensor,     # [B,3,H,W]
        x_bmode: torch.Tensor,       # [B,1,H,W]
        repaint_mask: torch.Tensor,  # [B,1,H,W], 1 = repaint/copypaste region
    ):
        d_tokens = self.doppler_encoder(x_doppler)  # [B,Nd,D]
        b_tokens = self.bmode_encoder(x_bmode)      # [B,Nb,D]

        patch_mask = patch_mask_from_pixel_mask(repaint_mask, self.patch_size)  # [B,Nb]

        # hard masking is NOT used
        repaint_prior = 1.0 - self.alpha_prior * patch_mask
        repaint_prior = repaint_prior.clamp(0.05, 1.0)

        d_norm = self.norm_d(d_tokens)
        b_norm = self.norm_b(b_tokens)

        reliability = self.rel_head(b_norm, repaint_prior)  # [B,Nb]

        cross = self.cross_attn(d_norm, b_norm, reliability)
        fused = d_tokens + cross
        fused = fused + self.ffn(fused)

        # structure bypass: preserve global B-mode structure instead of zeroing it out
        b_global = b_tokens.mean(dim=1)
        fused_global = fused.mean(dim=1) + self.structure_bypass_weight * b_global

        logits = self.cls_head(fused_global)

        return {
            "logits": logits,
            "reliability": reliability,
            "repaint_prior": repaint_prior,
            "patch_mask": patch_mask,
            "doppler_tokens": d_tokens,
            "bmode_tokens": b_tokens,
            "fused_tokens": fused,
        }



# Loss
@dataclass
class LossWeights:
    cls: float = 1.0
    prior: float = 0.5
    smooth: float = 0.05
    repaint_sep: float = 0.2


def suppress_rebalance_loss(
    outputs: dict,
    targets: torch.Tensor,
    weights: LossWeights = LossWeights(),
) -> dict:
    logits = outputs["logits"]
    reliability = outputs["reliability"]
    repaint_prior = outputs["repaint_prior"]
    patch_mask = outputs["patch_mask"]

    loss_cls = F.cross_entropy(logits, targets)

    # reliability should roughly follow prior in early-stage study setting
    loss_prior = F.l1_loss(reliability, repaint_prior)

    # smooth reliability along token sequence
    loss_smooth = total_variation_1d(reliability)

    # repaint region should have lower reliability than clean region
    repaint_mean = (reliability * patch_mask).sum() / (patch_mask.sum() + 1e-6)
    clean_mask = 1.0 - patch_mask
    clean_mean = (reliability * clean_mask).sum() / (clean_mask.sum() + 1e-6)
    loss_repaint_sep = F.relu(repaint_mean - clean_mean + 0.1)

    # L=λcls​Lcls​+λprior​Lprior​+λsmooth​Lsmooth​+λsep​Lsep​
    total = (
        weights.cls * loss_cls
        + weights.prior * loss_prior
        + weights.smooth * loss_smooth
        + weights.repaint_sep * loss_repaint_sep
    )

    return {
        "loss": total,
        "loss_cls": loss_cls.detach(),
        "loss_prior": loss_prior.detach(),
        "loss_smooth": loss_smooth.detach(),
        "loss_repaint_sep": loss_repaint_sep.detach(),
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SuppressRebalanceFusionModel(
        num_classes=3,
        dim=128,
        patch_size=16,
        num_heads=4,
        alpha_prior=0.5,
    ).to(device)

    B, H, W = 4, 224, 224
    x_d = torch.randn(B, 3, H, W, device=device)
    x_b = torch.randn(B, 1, H, W, device=device)
    repaint_mask = (torch.rand(B, 1, H, W, device=device) > 0.8).float()
    y = torch.randint(0, 3, (B,), device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    model.train()
    outputs = model(x_d, x_b, repaint_mask)
    loss_dict = suppress_rebalance_loss(outputs, y)

    optimizer.zero_grad()
    loss_dict["loss"].backward()
    optimizer.step()

    print("=== Suppress & Rebalance Demo ===")
    for k, v in loss_dict.items():
        print(f"{k}: {float(v):.6f}")
    print("logits shape:", outputs["logits"].shape)
    print("reliability shape:", outputs["reliability"].shape)


if __name__ == "__main__":
    main()
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SuppressRebalanceFusionModel(
        num_classes=3,
        dim=128,
        patch_size=16,
        num_heads=4,
        alpha_prior=0.5,
    ).to(device)

    B, H, W = 4, 224, 224
    x_d = torch.randn(B, 3, H, W, device=device)
    x_b = torch.randn(B, 1, H, W, device=device)
    repaint_mask = (torch.rand(B, 1, H, W, device=device) > 0.8).float()
    y = torch.randint(0, 3, (B,), device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    model.train()
    outputs = model(x_d, x_b, repaint_mask)
    loss_dict = suppress_rebalance_loss(outputs, y)

    optimizer.zero_grad()
    loss_dict["loss"].backward()
    optimizer.step()

    print("=== Suppress & Rebalance Demo ===")
    for k, v in loss_dict.items():
        print(f"{k}: {float(v):.6f}")
    print("logits shape:", outputs["logits"].shape)
    print("reliability shape:", outputs["reliability"].shape)


if __name__ == "__main__":
    main()