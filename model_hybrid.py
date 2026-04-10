"""
model_hybrid.py — Hybrid Hand Pose Estimation with Auxiliary Heatmap Head
=========================================================================
Extends SDFHandPoseNet (model_sdf.py) with three improvements:

  1. Auxiliary heatmap head — lightweight conv head predicting 21 Gaussian
     heatmaps, trained with an auxiliary MSE loss.  Forces the backbone to
     learn spatially-precise features that benefit the 2D localisation task.

  2. Joint-centred point sampling — during training, 50% of 3D query points
     are sampled uniformly (background coverage) and 50% are sampled as
     Gaussian noise around the GT joint locations (sigma=0.1 in normalised
     space).  At inference a dense regular grid is used as in model_sdf.py.

  3. use_sdf=False by default — density gate is disabled (density=1.0) so
     the unsupervised SDF decoder does not corrupt image features before the
     joint attention head.

Architecture overview:
  ResNet-50 + U-Net decoder  → 256-ch feature map  (B, 256, 128, 128)
  HeatmapHead                → 21 heatmaps         (B, 21, 128, 128)
  SDF decoder + density gate → point features      (B, N, 256)
  6-layer Transformer + cross-attention → joint queries → 2.5D output

Forward signature:
  forward(x, gt_joints_norm=None)
    x              : (B, 3, 128, 128)   input RGB
    gt_joints_norm : (B, 21, 3) or None  GT joint locations in [-1,1]^3
                     (u=pixel_x/64-1, v=pixel_y/64-1, d=depth_rel_normalised)
                     Used for joint-centred point sampling during training only.

Returns:
  pose_2d   : (B, K, 2)    2D pixel coords in [0, IMG_SIZE]
  depth_rel : (B, K)       root-relative scale-normalised depth
  hm_logits : (B, K, H, W) auxiliary heatmap logits (for heatmap loss)
  sdf_vals  : (B, N)       per-point SDF predictions
  pts       : (B, N, 3)    the query points used
"""

import math
from typing import Optional, cast
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.checkpoint import checkpoint


# ── Constants ─────────────────────────────────────────────────────────────────

K           = 21      # hand keypoints
N_PTS       = 2000    # 3D query points during training (50% uniform + 50% joint-centred)
N_PTS_GRID  = 8192    # dense candidate grid at inference
N_PTS_KEEP  = 600     # nearest-surface points kept after SDF filtering
FEAT_DIM    = 256     # channel width throughout
IMG_SIZE    = 128     # input / output image resolution
SDF_CLAMP   = 0.15    # SDF clamping distance in metres


# ── Fourier Positional Encoding ───────────────────────────────────────────────

class FourierPosEnc(nn.Module):
    """
    Sinusoidal Fourier positional encoding for 3D point coordinates.
    For L frequencies the output dimension is 3 + 2·L·3.
    Default L=6 gives 39 dimensions.
    """

    def __init__(self, num_freqs: int = 6):
        super().__init__()
        self.num_freqs = num_freqs
        freqs = 2.0 ** torch.arange(num_freqs, dtype=torch.float32)
        self.freqs: torch.Tensor
        self.register_buffer('freqs', freqs)    # (L,)

    @property
    def out_dim(self) -> int:
        return 3 + 2 * self.num_freqs * 3      # 3 + 2·L·3

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        """p: (B, N, 3)  →  (B, N, out_dim)"""
        parts = [p]
        for f in self.freqs:
            parts.append(torch.sin(f * p))
            parts.append(torch.cos(f * p))
        return torch.cat(parts, dim=-1)         # (B, N, out_dim)


# ── ResNet-50 + U-Net Decoder ─────────────────────────────────────────────────

def _dec_block(up_ch: int, skip_ch: int) -> nn.Sequential:
    """U-Net decoder block: two conv-BN-ReLU layers → FEAT_DIM channels."""
    return nn.Sequential(
        nn.Conv2d(up_ch + skip_ch, FEAT_DIM, 3, padding=1, bias=False),
        nn.BatchNorm2d(FEAT_DIM),
        nn.ReLU(inplace=True),
        nn.Conv2d(FEAT_DIM, FEAT_DIM, 3, padding=1, bias=False),
        nn.BatchNorm2d(FEAT_DIM),
        nn.ReLU(inplace=True),
    )


class ResNetUNet(nn.Module):
    """
    ResNet-50 encoder with five-stage U-Net decoder.

    Spatial sizes for 128×128 input:
        stem (stride 2)  : (B,   64,  64, 64)
        maxpool (stride 2): (B,   64,  32, 32)
        layer1           : (B,  256,  32, 32)
        layer2 (stride 2): (B,  512,  16, 16)
        layer3 (stride 2): (B, 1024,   8,  8)
        layer4 (stride 2): (B, 2048,   4,  4)

    Output: (B, 256, 128, 128) full-resolution feature map.
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        try:
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            resnet  = models.resnet50(weights=weights)
        except AttributeError:
            resnet  = models.resnet50(pretrained=pretrained)  # type: ignore[call-arg]

        self.stem   = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.pool   = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.dec4 = _dec_block(2048, 1024)
        self.dec3 = _dec_block(FEAT_DIM,  512)
        self.dec2 = _dec_block(FEAT_DIM,  256)
        self.dec1 = _dec_block(FEAT_DIM,   64)
        self.dec0 = nn.Sequential(
            nn.Conv2d(FEAT_DIM, FEAT_DIM, 3, padding=1, bias=False),
            nn.BatchNorm2d(FEAT_DIM),
            nn.ReLU(inplace=True),
            nn.Conv2d(FEAT_DIM, FEAT_DIM, 3, padding=1, bias=False),
            nn.BatchNorm2d(FEAT_DIM),
            nn.ReLU(inplace=True),
        )
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s0 = cast(torch.Tensor, checkpoint(self.stem,   x,  use_reentrant=False))
        s1 = self.pool(s0)
        e1 = cast(torch.Tensor, checkpoint(self.layer1, s1, use_reentrant=False))
        e2 = cast(torch.Tensor, checkpoint(self.layer2, e1, use_reentrant=False))
        e3 = cast(torch.Tensor, checkpoint(self.layer3, e2, use_reentrant=False))
        e4 = cast(torch.Tensor, checkpoint(self.layer4, e3, use_reentrant=False))

        d = self.dec4(torch.cat([self.up(e4), e3], dim=1))
        d = self.dec3(torch.cat([self.up(d),  e2], dim=1))
        d = self.dec2(torch.cat([self.up(d),  e1], dim=1))
        d = self.dec1(torch.cat([self.up(d),  s0], dim=1))
        d = self.dec0(self.up(d))
        return d


# ── SDF Field Decoder MLP ─────────────────────────────────────────────────────

class SDFDecoder(nn.Module):
    """3-layer MLP: (img_feat ‖ pos_enc) → scalar SDF value per point."""

    def __init__(self, feat_dim: int = FEAT_DIM, pos_dim: Optional[int] = None):
        super().__init__()
        if pos_dim is None:
            pos_dim = FourierPosEnc().out_dim
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim + pos_dim, FEAT_DIM),
            nn.ReLU(inplace=True),
            nn.Linear(FEAT_DIM, FEAT_DIM),
            nn.ReLU(inplace=True),
            nn.Linear(FEAT_DIM, 1),
            nn.Tanh(),
        )

    def forward(self, img_feats: torch.Tensor,
                pos_enc: torch.Tensor) -> torch.Tensor:
        """
        img_feats : (B, N, feat_dim)
        pos_enc   : (B, N, pos_dim)
        Returns   : (B, N, 1)
        """
        return self.mlp(torch.cat([img_feats, pos_enc], dim=-1))


# ── Attention-Based Joint Regression Head ─────────────────────────────────────

class JointQueryAttention(nn.Module):
    """
    6-layer self-attention on point features, then K=21 joint queries
    cross-attend → linear head → (B, K, 3) raw 2.5D predictions.
    """

    def __init__(self, feat_dim: int = FEAT_DIM, num_joints: int = K,
                 n_heads: int = 8, n_self_layers: int = 6):
        super().__init__()
        self.joint_queries = nn.Parameter(torch.empty(num_joints, feat_dim))
        nn.init.trunc_normal_(self.joint_queries, std=0.02)

        self.point_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=feat_dim,
                nhead=n_heads,
                dim_feedforward=feat_dim * 2,
                dropout=0.1,
                batch_first=True,
                norm_first=True,
            ),
            num_layers=n_self_layers,
            enable_nested_tensor=False,
        )

        self.cross_attn    = nn.MultiheadAttention(
            feat_dim, n_heads, batch_first=True, dropout=0.1)
        self.cross_norm_q  = nn.LayerNorm(feat_dim)
        self.cross_norm_kv = nn.LayerNorm(feat_dim)
        self.cross_ff      = nn.Sequential(
            nn.Linear(feat_dim, feat_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(feat_dim * 2, feat_dim),
        )
        self.cross_ff_norm = nn.LayerNorm(feat_dim)
        self.out_head      = nn.Linear(feat_dim, 3)

    def forward(self, point_feats: torch.Tensor) -> torch.Tensor:
        """
        point_feats : (B, N, feat_dim)
        Returns     : (B, K, 3)   raw [x_raw, y_raw, z_rel]
        """
        B = point_feats.shape[0]
        pts = self.point_encoder(point_feats)
        q = self.joint_queries.unsqueeze(0).expand(B, -1, -1)
        attn_out, _ = self.cross_attn(
            self.cross_norm_q(q),
            self.cross_norm_kv(pts),
            pts,
        )
        q = q + attn_out
        q = q + self.cross_ff(self.cross_ff_norm(q))
        return self.out_head(q)


# ── Auxiliary Heatmap Head ────────────────────────────────────────────────────

class HeatmapHead(nn.Module):
    """
    Lightweight conv head predicting 21 heatmaps at 128×128 resolution.

    Architecture:
        conv(256→256, 3×3, BN, ReLU)
        conv(256→256, 3×3, BN, ReLU)
        conv(256→21,  1×1)  ← no activation; logits for MSE loss

    Input:  (B, 256, 128, 128) feature map from ResNetUNet
    Output: (B, 21, 128, 128)  heatmap logits
    """

    def __init__(self, in_ch: int = FEAT_DIM, num_joints: int = K):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_ch,   in_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch,   in_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, num_joints, 1),   # 1×1 projection to K channels
        )

    def forward(self, feat_map: torch.Tensor) -> torch.Tensor:
        """
        feat_map : (B, 256, H, W)
        Returns  : (B, K, H, W)  heatmap logits
        """
        return self.head(feat_map)


# ── Hybrid Model ──────────────────────────────────────────────────────────────

class HybridHandPoseNet(nn.Module):
    """
    Hybrid hand pose network combining:
      - SDF-inspired point cloud + attention regression (model_sdf.py)
      - Auxiliary heatmap head for spatially precise feature learning
      - Joint-centred point sampling during training

    Input:
        x              : (B, 3, 128, 128)
        gt_joints_norm : (B, 21, 3) or None
                         GT joints in [-1,1]^3 model space.
                         Coordinates:  u = pixel_x / 64 - 1
                                       v = pixel_y / 64 - 1
                                       d = depth_rel (normalised)
                         When provided during training, used for joint-centred
                         point sampling.  Ignored at inference (eval mode).

    Returns:
        pose_2d   : (B, K, 2)    2D pixel coords in [0, IMG_SIZE]
        depth_rel : (B, K)       root-relative scale-normalised depth
        hm_logits : (B, K, H, W) auxiliary heatmap logits
        sdf_vals  : (B, N)       per-point SDF predictions
        pts       : (B, N, 3)    query points used
    """

    def __init__(self, num_kpts: int = K, n_pts: int = N_PTS,
                 pretrained_backbone: bool = True,
                 use_sdf: bool = False):
        """
        use_sdf : enable SDF density gating.
                  False by default — density=1.0, no gating.
        """
        super().__init__()
        self.num_kpts = num_kpts
        self.n_pts    = n_pts
        self.use_sdf  = use_sdf
        self._pts_buf: Optional[torch.Tensor] = None

        self.backbone     = ResNetUNet(pretrained=pretrained_backbone)
        self.heatmap_head = HeatmapHead(in_ch=FEAT_DIM, num_joints=num_kpts)
        self.pos_enc      = FourierPosEnc(num_freqs=6)
        self.sdf_dec      = SDFDecoder(feat_dim=FEAT_DIM,
                                       pos_dim=self.pos_enc.out_dim)

        # Project concat(pos_enc, density·img_feat) → FEAT_DIM
        self.feat_proj = nn.Sequential(
            nn.Linear(self.pos_enc.out_dim + FEAT_DIM, FEAT_DIM),
            nn.ReLU(inplace=True),
        )

        self.density_beta = nn.Parameter(torch.tensor(0.1))
        self.joint_head   = JointQueryAttention(
            feat_dim=FEAT_DIM, num_joints=num_kpts)

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _sample_feats(feat_map: torch.Tensor,
                      pts_uv: torch.Tensor) -> torch.Tensor:
        """
        Pixel-aligned feature extraction via bilinear grid_sample.

        feat_map : (B, C, H, W)
        pts_uv   : (B, N, 2)  normalised image coords in [-1, 1]
        Returns  : (B, N, C)
        """
        grid = pts_uv.unsqueeze(1)                   # (B, 1, N, 2)
        out  = F.grid_sample(feat_map, grid,
                             mode='bilinear', align_corners=True,
                             padding_mode='border')   # (B, C, 1, N)
        return out.squeeze(2).permute(0, 2, 1)        # (B, N, C)

    def _sample_points(self, B: int, device: torch.device,
                       gt_joints_norm: Optional[torch.Tensor] = None
                       ) -> torch.Tensor:
        """
        Sample 3D query points in normalised space [-1, 1]^3.

        Training with gt_joints_norm provided:
            50% uniform random  — broad background coverage.
            50% Gaussian around GT joint locations  (sigma=0.1)
            — concentrates query density near the hand surface for
              better feature attention from joint regions.

        Training without gt_joints_norm:
            N_PTS uniform random points (fallback, same as model_sdf.py).

        Inference (eval mode):
            Dense regular grid of ~N_PTS_GRID points in [-0.9, 0.9]^3.
        """
        if self.training:
            n_uniform = self.n_pts // 2
            n_joint   = self.n_pts - n_uniform

            # Uniform half
            pts_uniform = torch.empty(B, n_uniform, 3,
                                      device=device).uniform_(-1.0, 1.0)

            if gt_joints_norm is not None:
                # gt_joints_norm: (B, K, 3) in [-1, 1]^3
                # For each sample in batch, pick n_joint points from K joints
                # with replacement, then add Gaussian noise (sigma=0.1).
                sigma = 0.1

                # Randomly select one of the 21 joints for each point
                # joint_idx : (B, n_joint)
                joint_idx = torch.randint(0, self.num_kpts, (B, n_joint),
                                          device=device)

                # Gather the corresponding joint coordinates
                # Expand joint_idx for gather: (B, n_joint, 3)
                idx_exp      = joint_idx.unsqueeze(-1).expand(-1, -1, 3)
                joint_centres = gt_joints_norm.gather(1, idx_exp)  # (B, n_joint, 3)

                # Add Gaussian noise and clamp to [-1, 1]
                noise     = torch.randn_like(joint_centres) * sigma
                pts_joint = torch.clamp(joint_centres + noise, -1.0, 1.0)
            else:
                # Fallback: all uniform
                pts_joint = torch.empty(B, n_joint, 3,
                                        device=device).uniform_(-1.0, 1.0)

            return torch.cat([pts_uniform, pts_joint], dim=1)  # (B, N_PTS, 3)

        # ── Dense candidate grid for inference ────────────────────────────
        side = math.ceil(N_PTS_GRID ** (1.0 / 3))
        lin  = torch.linspace(-0.9, 0.9, side, device=device)
        gu, gv, gd = torch.meshgrid(lin, lin, lin, indexing='ij')  # type: ignore[call-overload]
        grid = torch.stack([gu, gv, gd], dim=-1).reshape(1, -1, 3)
        grid = grid.expand(B, -1, -1).contiguous()
        return grid

    # ── Forward pass ─────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor,
                gt_joints_norm: Optional[torch.Tensor] = None):
        """
        x              : (B, 3, 128, 128)
        gt_joints_norm : (B, 21, 3) or None — used only in training mode.

        Returns
        -------
        pose_2d   : (B, K, 2)
        depth_rel : (B, K)
        hm_logits : (B, K, 128, 128)
        sdf_vals  : (B, N)
        pts       : (B, N, 3)
        """
        B = x.shape[0]

        # 1. Feature map from backbone
        feat_map  = self.backbone(x)                              # (B, 256, H, W)

        # 2. Auxiliary heatmap head (always active)
        hm_logits = self.heatmap_head(feat_map)                   # (B, K, H, W)

        # 3. Query points
        pts = self._sample_points(B, x.device,
                                  gt_joints_norm if self.training else None)

        # 4. Per-point features
        pos_enc   = self.pos_enc(pts)                             # (B, N, 39)
        img_feats = self._sample_feats(feat_map, pts[:, :, :2])  # (B, N, 256)

        # 5. SDF decoder
        sdf_tanh = self.sdf_dec(img_feats, pos_enc)              # (B, N, 1) ∈ (-1,1)
        sdf_vals = sdf_tanh * SDF_CLAMP                          # metres

        # 6. SDF-guided nearest-surface filtering at inference (only when use_sdf=True)
        if self.use_sdf and not self.training:
            _, keep_idx = sdf_vals.abs().squeeze(-1).sort(dim=1)
            keep_idx    = keep_idx[:, :N_PTS_KEEP]
            idx_exp     = keep_idx.unsqueeze(-1)
            pts         = pts.gather(1, idx_exp.expand(-1, -1, 3))
            pos_enc     = pos_enc.gather(1, idx_exp.expand(-1, -1, pos_enc.shape[-1]))
            img_feats   = img_feats.gather(1, idx_exp.expand(-1, -1, FEAT_DIM))
            sdf_tanh    = sdf_tanh.gather(1, idx_exp.expand(-1, -1, 1))
            sdf_vals    = sdf_vals.gather(1, idx_exp.expand(-1, -1, 1))

        # 7. Density modulation
        if self.use_sdf:
            beta    = torch.clamp(self.density_beta, min=2e-3)
            density = torch.exp(-sdf_vals.abs() / beta)          # (B, N, 1)
        else:
            density = torch.ones_like(sdf_vals)                  # (B, N, 1) = 1.0

        # 8. Feature enhancement: concat(pos_enc, density·img_feat) → FEAT_DIM
        enhanced_raw = torch.cat([pos_enc, density * img_feats], dim=-1)  # (B, N, 295)
        enhanced     = self.feat_proj(enhanced_raw)                        # (B, N, 256)

        # 9. Joint regression via attention
        joint_raw = self.joint_head(enhanced)                     # (B, K, 3)

        # 10. Decode to 2.5D output
        pose_2d   = torch.sigmoid(joint_raw[:, :, :2]) * IMG_SIZE # (B, K, 2)
        depth_rel = joint_raw[:, :, 2]                             # (B, K)

        return pose_2d, depth_rel, hm_logits, sdf_vals.squeeze(-1), pts


# ── Quick sanity check ────────────────────────────────────────────────────────

if __name__ == '__main__':
    net = HybridHandPoseNet(pretrained_backbone=False)
    total = sum(p.numel() for p in net.parameters())
    print(f'HybridHandPoseNet  params: {total:,}  ({total * 4 / 1e6:.1f} MB)')

    x = torch.randn(2, 3, 128, 128)
    gt_jn = torch.zeros(2, 21, 3).uniform_(-0.5, 0.5)

    # Training forward (with joint-centred sampling)
    net.train()
    with torch.no_grad():
        p2d, dz, hml, sdf, pts = net(x, gt_joints_norm=gt_jn)
    print(f'[train] pose_2d   : {p2d.shape}')    # (2, 21, 2)
    print(f'[train] depth_rel : {dz.shape}')     # (2, 21)
    print(f'[train] hm_logits : {hml.shape}')    # (2, 21, 128, 128)
    print(f'[train] sdf_vals  : {sdf.shape}')    # (2, 2000)
    print(f'[train] pts       : {pts.shape}')    # (2, 2000, 3)

    # Inference forward (dense grid, no gt_joints_norm)
    net.eval()
    with torch.no_grad():
        p2d, dz, hml, sdf, pts = net(x)
    print(f'[eval]  pose_2d   : {p2d.shape}')
    print(f'[eval]  sdf_vals  : {sdf.shape}')
    print(f'[eval]  pts       : {pts.shape}')
