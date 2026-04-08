"""
model_sdf_mano.py — HOISDF-Inspired Hand Pose Estimation with MANO Output
==========================================================================
Extends model_sdf.py by adding a MANO parameter prediction head alongside
the existing 2.5D joint regression head.

Based on:
  Qi et al. "HOISDF: Constraining 3D Hand-Object Pose Estimation with
  Global Signed Distance Fields."  CVPR 2024.  arXiv:2402.17062

Architecture additions over model_sdf.py:
  • MANOParamHead — a second cross-attention head that reads the same
    density-enhanced point features as JointQueryAttention and outputs
    MANO pose (θ ∈ R^48) and shape (β ∈ R^10) parameters.

    - 16 learnable pose queries (one per MANO joint) cross-attend to
      the enhanced point features → each query → Linear(256, 3) → 48-d θ
    - 1 learnable shape query cross-attends to the same features → Linear(256, 10) → β
    - Both heads share one cross-attention block (17 queries total)

  This hybrid design keeps the 2.5D head for precise pixel-level supervision
  while adding the MANO head for structured articulation-aware output.
  The MANO layer itself is NOT required at training time — the pose/shape
  parameters are supervised directly against the FreiHAND ground-truth
  MANO annotations (training_mano.json).

Output format:
    pose_2d    : (B, K, 2)   2D pixel coords  (from 2.5D head, same as model_sdf.py)
    depth_rel  : (B, K)      root-relative depth (from 2.5D head)
    sdf_vals   : (B, N_PTS)  per-point SDF predictions
    pts        : (B, N_PTS, 3) query points matching sdf_vals
    mano_pose  : (B, 48)     predicted MANO pose parameters
    mano_shape : (B, 10)     predicted MANO shape parameters

evaluate.py compatibility:
    The first four outputs are identical to model_sdf.py, so evaluate.py
    works without modification when using model_type='sdf'.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# ── Constants ─────────────────────────────────────────────────────────────────

K              = 21    # hand keypoints
N_PTS          = 512   # 3D query points  (8³)
FEAT_DIM       = 256
IMG_SIZE       = 128
MANO_POSE_DIM  = 48    # 16 joints × 3 axis-angle
MANO_SHAPE_DIM = 10    # shape PCA coefficients
MANO_JOINTS    = 16    # number of MANO kinematic joints (excl. tips)


# ── Fourier Positional Encoding  (unchanged from model_sdf.py) ───────────────

class FourierPosEnc(nn.Module):
    def __init__(self, num_freqs: int = 6):
        super().__init__()
        self.num_freqs = num_freqs
        freqs = 2.0 ** torch.arange(num_freqs, dtype=torch.float32)
        self.register_buffer('freqs', freqs)

    @property
    def out_dim(self) -> int:
        return 3 + 2 * self.num_freqs * 3

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        parts = [p]
        for f in self.freqs:
            parts.append(torch.sin(f * p))
            parts.append(torch.cos(f * p))
        return torch.cat(parts, dim=-1)


# ── ResNet-50 + U-Net Decoder  (unchanged from model_sdf.py) ─────────────────

def _dec_block(up_ch: int, skip_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(up_ch + skip_ch, FEAT_DIM, 3, padding=1, bias=False),
        nn.BatchNorm2d(FEAT_DIM),
        nn.ReLU(inplace=True),
        nn.Conv2d(FEAT_DIM, FEAT_DIM, 3, padding=1, bias=False),
        nn.BatchNorm2d(FEAT_DIM),
        nn.ReLU(inplace=True),
    )


class ResNetUNet(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        try:
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            resnet  = models.resnet50(weights=weights)
        except AttributeError:
            resnet  = models.resnet50(pretrained=pretrained)

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
        s0 = self.stem(x)
        s1 = self.pool(s0)
        e1 = self.layer1(s1)
        e2 = self.layer2(e1)
        e3 = self.layer3(e2)
        e4 = self.layer4(e3)

        d = self.dec4(torch.cat([self.up(e4), e3], dim=1))
        d = self.dec3(torch.cat([self.up(d),  e2], dim=1))
        d = self.dec2(torch.cat([self.up(d),  e1], dim=1))
        d = self.dec1(torch.cat([self.up(d),  s0], dim=1))
        d = self.dec0(self.up(d))
        return d


# ── SDF Field Decoder MLP  (unchanged from model_sdf.py) ─────────────────────

class SDFDecoder(nn.Module):
    def __init__(self, feat_dim: int = FEAT_DIM, pos_dim: int = None):
        super().__init__()
        if pos_dim is None:
            pos_dim = FourierPosEnc().out_dim
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim + pos_dim, FEAT_DIM),
            nn.ReLU(inplace=True),
            nn.Linear(FEAT_DIM, FEAT_DIM),
            nn.ReLU(inplace=True),
            nn.Linear(FEAT_DIM, 1),
        )

    def forward(self, img_feats: torch.Tensor,
                pos_enc: torch.Tensor) -> torch.Tensor:
        return self.mlp(torch.cat([img_feats, pos_enc], dim=-1))


# ── 2.5D Joint Regression Head  (unchanged from model_sdf.py) ────────────────

class JointQueryAttention(nn.Module):
    """
    Cross-attention based 2.5D joint regression (from model_sdf.py).
    Produces (B, K, 3) raw predictions → decoded to pose_2d and depth_rel.
    """

    def __init__(self, feat_dim: int = FEAT_DIM, num_joints: int = K,
                 n_heads: int = 8, n_self_layers: int = 6):
        super().__init__()
        self.joint_queries = nn.Parameter(torch.empty(num_joints, feat_dim))
        nn.init.trunc_normal_(self.joint_queries, std=0.02)

        self.point_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=feat_dim, nhead=n_heads,
                dim_feedforward=feat_dim * 2,
                dropout=0.0, batch_first=True, norm_first=True,
            ),
            num_layers=n_self_layers,
            enable_nested_tensor=False,
        )
        self.cross_attn    = nn.MultiheadAttention(feat_dim, n_heads, batch_first=True, dropout=0.0)
        self.cross_norm_q  = nn.LayerNorm(feat_dim)
        self.cross_norm_kv = nn.LayerNorm(feat_dim)
        self.cross_ff      = nn.Sequential(
            nn.Linear(feat_dim, feat_dim * 2), nn.GELU(), nn.Linear(feat_dim * 2, feat_dim))
        self.cross_ff_norm = nn.LayerNorm(feat_dim)
        self.out_head      = nn.Linear(feat_dim, 3)

    def forward(self, point_feats: torch.Tensor):
        """point_feats: (B, N, D) → (B, K, 3) raw, and encoded_pts (B, N, D)"""
        B = point_feats.shape[0]
        pts = self.point_encoder(point_feats)
        q = self.joint_queries.unsqueeze(0).expand(B, -1, -1)
        attn_out, _ = self.cross_attn(self.cross_norm_q(q), self.cross_norm_kv(pts), pts)
        q = q + attn_out
        q = q + self.cross_ff(self.cross_ff_norm(q))
        return self.out_head(q), pts   # (B, K, 3),  (B, N, D)


# ── MANO Parameter Head  (new) ────────────────────────────────────────────────

class MANOParamHead(nn.Module):
    """
    Predict MANO pose (θ ∈ R^48) and shape (β ∈ R^10) from point features.

    Paper §3.3 (adapted):
      "16 learnable queries for MANO parameters ... cross-attention with
       hand query point features."

    We use 16 pose queries (one per MANO joint) and 1 shape query combined
    into a single cross-attention block (17 queries total) for efficiency.

    Inputs:
        encoded_pts : (B, N, feat_dim)  — self-attention-encoded point
                      features from JointQueryAttention.point_encoder.
                      Sharing the encoder avoids running it twice.

    Outputs:
        mano_pose  : (B, 48)   axis-angle parameters for 16 MANO joints
        mano_shape : (B, 10)   MANO shape PCA coefficients
    """

    def __init__(self, feat_dim: int = FEAT_DIM, n_heads: int = 8):
        super().__init__()
        n_queries = MANO_JOINTS + 1   # 16 pose + 1 shape = 17

        self.queries = nn.Parameter(torch.empty(n_queries, feat_dim))
        nn.init.trunc_normal_(self.queries, std=0.02)

        self.cross_attn    = nn.MultiheadAttention(feat_dim, n_heads, batch_first=True, dropout=0.0)
        self.cross_norm_q  = nn.LayerNorm(feat_dim)
        self.cross_norm_kv = nn.LayerNorm(feat_dim)
        self.cross_ff      = nn.Sequential(
            nn.Linear(feat_dim, feat_dim * 2), nn.GELU(), nn.Linear(feat_dim * 2, feat_dim))
        self.cross_ff_norm = nn.LayerNorm(feat_dim)

        # 16 pose queries → 3 axis-angle each → (B, 48)
        self.pose_head  = nn.Linear(feat_dim, 3)
        # 1 shape query  → 10 PCA coefficients → (B, 10)
        self.shape_head = nn.Linear(feat_dim, MANO_SHAPE_DIM)

    def forward(self, encoded_pts: torch.Tensor):
        """
        encoded_pts : (B, N, feat_dim)
        Returns     : mano_pose (B, 48),  mano_shape (B, 10)
        """
        B  = encoded_pts.shape[0]
        q  = self.queries.unsqueeze(0).expand(B, -1, -1)   # (B, 17, D)
        kv = self.cross_norm_kv(encoded_pts)
        attn_out, _ = self.cross_attn(self.cross_norm_q(q), kv, kv)
        q = q + attn_out
        q = q + self.cross_ff(self.cross_ff_norm(q))        # (B, 17, D)

        pose_feats  = q[:, :MANO_JOINTS]                   # (B, 16, D)
        shape_feat  = q[:, MANO_JOINTS]                    # (B, D)

        mano_pose   = self.pose_head(pose_feats).reshape(B, MANO_POSE_DIM)  # (B, 48)
        mano_shape  = self.shape_head(shape_feat)                            # (B, 10)

        return mano_pose, mano_shape


# ── Full Model ────────────────────────────────────────────────────────────────

class SDFHandPoseNetMANO(nn.Module):
    """
    HOISDF-inspired hand pose model with MANO parameter output.

    Extends SDFHandPoseNet (model_sdf.py) with a MANO parameter head.
    The self-attention point encoder from JointQueryAttention is shared
    with MANOParamHead so it only runs once per forward pass.

    Input:  (B, 3, 128, 128)
    Outputs:
        pose_2d    : (B, K, 2)    2D pixel coords in [0, IMG_SIZE]
        depth_rel  : (B, K)       root-relative scale-normalised depth
        sdf_vals   : (B, N_PTS)   per-point SDF predictions
        pts        : (B, N_PTS, 3) query points
        mano_pose  : (B, 48)      MANO pose parameters
        mano_shape : (B, 10)      MANO shape parameters
    """

    def __init__(self, num_kpts: int = K, n_pts: int = N_PTS,
                 pretrained_backbone: bool = True):
        super().__init__()
        self.num_kpts = num_kpts
        self.n_pts    = n_pts

        self.backbone  = ResNetUNet(pretrained=pretrained_backbone)
        self.pos_enc   = FourierPosEnc(num_freqs=6)
        self.sdf_dec   = SDFDecoder(feat_dim=FEAT_DIM, pos_dim=self.pos_enc.out_dim)
        self.feat_proj = nn.Sequential(
            nn.Linear(self.pos_enc.out_dim + 2 * FEAT_DIM, FEAT_DIM),
            nn.ReLU(inplace=True),
        )
        self.joint_head = JointQueryAttention(feat_dim=FEAT_DIM, num_joints=num_kpts)
        self.mano_head  = MANOParamHead(feat_dim=FEAT_DIM)

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _sample_feats(feat_map: torch.Tensor, pts_uv: torch.Tensor) -> torch.Tensor:
        grid = pts_uv.unsqueeze(1)
        out  = F.grid_sample(feat_map, grid, mode='bilinear',
                             align_corners=True, padding_mode='border')
        return out.squeeze(2).permute(0, 2, 1)

    def _sample_points(self, B: int, device: torch.device) -> torch.Tensor:
        if self.training:
            return torch.rand(B, self.n_pts, 3, device=device) * 2 - 1
        side = math.ceil(self.n_pts ** (1 / 3))
        lin  = torch.linspace(-0.9, 0.9, side, device=device)
        gu, gv, gd = torch.meshgrid(lin, lin, lin, indexing='ij')
        pts = torch.stack([gu, gv, gd], dim=-1).reshape(1, -1, 3)
        return pts[:, :self.n_pts].expand(B, -1, -1).contiguous()

    # ── Forward pass ─────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor):
        """
        x : (B, 3, IMG_SIZE, IMG_SIZE)

        Returns
        -------
        pose_2d    : (B, K, 2)
        depth_rel  : (B, K)
        sdf_vals   : (B, N_PTS)
        pts        : (B, N_PTS, 3)
        mano_pose  : (B, 48)
        mano_shape : (B, 10)
        """
        B = x.shape[0]

        # 1. Pixel-aligned feature map
        feat_map = self.backbone(x)                              # (B, 256, H, W)

        # 2. Sample 3D query points
        pts = self._sample_points(B, x.device)                  # (B, N, 3)

        # 3. Positional encoding + pixel-aligned features
        pos_enc   = self.pos_enc(pts)                            # (B, N, 39)
        img_feats = self._sample_feats(feat_map, pts[:, :, :2]) # (B, N, 256)

        # 4. SDF prediction
        sdf_vals = self.sdf_dec(img_feats, pos_enc)             # (B, N, 1)

        # 5. Density modulation
        density = torch.sigmoid(-sdf_vals.abs())                # (B, N, 1)

        # 6. Enhanced point features
        enhanced = self.feat_proj(
            torch.cat([pos_enc, img_feats, density * img_feats], dim=-1))  # (B, N, 256)

        # 7a. 2.5D joint regression head
        #     Also returns self-attention-encoded points for reuse by MANO head.
        joint_raw, encoded_pts = self.joint_head(enhanced)      # (B, K, 3), (B, N, 256)

        pose_2d   = torch.sigmoid(joint_raw[:, :, :2]) * IMG_SIZE  # (B, K, 2)
        depth_rel = joint_raw[:, :, 2]                              # (B, K)

        # 7b. MANO parameter head  (reuses encoded_pts — no double self-attn)
        mano_pose, mano_shape = self.mano_head(encoded_pts)     # (B, 48), (B, 10)

        return pose_2d, depth_rel, sdf_vals.squeeze(-1), pts, mano_pose, mano_shape


# ── Quick sanity check ────────────────────────────────────────────────────────

if __name__ == '__main__':
    net   = SDFHandPoseNetMANO(pretrained_backbone=False)
    total = sum(p.numel() for p in net.parameters())
    print(f'SDFHandPoseNetMANO   params: {total:,}  ({total * 4 / 1e6:.1f} MB)')

    x = torch.randn(2, 3, 128, 128)
    with torch.no_grad():
        p2d, dz, sdf, pts, mpose, mshape = net(x)
    print(f'pose_2d    : {p2d.shape}')      # (2, 21, 2)
    print(f'depth_rel  : {dz.shape}')       # (2, 21)
    print(f'sdf_vals   : {sdf.shape}')      # (2, 512)
    print(f'pts        : {pts.shape}')      # (2, 512, 3)
    print(f'mano_pose  : {mpose.shape}')    # (2, 48)
    print(f'mano_shape : {mshape.shape}')   # (2, 10)
