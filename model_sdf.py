"""
model_sdf.py — HOISDF-Inspired Hand Pose Estimation
=====================================================
Adapted from:

  Qi et al. "HOISDF: Constraining 3D Hand-Object Pose Estimation with
  Global Signed Distance Fields."  CVPR 2024.  arXiv:2402.17062

Architecture:
  1. ResNet-50 + U-Net decoder  → pixel-aligned feature map  (B, 256, H, W)
  2. N_PTS 3D points sampled in normalised image+depth space  [-1, 1]^3
  3. Pixel-aligned feature extraction per point (F.grid_sample)
  4. Fourier positional encoding of the 3D coordinates
  5. SDF field decoder (3-layer MLP) → signed distance per point
  6. Density modulation:  σ = sigmoid(−|sdf|)
       Near-surface points (|sdf| ≈ 0) get σ ≈ 1; far-surface points σ ≈ 0.
       Modulated feature = σ · img_feat  (paper §3.2 "density-weighted")
  7. Enhanced point features: concat(pos_enc, img_feat, σ·img_feat) → 256-d
  8. 6-layer Multi-Head Self-Attention on point features  (paper §3.3)
  9. K=21 learnable joint queries cross-attend to point features
 10. Linear head  → (x, y, z_rel) per joint

Output format matches model.py 2.5D convention:
    pose_2d   : (B, K, 2)    pixel coordinates in [0, IMG_SIZE]
    depth_rel : (B, K)       root-relative scale-normalised depth  z^r_k
    sdf_vals  : (B, N_PTS)   per-point SDF predictions (for optional aux loss)

Note on train.py compatibility:
    model.py returns 4 values (pose_2d, depth_rel, hm_logits, hm_probs).
    This model returns 3 values (pose_2d, depth_rel, sdf_vals).
    The heatmap loss term in train.py should be replaced with an SDF loss
    (L1 against ground-truth signed distances) or simply omitted.

Coordinate convention:
    3D query points are in normalised space [-1, 1]^3:
      (u, v)  →  image grid_sample coordinates (u=-1 is left, u=+1 is right)
      d       →  normalised depth (root-relative, scale-normalised)
    The joint regression head produces raw (x, y, z); x and y are passed
    through sigmoid and scaled to [0, IMG_SIZE] pixel space, matching the
    soft-argmax output of model.py.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# ── Constants ─────────────────────────────────────────────────────────────────

K            = 21    # hand keypoints
N_PTS        = 2048  # 3D query points sampled per image during training
N_PTS_GRID   = 8192  # dense candidate grid at inference before SDF filtering
N_PTS_KEEP   = 600   # nearest-surface points kept after SDF filtering (paper: 600)
FEAT_DIM     = 256   # channel width throughout
IMG_SIZE     = 128   # input / output image resolution


# ── Fourier Positional Encoding ───────────────────────────────────────────────

class FourierPosEnc(nn.Module):
    """
    Sinusoidal Fourier positional encoding for 3D point coordinates.

    Paper §3.2: "Fourier positional encoding of the 3D query point."

    For L frequencies the output dimension is 3 + 2·L·3 (raw coordinates
    are prepended).  Default L=6 gives 3 + 36 = 39 dimensions.
    """

    def __init__(self, num_freqs: int = 6):
        super().__init__()
        self.num_freqs = num_freqs
        freqs = 2.0 ** torch.arange(num_freqs, dtype=torch.float32)
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
    """
    U-Net decoder block: two conv-BN-ReLU layers.

    Input channels = up_ch (from upsampled previous level) + skip_ch
    (from encoder skip connection).  Output is always FEAT_DIM channels.
    """
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

    Paper §3.1: "ResNet-50 with U-Net decoder architecture ... 256 feature
    channels ... hierarchical encoder-decoder with skip connections."

    Spatial sizes for 128×128 input:
        conv1 (stem, stride 2)  : (B,   64,  64, 64)
        maxpool (stride 2)      : (B,   64,  32, 32)
        layer1                  : (B,  256,  32, 32)
        layer2 (stride 2)       : (B,  512,  16, 16)
        layer3 (stride 2)       : (B, 1024,   8,  8)
        layer4 (stride 2)       : (B, 2048,   4,  4)

    Decoder reverses this with five bilinear 2× upsamplings back to 128×128.

    Input:  (B, 3,   128, 128)
    Output: (B, 256, 128, 128)  full-resolution pixel-aligned feature map
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        try:
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            resnet  = models.resnet50(weights=weights)
        except AttributeError:          # older torchvision (<0.13)
            resnet  = models.resnet50(pretrained=pretrained)

        # ── Encoder ───────────────────────────────────────────────────────
        self.stem   = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.pool   = resnet.maxpool
        self.layer1 = resnet.layer1     # 256-ch, same spatial as after pool
        self.layer2 = resnet.layer2     # 512-ch, /2
        self.layer3 = resnet.layer3     # 1024-ch, /2
        self.layer4 = resnet.layer4     # 2048-ch, /2

        # ── Decoder ───────────────────────────────────────────────────────
        # Skip connections by spatial size (for 128×128 input):
        #   dec4: up(e4) 4→8,   cat e3  (8×8,  1024ch)
        #   dec3: up    8→16,   cat e2  (16×16, 512ch)
        #   dec2: up   16→32,   cat e1  (32×32, 256ch)
        #   dec1: up   32→64,   cat s0  (64×64,  64ch)  ← s0 = after conv1, before maxpool
        #   dec0: up   64→128,  no skip  (plain conv to reach input resolution)
        self.dec4 = _dec_block(2048, 1024)      # 4×4   → 8×8
        self.dec3 = _dec_block(FEAT_DIM,  512)  # 8×8   → 16×16
        self.dec2 = _dec_block(FEAT_DIM,  256)  # 16×16 → 32×32
        self.dec1 = _dec_block(FEAT_DIM,   64)  # 32×32 → 64×64  (skip: s0)
        self.dec0 = nn.Sequential(               # 64×64 → 128×128 (no skip)
            nn.Conv2d(FEAT_DIM, FEAT_DIM, 3, padding=1, bias=False),
            nn.BatchNorm2d(FEAT_DIM),
            nn.ReLU(inplace=True),
            nn.Conv2d(FEAT_DIM, FEAT_DIM, 3, padding=1, bias=False),
            nn.BatchNorm2d(FEAT_DIM),
            nn.ReLU(inplace=True),
        )

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s0 = self.stem(x)           # (B,   64,  64, 64)  ← skip for dec1
        s1 = self.pool(s0)          # (B,   64,  32, 32)
        e1 = self.layer1(s1)        # (B,  256,  32, 32)
        e2 = self.layer2(e1)        # (B,  512,  16, 16)
        e3 = self.layer3(e2)        # (B, 1024,   8,  8)
        e4 = self.layer4(e3)        # (B, 2048,   4,  4)

        d = self.dec4(torch.cat([self.up(e4), e3], dim=1))  # (B, 256,   8,  8)
        d = self.dec3(torch.cat([self.up(d),  e2], dim=1))  # (B, 256,  16, 16)
        d = self.dec2(torch.cat([self.up(d),  e1], dim=1))  # (B, 256,  32, 32)
        d = self.dec1(torch.cat([self.up(d),  s0], dim=1))  # (B, 256,  64, 64)
        d = self.dec0(self.up(d))                            # (B, 256, 128,128)
        return d


# ── SDF Field Decoder MLP ─────────────────────────────────────────────────────

class SDFDecoder(nn.Module):
    """
    3-layer MLP that predicts a signed distance value per 3D query point.

    Paper §3.2: "field decoders ... separate 3-layer MLPs ... predict the
    signed distance value for that point."

    Input: concatenation of pixel-aligned image features (FEAT_DIM) and
           Fourier positional encoding (pos_dim).
    Output: scalar SDF value per point.
    """

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
        """
        img_feats : (B, N, feat_dim)
        pos_enc   : (B, N, pos_dim)
        Returns   : (B, N, 1)  signed distance predictions
        """
        return self.mlp(torch.cat([img_feats, pos_enc], dim=-1))


# ── Attention-Based Joint Regression Head ─────────────────────────────────────

class JointQueryAttention(nn.Module):
    """
    Cross-attention based 3D joint position regression.

    Paper §3.3: "Six Multi-Head Self-Attention layers process the enhanced
    point features ... learnable queries ... cross-attention with hand query
    point features."

    Adapted for direct 2.5D regression without MANO (no shape parameters):
      • 6-layer pre-norm Transformer encoder (self-attention on point features)
      • K=21 learnable joint queries cross-attend to encoded point features
      • A single cross-attention block with residual + feed-forward
      • Linear output head: (B, K, 3) raw predictions for (x, y, z_rel)
    """

    def __init__(self, feat_dim: int = FEAT_DIM, num_joints: int = K,
                 n_heads: int = 8, n_self_layers: int = 6):
        super().__init__()

        # Learnable joint query embeddings (one per keypoint)
        self.joint_queries = nn.Parameter(torch.empty(num_joints, feat_dim))
        nn.init.trunc_normal_(self.joint_queries, std=0.02)

        # Self-attention stack on point features (paper: 6 MHSA layers)
        # enable_nested_tensor=False avoids a warning when norm_first=True.
        self.point_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=feat_dim,
                nhead=n_heads,
                dim_feedforward=feat_dim * 2,
                dropout=0.0,
                batch_first=True,
                norm_first=True,        # pre-norm for stability
            ),
            num_layers=n_self_layers,
            enable_nested_tensor=False,
        )

        # Cross-attention: joint queries attend to encoded point features
        self.cross_attn      = nn.MultiheadAttention(
            feat_dim, n_heads, batch_first=True, dropout=0.0)
        self.cross_norm_q    = nn.LayerNorm(feat_dim)
        self.cross_norm_kv   = nn.LayerNorm(feat_dim)
        self.cross_ff        = nn.Sequential(
            nn.Linear(feat_dim, feat_dim * 2),
            nn.GELU(),
            nn.Linear(feat_dim * 2, feat_dim),
        )
        self.cross_ff_norm   = nn.LayerNorm(feat_dim)

        # Output head: feat_dim → 3  (raw x, y, z_rel)
        self.out_head = nn.Linear(feat_dim, 3)

    def forward(self, point_feats: torch.Tensor) -> torch.Tensor:
        """
        point_feats : (B, N, feat_dim)
        Returns     : (B, K, 3)   raw predictions  [x_raw, y_raw, z_rel]
        """
        B = point_feats.shape[0]

        # Self-attention on point features
        pts = self.point_encoder(point_feats)       # (B, N, feat_dim)

        # Cross-attention: K joint queries attend to N point features
        q = self.joint_queries.unsqueeze(0).expand(B, -1, -1)  # (B, K, feat_dim)
        attn_out, _ = self.cross_attn(
            self.cross_norm_q(q),
            self.cross_norm_kv(pts),
            pts,
        )
        q = q + attn_out                            # residual
        q = q + self.cross_ff(self.cross_ff_norm(q))

        return self.out_head(q)                     # (B, K, 3)


# ── Full Model ────────────────────────────────────────────────────────────────

class SDFHandPoseNet(nn.Module):
    """
    HOISDF-inspired single-view hand joint pose estimation.

    Based on Qi et al. "HOISDF: Constraining 3D Hand-Object Pose Estimation
    with Global Signed Distance Fields."  CVPR 2024.  arXiv:2402.17062

    Simplified for hand-only 2.5D joint prediction (no MANO parameters,
    no object branch).  Output format matches SingleViewModel in model.py.

    Input:  (B, 3, 128, 128)  RGB, float32, [0, 1]
    Outputs:
        pose_2d   : (B, K, 2)    2D pixel coords in [0, IMG_SIZE]
        depth_rel : (B, K)       root-relative scale-normalised depth  z^r_k
        sdf_vals  : (B, N_PTS)   per-point SDF predictions

    Training note:
        sdf_vals can be supervised with ground-truth signed distances computed
        from the 3D pose (e.g. L1 distance of each point to the nearest joint).
        If not used, simply ignore this output and train with the xy/depth loss
        terms from train.py (removing the heatmap loss term).
    """

    def __init__(self, num_kpts: int = K, n_pts: int = N_PTS,
                 pretrained_backbone: bool = True):
        super().__init__()
        self.num_kpts = num_kpts
        self.n_pts    = n_pts

        self.backbone = ResNetUNet(pretrained=pretrained_backbone)
        self.pos_enc  = FourierPosEnc(num_freqs=6)
        self.sdf_dec  = SDFDecoder(feat_dim=FEAT_DIM,
                                   pos_dim=self.pos_enc.out_dim)

        # Project concat(pos_enc, img_feat, σ·img_feat) → FEAT_DIM
        # Input dim = pos_enc.out_dim + 2 × FEAT_DIM  (39 + 512 = 551)
        self.feat_proj = nn.Sequential(
            nn.Linear(self.pos_enc.out_dim + 2 * FEAT_DIM, FEAT_DIM),
            nn.ReLU(inplace=True),
        )

        self.joint_head = JointQueryAttention(
            feat_dim=FEAT_DIM, num_joints=num_kpts)

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _sample_feats(feat_map: torch.Tensor,
                      pts_uv: torch.Tensor) -> torch.Tensor:
        """
        Pixel-aligned feature extraction via bilinear grid_sample.

        feat_map : (B, C, H, W)
        pts_uv   : (B, N, 2)  normalised image coords in [-1, 1]
                              (u=col direction, v=row direction, grid_sample convention)
        Returns  : (B, N, C)
        """
        grid = pts_uv.unsqueeze(1)                  # (B, 1, N, 2)
        out  = F.grid_sample(feat_map, grid,
                             mode='bilinear', align_corners=True,
                             padding_mode='border')  # (B, C, 1, N)
        return out.squeeze(2).permute(0, 2, 1)       # (B, N, C)

    def _sample_points(self, B: int,
                       device: torch.device) -> torch.Tensor:
        """
        Sample 3D query points in normalised space [-1, 1]^3.

        Training:  N_PTS=2048 uniform random points — broad coverage gives
                   varied SDF supervision and reduces the chance the attention
                   head sees only empty-space points.

        Inference: SDF-guided nearest-surface filtering (paper §3.2).
                   1. Build a dense N_PTS_GRID=8192 candidate grid.
                   2. Run a lightweight forward pass through the SDF decoder
                      to get per-point signed distances.
                   3. Sort by |SDF| ascending and keep N_PTS_KEEP=600 points
                      (those closest to the predicted hand surface).
                   This mirrors the paper's 64³ voxel grid + top-600 selection
                   and gives the attention head much more informative points.
        """
        if self.training:
            return torch.rand(B, self.n_pts, 3, device=device) * 2 - 1

        # ── Dense candidate grid ──────────────────────────────────────────
        side = math.ceil(N_PTS_GRID ** (1 / 3))          # ≈ 21 → 21³ = 9261
        lin  = torch.linspace(-0.9, 0.9, side, device=device)
        gu, gv, gd = torch.meshgrid(lin, lin, lin, indexing='ij')
        grid = torch.stack([gu, gv, gd], dim=-1).reshape(1, -1, 3)  # (1, G, 3)
        grid = grid.expand(B, -1, -1).contiguous()                   # (B, G, 3)
        return grid    # full grid returned; SDF filtering happens in forward()

    # ── Forward pass ─────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor):
        """
        x : (B, 3, IMG_SIZE, IMG_SIZE)

        Returns
        -------
        pose_2d   : (B, K, 2)    2D pixel coordinates in [0, IMG_SIZE]
        depth_rel : (B, K)       root-relative scale-normalised depth
        sdf_vals  : (B, N_PTS)   per-point signed distance predictions
        pts       : (B, N_PTS, 3) the query points that produced sdf_vals,
                    returned so the caller can compute SDF supervision without
                    re-sampling (sdf_vals[i] corresponds to pts[i]).
        """
        B = x.shape[0]

        # 1. Extract full-resolution pixel-aligned feature map
        feat_map = self.backbone(x)                              # (B, 256, H, W)

        # 2. Sample 3D query points
        #    Training : N_PTS=2048 random points
        #    Inference: dense grid (N_PTS_GRID points)
        pts = self._sample_points(B, x.device)                  # (B, N, 3)

        # 3. Positional encoding + pixel-aligned features
        pos_enc   = self.pos_enc(pts)                            # (B, N, 39)
        img_feats = self._sample_feats(feat_map, pts[:, :, :2]) # (B, N, 256)

        # 4. SDF field decoder: predict signed distance per point
        sdf_vals = self.sdf_dec(img_feats, pos_enc)             # (B, N, 1)

        # 5. SDF-guided nearest-surface filtering at inference  (Fix 2)
        #    Sort all N_PTS_GRID candidates by |SDF| and keep the N_PTS_KEEP
        #    nearest-surface points — equivalent to the paper's top-600 selection.
        #    At training we skip this to keep gradients flowing through all points.
        if not self.training:
            _, keep_idx = sdf_vals.abs().squeeze(-1).sort(dim=1)  # (B, N) ascending
            keep_idx  = keep_idx[:, :N_PTS_KEEP]                  # (B, N_PTS_KEEP)
            idx_exp   = keep_idx.unsqueeze(-1)
            pts       = pts.gather(1, idx_exp.expand(-1, -1, 3))
            pos_enc   = pos_enc.gather(1, idx_exp.expand(-1, -1, pos_enc.shape[-1]))
            img_feats = img_feats.gather(1, idx_exp.expand(-1, -1, FEAT_DIM))
            sdf_vals  = sdf_vals.gather(1, idx_exp.expand(-1, -1, 1))

        # 6. Density modulation  (paper §3.2)
        #    σ = sigmoid(−|sdf|):  σ ≈ 1 near surface, σ ≈ 0 far from surface
        density = torch.sigmoid(-sdf_vals.abs())                # (B, N, 1)

        # 7. Feature enhancement: concat(pos_enc, img_feat, σ·img_feat) → 256-d
        enhanced_raw = torch.cat(
            [pos_enc, img_feats, density * img_feats], dim=-1)  # (B, N, 551)
        enhanced = self.feat_proj(enhanced_raw)                 # (B, N, 256)

        # 8. Attention-based joint regression
        joint_raw = self.joint_head(enhanced)                   # (B, K, 3)

        # 9. Decode to 2.5D output format
        pose_2d   = torch.sigmoid(joint_raw[:, :, :2]) * IMG_SIZE  # (B, K, 2)
        depth_rel = joint_raw[:, :, 2]                              # (B, K)

        return pose_2d, depth_rel, sdf_vals.squeeze(-1), pts


# ── Quick sanity check ────────────────────────────────────────────────────────

if __name__ == '__main__':
    net = SDFHandPoseNet(pretrained_backbone=False)
    total = sum(p.numel() for p in net.parameters())
    print(f'SDFHandPoseNet   params: {total:,}  ({total * 4 / 1e6:.1f} MB)')

    x = torch.randn(2, 3, 128, 128)
    with torch.no_grad():
        p2d, dz, sdf, pts = net(x)
    print(f'pose_2d   : {p2d.shape}')   # (2, 21, 2)
    print(f'depth_rel : {dz.shape}')    # (2, 21)
    print(f'sdf_vals  : {sdf.shape}')   # (2, 512)
    print(f'pts       : {pts.shape}')   # (2, 512, 3)
