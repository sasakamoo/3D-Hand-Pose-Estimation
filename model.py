"""
model.py — Latent 2.5D Heatmap Regression Network
===================================================
Exact implementation of the single-view network described in:

  Iqbal et al. "Hand Pose Estimation via Latent 2.5D Heatmap Regression"
  ECCV 2018.  arXiv:1804.09534

as used by:

  Zimmermann et al. "FreiHAND: A Dataset for Markerless Capture of Hand
  Pose and Shape from Single RGB Images."  ICCV 2019.  arXiv:1909.04349

Architecture (from paper §5.2 and appendix):
  - Encoder-Decoder with skip connections (U-Net style)
  - Fixed 256 channels in every convolutional layer
  - Input: 128×128 RGB image
  - Output: 2K feature maps at 128×128 (same resolution as input)
      K channels → latent 2D heatmaps  H*_2D
      K channels → latent depth maps   H*_z
  - Learnable per-keypoint temperature β_k (sharpness of softmax)
  - Latent soft-argmax extracts (x, y, z_rel) from the feature maps

2.5D representation (paper §3.1, Eq.2):
  - 2D pixel coords (x_k, y_k) in [0, W] × [0, H]
  - Root-relative depth  z^r_k = Z_k - Z_root   (in scale-normalised units)
  - Scale normalisation: s = ||P_n - P_parent(n)||  for reference bone
    (index MCP joint → wrist, paper §3.2), constant C = 1

Soft-argmax (paper §4.2, Eqs. 13-15):
  H_2D_k = softmax(β_k · H*_2D_k)           (per-keypoint temperature)
  p_k     = Σ_p  H_2D_k(p) · p              (2D position)
  z^r_k   = Σ_p  H_2D_k(p) ⊙ H*_z_k(p)    (depth via Hadamard product)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Number of hand keypoints ───────────────────────────────────────────────
K = 21


# ============================================================================
# Building blocks
# ============================================================================

def conv_bn_relu(in_ch, out_ch, kernel=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class EncoderBlock(nn.Module):
    """Two conv-BN-ReLU + 2× downsampling via stride-2 conv."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = conv_bn_relu(in_ch,  out_ch)
        self.conv2 = conv_bn_relu(out_ch, out_ch)
        self.down  = nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1, bias=False)
        self.bn_d  = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        skip = x                          # stored for skip connection
        x    = F.relu(self.bn_d(self.down(x)), inplace=True)
        return x, skip


class DecoderBlock(nn.Module):
    """Bilinear 2× upsample + skip add + two conv-BN-ReLU."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up    = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv1 = conv_bn_relu(in_ch,  out_ch)
        self.conv2 = conv_bn_relu(out_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        x = x + skip                      # element-wise addition (paper Fig.1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


# ============================================================================
# Full Encoder-Decoder backbone
# ============================================================================

class EncoderDecoder(nn.Module):
    """
    U-Net style backbone.

    Paper §5.2: "Encoder-Decoder network architecture with skip connections
    and fixed number of channels (256) in each convolutional layer."

    Input : (B, 3, 128, 128)
    Output: (B, 256, 128, 128)   — feature map at input resolution
    """

    CH = 256   # paper: fixed 256 channels everywhere

    def __init__(self):
        super().__init__()
        CH = self.CH

        # Stem — brings 3 → CH without downsampling
        self.stem = nn.Sequential(
            conv_bn_relu(3, CH, kernel=7, stride=1, padding=3),
        )

        # Encoder: 128→64→32→16→8
        self.enc1 = EncoderBlock(CH, CH)   # 128→64
        self.enc2 = EncoderBlock(CH, CH)   # 64→32
        self.enc3 = EncoderBlock(CH, CH)   # 32→16
        self.enc4 = EncoderBlock(CH, CH)   # 16→8

        # Bottleneck
        self.bottleneck = nn.Sequential(
            conv_bn_relu(CH, CH),
            conv_bn_relu(CH, CH),
        )

        # Decoder: 8→16→32→64→128
        self.dec4 = DecoderBlock(CH, CH)
        self.dec3 = DecoderBlock(CH, CH)
        self.dec2 = DecoderBlock(CH, CH)
        self.dec1 = DecoderBlock(CH, CH)

    def forward(self, x):
        x = self.stem(x)                        # (B, 256, 128, 128)
        x, s1 = self.enc1(x)                   # (B, 256, 64, 64)
        x, s2 = self.enc2(x)                   # (B, 256, 32, 32)
        x, s3 = self.enc3(x)                   # (B, 256, 16, 16)
        x, s4 = self.enc4(x)                   # (B, 256, 8, 8)
        x = self.bottleneck(x)
        x = self.dec4(x, s4)                   # (B, 256, 16, 16)
        x = self.dec3(x, s3)                   # (B, 256, 32, 32)
        x = self.dec2(x, s2)                   # (B, 256, 64, 64)
        x = self.dec1(x, s1)                   # (B, 256, 128, 128)
        return x


# ============================================================================
# Latent 2.5D Heatmap Head
# ============================================================================

class Latent25DHeatmapHead(nn.Module):
    """
    Converts 256-channel feature map → 2K channels → (x,y,z_rel) per keypoint.

    Paper §4.2, Eqs. 13-15:
      H_2D_k  = softmax(β_k · H*_2D_k)        learnable temperature β_k
      p_k     = Σ_p H_2D_k(p) · p             soft-argmax for 2D location
      z^r_k   = Σ_p H_2D_k(p) ⊙ H*_z_k(p)   depth via Hadamard product

    Key implementation notes:
      - β is initialised to 10 (not 1). With β=1 over 128×128=16384 pixels
        the softmax is nearly uniform (each value ≈ 1/16384), gradients
        through soft-argmax effectively vanish. β=10 sharpens the initial
        distribution enough for gradients to flow.
      - We return BOTH H_2d (probability map, post-softmax) AND
        H_star_2d (logits, pre-softmax). The heatmap loss must be applied
        to the logits space (or use a proper divergence), not to the
        post-softmax probabilities whose values are ~1e-5.

    Outputs:
      pose_2d   : (B, K, 2)     normalised coords in [0, 1]
      depth_rel : (B, K)        root-relative scale-normalised depth
      hm_logits : (B, K, H, W)  pre-softmax logits H*_2D  ← use for loss
      hm_probs  : (B, K, H, W)  post-softmax probs  H_2D  ← use for vis
    """

    def __init__(self, in_ch=256, num_kpts=K):
        super().__init__()
        self.num_kpts = num_kpts
        self.head = nn.Conv2d(in_ch, 2 * num_kpts, kernel_size=1)

        # β initialised to 10: gives enough sharpness for gradients to flow
        # through soft-argmax without being so large that it acts like argmax
        self.beta = nn.Parameter(torch.full((num_kpts,), 10.0))

    def forward(self, features):
        B, _, H, W = features.shape
        out = self.head(features)                            # (B, 2K, H, W)

        H_star_2d = out[:, :self.num_kpts]                  # (B, K, H, W) logits
        H_star_z  = out[:, self.num_kpts:]                  # (B, K, H, W)

        # Eq.13 — temperature-scaled softmax
        beta   = self.beta.view(1, self.num_kpts, 1, 1)
        scaled = (beta * H_star_2d).view(B, self.num_kpts, -1)   # (B,K,H*W)
        H_2d   = torch.softmax(scaled, dim=2).view(B, self.num_kpts, H, W)

        # Eq.14 — soft-argmax in PIXEL coordinates [0, W-1] / [0, H-1]
        # IMPORTANT: grid must match the space GT lives in.
        # GT keypoints are in [0, IMG_SIZE] pixels.
        # linspace(0, W-1, W) gives pixel indices 0,1,2,...,W-1.
        ys = torch.arange(H, dtype=torch.float32, device=features.device)  # 0..H-1
        xs = torch.arange(W, dtype=torch.float32, device=features.device)  # 0..W-1
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')             # (H, W)

        # Weighted sum → pixel coordinates
        x_coord = (H_2d * grid_x).view(B, self.num_kpts, -1).sum(2)  # (B, K)
        y_coord = (H_2d * grid_y).view(B, self.num_kpts, -1).sum(2)  # (B, K)
        pose_2d = torch.stack([x_coord, y_coord], dim=2)              # (B, K, 2) in pixels

        # Eq.15 — depth
        depth_rel = (H_2d * H_star_z).view(B, self.num_kpts, -1).sum(2)

        return pose_2d, depth_rel, H_star_2d, H_2d


# ============================================================================
# Complete single-view network (as used in FreiHAND paper)
# ============================================================================

class SingleViewModel(nn.Module):
    """
    Full single-view hand pose network from Iqbal et al. ECCV 2018,
    as referenced and used in Zimmermann et al. (FreiHAND) ICCV 2019.

    Paper notes (FreiHAND §3.2):
      "We train the single-view network using the same hyper-parameter
      choices as Iqbal et al. However, we use only a single stage and
      reduce the number of channels in the network layers."

    This implementation uses a single stage (no iterative refinement)
    and 256 channels as specified in §5.2 of the Iqbal paper.

    Input:  (B, 3, 128, 128)  RGB, float32, pixel values in [0, 1]
    Outputs:
        pose_2d  : (B, K, 2)  2D keypoint positions, normalised [0,1]
        depth_rel: (B, K)     root-relative scale-normalised depth z^r_k
        heatmaps : (B, K, H, W)  probability maps (for visualisation / loss)
    """

    def __init__(self, num_kpts: int = K):
        super().__init__()
        self.backbone = EncoderDecoder()
        self.head     = Latent25DHeatmapHead(in_ch=256, num_kpts=num_kpts)

    def forward(self, x):
        """
        Returns:
            pose_2d   : (B, K, 2)
            depth_rel : (B, K)
            hm_logits : (B, K, H, W)  pre-softmax  ← supervise with heatmap loss
            hm_probs  : (B, K, H, W)  post-softmax ← use for visualisation
        """
        features = self.backbone(x)
        pose_2d, depth_rel, hm_logits, hm_probs = self.head(features)
        return pose_2d, depth_rel, hm_logits, hm_probs


# ============================================================================
# 3D pose reconstruction from 2.5D (paper §3.2, Eqs. 4-6)
# ============================================================================

def reconstruct_3d_from_25d(pose_2d_px: torch.Tensor,
                             depth_rel:  torch.Tensor,
                             K_mat:      torch.Tensor,
                             img_size:   int = 128,
                             ref_bone_idx: tuple = (0, 5),
                             C: float = 1.0) -> torch.Tensor:
    """
    Recover scale-normalised 3D pose Pˆ from 2.5D representation.

    Paper §3.2, Eq. 5-6:
      Given (x_n, y_n, z^r_n) and (x_m, y_m, z^r_m) for the reference bone,
      solve the quadratic equation for Z_root, then back-project all joints.

    Args:
        pose_2d_px : (B, K, 2)  pixel coordinates (x, y) in [0, img_size]
        depth_rel  : (B, K)     root-relative normalised depth z^r_k
        K_mat      : (B, 3, 3)  camera intrinsic matrix
        img_size   : side length of input image in pixels
        ref_bone_idx: (n, m) indices of reference bone joints
                     paper uses (0=wrist, 5=index MCP)
        C          : target bone length after normalisation (paper: C=1)

    Returns:
        pose_3d    : (B, K, 3)  scale-normalised 3D pose  Pˆ
    """
    B, num_kpts, _ = pose_2d_px.shape
    device = pose_2d_px.device

    n_idx, m_idx = ref_bone_idx

    # Pixel coords already in [0, img_size] when we pass pose_2d_px * img_size
    xn = pose_2d_px[:, n_idx, 0]   # (B,)
    yn = pose_2d_px[:, n_idx, 1]
    xm = pose_2d_px[:, m_idx, 0]
    ym = pose_2d_px[:, m_idx, 1]
    zrn = depth_rel[:, n_idx]
    zrm = depth_rel[:, m_idx]

    # Quadratic coefficients (paper Eq. 6)
    a = (xn - xm)**2 + (yn - ym)**2
    b = (zrn * (xn**2 + yn**2 - xn*xm - yn*ym) +
         zrm * (xm**2 + ym**2 - xn*xm - yn*ym))
    c = ((xn*zrn - xm*zrm)**2 + (yn*zrn - ym*zrm)**2 +
         (zrn - zrm)**2 - C**2)

    # Solve for Z_root — take the positive (in-front-of-camera) root
    discriminant = torch.clamp(b**2 - 4*a*c, min=0.0)
    a_safe = a.clamp(min=1e-8)
    z_root = (-b + torch.sqrt(discriminant)) / (2 * a_safe)   # (B,)

    # Full depth of each keypoint
    Z_k = z_root.unsqueeze(1) + depth_rel        # (B, K)

    # Back-project using intrinsics K_mat:  X = (x - cx)*Z / fx,  Y = (y-cy)*Z/fy
    # K_mat: [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    fx = K_mat[:, 0, 0].unsqueeze(1)  # (B, 1)
    fy = K_mat[:, 1, 1].unsqueeze(1)
    cx = K_mat[:, 0, 2].unsqueeze(1)
    cy = K_mat[:, 1, 2].unsqueeze(1)

    x_px = pose_2d_px[:, :, 0]   # (B, K)
    y_px = pose_2d_px[:, :, 1]

    X_k = (x_px - cx) * Z_k / fx
    Y_k = (y_px - cy) * Z_k / fy

    pose_3d = torch.stack([X_k, Y_k, Z_k], dim=2)   # (B, K, 3)
    return pose_3d


if __name__ == '__main__':
    model = SingleViewModel(num_kpts=21)
    total = sum(p.numel() for p in model.parameters())
    print(f"SingleViewModel   params: {total:,}  ({total*4/1e6:.1f} MB)")

    x = torch.randn(2, 3, 128, 128)
    p2d, depth, hm = model(x)
    print(f"pose_2d   : {p2d.shape}")      # (2, 21, 2)
    print(f"depth_rel : {depth.shape}")    # (2, 21)
    print(f"heatmaps  : {hm.shape}")       # (2, 21, 128, 128)