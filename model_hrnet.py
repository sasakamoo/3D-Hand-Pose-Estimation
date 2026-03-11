"""
model_hrnet.py — HRNet Backbone + Latent 2.5D Heatmap Head
============================================================
Drop-in replacement for model.py.

Key difference from model.py
------------------------------
model.py uses a U-Net encoder-decoder:
  - Encodes to low-resolution (8×8) then decodes back up
  - High-resolution detail is lost in the bottleneck and recovered
    imperfectly through skip connections

This file uses HRNet (High-Resolution Net, Wang et al. CVPR 2019):
  - Maintains MULTIPLE parallel branches at DIFFERENT resolutions
    simultaneously throughout the entire network
  - High-resolution branch NEVER downsampled to a bottleneck
  - Branches exchange information via repeated multi-scale fusion
  - Final feature map is full-resolution (128×128) with rich
    multi-scale context — ideal for precise keypoint localisation

HRNet architecture used here (HRNet-W32, hand-pose variant):
  Stage 1: Single branch, 64ch @ 128×128
           (standard ResNet stem, no downsampling of main branch)
  Stage 2: 2 branches: 32ch@128² + 64ch@64²
           1 fusion module
  Stage 3: 3 branches: 32ch@128² + 64ch@64² + 128ch@32²
           4 fusion modules
  Stage 4: 4 branches: 32ch@128² + 64ch@64² + 128ch@32² + 256ch@16²
           3 fusion modules
  Output:  Concatenate all branches (upsampled to 128²) → 480ch
           1×1 conv → 256ch → Latent25DHeatmapHead

All other components (Latent25DHeatmapHead, reconstruct_3d_from_25d,
loss functions) are IDENTICAL to model.py so train.py / overfit_test.py
work without any changes — just swap the import:

    # model.py (original U-Net)
    from model import SingleViewModel

    # model_hrnet.py (HRNet backbone)
    from model_hrnet import SingleViewModel

References:
  Wang et al. "Deep High-Resolution Representation Learning for Visual
  Recognition." CVPR 2019 / TPAMI 2020.  arXiv:1908.07919
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

K = 21   # number of hand keypoints


# ============================================================================
# Shared building blocks
# ============================================================================

def conv_bn_relu(in_ch, out_ch, kernel=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel, stride=stride,
                  padding=padding, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class BasicBlock(nn.Module):
    """
    Standard ResNet BasicBlock: 3×3 → BN → ReLU → 3×3 → BN + residual.
    Used as the repeated unit within each HRNet branch.
    """
    expansion = 1

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride,
                               padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)

        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        if self.downsample:
            identity = self.downsample(x)
        return F.relu(out + identity, inplace=True)


class Bottleneck(nn.Module):
    """
    ResNet Bottleneck: 1×1 → 3×3 → 1×1 with 4× channel expansion.
    Used only in Stage 1 (the initial stem stage).
    """
    expansion = 4

    def __init__(self, in_ch, width, stride=1):
        super().__init__()
        out_ch = width * self.expansion
        self.conv1 = nn.Conv2d(in_ch, width, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, 3, stride=stride,
                               padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, out_ch, 1, bias=False)
        self.bn3   = nn.BatchNorm2d(out_ch)

        self.downsample = None
        if in_ch != out_ch or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))
        if self.downsample:
            identity = self.downsample(x)
        return F.relu(out + identity, inplace=True)


def make_branch(in_ch, out_ch, num_blocks):
    """Stack of BasicBlocks for one HRNet branch."""
    blocks = [BasicBlock(in_ch, out_ch)]
    for _ in range(num_blocks - 1):
        blocks.append(BasicBlock(out_ch, out_ch))
    return nn.Sequential(*blocks)


# ============================================================================
# Multi-Scale Fusion Module
# ============================================================================

class FusionModule(nn.Module):
    """
    HRNet multi-resolution fusion.

    Each branch receives information from ALL other branches.
    - Same resolution:   identity (no conv)
    - Higher → lower:   stride-2 conv for each scale step down
    - Lower  → higher:  bilinear upsample + 1×1 conv to match channels

    branch_channels: list of channel counts per branch, e.g. [32, 64, 128]
    The i-th branch has branch_channels[i] channels.
    After fusion, each branch keeps the same number of channels.
    """

    def __init__(self, branch_channels):
        super().__init__()
        self.num_branches = len(branch_channels)
        self.fuse_layers  = nn.ModuleList()

        for i in range(self.num_branches):        # target branch i
            branch_fuse = nn.ModuleList()
            for j in range(self.num_branches):    # source branch j
                if i == j:
                    branch_fuse.append(nn.Identity())
                elif j < i:
                    # Source j is higher-res → downsample to match i
                    # Need (i - j) strided convolutions
                    layers = []
                    for k in range(i - j):
                        in_c  = branch_channels[j]
                        out_c = branch_channels[i] if k == (i-j-1) else branch_channels[j]
                        layers += [
                            nn.Conv2d(in_c, out_c, 3, stride=2,
                                      padding=1, bias=False),
                            nn.BatchNorm2d(out_c),
                        ]
                        if k < (i - j - 1):
                            layers.append(nn.ReLU(inplace=True))
                    branch_fuse.append(nn.Sequential(*layers))
                else:
                    # Source j is lower-res → upsample to match i
                    branch_fuse.append(nn.Sequential(
                        nn.Conv2d(branch_channels[j], branch_channels[i],
                                  1, bias=False),
                        nn.BatchNorm2d(branch_channels[i]),
                    ))
            self.fuse_layers.append(branch_fuse)

    def forward(self, branches):
        """
        branches: list of tensors, one per branch
        Returns: list of fused tensors (same length)
        """
        fused = []
        for i in range(self.num_branches):
            y = None
            for j in range(self.num_branches):
                x_j = branches[j]
                t   = self.fuse_layers[i][j](x_j)
                if i < j:
                    # j is lower-res, need to upsample to branch i resolution
                    t = F.interpolate(t, size=branches[i].shape[2:],
                                      mode='bilinear', align_corners=False)
                y = t if y is None else y + t
            fused.append(F.relu(y, inplace=True))
        return fused


# ============================================================================
# HRNet Backbone
# ============================================================================

class HRNetBackbone(nn.Module):
    """
    HRNet-W40 adapted for 128×128 hand pose input.

    Changes from original W32:
      Branch 0 widened 32→48ch — the high-res branch that directly
      feeds localisation was too narrow (32ch) to capture fine spatial
      detail. 48ch gives it more representational capacity.
      All other branches unchanged (64/128/256).

    Channel widths:
        Branch 0 (full res,  128²): 48 ch  ← widened from 32
        Branch 1 (half res,   64²): 64 ch
        Branch 2 (quarter,    32²): 128 ch
        Branch 3 (eighth,     16²): 256 ch

    Output: concatenate all branches → 496ch
            refine with 3×3 conv block → 256ch

    Input:  (B, 3, 128, 128)
    Output: (B, 256, 128, 128)
    """

    # Widened W32: branch 0 = 48 instead of 32
    BRANCH_CH = [48, 64, 128, 256]

    def __init__(self, use_checkpoint: bool = False):
        super().__init__()
        # Gradient checkpointing: recompute activations on backward instead of
        # storing them. Saves ~60% activation memory at cost of ~30% slower
        # training. Enable when batch_size > 2 causes OOM on your GPU.
        self.use_checkpoint = use_checkpoint
        C = self.BRANCH_CH

        # ── Stem ─────────────────────────────────────────────────────────────
        self.stem = nn.Sequential(
            conv_bn_relu(3,  64, kernel=3, stride=1, padding=1),
            conv_bn_relu(64, 64, kernel=3, stride=1, padding=1),
        )

        # ── Stage 1 ───────────────────────────────────────────────────────────
        self.stage1 = nn.Sequential(
            Bottleneck(64, 64),
            Bottleneck(256, 64),
            Bottleneck(256, 64),
            Bottleneck(256, 64),
        )

        # ── Transition 1→2 ────────────────────────────────────────────────────
        self.trans1 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(256, C[0], 3, padding=1, bias=False),
                nn.BatchNorm2d(C[0]), nn.ReLU(inplace=True)),
            nn.Sequential(
                nn.Conv2d(256, C[1], 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(C[1]), nn.ReLU(inplace=True)),
        ])

        # ── Stage 2 ───────────────────────────────────────────────────────────
        self.stage2_b0 = make_branch(C[0], C[0], num_blocks=4)
        self.stage2_b1 = make_branch(C[1], C[1], num_blocks=4)
        self.fusion2   = FusionModule([C[0], C[1]])

        # ── Transition 2→3 ────────────────────────────────────────────────────
        self.trans2 = nn.Sequential(
            nn.Conv2d(C[1], C[2], 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(C[2]), nn.ReLU(inplace=True))

        # ── Stage 3 ───────────────────────────────────────────────────────────
        self.stage3 = nn.ModuleList()
        for _ in range(4):
            self.stage3.append(nn.ModuleDict({
                'b0':     make_branch(C[0], C[0], num_blocks=4),
                'b1':     make_branch(C[1], C[1], num_blocks=4),
                'b2':     make_branch(C[2], C[2], num_blocks=4),
                'fusion': FusionModule([C[0], C[1], C[2]]),
            }))

        # ── Transition 3→4 ────────────────────────────────────────────────────
        self.trans3 = nn.Sequential(
            nn.Conv2d(C[2], C[3], 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(C[3]), nn.ReLU(inplace=True))

        # ── Stage 4 ───────────────────────────────────────────────────────────
        self.stage4 = nn.ModuleList()
        for _ in range(3):
            self.stage4.append(nn.ModuleDict({
                'b0':     make_branch(C[0], C[0], num_blocks=4),
                'b1':     make_branch(C[1], C[1], num_blocks=4),
                'b2':     make_branch(C[2], C[2], num_blocks=4),
                'b3':     make_branch(C[3], C[3], num_blocks=4),
                'fusion': FusionModule([C[0], C[1], C[2], C[3]]),
            }))

        # ── Output refinement ─────────────────────────────────────────────────
        # Replacing the original 1×1 projection with a two-step block:
        #   Step 1: 3×3 conv to spatially refine the concatenated multi-scale
        #           features. A 1×1 cannot mix spatial context across the
        #           misaligned upsampled branches — 3×3 does.
        #   Step 2: 1×1 conv to project down to 256ch for the heatmap head.
        # This is the main fix for L_xy stalling: the 1×1 projection was
        # producing features that the soft-argmax couldn't localise precisely.
        out_ch = sum(C)   # 48+64+128+256 = 496
        self.out_proj = nn.Sequential(
            # 3×3 spatial refinement
            nn.Conv2d(out_ch, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # 3×3 again to deepen the localisation representation
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

    def _ckpt(self, fn, *args):
        """Run fn(*args) with gradient checkpointing if enabled."""
        if self.use_checkpoint:
            return checkpoint(fn, *args, use_reentrant=False)
        return fn(*args)

    def forward(self, x):
        # Stem  (B, 3, 128, 128) → (B, 64, 128, 128)
        x = self._ckpt(self.stem, x)

        # Stage 1  → (B, 256, 128, 128)
        x = self._ckpt(self.stage1, x)

        # Transition 1→2
        b0 = self.trans1[0](x)              # (B, 48,  128, 128)
        b1 = self.trans1[1](x)              # (B, 64,   64,  64)
        branches = [b0, b1]

        # Stage 2 — checkpoint each branch separately
        branches[0] = self._ckpt(self.stage2_b0, branches[0])
        branches[1] = self._ckpt(self.stage2_b1, branches[1])
        branches = self.fusion2(branches)

        # Transition 2→3
        b2 = self.trans2(branches[1])       # (B, 128, 32, 32)
        branches.append(b2)

        # Stage 3 — checkpoint each branch block independently.
        # Stage 3 has 4 fusion blocks × 3 branches = 12 branch passes,
        # all at 128²/64²/32² resolution — biggest memory consumer.
        for blk in self.stage3:
            branches[0] = self._ckpt(blk['b0'], branches[0])
            branches[1] = self._ckpt(blk['b1'], branches[1])
            branches[2] = self._ckpt(blk['b2'], branches[2])
            branches = blk['fusion'](branches)

        # Transition 3→4
        b3 = self.trans3(branches[2])       # (B, 256, 16, 16)
        branches.append(b3)

        # Stage 4 — checkpoint each branch block.
        # 3 fusion blocks × 4 branches = 12 branch passes.
        for blk in self.stage4:
            branches[0] = self._ckpt(blk['b0'], branches[0])
            branches[1] = self._ckpt(blk['b1'], branches[1])
            branches[2] = self._ckpt(blk['b2'], branches[2])
            branches[3] = self._ckpt(blk['b3'], branches[3])
            branches = blk['fusion'](branches)

        # Aggregate: upsample all branches to 128², concatenate
        H, W = branches[0].shape[2], branches[0].shape[3]
        agg = [branches[0]]
        for b in branches[1:]:
            agg.append(F.interpolate(b, size=(H, W),
                                     mode='bilinear', align_corners=False))
        x = torch.cat(agg, dim=1)           # (B, 496, 128, 128)

        return self.out_proj(x)             # (B, 256, 128, 128)


# ============================================================================
# Latent 2.5D Heatmap Head  (IDENTICAL to model.py)
# ============================================================================

class Latent25DHeatmapHead(nn.Module):
    """
    Identical to model.py — no changes.
    Converts 256-channel feature map → (pose_2d, depth_rel, hm_logits, hm_probs)
    """

    def __init__(self, in_ch=256, num_kpts=K):
        super().__init__()
        self.num_kpts = num_kpts
        self.head = nn.Conv2d(in_ch, 2 * num_kpts, kernel_size=1)
        self.beta = nn.Parameter(torch.full((num_kpts,), 10.0))

    def forward(self, features):
        B, _, H, W = features.shape
        out = self.head(features)

        H_star_2d = out[:, :self.num_kpts]
        H_star_z  = out[:, self.num_kpts:]

        beta   = self.beta.view(1, self.num_kpts, 1, 1)
        scaled = (beta * H_star_2d).view(B, self.num_kpts, -1)
        H_2d   = torch.softmax(scaled, dim=2).view(B, self.num_kpts, H, W)

        ys = torch.arange(H, dtype=torch.float32, device=features.device)
        xs = torch.arange(W, dtype=torch.float32, device=features.device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')

        x_coord = (H_2d * grid_x).view(B, self.num_kpts, -1).sum(2)
        y_coord = (H_2d * grid_y).view(B, self.num_kpts, -1).sum(2)
        pose_2d = torch.stack([x_coord, y_coord], dim=2)

        depth_rel = (H_2d * H_star_z).view(B, self.num_kpts, -1).sum(2)

        return pose_2d, depth_rel, H_star_2d, H_2d


# ============================================================================
# Complete model  (same API as model.py SingleViewModel)
# ============================================================================

class SingleViewModel(nn.Module):
    """
    HRNet-W40 + Latent 2.5D Heatmap Head.

    Drop-in replacement for model.py SingleViewModel.
    Identical inputs/outputs — swap the import, nothing else changes.

    Args:
        num_kpts      : number of keypoints (default 21)
        use_checkpoint: enable gradient checkpointing to save GPU memory.
                        Saves ~60% activation memory, ~30% slower training.
                        Set True when batch_size > 2 causes OOM.

    Input:  (B, 3, 128, 128)
    Output: pose_2d   (B, K, 2)      pixel-space 2D keypoints
            depth_rel (B, K)         root-relative depth
            hm_logits (B, K, H, W)   pre-softmax  ← heatmap loss
            hm_probs  (B, K, H, W)   post-softmax ← visualisation
    """

    def __init__(self, num_kpts: int = K, use_checkpoint: bool = False):
        super().__init__()
        self.backbone = HRNetBackbone(use_checkpoint=use_checkpoint)
        self.head     = Latent25DHeatmapHead(in_ch=256, num_kpts=num_kpts)

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)


# ============================================================================
# 3D reconstruction  (IDENTICAL to model.py)
# ============================================================================

def reconstruct_3d_from_25d(pose_2d_px, depth_rel, K_mat,
                             img_size=128, ref_bone_idx=(0, 5), C=1.0):
    """Identical to model.py — see that file for full docstring."""
    B = pose_2d_px.shape[0]
    n_idx, m_idx = ref_bone_idx

    xn, yn   = pose_2d_px[:, n_idx, 0], pose_2d_px[:, n_idx, 1]
    xm, ym   = pose_2d_px[:, m_idx, 0], pose_2d_px[:, m_idx, 1]
    zrn, zrm = depth_rel[:, n_idx],     depth_rel[:, m_idx]

    a = (xn-xm)**2 + (yn-ym)**2
    b = (zrn*(xn**2+yn**2-xn*xm-yn*ym) +
         zrm*(xm**2+ym**2-xn*xm-yn*ym))
    c = ((xn*zrn-xm*zrm)**2 + (yn*zrn-ym*zrm)**2 +
         (zrn-zrm)**2 - C**2)

    disc   = torch.clamp(b**2 - 4*a*c, min=0.0)
    z_root = (-b + torch.sqrt(disc)) / (2*a.clamp(min=1e-8))

    Z_k = z_root.unsqueeze(1) + depth_rel
    fx  = K_mat[:, 0, 0].unsqueeze(1)
    fy  = K_mat[:, 1, 1].unsqueeze(1)
    cx  = K_mat[:, 0, 2].unsqueeze(1)
    cy  = K_mat[:, 1, 2].unsqueeze(1)

    X_k = (pose_2d_px[:, :, 0] - cx) * Z_k / fx
    Y_k = (pose_2d_px[:, :, 1] - cy) * Z_k / fy
    return torch.stack([X_k, Y_k, Z_k], dim=2)


# ============================================================================
# Quick sanity check
# ============================================================================

if __name__ == '__main__':
    from model import SingleViewModel as UNetModel

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x      = torch.randn(2, 3, 128, 128, device=device)

    # HRNet model
    hrnet  = SingleViewModel(num_kpts=21).to(device)
    p2d, depth, hm_logits, hm_probs = hrnet(x)

    hr_params = sum(p.numel() for p in hrnet.parameters())
    print('=== HRNet backbone ===')
    print(f'  Params     : {hr_params:,}  ({hr_params*4/1e6:.1f} MB)')
    print(f'  pose_2d    : {tuple(p2d.shape)}      (should be (2,21,2))')
    print(f'  depth_rel  : {tuple(depth.shape)}     (should be (2,21))')
    print(f'  hm_logits  : {tuple(hm_logits.shape)} (should be (2,21,128,128))')
    print(f'  hm_probs   : {tuple(hm_probs.shape)}  (should be (2,21,128,128))')

    # U-Net model for comparison
    unet   = UNetModel(num_kpts=21).to(device)
    un_params = sum(p.numel() for p in unet.parameters())
    print('\n=== U-Net backbone (model.py) ===')
    print(f'  Params     : {un_params:,}  ({un_params*4/1e6:.1f} MB)')

    print('\n=== Backbone comparison ===')
    print(f'  HRNet  params : {hr_params:,}')
    print(f'  U-Net  params : {un_params:,}')
    print(f'  Ratio         : {hr_params/un_params:.2f}×')

    # Backward pass check
    loss = p2d.sum() + depth.sum()
    loss.backward()
    print('\n  Backward pass: OK')