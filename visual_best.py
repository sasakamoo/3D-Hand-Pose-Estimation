"""
visual_best.py — Find and visualize the best-predicted sample in the dataset
=============================================================================
Scans every sample in the chosen split, ranks by 3D MPJPE (lowest = best),
then for the top sample saves four individual images plus one side-by-side
3D comparison figure into  best_results/

Output files
------------
  best_results/
    raw_image.png            — unaltered RGB image
    overlay_gt_2d.png        — image + ground-truth 2D skeleton (orange)
    overlay_pred_2d.png      — image + predicted  2D skeleton  (green)
    sidebyside_3d.png        — 3D GT (left) vs 3D Predicted (right)

Usage
-----
    python visual_best.py --model sdf_best_model.pt --data-root /path/to/FreiHAND
    python visual_best.py --model hybrid_best_model.pt --model-type hybrid \\
                          --data-root /path/to/FreiHAND --split val
"""

import matplotlib
matplotlib.use('Agg')

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm

from model        import SingleViewModel, reconstruct_3d_from_25d
from model_sdf    import SDFHandPoseNet, N_PTS
from model_hybrid import HybridHandPoseNet
from dataset      import FreiHANDDataset, IMG_SIZE

# ── Skeleton definition ───────────────────────────────────────────────────────

CONNECTIONS = [
    [0, 1],[1, 2],[2, 3],[3, 4],
    [0, 5],[5, 6],[6, 7],[7, 8],
    [0, 9],[9,10],[10,11],[11,12],
    [0,13],[13,14],[14,15],[15,16],
    [0,17],[17,18],[18,19],[19,20],
]

FINGER_COLORS = np.array([
    [0.85, 0.85, 0.85],
    [0.20, 1.00, 0.20],[0.20, 1.00, 0.20],[0.20, 1.00, 0.20],[0.20, 1.00, 0.20],
    [0.20, 0.60, 1.00],[0.20, 0.60, 1.00],[0.20, 0.60, 1.00],[0.20, 0.60, 1.00],
    [1.00, 0.85, 0.00],[1.00, 0.85, 0.00],[1.00, 0.85, 0.00],[1.00, 0.85, 0.00],
    [1.00, 0.45, 0.10],[1.00, 0.45, 0.10],[1.00, 0.45, 0.10],[1.00, 0.45, 0.10],
    [0.90, 0.20, 0.90],[0.90, 0.20, 0.90],[0.90, 0.20, 0.90],[0.70, 0.10, 1.00],
], dtype=np.float64)

GT_COLORS = np.full((21, 3), [0.55, 0.65, 0.85], dtype=np.float64)
GT_COLORS[0] = [0.85, 0.85, 0.85]

BG_COLOR   = '#0D0D0D'
PANEL_FACE = (0.06, 0.06, 0.06, 1.0)
GRID_COLOR = (0.22, 0.22, 0.22, 0.9)
TICK_COLOR = '#555555'


# ── Drawing helpers ───────────────────────────────────────────────────────────

def _style_2d_ax(ax):
    ax.set_facecolor('black')
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor('#333333')


def _draw_skel_2d(ax, pts, color, lw=1.8, ls='-',
                  alpha_bone=0.92, alpha_joint=0.95, ms=26):
    for s, e in CONNECTIONS:
        ax.plot([pts[s, 0], pts[e, 0]],
                [pts[s, 1], pts[e, 1]],
                color=color, lw=lw, linestyle=ls,
                alpha=alpha_bone, solid_capstyle='round')
    ax.scatter(pts[:, 0], pts[:, 1],
               s=ms, color=color, alpha=alpha_joint,
               edgecolors='white', linewidths=0.4, zorder=5)


def style_3d_ax(ax, title, elev=22, azim=-55):
    ax.set_facecolor(BG_COLOR)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = True
        pane.set_facecolor(PANEL_FACE)
        pane.set_edgecolor('#1a1a1a')
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis._axinfo['grid']['color']     = GRID_COLOR
        axis._axinfo['grid']['linewidth'] = 0.6
        axis._axinfo['tick']['color']     = TICK_COLOR
    ax.tick_params(colors=TICK_COLOR, labelsize=6)
    ax.xaxis.label.set_color(TICK_COLOR)
    ax.yaxis.label.set_color(TICK_COLOR)
    ax.zaxis.label.set_color(TICK_COLOR)
    ax.set_xlabel('X', fontsize=7, labelpad=2)
    ax.set_ylabel('Y', fontsize=7, labelpad=2)
    ax.set_zlabel('Z', fontsize=7, labelpad=2)
    ax.set_title(title, color='white', fontsize=11, pad=6, fontweight='bold')
    ax.view_init(elev=elev, azim=azim)


def draw_skeleton_3d(ax, joints, colors, lw=2.0, ms=34):
    for s, e in CONNECTIONS:
        col = colors[s]
        ax.plot([joints[s, 0], joints[e, 0]],
                [joints[s, 1], joints[e, 1]],
                [joints[s, 2], joints[e, 2]],
                color=col, lw=lw, alpha=0.95, solid_capstyle='round')
    for pt, col in zip(joints, colors):
        ax.scatter(*pt, color=col, s=ms, zorder=5,
                   edgecolors='white', linewidths=0.4)


def set_equal_axes_3d(ax, joints_a, joints_b=None):
    pts = joints_a if joints_b is None else np.vstack([joints_a, joints_b])
    lo, hi = pts.min(0), pts.max(0)
    mid = (lo + hi) / 2
    r   = (hi - lo).max() / 2 * 1.15
    ax.set_xlim(mid[0]-r, mid[0]+r)
    ax.set_ylim(mid[1]-r, mid[1]+r)
    ax.set_zlim(mid[2]-r, mid[2]+r)


# ── Individual image savers ───────────────────────────────────────────────────

def save_raw_image(img_rgb: np.ndarray, path: Path):
    """Save the unaltered RGB image."""
    fig, ax = plt.subplots(figsize=(5, 5), facecolor='black')
    ax.imshow(img_rgb, aspect='equal')
    _style_2d_ax(ax)
    ax.set_title('Input Image', color='white', fontsize=11,
                 fontweight='bold', pad=6)
    fig.tight_layout(pad=0.3)
    fig.savefig(str(path), dpi=150, bbox_inches='tight', facecolor='black')
    plt.close(fig)


def save_gt_overlay(img_rgb: np.ndarray, gt_px: np.ndarray,
                    path: Path, e2d: float):
    """Save image with ground-truth skeleton overlay (orange)."""
    fig, ax = plt.subplots(figsize=(5, 5), facecolor='black')
    ax.imshow(img_rgb, aspect='equal')
    _style_2d_ax(ax)
    _draw_skel_2d(ax, gt_px, '#FF8C00')
    ax.set_title(f'Ground Truth 2D  ·  ref MPJPE-2D = {e2d:.2f} px',
                 color='white', fontsize=10, fontweight='bold', pad=6)
    fig.tight_layout(pad=0.3)
    fig.savefig(str(path), dpi=150, bbox_inches='tight', facecolor='black')
    plt.close(fig)


def save_pred_overlay(img_rgb: np.ndarray, pred_px: np.ndarray,
                      path: Path, e2d: float):
    """Save image with predicted skeleton overlay (green)."""
    fig, ax = plt.subplots(figsize=(5, 5), facecolor='black')
    ax.imshow(img_rgb, aspect='equal')
    _style_2d_ax(ax)
    _draw_skel_2d(ax, pred_px, '#00FF66')
    ax.set_title(f'Predicted 2D  ·  MPJPE-2D = {e2d:.2f} px',
                 color='white', fontsize=10, fontweight='bold', pad=6)
    fig.tight_layout(pad=0.3)
    fig.savefig(str(path), dpi=150, bbox_inches='tight', facecolor='black')
    plt.close(fig)


def save_3d_sidebyside(p3: np.ndarray, g3: np.ndarray, path: Path,
                       sample_idx: int, e3d: float, e3d_mm: float):
    """Save a side-by-side 3D GT / 3D Predicted figure."""
    fig = plt.figure(figsize=(13, 6), facecolor=BG_COLOR)
    fig.patch.set_facecolor(BG_COLOR)

    gs = gridspec.GridSpec(1, 2, figure=fig,
                           left=0.04, right=0.96,
                           bottom=0.06, top=0.88,
                           wspace=0.10)

    ax_gt   = fig.add_subplot(gs[0], projection='3d')
    ax_pred = fig.add_subplot(gs[1], projection='3d')

    style_3d_ax(ax_gt,   '3D Ground Truth',  elev=22, azim=-55)
    style_3d_ax(ax_pred, '3D Predicted',     elev=22, azim=-55)

    draw_skeleton_3d(ax_gt,   g3, GT_COLORS,    lw=2.2, ms=36)
    draw_skeleton_3d(ax_pred, p3, FINGER_COLORS, lw=2.2, ms=36)

    # Shared axis limits so scale is comparable
    set_equal_axes_3d(ax_gt,   g3, p3)
    set_equal_axes_3d(ax_pred, p3, g3)

    fig.suptitle(
        f'Best Sample  ·  idx {sample_idx}  ·  3D MPJPE = {e3d:.4f} (norm)  /  {e3d_mm:.2f} mm',
        color='white', fontsize=12, fontweight='bold', y=0.97,
    )

    fig.add_artist(
        plt.Line2D([0.50, 0.50], [0.04, 0.94],
                   transform=fig.transFigure,
                   color='#333333', lw=0.8))

    fig.savefig(str(path), dpi=150, bbox_inches='tight', facecolor=BG_COLOR)
    plt.close(fig)


# ── Inference helper ──────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(model, model_type, batch, device):
    img_t = batch['image'].unsqueeze(0).to(device)
    K_t   = batch['K_mat'].unsqueeze(0).to(device)

    if model_type == 'sdf':
        ext_pts = torch.empty(1, N_PTS, 3, device=device).uniform_(-1., 1.)
        pred_2d, pred_z, _, _ = model(img_t, ext_pts=ext_pts)
    elif model_type == 'hybrid':
        pred_2d, pred_z, _, _, _ = model(img_t)
    else:
        pred_2d, pred_z, _, _ = model(img_t)

    gt_2d_t = batch['pose_2d_gt'].unsqueeze(0).to(device)
    gt_z_t  = batch['depth_rel_gt'].unsqueeze(0).to(device)

    p3 = reconstruct_3d_from_25d(pred_2d, pred_z, K_t, img_size=IMG_SIZE
                                  ).cpu().numpy()[0]
    g3 = reconstruct_3d_from_25d(gt_2d_t,  gt_z_t,  K_t, img_size=IMG_SIZE
                                  ).cpu().numpy()[0]
    p3 -= p3[0]; g3 -= g3[0]   # root-relative

    pred_2d_px = pred_2d[0].cpu().numpy()
    gt_2d_px   = batch['pose_2d_gt'].numpy()
    scale_m    = float(batch['scale_factor'])

    e2d    = float(np.linalg.norm(pred_2d_px - gt_2d_px, axis=-1).mean())
    e3d    = float(np.linalg.norm(p3 - g3, axis=-1).mean())
    e3d_mm = e3d * scale_m * 1000

    return pred_2d_px, gt_2d_px, p3, g3, e2d, e3d, e3d_mm


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',      type=str, required=True)
    parser.add_argument('--model-type', type=str, default='sdf',
                        choices=['heatmap', 'sdf', 'hybrid'])
    parser.add_argument('--data-root',  type=str,
                        default='/home/kghasemz/scratch/datasets/FreiHand')
    parser.add_argument('--split',      type=str, default='val',
                        choices=['train', 'val'])
    parser.add_argument('--output-dir', type=str, default='best_results')
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── Load model ────────────────────────────────────────────────────────
    if args.model_type == 'sdf':
        model = SDFHandPoseNet(num_kpts=21, pretrained_backbone=False)
    elif args.model_type == 'hybrid':
        model = HybridHandPoseNet(num_kpts=21, pretrained_backbone=False)
    else:
        model = SingleViewModel(num_kpts=21)

    ckpt = torch.load(args.model, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    model = model.to(device).eval()
    print(f'Loaded {args.model_type} model  [{args.model}]')
    if 'epoch' in ckpt:
        print(f'  Epoch {ckpt["epoch"]}  |  best MPJPE {ckpt.get("best_mpjpe", 0):.4f}')

    # ── Dataset ───────────────────────────────────────────────────────────
    ds = FreiHANDDataset(args.data_root, split=args.split, augment=False)
    print(f'\nScanning {len(ds)} samples (split={args.split}) …')

    # ── Pass 1: score every sample ────────────────────────────────────────
    best_idx   = -1
    best_e3d   = float('inf')
    all_e3d_mm = []

    for i in tqdm(range(len(ds)), desc='Scoring', ncols=80):
        batch = ds[i]
        try:
            _, _, _, _, _, e3d, e3d_mm = run_inference(
                model, args.model_type, batch, device)
        except Exception as exc:
            print(f'  Warning: sample {i} failed ({exc}), skipping.')
            all_e3d_mm.append(float('inf'))
            continue

        all_e3d_mm.append(e3d_mm)
        if e3d < best_e3d:
            best_e3d = e3d
            best_idx = i

    print(f'\nBest sample  :  dataset index {best_idx}  '
          f'(3D MPJPE = {best_e3d:.4f} norm / {all_e3d_mm[best_idx]:.2f} mm)')
    print(f'Mean 3D MPJPE:  {np.mean([v for v in all_e3d_mm if v < float("inf")]):.2f} mm')

    # ── Pass 2: re-infer best sample and save outputs ─────────────────────
    batch = ds[best_idx]
    pred_px, gt_px, p3, g3, e2d, e3d, e3d_mm = run_inference(
        model, args.model_type, batch, device)

    img_rgb = (batch['image'].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    sample_idx = int(batch['sample_idx'])

    raw_path   = out_dir / 'raw_image.png'
    gt_path    = out_dir / 'overlay_gt_2d.png'
    pred_path  = out_dir / 'overlay_pred_2d.png'
    sbs_path   = out_dir / 'sidebyside_3d.png'

    save_raw_image(img_rgb, raw_path)
    save_gt_overlay(img_rgb, gt_px,   gt_path,   e2d)
    save_pred_overlay(img_rgb, pred_px, pred_path, e2d)
    save_3d_sidebyside(p3, g3, sbs_path, sample_idx, e3d, e3d_mm)

    print(f'\nSaved to {out_dir}/')
    for p in [raw_path, gt_path, pred_path, sbs_path]:
        print(f'  {p.name}')


if __name__ == '__main__':
    main()
