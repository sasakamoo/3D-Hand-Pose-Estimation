"""
visualize.py — Evaluate and visualize model predictions on dataset samples
===========================================================================
Per-sample figure with three panels on a dark background:
  Left  : RGB image with GT (orange dashed) and Pred (green solid) overlaid
  Middle: 3D Ground Truth skeleton on black + grid
  Right : 3D Predicted skeleton on black + grid

In interactive mode (--interactive) both 3D panels share a synchronized
camera — rotating one rotates the other.  Without the flag, each sample
is saved as a PNG.

Usage:
    # Save PNGs:
    python visualize.py --model sdf_best_model.pt --data-root /path/to/FreiHAND
    # Interactive viewer (one sample at a time, press any key to advance):
    python visualize.py --model sdf_best_model.pt --data-root /path/to/FreiHAND --interactive
"""

# ── Backend must be set before pyplot import ──────────────────────────────────
import argparse as _ap
_pre = _ap.ArgumentParser(add_help=False)
_pre.add_argument('--interactive', action='store_true')
_pre_args, _ = _pre.parse_known_args()

import matplotlib
if _pre_args.interactive:
    try:
        matplotlib.use('TkAgg')
    except Exception:
        matplotlib.use('Qt5Agg')
else:
    matplotlib.use('Agg')

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


from model        import SingleViewModel, reconstruct_3d_from_25d
from model_sdf    import SDFHandPoseNet, N_PTS
from model_hybrid import HybridHandPoseNet
from dataset      import FreiHANDDataset, IMG_SIZE

# ── Skeleton ──────────────────────────────────────────────────────────────────

CONNECTIONS = [
    [0, 1],[1, 2],[2, 3],[3, 4],
    [0, 5],[5, 6],[6, 7],[7, 8],
    [0, 9],[9,10],[10,11],[11,12],
    [0,13],[13,14],[14,15],[15,16],
    [0,17],[17,18],[18,19],[19,20],
]

# Vivid finger colours for predicted skeleton
FINGER_COLORS = np.array([
    [0.85, 0.85, 0.85],  # 0  wrist
    [0.20, 1.00, 0.20],  # 1–4  index  (bright green)
    [0.20, 1.00, 0.20],
    [0.20, 1.00, 0.20],
    [0.20, 1.00, 0.20],
    [0.20, 0.60, 1.00],  # 5–8  middle (sky blue)
    [0.20, 0.60, 1.00],
    [0.20, 0.60, 1.00],
    [0.20, 0.60, 1.00],
    [1.00, 0.85, 0.00],  # 9–12 ring   (gold)
    [1.00, 0.85, 0.00],
    [1.00, 0.85, 0.00],
    [1.00, 0.85, 0.00],
    [1.00, 0.45, 0.10],  # 13–16 pinky (orange)
    [1.00, 0.45, 0.10],
    [1.00, 0.45, 0.10],
    [1.00, 0.45, 0.10],
    [0.90, 0.20, 0.90],  # 17–20 thumb (magenta)
    [0.90, 0.20, 0.90],
    [0.90, 0.20, 0.90],
    [0.70, 0.10, 1.00],
], dtype=np.float64)

# Muted grey-blue for GT skeleton
GT_COLORS = np.full((21, 3), [0.55, 0.65, 0.85], dtype=np.float64)
GT_COLORS[0] = [0.85, 0.85, 0.85]  # wrist slightly brighter

BG_COLOR   = '#0D0D0D'   # near-black figure background
PANEL_FACE = (0.06, 0.06, 0.06, 1.0)   # 3D pane fill
GRID_COLOR = (0.22, 0.22, 0.22, 0.9)
TICK_COLOR = '#555555'


# ── Styling helpers ───────────────────────────────────────────────────────────

def style_3d_ax(ax, title: str, elev: float = 20, azim: float = -60):
    """Apply dark theme + grid to a 3D axis."""
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
    ax.set_title(title, color='white', fontsize=10, pad=6, fontweight='bold')
    ax.view_init(elev=elev, azim=azim)


def draw_skeleton_3d(ax, joints: np.ndarray, colors: np.ndarray,
                     lw: float = 1.8, ms: float = 28):
    """Draw a 3D hand skeleton on a matplotlib 3D axis."""
    for s, e in CONNECTIONS:
        col = colors[s]
        ax.plot([joints[s, 0], joints[e, 0]],
                [joints[s, 1], joints[e, 1]],
                [joints[s, 2], joints[e, 2]],
                color=col, lw=lw, alpha=0.95, solid_capstyle='round')
    for pt, col in zip(joints, colors):
        ax.scatter(*pt, color=col, s=ms, zorder=5,
                   edgecolors='white', linewidths=0.4)


def set_equal_axes_3d(ax, joints_a: np.ndarray, joints_b: np.ndarray | None = None):
    """Force equal axis scaling so the hand doesn't look distorted."""
    pts = joints_a if joints_b is None else np.vstack([joints_a, joints_b])
    lo, hi = pts.min(0), pts.max(0)
    mid = (lo + hi) / 2
    r   = (hi - lo).max() / 2 * 1.15
    ax.set_xlim(mid[0]-r, mid[0]+r)
    ax.set_ylim(mid[1]-r, mid[1]+r)
    ax.set_zlim(mid[2]-r, mid[2]+r)


# ── 2D overlay ────────────────────────────────────────────────────────────────

def _draw_skel_2d(ax, pts, color, lw, ls, alpha_bone, alpha_joint, ms):
    for s, e in CONNECTIONS:
        ax.plot([pts[s, 0], pts[e, 0]],
                [pts[s, 1], pts[e, 1]],
                color=color, lw=lw, linestyle=ls, alpha=alpha_bone,
                solid_capstyle='round')
    ax.scatter(pts[:, 0], pts[:, 1],
               s=ms, color=color, alpha=alpha_joint,
               edgecolors='white', linewidths=0.4, zorder=5)


def _style_2d_ax(ax):
    ax.set_facecolor('black')
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor('#333333')


def draw_2d_gt(ax, img_rgb: np.ndarray, gt_px: np.ndarray):
    """Image with GT skeleton only (orange solid)."""
    ax.imshow(img_rgb, aspect='equal')
    _style_2d_ax(ax)
    _draw_skel_2d(ax, gt_px, '#FF8C00', lw=1.8, ls='-',
                  alpha_bone=0.90, alpha_joint=0.92, ms=22)


def draw_2d_pred(ax, img_rgb: np.ndarray, pred_px: np.ndarray):
    """Image with Pred skeleton only (green solid)."""
    ax.imshow(img_rgb, aspect='equal')
    _style_2d_ax(ax)
    _draw_skel_2d(ax, pred_px, '#00FF66', lw=1.8, ls='-',
                  alpha_bone=0.95, alpha_joint=0.95, ms=22)


# ── Per-sample figure ─────────────────────────────────────────────────────────


def make_figure(img_rgb, pred_px, gt_px, p3, g3,
                idx, e2d, e3d, e3d_mm):
    """
    Build a dark-themed 4-panel figure:
      [2D GT | 2D Pred | 3D GT | 3D Pred]

    Returns (fig, ax_gt, ax_pred) so the caller can wire up camera sync.
    """
    fig = plt.figure(figsize=(20, 6), facecolor=BG_COLOR)
    fig.patch.set_facecolor(BG_COLOR)

    gs = gridspec.GridSpec(1, 4, figure=fig,
                           left=0.02, right=0.98,
                           bottom=0.06, top=0.88,
                           wspace=0.08)

    # ── Panel 0: 2D GT overlay ────────────────────────────────────────────
    ax_2d_gt = fig.add_subplot(gs[0])
    draw_2d_gt(ax_2d_gt, img_rgb, gt_px)
    ax_2d_gt.set_title('2D Ground Truth',
                        color='white', fontsize=9, fontweight='bold', pad=6)

    # ── Panel 1: 2D Pred overlay ──────────────────────────────────────────
    ax_2d_pred = fig.add_subplot(gs[1])
    draw_2d_pred(ax_2d_pred, img_rgb, pred_px)
    ax_2d_pred.set_title('2D Predicted',
                          color='white', fontsize=9, fontweight='bold', pad=6)

    # ── Panel 2: 3D GT ────────────────────────────────────────────────────
    ax_gt = fig.add_subplot(gs[2], projection='3d')
    style_3d_ax(ax_gt, '3D Ground Truth', elev=22, azim=-55)
    draw_skeleton_3d(ax_gt, g3, GT_COLORS, lw=2.0, ms=32)
    set_equal_axes_3d(ax_gt, g3)

    # ── Panel 3: 3D Predicted ─────────────────────────────────────────────
    ax_pred = fig.add_subplot(gs[3], projection='3d')
    style_3d_ax(ax_pred, '3D Predicted', elev=22, azim=-55)
    draw_skeleton_3d(ax_pred, p3, FINGER_COLORS, lw=2.0, ms=32)
    set_equal_axes_3d(ax_pred, p3)

    # ── Title ─────────────────────────────────────────────────────────────
    fig.suptitle(
        f'Sample {idx}  ·  2D err = {e2d:.2f} px  ·  3D MPJPE = {e3d:.4f} (norm)  /  {e3d_mm:.1f} mm',
        color='white', fontsize=11, fontweight='bold', y=0.97,
    )

    # Subtle separator lines between panels
    for xpos in [0.265, 0.510, 0.755]:
        fig.add_artist(
            plt.Line2D([xpos, xpos], [0.04, 0.95],
                       transform=fig.transFigure,
                       color='#333333', lw=0.8))

    return fig, ax_gt, ax_pred


# ── Camera sync callback (interactive only) ───────────────────────────────────

def wire_camera_sync(fig, ax_gt, ax_pred):
    """
    Synchronize the two 3D viewpoints: rotating one panel mirrors the other.
    Listens on both motion_notify (live drag) and button_release (end of drag)
    so the cameras stay in sync regardless of backend behaviour.
    """
    _syncing = [False]

    def _sync(event):
        if _syncing[0]:
            return
        if event.inaxes == ax_gt:
            _syncing[0] = True
            ax_pred.view_init(elev=ax_gt.elev, azim=ax_gt.azim)
            fig.canvas.draw_idle()
            _syncing[0] = False
        elif event.inaxes == ax_pred:
            _syncing[0] = True
            ax_gt.view_init(elev=ax_pred.elev, azim=ax_pred.azim)
            fig.canvas.draw_idle()
            _syncing[0] = False

    fig.canvas.mpl_connect('motion_notify_event', _sync)
    fig.canvas.mpl_connect('button_release_event', _sync)


# ── Open3D PLY export (unchanged from original) ───────────────────────────────

JOINT_COLORS_NP = FINGER_COLORS.copy()
GT_COLOR_NP     = np.array([0.6, 0.6, 0.6], dtype=np.float64)


def export_ply(p3, g3, idx, out_dir, o3d):
    offset  = np.array([max(p3[:, 0].max() - g3[:, 0].min() + 0.2, 0.5), 0, 0])
    g3_off  = g3 + offset
    all_pts = np.vstack([p3, g3_off])
    all_col = np.vstack([JOINT_COLORS_NP, np.tile(GT_COLOR_NP, (21, 1))])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_pts)
    pcd.colors = o3d.utility.Vector3dVector(all_col)
    path = str(out_dir / f'3d_sample_{idx:04d}.ply')
    o3d.io.write_point_cloud(path, pcd)
    return path


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',       type=str, required=True)
    parser.add_argument('--model-type',  type=str, default='sdf',
                        choices=['heatmap', 'sdf', 'hybrid'])
    parser.add_argument('--data-root',   type=str,
                        default='/home/kghasemz/scratch/datasets/FreiHand')
    parser.add_argument('--split',       type=str, default='val',
                        choices=['train', 'val'])
    parser.add_argument('--n-samples',   type=int, default=8)
    parser.add_argument('--start-idx',   type=int, default=0)
    parser.add_argument('--output-dir',  type=str, default='visualizations')
    parser.add_argument('--interactive', action='store_true',
                        help='Show interactive rotating viewer (one sample at a time)')
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
    ckpt  = torch.load(args.model, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    model = model.to(device).eval()
    print(f'Loaded {args.model_type} model  [{args.model}]')
    if 'epoch' in ckpt:
        print(f'  Epoch {ckpt["epoch"]}  |  best MPJPE {ckpt.get("best_mpjpe", 0):.4f}\n')

    # ── Open3D (optional) ─────────────────────────────────────────────────
    try:
        import open3d as o3d
        has_o3d = True
    except ImportError:
        has_o3d = False
        print('  open3d not found — skipping PLY export\n')

    # ── Dataset ───────────────────────────────────────────────────────────
    ds = FreiHANDDataset(args.data_root, split=args.split, augment=False)
    n  = min(args.n_samples, len(ds) - args.start_idx)
    print(f'Visualizing {n} samples  (split={args.split}, '
          f'idx {args.start_idx}–{args.start_idx + n - 1})\n')

    errs_2d, errs_3d, errs_3d_mm = [], [], []

    for local_i in range(n):
        idx   = args.start_idx + local_i
        batch = ds[idx]

        img_t = batch['image'].unsqueeze(0).to(device)
        K_t   = batch['K_mat'].unsqueeze(0).to(device)
        gt_2d = batch['pose_2d_gt'].numpy()      # (21,2) px

        with torch.no_grad():
            if args.model_type == 'sdf':
                ext_pts = torch.empty(1, N_PTS, 3, device=device).uniform_(-1., 1.)
                pred_2d, pred_z, _, _ = model(img_t, ext_pts=ext_pts)
            elif args.model_type == 'hybrid':
                pred_2d, pred_z, _, _, _ = model(img_t)
            else:
                pred_2d, pred_z, _, _ = model(img_t)

        pred_2d_px = pred_2d[0].cpu().numpy()    # (21,2) px

        # 3D reconstruction
        gt_2d_t = batch['pose_2d_gt'].unsqueeze(0).to(device)
        gt_z_t  = batch['depth_rel_gt'].unsqueeze(0).to(device)
        p3 = reconstruct_3d_from_25d(pred_2d, pred_z, K_t, img_size=IMG_SIZE
                                     ).cpu().numpy()[0]
        g3 = reconstruct_3d_from_25d(gt_2d_t,  gt_z_t,  K_t, img_size=IMG_SIZE
                                     ).cpu().numpy()[0]
        p3 -= p3[0]; g3 -= g3[0]   # root-relative

        scale_m  = float(batch['scale_factor'])   # wrist→index MCP in metres
        e2d      = float(np.linalg.norm(pred_2d_px - gt_2d, axis=-1).mean())
        e3d      = float(np.linalg.norm(p3 - g3, axis=-1).mean())
        e3d_mm   = e3d * scale_m * 1000
        errs_2d.append(e2d)
        errs_3d.append(e3d)
        errs_3d_mm.append(e3d_mm)

        img_rgb = (batch['image'].permute(1, 2, 0).numpy() * 255).astype(np.uint8)

        fig, ax_gt, ax_pred = make_figure(
            img_rgb, pred_2d_px, gt_2d, p3, g3, idx, e2d, e3d, e3d_mm,
        )

        if args.interactive:
            wire_camera_sync(fig, ax_gt, ax_pred)
            plt.show(block=True)
        else:
            png_path = out_dir / f'2d_sample_{idx:04d}.png'
            plt.savefig(str(png_path), dpi=150, bbox_inches='tight',
                        facecolor=BG_COLOR)
            plt.close(fig)

        # PLY export
        ply_tag = ''
        if has_o3d:
            ply_path = export_ply(p3, g3, idx, out_dir, o3d)
            ply_tag  = f'  + {Path(ply_path).name}'

        print(f'  [{local_i+1:>2}/{n}]  idx={idx:>5}  '
              f'2D={e2d:6.2f}px  3D={e3d:.4f} ({e3d_mm:.1f}mm)'
              + (f'  → {png_path.name}' if not args.interactive else '')
              + ply_tag)

    print(f'\nMean 2D err  : {np.mean(errs_2d):.3f} px')
    print(f'Mean 3D MPJPE: {np.mean(errs_3d):.4f} (norm)  /  {np.mean(errs_3d_mm):.1f} mm')
    if not args.interactive:
        print(f'\nPNGs saved to: {out_dir}/')
    if has_o3d:
        print(f'PLYs saved to: {out_dir}/')
        print(f'\nTo view PLYs:')
        print(f'  python view_3d.py --ply-dir {out_dir} --sample {args.start_idx}')


if __name__ == '__main__':
    main()
