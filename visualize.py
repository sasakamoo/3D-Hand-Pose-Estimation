"""
visualize.py — Evaluate and visualize model predictions on dataset samples
===========================================================================
For each sample:
  1. Saves a 2D skeleton PNG (predicted in green, GT in orange) overlaid on image
  2. Saves Open3D .ply files (joints + skeleton) for 3D inspection

Usage:
    python3 visualize.py --model best_model.pt --data-root /path/to/FreiHAND
    python3 visualize.py --model best_model.pt --data-root /path/to/FreiHAND --n-samples 16
    python3 visualize.py --model best_model.pt --data-root /path/to/FreiHAND --split train

After running:
    python3 view_3d.py --sample 0     # interactive 3D viewer
"""

import argparse
import os
from pathlib import Path
import numpy as np
import cv2
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model     import SingleViewModel, reconstruct_3d_from_25d
from model_sdf import SDFHandPoseNet
from dataset   import FreiHANDDataset, IMG_SIZE

CONNECTIONS = [
    [0,1],[1,2],[2,3],[3,4],
    [0,5],[5,6],[6,7],[7,8],
    [0,9],[9,10],[10,11],[11,12],
    [0,13],[13,14],[14,15],[15,16],
    [0,17],[17,18],[18,19],[19,20],
]

# Per-joint colors (float RGB [0,1]), one per finger
JOINT_COLORS_NP = np.array([
    [0.8, 0.8, 0.8],   # 0  wrist     — grey
    [0.0, 0.9, 0.0],   # 1  index
    [0.0, 0.9, 0.0],   # 2
    [0.0, 0.9, 0.0],   # 3
    [0.0, 0.9, 0.0],   # 
    [0.0, 0.5, 1.0],   # 5  middle
    [0.0, 0.5, 1.0],   # 6
    [0.0, 0.5, 1.0],   # 7
    [0.0, 0.5, 1.0],   # 8
    [1.0, 0.8, 0.0],   # 9  ring
    [1.0, 0.8, 0.0],   # 10
    [1.0, 0.8, 0.0],   # 11
    [1.0, 0.8, 0.0],   # 12
    [1.0, 0.4, 0.0],   # 13 pinky
    [1.0, 0.4, 0.0],   # 14
    [1.0, 0.4, 0.0],   # 15
    [1.0, 0.4, 0.0],   # 16
    [0.9, 0.0, 0.9],   # 17 thumb
    [0.9, 0.0, 0.9],   # 18
    [0.9, 0.0, 0.9],   # 19
    [0.6, 0.1, 0.9],   # 20
], dtype=np.float64)

GT_COLOR = np.array([0.6, 0.6, 0.6], dtype=np.float64)   # grey for GT


# ─────────────────────────────────────────────────────────────────────────────
# 2D drawing
# ─────────────────────────────────────────────────────────────────────────────

def draw_skeleton_2d(img, kpts_px, color_bone, color_joint, W, H,
                     line_thickness=1, joint_radius=2):
    vis = img.copy()
    for s, e in CONNECTIONS:
        p1 = tuple(np.clip(kpts_px[s].astype(int), [0, 0], [W-1, H-1]))
        p2 = tuple(np.clip(kpts_px[e].astype(int), [0, 0], [W-1, H-1]))
        cv2.line(vis, p1, p2, color_bone, line_thickness)
    for pt in kpts_px:
        c = tuple(np.clip(pt.astype(int), [0, 0], [W-1, H-1]))
        cv2.circle(vis, c, joint_radius, color_joint, -1)
    return vis


# ─────────────────────────────────────────────────────────────────────────────
# Open3D helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_hand_pcd(joints, colors_per_joint, o3d):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(joints.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors_per_joint)
    return pcd


def make_hand_lineset(joints, connections, color, o3d):
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(joints.astype(np.float64))
    ls.lines  = o3d.utility.Vector2iVector(np.array(connections, dtype=np.int32))
    if isinstance(color, np.ndarray) and color.ndim == 1:
        color = np.tile(color, (len(connections), 1))
    ls.colors = o3d.utility.Vector3dVector(np.array(color, dtype=np.float64))
    return ls


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',       type=str, required=True,
                        help='Path to checkpoint')
    parser.add_argument('--model-type',  type=str, default='heatmap',
                        choices=['heatmap', 'sdf'],
                        help='heatmap = SingleViewModel, sdf = SDFHandPoseNet')
    parser.add_argument('--data-root',  type=str, default='/home/kghasemz/projects/def-vislearn/kghasemz/dataset')
    parser.add_argument('--split',      type=str, default='val',
                        choices=['train', 'val'])
    parser.add_argument('--n-samples',  type=int, default=8,
                        help='How many samples to visualize')
    parser.add_argument('--start-idx',  type=int, default=0,
                        help='Dataset index of the first sample')
    parser.add_argument('--output-dir', type=str, default='visualizations')
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── Load model ────────────────────────────────────────────────────────
    if args.model_type == 'sdf':
        model = SDFHandPoseNet(num_kpts=21, pretrained_backbone=False)
    else:
        model = SingleViewModel(num_kpts=21)
    ckpt  = torch.load(args.model, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    model = model.to(device).eval()
    print(f'Loaded {args.model_type} model from {args.model}')
    if 'epoch' in ckpt:
        print(f'  Epoch {ckpt["epoch"]}  |  best MPJPE {ckpt.get("best_mpjpe", "?"):.4f}')

    # ── Dataset ───────────────────────────────────────────────────────────
    ds = FreiHANDDataset(args.data_root, split=args.split, augment=False)
    n  = min(args.n_samples, len(ds) - args.start_idx)
    print(f'\nVisualizing {n} samples from {args.split} split '
          f'(idx {args.start_idx}–{args.start_idx+n-1})\n')

    errs_2d_all, errs_3d_all = [], []
    ply_pairs = []   # (joints_path, skeleton_path) per sample

    # ── Try importing open3d once ─────────────────────────────────────────
    # DOWNLOAD PYTHON 3.10.13 ------ IMPORTANT
    try:
        import open3d as o3d
        has_o3d = True
    except ImportError:
        has_o3d = False
        print('  open3d not installed — skipping .ply export')
        print('  Install with: pip install open3d\n')

    # ── Per-sample loop ───────────────────────────────────────────────────
    for local_i in range(n):
        idx   = args.start_idx + local_i
        batch = ds[idx]

        img_t  = batch['image'].unsqueeze(0).to(device)
        K_t    = batch['K_mat'].unsqueeze(0).to(device)
        gt_2d  = batch['pose_2d_gt'].numpy()      # (21, 2) px
        K_mat  = batch['K_mat'].numpy()           # (3, 3)

        with torch.no_grad():
            pred_2d, pred_z, _, _ = model(img_t)

        pred_2d_px = pred_2d[0].cpu().numpy()     # (21, 2)

        # 3D reconstruction — same quadratic solve for both pred and GT
        pose_3d_pred = reconstruct_3d_from_25d(
            pred_2d, pred_z, K_t, img_size=IMG_SIZE).cpu().numpy()[0]

        gt_2d_t = batch['pose_2d_gt'].unsqueeze(0).to(device)
        gt_z_t  = batch['depth_rel_gt'].unsqueeze(0).to(device)
        pose_3d_gt = reconstruct_3d_from_25d(
            gt_2d_t, gt_z_t, K_t, img_size=IMG_SIZE).cpu().numpy()[0]

        # Root-relative
        p3 = pose_3d_pred - pose_3d_pred[0]
        g3 = pose_3d_gt   - pose_3d_gt[0]

        # ── 2D image save ──────────────────────────────────────────────
        img_np = (batch['image'].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        H_im, W_im = img_np.shape[:2]

        vis_pred = draw_skeleton_2d(img_np, pred_2d_px,
                                    (0, 220, 0), (0, 0, 255),
                                    W_im, H_im,
                                    line_thickness=1, joint_radius=2)
        vis_gt   = draw_skeleton_2d(img_np, gt_2d,
                                    (255, 140, 0), (220, 0, 0),
                                    W_im, H_im,
                                    line_thickness=1, joint_radius=2)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(img_np)
        axes[0].set_title('Input', fontsize=9)
        axes[0].axis('off')
        axes[1].imshow(vis_pred)
        axes[1].set_title('Predicted  (green / blue)', fontsize=9)
        axes[1].axis('off')
        axes[2].imshow(vis_gt)
        axes[2].set_title('GT  (orange / red)', fontsize=9)
        axes[2].axis('off')

        e2d = np.linalg.norm(pred_2d_px - gt_2d, axis=-1).mean()
        e3d = np.linalg.norm(p3 - g3, axis=-1).mean()
        errs_2d_all.append(e2d)
        errs_3d_all.append(e3d)

        plt.suptitle(f'Sample {idx} — 2D err={e2d:.2f}px  3D MPJPE={e3d:.4f}',
                     fontsize=9)
        plt.tight_layout()
        png_path = out_dir / f'2d_sample_{idx:04d}.png'
        plt.savefig(str(png_path), dpi=150, bbox_inches='tight')
        plt.close()

        # ── PLY save ───────────────────────────────────────────────────
        if has_o3d:
            # Offset GT to the right so both hands are visible in viewer
            offset = np.array([max(p3[:, 0].max() - g3[:, 0].min() + 0.2, 0.5), 0, 0])
            g3_off = g3 + offset

            # Save joints as a point cloud PLY (pred 0–20, GT 21–41).
            # write_line_set to PLY uses a non-standard 'edge' element that
            # many readers silently ignore — so we only save points here and
            # reconstruct the LineSet in-memory inside view_3d.py.
            all_pts    = np.vstack([p3, g3_off])
            all_colors = np.vstack([JOINT_COLORS_NP,
                                    np.tile(GT_COLOR, (21, 1))])
            comb_pcd = o3d.geometry.PointCloud()
            comb_pcd.points = o3d.utility.Vector3dVector(all_pts)
            comb_pcd.colors = o3d.utility.Vector3dVector(all_colors)

            pts_path = str(out_dir / f'3d_sample_{idx:04d}.ply')
            o3d.io.write_point_cloud(pts_path, comb_pcd)
            ply_pairs.append(pts_path)

        ply_name = f'3d_sample_{idx:04d}.ply'
        print(f'  [{local_i+1:>2}/{n}]  idx={idx:>5}  '
              f'2D={e2d:6.2f}px  3D MPJPE={e3d:.4f}  '
              f'→ {png_path.name}'
              + (f'  + {ply_name}' if has_o3d else ''))

    # ── Summary ───────────────────────────────────────────────────────────
    print(f'\nMean 2D error  : {np.mean(errs_2d_all):.3f} px')
    print(f'Mean 3D MPJPE  : {np.mean(errs_3d_all):.4f}  (norm units, C=1 bone)')
    print(f'\n2D PNGs saved to  : {out_dir}/')
    if has_o3d and ply_pairs:
        print(f'PLY files saved to: {out_dir}/')

    if has_o3d and ply_pairs:
        print(f'\nTo view interactively:')
        print(f'  python view_3d.py --ply-dir {out_dir} --sample {args.start_idx}')


if __name__ == '__main__':
    main()
