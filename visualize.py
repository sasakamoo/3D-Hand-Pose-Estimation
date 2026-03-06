"""
visualize.py — Visualize model predictions
==========================================
Shows:
  1. 2D skeleton overlaid on image (predicted vs GT)
  2. 3D skeleton in normalised space (predicted vs GT)
  3. Error table per joint
"""

import argparse
from pathlib import Path
import numpy as np
import cv2
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model   import SingleViewModel, reconstruct_3d_from_25d
from dataset import FreiHANDDataset, IMG_SIZE

CONNECTIONS = [
    [0,1],[1,2],[2,3],[3,4],
    [0,5],[5,6],[6,7],[7,8],
    [0,9],[9,10],[10,11],[11,12],
    [0,13],[13,14],[14,15],[15,16],
    [0,17],[17,18],[18,19],[19,20],
]


def draw_skeleton(img, kpts_px, color_bone, color_joint, W, H):
    vis = img.copy()
    for s, e in CONNECTIONS:
        p1 = tuple(np.clip(kpts_px[s].astype(int), [0,0], [W-1,H-1]))
        p2 = tuple(np.clip(kpts_px[e].astype(int), [0,0], [W-1,H-1]))
        cv2.line(vis, p1, p2, color_bone, 2)
    for pt in kpts_px:
        c = tuple(np.clip(pt.astype(int), [0,0], [W-1,H-1]))
        cv2.circle(vis, c, 4, color_joint, -1)
    return vis


def visualize(model_path, data_root, sample_idx, output_dir):
    output_dir = Path(output_dir); output_dir.mkdir(exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model = SingleViewModel(num_kpts=21)
    ckpt  = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    model = model.to(device).eval()
    print(f'✓ Loaded model from {model_path}')

    # Load sample
    ds = FreiHANDDataset(data_root, split='val', augment=False)
    batch = ds[sample_idx % len(ds)]

    img_t   = batch['image'].unsqueeze(0).to(device)
    gt_2d   = batch['pose_2d_gt'].numpy()       # (K, 2) pixels in 128x128
    gt_z    = batch['depth_rel_gt'].numpy()     # (K,)
    K_mat   = batch['K_mat'].numpy()            # (3, 3)

    with torch.no_grad():
        pred_2d, pred_z, _, hm_probs = model(img_t)

    # pred_2d is already in pixel space (0..127)
    pred_2d_px = pred_2d[0].cpu().numpy()   # (K, 2)
    pred_z_np  = pred_z[0].cpu().numpy()    # (K,)

    # ── Reconstruct 3D ────────────────────────────────────────────────────
    K_t = batch['K_mat'].unsqueeze(0)
    pose_3d_pred = reconstruct_3d_from_25d(
        pred_2d, pred_z, K_t.to(device), img_size=IMG_SIZE).cpu().numpy()[0]

    # GT 3D (back-project)
    gt_z_root = gt_z[0]
    gt_Z      = gt_z + gt_z_root
    fx, fy    = K_mat[0,0], K_mat[1,1]
    cx, cy    = K_mat[0,2], K_mat[1,2]
    gt_X      = (gt_2d[:,0] - cx) * gt_Z / fx
    gt_Y      = (gt_2d[:,1] - cy) * gt_Z / fy
    pose_3d_gt = np.stack([gt_X, gt_Y, gt_Z], axis=1)

    # ── 2D visualisation ─────────────────────────────────────────────────
    img_np = (batch['image'].permute(1,2,0).numpy() * 255).astype(np.uint8)
    H, W   = img_np.shape[:2]

    vis_pred = draw_skeleton(img_np, pred_2d_px,    (0,255,0),   (0,0,255), W, H)
    vis_gt   = draw_skeleton(img_np, gt_2d,         (255,165,0), (255,0,0), W, H)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(vis_pred); axes[0].set_title('Predicted  (green)'); axes[0].axis('off')
    axes[1].imshow(vis_gt);   axes[1].set_title('GT  (orange)');        axes[1].axis('off')
    plt.suptitle(f'Sample {sample_idx} — 2D Projection', fontsize=13)
    plt.tight_layout()
    out = output_dir / '2d_vis.png'
    plt.savefig(str(out), dpi=150, bbox_inches='tight'); plt.close()
    print(f'✓ 2D vis → {out}')

    # ── 3D visualisation (root-relative) ─────────────────────────────────
    p3_pred = pose_3d_pred - pose_3d_pred[0]
    p3_gt   = pose_3d_gt   - pose_3d_gt[0]

    fig = plt.figure(figsize=(14, 6))
    for col, (pts, title, color) in enumerate([
        (p3_pred, 'Predicted', 'red'),
        (p3_gt,   'GT',        'lime'),
    ]):
        ax = fig.add_subplot(1, 2, col+1, projection='3d')
        ax.scatter(pts[:,0], pts[:,1], pts[:,2], c=color, s=60)
        for s, e in CONNECTIONS:
            ax.plot(pts[[s,e],0], pts[[s,e],1], pts[[s,e],2],
                    color='blue' if color=='red' else 'darkorange', lw=1.5)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    plt.suptitle('3D Pose (root-relative, scale-normalised)', fontsize=13)
    plt.tight_layout()
    out = output_dir / '3d_vis.png'
    plt.savefig(str(out), dpi=150, bbox_inches='tight'); plt.close()
    print(f'✓ 3D vis → {out}')

    # ── Error table ───────────────────────────────────────────────────────
    print(f'\n{"Jnt":>4} {"Pred_x":>8} {"Pred_y":>8} {"GT_x":>8} {"GT_y":>8} '
          f'{"2D_err":>8} {"3D_err":>8}')
    print('-' * 60)
    errs_2d, errs_3d = [], []
    for k in range(21):
        e2 = np.linalg.norm(pred_2d_px[k] - gt_2d[k])
        e3 = np.linalg.norm(p3_pred[k] - p3_gt[k])
        errs_2d.append(e2); errs_3d.append(e3)
        print(f'{k:>4} {pred_2d_px[k,0]:>8.2f} {pred_2d_px[k,1]:>8.2f} '
              f'{gt_2d[k,0]:>8.2f} {gt_2d[k,1]:>8.2f} '
              f'{e2:>8.3f} {e3:>8.4f}')
    print(f'\nMean 2D error : {np.mean(errs_2d):.3f} px')
    print(f'Mean 3D error : {np.mean(errs_3d):.4f} (norm units, C=1 bone)')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',      type=str, required=True)
    parser.add_argument('--data-root',  type=str, default='/home/kia/Dataset')
    parser.add_argument('--sample-idx', type=int, default=0)
    parser.add_argument('--output-dir', type=str, default='visualizations')
    args = parser.parse_args()
    visualize(args.model, args.data_root, args.sample_idx, args.output_dir)


if __name__ == '__main__':
    main()