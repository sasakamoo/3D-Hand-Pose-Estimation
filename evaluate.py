"""
evaluate.py — Full quantitative evaluation of a trained model
==============================================================
Metrics (standard in 3D hand pose literature / FreiHAND leaderboard):

  2D:
    • MEPE-2D    Mean End-Point Error in pixels (per-joint, root-relative)
    • PCK-2D     Percentage of Correct Keypoints at 2/5/10/15 px thresholds
    • AUC-2D     Area under the PCK-2D curve  (0–30 px, 50 steps)

  3D (root-relative, scale-normalised with C=1 reference bone):
    • MPJPE      Mean Per-Joint Position Error   (no alignment)
    • PA-MPJPE   MPJPE after Procrustes alignment (optimal rot + scale + trans)
    • PCK-3D     Percentage of Correct Keypoints at 0.05/0.10/0.20 thresholds
    • AUC-3D     Area under the PCK-3D curve  (0–0.5, 50 steps)
    • Bone-Err   Mean absolute bone-length error (normalised units)

  Outputs saved to --output-dir:
    eval_summary.txt       — all scalar metrics
    per_joint_error.png    — bar chart of per-joint 3D error
    pck_curves.png         — PCK-2D and PCK-3D curves with AUC annotations

Usage:
    python3 evaluate.py --model best_model.pt --data-root /path/to/FreiHAND
    python3 evaluate.py --model best_model.pt --data-root /path/to/FreiHAND --split train
    python3 evaluate.py --model best_model.pt --data-root /path/to/FreiHAND --n-samples 1000
"""

import argparse
import os
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from model     import SingleViewModel, reconstruct_3d_from_25d
from model_sdf import SDFHandPoseNet
from dataset   import FreiHANDDataset, IMG_SIZE

# ─────────────────────────────────────────────────────────────────────────────
# Skeleton / joint metadata
# ─────────────────────────────────────────────────────────────────────────────

CONNECTIONS = [
    [0,1],[1,2],[2,3],[3,4],
    [0,5],[5,6],[6,7],[7,8],
    [0,9],[9,10],[10,11],[11,12],
    [0,13],[13,14],[14,15],[15,16],
    [0,17],[17,18],[18,19],[19,20],
]

JOINT_NAMES = [
    'Wrist',
    'Idx-MCP','Idx-PIP','Idx-DIP','Idx-Tip',
    'Mid-MCP','Mid-PIP','Mid-DIP','Mid-Tip',
    'Rng-MCP','Rng-PIP','Rng-DIP','Rng-Tip',
    'Pnk-MCP','Pnk-PIP','Pnk-DIP','Pnk-Tip',
    'Thm-CMC','Thm-MCP','Thm-IP', 'Thm-Tip',
]

# ─────────────────────────────────────────────────────────────────────────────
# Procrustes alignment  (used for PA-MPJPE)
# ─────────────────────────────────────────────────────────────────────────────

def procrustes_align(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """
    Align `pred` (K, 3) to `gt` (K, 3) via similarity Procrustes
    (optimal rotation + isotropic scale + translation).
    Returns the aligned prediction, same shape as `pred`.

    Algorithm (Umeyama 1991):
      1. Centre both clouds.
      2. SVD of  gt_c.T @ pred_c  →  U, S, V
      3. Handle reflection: if det(U @ V.T) < 0, flip last column.
      4. Rotation R = U @ V.T
      5. Scale    s = trace(S) / ||pred_c||_F²
      6. Translation t = mean_gt − s * mean_pred @ R.T
    """
    mu_p = pred.mean(0)
    mu_g = gt.mean(0)
    pred_c = pred - mu_p
    gt_c   = gt   - mu_g

    var_p = (pred_c ** 2).sum()
    if var_p < 1e-10:
        return pred.copy()

    M = gt_c.T @ pred_c                            # (3, 3)
    U, S, Vt = np.linalg.svd(M)                    # U (3,3), S (3,), Vt (3,3)
    V = Vt.T

    # Fix reflection
    d = np.linalg.det(U @ V.T)
    diag = np.ones(3); diag[-1] = np.sign(d)
    R = U @ (np.diag(diag) @ V.T)                  # (3, 3)

    s = (S * diag).sum() / var_p
    t = mu_g - s * (R @ mu_p)

    return s * (pred @ R.T) + t


# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────────────────────

def pck_curve(errors: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """
    errors     : (N, K)  per-joint errors across N samples
    thresholds : (T,)    threshold values to sweep
    returns    : (T,)    fraction of joints within each threshold
    """
    pck = np.array([(errors <= t).mean() for t in thresholds])
    return pck


def auc_from_pck(thresholds: np.ndarray, pck: np.ndarray) -> float:
    """Normalised AUC ∈ [0, 1] via trapezoidal integration."""
    return float(np.trapz(pck, thresholds) / (thresholds[-1] - thresholds[0]))


def bone_length_error(pred_3d: np.ndarray, gt_3d: np.ndarray) -> float:
    """
    Mean absolute error of bone lengths.
    pred_3d / gt_3d : (N, K, 3)
    """
    errors = []
    for s, e in CONNECTIONS:
        pred_len = np.linalg.norm(pred_3d[:, s] - pred_3d[:, e], axis=-1)  # (N,)
        gt_len   = np.linalg.norm(gt_3d[:, s]   - gt_3d[:, e],   axis=-1)
        errors.append(np.abs(pred_len - gt_len))
    return float(np.mean(errors))


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation loop
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_eval(model, loader, device):
    """
    Returns:
        pred_2d_all : (N, 21, 2)
        gt_2d_all   : (N, 21, 2)
        pred_3d_all : (N, 21, 3)  root-relative
        gt_3d_all   : (N, 21, 3)  root-relative
    """
    model.eval()
    pred_2d_list, gt_2d_list = [], []
    pred_3d_list, gt_3d_list = [], []

    for batch in tqdm(loader, desc='Evaluating', ncols=90, leave=True):
        imgs  = batch['image'].to(device, non_blocking=True)
        gt_2d = batch['pose_2d_gt'].to(device, non_blocking=True)
        gt_z  = batch['depth_rel_gt'].to(device, non_blocking=True)
        K_mat = batch['K_mat'].to(device, non_blocking=True)

        pred_2d, pred_z, _, _ = model(imgs)

        pred_3d = reconstruct_3d_from_25d(pred_2d, pred_z, K_mat, img_size=IMG_SIZE)
        gt_3d   = reconstruct_3d_from_25d(gt_2d,   gt_z,   K_mat, img_size=IMG_SIZE)

        # Root-relative (joint 0 = wrist)
        pred_3d = pred_3d - pred_3d[:, 0:1]
        gt_3d   = gt_3d   - gt_3d[:, 0:1]

        pred_2d_list.append(pred_2d.cpu().numpy())
        gt_2d_list.append(gt_2d.cpu().numpy())
        pred_3d_list.append(pred_3d.cpu().numpy())
        gt_3d_list.append(gt_3d.cpu().numpy())

    return (np.concatenate(pred_2d_list),
            np.concatenate(gt_2d_list),
            np.concatenate(pred_3d_list),
            np.concatenate(gt_3d_list))


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_per_joint_error(per_joint_3d: np.ndarray, out_path: str):
    """Bar chart of mean 3D error per joint."""
    fig, ax = plt.subplots(figsize=(14, 4))
    colors = (
        ['#888888'] +                   # wrist
        ['#00cc00'] * 4 +               # index
        ['#0080ff'] * 4 +               # middle
        ['#ffcc00'] * 4 +               # ring
        ['#ff6600'] * 4 +               # pinky
        ['#cc00cc'] * 4                 # thumb
    )
    bars = ax.bar(range(21), per_joint_3d, color=colors, edgecolor='black', linewidth=0.4)
    ax.set_xticks(range(21))
    ax.set_xticklabels(JOINT_NAMES, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('Mean 3D error (norm units)')
    ax.set_title('Per-joint 3D error  (lower is better)')
    ax.axhline(per_joint_3d.mean(), color='red', linestyle='--',
               linewidth=1, label=f'mean={per_joint_3d.mean():.4f}')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_pck_curves(thr_2d, pck_2d, auc_2d,
                    thr_3d, pck_3d, auc_3d,
                    thr_3d_pa, pck_3d_pa, auc_3d_pa,
                    out_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ── 2D PCK ──────────────────────────────────────────────────────────
    axes[0].plot(thr_2d, pck_2d * 100, lw=2, color='steelblue')
    axes[0].fill_between(thr_2d, pck_2d * 100, alpha=0.15, color='steelblue')
    axes[0].set_xlabel('Threshold (px)')
    axes[0].set_ylabel('PCK (%)')
    axes[0].set_title(f'PCK-2D  (AUC={auc_2d:.3f})')
    axes[0].set_xlim(thr_2d[0], thr_2d[-1])
    axes[0].set_ylim(0, 100)
    axes[0].grid(alpha=0.3)
    # Mark common thresholds
    for t, c in [(5, 'green'), (10, 'orange'), (15, 'red')]:
        idx = np.searchsorted(thr_2d, t)
        axes[0].axvline(t, color=c, linestyle=':', linewidth=1)
        axes[0].text(t + 0.3, 5, f'{t}px\n{pck_2d[idx]*100:.1f}%',
                     color=c, fontsize=7)

    # ── 3D PCK ──────────────────────────────────────────────────────────
    axes[1].plot(thr_3d, pck_3d * 100,    lw=2, color='steelblue',
                 label=f'MPJPE-aligned   AUC={auc_3d:.3f}')
    axes[1].plot(thr_3d_pa, pck_3d_pa * 100, lw=2, color='darkorange',
                 linestyle='--', label=f'PA-MPJPE-aligned  AUC={auc_3d_pa:.3f}')
    axes[1].fill_between(thr_3d, pck_3d * 100, alpha=0.12, color='steelblue')
    axes[1].fill_between(thr_3d_pa, pck_3d_pa * 100, alpha=0.12, color='darkorange')
    axes[1].set_xlabel('Threshold (norm units)')
    axes[1].set_ylabel('PCK (%)')
    axes[1].set_title('PCK-3D  (norm units, C=1 ref bone)')
    axes[1].set_xlim(thr_3d[0], thr_3d[-1])
    axes[1].set_ylim(0, 100)
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',       type=str, required=True)
    parser.add_argument('--model-type',  type=str, default='heatmap',
                        choices=['heatmap', 'sdf'],
                        help='heatmap = SingleViewModel (model.py), sdf = SDFHandPoseNet (model_sdf.py)')
    parser.add_argument('--data-root',   type=str, default='/home/kghasemz/scratch/datasets/FreiHand')
    parser.add_argument('--split',       type=str, default='val',
                        choices=['train', 'val'])
    parser.add_argument('--n-samples',   type=int, default=None,
                        help='Evaluate on first N samples (default: full split)')
    parser.add_argument('--batch-size',  type=int, default=16)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--output-dir',  type=str, default='eval_results')
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── Load model ────────────────────────────────────────────────────────
    if args.model_type == 'sdf':
        model = SDFHandPoseNet(num_kpts=21, pretrained_backbone=False)
    else:
        model = SingleViewModel(num_kpts=21)
    ckpt  = torch.load(args.model, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    model = model.to(device).eval()

    epoch_info = ''
    if 'epoch' in ckpt:
        epoch_info = f'epoch {ckpt["epoch"]}'
    print(f'\nModel  : {args.model}  ({epoch_info})')
    print(f'Split  : {args.split}')
    print(f'Device : {device}\n')

    # ── Dataloader ────────────────────────────────────────────────────────
    ds = FreiHANDDataset(args.data_root, split=args.split, augment=False)
    if args.n_samples is not None:
        from torch.utils.data import Subset
        ds = Subset(ds, range(min(args.n_samples, len(ds))))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
    print(f'Evaluating {len(ds)} samples  ({len(loader)} batches)...\n')

    # ── Collect predictions ───────────────────────────────────────────────
    pred_2d, gt_2d, pred_3d, gt_3d = run_eval(model, loader, device)
    # pred_2d / gt_2d : (N, 21, 2)   pixels
    # pred_3d / gt_3d : (N, 21, 3)   root-relative, norm units

    N = pred_2d.shape[0]

    # ── 2D metrics ────────────────────────────────────────────────────────
    err_2d = np.linalg.norm(pred_2d - gt_2d, axis=-1)    # (N, 21)
    mepe_2d = err_2d.mean()

    thr_2d  = np.linspace(0, 30, 50)
    pck_2d  = pck_curve(err_2d, thr_2d)
    auc_2d  = auc_from_pck(thr_2d, pck_2d)

    pck_2d_at = {}
    for t in [2, 5, 10, 15]:
        idx = np.searchsorted(thr_2d, t)
        pck_2d_at[t] = float(pck_2d[min(idx, len(pck_2d)-1)] * 100)

    per_joint_2d = err_2d.mean(axis=0)   # (21,)

    # ── 3D metrics — MPJPE (no alignment) ─────────────────────────────────
    err_3d = np.linalg.norm(pred_3d - gt_3d, axis=-1)    # (N, 21)
    mpjpe  = err_3d.mean()

    thr_3d = np.linspace(0, 0.5, 50)
    pck_3d = pck_curve(err_3d, thr_3d)
    auc_3d = auc_from_pck(thr_3d, pck_3d)

    pck_3d_at = {}
    for t in [0.05, 0.10, 0.20]:
        idx = np.searchsorted(thr_3d, t)
        pck_3d_at[t] = float(pck_3d[min(idx, len(pck_3d)-1)] * 100)

    per_joint_3d = err_3d.mean(axis=0)   # (21,)

    # ── 3D metrics — PA-MPJPE (Procrustes aligned) ────────────────────────
    err_3d_pa = np.zeros((N, 21), dtype=np.float32)
    for i in range(N):
        pred_aligned   = procrustes_align(pred_3d[i], gt_3d[i])
        err_3d_pa[i]   = np.linalg.norm(pred_aligned - gt_3d[i], axis=-1)

    pa_mpjpe = err_3d_pa.mean()

    thr_3d_pa = np.linspace(0, 0.5, 50)
    pck_3d_pa = pck_curve(err_3d_pa, thr_3d_pa)
    auc_3d_pa = auc_from_pck(thr_3d_pa, pck_3d_pa)

    # ── Bone-length error ──────────────────────────────────────────────────
    bone_err = bone_length_error(pred_3d, gt_3d)

    # ── Print results ─────────────────────────────────────────────────────
    sep = '─' * 55
    print(f'\n{sep}')
    print(f'  EVALUATION RESULTS   ({N} samples, {args.split} split)')
    print(sep)

    print(f'\n  2D metrics  (pixel space, 128×128 image)')
    print(f'    MEPE-2D          : {mepe_2d:.3f} px')
    print(f'    PCK-2D @  2 px   : {pck_2d_at[2]:.2f} %')
    print(f'    PCK-2D @  5 px   : {pck_2d_at[5]:.2f} %')
    print(f'    PCK-2D @ 10 px   : {pck_2d_at[10]:.2f} %')
    print(f'    PCK-2D @ 15 px   : {pck_2d_at[15]:.2f} %')
    print(f'    AUC-2D (0–30px)  : {auc_2d:.4f}')

    print(f'\n  3D metrics  (root-relative, normalised units, C=1 ref bone)')
    print(f'    MPJPE            : {mpjpe:.4f}')
    print(f'    PA-MPJPE         : {pa_mpjpe:.4f}')
    print(f'    PCK-3D @ 0.05    : {pck_3d_at[0.05]:.2f} %')
    print(f'    PCK-3D @ 0.10    : {pck_3d_at[0.10]:.2f} %')
    print(f'    PCK-3D @ 0.20    : {pck_3d_at[0.20]:.2f} %')
    print(f'    AUC-3D (0–0.5)   : {auc_3d:.4f}')
    print(f'    AUC-3D PA(0–0.5) : {auc_3d_pa:.4f}')
    print(f'    Bone-length err  : {bone_err:.4f}')
    print(f'\n{sep}\n')

    # ── Save text summary ─────────────────────────────────────────────────
    summary_path = out_dir / 'eval_summary.txt'
    with open(str(summary_path), 'w') as f:
        f.write(f'Model      : {args.model}\n')
        if epoch_info:
            f.write(f'Epoch      : {epoch_info}\n')
        f.write(f'Split      : {args.split}\n')
        f.write(f'N samples  : {N}\n\n')
        f.write(f'--- 2D ---\n')
        f.write(f'MEPE-2D         : {mepe_2d:.3f} px\n')
        f.write(f'PCK-2D @  2px   : {pck_2d_at[2]:.2f} %\n')
        f.write(f'PCK-2D @  5px   : {pck_2d_at[5]:.2f} %\n')
        f.write(f'PCK-2D @ 10px   : {pck_2d_at[10]:.2f} %\n')
        f.write(f'PCK-2D @ 15px   : {pck_2d_at[15]:.2f} %\n')
        f.write(f'AUC-2D (0-30px) : {auc_2d:.4f}\n\n')
        f.write(f'--- 3D ---\n')
        f.write(f'MPJPE           : {mpjpe:.4f}\n')
        f.write(f'PA-MPJPE        : {pa_mpjpe:.4f}\n')
        f.write(f'PCK-3D @ 0.05   : {pck_3d_at[0.05]:.2f} %\n')
        f.write(f'PCK-3D @ 0.10   : {pck_3d_at[0.10]:.2f} %\n')
        f.write(f'PCK-3D @ 0.20   : {pck_3d_at[0.20]:.2f} %\n')
        f.write(f'AUC-3D (0-0.5)  : {auc_3d:.4f}\n')
        f.write(f'AUC-3D PA(0-0.5): {auc_3d_pa:.4f}\n')
        f.write(f'Bone-length err : {bone_err:.4f}\n\n')
        f.write(f'--- Per-joint 3D error ---\n')
        for k, (name, e) in enumerate(zip(JOINT_NAMES, per_joint_3d)):
            f.write(f'  {k:>2} {name:<12} : {e:.4f}\n')
    print(f'Summary saved  → {summary_path}')

    # ── Plots ─────────────────────────────────────────────────────────────
    pj_path  = str(out_dir / 'per_joint_error.png')
    pck_path = str(out_dir / 'pck_curves.png')

    plot_per_joint_error(per_joint_3d, pj_path)
    print(f'Per-joint plot → {pj_path}')

    plot_pck_curves(thr_2d,    pck_2d,    auc_2d,
                    thr_3d,    pck_3d,    auc_3d,
                    thr_3d_pa, pck_3d_pa, auc_3d_pa,
                    pck_path)
    print(f'PCK curves     → {pck_path}')


if __name__ == '__main__':
    main()
