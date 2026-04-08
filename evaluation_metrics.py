"""
evaluation_metrics.py — Full FreiHAND evaluation with CSV output
=================================================================
Runs a trained SDFHandPoseNet model over the entire FreiHAND validation
set and writes all performance metrics to a CSV file.

Primary metric: MPJPE (Mean Per-Joint Position Error, normalised units)

Additional metrics per row:
  - PA-MPJPE   (Procrustes-aligned MPJPE)
  - MEPE-2D    (Mean 2D pixel error)
  - PCK-2D     at 2, 5, 10, 15 px
  - PCK-3D     at 0.05, 0.10, 0.20
  - AUC-2D     (0-30 px)
  - AUC-3D     (0-0.5)
  - AUC-3D-PA  (0-0.5, after Procrustes)
  - Bone-Err   (mean absolute bone-length error)
  - Per-joint 3D error for all 21 joints

Usage:
    python evaluation_metrics.py --model sdf_best_model.pt --data-root /path/to/FreiHand
    python evaluation_metrics.py --model sdf_best_model.pt --data-root /path/to/FreiHand --split train
    python evaluation_metrics.py --model sdf_best_model.pt --data-root /path/to/FreiHand --output results.csv
"""

import argparse
import csv
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model_sdf import SDFHandPoseNet
from model     import reconstruct_3d_from_25d
from dataset   import FreiHANDDataset, IMG_SIZE


# ── Skeleton metadata ─────────────────────────────────────────────────────────

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


# ── Metric helpers ────────────────────────────────────────────────────────────

def procrustes_align(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Similarity Procrustes alignment of pred (K,3) to gt (K,3)."""
    mu_p, mu_g = pred.mean(0), gt.mean(0)
    pred_c, gt_c = pred - mu_p, gt - mu_g
    var_p = (pred_c ** 2).sum()
    if var_p < 1e-10:
        return pred.copy()
    M = gt_c.T @ pred_c
    U, S, Vt = np.linalg.svd(M)
    V = Vt.T
    d = np.linalg.det(U @ V.T)
    diag = np.ones(3); diag[-1] = np.sign(d)
    R = U @ (np.diag(diag) @ V.T)
    s = (S * diag).sum() / var_p
    t = mu_g - s * (R @ mu_p)
    return s * (pred @ R.T) + t


def pck(errors: np.ndarray, threshold: float) -> float:
    """Fraction of joints with error <= threshold."""
    return float((errors <= threshold).mean())


def auc(errors: np.ndarray, max_threshold: float, steps: int = 50) -> float:
    """Normalised AUC under the PCK curve from 0 to max_threshold."""
    thresholds = np.linspace(0, max_threshold, steps)
    pck_vals   = np.array([(errors <= t).mean() for t in thresholds])
    return float(np.trapz(pck_vals, thresholds) / max_threshold)


def bone_length_error(pred_3d: np.ndarray, gt_3d: np.ndarray) -> float:
    """Mean absolute bone-length error across all bones and samples."""
    errors = []
    for s, e in CONNECTIONS:
        pred_len = np.linalg.norm(pred_3d[:, s] - pred_3d[:, e], axis=-1)
        gt_len   = np.linalg.norm(gt_3d[:,  s]  - gt_3d[:,  e],  axis=-1)
        errors.append(np.abs(pred_len - gt_len))
    return float(np.mean(errors))


# ── Inference loop ────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(model, loader, device):
    """
    Returns:
        pred_2d : (N, 21, 2)  predicted 2D pixel coords
        gt_2d   : (N, 21, 2)  GT 2D pixel coords
        pred_3d : (N, 21, 3)  predicted 3D, root-relative
        gt_3d   : (N, 21, 3)  GT 3D, root-relative
    """
    model.eval()
    pred_2d_list, gt_2d_list = [], []
    pred_3d_list, gt_3d_list = [], []

    for batch in tqdm(loader, desc='Running inference', ncols=90):
        imgs  = batch['image'].to(device, non_blocking=True)
        gt_2d = batch['pose_2d_gt'].to(device, non_blocking=True)
        gt_z  = batch['depth_rel_gt'].to(device, non_blocking=True)
        K_mat = batch['K_mat'].to(device, non_blocking=True)

        pred_2d, pred_z, _, _ = model(imgs)

        pred_3d = reconstruct_3d_from_25d(pred_2d, pred_z, K_mat, img_size=IMG_SIZE)
        gt_3d   = reconstruct_3d_from_25d(gt_2d,   gt_z,   K_mat, img_size=IMG_SIZE)

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


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',       type=str, required=True,
                        help='Path to sdf_best_model.pt checkpoint')
    parser.add_argument('--data-root',   type=str, default='/home/kghasemz/scratch/datasets/FreiHand')
    parser.add_argument('--split',       type=str, default='val',
                        choices=['train', 'val'])
    parser.add_argument('--batch-size',  type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--output',      type=str, default='evaluation_metrics.csv',
                        help='Output CSV file path')
    parser.add_argument('--device',      type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    device = torch.device(args.device)

    # ── Load model ────────────────────────────────────────────────────────
    model = SDFHandPoseNet(num_kpts=21, pretrained_backbone=False)
    ckpt  = torch.load(args.model, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    model = model.to(device).eval()

    epoch_str = f'epoch {ckpt["epoch"]}' if 'epoch' in ckpt else 'unknown epoch'
    print(f'\nModel  : {args.model}  ({epoch_str})')
    print(f'Split  : {args.split}')
    print(f'Device : {device}')

    # ── Dataset ───────────────────────────────────────────────────────────
    ds     = FreiHANDDataset(args.data_root, split=args.split, augment=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True,
                        persistent_workers=(args.num_workers > 0))
    print(f'Samples: {len(ds)}  ({len(loader)} batches)\n')

    # ── Inference ─────────────────────────────────────────────────────────
    pred_2d, gt_2d, pred_3d, gt_3d = run_inference(model, loader, device)
    N = pred_2d.shape[0]

    # ── 2D metrics ────────────────────────────────────────────────────────
    err_2d  = np.linalg.norm(pred_2d - gt_2d, axis=-1)   # (N, 21)
    mepe_2d = float(err_2d.mean())
    auc_2d  = auc(err_2d, max_threshold=30.0)
    pck_2d  = {t: pck(err_2d, t) * 100 for t in [2, 5, 10, 15]}

    # ── 3D metrics ────────────────────────────────────────────────────────
    err_3d  = np.linalg.norm(pred_3d - gt_3d, axis=-1)   # (N, 21)
    mpjpe   = float(err_3d.mean())
    auc_3d  = auc(err_3d, max_threshold=0.5)
    pck_3d  = {t: pck(err_3d, t) * 100 for t in [0.05, 0.10, 0.20]}

    # ── PA-MPJPE ──────────────────────────────────────────────────────────
    err_3d_pa = np.zeros((N, 21), dtype=np.float32)
    print('Computing PA-MPJPE...')
    for i in tqdm(range(N), ncols=90, desc='Procrustes'):
        aligned       = procrustes_align(pred_3d[i], gt_3d[i])
        err_3d_pa[i]  = np.linalg.norm(aligned - gt_3d[i], axis=-1)
    pa_mpjpe = float(err_3d_pa.mean())
    auc_3d_pa = auc(err_3d_pa, max_threshold=0.5)

    # ── Per-joint 3D error ────────────────────────────────────────────────
    per_joint = err_3d.mean(axis=0)   # (21,)

    # ── Bone-length error ─────────────────────────────────────────────────
    bone_err = bone_length_error(pred_3d, gt_3d)

    # ── Print summary ─────────────────────────────────────────────────────
    print(f'\n{"─"*55}')
    print(f'  MPJPE          : {mpjpe:.4f}')
    print(f'  PA-MPJPE       : {pa_mpjpe:.4f}')
    print(f'  MEPE-2D        : {mepe_2d:.3f} px')
    print(f'  AUC-2D (0-30px): {auc_2d:.4f}')
    print(f'  AUC-3D (0-0.5) : {auc_3d:.4f}')
    print(f'  AUC-3D PA      : {auc_3d_pa:.4f}')
    print(f'  Bone-length err: {bone_err:.4f}')
    print(f'{"─"*55}\n')

    # ── Build CSV rows ────────────────────────────────────────────────────
    # Row 1: aggregate metrics
    # Row 2: per-joint 3D errors
    summary_row = {
        'metric':           'aggregate',
        'model':            args.model,
        'epoch':            ckpt.get('epoch', ''),
        'split':            args.split,
        'N_samples':        N,
        'MPJPE':            round(mpjpe, 6),
        'PA_MPJPE':         round(pa_mpjpe, 6),
        'MEPE_2D_px':       round(mepe_2d, 6),
        'PCK_2D_2px':       round(pck_2d[2],  4),
        'PCK_2D_5px':       round(pck_2d[5],  4),
        'PCK_2D_10px':      round(pck_2d[10], 4),
        'PCK_2D_15px':      round(pck_2d[15], 4),
        'AUC_2D_0_30px':    round(auc_2d,   6),
        'PCK_3D_0.05':      round(pck_3d[0.05], 4),
        'PCK_3D_0.10':      round(pck_3d[0.10], 4),
        'PCK_3D_0.20':      round(pck_3d[0.20], 4),
        'AUC_3D_0_0.5':     round(auc_3d,    6),
        'AUC_3D_PA_0_0.5':  round(auc_3d_pa, 6),
        'Bone_length_err':  round(bone_err, 6),
    }
    # Add per-joint columns to the aggregate row
    for name, err in zip(JOINT_NAMES, per_joint):
        summary_row[f'joint_{name}'] = round(float(err), 6)

    # Per-joint breakdown rows (one row per joint for easy charting)
    joint_rows = []
    for k, (name, err) in enumerate(zip(JOINT_NAMES, per_joint)):
        joint_rows.append({
            'metric':    'per_joint',
            'joint_idx': k,
            'joint_name': name,
            'MPJPE_3D':  round(float(err), 6),
        })

    # ── Write CSV ─────────────────────────────────────────────────────────
    out_path = args.output
    with open(out_path, 'w', newline='') as f:
        # --- Sheet 1: aggregate row ---
        agg_fieldnames = list(summary_row.keys())
        writer = csv.DictWriter(f, fieldnames=agg_fieldnames)
        writer.writeheader()
        writer.writerow(summary_row)

        # blank separator
        f.write('\n')

        # --- Sheet 2: per-joint rows ---
        joint_fieldnames = ['metric', 'joint_idx', 'joint_name', 'MPJPE_3D']
        writer2 = csv.DictWriter(f, fieldnames=joint_fieldnames)
        writer2.writeheader()
        writer2.writerows(joint_rows)

    print(f'CSV saved -> {os.path.abspath(out_path)}')


if __name__ == '__main__':
    main()
