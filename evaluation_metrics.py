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
from torch.amp.autocast_mode import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from model_sdf import SDFHandPoseNet, N_PTS
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
        pred_2d       : (N, 21, 2)  predicted 2D pixel coords
        gt_2d         : (N, 21, 2)  GT 2D pixel coords
        pred_3d       : (N, 21, 3)  predicted 3D, root-relative (normalised units)
        gt_3d         : (N, 21, 3)  GT 3D, root-relative (normalised units)
        scale_factors : (N,)        per-sample bone scale s (metres); multiply
                                    normalised units by s*1000 to get mm
    """
    model.eval()
    pred_2d_list, gt_2d_list = [], []
    pred_3d_list, gt_3d_list = [], []
    scale_list = []

    for batch in tqdm(loader, desc='Running inference', ncols=90):
        imgs  = batch['image'].to(device, non_blocking=True)
        gt_2d = batch['pose_2d_gt'].to(device, non_blocking=True)
        gt_z  = batch['depth_rel_gt'].to(device, non_blocking=True)
        K_mat = batch['K_mat'].to(device, non_blocking=True)

        # Sample N_PTS random points explicitly to avoid the dense 9261-pt
        # eval grid in _sample_points, which makes Transformer O(N²) ~21× slower.
        ext_pts = torch.empty(imgs.shape[0], N_PTS, 3, device=device).uniform_(-1.0, 1.0)
        with autocast('cuda', enabled=(device.type == 'cuda')):
            pred_2d, pred_z, _, _ = model(imgs, ext_pts=ext_pts)

        pred_3d = reconstruct_3d_from_25d(pred_2d, pred_z, K_mat, img_size=IMG_SIZE)
        gt_3d   = reconstruct_3d_from_25d(gt_2d,   gt_z,   K_mat, img_size=IMG_SIZE)

        pred_3d = pred_3d - pred_3d[:, 0:1]
        gt_3d   = gt_3d   - gt_3d[:, 0:1]

        pred_2d_list.append(pred_2d.cpu().numpy())
        gt_2d_list.append(gt_2d.cpu().numpy())
        pred_3d_list.append(pred_3d.cpu().numpy())
        gt_3d_list.append(gt_3d.cpu().numpy())
        scale_list.append(batch['scale_factor'].numpy())   # (B,) metres

    return (np.concatenate(pred_2d_list),
            np.concatenate(gt_2d_list),
            np.concatenate(pred_3d_list),
            np.concatenate(gt_3d_list),
            np.concatenate(scale_list))


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
    pred_2d, gt_2d, pred_3d, gt_3d, scale_factors = run_inference(model, loader, device)
    N = pred_2d.shape[0]

    # scale_factors (N,) in metres — multiply by 1000 to get mm conversion factor
    # per-sample: normalised_units * scale_factors[i] * 1000 = mm
    s_mm = scale_factors * 1000.0   # (N,)

    # ── 2D metrics (pixels only — no mm equivalent without sensor size) ───
    err_2d  = np.linalg.norm(pred_2d - gt_2d, axis=-1)   # (N, 21)
    mepe_2d = float(err_2d.mean())
    auc_2d  = auc(err_2d, max_threshold=30.0)
    pck_2d  = {t: pck(err_2d, t) * 100 for t in [2, 5, 10, 15]}

    # ── 3D metrics — normalised units ─────────────────────────────────────
    err_3d  = np.linalg.norm(pred_3d - gt_3d, axis=-1)   # (N, 21)
    mpjpe   = float(err_3d.mean())
    auc_3d  = auc(err_3d, max_threshold=0.5)
    pck_3d  = {t: pck(err_3d, t) * 100 for t in [0.05, 0.10, 0.20]}

    # ── 3D metrics — mm (per-sample rescale then average) ─────────────────
    # Each sample has its own scale factor s; err_mm[i] = err_norm[i] * s_mm[i]
    err_3d_mm  = err_3d * s_mm[:, None]                   # (N, 21)
    mpjpe_mm   = float(err_3d_mm.mean())
    auc_3d_mm  = auc(err_3d_mm, max_threshold=50.0)
    pck_3d_mm  = {t: pck(err_3d_mm, t) * 100 for t in [20.0, 30.0, 50.0]}

    # ── PA-MPJPE — normalised and mm ──────────────────────────────────────
    err_3d_pa    = np.zeros((N, 21), dtype=np.float32)
    err_3d_pa_mm = np.zeros((N, 21), dtype=np.float32)
    print('Computing PA-MPJPE...')
    for i in tqdm(range(N), ncols=90, desc='Procrustes'):
        aligned          = procrustes_align(pred_3d[i], gt_3d[i])
        err_3d_pa[i]     = np.linalg.norm(aligned - gt_3d[i], axis=-1)
        err_3d_pa_mm[i]  = err_3d_pa[i] * s_mm[i]
    pa_mpjpe    = float(err_3d_pa.mean())
    pa_mpjpe_mm = float(err_3d_pa_mm.mean())
    auc_3d_pa    = auc(err_3d_pa,    max_threshold=0.5)
    auc_3d_pa_mm = auc(err_3d_pa_mm, max_threshold=50.0)

    # ── Per-joint 3D error — normalised and mm ────────────────────────────
    per_joint    = err_3d.mean(axis=0)      # (21,) normalised
    per_joint_mm = err_3d_mm.mean(axis=0)   # (21,) mm

    # ── Bone-length error — normalised and mm ─────────────────────────────
    bone_err    = bone_length_error(pred_3d,         gt_3d)
    # Scale each sample's bone error by its own s_mm then average
    bone_err_mm = bone_length_error(pred_3d * s_mm[:, None, None],
                                    gt_3d   * s_mm[:, None, None])

    # ── Print summary ─────────────────────────────────────────────────────
    W = 60
    print(f'\n{"─"*W}')
    print(f'  {"Metric":<28}  {"Norm units":>12}  {"mm":>10}')
    print(f'  {"─"*28}  {"─"*12}  {"─"*10}')
    print(f'  {"MPJPE":<28}  {mpjpe:>12.4f}  {mpjpe_mm:>10.2f}')
    print(f'  {"PA-MPJPE":<28}  {pa_mpjpe:>12.4f}  {pa_mpjpe_mm:>10.2f}')
    print(f'  {"Bone-length err":<28}  {bone_err:>12.4f}  {bone_err_mm:>10.2f}')
    print(f'  {"─"*28}  {"─"*12}  {"─"*10}')
    print(f'  {"AUC-3D (0–0.5 / 0–50mm)":<28}  {auc_3d:>12.4f}  {auc_3d_mm:>10.4f}')
    print(f'  {"AUC-3D PA (0–0.5 / 0–50mm)":<28}  {auc_3d_pa:>12.4f}  {auc_3d_pa_mm:>10.4f}')
    print(f'  {"─"*28}  {"─"*12}  {"─"*10}')
    print(f'  {"PCK-3D @0.05 / @20mm":<28}  {pck_3d[0.05]:>11.2f}%  {pck_3d_mm[20.0]:>9.2f}%')
    print(f'  {"PCK-3D @0.10 / @30mm":<28}  {pck_3d[0.10]:>11.2f}%  {pck_3d_mm[30.0]:>9.2f}%')
    print(f'  {"PCK-3D @0.20 / @50mm":<28}  {pck_3d[0.20]:>11.2f}%  {pck_3d_mm[50.0]:>9.2f}%')
    print(f'  {"─"*28}  {"─"*12}  {"─"*10}')
    print(f'  {"MEPE-2D":<28}  {mepe_2d:>10.3f}px  {"(pixels only)":>10}')
    print(f'  {"AUC-2D (0–30px)":<28}  {auc_2d:>12.4f}  {"—":>10}')
    print(f'  {"PCK-2D @2px":<28}  {pck_2d[2]:>11.2f}%  {"—":>10}')
    print(f'  {"PCK-2D @5px":<28}  {pck_2d[5]:>11.2f}%  {"—":>10}')
    print(f'  {"PCK-2D @10px":<28}  {pck_2d[10]:>11.2f}%  {"—":>10}')
    print(f'  {"PCK-2D @15px":<28}  {pck_2d[15]:>11.2f}%  {"—":>10}')
    print(f'{"─"*W}')
    print(f'\n  Per-joint MPJPE (norm / mm):')
    for name, e_n, e_mm in zip(JOINT_NAMES, per_joint, per_joint_mm):
        print(f'    {name:<12}  {e_n:.4f}  /  {e_mm:.2f} mm')
    print(f'{"─"*W}\n')

    # ── Build CSV rows ────────────────────────────────────────────────────
    summary_row = {
        'metric':                'aggregate',
        'model':                 args.model,
        'epoch':                 ckpt.get('epoch', ''),
        'split':                 args.split,
        'N_samples':             N,
        # --- normalised units ---
        'MPJPE':                 round(mpjpe,       6),
        'PA_MPJPE':              round(pa_mpjpe,    6),
        'AUC_3D_0_0.5':         round(auc_3d,      6),
        'AUC_3D_PA_0_0.5':      round(auc_3d_pa,   6),
        'PCK_3D_0.05':          round(pck_3d[0.05],4),
        'PCK_3D_0.10':          round(pck_3d[0.10],4),
        'PCK_3D_0.20':          round(pck_3d[0.20],4),
        'Bone_length_err':       round(bone_err,    6),
        # --- mm ---
        'MPJPE_mm':              round(mpjpe_mm,       4),
        'PA_MPJPE_mm':           round(pa_mpjpe_mm,    4),
        'AUC_3D_0_50mm':        round(auc_3d_mm,      6),
        'AUC_3D_PA_0_50mm':     round(auc_3d_pa_mm,   6),
        'PCK_3D_20mm':          round(pck_3d_mm[20.0],4),
        'PCK_3D_30mm':          round(pck_3d_mm[30.0],4),
        'PCK_3D_50mm':          round(pck_3d_mm[50.0],4),
        'Bone_length_err_mm':    round(bone_err_mm,    4),
        # --- 2D (pixels only) ---
        'MEPE_2D_px':            round(mepe_2d,     6),
        'AUC_2D_0_30px':        round(auc_2d,       6),
        'PCK_2D_2px':           round(pck_2d[2],    4),
        'PCK_2D_5px':           round(pck_2d[5],    4),
        'PCK_2D_10px':          round(pck_2d[10],   4),
        'PCK_2D_15px':          round(pck_2d[15],   4),
    }
    # Per-joint columns: both units
    for name, e_n, e_mm in zip(JOINT_NAMES, per_joint, per_joint_mm):
        summary_row[f'joint_{name}']      = round(float(e_n),  6)
        summary_row[f'joint_{name}_mm']   = round(float(e_mm), 4)

    # Per-joint breakdown rows (one row per joint, both units)
    joint_rows = []
    for k, (name, e_n, e_mm) in enumerate(zip(JOINT_NAMES, per_joint, per_joint_mm)):
        joint_rows.append({
            'metric':     'per_joint',
            'joint_idx':  k,
            'joint_name': name,
            'MPJPE_3D':   round(float(e_n),  6),
            'MPJPE_3D_mm': round(float(e_mm), 4),
        })

    # ── Write CSV ─────────────────────────────────────────────────────────
    out_path = args.output
    with open(out_path, 'w', newline='') as f:
        agg_fieldnames = list(summary_row.keys())
        writer = csv.DictWriter(f, fieldnames=agg_fieldnames)
        writer.writeheader()
        writer.writerow(summary_row)

        f.write('\n')

        joint_fieldnames = ['metric', 'joint_idx', 'joint_name', 'MPJPE_3D', 'MPJPE_3D_mm']
        writer2 = csv.DictWriter(f, fieldnames=joint_fieldnames)
        writer2.writeheader()
        writer2.writerows(joint_rows)

    print(f'CSV saved -> {os.path.abspath(out_path)}')


if __name__ == '__main__':
    main()
