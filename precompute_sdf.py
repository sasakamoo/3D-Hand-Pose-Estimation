# -*- coding: utf-8 -*-
"""
precompute_sdf.py -- Precompute GT 3D joint positions in model-normalised space
================================================================================
Computes the normalised (u, v, d) joint coordinates for every FreiHAND sample
and saves them as a single numpy array.  Training then uses 3D bone-segment
distances as proper SDF supervision instead of the crude 2D joint-proximity proxy.

Output
------
sdf_joints_norm.npy  :  float32  (N_total, 21, 3)
    Axis-2 layout:  [u_norm, v_norm, d_norm]
      u_norm, v_norm  in [-1, 1]  (normalised image plane, same as model pts[:,:,:2])
      d_norm          in model depth_rel units  (root-relative, scale C=1 bone)

Usage
-----
    python precompute_sdf.py --data-root /path/to/FreiHAND
    python precompute_sdf.py --data-root /path/to/FreiHAND --out sdf_joints_norm.npy
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Must match dataset.py exactly
SCALE_REF_BONE = (0, 5)   # wrist -> index MCP
C_NORM         = 1.0
IMG_SIZE       = 128


def compute_joints_norm(xyz: np.ndarray,
                        K_mat: np.ndarray,
                        orig_h: int,
                        orig_w: int) -> np.ndarray:
    """
    Replicate the dataset.py normalisation pipeline for a single sample.

    Returns
    -------
    joints_norm : (21, 3)   [u_norm, v_norm, d_norm] in model space
    """
    # 1. Scale normalisation  (dataset.py lines 99-105)
    n, m = SCALE_REF_BONE
    s = float(np.linalg.norm(xyz[n] - xyz[m]) + 1e-8)
    xyz_norm = xyz * (C_NORM / s)

    # 2. Root-relative depth
    z_root    = xyz_norm[0, 2]
    depth_rel = xyz_norm[:, 2] - z_root          # (21,)

    # 3. Perspective projection to original pixel coords
    P   = xyz_norm.T                              # (3, 21)
    p   = K_mat @ P                              # (3, 21)
    uv  = (p[:2] / (p[2:3] + 1e-8)).T           # (21, 2) original pixels

    # 4. Resize 2D coords to IMG_SIZE  (dataset.py lines 119-120)
    scale_x = IMG_SIZE / orig_w
    scale_y = IMG_SIZE / orig_h
    uv_r    = uv.copy()
    uv_r[:, 0] *= scale_x
    uv_r[:, 1] *= scale_y

    # 5. Convert pixel coords to model normalised image space [-1, 1]
    #    same transform as pts[:,:,:2] in SDFHandPoseNet._sample_points
    u_norm = uv_r[:, 0] / (IMG_SIZE / 2.0) - 1.0   # (21,)
    v_norm = uv_r[:, 1] / (IMG_SIZE / 2.0) - 1.0   # (21,)

    joints_norm = np.stack([u_norm, v_norm, depth_rel], axis=-1)  # (21, 3)
    return joints_norm.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str,
                        default='/home/kghasemz/scratch/datasets/FreiHand')
    parser.add_argument('--out', type=str, default='sdf_joints_norm.npy',
                        help='Output path for the precomputed array')
    args = parser.parse_args()

    base = Path(args.data_root)

    print('Loading FreiHAND JSON labels...')
    with open(base / 'training_K.json')     as f: Ks     = json.load(f)
    with open(base / 'training_xyz.json')   as f: xyzs   = json.load(f)

    N = len(Ks)
    print(f'Total samples: {N}')

    # FreiHAND images are 224x224
    # (all training images are the same resolution)
    orig_h, orig_w = 224, 224

    # Try to read actual image size from the first sample to be safe
    img_dir = base / 'training' / 'rgb'
    try:
        import cv2
        sample_img = cv2.imread(str(img_dir / f'{0:08d}.jpg'))
        if sample_img is not None:
            orig_h, orig_w = sample_img.shape[:2]
    except Exception:
        pass
    print(f'Image size: {orig_h}x{orig_w}')

    joints_all = np.zeros((N, 21, 3), dtype=np.float32)

    for i in tqdm(range(N), desc='Computing normalised joints', ncols=90):
        K_mat = np.array(Ks[i],   dtype=np.float32)
        xyz   = np.array(xyzs[i], dtype=np.float32)
        joints_all[i] = compute_joints_norm(xyz, K_mat, orig_h, orig_w)

    np.save(args.out, joints_all)
    size_mb = joints_all.nbytes / 1e6
    print(f'\nSaved {args.out}  '
          f'shape={joints_all.shape}  dtype={joints_all.dtype}  '
          f'size={size_mb:.1f} MB')
    print('\nDepth range (d_norm):  '
          f'min={joints_all[:,:,2].min():.3f}  '
          f'max={joints_all[:,:,2].max():.3f}')
    print('UV range:              '
          f'min={joints_all[:,:,:2].min():.3f}  '
          f'max={joints_all[:,:,:2].max():.3f}')


if __name__ == '__main__':
    main()
