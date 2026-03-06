"""
dataset.py — FreiHAND Dataset loader with correct 2.5D target preparation
==========================================================================

Implements the normalization described in Iqbal et al. (§3.1, Eq. 2):

  s  = ||P_n - P_parent(n)||   (length of reference bone before normalisation)
  Pˆ = (C / s) · P             (scale-normalised 3D pose, C = 1)

Reference bone: wrist (joint 0) ↔ index MCP (joint 5).   C = 1 (paper §3.2).

After normalisation the network targets are:
  - 2D pixel coords  (x_k, y_k)  in the resized 128×128 image
  - Root-relative depth  z^r_k = Zˆ_k - Zˆ_root  for each joint k

Data augmentation (paper §3.2):
  colour jitter, scale, translation, rotation around the optical axis.
"""

import os
import json
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import random


# Reference bone for scale normalisation (paper §3.2)
SCALE_REF_BONE = (0, 5)   # wrist → index MCP
C_NORM = 1.0              # target bone length

IMG_SIZE = 128            # paper §5.2


class FreiHANDDataset(Dataset):
    """
    FreiHAND dataset returning 2.5D targets.

    Labels returned:
        image        : (3, 128, 128)  float32 in [0, 1]
        pose_2d_gt   : (K, 2)  2D pixel coords in [0, 128] of resized image
        depth_rel_gt : (K,)    root-relative scale-normalised depth  z^r_k
        K_mat        : (3, 3)  camera intrinsics (adjusted for crop/resize)
        scale_factor : scalar  s used for normalisation  (to undo later)
        xyz_raw      : (K, 3)  raw metric XYZ in metres  (for validation)
    """

    def __init__(self, root: str, split: str = 'train',
                 augment: bool = False):
        """
        Args:
            root    : path to dataset root (contains FreiHAND_pub_v2/)
            split   : 'train' or 'val'
            augment : apply colour/scale/rotation augmentation
        """
        assert split in ('train', 'val')
        self.root    = root
        self.split   = split
        self.augment = augment

        base_dir = os.path.join(root, 'FreiHAND_pub_v2')
        with open(os.path.join(base_dir, 'training_K.json'))     as f: self.Ks     = json.load(f)
        with open(os.path.join(base_dir, 'training_scale.json')) as f: self.scales = json.load(f)
        with open(os.path.join(base_dir, 'training_xyz.json'))   as f: self.xyzs   = json.load(f)

        self.base_dir  = base_dir
        self.n_total   = len(self.Ks)

        # 90/10 split
        n_train = int(self.n_total * 0.9)
        if split == 'train':
            self.indices = list(range(n_train))
        else:
            self.indices = list(range(n_train, self.n_total))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]

        # ── Load image ────────────────────────────────────────────────────
        img_path = os.path.join(self.base_dir, 'training', 'rgb',
                                f'{idx:08d}.jpg')
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)     # (H, W, 3) uint8

        orig_h, orig_w = img.shape[:2]

        # ── Raw labels ────────────────────────────────────────────────────
        K_mat = np.array(self.Ks[idx],     dtype=np.float32)  # (3,3)
        xyz   = np.array(self.xyzs[idx],   dtype=np.float32)  # (K,3) metres

        # ── Scale normalisation (Iqbal §3.1, Eq.2) ───────────────────────
        n, m = SCALE_REF_BONE
        s = float(np.linalg.norm(xyz[n] - xyz[m]) + 1e-8)
        xyz_norm = xyz * (C_NORM / s)                   # scale-normalised

        # Root depth and root-relative depth
        z_root    = xyz_norm[0, 2]                      # wrist Z (normalised)
        depth_rel = xyz_norm[:, 2] - z_root             # (K,) z^r_k

        # ── Project normalised 3D → 2D pixel coords ──────────────────────
        # p_k = (K @ P_k) / Z_k  (perspective projection)
        P = xyz_norm.T                                  # (3, K)
        p = K_mat @ P                                   # (3, K)
        uv = (p[:2] / (p[2:3] + 1e-8)).T               # (K, 2) in original px

        # ── Resize image and adjust intrinsics ───────────────────────────
        scale_x = IMG_SIZE / orig_w
        scale_y = IMG_SIZE / orig_h
        img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        # Adjust 2D coords for resize
        uv_resized = uv.copy()
        uv_resized[:, 0] *= scale_x
        uv_resized[:, 1] *= scale_y

        # Adjust intrinsics for resize
        K_resized = K_mat.copy()
        K_resized[0] *= scale_x   # fx, cx
        K_resized[1] *= scale_y   # fy, cy

        # ── Augmentation ──────────────────────────────────────────────────
        if self.augment:
            img_resized, uv_resized, K_resized = self._augment(
                img_resized, uv_resized, K_resized)

        # ── To tensors ────────────────────────────────────────────────────
        img_t       = torch.from_numpy(
            img_resized.astype(np.float32) / 255.0).permute(2, 0, 1)
        pose_2d_gt  = torch.from_numpy(uv_resized.astype(np.float32))
        depth_rel_t = torch.from_numpy(depth_rel.astype(np.float32))
        K_t         = torch.from_numpy(K_resized)
        xyz_raw_t   = torch.from_numpy(xyz)

        return {
            'image':        img_t,            # (3, 128, 128)
            'pose_2d_gt':   pose_2d_gt,       # (K, 2)
            'depth_rel_gt': depth_rel_t,      # (K,)
            'K_mat':        K_t,              # (3, 3)
            'scale_factor': torch.tensor(s),  # scalar
            'xyz_raw':      xyz_raw_t,        # (K, 3) metres
        }

    # ── Augmentation helpers ──────────────────────────────────────────────────

    def _augment(self, img, uv, K):
        """
        Standard augmentations from FreiHAND §3.2:
          - Colour jitter
          - Scale and translation
          - Rotation around optical axis
        """
        H, W = img.shape[:2]

        # Colour jitter
        if random.random() < 0.8:
            img = self._colour_jitter(img)

        # Scale + translation (random crop/zoom within ±20%)
        scale = random.uniform(0.8, 1.2)
        tx    = random.uniform(-0.1, 0.1) * W
        ty    = random.uniform(-0.1, 0.1) * H
        M_st  = np.array([[scale, 0, tx + (1-scale)*W/2],
                          [0, scale, ty + (1-scale)*H/2]], dtype=np.float32)
        img = cv2.warpAffine(img, M_st, (W, H), borderMode=cv2.BORDER_REFLECT)

        # Adjust 2D coords and K for scale+translation
        uv_h   = np.hstack([uv, np.ones((uv.shape[0], 1))])   # homogeneous
        uv     = (M_st @ uv_h.T).T
        K[0,2] = K[0,2] * scale + tx + (1-scale)*W/2
        K[1,2] = K[1,2] * scale + ty + (1-scale)*H/2
        K[0,0] *= scale
        K[1,1] *= scale

        # Rotation around optical axis
        angle = random.uniform(-30, 30)
        cx, cy = W / 2, H / 2
        M_rot  = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        img    = cv2.warpAffine(img, M_rot, (W, H), borderMode=cv2.BORDER_REFLECT)
        uv_h   = np.hstack([uv, np.ones((uv.shape[0], 1))])
        uv     = (M_rot @ uv_h.T).T

        # Clip to image boundaries
        uv[:, 0] = np.clip(uv[:, 0], 0, W - 1)
        uv[:, 1] = np.clip(uv[:, 1], 0, H - 1)

        return img, uv, K

    @staticmethod
    def _colour_jitter(img, brightness=0.3, contrast=0.3,
                       saturation=0.3, hue=0.05):
        img_pil = TF.to_pil_image(img)
        img_pil = TF.adjust_brightness(img_pil, 1 + random.uniform(-brightness, brightness))
        img_pil = TF.adjust_contrast(img_pil,   1 + random.uniform(-contrast,   contrast))
        img_pil = TF.adjust_saturation(img_pil, 1 + random.uniform(-saturation, saturation))
        img_pil = TF.adjust_hue(img_pil,            random.uniform(-hue, hue))
        return np.array(img_pil)
