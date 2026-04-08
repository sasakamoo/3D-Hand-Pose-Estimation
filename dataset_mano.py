"""
dataset_mano.py — FreiHAND Dataset loader with MANO parameter targets
======================================================================
Extends dataset.py with ground-truth MANO pose and shape parameters
from training_mano.json.

FreiHAND training_mano.json format:
    A JSON list of N entries, one per sample.  Each entry is either:
      (a)  [pose_list_48, shape_list_10]   — list-of-lists (most common)
      (b)  {"pose": [...48...], "shape": [...10...]}  — dict

    pose  : 48 floats  — axis-angle rotations for 16 MANO joints
              [0:3]   global orientation (wrist)
              [3:48]  finger joint rotations (15 joints × 3)
    shape : 10 floats  — MANO shape (PCA) coefficients

Additional batch keys returned (on top of dataset.py keys):
    mano_pose  : (48,)  float32  MANO pose parameters
    mano_shape : (10,)  float32  MANO shape parameters

Note on augmentation and MANO params:
    Image-space augmentations (colour jitter, scale+translation) do not
    affect joint angles or shape, so mano_pose and mano_shape are returned
    as-is regardless of augmentation.  In-plane rotation DOES change the
    global orientation component of mano_pose (first 3 values).  We apply
    the rotation correction so that the predicted MANO params remain
    consistent with the augmented image.
"""

import os
import json
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import random


# Reference bone for scale normalisation  (inherited from dataset.py)
SCALE_REF_BONE = (0, 5)
C_NORM         = 1.0
IMG_SIZE       = 128

# MANO parameter dimensions
MANO_POSE_DIM  = 48   # 16 joints × 3 axis-angle
MANO_SHAPE_DIM = 10   # shape PCA coefficients


def _parse_mano_entry(entry):
    """
    Parse one entry from training_mano.json into (pose_48, shape_10) numpy arrays.
    Handles both list-of-lists and dict formats.
    """
    if isinstance(entry, dict):
        pose  = np.array(entry['pose'],  dtype=np.float32)
        shape = np.array(entry['shape'], dtype=np.float32)
    else:
        # list/tuple: [pose_48, shape_10]
        pose  = np.array(entry[0], dtype=np.float32)
        shape = np.array(entry[1], dtype=np.float32)

    assert pose.shape  == (MANO_POSE_DIM,),  f'Expected pose (48,), got {pose.shape}'
    assert shape.shape == (MANO_SHAPE_DIM,), f'Expected shape (10,), got {shape.shape}'
    return pose, shape


def _rotate_global_orient(pose: np.ndarray, angle_deg: float) -> np.ndarray:
    """
    Correct the MANO global orientation (pose[0:3], axis-angle) for an
    in-plane image rotation of `angle_deg` degrees around the optical axis.

    An in-plane rotation R_z(θ) applied to the image is equivalent to
    pre-multiplying the global orientation rotation matrix by R_z(θ)^T = R_z(-θ).

    Steps:
      1. Convert axis-angle pose[0:3] → rotation matrix R_hand
      2. Build R_z(-θ) for the in-plane correction
      3. R_new = R_z(-θ) @ R_hand
      4. Convert R_new back to axis-angle
    """
    import cv2 as _cv2

    theta = np.deg2rad(angle_deg)
    c, s  = np.cos(theta), np.sin(theta)
    # Rotation around Z axis by -theta  (compensates for image rotation by +theta)
    Rz = np.array([[ c,  s, 0],
                   [-s,  c, 0],
                   [ 0,  0, 1]], dtype=np.float32)

    aa  = pose[:3].copy()
    R_hand, _ = _cv2.Rodrigues(aa)
    R_new     = Rz @ R_hand
    aa_new, _ = _cv2.Rodrigues(R_new)

    pose_new       = pose.copy()
    pose_new[:3]   = aa_new.flatten()
    return pose_new


class FreiHANDDatasetMANO(Dataset):
    """
    FreiHAND dataset returning both 2.5D targets and MANO parameters.

    Labels returned:
        image        : (3, 128, 128)  float32 in [0, 1]
        pose_2d_gt   : (K, 2)   2D pixel coords in [0, 128]
        depth_rel_gt : (K,)     root-relative scale-normalised depth z^r_k
        K_mat        : (3, 3)   camera intrinsics (adjusted for crop/resize)
        scale_factor : scalar   s used for normalisation
        xyz_raw      : (K, 3)   raw metric XYZ in metres
        mano_pose    : (48,)    MANO pose parameters  (axis-angle, 16 joints)
        mano_shape   : (10,)    MANO shape parameters (PCA coefficients)
    """

    def __init__(self, root: str, split: str = 'train', augment: bool = False):
        assert split in ('train', 'val')
        self.root    = root
        self.split   = split
        self.augment = augment

        base_dir = root
        with open(os.path.join(base_dir, 'training_K.json'))     as f: self.Ks     = json.load(f)
        with open(os.path.join(base_dir, 'training_scale.json')) as f: self.scales = json.load(f)
        with open(os.path.join(base_dir, 'training_xyz.json'))   as f: self.xyzs   = json.load(f)
        with open(os.path.join(base_dir, 'training_mano.json'))  as f: self.manos  = json.load(f)

        self.base_dir = base_dir
        self.n_total  = len(self.Ks)

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
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        orig_h, orig_w = img.shape[:2]

        # ── Raw labels ────────────────────────────────────────────────────
        K_mat              = np.array(self.Ks[idx],   dtype=np.float32)   # (3,3)
        xyz                = np.array(self.xyzs[idx], dtype=np.float32)   # (K,3)
        mano_pose, mano_shape = _parse_mano_entry(self.manos[idx])         # (48,), (10,)

        # ── Scale normalisation ───────────────────────────────────────────
        n, m   = SCALE_REF_BONE
        s      = float(np.linalg.norm(xyz[n] - xyz[m]) + 1e-8)
        xyz_norm = xyz * (C_NORM / s)

        z_root    = xyz_norm[0, 2]
        depth_rel = xyz_norm[:, 2] - z_root

        # ── Project normalised 3D → 2D pixel coords ──────────────────────
        P  = xyz_norm.T
        p  = K_mat @ P
        uv = (p[:2] / (p[2:3] + 1e-8)).T

        # ── Resize image and adjust intrinsics ───────────────────────────
        scale_x     = IMG_SIZE / orig_w
        scale_y     = IMG_SIZE / orig_h
        img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        uv_resized  = uv.copy()
        uv_resized[:, 0] *= scale_x
        uv_resized[:, 1] *= scale_y

        K_resized = K_mat.copy()
        K_resized[0] *= scale_x
        K_resized[1] *= scale_y

        # ── Augmentation ──────────────────────────────────────────────────
        rot_angle = 0.0   # track rotation for MANO correction
        if self.augment:
            img_resized, uv_resized, K_resized, rot_angle = self._augment(
                img_resized, uv_resized, K_resized)
        if rot_angle != 0.0:
            mano_pose = _rotate_global_orient(mano_pose, rot_angle)

        # ── To tensors ────────────────────────────────────────────────────
        img_t        = torch.from_numpy(
            img_resized.astype(np.float32) / 255.0).permute(2, 0, 1)
        pose_2d_gt   = torch.from_numpy(uv_resized.astype(np.float32))
        depth_rel_t  = torch.from_numpy(depth_rel.astype(np.float32))
        K_t          = torch.from_numpy(K_resized)
        xyz_raw_t    = torch.from_numpy(xyz)
        mano_pose_t  = torch.from_numpy(mano_pose)
        mano_shape_t = torch.from_numpy(mano_shape)

        return {
            'image':        img_t,            # (3, 128, 128)
            'pose_2d_gt':   pose_2d_gt,       # (K, 2)
            'depth_rel_gt': depth_rel_t,      # (K,)
            'K_mat':        K_t,              # (3, 3)
            'scale_factor': torch.tensor(s),  # scalar
            'xyz_raw':      xyz_raw_t,        # (K, 3)
            'mano_pose':    mano_pose_t,      # (48,)
            'mano_shape':   mano_shape_t,     # (10,)
        }

    # ── Augmentation helpers ──────────────────────────────────────────────────

    def _augment(self, img, uv, K):
        """
        Returns (img, uv, K, rot_angle_deg) — rot_angle is passed back so
        the caller can apply the corresponding MANO global orient correction.
        """
        H, W = img.shape[:2]

        # Colour jitter
        if random.random() < 0.8:
            img = self._colour_jitter(img)

        # Scale + translation
        scale = random.uniform(0.8, 1.2)
        tx    = random.uniform(-0.1, 0.1) * W
        ty    = random.uniform(-0.1, 0.1) * H
        M_st  = np.array([[scale, 0, tx + (1-scale)*W/2],
                          [0, scale, ty + (1-scale)*H/2]], dtype=np.float32)
        img   = cv2.warpAffine(img, M_st, (W, H), borderMode=cv2.BORDER_REFLECT)

        uv_h   = np.hstack([uv, np.ones((uv.shape[0], 1))])
        uv     = (M_st @ uv_h.T).T
        K[0,2] = K[0,2] * scale + tx + (1-scale)*W/2
        K[1,2] = K[1,2] * scale + ty + (1-scale)*H/2
        K[0,0] *= scale
        K[1,1] *= scale

        # Rotation around optical axis
        angle  = random.uniform(-30, 30)
        cx, cy = W / 2, H / 2
        M_rot  = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        img    = cv2.warpAffine(img, M_rot, (W, H), borderMode=cv2.BORDER_REFLECT)
        uv_h   = np.hstack([uv, np.ones((uv.shape[0], 1))])
        uv     = (M_rot @ uv_h.T).T

        pp     = np.array([K[0, 2], K[1, 2]])
        pp_new = M_rot[:, :2] @ pp + M_rot[:, 2]
        K[0, 2] = pp_new[0]
        K[1, 2] = pp_new[1]

        uv[:, 0] = np.clip(uv[:, 0], 0, W - 1)
        uv[:, 1] = np.clip(uv[:, 1], 0, H - 1)

        return img, uv, K, angle

    @staticmethod
    def _colour_jitter(img, brightness=0.3, contrast=0.3,
                       saturation=0.3, hue=0.05):
        img_pil = TF.to_pil_image(img)
        img_pil = TF.adjust_brightness(img_pil, 1 + random.uniform(-brightness, brightness))
        img_pil = TF.adjust_contrast(img_pil,   1 + random.uniform(-contrast,   contrast))
        img_pil = TF.adjust_saturation(img_pil, 1 + random.uniform(-saturation, saturation))
        img_pil = TF.adjust_hue(img_pil,            random.uniform(-hue, hue))
        return np.array(img_pil)
