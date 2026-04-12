# ------------------------------------------------------------------------------
# FreiHAND Dataset loader for HOISDF codebase — with real MANO params
# Produces the exact same inputs/targets/meta_info dict structure
# as the original ho3d.py, but without any object-related fields.
#
# Change from previous version:
#   mano_param is now loaded from training_mano.json (pose 48-dim + shape 10-dim)
#   instead of being all zeros. This enables real MANO mesh supervision.
# ------------------------------------------------------------------------------

import os
import json
import random

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageFile, ImageFilter
from torch.utils import data

ImageFile.LOAD_TRUNCATED_IMAGES = True

from main.config import cfg


def load_K_xyz(root):
    with open(os.path.join(root, 'training_K.json'),   'r') as f:
        Ks = json.load(f)
    with open(os.path.join(root, 'training_xyz.json'), 'r') as f:
        xyzs = json.load(f)
    return Ks, xyzs


def load_mano_params(root):
    """
    Load MANO pose+shape params from training_mano.json.
    FreiHAND format: list of [[pose(48) + shape(10) + trans(3)]]
    data[i][0] is a flat 61-element list — we take first 58 (pose+shape).
    Returns list of (58,) float32 arrays, or None if file missing.
    """
    mano_path = os.path.join(root, 'training_mano.json')
    if not os.path.exists(mano_path):
        print(f'  WARNING: training_mano.json not found — MANO params set to zeros.')
        return None
    with open(mano_path, 'r') as f:
        raw = json.load(f)
    params = []
    for entry in raw:
        flat = np.array(entry[0], dtype=np.float32)  # (61,) pose+shape+trans
        params.append(flat[:58])                      # (58,) drop translation
    print(f'  Loaded MANO params: {len(params)} samples')
    return params


def projectPoints(xyz, K):
    xyz = np.array(xyz, dtype=np.float32)
    K   = np.array(K,   dtype=np.float32)
    uv  = (K @ xyz.T).T
    return uv[:, :2] / uv[:, 2:3]


def get_bbox_joints(joints_uv, bbox_factor=1.5):
    u_min, v_min = joints_uv.min(0)
    u_max, v_max = joints_uv.max(0)
    center = np.array([(u_min + u_max) / 2, (v_min + v_max) / 2])
    size   = max(u_max - u_min, v_max - v_min) * bbox_factor
    bbox   = np.array([
        center[0] - size / 2, center[1] - size / 2,
        center[0] + size / 2, center[1] + size / 2,
    ], dtype=np.float32)
    return bbox


def get_affine_transform(bbox, out_res):
    x1, y1, x2, y2 = bbox
    src = np.float32([[x1, y1], [x2, y1], [x1, y2]])
    dst = np.float32([[0,  0],  [out_res, 0], [0, out_res]])
    return cv2.getAffineTransform(src, dst)


def transform_coords(pts, M):
    pts_h = np.hstack([pts, np.ones((len(pts), 1), dtype=np.float32)])
    return (M @ pts_h.T).T


def transform_img(img_pil, M, out_res):
    img_np = np.array(img_pil)
    warped = cv2.warpAffine(img_np, M, (out_res, out_res),
                            flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT)
    return Image.fromarray(warped)


def color_jitter(img, brightness=0.5, saturation=0.5, hue=0.15, contrast=0.5):
    funcs = [
        lambda x: transforms.functional.adjust_brightness(x, 1 + random.uniform(-brightness, brightness)),
        lambda x: transforms.functional.adjust_contrast(x,   1 + random.uniform(-contrast,   contrast)),
        lambda x: transforms.functional.adjust_saturation(x, 1 + random.uniform(-saturation, saturation)),
        lambda x: transforms.functional.adjust_hue(x,            random.uniform(-hue, hue)),
    ]
    random.shuffle(funcs)
    for fn in funcs:
        img = fn(img)
    return img


def generate_hand_sdf_points(joints_3d, hand_root, num_points, sdf_scale, noise=0.02):
    joints_local = joints_3d - hand_root[None]
    idx   = np.random.randint(0, len(joints_local), size=num_points)
    delta = np.random.randn(num_points, 3).astype(np.float32) * noise
    pts   = joints_local[idx] + delta
    dists_to_joints = np.linalg.norm(
        pts[:, None, :] - joints_local[None, :, :], axis=-1)
    approx_sdf = dists_to_joints.min(axis=1) - 0.01
    approx_sdf = approx_sdf.astype(np.float32)
    pts_scaled = pts * sdf_scale
    return pts_scaled, approx_sdf


class Dataset(data.Dataset):
    """
    FreiHAND dataset formatted to match HOISDF's ho3d.py interface exactly.

    targets['mano_param'] is now a real (58,) tensor:
        [:48] = MANO pose params  (axis-angle, 16 joints × 3)
        [48:] = MANO shape params (10 PCA betas)

    If training_mano.json is missing, falls back to zeros (no MANO supervision).
    """

    def __init__(
        self,
        mode='train',
        max_rot=np.pi / 6,
        scale_jittering=0.2,
        center_jittering=0.1,
        hue=0.15,
        saturation=0.5,
        contrast=0.5,
        brightness=0.5,
        blur_radius=0.5,
    ):
        assert mode in ('train', 'evaluation')
        self.root        = cfg.freihand_data_dir
        self.mode        = mode
        self.joint_num   = 21
        self.inp_res     = cfg.input_img_shape[0]
        self.heatmap_res = cfg.output_hm_shape[0]
        self.transform   = transforms.ToTensor()

        self.num_samp_hand  = cfg.num_samp_hand
        self.hand_sdf_scale = cfg.hand_sdf_scale
        self.dist           = cfg.points_filter_dist

        self.max_rot          = max_rot
        self.scale_jittering  = scale_jittering
        self.center_jittering = center_jittering
        self.hue         = hue
        self.saturation  = saturation
        self.contrast    = contrast
        self.brightness  = brightness
        self.blur_radius = blur_radius

        # Load annotations
        Ks, xyzs = load_K_xyz(self.root)
        n_total  = len(Ks)

        # 90 / 10 split
        n_train = int(n_total * 0.9)
        if mode == 'train':
            self.indices = list(range(n_train))
        else:
            self.indices = list(range(n_train, n_total))

        self.Ks   = [np.array(Ks[i],   dtype=np.float32) for i in self.indices]
        self.xyzs = [np.array(xyzs[i], dtype=np.float32) for i in self.indices]

        # Load MANO params (real or zeros)
        all_mano = load_mano_params(self.root)
        if all_mano is not None:
            self.mano_params = [all_mano[i] for i in self.indices]
            self.has_mano    = True
        else:
            self.mano_params = None
            self.has_mano    = False

        # Pre-compute 2D projections
        self.joints_uv = [
            projectPoints(self.xyzs[i], self.Ks[i])
            for i in range(len(self.indices))
        ]

        print(f'FreiHAND {mode}: {len(self.indices)} samples loaded'
              f'  (MANO params: {"YES" if self.has_mano else "NO — zeros"})')

    def __len__(self):
        return len(self.indices)

    def _augment(self, img, joints_uv, joints_3d, K):
        img       = img.copy()
        joints_uv = joints_uv.copy()
        joints_3d = joints_3d.copy()
        K         = K.copy()

        bbox = get_bbox_joints(joints_uv, bbox_factor=1.5)
        cx   = (bbox[0] + bbox[2]) / 2
        cy   = (bbox[1] + bbox[3]) / 2
        size = (bbox[2] - bbox[0] + bbox[3] - bbox[1]) / 2

        cx  += self.center_jittering * size * np.random.uniform(-1, 1)
        cy  += self.center_jittering * size * np.random.uniform(-1, 1)
        scale = np.clip(self.scale_jittering * np.random.randn() + 1,
                        1 - self.scale_jittering, 1 + self.scale_jittering)
        size *= scale

        angle     = np.random.uniform(-self.max_rot, self.max_rot)
        angle_deg = np.degrees(angle)

        bbox_sq = np.array([cx - size, cy - size, cx + size, cy + size])
        M = get_affine_transform(bbox_sq, self.inp_res)

        rot_center = (self.inp_res / 2, self.inp_res / 2)
        R = cv2.getRotationMatrix2D(rot_center, -angle_deg, 1.0)
        M_full = np.vstack([R, [0, 0, 1]]) @ np.vstack([M, [0, 0, 1]])
        M_full = M_full[:2]

        joints_uv = transform_coords(joints_uv, M_full)
        img       = transform_img(img, M_full, self.inp_res)

        blur = random.random() * self.blur_radius
        img  = img.filter(ImageFilter.GaussianBlur(blur))
        img  = color_jitter(img,
                            brightness=self.brightness,
                            saturation=self.saturation,
                            hue=self.hue,
                            contrast=self.contrast)

        rot_3d = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle),  np.cos(angle), 0],
            [0,              0,             1],
        ], dtype=np.float32)
        joints_3d = joints_3d @ rot_3d.T

        cx_old, cy_old = K[0, 2], K[1, 2]
        fx_old, fy_old = K[0, 0], K[1, 1]
        crop_size = 2 * size
        s = self.inp_res / crop_size
        K = K.copy()
        K[0, 0] = fx_old * s
        K[1, 1] = fy_old * s
        pp_old  = np.array([[cx_old, cy_old, 1.0]], dtype=np.float32)
        pp_new  = (M_full @ pp_old.T).T
        K[0, 2] = pp_new[0, 0]
        K[1, 2] = pp_new[0, 1]

        return img, joints_uv, joints_3d, K

    def _crop_eval(self, img, joints_uv, K):
        bbox      = get_bbox_joints(joints_uv, bbox_factor=1.5)
        M         = get_affine_transform(bbox, self.inp_res)
        joints_uv = transform_coords(joints_uv, M)
        img       = transform_img(img, M, self.inp_res)
        K_h       = np.eye(3, dtype=np.float32)
        K_h[:2]   = M
        K         = K_h @ K
        return img, joints_uv, K

    def __getitem__(self, i):
        local_idx = i
        abs_idx   = self.indices[i]

        img_path = os.path.join(
            self.root, 'training', 'rgb', f'{abs_idx:08d}.jpg')
        img = Image.open(img_path).convert('RGB')

        K         = self.Ks[local_idx].copy()
        joints_3d = self.xyzs[local_idx].copy()
        joints_uv = self.joints_uv[local_idx].copy()

        if self.mode == 'train':
            img, joints_uv, joints_3d, K = self._augment(
                img, joints_uv, joints_3d, K)
        else:
            img, joints_uv, K = self._crop_eval(img, joints_uv, K)

        hand_root    = joints_3d[0].copy()
        joints_3d_rel = joints_3d - hand_root[None]

        bbox_hand    = get_bbox_joints(joints_uv, bbox_factor=1.2)
        joints_uv_hm = joints_uv / self.inp_res * self.heatmap_res

        hand_sdf_pts, hand_sdf_vals = generate_hand_sdf_points(
            joints_3d, hand_root,
            num_points=self.num_samp_hand,
            sdf_scale=self.hand_sdf_scale,
        )
        hand_pre_pts, _ = generate_hand_sdf_points(
            joints_3d, hand_root,
            num_points=self.num_samp_hand,
            sdf_scale=self.hand_sdf_scale,
            noise=self.dist,
        )

        # MANO param: real (58,) if available, else zeros
        if self.has_mano:
            mano_param = torch.from_numpy(self.mano_params[local_idx])
        else:
            mano_param = torch.zeros(58, dtype=torch.float32)

        img_t = self.transform(np.asarray(img).astype(np.float32)) / 255.0

        inputs = {
            'img':              img_t,
            'hand_sdf_points':  torch.from_numpy(hand_sdf_pts.astype(np.float32)),
            'hand_pre_points':  torch.from_numpy(hand_pre_pts.astype(np.float32)),
        }

        targets = {
            'joint_coord':          torch.from_numpy(joints_uv_hm.astype(np.float32)),
            'joint_cam_no_trans':   torch.from_numpy((joints_3d_rel * 1000).astype(np.float32)),
            'hand_sdf':             torch.from_numpy(hand_sdf_vals),
            'mano_param':           mano_param,
        }

        meta_info = {
            'cam_intr':     torch.from_numpy(K),
            'mano_root':    torch.from_numpy(hand_root),
            'bbox_hand':    torch.from_numpy(bbox_hand),
        }

        return inputs, targets, meta_info