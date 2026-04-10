# -*- coding: utf-8 -*-
"""
view_3d.py -- Interactive 3D hand pose viewer (dual-camera)
============================================================
Opens two independent Open3D windows side-by-side:
  Left  window  — Ground Truth joints (grey)
  Right window  — Predicted joints    (coloured by finger)

Each window has its own camera so you can rotate/zoom them independently.
Both windows must be closed (press Q in each) to quit.

Hotkeys (focus the window you want to control first):
  R — toggle auto-spin for that window
  Q — close that window

Usage:
    python view_3d.py --sample 0
    python view_3d.py --sample 3 --ply-dir visualizations
"""

import argparse
import sys
import time
from pathlib import Path
import numpy as np
import open3d as o3d

CONNECTIONS = [
    [0,1],[1,2],[2,3],[3,4],
    [0,5],[5,6],[6,7],[7,8],
    [0,9],[9,10],[10,11],[11,12],
    [0,13],[13,14],[14,15],[15,16],
    [0,17],[17,18],[18,19],[19,20],
]

# Spin parameters
SPIN_STEP_PX = 4.0   # horizontal pixel-drag equivalent per tick
SPIN_FPS     = 30    # max spin updates per second

parser = argparse.ArgumentParser()
parser.add_argument('--sample',  type=int, default=0,
                    help='Sample index to display (matches 3d_pred/gt_XXXX.ply)')
parser.add_argument('--ply-dir', type=str, default='visualizations',
                    help='Directory containing the .ply files from visualize.py')
args = parser.parse_args()

ply_dir   = Path(args.ply_dir)
pred_path = ply_dir / f'3d_pred_{args.sample:04d}.ply'
gt_path   = ply_dir / f'3d_gt_{args.sample:04d}.ply'

# ── Validate files ────────────────────────────────────────────────────────────
missing = [p for p in (pred_path, gt_path) if not p.exists()]
if missing:
    for p in missing:
        print(f'ERROR: file not found: {p}')
    available = sorted(ply_dir.glob('3d_pred_*.ply')) if ply_dir.exists() else []
    if available:
        indices = [int(p.stem.split('_')[-1]) for p in available]
        print(f'Available sample indices in "{args.ply_dir}": {indices}')
        print(f'Example: python view_3d.py --ply-dir {args.ply_dir} --sample {indices[0]}')
    else:
        print(f'No 3d_pred_*.ply files found in "{args.ply_dir}".')
        print('Run visualize.py first to generate them.')
    sys.exit(1)

# ── Load point clouds ─────────────────────────────────────────────────────────
def load_pcd_with_skeleton(ply_path):
    pcd    = o3d.io.read_point_cloud(str(ply_path))
    pts    = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    if len(pts) < 21:
        print(f'ERROR: expected 21 points in {ply_path}, got {len(pts)}.')
        sys.exit(1)

    bone_cols = [colors[s].tolist() for s, e in CONNECTIONS]
    ls        = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(pts)
    ls.lines  = o3d.utility.Vector2iVector(
                    np.array(CONNECTIONS, dtype=np.int32))
    ls.colors = o3d.utility.Vector3dVector(np.array(bone_cols))

    return pcd, ls

pred_pcd, pred_ls = load_pcd_with_skeleton(pred_path)
gt_pcd,   gt_ls   = load_pcd_with_skeleton(gt_path)

print(f'Loaded sample {args.sample}')
print('Controls : left-drag=rotate  scroll=zoom  right-drag=pan')
print('Hotkeys  : R=toggle spin  Q=quit window')
print('(Close both windows to exit)\n')

# ── Spin state ────────────────────────────────────────────────────────────────
spinning   = {'gt': False, 'pred': False}
last_tick  = {'gt': 0.0,   'pred': 0.0}

def make_spin_toggle(name):
    def _callback(vis):
        spinning[name] = not spinning[name]
        print(f'  Spin {"ON " if spinning[name] else "OFF"} ({name})')
        return False
    return _callback

# ── Create two independent visualizer windows ─────────────────────────────────
BG = np.array([0.05, 0.05, 0.05])

def _make_vis(title, pcd, ls, x_offset, spin_name):
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name=title, width=720, height=720,
                      left=x_offset, top=80)
    vis.add_geometry(pcd)
    vis.add_geometry(ls)
    opt = vis.get_render_option()
    opt.background_color      = BG
    opt.point_size            = 6.0
    opt.line_width            = 2.0
    opt.show_coordinate_frame = False
    vis.register_key_callback(ord('R'), make_spin_toggle(spin_name))
    return vis

vis_gt   = _make_vis(f'Ground Truth — sample {args.sample}  [R=spin]',
                     gt_pcd,   gt_ls,   80,  'gt')
vis_pred = _make_vis(f'Predicted    — sample {args.sample}  [R=spin]',
                     pred_pcd, pred_ls, 820, 'pred')

# ── Run both windows in a shared poll loop ────────────────────────────────────
_spin_interval = 1.0 / SPIN_FPS

while True:
    alive_gt   = vis_gt.poll_events()
    alive_pred = vis_pred.poll_events()

    if not alive_gt and not alive_pred:
        break

    now = time.monotonic()

    if alive_gt:
        if spinning['gt'] and now - last_tick['gt'] >= _spin_interval:
            vis_gt.get_view_control().rotate(SPIN_STEP_PX, 0.0)
            last_tick['gt'] = now
        vis_gt.update_renderer()

    if alive_pred:
        if spinning['pred'] and now - last_tick['pred'] >= _spin_interval:
            vis_pred.get_view_control().rotate(SPIN_STEP_PX, 0.0)
            last_tick['pred'] = now
        vis_pred.update_renderer()

vis_gt.destroy_window()
vis_pred.destroy_window()
