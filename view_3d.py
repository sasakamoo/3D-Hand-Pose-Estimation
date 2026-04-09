# -*- coding: utf-8 -*-
"""
view_3d.py -- Interactive 3D hand pose viewer
==============================================
Reads the .ply files saved by visualize.py and displays predicted vs GT
joints and skeleton in an Open3D window.

Usage:
    python view_3d.py --sample 0
    python view_3d.py --sample 3 --ply-dir visualizations
"""

import argparse
import sys
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

parser = argparse.ArgumentParser()
parser.add_argument('--sample',  type=int, default=0,
                    help='Sample index to display (matches 3d_sample_XXXX.ply)')
parser.add_argument('--ply-dir', type=str, default='visualizations',
                    help='Directory containing the .ply files from visualize.py')
args = parser.parse_args()

ply_path = Path(args.ply_dir) / f'3d_sample_{args.sample:04d}.ply'

if not ply_path.exists():
    # List available PLY files to help the user pick a valid sample
    ply_dir = Path(args.ply_dir)
    available = sorted(ply_dir.glob('3d_sample_*.ply')) if ply_dir.exists() else []
    print(f'ERROR: file not found: {ply_path}')
    if available:
        indices = [int(p.stem.split('_')[-1]) for p in available]
        print(f'Available sample indices in "{args.ply_dir}": {indices}')
        print(f'Example: python view_3d.py --ply-dir {args.ply_dir} --sample {indices[0]}')
    else:
        print(f'No 3d_sample_*.ply files found in "{args.ply_dir}".')
        print('Run visualize.py first to generate them.')
    sys.exit(1)

pcd    = o3d.io.read_point_cloud(str(ply_path))
pts    = np.asarray(pcd.points)
colors = np.asarray(pcd.colors)

if len(pts) < 42:
    print(f'ERROR: expected 42 points (21 pred + 21 GT) in {ply_path}, '
          f'got {len(pts)}. The file may be corrupt — re-run visualize.py.')
    sys.exit(1)

# Rebuild LineSet in-memory — avoids non-standard PLY edge element
gt_conn     = [[s + 21, e + 21] for s, e in CONNECTIONS]
all_conn    = CONNECTIONS + gt_conn
bone_colors = (
    [colors[s].tolist() for s, e in CONNECTIONS] +
    [[0.6, 0.6, 0.6]] * len(CONNECTIONS)
)

ls        = o3d.geometry.LineSet()
ls.points = o3d.utility.Vector3dVector(pts)
ls.lines  = o3d.utility.Vector2iVector(np.array(all_conn, dtype=np.int32))
ls.colors = o3d.utility.Vector3dVector(np.array(bone_colors))

print(f'Loaded {ply_path}')
print(f'Sample {args.sample}  |  Predicted (coloured)  ·  GT (grey, offset right)')
print('Controls: left-drag=rotate  scroll=zoom  right-drag=pan  Q=quit')

vis = o3d.visualization.Visualizer()
vis.create_window(window_name=f'3D Pose — sample {args.sample}',
                  width=1280, height=720)
vis.add_geometry(pcd)
vis.add_geometry(ls)

opt = vis.get_render_option()
opt.background_color      = np.array([0.05, 0.05, 0.05])
opt.point_size            = 6.0
opt.line_width            = 2.0
opt.show_coordinate_frame = False

vis.run()
vis.destroy_window()
