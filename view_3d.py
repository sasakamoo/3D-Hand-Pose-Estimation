# -*- coding: utf-8 -*-
"""
view_3d.py -- Interactive 3D hand pose viewer
==============================================
Reads the .ply files saved by visualize.py and displays predicted vs GT
joints and skeleton in an Open3D window.

Usage:
    python view_3d.py --sample 0
    python view_3d.py --sample 3 --ply-dir eval_results
"""

import argparse
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

ply_path = f'{args.ply_dir}/3d_sample_{args.sample:04d}.ply'

pcd    = o3d.io.read_point_cloud(ply_path)
pts    = np.asarray(pcd.points)    # (42, 3): 0-20 = pred, 21-41 = GT
colors = np.asarray(pcd.colors)

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

print(f'Sample {args.sample}  |  Predicted (coloured)  GT (grey)')
print('Controls: left-drag=rotate  scroll=zoom  right-drag=pan  Q=quit')

o3d.visualization.draw_geometries(
    [pcd, ls],
    window_name=f'3D Pose -- sample {args.sample}',
    width=1024,
    height=768,
)
