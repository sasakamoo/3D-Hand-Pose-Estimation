import argparse, numpy as np, open3d as o3d
CONNECTIONS = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],
               [0,9],[9,10],[10,11],[11,12],[0,13],[13,14],[14,15],[15,16],
               [0,17],[17,18],[18,19],[19,20]]
parser = argparse.ArgumentParser()
parser.add_argument("--sample", type=int, default=0)
args = parser.parse_args()
i = args.sample
pcd = o3d.io.read_point_cloud(f"3d_sample_{i:04d}.ply")
pts = np.asarray(pcd.points)   # (42,3): 0-20=pred, 21-41=GT
colors = np.asarray(pcd.colors)
# Rebuild LineSet in-memory — avoids non-standard PLY edge element
gt_conn = [[s+21, e+21] for s,e in CONNECTIONS]
all_conn = CONNECTIONS + gt_conn
bone_colors = [colors[s].tolist() for s,e in CONNECTIONS] + [[0.6,0.6,0.6]]*len(CONNECTIONS)
ls = o3d.geometry.LineSet()
ls.points = o3d.utility.Vector3dVector(pts)
ls.lines  = o3d.utility.Vector2iVector(np.array(all_conn, dtype=np.int32))
ls.colors = o3d.utility.Vector3dVector(np.array(bone_colors))
print(f"Sample {i}  |  LEFT=Predicted (coloured)  RIGHT=GT (grey)")
print("Controls: left-drag=rotate  scroll=zoom  right-drag=pan  Q=quit")
o3d.visualization.draw_geometries([pcd, ls],
    window_name=f"3D Pose — sample {i}",
    width=1024, height=768)
