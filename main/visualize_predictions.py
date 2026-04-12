"""
visualize_predictions.py
=========================
Load a trained HOISDF checkpoint, run inference on N random FreiHAND
samples, and save per-sample visualisations + PLY point clouds.

Outputs (all inside --save_dir):
    sample_XXXX_vis.png      — 5-panel figure: GT 2D | Pred 2D | Overlay | 3D front | 3D side
    sample_XXXX_gt.ply       — GT 3D joint positions as a point cloud
    sample_XXXX_pred.ply     — Predicted 3D joint positions as a point cloud
    overview.png             — all samples tiled in one grid

Usage (run from HOISDF-main/):
    python main/visualize_predictions.py \\
        --model_path  /scratch/kghasemz/hoisdf_freihand/outputs/model_dump/freihand_run1/snapshot_70.pth.tar \\
        --freihand_dir /home/kghasemz/projects/def-vislearn/kghasemz/dataset \\
        --n_samples   16 \\
        --save_dir    /scratch/kghasemz/hoisdf_eval \\
        --seed        42
"""

# ── headless display BEFORE any other import ──────────────────────────────────
import os
os.environ['MPLBACKEND']   = 'Agg'
os.environ['DISPLAY']      = ''
os.environ['MPLCONFIGDIR'] = '/tmp'

import sys, argparse, random
import numpy as np
import torch

# ── path setup ────────────────────────────────────────────────────────────────
_file_dir = os.path.dirname(os.path.abspath(__file__))
_root_dir = os.path.dirname(_file_dir)
for _p in [_root_dir, _file_dir,
           os.path.join(_root_dir, 'common'),
           os.path.join(_root_dir, 'data')]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── skeleton topology ─────────────────────────────────────────────────────────
CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
]
_FINGER_BGR = [(0,220,0),(255,100,0),(0,200,255),(0,100,255),(200,0,200)]
_FINGER_MPL = ['#00dd00','#0064ff','#00c8ff','#ff6400','#c800c8']
BONE_BGR    = [c for c in _FINGER_BGR for _ in range(4)]
BONE_MPL    = [c for c in _FINGER_MPL for _ in range(4)]

# joint colours for PLY (one RGB per joint, cycling through finger colours)
_JOINT_RGB_FINGER = [
    (0, 220, 0),    # thumb
    (255, 100, 0),  # index
    (0, 200, 255),  # middle
    (0, 100, 255),  # ring
    (200, 0, 200),  # pinky
]
JOINT_RGB = [(128,128,128)]  # wrist = grey
for fi in range(5):
    r,g,b = _JOINT_RGB_FINGER[fi]
    for _ in range(4):
        JOINT_RGB.append((r,g,b))


# ─────────────────────────────────────────────────────────────────────────────
# Drawing helpers
# ─────────────────────────────────────────────────────────────────────────────

def draw_2d(img_bgr, kpts, bone_color=None, jcolor=(255,255,255), thick=2, r=4):
    import cv2
    vis = img_bgr.copy()
    for bi,(s,e) in enumerate(CONNECTIONS):
        c  = bone_color if bone_color else BONE_BGR[bi]
        p1 = tuple(np.clip(kpts[s].astype(int), 0, 4096))
        p2 = tuple(np.clip(kpts[e].astype(int), 0, 4096))
        cv2.line(vis, p1, p2, c, thick, cv2.LINE_AA)
    for ji, pt in enumerate(kpts):
        r_,g_,b_ = JOINT_RGB[ji]
        cv2.circle(vis, tuple(np.clip(pt.astype(int), 0, 4096)),
                   r, (b_,g_,r_), -1, cv2.LINE_AA)
    return vis


def draw_3d(ax, joints, bone_colors, lw=1.5, alpha=0.9,
            jcolor='white', jsize=20, label=None):
    for bi,(s,e) in enumerate(CONNECTIONS):
        ax.plot([joints[s,0],joints[e,0]],
                [joints[s,1],joints[e,1]],
                [joints[s,2],joints[e,2]],
                color=bone_colors[bi], lw=lw, alpha=alpha,
                label=label if bi==0 else None)
    ax.scatter(joints[:,0], joints[:,1], joints[:,2],
               c=jcolor, s=jsize, zorder=5)


def style_3d(ax):
    ax.set_facecolor('#111111')
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor('#333333')
    ax.grid(False)
    ax.tick_params(colors='#555555', labelsize=5)


# ─────────────────────────────────────────────────────────────────────────────
# PLY export
# ─────────────────────────────────────────────────────────────────────────────

def save_ply(path, joints_xyz, label='hand'):
    """
    Save 21 joint positions as a PLY file.

    Each joint becomes a vertex coloured by finger.
    Edges (bones) are written as 'line' elements so viewers like MeshLab
    can display the skeleton.

    joints_xyz : (21, 3) float  in any consistent unit (metres recommended)
    """
    n_verts = joints_xyz.shape[0]
    n_edges = len(CONNECTIONS)

    lines = []
    lines.append('ply')
    lines.append('format ascii 1.0')
    lines.append(f'comment {label} skeleton — HOISDF hand-only')
    lines.append(f'element vertex {n_verts}')
    lines.append('property float x')
    lines.append('property float y')
    lines.append('property float z')
    lines.append('property uchar red')
    lines.append('property uchar green')
    lines.append('property uchar blue')
    lines.append(f'element edge {n_edges}')
    lines.append('property int vertex1')
    lines.append('property int vertex2')
    lines.append('end_header')

    for ji in range(n_verts):
        x, y, z = joints_xyz[ji]
        r, g, b = JOINT_RGB[ji]
        lines.append(f'{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}')

    for s, e in CONNECTIONS:
        lines.append(f'{s} {e}')

    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


# ─────────────────────────────────────────────────────────────────────────────
# Per-sample visualisation
# ─────────────────────────────────────────────────────────────────────────────

def visualise_sample(sid, img_tensor, gt_2d, pred_2d,
                     gt_3d, pred_3d, err_mm, out_path):
    """
    5-panel figure:
        [GT 2D] [Pred 2D] [Overlay] [3D Front] [3D Side]

    img_tensor : (3,H,W) float in [0,1]
    gt_2d      : (21,2) in heatmap pixels
    pred_2d    : (21,2) in heatmap pixels
    gt_3d      : (21,3) root-relative metres
    pred_3d    : (21,3) root-relative metres
    err_mm     : scalar float
    """
    import cv2
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D   # noqa: F401

    hm_res  = gt_2d[:,0].max() + 1 if gt_2d.max() > 10 else 128
    inp_res = img_tensor.shape[-1]

    # Convert tensor to uint8 BGR for cv2
    img_np  = (img_tensor.permute(1,2,0).numpy() * 255).astype(np.uint8)
    img_bgr = img_np[:,:,::-1].copy()

    # Scale 2D coords from heatmap space → image space
    scale = inp_res / hm_res
    gt_2d_img   = gt_2d   * scale
    pred_2d_img = pred_2d * scale

    gt_panel   = draw_2d(img_bgr, gt_2d_img,   bone_color=None,     jcolor=(200,200,200))
    pred_panel = draw_2d(img_bgr, pred_2d_img, bone_color=None,     jcolor=(50,255,50))
    overlay    = draw_2d(img_bgr, gt_2d_img,   bone_color=(80,80,80), jcolor=(200,200,200))
    overlay    = draw_2d(overlay,  pred_2d_img, bone_color=None,     jcolor=(50,255,50))

    fig = plt.figure(figsize=(18, 4), facecolor='#0a0a0a')
    fig.suptitle(f'Sample {sid}   MJE = {err_mm:.1f} mm',
                 color='white', fontsize=10, y=1.01)

    # 2D panels
    for col, (panel, title) in enumerate([
        (gt_panel,   'GT 2D'),
        (pred_panel, f'Pred 2D  {err_mm:.1f}mm'),
        (overlay,    'Overlay'),
    ]):
        ax = fig.add_subplot(1, 5, col+1)
        ax.imshow(panel[:,:,::-1])    # BGR→RGB
        ax.set_title(title, color='white', fontsize=8)
        ax.axis('off')

    # 3D panels (front + side)
    for col, (elev, azim, title) in enumerate([
        (10,  -60, '3D Front'),
        (10,  30,  '3D Side'),
    ], start=3):
        ax3 = fig.add_subplot(1, 5, col+1, projection='3d')
        ax3.set_facecolor('#111111')
        style_3d(ax3)

        draw_3d(ax3, gt_3d,   BONE_MPL, lw=1.0, alpha=0.5,
                jcolor='#aaaaaa', jsize=10, label='GT')
        draw_3d(ax3, pred_3d, BONE_MPL, lw=2.0, alpha=0.95,
                jcolor='#00ff50', jsize=20, label='Pred')

        # Equal axes
        all_pts = np.vstack([gt_3d, pred_3d])
        c = all_pts.mean(0)
        r = np.abs(all_pts - c).max() * 1.2
        ax3.set_xlim(c[0]-r, c[0]+r)
        ax3.set_ylim(c[1]-r, c[1]+r)
        ax3.set_zlim(c[2]-r, c[2]+r)
        ax3.view_init(elev=elev, azim=azim)
        ax3.set_title(title, color='white', fontsize=8)
        if col == 3:
            ax3.legend(fontsize=6, facecolor='#222222',
                       labelcolor='white', loc='upper left')

    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Overview grid
# ─────────────────────────────────────────────────────────────────────────────

def overview_grid(png_paths, out_path, n_cols=4):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image

    imgs = [np.array(Image.open(p)) for p in png_paths]
    n    = len(imgs)
    nrow = (n + n_cols - 1) // n_cols
    h, w = imgs[0].shape[:2]

    fig, axes = plt.subplots(nrow, n_cols,
                             figsize=(n_cols * 4.5, nrow * 1.5),
                             facecolor='#0a0a0a')
    axes = np.array(axes).reshape(-1)
    for ax, img in zip(axes, imgs):
        ax.imshow(img);  ax.axis('off')
    for ax in axes[n:]:
        ax.axis('off')
    plt.tight_layout(pad=0.3)
    fig.savefig(out_path, dpi=100, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'  Overview → {out_path}')


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model_path',   type=str, required=True,
                   help='Path to snapshot .pth.tar checkpoint')
    p.add_argument('--freihand_dir', type=str, required=True,
                   help='FreiHAND dataset root')
    p.add_argument('--n_samples',   type=int, default=16,
                   help='Number of random samples to visualise')
    p.add_argument('--save_dir',    type=str,
                   default=os.path.join(_root_dir, 'eval_output'),
                   help='Where to write output files')
    p.add_argument('--seed',        type=int, default=42,
                   help='Random seed for sample selection')
    p.add_argument('--split',       type=str, default='evaluation',
                   choices=['train','evaluation'],
                   help='Dataset split to sample from')
    p.add_argument('--batch_size',  type=int, default=8,
                   help='Inference batch size')
    return p.parse_args()


def main():
    args = parse_args()

    # ── device ────────────────────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # ── output dirs ───────────────────────────────────────────────────────────
    vis_dir = os.path.join(args.save_dir, 'visualisations')
    ply_dir = os.path.join(args.save_dir, 'ply')
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(ply_dir, exist_ok=True)

    # ── dataset ───────────────────────────────────────────────────────────────
    from main.config import cfg
    cfg.freihand_data_dir = args.freihand_dir
    cfg.calc_mutliscale_dim(cfg.use_big_decoder, cfg.resnet_type)

    from freihand import Dataset as FreiHandDataset
    ds = FreiHandDataset(args.split)
    print(f'Dataset: {len(ds)} samples  (split={args.split})')

    # Random sample indices
    random.seed(args.seed)
    np.random.seed(args.seed)
    indices = sorted(random.sample(range(len(ds)), min(args.n_samples, len(ds))))
    print(f'Selected {len(indices)} samples: {indices[:8]}{"..." if len(indices)>8 else ""}')

    from torch.utils.data import DataLoader, Subset
    subset = Subset(ds, indices)
    loader = DataLoader(subset, batch_size=args.batch_size,
                        shuffle=False, num_workers=4, pin_memory=True)

    # ── model ─────────────────────────────────────────────────────────────────
    print(f'\nLoading checkpoint: {args.model_path}')
    from main.model import get_model
    model = get_model('test').to(device)

    ckpt = torch.load(args.model_path, map_location=device)
    # Handle different checkpoint formats
    if 'network' in ckpt:
        state = ckpt['network']
    elif 'model_state' in ckpt:
        state = ckpt['model_state']
    elif 'state_dict' in ckpt:
        state = ckpt['state_dict']
    else:
        state = ckpt

    # Strip DataParallel prefix if present
    state = {k.replace('module.',''):v for k,v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f'  Missing keys  ({len(missing)}): {missing[:3]}...')
    if unexpected:
        print(f'  Unexpected    ({len(unexpected)}): {unexpected[:3]}...')
    model.eval()
    print('  Model loaded OK')

    # ── inference ─────────────────────────────────────────────────────────────
    inp_res = cfg.input_img_shape[0]   # 256
    hm_res  = cfg.output_hm_shape[0]  # 128

    all_pred_j21  = []   # (N,21,3) root-relative metres
    all_gt_j21    = []   # (N,21,3) root-relative metres
    all_gt_2d     = []   # (N,21,2) heatmap pixels
    all_mano_root = []   # (N,3)    camera-space metres
    all_cam_intr  = []   # (N,3,3)
    all_img       = []   # (N,3,H,W)

    print('\nRunning inference ...')
    with torch.no_grad():
        for batch_i, (inputs, targets, meta) in enumerate(loader):
            # Move to device
            inputs_d = {k: v.to(device) if torch.is_tensor(v) else v
                        for k,v in inputs.items()}
            targets_d = {k: v.to(device) if torch.is_tensor(v) else v
                         for k,v in targets.items()}
            meta_d    = {k: v.to(device) if torch.is_tensor(v) else v
                         for k,v in meta.items()}

            out = model(inputs_d, targets_d, meta_d, 'eval', epoch_cnt=1e8)

            pred_j20 = out['hand_joints_out'].cpu()   # (B,20,3) rr metres
            B = pred_j20.shape[0]
            # Prepend wrist (root) as zeros
            wrist = torch.zeros(B, 1, 3)
            pred_j21 = torch.cat([wrist, pred_j20], dim=1)  # (B,21,3)

            all_pred_j21.append(pred_j21.numpy())
            all_gt_j21.append(
                (targets['joint_cam_no_trans'] / 1000).numpy())   # mm→m
            all_gt_2d.append(targets['joint_coord'].numpy())
            all_mano_root.append(meta['mano_root'].numpy())
            all_cam_intr.append(meta['cam_intr'].numpy())
            all_img.append(inputs['img'].numpy())

            done = min((batch_i+1)*args.batch_size, len(indices))
            print(f'  {done}/{len(indices)} samples processed', end='\r')

    print()

    # Concatenate all batches
    all_pred_j21  = np.concatenate(all_pred_j21,  axis=0)
    all_gt_j21    = np.concatenate(all_gt_j21,    axis=0)
    all_gt_2d     = np.concatenate(all_gt_2d,     axis=0)
    all_mano_root = np.concatenate(all_mano_root, axis=0)
    all_cam_intr  = np.concatenate(all_cam_intr,  axis=0)
    all_img       = np.concatenate(all_img,       axis=0)

    # ── per-sample MJE ────────────────────────────────────────────────────────
    pred_rr = all_pred_j21 - all_pred_j21[:,0:1]
    gt_rr   = all_gt_j21   - all_gt_j21[:,0:1]
    per_mje = np.linalg.norm(pred_rr - gt_rr, axis=-1).mean(axis=-1) * 1000
    mean_mje = per_mje.mean()
    print(f'\nMean MJE over {len(indices)} samples: {mean_mje:.1f} mm')

    # ── save ──────────────────────────────────────────────────────────────────
    print('\nSaving visualisations and PLY files ...')
    png_paths = []

    for si in range(len(indices)):
        sid = indices[si]

        pred_j21_i = all_pred_j21[si]   # (21,3) rr metres
        gt_j21_i   = all_gt_j21[si]     # (21,3) rr metres
        mano_root  = all_mano_root[si]  # (3,)   cam metres
        K          = all_cam_intr[si]   # (3,3)

        # ── 2D projection of predicted joints ─────────────────────────────
        pred_cam = pred_j21_i + mano_root[None, :]          # cam-space m
        proj     = (K @ pred_cam.T).T                        # (21,3)
        depth    = proj[:, 2:3].clip(min=1e-6)
        pred_2d  = proj[:, :2] / depth                      # inp_res pixels
        pred_2d_hm = pred_2d / inp_res * hm_res             # hm pixels

        err_mm = float(per_mje[si])

        # ── visualisation PNG ─────────────────────────────────────────────
        png_path = os.path.join(vis_dir, f'sample_{sid:05d}_vis.png')
        visualise_sample(
            sid        = sid,
            img_tensor = torch.from_numpy(all_img[si]),
            gt_2d      = all_gt_2d[si],
            pred_2d    = pred_2d_hm,
            gt_3d      = gt_j21_i,
            pred_3d    = pred_j21_i,
            err_mm     = err_mm,
            out_path   = png_path,
        )
        png_paths.append(png_path)

        # ── PLY export ────────────────────────────────────────────────────
        # Root-relative metres (good for comparison)
        gt_rr_i   = gt_j21_i   - gt_j21_i[0:1]
        pred_rr_i = pred_j21_i - pred_j21_i[0:1]

        ply_gt   = os.path.join(ply_dir, f'sample_{sid:05d}_gt.ply')
        ply_pred = os.path.join(ply_dir, f'sample_{sid:05d}_pred.ply')
        save_ply(ply_gt,   gt_rr_i,   label='gt')
        save_ply(ply_pred, pred_rr_i, label='pred')

        print(f'  [{si+1:>3}/{len(indices)}]  sample {sid:5d}  '
              f'MJE={err_mm:6.1f}mm  → {os.path.basename(png_path)}')

    # ── summary stats ─────────────────────────────────────────────────────────
    print(f'\n{"─"*55}')
    print(f'  Samples evaluated : {len(indices)}')
    print(f'  Mean MJE          : {mean_mje:.1f} mm')
    print(f'  Median MJE        : {np.median(per_mje):.1f} mm')
    print(f'  Best  MJE         : {per_mje.min():.1f} mm  (sample {indices[per_mje.argmin()]})')
    print(f'  Worst MJE         : {per_mje.max():.1f} mm  (sample {indices[per_mje.argmax()]})')
    print(f'{"─"*55}')

    # ── overview grid ─────────────────────────────────────────────────────────
    ov_path = os.path.join(args.save_dir, 'overview.png')
    overview_grid(png_paths, ov_path, n_cols=min(4, len(indices)))

    # ── save MJE per sample as CSV ─────────────────────────────────────────────
    csv_path = os.path.join(args.save_dir, 'mje_per_sample.csv')
    with open(csv_path, 'w') as f:
        f.write('sample_idx,mje_mm\n')
        for si, idx in enumerate(indices):
            f.write(f'{idx},{per_mje[si]:.3f}\n')
    print(f'  MJE CSV  → {csv_path}')
    print(f'  PLY dir  → {ply_dir}/')
    print(f'  Vis dir  → {vis_dir}/')
    print(f'  Overview → {ov_path}')


if __name__ == '__main__':
    main()