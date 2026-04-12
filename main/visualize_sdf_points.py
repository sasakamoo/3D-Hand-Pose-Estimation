"""
visualize_sdf_points.py
========================
Visualise HOISDF hand SDF query points, matching the style in the paper figure:

    Left  : 2D projection onto the input image (skeleton + coloured SDF points)
    Centre: 3D scatter of SDF points coloured by SDF value, with skeleton + bbox
    Right : same scene from an alternative viewpoint

SDF points are coloured by their predicted (or GT-approximate) SDF value:
    blue  = near surface (|sdf| ~ 0)
    yellow/red = far from surface

Two modes
---------
--use_gt_sdf  (default)
    Uses the GT-approximate SDF values already stored in the dataset
    (targets['hand_sdf'] from freihand.py). No model needed, runs instantly.

--use_model_sdf (requires --model_path)
    Runs the full model forward pass and uses the *predicted* SDF values
    from hand_sdf_decoder on the sdf_infer grid points.  Shows what the
    model actually thinks the surface looks like.

Usage (run from HOISDF root)
-----------------------------
    # GT SDF points (no model needed):
    python main/visualize_sdf_points.py \\
        --freihand_dir /path/to/FreiHAND \\
        --save_dir     /path/to/output   \\
        --n_samples    8

    # Model-predicted SDF points:
    python main/visualize_sdf_points.py \\
        --freihand_dir /path/to/FreiHAND \\
        --model_path   /path/to/snapshot.pth.tar \\
        --use_model_sdf \\
        --save_dir     /path/to/output \\
        --n_samples    8

Outputs
-------
    sdf_vis/sample_XXXX.png   — 3-panel figure per sample
    sdf_overview.png          — all samples tiled
"""

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

# ── skeleton ──────────────────────────────────────────────────────────────────
CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
]
# finger colour palette (matches paper figure: magenta/yellow/cyan/green)
_FINGER_MPL = ['#ff44cc', '#ffdd00', '#00ddff', '#44ff44', '#ff8800']
BONE_MPL    = [_FINGER_MPL[f] for f in range(5) for _ in range(4)]
BONE_BGR    = [
    (200,  68, 255),  # magenta  — index
    (  0, 221, 255),  # yellow   — middle
    (255, 221,   0),  # cyan     — ring
    ( 68, 255,  68),  # green    — pinky
    (  0, 136, 255),  # orange   — thumb
]
BONE_BGR_LIST = [BONE_BGR[f] for f in range(5) for _ in range(4)]


# ── colour mapping ────────────────────────────────────────────────────────────

def sdf_to_rgba(sdf_vals, clamp=0.15, alpha=0.75):
    """
    Map SDF values → RGBA colours using a blue-cyan-yellow-red diverging map.

    sdf_vals : (N,) float  — signed distance, metres (can be clamped)
    Returns   : (N, 4) float in [0, 1]

    Colour convention that matches the paper figure:
        near 0  → bright cyan/teal  (on the surface)
        positive (outside, small) → yellow
        positive (outside, large) → red/orange
        negative (inside)  → dark blue
    """
    import matplotlib.pyplot as plt
    v = np.clip(sdf_vals, -clamp, clamp)
    # Normalise to [0, 1]:  -clamp → 0,  0 → 0.5,  +clamp → 1
    t = (v + clamp) / (2 * clamp)

    # Use 'RdYlBu_r': blue=inside, yellow=surface, red=far outside
    cmap = plt.get_cmap('RdYlBu_r')
    rgba = cmap(t).astype(np.float64)
    rgba[:, 3] = alpha
    return rgba


# ── 3D bounding box from joints ───────────────────────────────────────────────

def joints_bbox_lines(joints_rr):
    """
    Return list of (p1, p2) line segments for a tight axis-aligned bounding box
    around the joint cloud.  Scaled up slightly for visual clarity.

    joints_rr : (21, 3) root-relative metres
    Returns list of 12 pairs of (3,) float arrays.
    """
    mn = joints_rr.min(0) * 1.15
    mx = joints_rr.max(0) * 1.15
    # 8 corners
    corners = np.array([
        [mn[0], mn[1], mn[2]], [mx[0], mn[1], mn[2]],
        [mx[0], mx[1], mn[2]], [mn[0], mx[1], mn[2]],
        [mn[0], mn[1], mx[2]], [mx[0], mn[1], mx[2]],
        [mx[0], mx[1], mx[2]], [mn[0], mx[1], mx[2]],
    ])
    edges = [
        (0,1),(1,2),(2,3),(3,0),  # bottom face
        (4,5),(5,6),(6,7),(7,4),  # top face
        (0,4),(1,5),(2,6),(3,7),  # verticals
    ]
    return [(corners[a], corners[b]) for a, b in edges]


# ── 2D projection ─────────────────────────────────────────────────────────────

def project(pts_rr, mano_root, K):
    """
    pts_rr    : (N, 3) root-relative metres
    mano_root : (3,)   camera-space metres
    K         : (3, 3) camera intrinsics
    Returns   : (N, 2) image pixels
    """
    pts_cam = pts_rr + mano_root[None, :]
    proj    = (K @ pts_cam.T).T
    return proj[:, :2] / proj[:, 2:3].clip(min=1e-6)


# ── main visualisation ────────────────────────────────────────────────────────

def visualise_sdf(sid, img_tensor,
                  sdf_pts_rr, sdf_vals,
                  joints_rr,
                  mano_root, K,
                  inp_res, hm_res,
                  gt_2d_hm,
                  out_path):
    """
    3-panel figure replicating the paper figure style.

    Parameters
    ----------
    sid         : int     sample index (for title)
    img_tensor  : (3,H,W) float [0,1]
    sdf_pts_rr  : (N, 3)  SDF query points, root-relative metres
    sdf_vals    : (N,)    SDF values (metres, signed)
    joints_rr   : (21,3)  joint positions, root-relative metres
    mano_root   : (3,)    wrist in camera space metres
    K           : (3,3)   camera intrinsics
    inp_res     : int     input image resolution (256)
    hm_res      : int     heatmap resolution (128)
    gt_2d_hm    : (21,2)  GT 2D joints in heatmap pixels
    out_path    : str
    """
    import cv2 as _cv2
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D   # noqa: F401

    # ── colour-map SDF points ─────────────────────────────────────────────────
    rgba = sdf_to_rgba(sdf_vals)   # (N, 4)

    # ── 3D bbox lines ─────────────────────────────────────────────────────────
    bbox_segs = joints_bbox_lines(joints_rr)

    # ── 2D: project SDF points + joints to image space ─────────────────────
    sdf_2d     = project(sdf_pts_rr, mano_root, K)   # (N, 2) image pixels
    joints_2d  = project(joints_rr,  mano_root, K)   # (21,2) image pixels

    # Prepare image
    img_np  = (img_tensor.permute(1,2,0).numpy() * 255).astype(np.uint8)
    img_bgr = _cv2.cvtColor(img_np, _cv2.COLOR_RGB2BGR)
    img_bgr = _cv2.resize(img_bgr, (inp_res, inp_res))

    # ── figure layout ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(15, 5), facecolor='#0d0d0d')
    fig.suptitle(f'Sample {sid}  —  SDF query points  '
                 f'(blue=inside/near  →  red=far outside)',
                 color='#cccccc', fontsize=9, y=1.01)

    # ── Panel 1: 2D image with projected SDF points + skeleton ──────────────
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.set_facecolor('#0d0d0d')
    ax1.axis('off')
    ax1.set_title('2D projections\non input image',
                  color='#cccccc', fontsize=9, pad=4)

    ax1.imshow(_cv2.cvtColor(img_bgr, _cv2.COLOR_BGR2RGB))

    # SDF points projected to image
    ax1.scatter(sdf_2d[:, 0], sdf_2d[:, 1],
                c=rgba, s=4, linewidths=0, zorder=3, alpha=0.7)

    # Skeleton (GT 2D from heatmap → image space)
    scale = inp_res / hm_res
    gt_img = gt_2d_hm * scale
    for bi, (s, e) in enumerate(CONNECTIONS):
        ax1.plot([gt_img[s, 0], gt_img[e, 0]],
                 [gt_img[s, 1], gt_img[e, 1]],
                 color=BONE_MPL[bi], lw=1.5, alpha=0.9, zorder=4)
    ax1.scatter(gt_img[:, 0], gt_img[:, 1],
                c='white', s=12, zorder=5, linewidths=0)

    ax1.set_xlim(0, inp_res)
    ax1.set_ylim(inp_res, 0)   # y-flip to match image coords

    # ── Panel 2: 3D front view ────────────────────────────────────────────────
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    _style_3d(ax2, '3D output')

    _draw_sdf_3d(ax2, sdf_pts_rr, rgba, joints_rr, bbox_segs,
                 elev=20, azim=-55, pt_size=6)

    # ── Panel 3: alternative 3D view ─────────────────────────────────────────
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    _style_3d(ax3, 'Alternative view')

    _draw_sdf_3d(ax3, sdf_pts_rr, rgba, joints_rr, bbox_segs,
                 elev=10, azim=40, pt_size=6)

    # ── Colourbar (shared) ────────────────────────────────────────────────────
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    cax  = fig.add_axes([0.92, 0.15, 0.012, 0.7])
    norm = mcolors.Normalize(vmin=-0.15, vmax=0.15)
    sm   = cm.ScalarMappable(cmap='RdYlBu_r', norm=norm)
    sm.set_array([])
    cb   = fig.colorbar(sm, cax=cax)
    cb.set_label('SDF value (m)', color='#aaaaaa', fontsize=7)
    cb.ax.yaxis.set_tick_params(color='#aaaaaa', labelsize=6)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color='#aaaaaa')
    cb.outline.set_edgecolor('#444444')

    plt.tight_layout(rect=[0, 0, 0.91, 1])
    plt.savefig(out_path, dpi=130, bbox_inches='tight', facecolor='#0d0d0d')
    plt.close(fig)


def _style_3d(ax, title):
    ax.set_facecolor('#0d0d0d')
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor('#2a2a2a')
    ax.grid(False)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.xaxis.line.set_color('#2a2a2a')
    ax.yaxis.line.set_color('#2a2a2a')
    ax.zaxis.line.set_color('#2a2a2a')
    ax.set_title(title, color='#cccccc', fontsize=9, pad=2)


def _draw_sdf_3d(ax, sdf_pts, rgba, joints, bbox_segs, elev, azim, pt_size):
    """Draw SDF points, skeleton, and bounding box onto a 3D axes."""

    # ── SDF point cloud ───────────────────────────────────────────────────────
    ax.scatter(sdf_pts[:, 0], sdf_pts[:, 1], sdf_pts[:, 2],
               c=rgba, s=pt_size, linewidths=0, alpha=0.7, zorder=2)

    # ── Bounding box (cyan wireframe) ─────────────────────────────────────────
    for p1, p2 in bbox_segs:
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                color='#00eeff', lw=0.8, alpha=0.6, zorder=3)

    # ── Skeleton ──────────────────────────────────────────────────────────────
    for bi, (s, e) in enumerate(CONNECTIONS):
        ax.plot([joints[s,0], joints[e,0]],
                [joints[s,1], joints[e,1]],
                [joints[s,2], joints[e,2]],
                color=BONE_MPL[bi], lw=2.0, alpha=0.95, zorder=4)
    ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2],
               c='white', s=18, zorder=5, depthshade=False)

    # ── Axis limits (equal) ───────────────────────────────────────────────────
    all_pts = np.vstack([sdf_pts, joints])
    mid = all_pts.mean(0)
    r   = np.abs(all_pts - mid).max() * 1.1
    r   = max(r, 0.05)
    ax.set_xlim(mid[0]-r, mid[0]+r)
    ax.set_ylim(mid[1]-r, mid[1]+r)
    ax.set_zlim(mid[2]-r, mid[2]+r)
    ax.view_init(elev=elev, azim=azim)


def overview_grid(png_paths, out_path, n_cols=4):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    imgs = [plt.imread(p) for p in png_paths]
    n    = len(imgs)
    nc   = min(n_cols, n)
    nr   = (n + nc - 1) // nc
    fig, axes = plt.subplots(nr, nc, figsize=(nc * 5, nr * 2),
                             facecolor='#0d0d0d', squeeze=False)
    for ax, im in zip(axes.flatten(), imgs):
        ax.imshow(im); ax.axis('off')
    for ax in axes.flatten()[n:]:
        ax.axis('off')
    plt.tight_layout(pad=0.2)
    plt.savefig(out_path, dpi=100, bbox_inches='tight', facecolor='#0d0d0d')
    plt.close(fig)
    print(f'  Overview → {out_path}')


# ── model-based SDF inference (optional) ─────────────────────────────────────

def get_model_sdf_points(model, inputs_d, meta_d, device, cfg):
    """
    Run sdf_infer() through the model to get the near-surface point set
    with *predicted* SDF values. Returns (pts_rr_np, sdf_np) for sample 0.

    pts_rr_np : (N, 3) root-relative metres
    sdf_np    : (N,)   predicted SDF values (metres)
    """
    model.eval()
    with torch.no_grad():
        img_feat, enc_skip = model.backbone_net(inputs_d['img'])
        feature_pyramid, _ = model.decoder_net(img_feat, enc_skip)
        mano_root = meta_d['mano_root']
        cam_intr  = meta_d['cam_intr']
        bbox_hand = meta_d['bbox_hand']

        pts, sdf, _ = model.sdf_infer(
            feature_pyramid, mano_root, cam_intr,
            bbox_hand, cfg.hand_sdf_scale, cfg.num_samp_hand)

    # Take first sample in batch
    pts_scaled = pts[0].cpu().numpy()    # (N, 3) in SDF-scaled space
    sdf_np     = sdf[0, :, 0].cpu().numpy()  # (N,)

    # Convert from SDF-scaled space back to camera metres, then root-relative
    mano_root_np = meta_d['mano_root'][0].cpu().numpy()
    pts_cam  = pts_scaled / cfg.hand_sdf_scale + mano_root_np[None, :]
    pts_rr   = pts_cam - mano_root_np[None, :]   # root-relative
    return pts_rr, sdf_np


# ── args ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--freihand_dir', type=str, required=True)
    p.add_argument('--save_dir',     type=str,
                   default=os.path.join(_root_dir, 'sdf_vis_output'))
    p.add_argument('--n_samples',    type=int, default=8)
    p.add_argument('--seed',         type=int, default=42)
    p.add_argument('--split',        type=str, default='train',
                   choices=['train', 'evaluation'],
                   help='Use train split to get GT SDF targets')
    p.add_argument('--model_path',   type=str, default=None,
                   help='Checkpoint path. Required with --use_model_sdf.')
    p.add_argument('--use_model_sdf', action='store_true',
                   help='Use model-predicted SDF values instead of GT-approx.')
    p.add_argument('--gpu',          type=str, default='0')
    p.add_argument('--n_cols',       type=int, default=4,
                   help='Columns in overview grid')
    return p.parse_args()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    vis_dir = os.path.join(args.save_dir, 'sdf_vis')
    os.makedirs(vis_dir, exist_ok=True)

    # ── config ────────────────────────────────────────────────────────────────
    from main.config import cfg
    cfg.freihand_data_dir = args.freihand_dir
    cfg.set_args(args.gpu, 'sdf_vis', continue_train=False)
    cfg.create_run_dirs()
    cfg.calc_mutliscale_dim(cfg.use_big_decoder, cfg.resnet_type)
    cfg.num_samp_obj = 0

    inp_res = cfg.input_img_shape[0]   # 256
    hm_res  = cfg.output_hm_shape[0]  # 128

    # ── dataset ───────────────────────────────────────────────────────────────
    from freihand import Dataset as FreiHandDataset
    from torch.utils.data import DataLoader, Subset

    split = args.split
    if args.use_model_sdf and split == 'evaluation':
        print('  NOTE: evaluation split has no GT SDF, but --use_model_sdf '
              'will use model predictions anyway. Switching to train for GT joints.')

    ds = FreiHandDataset(split)
    print(f'  Dataset: {len(ds)} samples  (split={split})')

    random.seed(args.seed)
    idxs = random.sample(range(len(ds)), min(args.n_samples, len(ds)))
    print(f'  Samples: {sorted(idxs)}')

    # Batch size 1 so we can use model sdf_infer sample-by-sample easily
    loader = DataLoader(Subset(ds, idxs), batch_size=1,
                        shuffle=False, num_workers=0)

    # ── model (optional) ──────────────────────────────────────────────────────
    model = None
    if args.use_model_sdf:
        if args.model_path is None:
            raise ValueError('--use_model_sdf requires --model_path')
        from main.model import get_model
        model = get_model('test').to(device)
        ckpt  = torch.load(args.model_path, map_location='cpu')
        state = ckpt.get('network', ckpt.get('model_state', ckpt))
        state = {k.replace('module.', ''): v for k, v in state.items()}
        missing, _ = model.load_state_dict(state, strict=True)
        if missing:
            raise RuntimeError(f'Missing keys: {missing[:4]}')
        model.eval()
        print(f'  Model loaded: {args.model_path}')
        mode_label = 'model-predicted SDF'
    else:
        mode_label = 'GT-approximate SDF'

    print(f'  SDF source: {mode_label}')

    # ── process each sample ───────────────────────────────────────────────────
    png_paths = []

    for i, (inputs, targets, meta) in enumerate(loader):
        sid = idxs[i]

        inputs_d  = {k: v.to(device) if torch.is_tensor(v) else v
                     for k, v in inputs.items()}
        targets_d = {k: v.to(device) if torch.is_tensor(v) else v
                     for k, v in targets.items()}
        meta_d    = {k: v.to(device) if torch.is_tensor(v) else v
                     for k, v in meta.items()}

        mano_root_np = meta['mano_root'][0].numpy()   # (3,) cam metres
        K_np         = meta['cam_intr'][0].numpy()    # (3,3)

        # GT joints root-relative metres
        joints_mm    = targets['joint_cam_no_trans'][0].numpy()   # (21,3) mm
        joints_rr    = joints_mm / 1000.0                         # (21,3) m

        # 2D GT in heatmap pixels
        gt_2d_hm = targets['joint_coord'][0].numpy()   # (21,2)

        if args.use_model_sdf:
            # ── model-predicted SDF points ────────────────────────────────
            sdf_pts_rr, sdf_vals = get_model_sdf_points(
                model, inputs_d, meta_d, device, cfg)
        else:
            # ── GT-approximate SDF points from dataset ────────────────────
            # hand_sdf_points are in SDF-scaled space (scaled by hand_sdf_scale)
            # root-relative. Convert back to metres.
            sdf_pts_scaled = inputs['hand_sdf_points'][0].numpy()   # (N, 3)
            sdf_pts_rr     = sdf_pts_scaled / cfg.hand_sdf_scale    # (N, 3) m

            sdf_vals = targets['hand_sdf'][0].numpy()               # (N,)
            # hand_sdf from freihand.py is already in metres (approx dist)

        print(f'  [{i+1}/{len(idxs)}] sample {sid:5d}  '
              f'pts={sdf_pts_rr.shape[0]}  '
              f'sdf range=[{sdf_vals.min():.3f}, {sdf_vals.max():.3f}]')

        out_path = os.path.join(vis_dir, f'sample_{sid:05d}.png')
        visualise_sdf(
            sid        = sid,
            img_tensor = inputs['img'][0],
            sdf_pts_rr = sdf_pts_rr,
            sdf_vals   = sdf_vals,
            joints_rr  = joints_rr,
            mano_root  = mano_root_np,
            K          = K_np,
            inp_res    = inp_res,
            hm_res     = hm_res,
            gt_2d_hm   = gt_2d_hm,
            out_path   = out_path,
        )
        png_paths.append(out_path)

    # ── overview grid ─────────────────────────────────────────────────────────
    ov_path = os.path.join(args.save_dir, 'sdf_overview.png')
    overview_grid(png_paths, ov_path, n_cols=args.n_cols)
    print(f'\n  Done. Visualisations → {vis_dir}/')
    print(f'  Overview            → {ov_path}')


if __name__ == '__main__':
    main()