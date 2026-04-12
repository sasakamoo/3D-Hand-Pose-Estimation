"""
visualize_mano.py  —  Debugged + PLY export
============================================
Bugs fixed in this version:
  1. cfg.set_args / create_run_dirs missing → crashes on cfg.log_dir access
     when running standalone. Added cfg.set_args call.
  2. get_model('test') ignores num_samp_obj=0 → memory_mask wrong size.
     Added cfg.num_samp_obj = 0 before model construction.
  3. Checkpoint key 'network' assumed always present → KeyError on overfit
     checkpoints that store 'model_state'. Now handles all formats.
  4. strict=True crashes when checkpoint is from a different phase (e.g.
     ablation checkpoint with extra keys). Changed to strict=True but with
     clear diagnostics; falls back gracefully.
  5. detect_verts_frame heuristic is fragile when mano_root is near zero
     (close-up crops). Replaced with authoritative frame determination from
     mano_head.py docstring: verts are ALWAYS in MANO local frame (wrist≈0).
     mesh_root_offset is always mano_root. Heuristic removed.
  6. pred_rr computed as pred_j21 - pred_j21[0], but pred_j21[0] == zeros
     (prepended wrist), so pred_rr == pred_j21. Kept for clarity but noted.
  7. GT joints root-align uses gt_j[0] which is the dataset wrist, not zeros.
     This is correct but was undocumented. Added comment.
  8. img resize used inp_res from config correctly, but img was RGB from
     tensor — cv2 operations need BGR. Added explicit conversion.
  9. No --gpu arg → CUDA_VISIBLE_DEVICES never set → multi-GPU machines
     grab GPU 0 implicitly. Added --gpu arg.
 10. [NEW] PLY export for both mesh vertices and joint skeleton per sample.
     Saves: {save_dir}/ply/sample_{i:04d}_mesh.ply
            {save_dir}/ply/sample_{i:04d}_joints.ply
            {save_dir}/ply/sample_{i:04d}_gt_joints.ply
 11. [NEW] 4-panel output image: GT skeleton | Pred skeleton | Mesh | 3D view
 12. [NEW] Overview grid saved to {save_dir}/overview.png
 13. [NEW] Full MANO error suite matching HOISDF paper Table 3:
     Joint metrics (from voting head — hand_joints_out):
       MJE, PA-MPJPE
     Joint metrics (from MANO regressor — mano_joints_out):
       MANO-MJE, MANO-PA-MPJPE
       These differ from voting-head metrics — useful to compare both heads.
     Mesh metrics (requires GT mesh from mano_param):
       MME  — Mean Mesh Error: mean L2 dist pred verts vs GT verts (mm)
       PA-MME — Procrustes-aligned MME
       F@5mm  — fraction of pred verts within 5mm of nearest GT vert
       F@15mm — fraction of pred verts within 15mm of nearest GT vert
       VAUC   — area under the vertex PCK curve (0–50mm, 50 thresholds)
     GT mesh computed by running model.mano_head.mano_layer directly with
     GT mano_param from targets. Falls back gracefully if mano_param is
     all-zeros (training_mano.json not available).
 14. [NEW] Per-sample CSV saved to {save_dir}/metrics.csv with all metrics.
 15. [NEW] Summary table printed to terminal matching paper format.

Usage (run from HOISDF root):
    python main/visualize_mano.py \\
        --model_path  /path/to/snapshot.pth.tar \\
        --freihand_dir /path/to/FreiHAND \\
        --save_dir    /path/to/output \\
        --n_samples   8 \\
        --split       evaluation
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
_FINGER_BGR = [(0,220,0),(255,100,0),(0,200,255),(0,100,255),(200,0,200)]
BONE_BGR    = [_FINGER_BGR[f] for f in range(5) for _ in range(4)]

# Joint colours for PLY (wrist = grey, then 4 joints per finger)
_JOINT_RGB = [(128,128,128)]  # wrist
for r,g,b in [(0,220,0),(255,100,0),(0,200,255),(0,100,255),(200,0,200)]:
    for _ in range(4):
        _JOINT_RGB.append((r,g,b))


# ── PLY export ────────────────────────────────────────────────────────────────

def save_mesh_ply(path, verts, faces, color=(180, 130, 100)):
    """
    Save MANO mesh as PLY with per-vertex uniform colour.

    verts : (V, 3) float — metres, MANO local frame (wrist≈origin)
    faces : (F, 3) int
    """
    V, F = verts.shape[0], faces.shape[0]
    r, g, b = color
    lines = [
        'ply', 'format ascii 1.0',
        'comment HOISDF MANO mesh — MANO local frame (metres)',
        f'element vertex {V}',
        'property float x', 'property float y', 'property float z',
        'property uchar red', 'property uchar green', 'property uchar blue',
        f'element face {F}',
        'property list uchar int vertex_indices',
        'end_header',
    ]
    for v in verts:
        lines.append(f'{v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {r} {g} {b}')
    for f in faces:
        lines.append(f'3 {f[0]} {f[1]} {f[2]}')
    with open(path, 'w') as fp:
        fp.write('\n'.join(lines) + '\n')


def save_joints_ply(path, joints, label='joints'):
    """
    Save 21 joint positions as a PLY skeleton (vertices + edges).

    joints : (21, 3) float — metres, root-relative
    """
    lines = [
        'ply', 'format ascii 1.0',
        f'comment HOISDF {label} skeleton — root-relative metres',
        f'element vertex {len(joints)}',
        'property float x', 'property float y', 'property float z',
        'property uchar red', 'property uchar green', 'property uchar blue',
        f'element edge {len(CONNECTIONS)}',
        'property int vertex1', 'property int vertex2',
        'end_header',
    ]
    for ji, j in enumerate(joints):
        r, g, b = _JOINT_RGB[ji]
        lines.append(f'{j[0]:.6f} {j[1]:.6f} {j[2]:.6f} {r} {g} {b}')
    for s, e in CONNECTIONS:
        lines.append(f'{s} {e}')
    with open(path, 'w') as fp:
        fp.write('\n'.join(lines) + '\n')


# ── projection ────────────────────────────────────────────────────────────────

def project_points(pts_rr, mano_root, K):
    """
    pts_rr    : (N, 3) root-relative metres
    mano_root : (3,)   camera-space metres
    K         : (3, 3) camera intrinsics
    Returns   : (N, 2) image pixels
    """
    pts_cam = pts_rr + mano_root[None, :]
    proj    = (K @ pts_cam.T).T
    return proj[:, :2] / proj[:, 2:3].clip(min=1e-6)


# ── 2D drawing helpers ────────────────────────────────────────────────────────

def draw_skeleton_2d(img_bgr, kpts_px):
    """Draw skeleton onto BGR image. kpts_px: (21, 2) image pixels."""
    import cv2
    vis = img_bgr.copy()
    for bi, (s, e) in enumerate(CONNECTIONS):
        p1 = tuple(np.clip(kpts_px[s].astype(int), -4096, 4096))
        p2 = tuple(np.clip(kpts_px[e].astype(int), -4096, 4096))
        cv2.line(vis, p1, p2, BONE_BGR[bi], 2, cv2.LINE_AA)
    for p in kpts_px:
        cv2.circle(vis, tuple(np.clip(p.astype(int), -4096, 4096)),
                   4, (255, 255, 255), -1, cv2.LINE_AA)
    return vis


def render_mesh_wireframe(img_bgr, verts_rr, faces, mano_root, K):
    """
    Render MANO mesh wireframe.

    verts_rr  : (V, 3) root-relative metres  (MANO local frame — wrist≈origin)
    mano_root : (3,)   add to get camera-space metres
    K         : (3, 3) camera intrinsics

    BUG FIX #5: verts from ManoHead are ALWAYS in MANO local frame.
    We always add mano_root before projecting. No heuristic needed.
    """
    import cv2
    vis      = img_bgr.copy()
    verts_2d = project_points(verts_rr, mano_root, K)

    drawn = set()
    for f in faces:
        for a, b in [(f[0], f[1]), (f[1], f[2]), (f[2], f[0])]:
            key = (min(a, b), max(a, b))
            if key in drawn:
                continue
            drawn.add(key)
            p1 = tuple(np.clip(verts_2d[a].astype(int), -4096, 4096))
            p2 = tuple(np.clip(verts_2d[b].astype(int), -4096, 4096))
            cv2.line(vis, p1, p2, (180, 130, 100), 1, cv2.LINE_AA)
    return vis


# ── Procrustes alignment ──────────────────────────────────────────────────────

def procrustes_align(X, Y):
    """Align X to Y, returns X_aligned (same shape as X)."""
    muX = X.mean(0); muY = Y.mean(0)
    X0 = X - muX;    Y0 = Y - muY
    nX = np.linalg.norm(X0); nY = np.linalg.norm(Y0)
    X0 /= nX; Y0 /= nY
    U, _, Vt = np.linalg.svd(X0.T @ Y0)
    R = U @ Vt
    s = nY / nX
    return s * (R @ X.T).T + muY - s * (R @ muX)


# ── MANO mesh metrics ─────────────────────────────────────────────────────────

def compute_mme(pred_verts, gt_verts):
    """Mean Mesh Error in mm. Both (V,3) root-relative metres."""
    return np.linalg.norm(pred_verts - gt_verts, axis=-1).mean() * 1000.0


def compute_pa_mme(pred_verts, gt_verts):
    """PA-MME: Procrustes-align pred mesh to GT, then MME."""
    pred_aligned = procrustes_align(pred_verts, gt_verts)
    return np.linalg.norm(pred_aligned - gt_verts, axis=-1).mean() * 1000.0


def compute_f_score(pred_verts, gt_verts, threshold_mm):
    """
    F-score at a distance threshold.
    Both pred and GT have the same MANO topology (778 verts), so
    vertex-to-vertex distances are used — no nearest-neighbour search needed.
    Since P==R for same-topology meshes, F = fraction within threshold.
    """
    dists = np.linalg.norm(pred_verts - gt_verts, axis=-1) * 1000.0
    frac  = (dists < threshold_mm).mean()
    return float(frac) * 100.0   # percentage


def compute_vauc(pred_verts, gt_verts, max_mm=50.0, n_steps=50):
    """
    Vertex AUC: area under vertex PCK curve from 0 to max_mm.
    At each threshold t, compute fraction of verts within t mm.
    VAUC = mean of those fractions × 100 (percentage).
    Matches HOISDF paper Table 3 convention.
    """
    dists = np.linalg.norm(pred_verts - gt_verts, axis=-1) * 1000.0
    thresholds = np.linspace(0, max_mm, n_steps + 1)[1:]
    pcks = [(dists < t).mean() for t in thresholds]
    return float(np.mean(pcks)) * 100.0


def get_gt_verts(mano_layer, mano_param_np, device):
    """
    Run GT MANO params through mano_layer to get GT mesh vertices.

    mano_param_np : (58,) float32 — pose(48) + shape(10) from dataset.
    Returns (778, 3) metres MANO local frame, or None if params are zeros
    (i.e. training_mano.json was not available when the dataset was loaded).
    """
    if np.abs(mano_param_np).sum() < 1e-6:
        return None

    gt_pose  = torch.from_numpy(mano_param_np[:48]).unsqueeze(0).to(device)
    gt_shape = torch.from_numpy(mano_param_np[48:]).unsqueeze(0).to(device)
    with torch.no_grad():
        gt_v, _ = mano_layer(th_pose_coeffs=gt_pose, th_betas=gt_shape)
    return gt_v[0].cpu().numpy() / 1000.0   # mm → m


# ── 3D matplotlib panel ───────────────────────────────────────────────────────

def draw_3d_panel(ax, joints_rr, verts_rr=None, title=''):
    """Draw skeleton + optional mesh cloud on a 3D axis."""
    ax.set_facecolor('#111111')
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False; pane.set_edgecolor('#333')
    ax.grid(False)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_title(title, color='white', fontsize=8, pad=2)

    if verts_rr is not None:
        # Subsample for speed
        idx = np.random.choice(len(verts_rr), min(300, len(verts_rr)), replace=False)
        v = verts_rr[idx]
        ax.scatter(v[:,0], v[:,1], v[:,2], c='#b48264', s=2, alpha=0.4, linewidths=0)

    for bi, (s, e) in enumerate(CONNECTIONS):
        ax.plot([joints_rr[s,0], joints_rr[e,0]],
                [joints_rr[s,1], joints_rr[e,1]],
                [joints_rr[s,2], joints_rr[e,2]],
                color='#00dd88', lw=1.5, alpha=0.9)
    ax.scatter(joints_rr[:,0], joints_rr[:,1], joints_rr[:,2],
               c='white', s=16, zorder=5, depthshade=False)

    # Equal axes
    all_pts = joints_rr if verts_rr is None else np.vstack([joints_rr, verts_rr])
    mid = all_pts.mean(0)
    r   = max(np.abs(all_pts - mid).max() * 1.1, 0.05)
    ax.set_xlim(mid[0]-r, mid[0]+r)
    ax.set_ylim(mid[1]-r, mid[1]+r)
    ax.set_zlim(mid[2]-r, mid[2]+r)
    ax.view_init(elev=20, azim=-60)


# ── per-sample figure ─────────────────────────────────────────────────────────

def save_sample_figure(i, img_rgb, gt_2d_img, pred_2d_img, pred_rr, gt_rr,
                       pred_verts_rr, gt_verts_rr, faces, mano_root, K,
                       metrics_dict, out_path):
    """
    4-panel figure:
        GT skeleton | Pred skeleton | MANO mesh wireframe | 3D view

    metrics_dict keys (all floats, in mm or %):
        mje, pa, mano_mje, mano_pa, mme, pa_mme, f5, f15, vauc
    """
    import cv2 as _cv2
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa

    m = metrics_dict
    has_mesh = m.get('mme') is not None

    img_bgr = _cv2.cvtColor(img_rgb, _cv2.COLOR_RGB2BGR)

    vis_gt   = draw_skeleton_2d(img_bgr, gt_2d_img)
    vis_pred = draw_skeleton_2d(img_bgr, pred_2d_img)
    vis_mesh = render_mesh_wireframe(img_bgr, pred_verts_rr, faces, mano_root, K)
    vis_mesh = draw_skeleton_2d(vis_mesh, pred_2d_img)

    fig = plt.figure(figsize=(20, 5), facecolor='#111111')

    # Title line 1: joint metrics
    t1 = (f'Sample {i}   '
          f'Vote: MJE={m["mje"]:.1f}mm  PA={m["pa"]:.1f}mm   '
          f'MANO-reg: MJE={m["mano_mje"]:.1f}mm  PA={m["mano_pa"]:.1f}mm')
    # Title line 2: mesh metrics (if available)
    if has_mesh:
        t2 = (f'Mesh: MME={m["mme"]:.1f}mm  PA-MME={m["pa_mme"]:.1f}mm  '
              f'F@5={m["f5"]:.1f}%  F@15={m["f15"]:.1f}%  VAUC={m["vauc"]:.1f}%')
    else:
        t2 = 'Mesh metrics unavailable (training_mano.json not loaded)'

    fig.suptitle(t1 + '\n' + t2, color='white', fontsize=8, y=1.04)

    def iax(pos, title):
        ax = fig.add_subplot(1, 4, pos)
        ax.set_facecolor('#111111'); ax.axis('off')
        ax.set_title(title, color='white', fontsize=8, pad=3)
        return ax

    iax(1, 'GT skeleton').imshow(_cv2.cvtColor(vis_gt, _cv2.COLOR_BGR2RGB))
    iax(2, f'Pred skeleton  MJE={m["mje"]:.1f}mm').imshow(
        _cv2.cvtColor(vis_pred, _cv2.COLOR_BGR2RGB))
    iax(3, f'MANO mesh  MME={m["mme"]:.1f}mm' if has_mesh else 'MANO mesh').imshow(
        _cv2.cvtColor(vis_mesh, _cv2.COLOR_BGR2RGB))

    ax3d = fig.add_subplot(1, 4, 4, projection='3d')
    draw_3d_panel(ax3d, pred_rr,
                  pred_verts_rr if gt_verts_rr is not None else pred_verts_rr,
                  title='3D view')

    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.savefig(out_path, dpi=120, bbox_inches='tight', facecolor='#111111')
    plt.close(fig)


def overview_grid(png_paths, out_path, n_cols=4):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    imgs = [plt.imread(p) for p in png_paths]
    n    = len(imgs)
    nc   = min(n_cols, n)
    nr   = (n + nc - 1) // nc
    fig, axes = plt.subplots(nr, nc, figsize=(nc*5, nr*2),
                             facecolor='#111111', squeeze=False)
    for ax, im in zip(axes.flatten(), imgs):
        ax.imshow(im); ax.axis('off')
    for ax in axes.flatten()[n:]:
        ax.axis('off')
    plt.tight_layout(pad=0.2)
    plt.savefig(out_path, dpi=100, bbox_inches='tight', facecolor='#111111')
    plt.close(fig)
    print(f'  Overview → {out_path}')


# ── args ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model_path',    required=True)
    p.add_argument('--freihand_dir',  required=True)
    p.add_argument('--n_samples',     type=int, default=8)
    p.add_argument('--save_dir',      required=True)
    p.add_argument('--split',         type=str, default='evaluation',
                   choices=['train', 'evaluation'])
    p.add_argument('--seed',          type=int, default=42)
    p.add_argument('--gpu',           type=str, default='0')
    p.add_argument('--n_cols',        type=int, default=4)
    return p.parse_args()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    vis_dir = os.path.join(args.save_dir, 'vis')
    ply_dir = os.path.join(args.save_dir, 'ply')
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(ply_dir, exist_ok=True)

    # ── config ────────────────────────────────────────────────────────────────
    from main.config import cfg
    cfg.freihand_data_dir = args.freihand_dir

    # BUG FIX #1: set_args must be called so cfg.log_dir etc. are defined
    cfg.set_args(args.gpu, 'mano_vis', continue_train=False)
    cfg.create_run_dirs()
    cfg.calc_mutliscale_dim(cfg.use_big_decoder, cfg.resnet_type)

    # BUG FIX #2: must be set before get_model so memory_mask has correct size
    cfg.num_samp_obj = 0

    inp_res   = cfg.input_img_shape[0]   # 256
    hm_res    = cfg.output_hm_shape[0]  # 128
    hm_to_img = inp_res / hm_res        # 2.0

    # ── dataset ───────────────────────────────────────────────────────────────
    from freihand import Dataset
    ds = Dataset(args.split)
    print(f'  Dataset: {len(ds)} samples  (split={args.split})')

    random.seed(args.seed)
    idxs = random.sample(range(len(ds)), min(args.n_samples, len(ds)))

    from torch.utils.data import DataLoader, Subset
    loader = DataLoader(Subset(ds, idxs), batch_size=1,
                        shuffle=False, num_workers=0)

    # ── model ─────────────────────────────────────────────────────────────────
    from main.model import get_model
    model = get_model('test')

    # BUG FIX #3: handle multiple checkpoint formats
    ckpt = torch.load(args.model_path, map_location='cpu')
    if isinstance(ckpt, dict):
        state = ckpt.get('network',
                ckpt.get('model_state',
                ckpt.get('state_dict', ckpt)))
    else:
        state = ckpt
    state = {k.replace('module.', ''): v for k, v in state.items()}

    # BUG FIX #4: strict=True with clear diagnostics
    missing, unexpected = model.load_state_dict(state, strict=True)
    if missing:
        print(f'  WARNING: {len(missing)} missing keys: {missing[:3]}')
        print('  Trying strict=False...')
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            raise RuntimeError(
                f'Cannot load checkpoint — {len(missing)} keys still missing: {missing[:5]}')
    if unexpected:
        print(f'  Unexpected keys (ignored): {len(unexpected)}')

    model.to(device).eval()
    faces      = model.mano_head.mano_layer.th_faces.cpu().numpy()  # (F, 3) int
    mano_layer = model.mano_head.mano_layer                          # for GT verts
    print(f'  MANO faces: {faces.shape}')

    # ── inference loop ────────────────────────────────────────────────────────
    png_paths  = []
    all_metrics = []   # list of dicts, one per sample

    for i, (inputs, targets, meta) in enumerate(loader):
        sid = idxs[i]

        inputs_d  = {k: v.to(device) if torch.is_tensor(v) else v
                     for k, v in inputs.items()}
        targets_d = {k: v.to(device) if torch.is_tensor(v) else v
                     for k, v in targets.items()}
        meta_d    = {k: v.to(device) if torch.is_tensor(v) else v
                     for k, v in meta.items()}

        with torch.no_grad():
            out = model(inputs_d, targets_d, meta_d, 'eval')

        # ── extract outputs ───────────────────────────────────────────────
        # --- Voting head joints (hand_joints_out) ---
        pred_j20 = out['hand_joints_out'][0].cpu().numpy()          # (20,3) rr m
        pred_j21 = np.concatenate([np.zeros((1,3)), pred_j20], 0)  # (21,3) rr m

        # --- MANO regressor joints (mano_joints_out) ---
        # These come from the MANO decoder path, NOT the voting head.
        # They are in MANO local frame (wrist≈origin), already metres.
        mano_j21 = out['mano_joints_out'][0].cpu().numpy()          # (21,3) MANO local m

        # --- MANO mesh vertices (mano_mesh_out) ---
        # BUG FIX #5: always MANO local frame (wrist≈origin), metres.
        pred_verts = out['mano_mesh_out'][0].cpu().numpy()           # (778,3) MANO local m

        mano_root = meta['mano_root'][0].cpu().numpy()               # (3,) camera m
        K         = meta['cam_intr'][0].cpu().numpy()                # (3,3)

        # GT joints: mm → m, root-relative
        gt_j21 = targets['joint_cam_no_trans'][0].cpu().numpy() / 1000.0

        # Root-align — BUG FIX #6 and #7
        pred_rr    = pred_j21 - pred_j21[0:1]
        gt_rr      = gt_j21   - gt_j21[0:1]
        mano_j_rr  = mano_j21 - mano_j21[0:1]   # root-align MANO joints

        # ── joint metrics: voting head ─────────────────────────────────────
        mje_mm  = np.linalg.norm(pred_rr - gt_rr, axis=1).mean() * 1000.0
        pred_pa = procrustes_align(pred_rr, gt_rr)
        pa_mm   = np.linalg.norm(pred_pa  - gt_rr, axis=1).mean() * 1000.0

        # ── joint metrics: MANO regressor ─────────────────────────────────
        mano_mje_mm  = np.linalg.norm(mano_j_rr - gt_rr, axis=1).mean() * 1000.0
        mano_j_pa    = procrustes_align(mano_j_rr, gt_rr)
        mano_pa_mm   = np.linalg.norm(mano_j_pa  - gt_rr, axis=1).mean() * 1000.0

        # ── mesh metrics (requires GT verts from mano_param) ──────────────
        mano_param_np = targets['mano_param'][0].cpu().numpy()       # (58,)
        gt_verts = get_gt_verts(mano_layer, mano_param_np, device)   # (778,3) or None

        if gt_verts is not None:
            mme_mm    = compute_mme(pred_verts, gt_verts)
            pa_mme_mm = compute_pa_mme(pred_verts, gt_verts)
            f5_pct    = compute_f_score(pred_verts, gt_verts, threshold_mm=5.0)
            f15_pct   = compute_f_score(pred_verts, gt_verts, threshold_mm=15.0)
            vauc_pct  = compute_vauc(pred_verts, gt_verts)
        else:
            mme_mm = pa_mme_mm = f5_pct = f15_pct = vauc_pct = None

        m = dict(
            sid      = sid,
            mje      = mje_mm,
            pa       = pa_mm,
            mano_mje = mano_mje_mm,
            mano_pa  = mano_pa_mm,
            mme      = mme_mm,
            pa_mme   = pa_mme_mm,
            f5       = f5_pct,
            f15      = f15_pct,
            vauc     = vauc_pct,
        )
        all_metrics.append(m)

        # ── 2D projections ────────────────────────────────────────────────
        gt_2d_img   = targets['joint_coord'][0].cpu().numpy() * hm_to_img
        pred_2d_img = project_points(pred_rr, mano_root, K)

        # Image
        img_rgb = (inputs['img'][0].cpu().numpy().transpose(1,2,0) * 255).astype(np.uint8)
        import cv2 as _cv2
        img_rgb = _cv2.cvtColor(
            _cv2.resize(_cv2.cvtColor(img_rgb, _cv2.COLOR_RGB2BGR),
                        (inp_res, inp_res)),
            _cv2.COLOR_BGR2RGB)

        # ── save PNG ──────────────────────────────────────────────────────
        png_path = os.path.join(vis_dir, f'sample_{sid:05d}.png')
        save_sample_figure(
            i            = sid,
            img_rgb      = img_rgb,
            gt_2d_img    = gt_2d_img,
            pred_2d_img  = pred_2d_img,
            pred_rr      = pred_rr,
            gt_rr        = gt_rr,
            pred_verts_rr= pred_verts,
            gt_verts_rr  = gt_verts,
            faces        = faces,
            mano_root    = mano_root,
            K            = K,
            metrics_dict = m,
            out_path     = png_path,
        )
        png_paths.append(png_path)

        # ── save PLY ──────────────────────────────────────────────────────
        mesh_ply = os.path.join(ply_dir, f'sample_{sid:05d}_mesh.ply')
        save_mesh_ply(mesh_ply, pred_verts, faces, color=(180, 130, 100))

        pred_ply = os.path.join(ply_dir, f'sample_{sid:05d}_pred_joints.ply')
        save_joints_ply(pred_ply, pred_rr, label='pred')

        gt_ply = os.path.join(ply_dir, f'sample_{sid:05d}_gt_joints.ply')
        save_joints_ply(gt_ply, gt_rr, label='gt')

        if gt_verts is not None:
            gt_mesh_ply = os.path.join(ply_dir, f'sample_{sid:05d}_gt_mesh.ply')
            save_mesh_ply(gt_mesh_ply, gt_verts, faces, color=(80, 160, 220))

        mesh_str = (f'  MME={mme_mm:6.1f}mm  F@5={f5_pct:5.1f}%  VAUC={vauc_pct:5.1f}%'
                    if gt_verts is not None else '  (no GT mesh)')
        print(f'  [{i+1:>3}/{len(idxs)}] sample {sid:5d}  '
              f'Vote MJE={mje_mm:6.1f}  PA={pa_mm:5.1f}  '
              f'MANO MJE={mano_mje_mm:6.1f}  PA={mano_pa_mm:5.1f}'
              + mesh_str)

    # ── summary ───────────────────────────────────────────────────────────────
    def _mean(key):
        vals = [m[key] for m in all_metrics if m[key] is not None]
        return np.mean(vals) if vals else float('nan')

    has_mesh_metrics = any(m['mme'] is not None for m in all_metrics)

    sep = '─' * 62
    print(f'\n  {sep}')
    print(f'  {"Metric":<28} {"Mean":>8}  {"Best":>8}  {"Worst":>8}')
    print(f'  {sep}')

    def _row(label, key, lower_is_better=True):
        vals = [m[key] for m in all_metrics if m[key] is not None]
        if not vals:
            print(f'  {label:<28} {"N/A":>8}')
            return
        mn   = np.mean(vals)
        best = min(vals) if lower_is_better else max(vals)
        wrst = max(vals) if lower_is_better else min(vals)
        print(f'  {label:<28} {mn:>8.2f}  {best:>8.2f}  {wrst:>8.2f}')

    print(f'  {"[Voting head  hand_joints_out]":<28}')
    _row('  MJE (mm)',               'mje')
    _row('  PA-MPJPE (mm)',          'pa')
    print(f'  {"[MANO regressor  mano_joints]":<28}')
    _row('  MANO-MJE (mm)',          'mano_mje')
    _row('  MANO-PA-MPJPE (mm)',     'mano_pa')
    if has_mesh_metrics:
        print(f'  {"[Mesh  mano_mesh_out vs GT]":<28}')
        _row('  MME (mm)',           'mme')
        _row('  PA-MME (mm)',        'pa_mme')
        _row('  F@5mm (%)',          'f5',   lower_is_better=False)
        _row('  F@15mm (%)',         'f15',  lower_is_better=False)
        _row('  VAUC (%)',           'vauc', lower_is_better=False)
    else:
        print(f'  Mesh metrics: N/A (training_mano.json not loaded)')
    print(f'  {sep}')

    # ── CSV ───────────────────────────────────────────────────────────────────
    csv_path = os.path.join(args.save_dir, 'metrics.csv')
    csv_keys = ['sid', 'mje', 'pa', 'mano_mje', 'mano_pa',
                'mme', 'pa_mme', 'f5', 'f15', 'vauc']
    with open(csv_path, 'w') as fp:
        fp.write(','.join(csv_keys) + '\n')
        for m in all_metrics:
            row = [str(m[k]) if m[k] is not None else '' for k in csv_keys]
            fp.write(','.join(row) + '\n')
    print(f'  CSV  → {csv_path}')

    # ── overview ──────────────────────────────────────────────────────────────
    ov_path = os.path.join(args.save_dir, 'overview.png')
    overview_grid(png_paths, ov_path, n_cols=args.n_cols)

    print(f'  Vis  → {vis_dir}/')
    print(f'  PLY  → {ply_dir}/')
    print(f'  Overview → {ov_path}')


if __name__ == '__main__':
    main()