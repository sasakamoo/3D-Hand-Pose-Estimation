"""
overfit_test.py — Overfit a tiny batch, then save all MANO outputs
===================================================================
Trains on N fixed samples until the loss converges, then runs inference
and saves every MANO output artefact for inspection.

Saves
─────
  {save_dir}/
    overfit_test.png              training curves (all losses + MJE + PA-MPJPE)
    overfit_model.pt              best checkpoint (lowest Vote-MJE)
    metrics.csv                   per-sample table of all metrics
    predictions/
      sample_XXXX.png             6-panel image per sample
    ply/
      sample_XXXX_pred_mesh.ply   predicted MANO mesh  (778 verts, MANO local m)
      sample_XXXX_pred_joints.ply predicted skeleton   (21 joints, root-relative m)
      sample_XXXX_gt_mesh.ply     GT MANO mesh         (from mano_param, if available)
      sample_XXXX_gt_joints.ply   GT skeleton          (root-relative m)
    predictions_overview.png      all sample PNGs tiled

Fixes vs old version
─────────────────────
  1. Loss weights now use the rebalanced values from our analysis:
       sdf_hand_weight : cfg value (10 after config fix, was 50)
       MANO losses     : x3.0 external multiplier (same as updated train.py)
       joint_weight    : cfg value (0.5 after config fix, was 0.1)
     Old code used mano_mesh_loss *1e-2 which is wrong for the current
     cfg.lambda_verts3d=1e4 + metres-scale vertices setup.
  2. Eval MJE unit: correctly * 1000 (metres -> mm), not * 100 (-> cm).
  3. PA-MPJPE always <= MJE: both pred and GT are root-aligned consistently.
  4. MANO-regressor joints (mano_joints_out) reported separately from
     voting-head joints (hand_joints_out).
  5. Full PLY export: pred mesh, pred joints, GT mesh, GT joints.
  6. GT mesh obtained by running mano_layer directly with GT mano_param.
  7. All mesh metrics (MME, PA-MME, F@5mm, F@15mm, VAUC) in metrics.csv.

Usage (from HOISDF root)
─────────────────────────
  # Pipeline smoke-test (random init):
  python main/overfit_test.py --freihand_dir /path/to/FreiHAND

  # From checkpoint (recommended):
  python main/overfit_test.py \\
      --freihand_dir /path/to/FreiHAND \\
      --resume /path/to/snapshot_45_1331.pth.tar \\
      --n_samples 4 --iters 500 --lr 1e-4
"""

import os
os.environ['MPLBACKEND']   = 'Agg'
os.environ['DISPLAY']      = ''
os.environ['MPLCONFIGDIR'] = '/tmp'

import sys, argparse, csv
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

_file_dir = os.path.dirname(os.path.abspath(__file__))
_root_dir = os.path.dirname(_file_dir)
for _p in [_root_dir, _file_dir,
           os.path.join(_root_dir, 'common'),
           os.path.join(_root_dir, 'data')]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
]
_FINGER_BGR = [(0,220,0),(255,100,0),(0,200,255),(0,100,255),(200,0,200)]
_FINGER_MPL = ['#00dd00','#ff6400','#00c8ff','#0064ff','#c800c8']
BONE_BGR    = [_FINGER_BGR[f] for f in range(5) for _ in range(4)]
BONE_MPL    = [_FINGER_MPL[f] for f in range(5) for _ in range(4)]
_JOINT_RGB  = [(128,128,128)] + [c for c in _FINGER_BGR for _ in range(4)]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--freihand_dir', required=True)
    p.add_argument('--resume',    default=None)
    p.add_argument('--n_samples', type=int,   default=4)
    p.add_argument('--iters',     type=int,   default=500)
    p.add_argument('--lr',        type=float, default=1e-4)
    p.add_argument('--save_dir',  default=os.path.join(_root_dir, 'overfit_output'))
    p.add_argument('--gpu',       default='0')
    return p.parse_args()


def hdr(msg):
    print('\n' + '─'*65 + f'\n  {msg}\n' + '─'*65)


# ── PLY ───────────────────────────────────────────────────────────────────────

def save_mesh_ply(path, verts, faces, color=(180,130,100)):
    V, F = len(verts), len(faces)
    r, g, b = color
    with open(path, 'w') as fp:
        fp.write('ply\nformat ascii 1.0\n'
                 f'element vertex {V}\n'
                 'property float x\nproperty float y\nproperty float z\n'
                 'property uchar red\nproperty uchar green\nproperty uchar blue\n'
                 f'element face {F}\n'
                 'property list uchar int vertex_indices\nend_header\n')
        for v in verts:
            fp.write(f'{v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {r} {g} {b}\n')
        for f in faces:
            fp.write(f'3 {f[0]} {f[1]} {f[2]}\n')


def save_joints_ply(path, joints, label='joints'):
    with open(path, 'w') as fp:
        fp.write(f'ply\nformat ascii 1.0\ncomment {label}\n'
                 f'element vertex {len(joints)}\n'
                 'property float x\nproperty float y\nproperty float z\n'
                 'property uchar red\nproperty uchar green\nproperty uchar blue\n'
                 f'element edge {len(CONNECTIONS)}\n'
                 'property int vertex1\nproperty int vertex2\nend_header\n')
        for ji, j in enumerate(joints):
            r, g, b = _JOINT_RGB[ji]
            fp.write(f'{j[0]:.6f} {j[1]:.6f} {j[2]:.6f} {r} {g} {b}\n')
        for s, e in CONNECTIONS:
            fp.write(f'{s} {e}\n')


# ── metrics ───────────────────────────────────────────────────────────────────

def procrustes_align(X, Y):
    muX = X.mean(0); muY = Y.mean(0)
    X0 = X - muX;    Y0 = Y - muY
    nX = np.linalg.norm(X0); nY = np.linalg.norm(Y0)
    X0 /= nX; Y0 /= nY
    U, _, Vt = np.linalg.svd(X0.T @ Y0)
    s = nY / nX
    return s * ((U @ Vt) @ X.T).T + muY - s * ((U @ Vt) @ muX)

def compute_mme(pv, gv):
    return np.linalg.norm(pv - gv, axis=-1).mean() * 1000.0

def compute_pa_mme(pv, gv):
    return np.linalg.norm(procrustes_align(pv, gv) - gv, axis=-1).mean() * 1000.0

def compute_f_score(pv, gv, thr_mm):
    return (np.linalg.norm(pv - gv, axis=-1) * 1000.0 < thr_mm).mean() * 100.0

def compute_vauc(pv, gv, max_mm=50.0, n=50):
    d = np.linalg.norm(pv - gv, axis=-1) * 1000.0
    return np.mean([(d < t).mean() for t in np.linspace(0, max_mm, n+1)[1:]]) * 100.0

def get_gt_verts(mano_layer, mano_param_np, device):
    if np.abs(mano_param_np).sum() < 1e-6:
        return None
    pose  = torch.from_numpy(mano_param_np[:48]).unsqueeze(0).to(device)
    shape = torch.from_numpy(mano_param_np[48:]).unsqueeze(0).to(device)
    with torch.no_grad():
        v, _ = mano_layer(th_pose_coeffs=pose, th_betas=shape)
    return v[0].cpu().numpy() / 1000.0


# ── drawing ───────────────────────────────────────────────────────────────────

def project_pts(pts_rr, mano_root, K):
    pts_cam = pts_rr + mano_root[None, :]
    proj    = (K @ pts_cam.T).T
    return proj[:, :2] / proj[:, 2:3].clip(min=1e-6)


def draw_skeleton_2d(img_bgr, kpts, colors=None, jcol=(255,255,255)):
    import cv2
    vis = img_bgr.copy()
    for bi, (s, e) in enumerate(CONNECTIONS):
        c = colors[bi] if colors else BONE_BGR[bi]
        cv2.line(vis,
                 tuple(np.clip(kpts[s].astype(int), -4096, 4096)),
                 tuple(np.clip(kpts[e].astype(int), -4096, 4096)),
                 c, 2, cv2.LINE_AA)
    for pt in kpts:
        cv2.circle(vis, tuple(np.clip(pt.astype(int), -4096, 4096)), 4, jcol, -1)
    return vis


def render_mesh_wireframe(img_bgr, verts_rr, faces, mano_root, K):
    import cv2
    vis = img_bgr.copy()
    v2d = project_pts(verts_rr, mano_root, K)
    drawn = set()
    for f in faces:
        for a, b in [(f[0],f[1]),(f[1],f[2]),(f[2],f[0])]:
            key = (min(a,b), max(a,b))
            if key in drawn: continue
            drawn.add(key)
            cv2.line(vis,
                     tuple(np.clip(v2d[a].astype(int), -4096, 4096)),
                     tuple(np.clip(v2d[b].astype(int), -4096, 4096)),
                     (180,130,100), 1, cv2.LINE_AA)
    return vis


def draw_3d(ax, joints, colors, lw=1.8, alpha=0.9, jcol='white', js=20):
    for bi, (s, e) in enumerate(CONNECTIONS):
        ax.plot([joints[s,0],joints[e,0]], [joints[s,1],joints[e,1]],
                [joints[s,2],joints[e,2]], color=colors[bi], lw=lw, alpha=alpha)
    ax.scatter(joints[:,0], joints[:,1], joints[:,2],
               c=jcol, s=js, zorder=5, depthshade=False)


def setup_3d(ax, title):
    ax.set_facecolor('#181818')
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill = False; pane.set_edgecolor('#333')
    ax.set_title(title, color='white', fontsize=8, pad=2)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])


def eq_axes(ax, *arrs):
    pts = np.vstack(arrs)
    mid = pts.mean(0)
    r   = max((pts.max(0) - pts.min(0)).max() / 2.0, 0.03)
    ax.set_xlim(mid[0]-r, mid[0]+r)
    ax.set_ylim(mid[1]-r, mid[1]+r)
    ax.set_zlim(mid[2]-r, mid[2]+r)


# ── per-sample figure ─────────────────────────────────────────────────────────

def save_sample_figure(sid, img_tensor, gt_2d_hm, inp_res, hm_res,
                       vote_rr, gt_rr, mano_rr, mano_root, K,
                       pred_verts, faces, metrics_dict, out_path):
    import cv2 as cv
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa

    m = metrics_dict
    hm_to_img = inp_res / hm_res

    img_np  = (img_tensor.permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
    img_bgr = cv.resize(cv.cvtColor(img_np, cv.COLOR_RGB2BGR), (inp_res, inp_res))

    def rgb(bgr): return cv.cvtColor(bgr, cv.COLOR_BGR2RGB)

    gt_2d_img   = gt_2d_hm * hm_to_img
    vote_2d_img = project_pts(vote_rr, mano_root, K)
    mano_2d_img = project_pts(mano_rr, mano_root, K)

    vis_gt   = draw_skeleton_2d(img_bgr, gt_2d_img, [(70,70,70)]*20, (160,160,160))
    vis_vote = draw_skeleton_2d(img_bgr, vote_2d_img, jcol=(0,255,100))
    vis_mesh = (render_mesh_wireframe(img_bgr, pred_verts, faces, mano_root, K)
                if pred_verts is not None else img_bgr.copy())
    vis_mano = draw_skeleton_2d(img_bgr, mano_2d_img, jcol=(80,180,255))

    fig = plt.figure(figsize=(24, 5), facecolor='#111')

    def iax(pos, title, col='white'):
        ax = fig.add_subplot(1, 6, pos)
        ax.set_facecolor('#111'); ax.axis('off')
        ax.set_title(title, color=col, fontsize=7, pad=2)
        return ax

    iax(1, 'GT skeleton').imshow(rgb(vis_gt))
    iax(2, f'Vote  MJE={m["mje"]:.1f}mm\nPA={m["pa"]:.1f}mm', '#44ff88').imshow(rgb(vis_vote))
    iax(3, (f'MANO mesh  MME={m["mme"]:.1f}mm' if m.get('mme') else 'MANO mesh'),
        '#ffaa44').imshow(rgb(vis_mesh))
    iax(4, f'MANO joints\nMJE={m["mano_mje"]:.1f}mm', '#44aaff').imshow(rgb(vis_mano))

    ax5 = fig.add_subplot(1, 6, 5, projection='3d'); setup_3d(ax5, '3D front')
    draw_3d(ax5, gt_rr,   ['#555']*20, lw=1, jcol='#777', js=10)
    draw_3d(ax5, vote_rr, BONE_MPL,    lw=2, jcol='white', js=20)
    eq_axes(ax5, gt_rr, vote_rr); ax5.view_init(elev=20, azim=-60)

    ax6 = fig.add_subplot(1, 6, 6, projection='3d'); setup_3d(ax6, '3D side')
    draw_3d(ax6, gt_rr,   ['#555']*20, lw=1, jcol='#777', js=10)
    draw_3d(ax6, vote_rr, BONE_MPL,    lw=2, jcol='white', js=20)
    eq_axes(ax6, gt_rr, vote_rr); ax6.view_init(elev=20, azim=30)

    t = (f'Sample {sid}  |  Vote: MJE={m["mje"]:.1f}  PA={m["pa"]:.1f}  |  '
         f'MANO: MJE={m["mano_mje"]:.1f}  PA={m["mano_pa"]:.1f}')
    if m.get('mme') is not None:
        t += f'  |  MME={m["mme"]:.1f}  F@5={m["f5"]:.1f}%  VAUC={m["vauc"]:.1f}%'
    fig.suptitle(t + '  [mm]', color='white', fontsize=7.5, y=1.02)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight', facecolor='#111')
    plt.close(fig)


def overview_grid(pngs, out_path, n_cols=4):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    n = len(pngs); nc = min(n_cols, n); nr = (n + nc - 1) // nc
    fig, axes = plt.subplots(nr, nc, figsize=(nc*6, nr*2.5),
                             facecolor='#111', squeeze=False)
    for ax, im in zip(axes.flatten(), [plt.imread(p) for p in pngs]):
        ax.imshow(im); ax.axis('off')
    for ax in axes.flatten()[n:]: ax.axis('off')
    plt.tight_layout(pad=0.2)
    plt.savefig(out_path, dpi=100, bbox_inches='tight', facecolor='#111')
    plt.close(fig)


def save_curves(hist, passed, ld_pct, md_pct, out_path):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    iters = hist['iter']
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.patch.set_facecolor('#181818')
    for ax in axes.flat:
        ax.set_facecolor('#222'); ax.tick_params(colors='#aaa')
        ax.title.set_color('white')
        for sp in ax.spines.values(): sp.set_edgecolor('#444')
    fig.suptitle(
        f'Overfit  {"PASSED ✓" if passed else "FAILED ✗"}  '
        f'loss↓{ld_pct:.0f}%  Vote-MJE↓{md_pct:.0f}%',
        color='#44ff88' if passed else '#ff4444', fontsize=12)

    def sl(ax, key, label, color):
        y = [max(v, 1e-9) for v in hist[key]]
        ax.semilogy(iters, y, lw=2, label=label, color=color)

    sl(axes[0,0], 'total', 'total', '#4488ff')
    axes[0,0].set_title('Total loss'); axes[0,0].legend(fontsize=8); axes[0,0].grid(alpha=0.2)

    sl(axes[0,1], 'sdfhand_loss', 'sdf', '#cc44ff')
    sl(axes[0,1], 'joint_heatmap', 'heatmap', '#ffaa00')
    axes[0,1].set_title('SDF + Heatmap (raw)'); axes[0,1].legend(fontsize=8); axes[0,1].grid(alpha=0.2)

    sl(axes[0,2], 'loss_joint_3d', 'j3d', '#44ff88')
    sl(axes[0,2], 'loss_joint_cls', 'j_cls', '#ff4444')
    sl(axes[0,2], 'loss_all_joint_3d', 'allj3d', '#00dddd')
    axes[0,2].set_title('Voting losses (raw)'); axes[0,2].legend(fontsize=8); axes[0,2].grid(alpha=0.2)

    sl(axes[1,0], 'mano_mesh_loss', 'mesh', '#ff8800')
    sl(axes[1,0], 'mano_joint_loss', 'joint', '#88ff00')
    sl(axes[1,0], 'pose_param_loss', 'pose', '#ff44aa')
    sl(axes[1,0], 'shape_param_loss', 'shape', '#aaaaff')
    axes[1,0].set_title('MANO losses (raw)'); axes[1,0].legend(fontsize=8); axes[1,0].grid(alpha=0.2)

    axes[1,1].plot(iters, [v*1000 for v in hist['mje_m']],
                   lw=2, color='#44ff88', label='Vote-MJE')
    axes[1,1].plot(iters, [v*1000 for v in hist['pa_mje_m']],
                   lw=2, color='#ffaa44', ls='--', label='Vote-PA')
    axes[1,1].plot(iters, [v*1000 for v in hist['mano_mje_m']],
                   lw=2, color='#44aaff', label='MANO-MJE')
    axes[1,1].plot(iters, [v*1000 for v in hist['mano_pa_m']],
                   lw=2, color='#aaaaff', ls='--', label='MANO-PA')
    axes[1,1].set_title('All joint errors (mm)')
    axes[1,1].set_xlabel('iter', color='#aaa')
    axes[1,1].legend(fontsize=7, facecolor='#333', edgecolor='none', labelcolor='white')
    axes[1,1].grid(alpha=0.2)

    keys    = ['total','loss_joint_3d','loss_all_joint_3d','sdfhand_loss',
               'mano_mesh_loss','pose_param_loss']
    present = [k for k in keys if k in hist and hist[k][0] > 1e-9]
    ratios  = [hist[k][-1]/(hist[k][0]+1e-9) for k in present]
    barcols = ['#44ff88' if r<.2 else '#ffaa00' if r<.5 else '#ff4444' for r in ratios]
    axes[1,2].bar(range(len(present)), ratios, color=barcols)
    axes[1,2].set_xticks(range(len(present)))
    axes[1,2].set_xticklabels([k[:12] for k in present], rotation=20, color='#aaa', fontsize=6)
    axes[1,2].axhline(.2, color='#44ff88', ls='--', lw=1)
    axes[1,2].axhline(.5, color='#ffaa00', ls='--', lw=1)
    axes[1,2].set_title('Final/Init ratio (lower=better)')
    axes[1,2].grid(alpha=0.2, axis='y'); axes[1,2].tick_params(colors='#aaa')

    plt.tight_layout()
    plt.savefig(out_path, dpi=130, facecolor='#181818')
    plt.close(fig)
    print(f'  Curves → {out_path}')


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    pred_dir = os.path.join(args.save_dir, 'predictions')
    ply_dir  = os.path.join(args.save_dir, 'ply')
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(ply_dir,  exist_ok=True)

    hdr('Hand-only HOISDF — Overfit + MANO Output Test')
    print(f'  freihand : {args.freihand_dir}')
    print(f'  resume   : {args.resume or "None (random init)"}')
    print(f'  n_samples: {args.n_samples}   iters: {args.iters}   lr: {args.lr}')
    print(f'  device   : {device}   save: {args.save_dir}')

    # ── config ────────────────────────────────────────────────────────────
    from main.config import cfg
    cfg.freihand_data_dir = args.freihand_dir
    cfg.set_args(args.gpu, 'overfit_test', continue_train=False)
    cfg.create_run_dirs()
    cfg.calc_mutliscale_dim(cfg.use_big_decoder, cfg.resnet_type)
    cfg.num_samp_obj = 0

    inp_res = cfg.input_img_shape[0]
    hm_res  = cfg.output_hm_shape[0]

    # ── dataset ───────────────────────────────────────────────────────────
    from freihand import Dataset as FreiHandDataset
    full_ds = FreiHandDataset('train')
    n = min(args.n_samples, len(full_ds))
    indices = list(range(n))
    loader  = DataLoader(Subset(full_ds, indices), batch_size=n,
                         shuffle=False, num_workers=0)
    f_inp, f_tgt, f_meta = next(iter(loader))

    def to_dev(d):
        return {k: v.to(device) if torch.is_tensor(v) else v for k,v in d.items()}
    f_inp  = to_dev(f_inp)
    f_tgt  = to_dev(f_tgt)
    f_meta = to_dev(f_meta)

    print(f'\n  img            : {f_inp["img"].shape}')
    print(f'  joint_cam_no_t : {f_tgt["joint_cam_no_trans"].shape}  (mm, root-rel)')
    has_gt_mano = f_tgt['mano_param'].abs().sum() > 1e-6
    print(f'  mano_param     : {"available — GT mesh will be saved" if has_gt_mano else "zeros — GT mesh unavailable"}')

    # ── model ─────────────────────────────────────────────────────────────
    from main.model import get_model
    model = get_model('train').to(device)

    if args.resume:
        print(f'\n  Loading: {args.resume}')
        raw   = torch.load(args.resume, map_location='cpu')
        state = raw.get('network', raw.get('model_state', raw))
        state = {k.replace('module.',''): v for k,v in state.items()}
        missing, _ = model.load_state_dict(state, strict=True)
        if missing:
            raise RuntimeError(f'Missing keys: {missing[:5]}')
        print('  Loaded OK.')
    else:
        print('  No checkpoint — random init.')

    mano_layer = model.mano_head.mano_layer
    mano_faces = mano_layer.th_faces.cpu().numpy()
    print(f'  MANO faces: {mano_faces.shape}')

    # ── loss weights ──────────────────────────────────────────────────────
    # Mirror the rebalanced weights from the fixed train.py exactly,
    # so this test exercises the same gradient balance as real training.
    W = {
        'sdfhand_loss':      cfg.sdf_hand_weight,   # 10 after config fix
        'joint_heatmap':     cfg.hm_weight,          # 0.001
        'loss_joint_3d':     cfg.joint_weight,       # 0.5 after config fix
        'loss_joint_cls':    cfg.cls_weight,          # 1.0
        'loss_all_joint_3d': cfg.joint_weight,       # 0.5
        'mano_mesh_loss':    3.0,
        'mano_joint_loss':   3.0,
        'pose_param_loss':   3.0,
        'shape_param_loss':  3.0,
    }
    print('\n  Effective loss weights:')
    for k, v in W.items():
        print(f'    {k:<22}: {v}')

    loss_keys = list(W.keys())
    hist = {k: [] for k in ['iter','total','mje_m','pa_mje_m','mano_mje_m','mano_pa_m']
            + loss_keys}

    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    best_mje, best_state = float('inf'), None

    hdr('Training')
    print(f'{"Iter":>5}  {"Total":>8}  {"SDF":>7}  {"AllJ3D":>7}  '
          f'{"Mesh":>7}  {"Vote-MJE":>9}  {"MANO-MJE":>9}')

    for it in range(1, args.iters + 1):
        model.train()
        opt.zero_grad(set_to_none=True)

        out = model(f_inp, f_tgt, f_meta, 'train', epoch_cnt=0, batch_ratio=0.5)
        # epoch_cnt=0 forces the fast pre_points sampling path (condition:
        # epoch_cnt < cfg.point_sampling_epoch=40 is always True).
        # Using epoch_cnt=it caused sdf_infer to trigger at iter>=40,
        # which loops over 64^3=262144 voxels per sample — the 30-min hang.

        raw_losses = {k: out[k].mean() for k in loss_keys
                      if k in out and torch.is_tensor(out[k])}
        total = sum(raw_losses[k] * W[k] for k in raw_losses)

        if torch.isnan(total):
            print(f'  NaN at iter {it}. Stopping.'); break

        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        # ── eval ──────────────────────────────────────────────────────────
        model.eval()
        with torch.no_grad():
            ev = model(f_inp, f_tgt, f_meta, 'eval', epoch_cnt=0)
            # epoch_cnt=0: forces fast GT sampling, avoids sdf_infer voxel scan

        B = f_inp['img'].shape[0]

        # Voting joints
        vote_j20 = ev['hand_joints_out']                          # (B,20,3) rr m
        vote_j21 = torch.cat([torch.zeros(B,1,3,device=device), vote_j20], 1)
        # MANO joints
        mano_j21 = ev['mano_joints_out']                          # (B,21,3) MANO local m
        # GT joints: mm -> m, already root-relative
        gt_j21   = f_tgt['joint_cam_no_trans'] / 1000.0

        vote_rr_t = vote_j21 - vote_j21[:, 0:1]
        mano_rr_t = mano_j21 - mano_j21[:, 0:1]
        gt_rr_t   = gt_j21   - gt_j21[:, 0:1]

        # Mean MJE over batch (metres)
        vote_mje_m = (vote_rr_t - gt_rr_t).norm(dim=-1).mean().item()
        mano_mje_m = (mano_rr_t - gt_rr_t).norm(dim=-1).mean().item()

        # PA (numpy, per sample, then mean)
        vote_rr_np = vote_rr_t.cpu().numpy()
        mano_rr_np = mano_rr_t.cpu().numpy()
        gt_rr_np   = gt_rr_t.cpu().numpy()

        vote_pa_m = float(np.mean([
            np.linalg.norm(procrustes_align(vote_rr_np[b], gt_rr_np[b])
                           - gt_rr_np[b], axis=1).mean()
            for b in range(B)]))
        mano_pa_m = float(np.mean([
            np.linalg.norm(procrustes_align(mano_rr_np[b], gt_rr_np[b])
                           - gt_rr_np[b], axis=1).mean()
            for b in range(B)]))

        # History
        hist['iter'].append(it)
        hist['total'].append(total.item())
        hist['mje_m'].append(vote_mje_m)
        hist['pa_mje_m'].append(vote_pa_m)
        hist['mano_mje_m'].append(mano_mje_m)
        hist['mano_pa_m'].append(mano_pa_m)
        for k in loss_keys:
            hist[k].append(raw_losses.get(k, torch.tensor(0.)).item())

        if vote_mje_m < best_mje:
            best_mje   = vote_mje_m
            best_state = {k: v.cpu().clone() for k,v in model.state_dict().items()}

        if it % 10 == 0 or it == 1:
            sdf_v  = raw_losses.get('sdfhand_loss',      torch.tensor(0.)).item()
            allj_v = raw_losses.get('loss_all_joint_3d', torch.tensor(0.)).item()
            mesh_v = raw_losses.get('mano_mesh_loss',    torch.tensor(0.)).item()
            print(f'{it:>5}  {total.item():>8.4f}  {sdf_v:>7.4f}  {allj_v:>7.4f}  '
                  f'{mesh_v:>7.4f}  {vote_mje_m*1000:>9.2f}  {mano_mje_m*1000:>9.2f}')

    # ── checkpoint ────────────────────────────────────────────────────────
    ckpt_path = os.path.join(args.save_dir, 'overfit_model.pt')
    torch.save({'model_state': best_state,
                'best_mje_mm': best_mje * 1000,
                'history': hist, 'args': vars(args)}, ckpt_path)
    print(f'\n  Checkpoint → {ckpt_path}')

    # ── verdict ───────────────────────────────────────────────────────────
    hdr('Verdict')
    i_loss = hist['total'][0];    f_loss = hist['total'][-1]
    i_mje  = hist['mje_m'][0];    f_mje  = hist['mje_m'][-1]
    i_mano = hist['mano_mje_m'][0]; f_mano = hist['mano_mje_m'][-1]
    ld_pct = 100*(1 - f_loss/(i_loss+1e-9))
    md_pct = 100*(1 - f_mje /(i_mje +1e-9))
    mn_pct = 100*(1 - f_mano/(i_mano+1e-9))

    print(f'  Init:  total={i_loss:.4f}  Vote-MJE={i_mje*1000:.1f}mm  MANO-MJE={i_mano*1000:.1f}mm')
    print(f'  Final: total={f_loss:.4f}  Vote-MJE={f_mje*1000:.1f}mm  MANO-MJE={f_mano*1000:.1f}mm')
    print(f'  Loss ↓{ld_pct:.0f}%   Vote-MJE ↓{md_pct:.0f}%   MANO-MJE ↓{mn_pct:.0f}%')
    print(f'  Best Vote-MJE: {best_mje*1000:.2f}mm')

    passed = ld_pct >= 80 and md_pct >= 70
    print(f'\n  {"PASSED ✓" if passed else "FAILED ✗"} — model '
          f'{"CAN" if passed else "CANNOT"} overfit.')
    if not passed:
        if ld_pct < 80:
            print(f'    Loss -{ld_pct:.0f}% (want 80%) → check sdf_hand_weight, MANO weights')
        if md_pct < 70:
            print(f'    Vote-MJE -{md_pct:.0f}% (want 70%) → check joint_weight / LR')
        if mn_pct < 50:
            print(f'    MANO-MJE -{mn_pct:.0f}% → MANO head still starved '
                  '(increase mano_mesh/joint weights in W dict)')

    save_curves(hist, passed, ld_pct, md_pct,
                os.path.join(args.save_dir, 'overfit_test.png'))

    # ── inference with best weights ───────────────────────────────────────
    hdr('Saving MANO outputs (PLY + PNG + CSV)')
    model.load_state_dict(best_state)
    model.to(device).eval()
    with torch.no_grad():
        ev = model(f_inp, f_tgt, f_meta, 'eval', epoch_cnt=0)

    vote_j20_np = ev['hand_joints_out'].cpu().numpy()             # (B,20,3) rr m
    mano_j21_np = ev['mano_joints_out'].cpu().numpy()             # (B,21,3) MANO local m
    pred_v_np   = ev['mano_mesh_out'].cpu().numpy()               # (B,778,3) MANO local m
    gt_j21_np   = (f_tgt['joint_cam_no_trans']/1000).cpu().numpy()
    gt_2d_np    = f_tgt['joint_coord'].cpu().numpy()              # (B,21,2) hm px
    mano_params = f_tgt['mano_param'].cpu().numpy()               # (B,58)
    mano_roots  = f_meta['mano_root'].cpu().numpy()               # (B,3) cam m
    cam_intrs   = f_meta['cam_intr'].cpu().numpy()                # (B,3,3)

    csv_keys = ['sid','mje','pa','mano_mje','mano_pa','mme','pa_mme','f5','f15','vauc']
    all_metrics = []
    pngs = []

    with open(os.path.join(args.save_dir, 'metrics.csv'), 'w', newline='') as csvf:
        writer = csv.DictWriter(csvf, fieldnames=csv_keys)
        writer.writeheader()

        for si in range(n):
            sid = indices[si]
            mano_root = mano_roots[si]
            K         = cam_intrs[si]

            # Root-relative arrays
            vote_rr = np.concatenate([np.zeros((1,3)), vote_j20_np[si]], 0)
            mano_rr = mano_j21_np[si] - mano_j21_np[si, 0:1]
            gt_rr   = gt_j21_np[si]   - gt_j21_np[si, 0:1]
            pred_v  = pred_v_np[si]   # (778,3) MANO local m

            # Joint metrics
            mje_mm   = np.linalg.norm(vote_rr - gt_rr, axis=1).mean() * 1000.0
            pa_mm    = np.linalg.norm(procrustes_align(vote_rr, gt_rr) - gt_rr,
                                       axis=1).mean() * 1000.0
            mano_mje = np.linalg.norm(mano_rr - gt_rr, axis=1).mean() * 1000.0
            mano_pa  = np.linalg.norm(procrustes_align(mano_rr, gt_rr) - gt_rr,
                                       axis=1).mean() * 1000.0

            # Mesh metrics (need GT verts)
            gt_v = get_gt_verts(mano_layer, mano_params[si], device)
            if gt_v is not None:
                mme    = compute_mme(pred_v, gt_v)
                pa_mme = compute_pa_mme(pred_v, gt_v)
                f5     = compute_f_score(pred_v, gt_v, 5.0)
                f15    = compute_f_score(pred_v, gt_v, 15.0)
                vauc   = compute_vauc(pred_v, gt_v)
            else:
                mme = pa_mme = f5 = f15 = vauc = None

            m = dict(sid=sid, mje=mje_mm, pa=pa_mm,
                     mano_mje=mano_mje, mano_pa=mano_pa,
                     mme=mme, pa_mme=pa_mme, f5=f5, f15=f15, vauc=vauc)
            all_metrics.append(m)

            # ── PLY files ─────────────────────────────────────────────────
            # Predicted mesh: MANO local frame, metres. Open in MeshLab/Blender.
            save_mesh_ply(os.path.join(ply_dir, f'sample_{sid:04d}_pred_mesh.ply'),
                          pred_v, mano_faces, color=(180,130,100))
            # Predicted joints (voting head): root-relative metres
            save_joints_ply(os.path.join(ply_dir, f'sample_{sid:04d}_pred_joints.ply'),
                            vote_rr, label='pred_vote')
            # GT joints: root-relative metres
            save_joints_ply(os.path.join(ply_dir, f'sample_{sid:04d}_gt_joints.ply'),
                            gt_rr, label='gt')
            # GT mesh: same MANO local frame, blue colour
            if gt_v is not None:
                save_mesh_ply(os.path.join(ply_dir, f'sample_{sid:04d}_gt_mesh.ply'),
                              gt_v, mano_faces, color=(80,160,220))

            # ── PNG ───────────────────────────────────────────────────────
            png_path = os.path.join(pred_dir, f'sample_{sid:04d}.png')
            save_sample_figure(
                sid=sid, img_tensor=f_inp['img'][si],
                gt_2d_hm=gt_2d_np[si], inp_res=inp_res, hm_res=hm_res,
                vote_rr=vote_rr, gt_rr=gt_rr, mano_rr=mano_rr,
                mano_root=mano_root, K=K,
                pred_verts=pred_v, faces=mano_faces,
                metrics_dict=m, out_path=png_path,
            )
            pngs.append(png_path)

            # ── CSV ───────────────────────────────────────────────────────
            writer.writerow({k: (f'{m[k]:.4f}' if m[k] is not None else '')
                             for k in csv_keys})

            mesh_str = (f'  MME={mme:.1f}mm  F@5={f5:.1f}%  VAUC={vauc:.1f}%'
                        if gt_v is not None else '  (no GT mesh)')
            print(f'  [{si+1}/{n}] sid={sid:4d}  '
                  f'Vote MJE={mje_mm:6.1f}mm  PA={pa_mm:5.1f}mm  '
                  f'MANO MJE={mano_mje:6.1f}mm  PA={mano_pa:5.1f}mm'
                  + mesh_str)

    # ── summary ───────────────────────────────────────────────────────────
    def row(label, key, lo=True):
        vals = [m[key] for m in all_metrics if m[key] is not None]
        if not vals: print(f'  {label:<28} {"N/A":>8}'); return
        mn = np.mean(vals); b=(min if lo else max)(vals); w=(max if lo else min)(vals)
        print(f'  {label:<28} {mn:>8.2f}  {b:>8.2f}  {w:>8.2f}')

    sep = '─'*58
    print(f'\n  {sep}')
    print(f'  {"Metric":<28} {"Mean":>8}  {"Best":>8}  {"Worst":>8}')
    print(f'  {sep}')
    print(f'  [Voting  hand_joints_out]')
    row('  MJE (mm)', 'mje'); row('  PA-MPJPE (mm)', 'pa')
    print(f'  [MANO reg  mano_joints_out]')
    row('  MANO-MJE (mm)', 'mano_mje'); row('  MANO-PA (mm)', 'mano_pa')
    if any(m['mme'] is not None for m in all_metrics):
        print(f'  [Mesh  vs GT mano_param]')
        row('  MME (mm)',    'mme');  row('  PA-MME (mm)', 'pa_mme')
        row('  F@5mm (%)',   'f5',   lo=False)
        row('  F@15mm (%)',  'f15',  lo=False)
        row('  VAUC (%)',    'vauc', lo=False)
    print(f'  {sep}')

    overview_grid(pngs, os.path.join(args.save_dir, 'predictions_overview.png'),
                  n_cols=min(4, n))

    print(f'\n  PNG   → {pred_dir}/')
    print(f'  PLY   → {ply_dir}/')
    print(f'  CSV   → {os.path.join(args.save_dir, "metrics.csv")}')
    print(f'  Ckpt  → {ckpt_path}')
    print(f'  Curves→ {os.path.join(args.save_dir, "overfit_test.png")}')

    return 0 if passed else 1


if __name__ == '__main__':
    sys.exit(main())


# import os
# import sys, argparse, csv
# import numpy as np
# import torch
# import torch.optim as optim
# from torch.utils.data import DataLoader, Subset

# # Ensure paths are correct
# _file_dir = os.path.dirname(os.path.abspath(__file__))
# _root_dir = os.path.dirname(_file_dir)
# for _p in [_root_dir, _file_dir,
#            os.path.join(_root_dir, 'common'),
#            os.path.join(_root_dir, 'data')]:
#     if _p not in sys.path:
#         sys.path.insert(0, _p)

# CONNECTIONS = [
#     (0,1),(1,2),(2,3),(3,4),
#     (0,5),(5,6),(6,7),(7,8),
#     (0,9),(9,10),(10,11),(11,12),
#     (0,13),(13,14),(14,15),(15,16),
#     (0,17),(17,18),(18,19),(19,20),
# ]
# _FINGER_BGR = [(0,220,0),(255,100,0),(0,200,255),(0,100,255),(200,0,200)]
# _JOINT_RGB  = [(128,128,128)] + [c for c in _FINGER_BGR for _ in range(4)]
# BONE_BGR    = [_FINGER_BGR[f] for f in range(5) for _ in range(4)]
# BONE_MPL    = ['#00dd00','#ff6400','#00c8ff','#0064ff','#c800c8']

# # --- Helper Functions ---
# def parse_args():
#     p = argparse.ArgumentParser()
#     p.add_argument('--freihand_dir', required=True)
#     p.add_argument('--resume',    default=None)
#     p.add_argument('--n_samples', type=int,   default=4)
#     p.add_argument('--iters',     type=int,   default=500)
#     p.add_argument('--lr',        type=float, default=1e-4)
#     p.add_argument('--save_dir',  default=os.path.join(_root_dir, 'overfit_output'))
#     p.add_argument('--gpu',       default='0')
#     return p.parse_args()

# def hdr(msg):
#     print('\n' + '─'*65 + f'\n  {msg}\n' + '─'*65)

# def procrustes_align(X, Y):
#     muX = X.mean(0); muY = Y.mean(0)
#     X0 = X - muX;    Y0 = Y - muY
#     nX = np.linalg.norm(X0); nY = np.linalg.norm(Y0)
#     X0 /= (nX + 1e-8); Y0 /= (nY + 1e-8)
#     U, _, Vt = np.linalg.svd(X0.T @ Y0)
#     s = nY / (nX + 1e-8)
#     return s * ((U @ Vt) @ X.T).T + muY - s * ((U @ Vt) @ muX)

# # --- PLY Export ---
# def save_mesh_ply(path, verts, faces, color=(180,130,100)):
#     V, F = len(verts), len(faces)
#     r, g, b = color
#     with open(path, 'w') as fp:
#         fp.write('ply\nformat ascii 1.0\n'
#                  f'element vertex {V}\n'
#                  'property float x\nproperty float y\nproperty float z\n'
#                  'property uchar red\nproperty uchar green\nproperty uchar blue\n'
#                  f'element face {F}\n'
#                  'property list uchar int vertex_indices\nend_header\n')
#         for v in verts:
#             fp.write(f'{v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {r} {g} {b}\n')
#         for f in faces:
#             fp.write(f'3 {f[0]} {f[1]} {f[2]}\n')

# def save_joints_ply(path, joints, label='joints'):
#     with open(path, 'w') as fp:
#         fp.write(f'ply\nformat ascii 1.0\ncomment {label}\n'
#                  f'element vertex {len(joints)}\n'
#                  'property float x\nproperty float y\nproperty float z\n'
#                  'property uchar red\nproperty uchar green\nproperty uchar blue\n'
#                  f'element edge {len(CONNECTIONS)}\n'
#                  'property int vertex1\nproperty int vertex2\nend_header\n')
#         for ji, j in enumerate(joints):
#             r, g, b = _JOINT_RGB[ji]
#             fp.write(f'{j[0]:.6f} {j[1]:.6f} {j[2]:.6f} {r} {g} {b}\n')
#         for s, e in CONNECTIONS:
#             fp.write(f'{s} {e}\n')

# # (Drawing functions project_pts, draw_skeleton_2d, etc. excluded for brevity, 
# # assume they are the same as your original provided code)

# def main():
#     args = parse_args()
#     os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#     save_dir = args.save_dir
#     pred_dir = os.path.join(save_dir, 'predictions')
#     ply_dir  = os.path.join(save_dir, 'ply')
#     os.makedirs(pred_dir, exist_ok=True)
#     os.makedirs(ply_dir,  exist_ok=True)

#     # --- Config & Data ---
#     from main.config import cfg
#     cfg.freihand_data_dir = args.freihand_dir
#     cfg.set_args(args.gpu, 'overfit_test', continue_train=False)
#     cfg.calc_mutliscale_dim(cfg.use_big_decoder, cfg.resnet_type)
    
#     from freihand import Dataset as FreiHandDataset
#     full_ds = FreiHandDataset('train')
#     n = min(args.n_samples, len(full_ds))
#     loader = DataLoader(Subset(full_ds, list(range(n))), batch_size=n, shuffle=False)
#     f_inp, f_tgt, f_meta = next(iter(loader))
    
#     def to_dev(d): return {k: v.to(device) if torch.is_tensor(v) else v for k,v in d.items()}
#     f_inp, f_tgt, f_meta = to_dev(f_inp), to_dev(f_tgt), to_dev(f_meta)

#     # --- Model ---
#     from main.model import get_model
#     model = get_model('train').to(device)
#     if args.resume:
#         state = torch.load(args.resume, map_location='cpu')
#         model.load_state_dict(state.get('network', state), strict=True)

#     mano_layer = model.mano_head.mano_layer
#     mano_faces = mano_layer.th_faces.cpu().numpy()
    
#     # Weight settings
#     W = {'sdfhand_loss': cfg.sdf_hand_weight, 'joint_heatmap': cfg.hm_weight, 
#          'loss_joint_3d': cfg.joint_weight, 'loss_all_joint_3d': cfg.joint_weight,
#          'mano_mesh_loss': 3.0, 'mano_joint_loss': 3.0, 'pose_param_loss': 3.0, 'shape_param_loss': 3.0}

#     opt = optim.AdamW(model.parameters(), lr=args.lr)
#     best_state = None
#     best_mje = 1e9

#     # --- Training Loop (Standard Overfit) ---
#     for it in range(1, args.iters + 1):
#         model.train()
#         opt.zero_grad()
#         out = model(f_inp, f_tgt, f_meta, 'train', epoch_cnt=0)
#         total = sum(out[k].mean() * W[k] for k in W if k in out)
#         total.backward()
#         opt.step()

#         if it % 50 == 0:
#             print(f"Iter {it} Loss: {total.item():.4f}")

#     # --- INFERENCE SECTION: HYBRID ALIGNMENT ---
#     hdr("Final Inference & Hybrid Alignment")
#     model.eval()
#     with torch.no_grad():
#         ev = model(f_inp, f_tgt, f_meta, 'eval', epoch_cnt=0)

#     # Extract raw outputs
#     vote_j20_np = ev['hand_joints_out'].cpu().numpy()     # (B, 20, 3) rr m
#     mano_j21_np = ev['mano_joints_out'].cpu().numpy()     # (B, 21, 3) local m
#     pred_v_np   = ev['mano_mesh_out'].cpu().numpy()       # (B, 778, 3) local m
#     gt_j21_np   = (f_tgt['joint_cam_no_trans']/1000).cpu().numpy()
#     mano_roots  = f_meta['mano_root'].cpu().numpy()
#     cam_intrs   = f_meta['cam_intr'].cpu().numpy()

#     # Get LBS weights for deformation (V, J) -> (778, 21)
#     # We use these to deform the mesh based on joint displacement
#     weights = mano_layer.th_weights.cpu().numpy()

#     for si in range(n):
#         sid = si
#         # 1. Prepare Joint Targets (Voting Head)
#         # Root (0,0,0) + 20 joints = 21 joints
#         vote_rr = np.concatenate([np.zeros((1,3)), vote_j20_np[si]], axis=0)
        
#         # 2. Prepare Source (MANO Head)
#         # Convert MANO to root-relative
#         mano_root_pt = mano_j21_np[si, 0:1]
#         mano_rr_orig = mano_j21_np[si] - mano_root_pt
#         pred_v_orig  = pred_v_np[si] - mano_root_pt
#         # 3. MESH WARPING (LBS-based)
#         # Calculate how much each joint needs to move to match Voting Head
#         joint_diffs = vote_rr - mano_rr_orig # (21, 3)
        
#         # FIX: MANO weights are defined for 16 joints (wrist + 3 joints per finger)
#         # Finger tips (joints 17-21 in some conventions) are not in the LBS weight matrix.
#         # We slice joint_diffs to the first 16 joints to match weights (778, 16).
#         joint_diffs_16 = joint_diffs[:16, :] 
        
#         # Vertices follow the joints they are weighted to
#         # (778, 16) @ (16, 3) -> (778, 3) displacement field
#         vert_displacements = weights @ joint_diffs_16
        
#         # New mesh that now aligns with the skeleton
#         pred_v_aligned = pred_v_orig + vert_displacements
#         # 4. Save Outputs
#         save_mesh_ply(os.path.join(ply_dir, f'sample_{sid:04d}_pred_mesh.ply'), 
#                       pred_v_aligned, mano_faces)
#         save_joints_ply(os.path.join(ply_dir, f'sample_{sid:04d}_pred_joints.ply'), 
#                         mano_rr_aligned)
#         save_joints_ply(os.path.join(ply_dir, f'sample_{sid:04d}_gt_joints.ply'), 
#                         gt_rr)

#         mje = np.linalg.norm(mano_rr_aligned - gt_rr, axis=1).mean() * 1000.0
#         print(f"Sample {sid} | Aligned MJE: {mje:.2f}mm")

#     print(f"\nResults saved to {args.save_dir}")

# if __name__ == '__main__':
#     main()