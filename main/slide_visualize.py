"""
slide_visualize.py — Slide-quality hand pose visualizations
============================================================
Selects every 1000th sample from the evaluation split (indices 0, 1000, 2000, …),
runs inference, and saves:

  {save_dir}/
    2d/
      sample_XXXXX.png   ← side-by-side GT vs Pred on the image
                            each finger a distinct colour, dark theme
    3d/
      sample_XXXXX.gif   ← 360° rotating 3D skeleton (pred + gt), dark bg

Usage (from HOISDF root):
    python main/slide_visualize.py \\
        --model_path  /path/to/snapshot_45_1331.pth.tar \\
        --freihand_dir /path/to/FreiHAND \\
        --save_dir    /path/to/slide_output \\
        --stride      1000 \\
        --gpu         0
"""

import os, sys, argparse
os.environ['MPLBACKEND']   = 'Agg'
os.environ['DISPLAY']      = ''
os.environ['MPLCONFIGDIR'] = '/tmp'

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

_file_dir = os.path.dirname(os.path.abspath(__file__))
_root_dir = os.path.dirname(_file_dir)
for _p in [_root_dir, _file_dir,
           os.path.join(_root_dir, 'common'),
           os.path.join(_root_dir, 'data')]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ── skeleton ──────────────────────────────────────────────────────────────────
# 21 joints: 0=wrist, then 4 fingers × 4 joints, then thumb × 4
CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),       # index
    (0,5),(5,6),(6,7),(7,8),       # middle
    (0,9),(9,10),(10,11),(11,12),  # ring
    (0,13),(13,14),(14,15),(15,16),# pinky
    (0,17),(17,18),(18,19),(19,20),# thumb
]

# Finger colour palette — rich, slide-friendly
FINGER_HEX = ['#FF6B6B', '#FFD93D', '#6BCB77', '#4D96FF', '#C77DFF']
# index=red  middle=yellow  ring=green  pinky=blue  thumb=violet
FINGER_RGB = [
    (255, 107, 107),   # index
    (255, 217,  61),   # middle
    (107, 203, 119),   # ring
    ( 77, 150, 255),   # pinky
    (199, 125, 255),   # thumb
]
# 4 bones per finger, in CONNECTIONS order
BONE_RGB   = [FINGER_RGB[f] for f in range(5) for _ in range(4)]
BONE_HEX   = [FINGER_HEX[f] for f in range(5) for _ in range(4)]
WRIST_RGB  = (220, 220, 220)

FINGER_NAMES = ['Index', 'Middle', 'Ring', 'Pinky', 'Thumb']


# ── args ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model_path',   required=True)
    p.add_argument('--freihand_dir', required=True)
    p.add_argument('--save_dir',     required=True)
    p.add_argument('--stride',       type=int,   default=1000)
    p.add_argument('--gpu',          default='0')
    p.add_argument('--gif_fps',      type=int,   default=30)
    p.add_argument('--gif_frames',   type=int,   default=90,
                   help='frames per 360° rotation')
    return p.parse_args()


# ── helpers ───────────────────────────────────────────────────────────────────

def project(pts_rr, mano_root, K):
    """Root-relative metres → image pixels."""
    cam = pts_rr + mano_root[None, :]
    p   = (K @ cam.T).T
    return p[:, :2] / p[:, 2:3].clip(min=1e-6)


def draw_hand_2d(ax, kpts_px, alpha=1.0, lw=2.5, r=6, label_finger=False):
    """Draw skeleton on matplotlib axes. kpts_px: (21,2)."""
    import matplotlib.patches as mpatches
    # bones
    for bi, (s, e) in enumerate(CONNECTIONS):
        c = [x/255 for x in BONE_RGB[bi]]
        ax.plot([kpts_px[s,0], kpts_px[e,0]],
                [kpts_px[s,1], kpts_px[e,1]],
                color=c, lw=lw, alpha=alpha, solid_capstyle='round',
                zorder=3)
    # joints
    ax.scatter(kpts_px[1:, 0], kpts_px[1:, 1],
               c=[f'#{FINGER_RGB[b//4][0]:02x}{FINGER_RGB[b//4][1]:02x}{FINGER_RGB[b//4][2]:02x}'
                  for b in range(20)],
               s=r**2, zorder=4, edgecolors='white', linewidths=0.8, alpha=alpha)
    # wrist
    ax.scatter([kpts_px[0,0]], [kpts_px[0,1]],
               c='white', s=r**2, zorder=4, edgecolors='#888', linewidths=0.8,
               alpha=alpha)


def draw_hand_3d(ax, joints, alpha=1.0, lw=2.0, js=40, elev=20, azim=0):
    """Draw 3D skeleton on 3D axes."""
    for bi, (s, e) in enumerate(CONNECTIONS):
        c = BONE_HEX[bi]
        ax.plot([joints[s,0], joints[e,0]],
                [joints[s,1], joints[e,1]],
                [joints[s,2], joints[e,2]],
                color=c, lw=lw, alpha=alpha,
                solid_capstyle='round', zorder=3)
    # finger joints
    colors_f = [BONE_HEX[b//4*4] for b in range(20)]
    ax.scatter(joints[1:,0], joints[1:,1], joints[1:,2],
               c=colors_f, s=js, zorder=4, depthshade=False,
               edgecolors='white', linewidths=0.5, alpha=alpha)
    # wrist
    ax.scatter([joints[0,0]], [joints[0,1]], [joints[0,2]],
               c='white', s=js, zorder=4, depthshade=False,
               edgecolors='#999', linewidths=0.5, alpha=alpha)


def style_3d(ax):
    """Dark minimal 3D axes."""
    ax.set_facecolor('#0d0d0d')
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor('#1a1a1a')
    ax.grid(False)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])


def eq_axes_3d(ax, *arrs):
    pts = np.vstack(arrs)
    mid = pts.mean(0)
    r   = max((pts.max(0) - pts.min(0)).max() / 2.0, 0.03)
    ax.set_xlim(mid[0]-r, mid[0]+r)
    ax.set_ylim(mid[1]-r, mid[1]+r)
    ax.set_zlim(mid[2]-r, mid[2]+r)


# ── 2D figure ─────────────────────────────────────────────────────────────────

def save_2d_figure(sid, img_tensor, gt_2d_hm, pred_rr, gt_rr,
                   mano_root, K, inp_res, hm_res, out_path):
    """
    Dark-theme side-by-side: GT | Predicted
    Each finger a distinct colour. Slide-ready.
    """
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import cv2

    hm_to_img = inp_res / hm_res   # 2.0

    # Image: tensor (C,H,W) -> (H,W,C) uint8, resize to inp_res
    img_np  = (img_tensor.permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
    img_bgr = cv2.resize(cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR),
                         (inp_res, inp_res))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # 2D coords
    gt_px   = gt_2d_hm * hm_to_img                          # (21,2) image px
    pred_px = project(pred_rr, mano_root, K)                 # (21,2) image px

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.patch.set_facecolor('#0d0d0d')

    titles = ['Ground Truth', 'Prediction']
    kpts_list = [gt_px, pred_px]

    for ax, title, kpts in zip(axes, titles, kpts_list):
        ax.set_facecolor('#0d0d0d')
        ax.imshow(img_rgb, extent=[0, inp_res, inp_res, 0], aspect='equal')
        ax.set_xlim(0, inp_res)
        ax.set_ylim(inp_res, 0)
        ax.axis('off')

        draw_hand_2d(ax, kpts, alpha=0.95, lw=2.8, r=6)

        ax.set_title(title, color='white', fontsize=16, fontweight='bold',
                     pad=10, fontfamily='monospace')

    # legend: one entry per finger
    handles = [mpatches.Patch(color=FINGER_HEX[i], label=FINGER_NAMES[i])
               for i in range(5)]
    handles += [mpatches.Patch(color='white', label='Wrist')]
    fig.legend(handles=handles, loc='lower center', ncol=6,
               facecolor='#1a1a1a', edgecolor='none',
               labelcolor='white', fontsize=11,
               bbox_to_anchor=(0.5, 0.0), framealpha=0.9)

    # sample id watermark
    fig.text(0.5, 0.97, f'Sample {sid}',
             ha='center', va='top', color='#555', fontsize=10,
             fontfamily='monospace')

    plt.tight_layout(rect=[0, 0.07, 1, 0.97])
    plt.savefig(out_path, dpi=150, bbox_inches='tight',
                facecolor='#0d0d0d', edgecolor='none')
    plt.close(fig)


# ── 3D GIF ────────────────────────────────────────────────────────────────────

def save_3d_gif(sid, pred_rr, gt_rr, out_path, n_frames=90, fps=30):
    """
    360° rotating 3D skeleton GIF.
    Left panel: GT (dimmer). Right panel: Pred (bright).
    Dark background, slide-ready.
    """
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    from PIL import Image
    import io

    frames = []
    for fi in range(n_frames):
        azim = fi * 360.0 / n_frames

        fig = plt.figure(figsize=(10, 5))
        fig.patch.set_facecolor('#0d0d0d')

        # GT panel
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        ax1.set_facecolor('#0d0d0d')
        style_3d(ax1)
        draw_hand_3d(ax1, gt_rr, alpha=0.9, lw=2.2, js=35)
        eq_axes_3d(ax1, gt_rr, pred_rr)
        ax1.view_init(elev=20, azim=azim)
        ax1.set_title('Ground Truth', color='white', fontsize=13,
                      fontweight='bold', pad=6, fontfamily='monospace')

        # Pred panel
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        ax2.set_facecolor('#0d0d0d')
        style_3d(ax2)
        draw_hand_3d(ax2, pred_rr, alpha=0.9, lw=2.2, js=35)
        eq_axes_3d(ax2, gt_rr, pred_rr)
        ax2.view_init(elev=20, azim=azim)
        ax2.set_title('Prediction', color='white', fontsize=13,
                      fontweight='bold', pad=6, fontfamily='monospace')

        fig.text(0.5, 0.97, f'Sample {sid}  |  3D hand pose',
                 ha='center', va='top', color='#444', fontsize=9,
                 fontfamily='monospace')

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100,
                    facecolor='#0d0d0d', edgecolor='none', bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        frames.append(Image.open(buf).copy())
        buf.close()

    # Save GIF
    duration_ms = int(1000 / fps)
    frames[0].save(
        out_path,
        save_all=True, append_images=frames[1:],
        duration=duration_ms, loop=0, optimize=False,
    )


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dir_2d = os.path.join(args.save_dir, '2d')
    dir_3d = os.path.join(args.save_dir, '3d')
    os.makedirs(dir_2d, exist_ok=True)
    os.makedirs(dir_3d, exist_ok=True)

    print(f'  Device   : {device}')
    print(f'  Checkpoint: {args.model_path}')
    print(f'  FreiHAND : {args.freihand_dir}')
    print(f'  Stride   : every {args.stride} samples')
    print(f'  Save to  : {args.save_dir}')

    # ── config ────────────────────────────────────────────────────────────
    from main.config import cfg
    cfg.freihand_data_dir = args.freihand_dir
    cfg.set_args(args.gpu, 'slide_vis', continue_train=False)
    cfg.create_run_dirs()
    cfg.calc_mutliscale_dim(cfg.use_big_decoder, cfg.resnet_type)
    cfg.num_samp_obj = 0

    inp_res = cfg.input_img_shape[0]   # 256
    hm_res  = cfg.output_hm_shape[0]   # 128

    # ── dataset ───────────────────────────────────────────────────────────
    from freihand import Dataset as FreiHandDataset
    ds = FreiHandDataset('evaluation')
    n  = len(ds)
    # Every stride-th sample: 0, 1000, 2000, …
    indices = list(range(0, n, args.stride))
    print(f'\n  Eval set  : {n} samples')
    print(f'  Selected  : {len(indices)} samples → {indices}')

    loader = DataLoader(
        Subset(ds, indices), batch_size=1,
        shuffle=False, num_workers=0,
    )

    # ── model ─────────────────────────────────────────────────────────────
    from main.model import get_model
    model = get_model('test').to(device)

    raw   = torch.load(args.model_path, map_location='cpu')
    state = raw.get('network', raw.get('model_state', raw))
    state = {k.replace('module.',''): v for k,v in state.items()}
    missing, _ = model.load_state_dict(state, strict=True)
    if missing:
        raise RuntimeError(f'Missing keys: {missing[:5]}')
    model.eval()
    print(f'  Checkpoint loaded OK.')

    # ── inference + visualise ─────────────────────────────────────────────
    for i, (inputs, targets, meta) in enumerate(loader):
        if i > 10:
            break
        sid = indices[i]

        inputs_d  = {k: v.to(device) if torch.is_tensor(v) else v
                     for k,v in inputs.items()}
        targets_d = {k: v.to(device) if torch.is_tensor(v) else v
                     for k,v in targets.items()}
        meta_d    = {k: v.to(device) if torch.is_tensor(v) else v
                     for k,v in meta.items()}

        with torch.no_grad():
            out = model(inputs_d, targets_d, meta_d, 'eval')

        # Voting head: (1,20,3) rr m → prepend wrist → (21,3)
        vote_j20 = out['hand_joints_out'][0].cpu().numpy()
        pred_rr  = np.concatenate([np.zeros((1,3)), vote_j20], 0)

        # GT: mm → m, root-relative
        gt_j21 = targets['joint_cam_no_trans'][0].cpu().numpy() / 1000.0
        gt_rr  = gt_j21 - gt_j21[0:1]

        mano_root = meta['mano_root'][0].cpu().numpy()
        K         = meta['cam_intr'][0].cpu().numpy()
        gt_2d_hm  = targets['joint_coord'][0].cpu().numpy()  # (21,2) hm px

        path_2d = os.path.join(dir_2d, f'sample_{sid:05d}.png')
        path_3d = os.path.join(dir_3d, f'sample_{sid:05d}.gif')

        save_2d_figure(sid, inputs['img'][0], gt_2d_hm, pred_rr, gt_rr,
                       mano_root, K, inp_res, hm_res, path_2d)

        save_3d_gif(sid, pred_rr, gt_rr, path_3d,
                    n_frames=args.gif_frames, fps=args.gif_fps)

        # MJE for info
        mje_mm = np.linalg.norm(pred_rr - gt_rr, axis=1).mean() * 1000.0
        print(f'  [{i+1:>3}/{len(indices)}] sample {sid:5d}  '
              f'MJE={mje_mm:6.1f}mm  → 2d/{os.path.basename(path_2d)}  '
              f'3d/{os.path.basename(path_3d)}')

    print(f'\n  Done.')
    print(f'  2D images : {dir_2d}/')
    print(f'  3D GIFs   : {dir_3d}/')


if __name__ == '__main__':
    main()