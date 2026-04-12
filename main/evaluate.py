"""
evaluate.py — Full evaluation of hand-only HOISDF on FreiHAND
==============================================================

Metric groups
─────────────
  [mm / absolute]
    MPJPE       root-relative mean per-joint error (mm)
    PA-MPJPE    Procrustes-aligned MPJPE (mm)
    3D-MPJPE    camera-space MPJPE, NOT root-relative (mm)

  [scale-normalized, still mm]
    NMPJPE      MPJPE after rescaling pred bone-length to match GT

  [dimensionless, 0–1 / 0–100%]
    PCK@20mm    % joints within 20mm  (joint metric)
    PCK@50mm    % joints within 50mm  (joint metric)
    AUC-J       area under joint PCK curve 0–50mm  (joint metric)
    F@5mm       fraction of vertices within 5mm    (mesh metric)
    F@15mm      fraction of vertices within 15mm   (mesh metric)
    VAUC        area under vertex PCK 0–50mm       (mesh metric)

  [bbox-normalized, dimensionless]
    MPJPE_N     MPJPE / bbox_diagonal  (0..1, no unit)
    PA_MPJPE_N  PA-MPJPE / bbox_diagonal

  Both joint heads (voting = hand_joints_out, MANO = mano_joints_out)
  evaluated separately for every metric.

Outputs  (--save_dir)
  {name}_results.json / .csv
  {name}_per_joint.png        per-joint MPJPE bar (both heads)
  {name}_pck_curve.png        PCK + AUC curve (both heads)
  all_results.json / all_results.csv
  comparison_table.txt        plain text + LaTeX table
  comparison_bar.png          multi-checkpoint bar chart

Usage
  python main/evaluate.py \\
      --freihand_dir /path/to/FreiHAND \\
      --model_path /path/to/snapshot_69.pth.tar \\
      --save_dir /scratch/kghasemz/eval_out

  # ablation: multiple checkpoints
  python main/evaluate.py \\
      --freihand_dir /path/to/FreiHAND \\
      --model_path snap_45.pth.tar snap_69.pth.tar \\
      --names ep45 ep69
"""

import os
os.environ['MPLBACKEND']   = 'Agg'
os.environ['DISPLAY']      = ''
os.environ['MPLCONFIGDIR'] = '/tmp'

import sys, argparse, json, csv
import numpy as np
import torch
from torch.utils.data import DataLoader

_file_dir = os.path.dirname(os.path.abspath(__file__))
_root_dir  = os.path.dirname(_file_dir)
for _p in [_root_dir, _file_dir,
           os.path.join(_root_dir, 'common'),
           os.path.join(_root_dir, 'data')]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

JOINT_NAMES = [
    'Wrist',
    'Thumb_MCP','Thumb_PIP','Thumb_DIP','Thumb_Tip',
    'Index_MCP','Index_PIP','Index_DIP','Index_Tip',
    'Mid_MCP',  'Mid_PIP',  'Mid_DIP',  'Mid_Tip',
    'Ring_MCP', 'Ring_PIP', 'Ring_DIP', 'Ring_Tip',
    'Pinky_MCP','Pinky_PIP','Pinky_DIP','Pinky_Tip',
]


# ─────────────────────────────────────────────────────────────────────────────
# Core geometry helpers
# ─────────────────────────────────────────────────────────────────────────────

def _procrustes(pred, gt):
    """
    Batch Procrustes alignment: scale + rotation, no shear.
    pred, gt : (N, J, 3)
    Returns  : aligned pred (N, J, 3)
    """
    N   = pred.shape[0]
    out = np.zeros_like(pred)
    for i in range(N):
        p, g = pred[i], gt[i]
        pc = p - p.mean(0);  gc = g - g.mean(0)
        ps = np.linalg.norm(pc); gs = np.linalg.norm(gc)
        if ps < 1e-9:
            out[i] = p; continue
        pn = pc / ps;  gn = gc / gs
        U, _, Vt = np.linalg.svd(gn.T @ pn)
        d = np.linalg.det(U @ Vt)
        R = U @ np.diag([1, 1, d]) @ Vt
        out[i] = gs * (pn @ R.T) + g.mean(0)
    return out


def _bbox_diag(gt_rr):
    """
    Per-sample bounding box diagonal of GT joints (metres).
    gt_rr : (N, J, 3) root-relative
    Returns (N,) diagonal lengths in metres.
    """
    lo  = gt_rr.min(axis=1)   # (N, 3)
    hi  = gt_rr.max(axis=1)
    return np.linalg.norm(hi - lo, axis=-1).clip(min=1e-6)  # (N,)


# ─────────────────────────────────────────────────────────────────────────────
# Joint metrics
# All pred/gt arrays are (N, 21, 3) numpy, root-relative, in METRES
# ─────────────────────────────────────────────────────────────────────────────

def metric_mpjpe(pred_rr, gt_rr):
    """MPJPE in mm. Returns (mean_mm, per_joint_mm (21,))."""
    err = np.linalg.norm(pred_rr - gt_rr, axis=-1)          # (N, 21)
    return err.mean() * 1000, err.mean(0) * 1000


def metric_pa_mpjpe(pred_rr, gt_rr):
    """PA-MPJPE in mm after Procrustes alignment."""
    aligned = _procrustes(pred_rr, gt_rr)
    err     = np.linalg.norm(aligned - gt_rr, axis=-1)
    return err.mean() * 1000, err.mean(0) * 1000


def metric_nmpjpe(pred_rr, gt_rr):
    """
    NMPJPE (scale-normalized MPJPE, mm).
    Scale each sample's prediction so its mean bone-length matches GT,
    then compute MPJPE.  Removes global scale ambiguity.
    """
    p_scale = np.linalg.norm(pred_rr, axis=-1).mean(1, keepdims=True)  # (N,1)
    g_scale = np.linalg.norm(gt_rr,   axis=-1).mean(1, keepdims=True)
    scale   = g_scale / (p_scale + 1e-9)                                # (N,1)
    pred_s  = pred_rr * scale[:, :, None]
    err     = np.linalg.norm(pred_s - gt_rr, axis=-1)
    return err.mean() * 1000, err.mean(0) * 1000


def metric_mpjpe_3d(pred_cam, gt_cam):
    """3D MPJPE: camera-space (NOT root-relative), mm."""
    err = np.linalg.norm(pred_cam - gt_cam, axis=-1)
    return err.mean() * 1000, err.mean(0) * 1000


def metric_mpjpe_norm(pred_rr, gt_rr):
    """
    Bbox-normalized MPJPE (dimensionless, 0..1).
    Divides per-sample MPJPE by the GT joint bounding-box diagonal.
    Returns (mean_ratio, per_joint_ratio (21,)).
    """
    diag = _bbox_diag(gt_rr)                          # (N,)
    err  = np.linalg.norm(pred_rr - gt_rr, axis=-1)  # (N,21)
    norm_err = err / diag[:, None]                     # (N,21)
    return norm_err.mean(), norm_err.mean(0)


def metric_pa_mpjpe_norm(pred_rr, gt_rr):
    """Bbox-normalized PA-MPJPE (dimensionless, 0..1)."""
    aligned  = _procrustes(pred_rr, gt_rr)
    diag     = _bbox_diag(gt_rr)
    err      = np.linalg.norm(aligned - gt_rr, axis=-1)
    norm_err = err / diag[:, None]
    return norm_err.mean(), norm_err.mean(0)


def metric_pck3d(pred_rr, gt_rr, thr_m):
    """PCK-3D: % joints within thr_m metres. Returns 0–100."""
    return (np.linalg.norm(pred_rr - gt_rr, axis=-1) < thr_m).mean() * 100


def metric_auc3d(pred_rr, gt_rr, n=100, max_m=0.05):
    """
    AUC under joint PCK curve from 0 to max_m metres.
    Returns (auc_pct 0–100, pck_array_pct, thresholds_mm).
    """
    thrs = np.linspace(0, max_m, n + 1)[1:]
    err  = np.linalg.norm(pred_rr - gt_rr, axis=-1)
    pcks = np.array([(err < t).mean() for t in thrs])
    auc  = np.trapz(pcks, thrs) / max_m * 100
    return auc, pcks * 100, thrs * 1000


# ─────────────────────────────────────────────────────────────────────────────
# Mesh metrics
# pred_v, gt_v : (N, 778, 3) metres
# ─────────────────────────────────────────────────────────────────────────────

def metric_mme(pv, gv):
    return np.linalg.norm(pv - gv, axis=-1).mean() * 1000


def metric_pa_mme(pv, gv):
    return np.linalg.norm(_procrustes(pv, gv) - gv, axis=-1).mean() * 1000


def metric_fscore(pv, gv, thr_mm):
    """F@thr: fraction of vertices within thr_mm. Returns 0–100."""
    return (np.linalg.norm(pv - gv, axis=-1) * 1000 < thr_mm).mean() * 100


def metric_vauc(pv, gv, n=50, max_mm=50.0):
    """Vertex AUC 0–max_mm mm. Returns 0–100."""
    thrs = np.linspace(0, max_mm, n + 1)[1:]
    err  = np.linalg.norm(pv - gv, axis=-1) * 1000
    return np.trapz([(err < t).mean() for t in thrs], thrs) / max_mm * 100


def get_gt_verts_batch(mano_layer, mp_np, device):
    """mp_np (N,58) -> (N,778,3) metres, or None if all zeros."""
    if np.abs(mp_np).sum() < 1e-6:
        return None
    pose  = torch.from_numpy(mp_np[:, :48]).float().to(device)
    shape = torch.from_numpy(mp_np[:, 48:]).float().to(device)
    with torch.no_grad():
        v, _ = mano_layer(th_pose_coeffs=pose, th_betas=shape)
    return v.cpu().numpy() / 1000.0


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

def run_inference(model, loader, device):
    model.eval()
    bufs = {k: [] for k in ['vote_rr','mano_rr','pred_mesh',
                             'gt_rr','gt_cam','vote_cam','mano_params']}
    with torch.no_grad():
        for bi, (inp, tgt, meta) in enumerate(loader):
            def td(d):
                return {k: v.to(device) if torch.is_tensor(v) else v
                        for k, v in d.items()}
            out = model(td(inp), td(tgt), td(meta), 'eval', epoch_cnt=1e8)
            B  = inp['img'].shape[0]
            mr = meta['mano_root'].cpu().numpy()

            v20     = out['hand_joints_out'].cpu().numpy()            # (B,20,3)
            v21     = np.concatenate([np.zeros((B,1,3)), v20], 1)
            vote_rr = v21 - v21[:, 0:1]
            vote_cam= vote_rr + mr[:, None, :]

            m21     = out['mano_joints_out'].cpu().numpy()            # (B,21,3)
            mano_rr = m21 - m21[:, 0:1]

            gt      = (tgt['joint_cam_no_trans'] / 1000).numpy()
            gt_rr   = gt - gt[:, 0:1]
            gt_cam  = gt_rr + mr[:, None, :]

            bufs['vote_rr'].append(vote_rr)
            bufs['mano_rr'].append(mano_rr)
            bufs['pred_mesh'].append(out['mano_mesh_out'].cpu().numpy())
            bufs['gt_rr'].append(gt_rr)
            bufs['gt_cam'].append(gt_cam)
            bufs['vote_cam'].append(vote_cam)
            bufs['mano_params'].append(tgt['mano_param'].numpy())

            if (bi + 1) % 20 == 0:
                n_done = min((bi + 1) * loader.batch_size, len(loader.dataset))
                print(f'  [{n_done}/{len(loader.dataset)}]', end='\r')
    print()
    return {k: np.concatenate(v, 0) for k, v in bufs.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

def _ax(ax):
    ax.set_facecolor('#1a1a1a')
    ax.tick_params(colors='#aaa', labelsize=8)
    for sp in ax.spines.values(): sp.set_color('#444')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', color='#333', ls='--', lw=0.5)


def plot_per_joint(v_pj, m_pj, title, path):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    x = np.arange(21); w = 0.38
    fig, ax = plt.subplots(figsize=(14, 4), facecolor='#0d0d0d')
    _ax(ax)
    b1 = ax.bar(x - w/2, v_pj, w, label='Voting head', color='#38BDF8', alpha=0.9)
    b2 = ax.bar(x + w/2, m_pj, w, label='MANO head',   color='#A78BFA', alpha=0.9)
    for bar, val in zip(list(b1)+list(b2), list(v_pj)+list(m_pj)):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.15,
                f'{val:.1f}', ha='center', va='bottom', color='white', fontsize=5.5)
    ax.set_xticks(x)
    ax.set_xticklabels(JOINT_NAMES, rotation=45, ha='right', fontsize=7, color='#aaa')
    ax.set_ylabel('Error (mm)', color='#aaa')
    ax.set_title(title, color='white', fontsize=10)
    ax.legend(fontsize=8, facecolor='#333', edgecolor='none', labelcolor='white')
    plt.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches='tight', facecolor='#0d0d0d')
    plt.close(fig)
    print(f'  Per-joint bar   → {path}')


def plot_pck(v_pcks, m_pcks, thrs, v_auc, m_auc, name, path):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 5), facecolor='#0d0d0d')
    _ax(ax)
    ax.plot(thrs, v_pcks, '#38BDF8', lw=2.5, label=f'Voting  AUC={v_auc:.1f}%')
    ax.fill_between(thrs, v_pcks, alpha=0.15, color='#38BDF8')
    ax.plot(thrs, m_pcks, '#A78BFA', lw=2.5, ls='--', label=f'MANO   AUC={m_auc:.1f}%')
    ax.fill_between(thrs, m_pcks, alpha=0.12, color='#A78BFA')
    ax.set_xlabel('Threshold (mm)', color='#aaa')
    ax.set_ylabel('PCK (%)', color='#aaa')
    ax.set_ylim(0, 105)
    ax.set_title(f'PCK-3D — {name}', color='white', fontsize=10)
    ax.legend(fontsize=9, facecolor='#333', edgecolor='none', labelcolor='white')
    plt.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches='tight', facecolor='#0d0d0d')
    plt.close(fig)
    print(f'  PCK curve       → {path}')


def plot_comparison_bar(all_results, path):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # show both mm and normalized columns
    cols = [
        ('mpjpe_mm',       'MPJPE',         'mm',   True),
        ('pa_mpjpe_mm',    'PA-MPJPE',      'mm',   True),
        ('nmpjpe_mm',      'NMPJPE',        'mm',   True),
        ('mpjpe_norm',     'MPJPE_N',       '×1e3', True),
        ('auc_vote',       'AUC-J',         '%',    False),
        ('pck20_vote',     'PCK@20mm',      '%',    False),
        ('f5_pct',         'F@5mm',         '%',    False),
        ('vauc_pct',       'VAUC',          '%',    False),
    ]
    names  = [r['name'] for r in all_results]
    colors = ['#38BDF8','#6EE7B7','#FCD34D','#F87171','#C084FC']
    fig, axes = plt.subplots(2, 4, figsize=(22, 8), facecolor='#0d0d0d')
    fig.suptitle('Metric Comparison', color='white', fontsize=13)
    for ax, (key, lbl, unit, lib) in zip(axes.flat, cols):
        _ax(ax)
        # for mpjpe_norm scale to ×1e3 for readability
        scale = 1e3 if key == 'mpjpe_norm' else 1.0
        vals  = [r.get(key, float('nan')) * scale for r in all_results]
        valid = [v for v in vals if not np.isnan(v)]
        if not valid:
            ax.set_title(f'{lbl} — N/A', color='#888', fontsize=9); continue
        bars = ax.bar(names, vals,
                      color=[colors[i % len(colors)] for i in range(len(names))],
                      width=0.55, alpha=0.9)
        mx = max(valid)
        for bar, val in zip(bars, vals):
            if np.isnan(val): continue
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+mx*0.01,
                    f'{val:.2f}', ha='center', va='bottom', color='white', fontsize=8)
        ax.set_title(f'{lbl} ({unit}) {"↓" if lib else "↑"}', color='white', fontsize=9)
        plt.setp(ax.get_xticklabels(), rotation=20, ha='right', color='#aaa')
    plt.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches='tight', facecolor='#0d0d0d')
    plt.close(fig)
    print(f'  Comparison bar  → {path}')


# ─────────────────────────────────────────────────────────────────────────────
# Comparison table — plain text + LaTeX
# ─────────────────────────────────────────────────────────────────────────────

def write_comparison_table(all_results, path):
    nan = float('nan')

    def _f(r, key, fmt='.2f', fallback='  N/A'):
        v = r.get(key, nan)
        return f'{v:{fmt}}' if not np.isnan(v) else fallback

    # ── plain text ─────────────────────────────────────────────────────────────
    hdr1 = (f'{"Method":<26}'
            f'{"MPJPE":>8} {"PA":>7} {"NMPJPE":>8} {"3D-MPJPE":>9}'
            f' | {"MPJPE_N":>8} {"PA_N":>7}'
            f' | {"PCK@20":>7} {"PCK@50":>7} {"AUC-J":>7}'
            f' | {"F@5":>6} {"F@15":>6} {"VAUC":>6}')
    hdr2 = (f'{"":26}'
            f'{"(mm)":>8} {"(mm)":>7} {"(mm)":>8} {"(mm)":>9}'
            f' | {"(×1e3)":>8} {"(×1e3)":>7}'
            f' | {"(%)":>7} {"(%)":>7} {"(%)":>7}'
            f' | {"(%)":>6} {"(%)":>6} {"(%)":>6}')
    sep  = '─' * len(hdr1)
    lines = [sep, hdr1, hdr2, sep]

    for r in all_results:
        nm   = _f(r, 'mpjpe_norm', fmt='.4f')
        pa_n = _f(r, 'pa_mpjpe_norm', fmt='.4f')
        # scale to ×1e3 for readability in table
        nm_s   = f'{float(nm)*1e3:.2f}'   if nm   != '  N/A' else '  N/A'
        pa_n_s = f'{float(pa_n)*1e3:.2f}' if pa_n != '  N/A' else '  N/A'
        line = (f'{r["name"]:<26}'
                f'{_f(r,"mpjpe_mm"):>8} {_f(r,"pa_mpjpe_mm"):>7}'
                f' {_f(r,"nmpjpe_mm"):>8} {_f(r,"mpjpe_3d_mm"):>9}'
                f' | {nm_s:>8} {pa_n_s:>7}'
                f' | {_f(r,"pck20_vote"):>7} {_f(r,"pck50_vote"):>7}'
                f' {_f(r,"auc_vote"):>7}'
                f' | {_f(r,"f5_pct"):>6} {_f(r,"f15_pct"):>6}'
                f' {_f(r,"vauc_pct"):>6}')
        lines.append(line)

    lines += [sep,
              'mm metrics: lower is better  |  normalized: MPJPE_N/PA_N = error/bbox_diag (×1e3)',
              'PCK/AUC/F-score: higher is better  (all dimensionless, 0–100%)']

    # ── LaTeX ──────────────────────────────────────────────────────────────────
    latex = [
        r'\begin{table*}[h]', r'\centering',
        r'\caption{Hand Pose Estimation Results on FreiHAND}',
        r'\begin{tabular}{l|cccc|cc|ccc|ccc}', r'\toprule',
        r'\multirow{2}{*}{Method}'
        r' & \multicolumn{4}{c|}{Absolute (mm $\downarrow$)}'
        r' & \multicolumn{2}{c|}{Norm. ($\times10^{-3}$ $\downarrow$)}'
        r' & \multicolumn{3}{c|}{Joint (\% $\uparrow$)}'
        r' & \multicolumn{3}{c}{Mesh (\% $\uparrow$)} \\',
        r' & MPJPE & PA & NMPJPE & 3D-MPJPE'
        r' & MPJPE$_N$ & PA$_N$'
        r' & PCK@20 & PCK@50 & AUC'
        r' & F@5 & F@15 & VAUC \\ \midrule',
    ]
    for r in all_results:
        def lf(key, scale=1.0, fmt='.2f', fb='--'):
            v = r.get(key, nan)
            return f'{v*scale:{fmt}}' if not np.isnan(v) else fb
        latex.append(
            f'{r["name"]}'
            f' & {lf("mpjpe_mm")} & {lf("pa_mpjpe_mm")} & {lf("nmpjpe_mm")} & {lf("mpjpe_3d_mm")}'
            f' & {lf("mpjpe_norm", scale=1e3)} & {lf("pa_mpjpe_norm", scale=1e3)}'
            f' & {lf("pck20_vote")} & {lf("pck50_vote")} & {lf("auc_vote")}'
            f' & {lf("f5_pct")} & {lf("f15_pct")} & {lf("vauc_pct")}'
            r' \\')
    latex += [r'\bottomrule', r'\end{tabular}', r'\end{table*}']

    out = '\n'.join(lines) + '\n\n' + '\n'.join(latex) + '\n'
    with open(path, 'w') as f:
        f.write(out)
    print('\n' + '\n'.join(lines))
    print(f'\n  Table           → {path}')


# ─────────────────────────────────────────────────────────────────────────────
# Evaluate one checkpoint
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_one(model_path, name, loader, device, save_dir, mano_layer):
    from main.model import get_model
    sep = '─' * 60
    print(f'\n{sep}\n  {name}\n  {model_path}\n{sep}')

    raw   = torch.load(model_path, map_location='cpu')
    state = raw.get('network', raw.get('model_state', raw))
    state = {k.replace('module.', ''): v for k, v in state.items()}
    model = get_model('test').to(device)
    missing, _ = model.load_state_dict(state, strict=False)
    if missing:
        print(f'  WARNING: {len(missing)} missing keys')

    print('  Running inference...')
    data = run_inference(model, loader, device)
    N = data['vote_rr'].shape[0]
    print(f'  {N} samples')

    vr = data['vote_rr'];    mr = data['mano_rr']
    gr = data['gt_rr'];      gc = data['gt_cam']
    vc = data['vote_cam'];   pv = data['pred_mesh']
    mp = data['mano_params']

    nan = float('nan')

    # ── Voting head ────────────────────────────────────────────────────────────
    v_mpjpe,    v_pj      = metric_mpjpe(vr, gr)
    v_pa,       v_paj     = metric_pa_mpjpe(vr, gr)
    v_n,        v_nj      = metric_nmpjpe(vr, gr)
    v_3d,       v_3dj     = metric_mpjpe_3d(vc, gc)
    v_norm,     v_normj   = metric_mpjpe_norm(vr, gr)
    v_pa_norm,  v_pa_normj= metric_pa_mpjpe_norm(vr, gr)
    v_pck20               = metric_pck3d(vr, gr, 0.020)
    v_pck50               = metric_pck3d(vr, gr, 0.050)
    v_auc, v_pcks, thrs   = metric_auc3d(vr, gr)

    # ── MANO head ──────────────────────────────────────────────────────────────
    m_mpjpe,    m_pj      = metric_mpjpe(mr, gr)
    m_pa,       m_paj     = metric_pa_mpjpe(mr, gr)
    m_n,        m_nj      = metric_nmpjpe(mr, gr)
    m_norm,     m_normj   = metric_mpjpe_norm(mr, gr)
    m_pa_norm,  m_pa_normj= metric_pa_mpjpe_norm(mr, gr)
    m_pck20               = metric_pck3d(mr, gr, 0.020)
    m_pck50               = metric_pck3d(mr, gr, 0.050)
    m_auc, m_pcks, _      = metric_auc3d(mr, gr)

    # ── Mesh ───────────────────────────────────────────────────────────────────
    gt_v = get_gt_verts_batch(mano_layer, mp, device)
    hm   = gt_v is not None
    mme_v = pamme_v = f5_v = f15_v = vauc_v = nan
    if hm:
        mme_v   = metric_mme(pv, gt_v)
        pamme_v = metric_pa_mme(pv, gt_v)
        f5_v    = metric_fscore(pv, gt_v, 5.0)
        f15_v   = metric_fscore(pv, gt_v, 15.0)
        vauc_v  = metric_vauc(pv, gt_v)

    # ── Print ──────────────────────────────────────────────────────────────────
    def row(lbl, vv, mv=None, unit='', fmt='.2f'):
        vs = f'{vv:{fmt}}'
        ms = f'   MANO: {mv:{fmt}}' if mv is not None else ''
        print(f'  {lbl:<26} {vs:>9} {unit}{ms}')

    print(f'\n  {"Metric":<26} {"Voting":>9}       MANO')
    print(f'  {sep}')
    print(f'  [Absolute errors]')
    row('MPJPE',          v_mpjpe, m_mpjpe,  'mm')
    row('PA-MPJPE',       v_pa,    m_pa,     'mm')
    row('NMPJPE',         v_n,     m_n,      'mm')
    row('3D-MPJPE',       v_3d,    None,     'mm')
    print(f'  [Bbox-normalized (dimensionless)]')
    row('MPJPE_N  (×1e3)',v_norm*1e3,   m_norm*1e3,   '')
    row('PA_N     (×1e3)',v_pa_norm*1e3,m_pa_norm*1e3,'')
    print(f'  [PCK / AUC  (dimensionless %)]')
    row('PCK@20mm',       v_pck20, m_pck20,  '%')
    row('PCK@50mm',       v_pck50, m_pck50,  '%')
    row('AUC-J (0-50mm)', v_auc,   m_auc,    '%')
    print(f'  [Mesh — MANO head vs GT]')
    if hm:
        row('MME',        mme_v,   None, 'mm')
        row('PA-MME',     pamme_v, None, 'mm')
        row('F@5mm',      f5_v,    None,  '%')
        row('F@15mm',     f15_v,   None,  '%')
        row('VAUC',       vauc_v,  None,  '%')
    else:
        print('  Mesh metrics: N/A (mano_param=zeros in eval split)')
    print(f'  {sep}')

    print('\n  Per-joint MPJPE [mm] — Voting / MANO:')
    for jn, ve, me in zip(JOINT_NAMES, v_pj, m_pj):
        print(f'    {jn:<18}  vote={ve:.2f}  mano={me:.2f}')

    # ── Result dict ────────────────────────────────────────────────────────────
    res = dict(
        name=name, checkpoint=model_path, n_samples=int(N),
        # absolute mm
        mpjpe_mm       = float(v_mpjpe),
        pa_mpjpe_mm    = float(v_pa),
        nmpjpe_mm      = float(v_n),
        mpjpe_3d_mm    = float(v_3d),
        # bbox-normalized (dimensionless)
        mpjpe_norm     = float(v_norm),
        pa_mpjpe_norm  = float(v_pa_norm),
        # PCK / AUC (%)
        pck20_vote     = float(v_pck20),
        pck50_vote     = float(v_pck50),
        auc_vote       = float(v_auc),
        # MANO head
        mpjpe_mano     = float(m_mpjpe),
        pa_mpjpe_mano  = float(m_pa),
        nmpjpe_mano    = float(m_n),
        mpjpe_norm_mano= float(m_norm),
        pa_norm_mano   = float(m_pa_norm),
        pck20_mano     = float(m_pck20),
        pck50_mano     = float(m_pck50),
        auc_mano       = float(m_auc),
        # mesh
        mme_mm         = float(mme_v),
        pa_mme_mm      = float(pamme_v),
        f5_pct         = float(f5_v),
        f15_pct        = float(f15_v),
        vauc_pct       = float(vauc_v),
        has_mesh       = hm,
        # per-joint arrays
        mpjpe_pj_vote  = v_pj.tolist(),
        pa_pj_vote     = v_paj.tolist(),
        mpjpe_pj_mano  = m_pj.tolist(),
        pa_pj_mano     = m_paj.tolist(),
        norm_pj_vote   = v_normj.tolist(),
        norm_pj_mano   = m_normj.tolist(),
    )

    # ── Save ───────────────────────────────────────────────────────────────────
    os.makedirs(save_dir, exist_ok=True)
    safe = name.replace('/', '_').replace(' ', '_')

    with open(os.path.join(save_dir, f'{safe}_results.json'), 'w') as f:
        json.dump(res, f, indent=2)

    csv_keys = ['name',
                'mpjpe_mm','pa_mpjpe_mm','nmpjpe_mm','mpjpe_3d_mm',
                'mpjpe_norm','pa_mpjpe_norm',
                'pck20_vote','pck50_vote','auc_vote',
                'mpjpe_mano','pa_mpjpe_mano','nmpjpe_mano',
                'mpjpe_norm_mano','pa_norm_mano',
                'pck20_mano','pck50_mano','auc_mano',
                'mme_mm','pa_mme_mm','f5_pct','f15_pct','vauc_pct']
    with open(os.path.join(save_dir, f'{safe}_results.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=csv_keys, extrasaction='ignore')
        w.writeheader()
        w.writerow({k: (f'{res[k]:.6f}' if isinstance(res.get(k), float)
                        else res.get(k, '')) for k in csv_keys})
    print(f'\n  CSV  → {os.path.join(save_dir, safe+"_results.csv")}')

    plot_per_joint(v_pj, m_pj,
                   f'Per-Joint MPJPE — {name}\nVoting={v_mpjpe:.2f}mm  MANO={m_mpjpe:.2f}mm',
                   os.path.join(save_dir, f'{safe}_per_joint.png'))
    plot_pck(v_pcks, m_pcks, thrs, v_auc, m_auc, name,
             os.path.join(save_dir, f'{safe}_pck_curve.png'))
    return res


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--freihand_dir', required=True)
    p.add_argument('--model_path',   nargs='+', required=True)
    p.add_argument('--names',        nargs='*', default=None)
    p.add_argument('--save_dir',
                   default=os.path.join(_root_dir, 'outputs', 'eval_metrics'))
    p.add_argument('--batch_size',   type=int, default=22)
    p.add_argument('--gpu',          default='0')
    p.add_argument('--num_workers',  type=int, default=4)
    return p.parse_args()


def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    names = args.names or [
        os.path.basename(os.path.dirname(p)) for p in args.model_path]
    assert len(names) == len(args.model_path), \
        f'--names count ({len(names)}) != checkpoints ({len(args.model_path)})'

    from main.config import cfg
    cfg.freihand_data_dir = args.freihand_dir
    cfg.num_samp_obj = 0
    cfg.set_args(args.gpu, 'evaluate', continue_train=False)
    cfg.create_run_dirs()
    cfg.calc_mutliscale_dim(cfg.use_big_decoder, cfg.resnet_type)

    from freihand import Dataset as FreiHandDataset
    eval_ds = FreiHandDataset('evaluation')
    loader  = DataLoader(eval_ds, batch_size=args.batch_size,
                         shuffle=False, num_workers=args.num_workers,
                         pin_memory=True)
    print(f'Eval set: {len(eval_ds)} samples')

    from main.model import get_model
    _tmp = get_model('test')
    mano_layer = _tmp.mano_head.mano_layer.to(device)
    del _tmp

    os.makedirs(args.save_dir, exist_ok=True)
    all_results = []

    for mp, nm in zip(args.model_path, names):
        r = evaluate_one(mp, nm, loader, device, args.save_dir, mano_layer)
        all_results.append(r)

    write_comparison_table(all_results,
                           os.path.join(args.save_dir, 'comparison_table.txt'))

    with open(os.path.join(args.save_dir, 'all_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)

    csv_keys = ['name',
                'mpjpe_mm','pa_mpjpe_mm','nmpjpe_mm','mpjpe_3d_mm',
                'mpjpe_norm','pa_mpjpe_norm',
                'pck20_vote','pck50_vote','auc_vote',
                'mpjpe_mano','pa_mpjpe_mano','nmpjpe_mano',
                'mpjpe_norm_mano','pa_norm_mano',
                'pck20_mano','pck50_mano','auc_mano',
                'mme_mm','pa_mme_mm','f5_pct','f15_pct','vauc_pct']
    with open(os.path.join(args.save_dir, 'all_results.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=csv_keys, extrasaction='ignore')
        w.writeheader()
        for r in all_results:
            w.writerow({k: (f'{r[k]:.6f}' if isinstance(r.get(k), float)
                            else r.get(k, '')) for k in csv_keys})
    print(f'\n  All CSV  → {os.path.join(args.save_dir,"all_results.csv")}')
    print(f'  All JSON → {os.path.join(args.save_dir,"all_results.json")}')

    if len(all_results) > 1:
        plot_comparison_bar(all_results,
                            os.path.join(args.save_dir, 'comparison_bar.png'))

    print(f'\n  Done → {args.save_dir}/')


if __name__ == '__main__':
    main()