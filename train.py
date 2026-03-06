"""
overfit_test.py — Sanity check: overfit on a tiny fixed batch
=============================================================

Root cause of previous failures
---------------------------------
The heatmap loss kept producing trivial solutions because plain MSE on a
128×128 map is dominated by ~16000 background pixels vs ~50 foreground pixels
near the peak. The network minimizes loss by outputting -10 everywhere
(matching the clamped background) while ignoring the peak entirely.

Solution: weighted MSE that upweights foreground pixels
  w_i = Gaussian(dist_i)   so near-peak pixels contribute much more
  L_hm = mean( w_i * (logit_i - target_i)^2 )

This forces the network to concentrate on getting the peak right.

Additionally: use sigma=6 (wider Gaussian) so more pixels carry signal,
and increase beta to balance the weighted loss scale.
"""

import argparse
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from torch.utils.data import DataLoader, Subset

from model   import SingleViewModel, reconstruct_3d_from_25d
from dataset import FreiHANDDataset, IMG_SIZE


# ============================================================================
# Weighted heatmap loss
# ============================================================================

def heatmap_weighted_mse(hm_logits: torch.Tensor,
                          kpts_px:   torch.Tensor,
                          sigma:     float = 6.0) -> torch.Tensor:
    """
    Weighted MSE between heatmap logits and log-Gaussian targets.

    Weight = Gaussian(pixel, GT)  → near-peak pixels dominate loss.
    Target = clamped log-Gaussian in [-10, 0].

    Without weighting, 16000 background pixels drown out the 50 peak pixels
    and the network learns to output -10 everywhere (trivial solution).

    Args:
        hm_logits : (B, K, H, W)  raw pre-softmax logits
        kpts_px   : (B, K, 2)     GT keypoint pixel coords in [0, IMG_SIZE]
        sigma     : Gaussian sigma in pixels (use 6 for 128×128 input)

    Returns:
        scalar loss
    """
    B, K, H, W = hm_logits.shape
    device = hm_logits.device

    ys = torch.arange(H, device=device, dtype=torch.float32)
    xs = torch.arange(W, device=device, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')  # (H, W)

    px = kpts_px[:, :, 0].unsqueeze(-1).unsqueeze(-1)   # (B, K, 1, 1)
    py = kpts_px[:, :, 1].unsqueeze(-1).unsqueeze(-1)

    dist2  = (grid_x - px)**2 + (grid_y - py)**2         # (B, K, H, W)

    # Target: clamped log-Gaussian  (peak=0, background=-10)
    target = torch.clamp(-dist2 / (2.0 * sigma**2), min=-10.0)

    # Weight: Gaussian (peak=1, background≈0) — foreground focus
    weight = torch.exp(-dist2 / (2.0 * sigma**2))        # (B, K, H, W)

    # Weighted MSE
    loss = (weight * (hm_logits - target) ** 2).sum() / (weight.sum() + 1e-8)
    return loss


# ============================================================================
# Normalised loss
# ============================================================================

def loss_full(pred_2d:   torch.Tensor,
              pred_z:    torch.Tensor,
              hm_logits: torch.Tensor,
              gt_2d:     torch.Tensor,
              gt_z:      torch.Tensor,
              alpha: float = 1.0,
              beta:  float = 1.0,
              sigma: float = 6.0):
    """
    L = L_xy_norm  +  α·L_z_norm  +  β·L_hm_norm

    All three terms are normalised to the same scale so α and β
    are meaningful weights (default 1.0), not magic compensators.

    Normalisation strategy:
      L_xy  : divide by IMG_SIZE² so pixel MSE → [0,1] range
               MSE(px) / 128² maps a 128px error to loss=1.0
      L_z   : already in normalised depth units (~[-2, 2])
               divide by 4.0 (= max_range²/4) to put in ~[0,1]
      L_hm  : weighted MSE of logits already in ~[0,10] range
               divide by 10.0 to normalise to ~[0,1]

    With this normalisation α=β=1 is a sensible default.
    Increase α to prioritise depth, increase β to prioritise heatmap peaks.
    """
    L_xy  = F.mse_loss(pred_2d, gt_2d) / (IMG_SIZE ** 2)   # → [0, 1]
    L_z   = F.mse_loss(pred_z,  gt_z)  / 4.0               # → [0, 1]
    L_hm  = heatmap_weighted_mse(hm_logits, gt_2d, sigma=sigma) / 10.0  # → [0,1]

    total = L_xy + alpha * L_z + beta * L_hm
    return total, L_xy.item(), L_z.item(), L_hm.item()


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str,   default='/home/kia/Dataset')
    parser.add_argument('--n-samples', type=int,   default=8)
    parser.add_argument('--epochs',    type=int,   default=500)
    parser.add_argument('--lr',        type=float, default=1e-3)
    parser.add_argument('--alpha',     type=float, default=1.0,
                        help='Depth loss weight')
    parser.add_argument('--beta',      type=float, default=1.0,
                        help='Heatmap loss weight')
    parser.add_argument('--sigma',     type=float, default=6.0,
                        help='Gaussian sigma in pixels (wider = more signal)')
    parser.add_argument('--device',    type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    print('\n' + '='*65)
    print('Overfit Test — weighted heatmap MSE loss')
    print('='*65)
    print(f'  Samples : {args.n_samples}')
    print(f'  Epochs  : {args.epochs}')
    print(f'  LR      : {args.lr}')
    print(f'  α       : {args.alpha}   (depth weight)')
    print(f'  β       : {args.beta}   (heatmap weight)')
    print(f'  σ       : {args.sigma} px')
    print(f'  Device  : {args.device}\n')

    device = torch.device(args.device)

    # Fixed tiny batch — no augmentation, no shuffle
    ds     = FreiHANDDataset(args.data_root, split='train', augment=False)
    subset = Subset(ds, list(range(min(args.n_samples, len(ds)))))
    loader = DataLoader(subset, batch_size=args.n_samples,
                        shuffle=False, num_workers=0)

    fixed = next(iter(loader))
    imgs  = fixed['image'].to(device)
    gt_2d = fixed['pose_2d_gt'].to(device)
    gt_z  = fixed['depth_rel_gt'].to(device)
    K_mat = fixed['K_mat']                    # (N, 3, 3) — keep on CPU for numpy ops

    print(f'Batch          : {imgs.shape}')
    print(f'GT 2D range    : [{gt_2d.min():.1f}, {gt_2d.max():.1f}] px')
    print(f'GT depth range : [{gt_z.min():.4f}, {gt_z.max():.4f}]')
    print(f'GT joint spread: {gt_2d.std(dim=1).mean().item():.2f} px\n')

    model  = SingleViewModel(num_kpts=21).to(device)
    opt    = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    # ReduceLROnPlateau: patience=40 so it only fires on genuine long plateaus,
    # not normal epoch-to-epoch oscillations. factor=0.5 halves LR each time.
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=0.5, patience=40, min_lr=1e-6, verbose=False)

    params = sum(p.numel() for p in model.parameters())
    print(f'Model params: {params:,}  ({params*4/1e6:.1f} MB)\n')

    hist = {k: [] for k in ('loss', 'L_xy', 'L_z', 'L_hm', 'err2d', 'lr')}

    print(f"{'Ep':>5} {'Loss':>10} {'L_xy':>10} {'L_z':>10} "
          f"{'L_hm':>10} {'Err2D(px)':>11} {'LR':>10}")
    print('-' * 75)

    for epoch in range(1, args.epochs + 1):
        model.train()
        pred_2d, pred_z, hm_logits, _ = model(imgs)
        loss, lxy, lz, lhm = loss_full(
            pred_2d, pred_z, hm_logits, gt_2d, gt_z,
            alpha=args.alpha, beta=args.beta, sigma=args.sigma)

        opt.zero_grad()
        loss.backward()

        if epoch == 1:
            gnorm = sum(p.grad.norm().item()**2
                        for p in model.parameters()
                        if p.grad is not None) ** 0.5
            print(f'  [Ep1 grad norm: {gnorm:.2f}]'
                  f'  [L_hm scale: {lhm:.4f}  — should be < 10]\n')

        # Reduced clip norm: 0.5 instead of 1.0
        # Spikes happen when a single large gradient step overshoots the minimum.
        # Tighter clipping keeps steps conservative.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        opt.step()

        # Step scheduler on the current loss — reduces LR at plateaus
        scheduler.step(loss.item())
        current_lr = opt.param_groups[0]['lr']

        with torch.no_grad():
            err2d = (pred_2d - gt_2d).norm(dim=-1).mean().item()

        for k, v in zip(('loss','L_xy','L_z','L_hm','err2d','lr'),
                        (loss.item(), lxy, lz, lhm, err2d, current_lr)):
            hist[k].append(v)

        if epoch % 10 == 0 or epoch == 1:
            print(f'{epoch:>5} {loss.item():>10.4f} {lxy:>10.6f} '
                  f'{lz:>10.6f} {lhm:>10.6f} {err2d:>11.4f} {current_lr:>10.2e}')

    # ── Final predictions on the SAME training batch ─────────────────────
    model.eval()
    with torch.no_grad():
        pred_2d, pred_z, _, _ = model(imgs)
    pred_px = pred_2d.cpu().numpy()    # (N, 21, 2) — already pixels
    gt_px   = gt_2d.cpu().numpy()     # (N, 21, 2)

    # ── Per-joint error ───────────────────────────────────────────────────
    print('\nPer-joint 2D error (px) on training batch:')
    print(f"{'Jnt':>4} {'Err(px)':>10}   {'Jnt':>4} {'Err(px)':>10}")
    print('-' * 35)
    for k in range(0, 21, 2):
        e1 = np.linalg.norm(pred_px[:, k]   - gt_px[:, k],   axis=-1).mean()
        e2_str = ''
        if k+1 < 21:
            e2 = np.linalg.norm(pred_px[:, k+1] - gt_px[:, k+1], axis=-1).mean()
            e2_str = f'   {k+1:>4} {e2:>10.3f}'
        print(f'{k:>4} {e1:>10.3f}{e2_str}')

    # ── Save model checkpoint ─────────────────────────────────────────────
    ckpt_path = 'overfit_checkpoint.pt'
    torch.save({
        'epoch':            args.epochs,
        'model_state':      model.state_dict(),
        'optimizer_state':  opt.state_dict(),
        'scheduler_state':  scheduler.state_dict(),
        'history':          hist,
        'args':             vars(args),
    }, ckpt_path)
    print(f'\n✓ Saved model → {ckpt_path}')

    # ── Loss curves ───────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    ep = range(1, args.epochs + 1)

    axes[0,0].semilogy(ep, hist['loss'],  lw=2,           label='Total')
    axes[0,0].semilogy(ep, hist['L_xy'],  linestyle='--', label='L_xy (norm)')
    axes[0,0].semilogy(ep, hist['L_z'],   linestyle=':',  label='L_z  (norm)')
    axes[0,0].semilogy(ep, hist['L_hm'],  linestyle='-.', label='L_hm (norm)')
    axes[0,0].set_title('All losses (log scale)'); axes[0,0].legend()
    axes[0,0].grid(alpha=0.3); axes[0,0].set_xlabel('Epoch')

    axes[0,1].plot(ep, hist['err2d'], color='green', lw=2)
    axes[0,1].axhline(2.0, color='red', linestyle='--', label='2 px target')
    axes[0,1].set_title('2D error (pixels)'); axes[0,1].legend()
    axes[0,1].grid(alpha=0.3); axes[0,1].set_xlabel('Epoch')

    axes[0,2].semilogy(ep, hist['lr'], color='brown', lw=2)
    axes[0,2].set_title('Learning rate (ReduceLROnPlateau)')
    axes[0,2].grid(alpha=0.3); axes[0,2].set_xlabel('Epoch')

    axes[1,0].semilogy(ep, hist['L_xy'], color='orange')
    axes[1,0].set_title('L_xy normalised (log)'); axes[1,0].grid(alpha=0.3)
    axes[1,0].set_xlabel('Epoch')

    axes[1,1].semilogy(ep, hist['L_hm'], color='purple')
    axes[1,1].set_title('L_hm normalised (log)'); axes[1,1].grid(alpha=0.3)
    axes[1,1].set_xlabel('Epoch')

    axes[1,2].semilogy(ep, hist['L_z'], color='steelblue')
    axes[1,2].set_title('L_z normalised (log)'); axes[1,2].grid(alpha=0.3)
    axes[1,2].set_xlabel('Epoch')

    plt.suptitle(f'Overfit test — {args.n_samples} samples / {args.epochs} epochs',
                 fontsize=13)
    plt.tight_layout()
    plt.savefig('overfit_curves.png', dpi=130)
    plt.close()
    print('✓ Saved overfit_curves.png')

    # ── Visualization on the SAME training batch ──────────────────────────
    # NOTE: intentionally using the training images, not a val split.
    # The model has only seen these N samples — generalization is not
    # expected here. We just want to confirm the network CAN memorize them.
    print('\nGenerating visualizations on training batch...')

    CONNECTIONS = [
        [0,1],[1,2],[2,3],[3,4],
        [0,5],[5,6],[6,7],[7,8],
        [0,9],[9,10],[10,11],[11,12],
        [0,13],[13,14],[14,15],[15,16],
        [0,17],[17,18],[18,19],[19,20],
    ]
    FINGER_COLORS = [
        (255, 255, 255),   # wrist connections — white
        (0,   255,   0),   # index  — green
        (0,   200, 255),   # middle — cyan
        (255, 165,   0),   # ring   — orange
        (255,   0, 128),   # pinky  — pink
        (160,  32, 240),   # thumb  — purple
    ]
    FINGER_GROUPS = [
        [0,1],[0,5],[0,9],[0,13],[0,17],   # wrist spokes
        [1,2],[2,3],[3,4],                  # index
        [5,6],[6,7],[7,8],                  # middle
        [9,10],[10,11],[11,12],             # ring
        [13,14],[14,15],[15,16],            # pinky
        [17,18],[18,19],[19,20],            # thumb
    ]
    FINGER_COLOR_MAP = {
        tuple([0,1]):0, tuple([0,5]):0, tuple([0,9]):0,
        tuple([0,13]):0, tuple([0,17]):0,
        tuple([1,2]):1,tuple([2,3]):1,tuple([3,4]):1,
        tuple([5,6]):2,tuple([6,7]):2,tuple([7,8]):2,
        tuple([9,10]):3,tuple([10,11]):3,tuple([11,12]):3,
        tuple([13,14]):4,tuple([14,15]):4,tuple([15,16]):4,
        tuple([17,18]):5,tuple([18,19]):5,tuple([19,20]):5,
    }

    def draw_hand(img, kpts, alpha=1.0):
        """Draw skeleton with per-finger colors onto a copy of img."""
        out = img.copy().astype(np.float32)
        overlay = img.copy().astype(np.float32)
        for s, e in CONNECTIONS:
            cidx = FINGER_COLOR_MAP.get(tuple(sorted([s,e])), 0)
            color = FINGER_COLORS[cidx]
            p1 = tuple(np.clip(kpts[s].astype(int), [0,0], [127,127]))
            p2 = tuple(np.clip(kpts[e].astype(int), [0,0], [127,127]))
            cv2.line(overlay, p1, p2, color, 2)
        for k_idx, pt in enumerate(kpts):
            c = tuple(np.clip(pt.astype(int), [0,0], [127,127]))
            cv2.circle(overlay, c, 4, (255,255,255), -1)
            cv2.circle(overlay, c, 3, (0,0,0), 1)
        return cv2.addWeighted(overlay, alpha, out, 1-alpha, 0).astype(np.uint8)

    N = imgs.shape[0]
    cols = min(N, 4)
    rows = int(np.ceil(N / cols))
    fig, axes = plt.subplots(rows, cols * 2,
                             figsize=(cols * 5, rows * 3.5))
    # Flatten axes for easy indexing
    if rows == 1 and cols * 2 == 2:
        axes = np.array([[axes[0], axes[1]]])
    elif rows == 1:
        axes = axes[np.newaxis, :]
    axes = axes.reshape(rows, cols * 2)

    imgs_np = (imgs.cpu().permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)

    for i in range(N):
        row = i // cols
        col = (i % cols) * 2

        img_bgr = cv2.cvtColor(imgs_np[i], cv2.COLOR_RGB2BGR)

        vis_pred = draw_hand(imgs_np[i], pred_px[i])
        vis_gt   = draw_hand(imgs_np[i], gt_px[i])

        axes[row, col].imshow(vis_pred)
        axes[row, col].set_title(f'Sample {i} — Pred', fontsize=9)
        axes[row, col].axis('off')
        err = np.linalg.norm(pred_px[i] - gt_px[i], axis=-1).mean()
        axes[row, col].set_xlabel(f'err={err:.1f}px', fontsize=8)

        axes[row, col+1].imshow(vis_gt)
        axes[row, col+1].set_title(f'Sample {i} — GT', fontsize=9)
        axes[row, col+1].axis('off')

    # Hide any unused axes
    for i in range(N, rows * cols):
        row = i // cols
        col = (i % cols) * 2
        axes[row, col].axis('off')
        axes[row, col+1].axis('off')

    plt.suptitle(
        f'Overfit viz — TRAINING BATCH ONLY ({N} samples)\n'
        f'Left=Predicted (white joints), Right=GT  |  '
        f'Mean 2D err: {hist["err2d"][-1]:.2f} px',
        fontsize=11)
    plt.tight_layout()
    plt.savefig('overfit_visualization.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('✓ Saved overfit_visualization.png')

    # ── 3D Visualization on the SAME training batch ───────────────────────
    print('Generating 3D visualizations...')

    # Reconstruct predicted 3D from 2.5D output
    with torch.no_grad():
        pred_2d_t, pred_z_t, _, _ = model(imgs)

    pred_3d = reconstruct_3d_from_25d(
        pred_2d_t, pred_z_t,
        K_mat.to(device),
        img_size=IMG_SIZE
    ).cpu().numpy()                          # (N, 21, 3)  scale-normalised

    # Reconstruct GT 3D by back-projecting GT 2D + depth
    gt_2d_np = gt_2d.cpu().numpy()          # (N, 21, 2)
    gt_z_np  = gt_z.cpu().numpy()           # (N, 21)
    K_np     = K_mat.numpy()                # (N, 3, 3)

    # Absolute depth = root_depth + relative_depth
    # We don't know absolute root depth from 2.5D, so use scale-normalised
    # depth directly: z_abs_k = z_rel_k  (root is at 0)
    gt_Z = gt_z_np                          # root-relative, root=0
    fx   = K_np[:, 0, 0:1];  cx = K_np[:, 0, 2:3]
    fy   = K_np[:, 1, 1:2];  cy = K_np[:, 1, 2:3]
    gt_X = (gt_2d_np[:, :, 0] - cx) * gt_Z / (fx + 1e-8)
    gt_Y = (gt_2d_np[:, :, 1] - cy) * gt_Z / (fy + 1e-8)
    gt_3d = np.stack([gt_X, gt_Y, gt_Z], axis=2)  # (N, 21, 3)

    # Root-centre both
    pred_3d_rc = pred_3d - pred_3d[:, 0:1, :]
    gt_3d_rc   = gt_3d   - gt_3d[:, 0:1, :]

    # Per-finger colors for 3D plots (matplotlib RGB 0-1)
    COLORS_3D = {
        'wrist':  (0.8, 0.8, 0.8),
        'index':  (0.0, 0.9, 0.0),
        'middle': (0.0, 0.7, 1.0),
        'ring':   (1.0, 0.6, 0.0),
        'pinky':  (1.0, 0.0, 0.5),
        'thumb':  (0.6, 0.1, 0.9),
    }
    BONE_COLOR_LIST = [
        COLORS_3D['wrist'],   # [0,1]
        COLORS_3D['index'],   # [1,2],[2,3],[3,4]
        COLORS_3D['index'],
        COLORS_3D['index'],
        COLORS_3D['wrist'],   # [0,5]
        COLORS_3D['middle'],  # [5,6],[6,7],[7,8]
        COLORS_3D['middle'],
        COLORS_3D['middle'],
        COLORS_3D['wrist'],   # [0,9]
        COLORS_3D['ring'],    # [9,10],[10,11],[11,12]
        COLORS_3D['ring'],
        COLORS_3D['ring'],
        COLORS_3D['wrist'],   # [0,13]
        COLORS_3D['pinky'],   # [13,14],[14,15],[15,16]
        COLORS_3D['pinky'],
        COLORS_3D['pinky'],
        COLORS_3D['wrist'],   # [0,17]
        COLORS_3D['thumb'],   # [17,18],[18,19],[19,20]
        COLORS_3D['thumb'],
        COLORS_3D['thumb'],
    ]

    def draw_3d_hand(ax, joints, bone_colors, title, marker='o',
                     alpha_bones=0.9, alpha_joints=1.0):
        """
        Draw a 3D hand skeleton on a matplotlib 3D axis.
        joints: (21, 3)  XYZ coordinates
        """
        for bi, (s, e) in enumerate(CONNECTIONS):
            col = bone_colors[bi]
            ax.plot([joints[s,0], joints[e,0]],
                    [joints[s,1], joints[e,1]],
                    [joints[s,2], joints[e,2]],
                    color=col, lw=2, alpha=alpha_bones)
        # Joint dots — colored by finger
        for ji, pt in enumerate(joints):
            # Determine finger from joint index
            if   ji == 0:             c = COLORS_3D['wrist']
            elif 1  <= ji <= 4:       c = COLORS_3D['index']
            elif 5  <= ji <= 8:       c = COLORS_3D['middle']
            elif 9  <= ji <= 12:      c = COLORS_3D['ring']
            elif 13 <= ji <= 16:      c = COLORS_3D['pinky']
            else:                     c = COLORS_3D['thumb']
            ax.scatter(*pt, color=c, s=30, zorder=5, alpha=alpha_joints,
                       edgecolors='black', linewidths=0.4)
        ax.set_title(title, fontsize=9, pad=3)
        ax.set_xlabel('X', fontsize=7); ax.set_ylabel('Y', fontsize=7)
        ax.set_zlabel('Z', fontsize=7)
        ax.tick_params(labelsize=6)

    def set_equal_axes(ax, joints_list):
        """Force equal aspect ratio on a 3D axis given a list of joint arrays."""
        all_pts = np.concatenate(joints_list, axis=0)
        ranges  = all_pts.max(0) - all_pts.min(0)
        max_r   = ranges.max() / 2.0 + 1e-6
        mid     = (all_pts.max(0) + all_pts.min(0)) / 2.0
        ax.set_xlim(mid[0]-max_r, mid[0]+max_r)
        ax.set_ylim(mid[1]-max_r, mid[1]+max_r)
        ax.set_zlim(mid[2]-max_r, mid[2]+max_r)

    # Each sample gets one row: [front view pred | front view GT | side view overlay]
    fig3d = plt.figure(figsize=(N * 3.5, 3 * 3.2))
    fig3d.suptitle(
        f'3D Pose — TRAINING BATCH  |  '
        f'Mean MPJPE: {np.mean(np.linalg.norm(pred_3d_rc - gt_3d_rc, axis=-1)):.4f} (norm units)',
        fontsize=12)

    for i in range(N):
        p3 = pred_3d_rc[i]   # (21, 3)
        g3 = gt_3d_rc[i]     # (21, 3)

        # --- Col 1: Predicted (front view) ---
        ax1 = fig3d.add_subplot(3, N, i + 1, projection='3d')
        draw_3d_hand(ax1, p3, BONE_COLOR_LIST,
                     title=f'S{i} Pred\n(front)', alpha_bones=0.9)
        set_equal_axes(ax1, [p3, g3])
        ax1.view_init(elev=0, azim=-90)   # front view

        # --- Col 2: GT (front view) ---
        ax2 = fig3d.add_subplot(3, N, N + i + 1, projection='3d')
        draw_3d_hand(ax2, g3,
                     [COLORS_3D['wrist']]*20,   # GT in uniform grey
                     title=f'S{i} GT\n(front)', alpha_bones=0.7)
        set_equal_axes(ax2, [p3, g3])
        ax2.view_init(elev=0, azim=-90)

        # --- Col 3: Overlay (side view) ---
        ax3 = fig3d.add_subplot(3, N, 2*N + i + 1, projection='3d')
        draw_3d_hand(ax3, p3, BONE_COLOR_LIST,
                     title=f'S{i} Overlay\n(side)', alpha_bones=0.9)
        draw_3d_hand(ax3, g3,
                     [(0.5, 0.5, 0.5)]*20,
                     title=f'S{i} Overlay\n(side)',
                     alpha_bones=0.4, alpha_joints=0.4)
        set_equal_axes(ax3, [p3, g3])
        ax3.view_init(elev=20, azim=0)    # side view

        # Per-sample MPJPE annotation
        mpjpe_i = np.linalg.norm(p3 - g3, axis=-1).mean()
        ax1.set_xlabel(f'MPJPE={mpjpe_i:.4f}', fontsize=7)

    plt.tight_layout()
    plt.savefig('overfit_3d_visualization.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('✓ Saved overfit_3d_visualization.png')

    # ── Pass / fail ───────────────────────────────────────────────────────
    final_2d = hist['err2d'][-1]
    print('\n' + '='*65)
    print('RESULT')
    print('='*65)
    print(f'  Final 2D error : {final_2d:.3f} px  (target < 2 px)')
    if final_2d < 2.0:
        print('\n  ✅ PASS — network memorized the training batch.')
        print('     Proceed to full training with train.py')
    elif final_2d < 8.0:
        print('\n  ⚠  PARTIAL — try --epochs 1000 or --lr 3e-3')
    else:
        print('\n  ❌ FAIL — check loss curves in overfit_curves.png')
        print('     Which loss is not decreasing?')
        print('     L_xy stuck → soft-argmax / coordinate space bug')
        print('     L_hm stuck → heatmap not peaking, try --sigma 10')
        print('     L_z  stuck → depth head issue')
    print('='*65 + '\n')


if __name__ == '__main__':
    main()