"""
train.py — Full training script for Latent 2.5D Heatmap Regression
====================================================================
Trains SingleViewModel (Iqbal et al. ECCV 2018) on the FreiHAND dataset.

Usage:
    python3 train.py --data-root /home/kia/Dataset
    python3 train.py --data-root /home/kia/Dataset --resume train_checkpoint.pt

Key differences from overfit_test.py:
  - Full DataLoader with shuffling and augmentation
  - Per-epoch validation with MPJPE metric
  - Checkpoint save/resume
  - Cosine LR schedule with linear warmup
"""

import argparse
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from model   import SingleViewModel, reconstruct_3d_from_25d
from dataset import FreiHANDDataset, IMG_SIZE


# ============================================================================
# Loss functions  (identical to overfit_test.py)
# ============================================================================

def heatmap_weighted_mse(hm_logits, kpts_px, ys, xs, sigma=6.0):
    """
    ys: (H,) and xs: (W,) are precomputed once per run and reused every batch.
    Weight is separable: exp(-dx²/2σ²) * exp(-dy²/2σ²) — avoids meshgrid.
    """
    B, K, H, W = hm_logits.shape
    px = kpts_px[:, :, 0].view(B, K, 1, 1)   # (B, K, 1, 1)
    py = kpts_px[:, :, 1].view(B, K, 1, 1)

    dx = xs.view(1, 1, 1, W) - px             # (B, K, 1, W)
    dy = ys.view(1, 1, H, 1) - py             # (B, K, H, 1)
    dist2 = dx ** 2 + dy ** 2                 # (B, K, H, W)

    inv_2s2 = 1.0 / (2.0 * sigma ** 2)
    target  = (-dist2 * inv_2s2).clamp(min=-10.0)
    weight  = torch.exp(-dist2 * inv_2s2)

    return (weight * (hm_logits - target) ** 2).sum() / (weight.sum() + 1e-8)


def loss_full(pred_2d, pred_z, hm_logits, gt_2d, gt_z,
              ys, xs, alpha=1.0, beta=1.0, sigma=6.0):
    L_xy = F.mse_loss(pred_2d, gt_2d) / (IMG_SIZE ** 2)
    L_z  = F.mse_loss(pred_z,  gt_z)  / 4.0
    L_hm = heatmap_weighted_mse(hm_logits, gt_2d, ys, xs, sigma=sigma) / 10.0
    total = L_xy + alpha * L_z + beta * L_hm
    return total, L_xy.item(), L_z.item(), L_hm.item()


# ============================================================================
# Validation — MPJPE in normalised units
# ============================================================================

@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    mpjpe_list = []
    for batch in loader:
        imgs  = batch['image'].to(device, non_blocking=True)
        gt_2d = batch['pose_2d_gt'].to(device, non_blocking=True)
        gt_z  = batch['depth_rel_gt'].to(device, non_blocking=True)
        K_mat = batch['K_mat'].to(device, non_blocking=True)

        with autocast():
            pred_2d, pred_z, _, _ = model(imgs)

        pred_3d = reconstruct_3d_from_25d(pred_2d.float(), pred_z.float(),
                                          K_mat, img_size=IMG_SIZE)
        gt_3d   = reconstruct_3d_from_25d(gt_2d,  gt_z,  K_mat, img_size=IMG_SIZE)

        pred_3d = pred_3d - pred_3d[:, 0:1]
        gt_3d   = gt_3d   - gt_3d[:, 0:1]

        mpjpe = (pred_3d - gt_3d).norm(dim=-1).mean(dim=-1)
        mpjpe_list.append(mpjpe.cpu())

    return torch.cat(mpjpe_list).mean().item()


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root',   type=str,   default='/home/kghasemz/scratch/datasets/FreiHand')
    parser.add_argument('--epochs',      type=int,   default=100)
    parser.add_argument('--batch-size',  type=int,   default=32)
    parser.add_argument('--lr',          type=float, default=1e-3)
    parser.add_argument('--alpha',       type=float, default=1.0,  help='Depth loss weight')
    parser.add_argument('--beta',        type=float, default=1.0,  help='Heatmap loss weight')
    parser.add_argument('--sigma',       type=float, default=6.0,  help='Heatmap Gaussian sigma (px)')
    parser.add_argument('--num-workers', type=int,   default=4)
    parser.add_argument('--resume',      type=str,   default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--save-dir',    type=str,   default='.',  help='Directory for checkpoints and plots')
    parser.add_argument('--device',      type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    print('\n' + '='*65)
    print('Full Training — Latent 2.5D Heatmap Regression')
    print('='*65)
    print(f'  Epochs      : {args.epochs}')
    print(f'  Batch size  : {args.batch_size}')
    print(f'  LR          : {args.lr}')
    print(f'  α (depth)   : {args.alpha}')
    print(f'  β (heatmap) : {args.beta}')
    print(f'  σ           : {args.sigma} px')
    print(f'  Device      : {args.device}')
    print(f'  Save dir    : {args.save_dir}\n')

    device = torch.device(args.device)

    # Fixed input size → let cuDNN benchmark and cache the fastest kernels
    torch.backends.cudnn.benchmark = True

    # ── Datasets & loaders ────────────────────────────────────────────────
    train_ds = FreiHANDDataset(args.data_root, split='train', augment=True)
    val_ds   = FreiHANDDataset(args.data_root, split='val',   augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=args.num_workers,
                              pin_memory=True, drop_last=True,
                              persistent_workers=(args.num_workers > 0))
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers,
                              pin_memory=True,
                              persistent_workers=(args.num_workers > 0))

    print(f'Train samples : {len(train_ds)}  ({len(train_loader)} batches/epoch)')
    print(f'Val samples   : {len(val_ds)}   ({len(val_loader)} batches)\n')

    # ── Model, optimiser, scheduler ───────────────────────────────────────
    model = SingleViewModel(num_kpts=21).to(device)
    opt   = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    # Cosine annealing with linear warmup (5% of total steps as warmup)
    total_steps  = args.epochs * len(train_loader)
    warmup_steps = max(1, int(0.05 * total_steps))

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    start_epoch = 1
    best_mpjpe  = float('inf')
    hist = {'train_loss': [], 'val_mpjpe': [], 'lr': []}

    # ── Resume from checkpoint ────────────────────────────────────────────
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        opt.load_state_dict(ckpt['optimizer_state'])
        scheduler.load_state_dict(ckpt['scheduler_state'])
        start_epoch = ckpt['epoch'] + 1
        best_mpjpe  = ckpt.get('best_mpjpe', float('inf'))
        hist        = ckpt.get('history', hist)
        print(f'Resumed from {args.resume}  (epoch {ckpt["epoch"]}, best MPJPE {best_mpjpe:.4f})\n')

    params = sum(p.numel() for p in model.parameters())
    print(f'Model params: {params:,}  ({params*4/1e6:.1f} MB)\n')

    # Mixed-precision scaler — no-op on CPU
    scaler = GradScaler(enabled=(device.type == 'cuda'))

    # Precompute pixel grid once — reused by heatmap loss every batch
    ys = torch.arange(IMG_SIZE, device=device, dtype=torch.float32)
    xs = torch.arange(IMG_SIZE, device=device, dtype=torch.float32)

    global_step = (start_epoch - 1) * len(train_loader)

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()

        # ── Train ─────────────────────────────────────────────────────────
        model.train()
        train_losses = []

        pbar = tqdm(train_loader, desc=f'Ep {epoch}/{args.epochs} [train]',
                    ncols=110, leave=True)

        for batch in pbar:
            imgs  = batch['image'].to(device, non_blocking=True)
            gt_2d = batch['pose_2d_gt'].to(device, non_blocking=True)
            gt_z  = batch['depth_rel_gt'].to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)   # faster than zero_grad()

            with autocast(enabled=(device.type == 'cuda')):
                pred_2d, pred_z, hm_logits, _ = model(imgs)
                loss, lxy, lz, lhm = loss_full(
                    pred_2d, pred_z, hm_logits, gt_2d, gt_z,
                    ys=ys, xs=xs,
                    alpha=args.alpha, beta=args.beta, sigma=args.sigma)

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            scheduler.step()
            global_step += 1

            train_losses.append((loss.item(), lxy, lz, lhm))

            # Update progress bar with running averages every step
            avg = np.mean(train_losses, axis=0)
            pbar.set_postfix(
                step=global_step,
                loss=f'{avg[0]:.4f}',
                xy=f'{avg[1]:.4f}',
                z=f'{avg[2]:.4f}',
                hm=f'{avg[3]:.4f}',
                lr=f'{opt.param_groups[0]["lr"]:.1e}',
            )

        pbar.close()
        avg_loss, avg_lxy, avg_lz, avg_lhm = np.mean(train_losses, axis=0)
        current_lr = opt.param_groups[0]['lr']
        epoch_time = time.time() - epoch_start

        # ── Validate ──────────────────────────────────────────────────────
        print(f'  Ep {epoch} validating...', end='\r')
        val_mpjpe = validate(model, val_loader, device)

        hist['train_loss'].append(avg_loss)
        hist['val_mpjpe'].append(val_mpjpe)
        hist['lr'].append(current_lr)

        print(f'Ep {epoch:>3}/{args.epochs}  '
              f'loss={avg_loss:.4f}  xy={avg_lxy:.4f}  z={avg_lz:.4f}  '
              f'hm={avg_lhm:.4f}  val_MPJPE={val_mpjpe:.4f}  '
              f'lr={current_lr:.1e}  time={epoch_time/60:.1f}min'
              + ('  ★ best' if val_mpjpe <= min(hist['val_mpjpe']) else ''))

        # ── Save best checkpoint ──────────────────────────────────────────
        if val_mpjpe < best_mpjpe:
            best_mpjpe = val_mpjpe
            torch.save({
                'epoch':           epoch,
                'model_state':     model.state_dict(),
                'optimizer_state': opt.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'best_mpjpe':      best_mpjpe,
                'history':         hist,
                'args':            vars(args),
            }, os.path.join(args.save_dir, 'best_model.pt'))

        # ── Save latest checkpoint every 10 epochs (for resuming) ─────────
        if epoch % 10 == 0:
            torch.save({
                'epoch':           epoch,
                'model_state':     model.state_dict(),
                'optimizer_state': opt.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'best_mpjpe':      best_mpjpe,
                'history':         hist,
                'args':            vars(args),
            }, os.path.join(args.save_dir, 'train_checkpoint.pt'))

    print(f'\nBest val MPJPE: {best_mpjpe:.4f} (normalised units)')
    print(f'Checkpoints saved to: {args.save_dir}/')

    # ── Plot training curves ───────────────────────────────────────────────
    ep = range(1, len(hist['train_loss']) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].semilogy(ep, hist['train_loss'], lw=2)
    axes[0].set_title('Train loss (log)'); axes[0].set_xlabel('Epoch')
    axes[0].grid(alpha=0.3)

    axes[1].plot(ep, hist['val_mpjpe'], lw=2, color='green')
    axes[1].set_title('Val MPJPE (norm units)'); axes[1].set_xlabel('Epoch')
    axes[1].grid(alpha=0.3)

    axes[2].semilogy(ep, hist['lr'], lw=2, color='brown')
    axes[2].set_title('Learning rate'); axes[2].set_xlabel('Epoch')
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, 'train_curves.png'), dpi=130)
    plt.close()
    print('✓ Saved train_curves.png')


if __name__ == '__main__':
    main()
