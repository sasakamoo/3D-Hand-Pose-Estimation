"""
train_sdf_mano.py — Training script for SDFHandPoseNetMANO
===========================================================
Trains the HOISDF-inspired model with MANO parameter output on FreiHAND.

Usage:
    python3 train_sdf_mano.py --data-root /path/to/FreiHand
    python3 train_sdf_mano.py --data-root /path/to/FreiHand --resume sdf_mano_checkpoint.pt

Differences from train_sdf.py:
  - Uses FreiHANDDatasetMANO  (loads training_mano.json)
  - Uses SDFHandPoseNetMANO   (adds MANOParamHead alongside 2.5D head)
  - Model returns 6 values:   (pose_2d, depth_rel, sdf_vals, pts, mano_pose, mano_shape)
  - Two additional loss terms:
      L_mano_pose  : MSE between predicted and GT MANO pose  (θ, 48-d)
      L_mano_shape : MSE between predicted and GT MANO shape (β, 10-d)
  - New args: --mano-pose-weight, --mano-shape-weight

Loss breakdown:
    total = L_xy
          + alpha  · L_z
          + gamma  · L_sdf
          + delta  · L_mano_pose
          + epsilon· L_mano_shape

    L_xy         : MSE on 2D pixel positions          / IMG_SIZE²
    L_z          : MSE on root-relative depth         / 4
    L_sdf        : L1 |sdf_pred| vs nearest-joint dist (pseudo GT)
    L_mano_pose  : MSE on 48-d axis-angle pose params / MANO_POSE_DIM
    L_mano_shape : MSE on 10-d shape params           / MANO_SHAPE_DIM
"""

import argparse
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast, GradScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from model_sdf_mano import SDFHandPoseNetMANO, MANO_POSE_DIM, MANO_SHAPE_DIM
from dataset_mano   import FreiHANDDatasetMANO, IMG_SIZE
from model          import reconstruct_3d_from_25d


# ============================================================================
# Loss functions
# ============================================================================

def sdf_pseudo_loss(sdf_vals: torch.Tensor,
                    pts: torch.Tensor,
                    gt_2d_px: torch.Tensor) -> torch.Tensor:
    """
    L1 loss: |sdf_pred| vs nearest-joint distance in normalised UV space.
    Identical to train_sdf.py — see that file for full explanation.
    """
    gt_uv  = gt_2d_px / (IMG_SIZE / 2.0) - 1.0           # (B, K, 2)
    pts_uv = pts[:, :, :2]                                 # (B, N, 2)
    diff   = pts_uv.unsqueeze(2) - gt_uv.unsqueeze(1)     # (B, N, K, 2)
    dist   = diff.norm(dim=-1)                             # (B, N, K)
    sdf_gt = dist.min(dim=-1).values                       # (B, N)
    return F.l1_loss(sdf_vals.abs(), sdf_gt)


def loss_full(pred_2d, pred_z, gt_2d, gt_z,
              sdf_vals, pts,
              pred_mano_pose, pred_mano_shape,
              gt_mano_pose,   gt_mano_shape,
              alpha=1.0, gamma=0.1, delta=1.0, eps=0.1):
    """
    Combined loss for SDFHandPoseNetMANO.

    Args:
        pred_2d         : (B, K, 2)
        pred_z          : (B, K)
        gt_2d           : (B, K, 2)
        gt_z            : (B, K)
        sdf_vals        : (B, N)
        pts             : (B, N, 3)
        pred_mano_pose  : (B, 48)
        pred_mano_shape : (B, 10)
        gt_mano_pose    : (B, 48)
        gt_mano_shape   : (B, 10)
        alpha   : depth loss weight
        gamma   : SDF pseudo-supervision weight   (0 = disabled)
        delta   : MANO pose loss weight
        eps     : MANO shape loss weight

    Returns:
        total, lxy, lz, lsdf, lmpose, lmshape   (all floats except total)
    """
    L_xy = F.mse_loss(pred_2d, gt_2d) / (IMG_SIZE ** 2)
    L_z  = F.mse_loss(pred_z,  gt_z)  / 4.0

    L_sdf = (sdf_pseudo_loss(sdf_vals, pts, gt_2d)
             if gamma > 0.0 else torch.zeros(1, device=pred_2d.device)[0])

    # MANO pose: normalise by dimensionality so loss scale is independent
    # of the 48 vs 10 parameter counts.
    L_mano_pose  = F.mse_loss(pred_mano_pose,  gt_mano_pose)  / MANO_POSE_DIM
    L_mano_shape = F.mse_loss(pred_mano_shape, gt_mano_shape) / MANO_SHAPE_DIM

    total = (L_xy
             + alpha * L_z
             + gamma * L_sdf
             + delta * L_mano_pose
             + eps   * L_mano_shape)

    return (total,
            L_xy.item(), L_z.item(), L_sdf.item(),
            L_mano_pose.item(), L_mano_shape.item())


# ============================================================================
# Validation — MPJPE in normalised units  (same metric as train_sdf.py)
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

        with autocast('cuda'):
            pred_2d, pred_z, _, _, _, _ = model(imgs)

        pred_3d = reconstruct_3d_from_25d(pred_2d.float(), pred_z.float(),
                                          K_mat, img_size=IMG_SIZE)
        gt_3d   = reconstruct_3d_from_25d(gt_2d, gt_z, K_mat, img_size=IMG_SIZE)

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
    parser.add_argument('--data-root',         type=str,   default='/home/kghasemz/scratch/datasets/FreiHand')
    parser.add_argument('--epochs',            type=int,   default=100)
    parser.add_argument('--batch-size',        type=int,   default=32)
    parser.add_argument('--lr',                type=float, default=1e-4)
    parser.add_argument('--alpha',             type=float, default=1.0,  help='Depth loss weight')
    parser.add_argument('--gamma',             type=float, default=0.1,  help='SDF pseudo-supervision weight')
    parser.add_argument('--mano-pose-weight',  type=float, default=1.0,  help='MANO pose loss weight (delta)')
    parser.add_argument('--mano-shape-weight', type=float, default=0.1,  help='MANO shape loss weight (epsilon)')
    parser.add_argument('--num-workers',       type=int,   default=4)
    parser.add_argument('--resume',            type=str,   default=None)
    parser.add_argument('--save-dir',          type=str,   default='.')
    parser.add_argument('--no-pretrain',       action='store_true')
    parser.add_argument('--device',            type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    print('\n' + '='*65)
    print('Full Training — HOISDF + MANO 2.5D Hand Pose')
    print('='*65)
    print(f'  Epochs            : {args.epochs}')
    print(f'  Batch size        : {args.batch_size}')
    print(f'  LR                : {args.lr}')
    print(f'  α (depth)         : {args.alpha}')
    print(f'  γ (SDF)           : {args.gamma}')
    print(f'  δ (MANO pose)     : {args.mano_pose_weight}')
    print(f'  ε (MANO shape)    : {args.mano_shape_weight}')
    print(f'  Pretrained        : {not args.no_pretrain}')
    print(f'  Device            : {args.device}')
    print(f'  Save dir          : {args.save_dir}\n')

    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = True

    # ── Datasets & loaders ────────────────────────────────────────────────
    train_ds = FreiHANDDatasetMANO(args.data_root, split='train', augment=True)
    val_ds   = FreiHANDDatasetMANO(args.data_root, split='val',   augment=False)

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
    model = SDFHandPoseNetMANO(
        num_kpts=21,
        pretrained_backbone=(not args.no_pretrain),
    ).to(device)

    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    warmup_steps = max(1, len(train_loader))   # 1 epoch warmup

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        epoch_frac = (step - warmup_steps) / len(train_loader)
        return 0.7 ** (epoch_frac / 5.0)

    scheduler   = optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    start_epoch = 1
    best_mpjpe  = float('inf')
    hist = {'train_loss': [], 'val_mpjpe': [], 'lr': []}

    # ── Resume from checkpoint ────────────────────────────────────────────
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state'])
        opt.load_state_dict(ckpt['optimizer_state'])
        scheduler.load_state_dict(ckpt['scheduler_state'])
        start_epoch = ckpt['epoch'] + 1
        best_mpjpe  = ckpt.get('best_mpjpe', float('inf'))
        hist        = ckpt.get('history', hist)
        print(f'Resumed from {args.resume}  (epoch {ckpt["epoch"]}, best MPJPE {best_mpjpe:.4f})\n')

    params = sum(p.numel() for p in model.parameters())
    print(f'Model params: {params:,}  ({params * 4 / 1e6:.1f} MB)\n')

    scaler      = GradScaler('cuda', enabled=(device.type == 'cuda'))
    global_step = (start_epoch - 1) * len(train_loader)

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()

        # ── Train ─────────────────────────────────────────────────────────
        model.train()
        train_losses = []

        pbar = tqdm(train_loader, desc=f'Ep {epoch}/{args.epochs} [train]',
                    ncols=120, leave=True)

        for batch in pbar:
            imgs       = batch['image'].to(device, non_blocking=True)
            gt_2d      = batch['pose_2d_gt'].to(device, non_blocking=True)
            gt_z       = batch['depth_rel_gt'].to(device, non_blocking=True)
            gt_mpose   = batch['mano_pose'].to(device, non_blocking=True)
            gt_mshape  = batch['mano_shape'].to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            with autocast('cuda', enabled=(device.type == 'cuda')):
                pred_2d, pred_z, sdf_vals, pts, pred_mpose, pred_mshape = model(imgs)

                (loss, lxy, lz, lsdf,
                 lmpose, lmshape) = loss_full(
                    pred_2d, pred_z, gt_2d, gt_z,
                    sdf_vals, pts,
                    pred_mpose, pred_mshape,
                    gt_mpose,   gt_mshape,
                    alpha=args.alpha,
                    gamma=args.gamma,
                    delta=args.mano_pose_weight,
                    eps=args.mano_shape_weight,
                )

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            scheduler.step()
            global_step += 1

            train_losses.append((loss.item(), lxy, lz, lsdf, lmpose, lmshape))

            avg = np.mean(train_losses, axis=0)
            pbar.set_postfix(
                loss=f'{avg[0]:.4f}',
                xy=f'{avg[1]:.4f}',
                z=f'{avg[2]:.4f}',
                sdf=f'{avg[3]:.4f}',
                mpose=f'{avg[4]:.4f}',
                mshape=f'{avg[5]:.4f}',
                lr=f'{opt.param_groups[0]["lr"]:.1e}',
            )

        pbar.close()
        avg = np.mean(train_losses, axis=0)
        avg_loss, avg_lxy, avg_lz, avg_lsdf, avg_lmpose, avg_lmshape = avg
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
              f'sdf={avg_lsdf:.4f}  mpose={avg_lmpose:.4f}  mshape={avg_lmshape:.4f}  '
              f'val_MPJPE={val_mpjpe:.4f}  lr={current_lr:.1e}  '
              f'time={epoch_time/60:.1f}min'
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
            }, os.path.join(args.save_dir, 'sdf_mano_best_model.pt'))

        if epoch % 10 == 0:
            torch.save({
                'epoch':           epoch,
                'model_state':     model.state_dict(),
                'optimizer_state': opt.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'best_mpjpe':      best_mpjpe,
                'history':         hist,
                'args':            vars(args),
            }, os.path.join(args.save_dir, 'sdf_mano_checkpoint.pt'))

    print(f'\nBest val MPJPE: {best_mpjpe:.4f} (normalised units)')
    print(f'Checkpoints saved to: {args.save_dir}/')

    # ── Plot training curves ──────────────────────────────────────────────
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
    plt.savefig(os.path.join(args.save_dir, 'sdf_mano_train_curves.png'), dpi=130)
    plt.close()
    print('✓ Saved sdf_mano_train_curves.png')


if __name__ == '__main__':
    main()
