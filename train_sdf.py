"""
train_sdf.py — Training script for SDFHandPoseNet
==================================================
Trains the HOISDF-inspired model (model_sdf.py) on the FreiHAND dataset.

Usage:
    python3 train_sdf.py --data-root /path/to/FreiHAND
    python3 train_sdf.py --data-root /path/to/FreiHAND --resume sdf_checkpoint.pt

SDF supervision:
    GT joint positions are derived on-the-fly from each batch's augmentation-
    corrected gt_2d and gt_z tensors, converted to normalised model space
    [-1,1]^3.  GT SDF values are then the minimum 3D distance from each query
    point to the nearest bone segment of these joints.  Computing on-the-fly
    ensures the SDF targets are always consistent with the augmented image the
    model receives.  Use --gamma to control the SDF loss weight (default 0.05).
"""

import argparse
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader


from model_sdf import SDFHandPoseNet
from dataset   import FreiHANDDataset, IMG_SIZE
from model     import reconstruct_3d_from_25d


# ============================================================================
# Loss functions
# ============================================================================

BONE_SEGMENTS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
]


def _point_to_segment_dist(p: torch.Tensor,
                            a: torch.Tensor,
                            b: torch.Tensor) -> torch.Tensor:
    """
    Minimum distance from points p to line segments a→b.

    Args:
        p : (B, N, 3)
        a : (B, 1, 3)  segment start
        b : (B, 1, 3)  segment end
    Returns:
        (B, N)  distance from each point to the nearest point on the segment
    """
    ab  = b - a                                      # (B, 1, 3)
    ap  = p - a                                      # (B, N, 3)
    denom = (ab * ab).sum(-1, keepdim=True).clamp(min=1e-8)  # (B, 1, 1)
    t   = ((ap * ab).sum(-1, keepdim=True) / denom).clamp(0, 1)  # (B, N, 1)
    closest = a + t * ab                             # (B, N, 3)
    return (p - closest).norm(dim=-1)                # (B, N)


def sdf_loss(sdf_vals: torch.Tensor,
             pts: torch.Tensor,
             gt_joints_norm: torch.Tensor) -> torch.Tensor:
    """
    SDF supervision via minimum 3D distance to the nearest hand bone segment.

    GT SDF values are the distance from each query point to the nearest bone
    segment, computed from precomputed normalised 3D joint positions
    (sdf_joints_norm.npy).  The model is trained to predict |sdf| ≈ this
    distance, which trains the density gate σ = sigmoid(-|sdf|) to be high
    near the hand surface and low away from it.

    Args:
        sdf_vals       : (B, N)    predicted SDF values
        pts            : (B, N, 3) query points in normalised [-1,1]^3 space
        gt_joints_norm : (B, K, 3) precomputed normalised 3D joint positions

    Returns:
        scalar L1 loss
    """
    seg_dists = []
    for s, e in BONE_SEGMENTS:
        a = gt_joints_norm[:, s:s+1, :]              # (B, 1, 3)
        b = gt_joints_norm[:, e:e+1, :]              # (B, 1, 3)
        seg_dists.append(_point_to_segment_dist(pts, a, b))  # (B, N)
    sdf_gt = torch.stack(seg_dists, dim=-1).min(dim=-1).values  # (B, N)
    return F.l1_loss(sdf_vals.abs(), sdf_gt)


def loss_full(pred_2d, pred_z, gt_2d, gt_z,
              sdf_vals=None, pts=None, gt_joints_norm=None,
              alpha=0.1, gamma=0.05):
    """
    Combined training loss for SDFHandPoseNet.

    Terms:
        L_xy  : L1 on 2D pixel positions normalised by IMG_SIZE
                → mean absolute error as fraction of image width  (~0.02–0.15)
        L_z   : L1 on root-relative depth in normalised units
                → mean absolute depth error in scale-norm units  (~0.05–0.50)
        L_sdf : L1 on predicted vs GT bone-segment SDF distances
                → requires gt_joints_norm from precompute_sdf.py

    Args:
        pred_2d        : (B, K, 2)  predicted 2D positions in pixels
        pred_z         : (B, K)     predicted depth_rel
        gt_2d          : (B, K, 2)  GT 2D positions in pixels
        gt_z           : (B, K)     GT depth_rel
        sdf_vals       : (B, N) or None
        pts            : (B, N, 3) or None
        gt_joints_norm : (B, K, 3) precomputed normalised 3D joints (required for SDF loss)
        alpha          : depth loss weight (~0.1 balances L_z with L_xy)
        gamma          : SDF loss weight (0 = disabled)

    Returns:
        total, L_xy.item(), L_z.item(), L_sdf.item()
    """
    L_xy = F.l1_loss(pred_2d, gt_2d) / IMG_SIZE   # mean abs pixel err / 128  → ~0.02–0.15
    L_z  = F.l1_loss(pred_z,  gt_z)               # mean abs depth err in norm units → ~0.05–0.5

    if gamma > 0.0 and sdf_vals is not None and pts is not None and gt_joints_norm is not None:
        L_sdf = sdf_loss(sdf_vals, pts, gt_joints_norm)
    else:
        L_sdf = torch.zeros(1, device=pred_2d.device)[0]

    total = L_xy + alpha * L_z + gamma * L_sdf
    return total, L_xy.item(), L_z.item(), L_sdf.item()


# ============================================================================
# Validation — MPJPE in normalised units  (identical to train.py)
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
            pred_2d, pred_z, _, _ = model(imgs)

        pred_3d = reconstruct_3d_from_25d(pred_2d.float(), pred_z.float(),
                                          K_mat, img_size=IMG_SIZE)
        gt_3d   = reconstruct_3d_from_25d(gt_2d, gt_z, K_mat,
                                          img_size=IMG_SIZE)

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
    parser.add_argument('--data-root',    type=str,   default='/home/kghasemz/scratch/datasets/FreiHand')
    parser.add_argument('--epochs',       type=int,   default=150)
    parser.add_argument('--batch-size',   type=int,   default=32)
    parser.add_argument('--lr',           type=float, default=1e-4)
    parser.add_argument('--alpha',        type=float, default=0.1,  help='Depth loss weight — ~0.1 balances L1 depth with L1 2D/IMG_SIZE')
    parser.add_argument('--gamma',        type=float, default=0.05,
                        help='SDF loss weight — requires --sdf-data (0 = disabled)')
    parser.add_argument('--freeze-backbone-epochs', type=int, default=10,
                        help='Freeze ResNet backbone for this many epochs at start (0 = no freeze)')
    parser.add_argument('--num-workers',  type=int,   default=4)
    parser.add_argument('--patience',      type=int,   default=20,
                        help='Early stopping patience in epochs (0 = disabled)')
    parser.add_argument('--resume',       type=str,   default=None)
    parser.add_argument('--save-dir',     type=str,   default='.')
    parser.add_argument('--no-pretrain',  action='store_true',
                        help='Do not load ImageNet weights for ResNet-50 backbone')
    parser.add_argument('--device',       type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    print('\n' + '='*65)
    print('Full Training — HOISDF-Inspired 2.5D Hand Pose')
    print('='*65)
    print(f'  Epochs      : {args.epochs}')
    print(f'  Batch size  : {args.batch_size}')
    print(f'  LR          : {args.lr}')
    print(f'  α (depth)   : {args.alpha}')
    print(f'  γ (SDF)     : {args.gamma}  (on-the-fly bone-segment SDF)')
    print(f'  Patience    : {args.patience}  (early stopping, 0=disabled)')
    print(f'  Pretrained  : {not args.no_pretrain}')
    print(f'  Device      : {args.device}')
    print(f'  Save dir    : {args.save_dir}\n')

    device = torch.device(args.device)
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
    model = SDFHandPoseNet(
        num_kpts=21,
        pretrained_backbone=(not args.no_pretrain),
    ).to(device)

    # Cosine annealing with 5% linear warmup (Fix 4).
    # Replaces the previous 0.7×/5-epoch exponential decay which dropped the LR
    # too aggressively, preventing the model from escaping early-training plateaus.
    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    total_steps  = args.epochs * len(train_loader)
    warmup_steps = max(1, int(0.05 * total_steps))   # 5% of total steps

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    start_epoch    = 1
    best_mpjpe     = float('inf')
    epochs_no_improve = 0
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

    # torch.compile requires Triton which is not supported on Windows.
    fwd_model = model

    scaler      = GradScaler('cuda', enabled=(device.type == 'cuda'))
    global_step = (start_epoch - 1) * len(train_loader)

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()

        # ── Backbone freeze / unfreeze ────────────────────────────────────
        if args.freeze_backbone_epochs > 0:
            freeze = (epoch <= args.freeze_backbone_epochs)
            for p in model.backbone.parameters():
                p.requires_grad = not freeze
            if epoch == 1:
                print(f'  Backbone FROZEN for first {args.freeze_backbone_epochs} epochs')
            elif epoch == args.freeze_backbone_epochs + 1:
                print(f'  Backbone UNFROZEN at epoch {epoch}')

        # ── Train ─────────────────────────────────────────────────────────
        model.train()
        train_losses = []

        pbar = tqdm(train_loader, desc=f'Ep {epoch:>3}/{args.epochs}',
                    ncols=180, leave=False)

        for batch in pbar:
            imgs  = batch['image'].to(device, non_blocking=True)
            gt_2d = batch['pose_2d_gt'].to(device, non_blocking=True)
            gt_z  = batch['depth_rel_gt'].to(device, non_blocking=True)

            # Derive GT joint positions in normalised model space on-the-fly.
            # gt_2d is already augmentation-corrected by the dataset — converting
            # here keeps SDF targets consistent with the augmented image.
            # u,v: pixel [0, IMG_SIZE] → normalised [-1, 1]  (matches pts[:,:,:2])
            # d  : depth_rel is already in scale-normalised units
            u_norm = gt_2d[:, :, 0] / (IMG_SIZE / 2.0) - 1.0  # (B, K)
            v_norm = gt_2d[:, :, 1] / (IMG_SIZE / 2.0) - 1.0  # (B, K)
            gt_joints_norm = torch.stack([u_norm, v_norm, gt_z], dim=-1)  # (B, K, 3)

            opt.zero_grad(set_to_none=True)

            with autocast('cuda', enabled=(device.type == 'cuda')):
                pred_2d, pred_z, sdf_vals, pts = fwd_model(imgs)

                loss, lxy, lz, lsdf = loss_full(
                    pred_2d, pred_z, gt_2d, gt_z,
                    sdf_vals=sdf_vals, pts=pts,
                    gt_joints_norm=gt_joints_norm,
                    alpha=args.alpha, gamma=args.gamma,
                )

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            scheduler.step()
            global_step += 1

            train_losses.append((loss.item(), lxy, lz, lsdf))

            avg = np.mean(train_losses, axis=0)
            pbar.set_postfix(loss=f'{avg[0]:.4f}', xy=f'{avg[1]:.4f}',
                             z=f'{avg[2]:.4f}', sdf=f'{avg[3]:.4f}',
                             lr=f'{opt.param_groups[0]["lr"]:.1e}')

        pbar.close()
        avg_loss, avg_lxy, avg_lz, avg_lsdf = np.mean(train_losses, axis=0)
        current_lr = opt.param_groups[0]['lr']
        epoch_time = time.time() - epoch_start

        # ── Validate ──────────────────────────────────────────────────────
        print(f'  Ep {epoch} validating...', end='\r')
        val_mpjpe = validate(fwd_model, val_loader, device)

        hist['train_loss'].append(avg_loss)
        hist['val_mpjpe'].append(val_mpjpe)
        hist['lr'].append(current_lr)

        if device.type == 'cuda':
            mem_alloc    = torch.cuda.memory_allocated()  / 1e9
            mem_reserved = torch.cuda.memory_reserved()   / 1e9
            torch.cuda.empty_cache()
            mem_str = f'  mem={mem_alloc:.1f}/{mem_reserved:.1f}GB'
        else:
            mem_str = ''

        print(f'Ep {epoch:>3}/{args.epochs}  '
              f'loss={avg_loss:.4f}  xy={avg_lxy:.4f}  z={avg_lz:.4f}  '
              f'sdf={avg_lsdf:.4f}  val_MPJPE={val_mpjpe:.4f}  '
              f'lr={current_lr:.1e}  time={epoch_time/60:.1f}min{mem_str}'
              + ('  ★ best' if val_mpjpe <= min(hist['val_mpjpe']) else ''))

        # ── Early stopping & best checkpoint ─────────────────────────────
        if val_mpjpe < best_mpjpe:
            best_mpjpe        = val_mpjpe
            epochs_no_improve = 0
            torch.save({
                'epoch':           epoch,
                'model_state':     model.state_dict(),
                'optimizer_state': opt.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'best_mpjpe':      best_mpjpe,
                'history':         hist,
                'args':            vars(args),
            }, os.path.join(args.save_dir, 'sdf_best_model.pt'))
        else:
            epochs_no_improve += 1
            if args.patience > 0 and epochs_no_improve >= args.patience:
                print(f'\nEarly stopping: no improvement for {args.patience} epochs. '
                      f'Best val MPJPE: {best_mpjpe:.4f}')
                break

        if epoch % 10 == 0:
            torch.save({
                'epoch':           epoch,
                'model_state':     model.state_dict(),
                'optimizer_state': opt.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'best_mpjpe':      best_mpjpe,
                'history':         hist,
                'args':            vars(args),
            }, os.path.join(args.save_dir, 'sdf_checkpoint.pt'))
 
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
    plt.savefig(os.path.join(args.save_dir, 'sdf_train_curves.png'), dpi=130)
    plt.close()
    print('✓ Saved sdf_train_curves.png')


if __name__ == '__main__':
    main()
