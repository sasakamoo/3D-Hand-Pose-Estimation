"""
train_hybrid.py — Training script for HybridHandPoseNet
========================================================
Trains the hybrid model (model_hybrid.py) on the FreiHAND dataset with three
additional loss terms compared to train_sdf.py:

  1. Auxiliary heatmap loss  (lambda_hm,   default 1.0)
     MSE between predicted heatmap logits and on-the-fly Gaussian GT heatmaps.

  2. Bone length regularisation  (lambda_bone, default 0.5)
     L1 between predicted and GT 3D bone lengths using FreiHAND bone segments.

  3. Joint-centred point sampling in the model's forward pass
     (50% uniform + 50% Gaussian around GT joints, sigma=0.1 normalised).

Usage:
    python train_hybrid.py --data-root ../Datasets/FreiHAND_pub_v2/

    # With custom loss weights:
    python train_hybrid.py --lambda-hm 1.5 --lambda-bone 0.3

    # Resume from checkpoint:
    python train_hybrid.py --resume hybrid_checkpoint.pt

Outputs:
    hybrid_best_model.pt      best validation MPJPE checkpoint
    hybrid_checkpoint.pt      periodic checkpoint (every 10 epochs)
    hybrid_train_curves.png   training curves plot
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

from model_hybrid import HybridHandPoseNet, IMG_SIZE, K
from dataset       import FreiHANDDataset
from model         import reconstruct_3d_from_25d


# ── FreiHAND bone segments ────────────────────────────────────────────────────
# Kinematic chain connections: (parent, child) joint index pairs.
# Matches the standard 21-joint hand skeleton layout used in FreiHAND.

BONE_SEGMENTS = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle
    (0, 9), (9, 10), (10, 11), (11, 12),
    # Ring
    (0, 13), (13, 14), (14, 15), (15, 16),
    # Pinky
    (0, 17), (17, 18), (18, 19), (19, 20),
]


# ============================================================================
# GT heatmap generation (vectorised — no Python loops over batch)
# ============================================================================

def make_gt_heatmaps(gt_2d: torch.Tensor,
                     hm_size: int = IMG_SIZE,
                     sigma: float = 2.0) -> torch.Tensor:
    """
    Generate GT Gaussian heatmaps from 2D pixel coordinates.

    Fully vectorised — no Python loops.  Uses broadcasting over the spatial
    grid to compute per-joint Gaussian in a single tensor operation.

    Args:
        gt_2d   : (B, K, 2)  GT 2D pixel coordinates  (x, y) in [0, hm_size]
        hm_size : spatial size of heatmaps (default 128)
        sigma   : Gaussian standard deviation in pixels (default 2.0)

    Returns:
        gt_hm   : (B, K, hm_size, hm_size)  Gaussian heatmaps, values in [0, 1]
                  Peak value is 1.0 at the GT location; falls off as a Gaussian.
    """
    B, num_joints, _ = gt_2d.shape
    device = gt_2d.device

    # Pixel grid: (1, 1, H, W) each
    ys = torch.arange(hm_size, dtype=torch.float32, device=device)
    xs = torch.arange(hm_size, dtype=torch.float32, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')   # (H, W)
    grid_x = grid_x.view(1, 1, hm_size, hm_size)             # (1, 1, H, W)
    grid_y = grid_y.view(1, 1, hm_size, hm_size)

    # GT joint positions: (B, K, 1, 1) for broadcasting
    mu_x = gt_2d[:, :, 0].unsqueeze(-1).unsqueeze(-1)        # (B, K, 1, 1)
    mu_y = gt_2d[:, :, 1].unsqueeze(-1).unsqueeze(-1)

    # Gaussian: exp(-((x - mu_x)^2 + (y - mu_y)^2) / (2 * sigma^2))
    dist_sq = (grid_x - mu_x) ** 2 + (grid_y - mu_y) ** 2   # (B, K, H, W)
    gt_hm   = torch.exp(-dist_sq / (2.0 * sigma ** 2))       # (B, K, H, W)
    return gt_hm


# ============================================================================
# Loss functions
# ============================================================================

def heatmap_loss(hm_logits: torch.Tensor,
                 gt_2d: torch.Tensor,
                 sigma: float = 2.0) -> torch.Tensor:
    """
    MSE loss between predicted heatmap logits and Gaussian GT heatmaps.

    The GT heatmaps are generated on-the-fly from gt_2d pixel coordinates.
    Peak value is 1.0 at the joint location, falling off as a Gaussian
    with sigma=2px.  The model is trained to match these targets directly
    from the raw conv output (logits), which is appropriate for MSE.

    Args:
        hm_logits : (B, K, H, W)  predicted heatmap logits (model output)
        gt_2d     : (B, K, 2)     GT 2D pixel coords in [0, IMG_SIZE]
        sigma     : Gaussian sigma in pixels

    Returns:
        scalar MSE loss
    """
    gt_hm = make_gt_heatmaps(gt_2d, hm_size=hm_logits.shape[-1], sigma=sigma)
    return F.mse_loss(hm_logits, gt_hm)


def bone_length_loss(pred_3d: torch.Tensor,
                     gt_3d: torch.Tensor) -> torch.Tensor:
    """
    L1 loss on predicted vs GT 3D bone lengths.

    Penalises implausible kinematic chain proportions without constraining
    the absolute scale (the network already has a separate depth term for that).

    Args:
        pred_3d : (B, K, 3)  predicted 3D joint positions
        gt_3d   : (B, K, 3)  GT 3D joint positions

    Returns:
        scalar L1 loss on bone lengths
    """
    pred_lengths = []
    gt_lengths   = []
    for (i, j) in BONE_SEGMENTS:
        pred_lengths.append((pred_3d[:, i] - pred_3d[:, j]).norm(dim=-1))  # (B,)
        gt_lengths.append(  (gt_3d[:, i]   - gt_3d[:, j]).norm(dim=-1))
    pred_lens = torch.stack(pred_lengths, dim=1)   # (B, num_bones)
    gt_lens   = torch.stack(gt_lengths,   dim=1)
    return F.l1_loss(pred_lens, gt_lens)


def loss_full(pred_2d: torch.Tensor,
              pred_z: torch.Tensor,
              gt_2d: torch.Tensor,
              gt_z: torch.Tensor,
              hm_logits: torch.Tensor,
              K_mat: torch.Tensor,
              alpha: float = 0.1,
              lambda_hm: float = 1.0,
              lambda_bone: float = 0.5) -> tuple:
    """
    Combined training loss for HybridHandPoseNet.

    Terms:
        L_xy   : L1 on 2D pixel positions normalised by IMG_SIZE
        L_z    : L1 on root-relative depth
        L_hm   : MSE on auxiliary heatmap predictions vs Gaussian GT
        L_bone : L1 on predicted vs GT 3D bone lengths

    Args:
        pred_2d    : (B, K, 2)     predicted 2D positions in pixels
        pred_z     : (B, K)        predicted depth_rel
        gt_2d      : (B, K, 2)     GT 2D positions in pixels
        gt_z       : (B, K)        GT depth_rel
        hm_logits  : (B, K, H, W)  predicted heatmap logits
        K_mat      : (B, 3, 3)     camera intrinsics
        alpha      : depth loss weight (default 0.1)
        lambda_hm  : heatmap loss weight (default 1.0)
        lambda_bone: bone length loss weight (default 0.5)

    Returns:
        (total, L_xy.item(), L_z.item(), L_hm.item(), L_bone.item())
    """
    L_xy = F.l1_loss(pred_2d, gt_2d) / IMG_SIZE
    L_z  = F.l1_loss(pred_z,  gt_z)

    L_hm = heatmap_loss(hm_logits, gt_2d, sigma=2.0)

    # Reconstruct 3D for bone length loss
    pred_3d = reconstruct_3d_from_25d(pred_2d, pred_z, K_mat, img_size=IMG_SIZE)
    gt_3d   = reconstruct_3d_from_25d(gt_2d,   gt_z,   K_mat, img_size=IMG_SIZE)

    L_bone = bone_length_loss(pred_3d, gt_3d)

    total = L_xy + alpha * L_z + lambda_hm * L_hm + lambda_bone * L_bone
    return total, L_xy.item(), L_z.item(), L_hm.item(), L_bone.item()


# ============================================================================
# gt_joints_norm helper
# ============================================================================

def compute_gt_joints_norm(gt_2d: torch.Tensor,
                           gt_z: torch.Tensor) -> torch.Tensor:
    """
    Convert GT 2D pixel coords + depth_rel into normalised [-1, 1]^3 model space.

    Convention (matches model_sdf.py point sampling space):
        u = pixel_x / (IMG_SIZE / 2) - 1   ∈ [-1, 1]
        v = pixel_y / (IMG_SIZE / 2) - 1   ∈ [-1, 1]
        d = depth_rel (already normalised, kept as-is but clamped to [-1,1])

    Args:
        gt_2d : (B, K, 2)  GT 2D pixel coords in [0, IMG_SIZE]
        gt_z  : (B, K)     GT root-relative normalised depth

    Returns:
        (B, K, 3)  GT joints in [-1, 1]^3
    """
    half = IMG_SIZE / 2.0
    u = gt_2d[:, :, 0] / half - 1.0              # x pixel → [-1, 1]
    v = gt_2d[:, :, 1] / half - 1.0              # y pixel → [-1, 1]
    d = torch.clamp(gt_z, -1.0, 1.0)             # depth_rel → [-1, 1]
    return torch.stack([u, v, d], dim=-1)          # (B, K, 3)


# ============================================================================
# Validation — MPJPE in normalised units
# ============================================================================

@torch.no_grad()
def validate(model: HybridHandPoseNet,
             loader: DataLoader,
             device: torch.device) -> float:
    """
    Compute mean MPJPE over the validation set.

    Aligns both predicted and GT 3D poses to the wrist joint (joint 0)
    before computing Euclidean distance — standard root-relative evaluation.

    Returns:
        mean MPJPE in normalised units (mean over joints and samples)
    """
    model.eval()
    mpjpe_list = []
    for batch in loader:
        imgs  = batch['image'].to(device, non_blocking=True)
        gt_2d = batch['pose_2d_gt'].to(device, non_blocking=True)
        gt_z  = batch['depth_rel_gt'].to(device, non_blocking=True)
        K_mat = batch['K_mat'].to(device, non_blocking=True)

        with autocast('cuda', enabled=(device.type == 'cuda')):
            # No gt_joints_norm at eval — dense grid is used automatically
            pred_2d, pred_z, _hm, _sdf, _pts = model(imgs)

        pred_3d = reconstruct_3d_from_25d(
            pred_2d.float(), pred_z.float(), K_mat, img_size=IMG_SIZE)
        gt_3d   = reconstruct_3d_from_25d(
            gt_2d, gt_z, K_mat, img_size=IMG_SIZE)

        # Root-relative alignment
        pred_3d = pred_3d - pred_3d[:, 0:1]
        gt_3d   = gt_3d   - gt_3d[:, 0:1]

        mpjpe = (pred_3d - gt_3d).norm(dim=-1).mean(dim=-1)   # (B,)
        mpjpe_list.append(mpjpe.cpu())

    return torch.cat(mpjpe_list).mean().item()


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Train HybridHandPoseNet on FreiHAND')
    parser.add_argument('--data-root',    type=str,
                        default='../Datasets/FreiHAND_pub_v2/',
                        help='Path to FreiHAND dataset root directory')
    parser.add_argument('--epochs',       type=int,   default=150)
    parser.add_argument('--batch-size',   type=int,   default=32)
    parser.add_argument('--lr',           type=float, default=1e-4)
    parser.add_argument('--alpha',        type=float, default=0.1,
                        help='Depth loss weight')
    parser.add_argument('--lambda-hm',   type=float, default=1.0,
                        help='Auxiliary heatmap loss weight')
    parser.add_argument('--lambda-bone', type=float, default=0.5,
                        help='Bone length regularisation loss weight')
    parser.add_argument('--freeze-backbone-epochs', type=int, default=10,
                        help='Freeze ResNet backbone for this many epochs at start')
    parser.add_argument('--num-workers',  type=int,   default=4)
    parser.add_argument('--patience',     type=int,   default=20,
                        help='Early stopping patience in epochs (0 = disabled)')
    parser.add_argument('--resume',       type=str,   default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--save-dir',     type=str,   default='.',
                        help='Directory to save checkpoints and plots')
    parser.add_argument('--no-pretrain',  action='store_true',
                        help='Do not load ImageNet weights for ResNet-50 backbone')
    parser.add_argument('--device',       type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    print('\n' + '=' * 65)
    print('Hybrid Hand Pose Training — with Heatmap + Bone Length Loss')
    print('=' * 65)
    print(f'  Epochs         : {args.epochs}')
    print(f'  Batch size     : {args.batch_size}')
    print(f'  LR             : {args.lr}')
    print(f'  α (depth)      : {args.alpha}')
    print(f'  λ_hm           : {args.lambda_hm}')
    print(f'  λ_bone         : {args.lambda_bone}')
    print(f'  Patience       : {args.patience}  (early stopping, 0=disabled)')
    print(f'  Pretrained     : {not args.no_pretrain}')
    print(f'  Device         : {args.device}')
    print(f'  Save dir       : {args.save_dir}\n')

    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = True

    # ── Datasets & loaders ────────────────────────────────────────────────
    train_ds = FreiHANDDataset(args.data_root, split='train', augment=True)
    val_ds   = FreiHANDDataset(args.data_root, split='val',   augment=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(args.num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
    )

    print(f'Train samples : {len(train_ds)}  ({len(train_loader)} batches/epoch)')
    print(f'Val samples   : {len(val_ds)}   ({len(val_loader)} batches)\n')

    # ── Model, optimiser, scheduler ───────────────────────────────────────
    model = HybridHandPoseNet(
        num_kpts=K,
        pretrained_backbone=(not args.no_pretrain),
        use_sdf=False,          # density=1.0, no SDF gating
    ).to(device)

    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    total_steps  = args.epochs * len(train_loader)
    warmup_steps = max(1, int(0.05 * total_steps))   # 5% linear warmup

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    start_epoch       = 1
    best_mpjpe        = float('inf')
    epochs_no_improve = 0
    hist: dict = {'train_loss': [], 'val_mpjpe': [], 'lr': [],
                  'L_xy': [], 'L_z': [], 'L_hm': [], 'L_bone': []}

    # ── Resume from checkpoint ────────────────────────────────────────────
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state'])
        opt.load_state_dict(ckpt['optimizer_state'])
        scheduler.load_state_dict(ckpt['scheduler_state'])
        start_epoch = ckpt['epoch'] + 1
        best_mpjpe  = ckpt.get('best_mpjpe', float('inf'))
        hist        = ckpt.get('history', hist)
        print(f'Resumed from {args.resume}  '
              f'(epoch {ckpt["epoch"]}, best MPJPE {best_mpjpe:.4f})\n')

    params = sum(p.numel() for p in model.parameters())
    print(f'Model params: {params:,}  ({params * 4 / 1e6:.1f} MB)\n')

    scaler      = GradScaler('cuda', enabled=(device.type == 'cuda'))
    global_step = (start_epoch - 1) * len(train_loader)

    # ── Training loop ─────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()

        # Backbone freeze / unfreeze
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
        train_losses: list = []

        pbar = tqdm(train_loader,
                    desc=f'Ep {epoch:>3}/{args.epochs}',
                    ncols=200, leave=False)

        for batch in pbar:
            imgs  = batch['image'].to(device, non_blocking=True)
            gt_2d = batch['pose_2d_gt'].to(device, non_blocking=True)
            gt_z  = batch['depth_rel_gt'].to(device, non_blocking=True)
            K_mat = batch['K_mat'].to(device, non_blocking=True)

            # Compute GT joint positions in normalised [-1,1]^3 model space
            # for joint-centred point sampling inside the model.
            gt_joints_norm = compute_gt_joints_norm(gt_2d, gt_z)  # (B, K, 3)

            opt.zero_grad(set_to_none=True)

            with autocast('cuda', enabled=(device.type == 'cuda')):
                pred_2d, pred_z, hm_logits, _sdf, _pts = model(
                    imgs, gt_joints_norm=gt_joints_norm)

                loss, lxy, lz, lhm, lbone = loss_full(
                    pred_2d, pred_z, gt_2d, gt_z,
                    hm_logits, K_mat,
                    alpha=args.alpha,
                    lambda_hm=args.lambda_hm,
                    lambda_bone=args.lambda_bone,
                )

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            scheduler.step()
            global_step += 1

            train_losses.append((loss.item(), lxy, lz, lhm, lbone))

            avg = np.mean(train_losses, axis=0)
            pbar.set_postfix(
                loss=f'{avg[0]:.4f}',
                xy=f'{avg[1]:.4f}',
                z=f'{avg[2]:.4f}',
                hm=f'{avg[3]:.4f}',
                bone=f'{avg[4]:.4f}',
                lr=f'{opt.param_groups[0]["lr"]:.1e}',
            )

        pbar.close()
        arr = np.mean(train_losses, axis=0)
        avg_loss, avg_lxy, avg_lz, avg_lhm, avg_lbone = arr
        current_lr = opt.param_groups[0]['lr']
        epoch_time = time.time() - epoch_start

        # ── Validate ──────────────────────────────────────────────────────
        print(f'  Ep {epoch} validating...', end='\r')
        val_mpjpe = validate(model, val_loader, device)

        hist['train_loss'].append(float(avg_loss))
        hist['val_mpjpe'].append(float(val_mpjpe))
        hist['lr'].append(float(current_lr))
        hist['L_xy'].append(float(avg_lxy))
        hist['L_z'].append(float(avg_lz))
        hist['L_hm'].append(float(avg_lhm))
        hist['L_bone'].append(float(avg_lbone))

        if device.type == 'cuda':
            mem_alloc    = torch.cuda.memory_allocated()  / 1e9
            mem_reserved = torch.cuda.memory_reserved()   / 1e9
            torch.cuda.empty_cache()
            mem_str = f'  mem={mem_alloc:.1f}/{mem_reserved:.1f}GB'
        else:
            mem_str = ''

        is_best = val_mpjpe <= min(hist['val_mpjpe'])
        print(
            f'Ep {epoch:>3}/{args.epochs}  '
            f'loss={avg_loss:.4f}  xy={avg_lxy:.4f}  z={avg_lz:.4f}  '
            f'hm={avg_lhm:.4f}  bone={avg_lbone:.4f}  '
            f'val_MPJPE={val_mpjpe:.4f}  '
            f'lr={current_lr:.1e}  time={epoch_time/60:.1f}min{mem_str}'
            + ('  * best' if is_best else '')
        )

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
            }, os.path.join(args.save_dir, 'hybrid_best_model.pt'))
        else:
            epochs_no_improve += 1
            if args.patience > 0 and epochs_no_improve >= args.patience:
                print(
                    f'\nEarly stopping: no improvement for {args.patience} epochs. '
                    f'Best val MPJPE: {best_mpjpe:.4f}')
                break

        # Periodic checkpoint every 10 epochs
        if epoch % 10 == 0:
            torch.save({
                'epoch':           epoch,
                'model_state':     model.state_dict(),
                'optimizer_state': opt.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'best_mpjpe':      best_mpjpe,
                'history':         hist,
                'args':            vars(args),
            }, os.path.join(args.save_dir, 'hybrid_checkpoint.pt'))

    print(f'\nBest val MPJPE: {best_mpjpe:.4f} (normalised units)')
    print(f'Checkpoints saved to: {args.save_dir}/')

    # ── Plot training curves ──────────────────────────────────────────────
    ep = range(1, len(hist['train_loss']) + 1)
    fig, axes = plt.subplots(2, 3, figsize=(18, 8))

    axes[0, 0].semilogy(ep, hist['train_loss'], lw=2)
    axes[0, 0].set_title('Train loss (log)')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(ep, hist['val_mpjpe'], lw=2, color='green')
    axes[0, 1].set_title('Val MPJPE (norm units)')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].grid(alpha=0.3)

    axes[0, 2].semilogy(ep, hist['lr'], lw=2, color='brown')
    axes[0, 2].set_title('Learning rate')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].grid(alpha=0.3)

    axes[1, 0].semilogy(ep, hist['L_xy'], lw=2, color='steelblue')
    axes[1, 0].set_title('L_xy (2D position, log)')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].semilogy(ep, hist['L_hm'], lw=2, color='darkorange')
    axes[1, 1].set_title('L_hm (heatmap, log)')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].grid(alpha=0.3)

    axes[1, 2].semilogy(ep, hist['L_bone'], lw=2, color='purple')
    axes[1, 2].set_title('L_bone (bone length, log)')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].grid(alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(args.save_dir, 'hybrid_train_curves.png')
    plt.savefig(plot_path, dpi=130)
    plt.close()
    print(f'Saved {plot_path}')


if __name__ == '__main__':
    main()
