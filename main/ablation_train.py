"""
ablation_train.py  —  Fixed & extended fine-tuning for ablation study
======================================================================
Bugs fixed vs previous version:
  1. Loss weights were missing — model returns raw unweighted losses,
     train.py multiplies them. Without this total_loss was ~100x wrong.
  2. Adam used instead of AdamW (original uses AdamW).
  3. Optimizer + scheduler state NOT restored — LR jumped to 5e-5 from 1e-5.
  4. strict=False on load_state_dict silently ignored key mismatches,
     leaving the backbone at ImageNet init instead of trained weights.
  5. bone_loss: wrist zero-prepend made gradient not flow through wrist —
     wrist-adjacent BONE_PAIRS excluded instead.
  6. reproj_loss weight=1.0 dominated total loss (should be ~0.05).
  7. Scheduler stepped every epoch but step_size=9 with only 11 epochs
     meant it never fired — now uses MultiStepLR to match original.
  8. mano-only loss filter was keyed on run_dir_name string (fragile) —
     replaced with explicit --mano_only flag.
  9. clip_grad_norm_ applied to all params including frozen ones —
     now clips only trainable params.
 10. [NEW] hand_transformer is now UNfrozen alongside mano_head. The
     transformer produces the query tokens that condition the MANO head;
     keeping it frozen meant the MANO head received fixed, untailored
     features and could not learn pose/shape effectively.
     Also unfreeze mano_query_embed, linear_pose, linear_shape so the
     full MANO conditioning pathway has gradient flow end-to-end.
     mano_mesh_loss weight lowered 1e-2 → 1e-3 to reduce oscillation.

Usage:
    python main/ablation_train.py \\
        --ablation      bone_loss \\
        --resume        /path/to/snapshot_69_1331.pth.tar \\
        --freihand_dir  /path/to/dataset \\
        --end_epoch     100 \\
        --run_dir_name  abl_bone_loss
"""

import os
os.environ['MPLBACKEND']   = 'Agg'
os.environ['DISPLAY']      = ''
os.environ['MPLCONFIGDIR'] = '/tmp'

import sys, argparse
import numpy as np
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn

_file_dir = os.path.dirname(os.path.abspath(__file__))
_root_dir = os.path.dirname(_file_dir)
for _p in [_root_dir, _file_dir,
           os.path.join(_root_dir, 'common'),
           os.path.join(_root_dir, 'data')]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ─────────────────────────────────────────────────────────────────────────────
# Loss weight application  (mirrors train.py exactly)
# ─────────────────────────────────────────────────────────────────────────────

def apply_loss_weights(out_dict, cfg):
    """
    Reproduce the exact loss weighting from train.py.
    Returns (total_loss, losses_dict) where losses_dict has weighted values.
    """
    losses = {k: v.mean() for k, v in out_dict.items()
              if '_out' not in k and torch.is_tensor(v) and v.ndim <= 1}

    if 'sdfhand_loss'      in losses: losses['sdfhand_loss']      *= cfg.sdf_hand_weight
    if 'joint_heatmap'     in losses: losses['joint_heatmap']     *= cfg.hm_weight
    if 'loss_joint_3d'     in losses: losses['loss_joint_3d']     *= cfg.joint_weight
    if 'loss_joint_cls'    in losses: losses['loss_joint_cls']    *= cfg.cls_weight
    if 'loss_all_joint_3d' in losses: losses['loss_all_joint_3d'] *= cfg.joint_weight
    # FIX: mano_mesh_loss lowered 1e-2 → 1e-3.
    # At 1e-2 the mesh loss (MSE in metres²) oscillates and destabilises
    # the newly-unfrozen transformer weights.
    if 'mano_mesh_loss'    in losses: losses['mano_mesh_loss']    *= 1e-3
    if 'mano_joint_loss'   in losses: losses['mano_joint_loss']   *= 1e-2
    if 'pose_param_loss'   in losses: losses['pose_param_loss']   *= 1e-4
    if 'shape_param_loss'  in losses: losses['shape_param_loss']  *= 1e-5

    total = sum(losses.values())
    return total, losses


# ─────────────────────────────────────────────────────────────────────────────
# Bone length consistency loss
# ─────────────────────────────────────────────────────────────────────────────

# FIX: wrist (joint 0) is a zero-constant with no gradient. Exclude all
# bone pairs that connect TO the wrist so those bones are not zero-grad.
BONE_PAIRS = [
    # finger chains only — no wrist connection
    (1,2),(2,3),(3,4),       # index
    (5,6),(6,7),(7,8),       # middle
    (9,10),(10,11),(11,12),  # ring
    (13,14),(14,15),(15,16), # pinky
    (17,18),(18,19),(19,20), # thumb
]

def bone_length_loss(pred_j21, gt_j21):
    """
    pred_j21 / gt_j21 : (B, 21, 3) root-relative metres
    Returns mean absolute bone length error normalised by number of bones.
    """
    loss = pred_j21.new_zeros(1)
    for s, e in BONE_PAIRS:
        pred_len = (pred_j21[:, e] - pred_j21[:, s]).norm(dim=-1)
        gt_len   = (gt_j21[:, e]   - gt_j21[:, s]).norm(dim=-1)
        loss     = loss + (pred_len - gt_len).abs().mean()
    return loss / len(BONE_PAIRS)


# ─────────────────────────────────────────────────────────────────────────────
# 2D reprojection loss
# ─────────────────────────────────────────────────────────────────────────────

def reprojection_loss(pred_j21_rr, mano_root, cam_intr, gt_2d_hm,
                      inp_res=256, hm_res=128):
    """
    pred_j21_rr : (B,21,3) root-relative metres
    mano_root   : (B,3)    wrist in camera space metres
    cam_intr    : (B,3,3)  augmented camera intrinsics
    gt_2d_hm    : (B,21,2) GT 2D in heatmap pixels (128px space)
    """
    pred_cam    = pred_j21_rr + mano_root.unsqueeze(1)
    proj        = torch.bmm(cam_intr, pred_cam.transpose(1, 2)).transpose(1, 2)
    depth       = proj[:, :, 2:3].clamp(min=1e-6)
    pred_2d_inp = proj[:, :, :2] / depth
    pred_2d_hm  = pred_2d_inp * (hm_res / inp_res)
    return (pred_2d_hm - gt_2d_hm).abs().mean() / hm_res


# ─────────────────────────────────────────────────────────────────────────────
# Metric
# ─────────────────────────────────────────────────────────────────────────────

def compute_mpjpe(pred, gt):
    """Root-relative MPJPE in metres."""
    pred = pred - pred[:, 0:1]
    gt   = gt   - gt[:, 0:1]
    return (pred - gt).norm(dim=-1).mean().item()


# ─────────────────────────────────────────────────────────────────────────────
# Args
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ablation', type=str, required=True,
                   choices=['baseline', 'bone_loss', 'reproj_loss', 'lr_sched', 'sdf_pts'])
    p.add_argument('--resume',       type=str, required=True)
    p.add_argument('--freihand_dir', type=str, required=True)
    p.add_argument('--run_dir_name', type=str, required=True)
    p.add_argument('--end_epoch',    type=int, default=100)
    p.add_argument('--bone_weight',   type=float, default=0.5)
    p.add_argument('--reproj_weight', type=float, default=0.05)
    p.add_argument('--mano_only', action='store_true',
                   help='Only back-prop MANO losses (use when fine-tuning MANO head)')
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cudnn.benchmark = True

    # ── config ────────────────────────────────────────────────────────────────
    from main.config import cfg
    cfg.freihand_data_dir = args.freihand_dir

    if args.ablation == 'sdf_pts':
        cfg.num_samp_hand = 1200
        print('  [ablation] num_samp_hand = 1200')

    cfg.set_args('0', args.run_dir_name, continue_train=False)
    cfg.create_run_dirs()
    cfg.calc_mutliscale_dim(cfg.use_big_decoder, cfg.resnet_type)

    inp_res = cfg.input_img_shape[0]
    hm_res  = cfg.output_hm_shape[0]

    # ── dataset ───────────────────────────────────────────────────────────────
    from freihand import Dataset as FreiHandDataset
    from torch.utils.data import DataLoader

    train_ds = FreiHandDataset('train')
    eval_ds  = FreiHandDataset('evaluation')
    train_loader = DataLoader(train_ds, batch_size=22, shuffle=True,
                              num_workers=8, pin_memory=True, drop_last=True)
    eval_loader  = DataLoader(eval_ds,  batch_size=22, shuffle=False,
                              num_workers=4, pin_memory=True)
    print(f'Train: {len(train_ds)}  Eval: {len(eval_ds)}')

    # ── load checkpoint ───────────────────────────────────────────────────────
    print(f'Loading checkpoint: {args.resume}')
    raw_ckpt    = torch.load(args.resume, map_location='cpu')
    state       = raw_ckpt.get('network', raw_ckpt.get('model_state', raw_ckpt))
    state       = {k.replace('module.', ''): v for k, v in state.items()}
    start_epoch = raw_ckpt.get('epoch', 69) + 1

    # ── build model & restore weights ─────────────────────────────────────────
    from main.model import get_model
    model = get_model('train')

    missing, unexpected = model.load_state_dict(state, strict=True)
    if missing:
        raise RuntimeError(
            f'Missing keys in checkpoint ({len(missing)}): {missing[:5]}\n'
            f'Cannot safely fine-tune.')
    if unexpected:
        print(f'  Unexpected keys (ignored): {len(unexpected)}')

    model = model.to(device)
    print(f'  Start epoch: {start_epoch}  End epoch: {args.end_epoch}')

    # ── sanity check ──────────────────────────────────────────────────────────
    model.eval()
    _mjes = []
    with torch.no_grad():
        for _i, (_inp, _tgt, _meta) in enumerate(eval_loader):
            if _i >= 15: break
            _id  = {k: v.to(device) if torch.is_tensor(v) else v for k, v in _inp.items()}
            _td  = {k: v.to(device) if torch.is_tensor(v) else v for k, v in _tgt.items()}
            _md  = {k: v.to(device) if torch.is_tensor(v) else v for k, v in _meta.items()}
            _out = model(_id, _td, _md, 'eval', epoch_cnt=1e8)
            _p20 = _out['hand_joints_out']
            _B   = _p20.shape[0]
            _p21 = torch.cat([torch.zeros(_B, 1, 3, device=device), _p20], dim=1)
            _g21 = _td['joint_cam_no_trans'] / 1000
            _p21r = _p21 - _p21[:, 0:1]
            _g21r = _g21 - _g21[:, 0:1]
            _mjes.append((_p21r - _g21r).norm(dim=-1).mean().item())
    sanity_mje = float(np.mean(_mjes)) * 1000
    print(f'  Sanity MJE: {sanity_mje:.2f} mm  (expect ~10 mm for fine-tune start)')
    if sanity_mje > 50.0:
        raise RuntimeError(
            f'Checkpoint loaded incorrectly — MJE={sanity_mje:.1f}mm. Aborting.')
    model.train()

    # ── freeze strategy ───────────────────────────────────────────────────────
    # FIX: hand_transformer is now TRAINABLE. It produces the query tokens
    # that condition the MANO head. Freezing it meant the MANO head always
    # received the same fixed features regardless of pose/shape targets,
    # making it impossible to learn better mesh predictions.
    #
    # Trainable modules (full MANO conditioning pathway):
    #   hand_transformer      — produces MANO query features
    #   mano_query_embed      — learnable query positional embeddings
    #   mano_head             — MANO regressor
    #   linear_pose           — pose parameter projection
    #   linear_shape          — shape parameter projection
    #
    # Frozen modules (heavy backbone — keep stable, save memory/time):
    #   backbone_net, decoder_net, hand_sdf_decoder
    #   linear_transformerin, linear_sdfin
    #   linear_handvote, linear_handcls
    FROZEN_MODULES = [
        'backbone_net',
        'decoder_net',
        'hand_sdf_decoder',
        'linear_transformerin',
        'linear_sdfin',
        'linear_handvote',
        'linear_handcls',
    ]
    # hand_transformer is intentionally NOT in the frozen list
    TRAINABLE_MODULES = [
        'hand_transformer',
        'mano_query_embed',
        'mano_head',
        'linear_pose',
        'linear_shape',
    ]

    n_frozen = 0
    for name, param in model.named_parameters():
        if any(name.startswith(m) for m in FROZEN_MODULES):
            param.requires_grad = False
            n_frozen += 1

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'  Frozen {n_frozen} param groups. Trainable params: {n_trainable:,}')
    print(f'  Trainable modules: {", ".join(TRAINABLE_MODULES)}')

    # ── optimiser ─────────────────────────────────────────────────────────────
    # Use a slightly lower LR for hand_transformer (it's pre-trained) vs
    # mano_head (needs more signal). Differential LR prevents the transformer
    # from being over-updated and forgetting its learned joint features.
    transformer_params = [p for n, p in model.named_parameters()
                          if p.requires_grad and n.startswith('hand_transformer')]
    mano_params        = [p for n, p in model.named_parameters()
                          if p.requires_grad and not n.startswith('hand_transformer')]

    optimizer = optim.AdamW([
        {'params': transformer_params, 'lr': 1e-5},   # conservative — pre-trained
        {'params': mano_params,        'lr': 3e-5},   # faster — mostly random init
    ], weight_decay=1e-4)

    print(f'  Optimizer: AdamW  transformer_lr=1e-5  mano_lr=3e-5')

    # Optimizer restore: skip if param group sizes differ (they will, since
    # the checkpoint optimizer covered all 110M params, ours covers ~9M).
    if 'optimizer' in raw_ckpt:
        ckpt_n_groups = len(raw_ckpt['optimizer'].get('param_groups', []))
        our_n_groups  = len(optimizer.param_groups)
        if ckpt_n_groups == our_n_groups:
            try:
                optimizer.load_state_dict(raw_ckpt['optimizer'])
                print('  Optimizer state restored')
            except ValueError as e:
                print(f'  Optimizer restore failed ({e}) — using fresh LR')
        else:
            print(f'  Optimizer param group mismatch '
                  f'(ckpt={ckpt_n_groups}, ours={our_n_groups}) — using fresh LR')
    else:
        print('  No optimizer state in checkpoint — using fresh LR')

    # ── scheduler ─────────────────────────────────────────────────────────────
    n_remaining = args.end_epoch - start_epoch

    if args.ablation == 'lr_sched':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_remaining, eta_min=1e-7)
        print('  [ablation] CosineAnnealingLR')
    else:
        m1 = start_epoch + n_remaining // 3
        m2 = start_epoch + 2 * n_remaining // 3
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[m1, m2], gamma=0.5)
        print(f'  MultiStepLR milestones: [{m1}, {m2}]')

    if 'lr_scheduler' in raw_ckpt and args.ablation != 'lr_sched':
        try:
            scheduler.load_state_dict(raw_ckpt['lr_scheduler'])
            print('  Scheduler state restored')
        except Exception:
            print('  Scheduler state incompatible, using fresh')

    # ── logging ───────────────────────────────────────────────────────────────
    log_path = os.path.join(cfg.log_dir, 'ablation_log.txt')
    os.makedirs(cfg.log_dir, exist_ok=True)

    def log(msg):
        print(msg, flush=True)
        with open(log_path, 'a') as f:
            f.write(msg + '\n')

    log(f'Ablation      : {args.ablation}')
    log(f'Resume        : {args.resume}')
    log(f'Sanity MJE    : {sanity_mje:.2f} mm')
    log(f'Epochs        : {start_epoch} -> {args.end_epoch}')
    log(f'LR transformer: {optimizer.param_groups[0]["lr"]:.2e}')
    log(f'LR mano       : {optimizer.param_groups[1]["lr"]:.2e}')
    log(f'Trainable     : {", ".join(TRAINABLE_MODULES)}')
    log(f'MANO only     : {args.mano_only}')
    if args.ablation == 'bone_loss':
        log(f'bone_weight   : {args.bone_weight}')
    if args.ablation == 'reproj_loss':
        log(f'reproj_weight : {args.reproj_weight}')
    log('-' * 60)

    best_mje = float('inf')

    # ── training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.end_epoch):
        model.train()
        epoch_losses = []
        epoch_extra  = []

        for itr, (inputs, targets, meta) in enumerate(train_loader):
            inputs_d  = {k: v.to(device) if torch.is_tensor(v) else v
                         for k, v in inputs.items()}
            targets_d = {k: v.to(device) if torch.is_tensor(v) else v
                         for k, v in targets.items()}
            meta_d    = {k: v.to(device) if torch.is_tensor(v) else v
                         for k, v in meta.items()}

            batch_ratio = itr / len(train_loader)
            out = model(inputs_d, targets_d, meta_d, 'train', epoch, batch_ratio)

            total_loss, loss_dict = apply_loss_weights(out, cfg)
            base_loss_val = total_loss.item()

            if args.mano_only:
                mano_only_keys = {'mano_mesh_loss', 'mano_joint_loss',
                                  'pose_param_loss', 'shape_param_loss'}
                mano_losses = [v for k, v in loss_dict.items()
                               if k in mano_only_keys and torch.is_tensor(v)]
                if mano_losses:
                    total_loss = sum(mano_losses)
                else:
                    total_loss = out['mano_mesh_out'].new_zeros(1).squeeze()
                base_loss_val = total_loss.item()

            extra_loss_val = 0.0

            # ── extra ablation losses ──────────────────────────────────────
            if args.ablation == 'bone_loss':
                pred_j20 = out['hand_joints_out']
                B        = pred_j20.shape[0]
                wrist    = torch.zeros(B, 1, 3, device=device)
                pred_j21 = torch.cat([wrist, pred_j20], dim=1)
                gt_j21   = targets_d['joint_cam_no_trans'] / 1000
                gt_j21rr = gt_j21 - gt_j21[:, 0:1]
                extra    = args.bone_weight * bone_length_loss(pred_j21, gt_j21rr)
                total_loss     = total_loss + extra
                extra_loss_val = extra.item()

            elif args.ablation == 'reproj_loss':
                pred_j20 = out['hand_joints_out']
                B        = pred_j20.shape[0]
                wrist    = torch.zeros(B, 1, 3, device=device)
                pred_j21 = torch.cat([wrist, pred_j20], dim=1)
                extra    = args.reproj_weight * reprojection_loss(
                    pred_j21, meta_d['mano_root'], meta_d['cam_intr'],
                    targets_d['joint_coord'], inp_res=inp_res, hm_res=hm_res)
                total_loss     = total_loss + extra
                extra_loss_val = extra.item()

            optimizer.zero_grad()
            total_loss.backward()

            trainable_params = [p for p in model.parameters() if p.requires_grad]
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)

            optimizer.step()

            epoch_losses.append(total_loss.item())
            epoch_extra.append(extra_loss_val)

            if itr % 100 == 0:
                ex = f'  extra={extra_loss_val:.4f}' if extra_loss_val > 0 else ''
                lrs = '/'.join(f'{pg["lr"]:.1e}' for pg in optimizer.param_groups)
                print(f'  Ep {epoch}/{args.end_epoch}  itr {itr}/{len(train_loader)}'
                      f'  loss={total_loss.item():.4f}  base={base_loss_val:.4f}'
                      f'{ex}  lr={lrs}', flush=True)

        scheduler.step()

        # ── eval ──────────────────────────────────────────────────────────────
        model.eval()
        mje_list = []
        with torch.no_grad():
            for inputs, targets, meta in eval_loader:
                inputs_d  = {k: v.to(device) if torch.is_tensor(v) else v
                             for k, v in inputs.items()}
                targets_d = {k: v.to(device) if torch.is_tensor(v) else v
                             for k, v in targets.items()}
                meta_d    = {k: v.to(device) if torch.is_tensor(v) else v
                             for k, v in meta.items()}
                out      = model(inputs_d, targets_d, meta_d, 'eval', epoch_cnt=1e8)
                pred_j20 = out['hand_joints_out']
                B        = pred_j20.shape[0]
                pred_j21 = torch.cat([torch.zeros(B, 1, 3, device=device), pred_j20], dim=1)
                gt_j21   = targets_d['joint_cam_no_trans'] / 1000
                mje_list.append(compute_mpjpe(pred_j21, gt_j21))

        eval_mje   = float(np.mean(mje_list)) * 1000
        mean_loss  = float(np.mean(epoch_losses))
        mean_extra = float(np.mean(epoch_extra))

        ex  = f'  extra={mean_extra:.4f}' if mean_extra > 0 else ''
        lrs = '/'.join(f'{pg["lr"]:.2e}' for pg in optimizer.param_groups)
        msg = (f'Epoch {epoch:3d}  loss={mean_loss:.4f}{ex}'
               f'  eval_MJE={eval_mje:.2f}mm  lr={lrs}')
        log(msg)

        # ── save ──────────────────────────────────────────────────────────────
        ckpt_data = {
            'epoch':        epoch,
            'network':      model.state_dict(),
            'optimizer':    optimizer.state_dict(),
            'lr_scheduler': scheduler.state_dict(),
            'ablation':     args.ablation,
            'eval_mje_mm':  eval_mje,
        }
        torch.save(ckpt_data,
                   os.path.join(cfg.model_dir,
                                f'snapshot_{epoch}_{len(train_loader)-1}.pth.tar'))

        if eval_mje < best_mje:
            best_mje  = eval_mje
            best_path = os.path.join(cfg.model_dir, 'best.pth.tar')
            torch.save(ckpt_data, best_path)
            log(f'  * New best: {eval_mje:.2f} mm  -> {best_path}')

    log('-' * 60)
    log(f'Best eval MJE : {best_mje:.2f} mm')
    log('Done.')


if __name__ == '__main__':
    main()