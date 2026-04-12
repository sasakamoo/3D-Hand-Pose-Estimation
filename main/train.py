# ------------------------------------------------------------------------------
# train.py — hand-only HOISDF on FreiHAND
# ------------------------------------------------------------------------------

import os, sys, glob, math, argparse
file_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(file_dir, '..'))

import torch
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DataParallel
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from main.config import cfg
from main.model import get_model
from common.logger import colorlogger
from common.timer import Timer
from common.metrics import eval_hand_joint


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--gpu',            type=str, default='0', dest='gpu_ids')
    p.add_argument('--continue',       action='store_true',   dest='continue_train')
    p.add_argument('--run_dir_name',   type=str, default='freihand_hand')
    p.add_argument('--end_epoch',      type=int, default=70)
    p.add_argument('--point_sampling_epoch', type=int, default=40)
    p.add_argument('--lr_drop',        type=int, default=9)
    p.add_argument('--freihand_dir',   type=str, required=True,
                   help='Path to FreiHAND root (contains training_K.json)')
    return p.parse_args()


def get_optimizer(model):
    optimizer    = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, cfg.lr_drop, gamma=cfg.lr_decay_gamma)
    return optimizer, lr_scheduler


def save_model(state, epoch, itr, model_dir, logger):
    path = os.path.join(model_dir, f'snapshot_{epoch}_{itr}.pth.tar')
    torch.save(state, path)
    logger.info(f'Saved checkpoint to {path}')


def load_model(model_dir, model, optimizer, lr_scheduler, logger):
    files  = glob.glob(os.path.join(model_dir, '*.pth.tar'))
    epochs = [int(f.split('snapshot_')[1].split('_')[0])
              for f in files if 'snapshot' in f]
    cur_epoch = max(epochs)
    itrs = [int(f.split(f'snapshot_{cur_epoch}_')[1].split('.')[0])
            for f in files if f'snapshot_{cur_epoch}_' in f]
    path = os.path.join(model_dir, f'snapshot_{cur_epoch}_{max(itrs)}.pth.tar')
    logger.info(f'Resuming from {path}')
    ckpt = torch.load(path)
    model.load_state_dict(ckpt['network'], strict=True)
    optimizer.load_state_dict(ckpt['optimizer'])
    lr_scheduler.load_state_dict(ckpt['lr_scheduler'])
    return ckpt['epoch'] + 1


def main():
    args = parse_args()
    cfg.freihand_data_dir = args.freihand_dir
    cfg.set_args(args.gpu_ids, args.run_dir_name, args.continue_train)
    cfg.create_run_dirs()
    cfg.end_epoch            = args.end_epoch
    cfg.point_sampling_epoch = args.point_sampling_epoch
    cfg.lr_drop              = args.lr_drop
    cfg.calc_mutliscale_dim(cfg.use_big_decoder, cfg.resnet_type)
    cudnn.benchmark = True

    logger = colorlogger(cfg.log_dir, log_name='train_logs.txt')
    writer = SummaryWriter(cfg.tensorboard_dir)

    # ── datasets ──────────────────────────────────────────────────────────
    sys.path.insert(0, os.path.join(os.path.dirname(file_dir), 'data'))
    from freihand import Dataset as FreiHandDataset

    logger.info('Creating FreiHAND train dataset...')
    train_ds = FreiHandDataset('train')
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.num_gpus * cfg.train_batch_size,
        shuffle=True,
        num_workers=cfg.num_thread,
        pin_memory=True,
        drop_last=True,
    )
    itr_per_epoch = math.ceil(len(train_ds) / cfg.num_gpus / cfg.train_batch_size)

    logger.info('Creating FreiHAND eval dataset...')
    eval_ds = FreiHandDataset('evaluation')
    eval_loader = DataLoader(
        eval_ds,
        batch_size=cfg.num_gpus * cfg.eval_batch_size,
        shuffle=False,
        num_workers=cfg.num_thread,
        pin_memory=True,
    )

    # ── model ─────────────────────────────────────────────────────────────
    logger.info('Building model...')
    model = get_model('train').cuda()
    model = DataParallel(model)
    model.train()

    optimizer, lr_scheduler = get_optimizer(model)
    start_epoch = 0
    if args.continue_train:
        start_epoch = load_model(
            cfg.model_dir, model, optimizer, lr_scheduler, logger)

    # ── training loop ─────────────────────────────────────────────────────
    for epoch in range(start_epoch, cfg.end_epoch):
        for pg in optimizer.param_groups:
            pg['lr'] = max(lr_scheduler.get_last_lr()[-1], 1e-5)

        for itr, (inputs, targets, meta_info) in enumerate(train_loader):
            batch_ratio = itr / itr_per_epoch
            optimizer.zero_grad()

            out_dict = model(inputs, targets, meta_info, 'train', epoch, batch_ratio)

            losses = {k: v.mean() for k, v in out_dict.items() if '_out' not in k}

            losses['sdfhand_loss']      *= cfg.sdf_hand_weight
            losses['joint_heatmap']     *= cfg.hm_weight
            losses['loss_joint_3d']     *= cfg.joint_weight
            losses['loss_joint_cls']    *= cfg.cls_weight
            losses['loss_all_joint_3d'] *= cfg.joint_weight
            # Rebalance: MANO losses had no external weight, giving them only
            # ~7% of total gradient vs SDF's 91%. Add external multipliers so
            # MANO and SDF each contribute ~40-50% of the learning signal.
            losses['mano_mesh_loss']    *= 3.0
            losses['mano_joint_loss']   *= 3.0
            losses['pose_param_loss']   *= 3.0
            losses['shape_param_loss']  *= 3.0

            total = sum(losses.values())
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if itr % 100 == 0:
                for k, v in losses.items():
                    writer.add_scalar(f'train/{k}', v.item(),
                                      epoch * itr_per_epoch + itr)
                writer.add_scalar('train/total', total.item(),
                                  epoch * itr_per_epoch + itr)

            logger.info(
                f'Ep {epoch}/{cfg.end_epoch} itr {itr}/{itr_per_epoch}  '
                f'lr={optimizer.param_groups[0]["lr"]:.2e}  '
                f'total={total.item():.4f}  '
                + '  '.join(f'{k}={v.item():.4f}' for k, v in losses.items())
            )

        lr_scheduler.step()

        # ── checkpoint + quick eval ───────────────────────────────────────
        save_gap = 1 if epoch >= cfg.point_sampling_epoch else 5
        if epoch % save_gap == 0:
            save_model({'epoch': epoch,
                        'network': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict()},
                       epoch, itr, cfg.model_dir, logger)

            model.eval()
            mje_total, n_total = 0.0, 0
            with torch.no_grad():
                for inputs_e, targets_e, meta_e in eval_loader:
                    od = model(inputs_e, targets_e, meta_e, 'eval', epoch)
                    hj = od['hand_joints_out']                    # (B, 20, 3)
                    hj_full = torch.cat([torch.zeros_like(hj[:, :1]), hj], 1)
                    gt_j = targets_e['joint_cam_no_trans'] / 1000  # mm → m
                    mje, _ = eval_hand_joint(hj_full, gt_j)
                    B = hj.shape[0]
                    mje_total += mje * B * 1000   # metres → mm  (was *100 = cm, wrong)
                    n_total   += B
            avg_mje = mje_total / max(n_total, 1)
            logger.info(f'Epoch {epoch}  eval MJE = {avg_mje:.2f} mm')
            writer.add_scalar('eval/MJE_mm', avg_mje, epoch)
            model.train()

    writer.close()
    logger.info('Training complete.')


if __name__ == '__main__':
    main()