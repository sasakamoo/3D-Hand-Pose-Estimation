# ------------------------------------------------------------------------------
# Config for HOISDF hand-only on FreiHAND
# ------------------------------------------------------------------------------

import os
import os.path as osp
import random
import sys

import numpy as np
import torch


def fix_seeds(s):
    np.random.seed(s); torch.manual_seed(s)
    torch.cuda.manual_seed_all(s); random.seed(s)
    torch.backends.cudnn.benchmark = False


def make_folder(p):
    os.makedirs(p, exist_ok=True)


def add_pypath(p):
    if p not in sys.path:
        sys.path.insert(0, p)


class Config:

    # ── dataset ───────────────────────────────────────────────────────────────
    dataset = 'freihand'

    # Set this to your FreiHAND root (contains training_K.json etc.)
    freihand_data_dir = None

    output_dir = 'outputs'

    # ── batch / workers ───────────────────────────────────────────────────────
    train_batch_size = 22
    eval_batch_size  = 22
    num_thread       = 8

    # ── SDF sampling ──────────────────────────────────────────────────────────
    num_samp_hand      = 600
    num_samp_obj       = 0       # no object branch
    points_filter_dist = 0.05
    random_ratio       = [0.3, 0.7]
    random_move_dist   = [0.03, 0.05, 0.07]

    hand_sdf_scale = 3.1
    obj_sdf_scale  = 3.1
    hand_cls_dist  = 0.04
    obj_cls_dist   = 0.05

    ClampingDistance = 0.15
    bins_n           = 64
    PointFeatSize    = 33
    ClassifierBranch = False

    # ── image / heatmap sizes ─────────────────────────────────────────────────
    input_img_shape = (256, 256)
    output_hm_shape = (128, 128, 128)
    sigma           = 2.5 / 2

    # ── model architecture ────────────────────────────────────────────────────
    resnet_type       = 50
    use_big_decoder   = True
    mutliscale_layers = ['stride2', 'stride4', 'stride8', 'stride16', 'stride32']

    def calc_mutliscale_dim(self, use_big_decoder_l, resnet_type_l):
        if use_big_decoder_l:
            self.mutliscale_dim = 128 + 256 + 512 + 1024 + 2048
        else:
            self.mutliscale_dim = 32 + 64 + 128 + 256 + 512

    # ── transformer ───────────────────────────────────────────────────────────
    hidden_dim      = 256
    dropout         = 0.1
    nheads          = 4
    dim_feedforward = 1024
    enc_layers      = 6
    dec_layers      = 4
    pre_norm        = False

    # MANO queries: 15 pose + 1 shape + 1 extra
    mano_num_queries       = 15 + 1 + 1
    mano_shape_indx        = 16
    use_inverse_kinematics = False

    # ── training ──────────────────────────────────────────────────────────────
    end_epoch            = 70
    point_sampling_epoch = 40
    lr                   = 1e-4
    lr_decay_gamma       = 0.7
    lr_drop              = 9

    # ── loss weights ──────────────────────────────────────────────────────────
    sdf_hand_weight  = 10      # was 50 — reduced to rebalance SDF vs MANO gradient share
    hm_weight        = 100 / 100000
    joint_weight     = 5 / 10  # was 1/10 — boosted to increase voting head signal
    cls_weight       = 1 / 1

    lambda_verts3d   = 1e4
    lambda_joints3d  = 1e4
    lambda_manopose  = 10
    lambda_manoshape = 0.1

    # ── directories ───────────────────────────────────────────────────────────
    cur_dir      = osp.dirname(os.path.abspath(__file__))
    root_dir     = osp.dirname(cur_dir)
    data_dir     = osp.join(root_dir, 'data')
    base_log_dir = osp.join(output_dir, 'log')

    # ── GPU ───────────────────────────────────────────────────────────────────
    gpu_ids        = '0'
    num_gpus       = 1
    continue_train = False

    def set_args(self, gpu_ids, model_dir_name, continue_train=False):
        self.gpu_ids        = gpu_ids
        self.num_gpus       = len(gpu_ids.split(','))
        self.continue_train = continue_train
        self.model_dir_name = model_dir_name
        self.setup_out_dirs(model_dir_name)
        os.environ['CUDA_VISIBLE_DEVICES'] = self.gpu_ids
        print(f'>>> Using GPU: {self.gpu_ids}')

    def setup_out_dirs(self, name):
        self.log_dir         = osp.join(self.output_dir, 'log',         name)
        self.model_dir       = osp.join(self.output_dir, 'model_dump',  name)
        self.tensorboard_dir = osp.join(self.output_dir, 'tensorboard', name)

    def create_run_dirs(self):
        for d in [self.log_dir, self.model_dir, self.tensorboard_dir]:
            make_folder(d)

    def create_log_dir(self):
        make_folder(self.log_dir)


cfg = Config()

sys.path.insert(0, osp.join(cfg.root_dir, 'common'))
add_pypath(osp.join(cfg.data_dir))
make_folder(cfg.base_log_dir)