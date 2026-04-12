# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Modified from HOISDF — hand-only version for FreiHAND.
# All object SDF decoder, object transformer, object segmentation,
# object rotation/translation prediction removed.
# ------------------------------------------------------------------------------

import random

import torch
import torch.nn as nn

from common.nets.layer import MLP
from common.nets.loss import JointHeatmapLoss, JointvoteLoss, SepSDFLoss, ManoLoss
from common.nets.module import BackboneNet, DecoderNet, DecoderNet_big
from common.nets.sdf_net import SDFDecoder
from common.nets.transformer import Transformer
from common.nets.mano_head import ManoHead
from common.utils.misc import get_mano_tgt_mask, get_mano_memory_mask
from common.utils.sdf_utils import get_nerf_embedder
from manopth.manopth.manolayer import ManoLayer
from main.config import cfg


class Model(nn.Module):
    def __init__(self, backbone_net, decoder_net, hand_sdf_decoder,
                 hand_transformer, mano_layer):
        super(Model, self).__init__()

        self.backbone_net     = backbone_net
        self.decoder_net      = decoder_net
        self.hand_sdf_decoder = hand_sdf_decoder
        self.hand_transformer = hand_transformer

        self.hand_sigmoid_beta = nn.Parameter(0.1 * torch.ones(1))

        output_dim = cfg.hidden_dim - cfg.PointFeatSize

        self.norm1 = nn.LayerNorm(cfg.mutliscale_dim)
        self.linear_transformerin = MLP(
            input_dim=cfg.mutliscale_dim,
            hidden_dim=[1024, 512, 256],
            output_dim=output_dim,
            num_layers=4,
            is_activation_last=True,
        )
        self.linear_sdfin = MLP(
            input_dim=cfg.mutliscale_dim,
            hidden_dim=[512],
            output_dim=int(cfg.hidden_dim),
            num_layers=2,
            is_activation_last=True,
        )

        coord_change_mat = torch.tensor(
            [[1., 0., 0.], [0., -1., 0.], [0., 0., -1.]], dtype=torch.float32)

        self.mano_query_embed = nn.Embedding(cfg.mano_num_queries, cfg.hidden_dim)
        self.mano_head        = ManoHead(mano_layer, coord_change_mat=coord_change_mat)

        self.linear_pose      = MLP(cfg.hidden_dim, cfg.hidden_dim, 6,      3)
        self.linear_shape     = MLP(cfg.hidden_dim, cfg.hidden_dim, 10,     3)
        self.linear_handvote  = MLP(cfg.hidden_dim, cfg.hidden_dim, 20 * 3, 4)
        self.linear_handcls   = MLP(cfg.hidden_dim, cfg.hidden_dim, 20,     3)

        self.joint_heatmap_loss = JointHeatmapLoss()
        self.hand_seg_loss      = torch.nn.BCELoss(reduction='none')
        self.joints_vote_loss   = JointvoteLoss()
        self.mano_loss          = ManoLoss(
            lambda_verts3d=cfg.lambda_verts3d,
            lambda_joints3d=cfg.lambda_joints3d,
            lambda_manopose=cfg.lambda_manopose,
            lambda_manoshape=cfg.lambda_manoshape,
        )
        self.sdf_loss = SepSDFLoss()

        self.freeze_stages()

    def freeze_stages(self):
        for name, param in self.backbone_net.named_parameters():
            if 'bn' in name:
                param.requires_grad = False

    def sdf_activation(self, inp, beta):
        beta.data.copy_(max(torch.zeros_like(beta.data) + 2e-3, beta.data))
        return torch.sigmoid(inp / beta) / beta

    def render_gaussian_heatmap(self, joint_coord):
        x  = torch.arange(cfg.output_hm_shape[2])
        y  = torch.arange(cfg.output_hm_shape[1])
        yy, xx = torch.meshgrid(y, x)
        xx = xx[None, None].cuda().float()
        yy = yy[None, None].cuda().float()
        x  = joint_coord[:, :, 0, None, None]
        y  = joint_coord[:, :, 1, None, None]
        hm = torch.exp(-(((xx-x)/cfg.sigma)**2)/2 - (((yy-y)/cfg.sigma)**2)/2)
        return torch.sum(hm, 1) * 255

    def _sample_feats(self, feature_pyramid, sdf_points, center_joint,
                      cam_intr, sdf_scale):
        """Project sdf_points to 2D, bilinearly sample multiscale features."""
        cam_pts = (sdf_points / sdf_scale) + center_joint[:, None, :]
        sdf_2D  = torch.bmm(cam_pts, cam_intr.transpose(1, 2))
        sdf_2D  = sdf_2D[:, :, :2] / sdf_2D[:, :, [2]]
        norm    = (torch.tensor(
            [cfg.input_img_shape[1]-1, cfg.input_img_shape[0]-1],
            dtype=torch.float32) / 2).to(sdf_2D.device)
        grids   = (sdf_2D.detach() - norm) / norm
        grids_t = grids.unsqueeze(1).to(
            feature_pyramid[cfg.mutliscale_layers[0]].device)
        feats = []
        for layer in cfg.mutliscale_layers:
            feats.append(nn.functional.grid_sample(
                feature_pyramid[layer], grids_t,
                padding_mode='border', align_corners=True))
        feats = torch.cat(feats, dim=1).squeeze(2).permute(0, 2, 1).contiguous()
        return feats, cam_pts

    def get_input_transformer(self, feature_pyramid, sdf_points, center_joint,
                              cam_intr, sdf_scale):
        feats, cam_pts = self._sample_feats(
            feature_pyramid, sdf_points, center_joint, cam_intr, sdf_scale)
        return self.linear_transformerin(feats), cam_pts

    def sdf_forward(self, feature_pyramid, sdf_points, center_joint,
                    cam_intr, sdf_scale):
        feats, cam_pts = self._sample_feats(
            feature_pyramid, sdf_points, center_joint, cam_intr, sdf_scale)
        pts_fea   = self.linear_sdfin(feats)
        nerf_emb, _ = get_nerf_embedder((cfg.PointFeatSize - 3) // 6)
        pos_enc   = nerf_emb(sdf_points.reshape(-1, 3))
        dec_in    = torch.cat([
            pts_fea.reshape(-1, pts_fea.shape[-1]),
            pos_enc,
            sdf_points.reshape(-1, 3),
        ], dim=1).contiguous()
        pred_sdf, _ = self.hand_sdf_decoder(dec_in)
        pred_sdf = pred_sdf.reshape(sdf_points.shape[0], sdf_points.shape[1], 1)
        pred_sdf = torch.clamp(pred_sdf, -cfg.ClampingDistance, cfg.ClampingDistance)
        pos_enc  = pos_enc.reshape(sdf_points.shape[0], sdf_points.shape[1], -1)
        return pred_sdf, pos_enc

    def sdf_infer(self, feature_pyramid, center_joint, cam_intr,
                  bbox_hand, sdf_scale, num_points):
        B = center_joint.shape[0]
        voxel_size = 2.0 / (cfg.bins_n - 1)
        idx_all = torch.arange(cfg.bins_n**3, out=torch.LongTensor())
        samples = torch.zeros(cfg.bins_n**3, 3)
        samples[:, 2] = idx_all % cfg.bins_n
        samples[:, 1] = (idx_all // cfg.bins_n) % cfg.bins_n
        samples[:, 0] = (idx_all // cfg.bins_n // cfg.bins_n) % cfg.bins_n
        samples = samples * voxel_size - 1.0

        pts_out = torch.zeros(B, num_points, 3).float().cuda()
        sdf_out = torch.zeros(B, num_points, 1).float().cuda()
        enc_out = torch.zeros(B, num_points, cfg.PointFeatSize-3).float().cuda()

        nerf_emb, _ = get_nerf_embedder((cfg.PointFeatSize - 3) // 6)

        for b in range(B):
            b_cam  = (samples.clone() / sdf_scale) + center_joint[b].cpu()
            b_bbox = bbox_hand[b].cpu()
            b_2d   = torch.mm(b_cam, cam_intr[b].T.cpu())
            b_2d   = b_2d[:, :2] / b_2d[:, [2]]
            mask   = ((b_2d[:, 0] > b_bbox[0]) & (b_2d[:, 0] < b_bbox[2]) &
                      (b_2d[:, 1] > b_bbox[1]) & (b_2d[:, 1] < b_bbox[3]))
            if mask.sum() == 0:
                continue
            b_2d   = b_2d[mask].unsqueeze(0).cuda()
            b_smp  = samples[mask].cuda()

            norm   = (torch.tensor(
                [cfg.input_img_shape[1]-1, cfg.input_img_shape[0]-1],
                dtype=torch.float32) / 2).to(b_2d.device)
            grids  = (b_2d - norm) / norm
            grids_t = grids.unsqueeze(1)

            feats = []
            for layer in cfg.mutliscale_layers:
                feats.append(nn.functional.grid_sample(
                    feature_pyramid[layer][b].unsqueeze(0), grids_t,
                    padding_mode='border', align_corners=True))
            feats   = torch.cat(feats, dim=1).squeeze(2).permute(0, 2, 1)
            pts_fea = self.linear_sdfin(feats)
            pos_enc = nerf_emb(b_smp)
            dec_in  = torch.cat([pts_fea.squeeze(0), pos_enc, b_smp], 1)

            b_sdf, _ = self.hand_sdf_decoder(dec_in)
            b_sdf = b_sdf.squeeze(1)
            _, si = torch.sort(b_sdf.abs().detach())
            si = si[:num_points]
            pts_out[b] = b_smp[si].detach()
            sdf_out[b] = b_sdf[si].unsqueeze(-1)
            enc_out[b] = pos_enc[si].detach()

        sdf_out = torch.clamp(sdf_out, -cfg.ClampingDistance, cfg.ClampingDistance)
        return pts_out, sdf_out, enc_out

    def forward(self, inputs, targets, meta_info, mode,
                epoch_cnt=1e8, batch_ratio=0):
        input_img = inputs['img']
        loss, out = {}, {}

        mano_root = meta_info['mano_root']
        cam_intr  = meta_info['cam_intr']

        # ── Backbone + Decoder ────────────────────────────────────────────
        img_feat, enc_skip = self.backbone_net(input_img)
        feature_pyramid, decoder_out = self.decoder_net(img_feat, enc_skip)

        # ── SDF + heatmap losses (train only) ─────────────────────────────
        if mode == 'train':
            hand_sdf_points = inputs['hand_sdf_points']
            hand_sdf_gt     = targets['hand_sdf']

            hand_sdf_sample, _ = self.sdf_forward(
                feature_pyramid, hand_sdf_points,
                mano_root, cam_intr, cfg.hand_sdf_scale)

            hand_sdf_gt_c = torch.clamp(
                hand_sdf_gt, -cfg.ClampingDistance, cfg.ClampingDistance)

            # SepSDFLoss signature: (hand, obj, hand_gt, obj_gt)
            # Pass hand values for both since we have no object.
            loss['sdfhand_loss'], _ = self.sdf_loss(
                hand_sdf_sample, hand_sdf_sample,
                hand_sdf_gt_c, hand_sdf_gt_c)

            joint_heatmap_out = decoder_out[:, 0]
            hand_seg_out      = decoder_out[:, 1]

            out['joint_heatmap_out']  = joint_heatmap_out
            out['hand_seg_pred_out']  = hand_seg_out

            target_hm = self.render_gaussian_heatmap(targets['joint_coord'])
            loss['joint_heatmap'] = self.joint_heatmap_loss(
                joint_heatmap_out, target_hm)

        # ── Point sampling ────────────────────────────────────────────────
        bbox_hand = meta_info['bbox_hand']
        p = random.uniform(0, 1)

        if (p < 0.4 or epoch_cnt < cfg.point_sampling_epoch) and mode == 'train':
            hand_pre_points = inputs['hand_pre_points']
            dist_range = cfg.random_move_dist[
                len([a for a in cfg.random_ratio if batch_ratio > a])]
            hand_points = (hand_pre_points
                           + torch.empty_like(hand_pre_points)
                           .uniform_(-dist_range, dist_range).cuda())
            hand_sdf, hand_posenc3d = self.sdf_forward(
                feature_pyramid, hand_points,
                mano_root, cam_intr, cfg.hand_sdf_scale)
        else:
            with torch.no_grad():
                hand_points, hand_sdf, hand_posenc3d = self.sdf_infer(
                    feature_pyramid, mano_root, cam_intr,
                    bbox_hand, cfg.hand_sdf_scale, cfg.num_samp_hand)

        # ── Transformer input ─────────────────────────────────────────────
        sigma_hand = self.sdf_activation(hand_sdf.detach(), self.hand_sigmoid_beta)

        hand_fea, hand_pts_cam = self.get_input_transformer(
            feature_pyramid, hand_points, mano_root, cam_intr, cfg.hand_sdf_scale)

        hand_pts_notrans = hand_pts_cam - mano_root[:, None, :]

        hand_transformer_in = (
            torch.cat([hand_pts_notrans, hand_posenc3d, hand_fea * sigma_hand], dim=2)
            .permute(1, 0, 2).contiguous()
        )
        hand_positions = torch.zeros_like(hand_transformer_in)

        tgt_mask    = get_mano_tgt_mask().to(hand_transformer_in.device)
        memory_mask = get_mano_memory_mask().to(hand_transformer_in.device)

        # ── Hand transformer ──────────────────────────────────────────────
        hand_transformer_out, memory, hand_encoder_out, attn_wts = (
            self.hand_transformer(
                src=hand_transformer_in,
                mask=None,
                pos_embed=hand_positions,
                src_mask=None,
                query_embed=self.mano_query_embed.weight,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
            )
        )

        # ── MANO predictions ──────────────────────────────────────────────
        mano_pose6d = self.linear_pose(
            hand_transformer_out[:, :cfg.mano_shape_indx])
        mano_shape  = self.linear_shape(
            hand_transformer_out[:, cfg.mano_shape_indx])

        mano_params = targets['mano_param'] if mode == 'train' else None
        pred_mano_results, gt_mano_results = self.mano_head(
            mano_pose6d, mano_shape, mano_params=mano_params)

        out['mano_mesh_out']   = pred_mano_results['verts3d'][-1]
        out['mano_joints_out'] = pred_mano_results['joints3d'][-1]

        # ── Joint voting ──────────────────────────────────────────────────
        hand_off = self.linear_handvote(hand_encoder_out[:, :cfg.num_samp_hand])
        hand_cls = self.linear_handcls(hand_encoder_out[:, :cfg.num_samp_hand])

        if mode == 'train':
            joints3d_gt = targets['joint_cam_no_trans'][:, 1:]   # skip wrist (index 0)
        else:
            joints3d_gt = torch.zeros(mano_root.shape[0], 20, 3).cuda()

        (loss['loss_joint_3d'],
         loss['loss_joint_cls'],
         loss['loss_all_joint_3d'],
         hand_joints) = self.joints_vote_loss(
            hand_pts_notrans, hand_off, hand_cls, joints3d_gt)

        out['hand_joints_out'] = hand_joints[-1]

        # ── MANO losses ───────────────────────────────────────────────────
        if mode == 'train':
            (loss['mano_mesh_loss'],
             loss['mano_joint_loss'],
             loss['pose_param_loss'],
             loss['shape_param_loss'],
             _, _) = self.mano_loss(pred_mano_results, gt_mano_results)

        return {**loss, **out}


# ── weight init ───────────────────────────────────────────────────────────────

def init_weights(m):
    if type(m) == nn.ConvTranspose2d:
        nn.init.normal_(m.weight, std=0.001)
    elif type(m) == nn.Conv2d:
        nn.init.normal_(m.weight, std=0.001)
        nn.init.constant_(m.bias, 0)
    elif type(m) == nn.BatchNorm2d:
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)
        nn.init.constant_(m.bias, 0)


def get_model(mode):
    backbone_net = BackboneNet()
    decoder_net  = DecoderNet_big() if cfg.use_big_decoder else DecoderNet()

    hand_sdf_decoder = SDFDecoder(
        latent_size=cfg.hidden_dim,
        point_feat_size=cfg.PointFeatSize,
        use_classifier=cfg.ClassifierBranch,
    )

    hand_transformer = Transformer(
        d_model=cfg.hidden_dim,
        dropout=cfg.dropout,
        nhead=cfg.nheads,
        dim_feedforward=cfg.dim_feedforward,
        num_encoder_layers=cfg.enc_layers,
        num_decoder_layers=cfg.dec_layers,
        normalize_before=cfg.pre_norm,
        return_intermediate_dec=True,
    )

    mano_layer = ManoLayer(
        ncomps=45, center_idx=0, flat_hand_mean=True,
        side='right', mano_root='tool/mano_models', use_pca=False,
    )

    if mode == 'train':
        backbone_net.init_weights()
        decoder_net.apply(init_weights)
        hand_sdf_decoder.apply(init_weights)
        hand_transformer.apply(init_weights)

    model = Model(backbone_net, decoder_net, hand_sdf_decoder,
                  hand_transformer, mano_layer)

    print(f'Backbone     : {sum(p.numel() for p in backbone_net.parameters() if p.requires_grad):,}')
    print(f'Decoder      : {sum(p.numel() for p in decoder_net.parameters() if p.requires_grad):,}')
    print(f'HandSDF      : {sum(p.numel() for p in hand_sdf_decoder.parameters() if p.requires_grad):,}')
    print(f'Transformer  : {sum(p.numel() for p in hand_transformer.parameters() if p.requires_grad):,}')
    print(f'Total        : {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')

    return model