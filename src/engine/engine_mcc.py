# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------
import math
from typing import Iterable
import os
import random
import torch
import numpy as np
import time

import util.misc as misc
import util.lr_sched as lr_sched

from src.fns import *
from tqdm import tqdm

def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    args=None):
    epoch_start_time = time.time()
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    for data_iter_step, samples in enumerate(data_loader):
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
        seen_xyz, valid_seen_xyz, unseen_xyz, unseen_rgb, labels, seen_images = prepare_data(samples, device, is_train=True, args=args)

        #with torch.cuda.amp.autocast():
        loss, _ = model(
            seen_images=seen_images,
            seen_xyz=seen_xyz,
            unseen_xyz=unseen_xyz,
            unseen_rgb=unseen_rgb,
            unseen_occupy=labels,
            valid_seen_xyz=valid_seen_xyz,
        )

        loss_occ = loss[0]
        loss_rgb = loss[1]
        loss = loss_occ + loss_rgb

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Warning: Loss is {}".format(loss_value))
            loss *= 0.0
            loss_value = 100.0

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    clip_grad=args.clip_grad,
                    update_grad=(data_iter_step + 1) % accum_iter == 0,
                    verbose=False)

        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()
        metric_logger.update(loss=loss_value)
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        #if (data_iter_step % args.print_every)== 0 and (data_iter_step > 0):
        #    print('[Epoch %02d] it=%03d/%03d, loss=%.4f' % (epoch, data_iter_step, len(data_loader), loss))

        if (data_iter_step % args.print_every)== 0 and (data_iter_step > 0):
            print('[Epoch %02d] it=%03d/%03d, loss=%.4f, loss_occ=%.4f, loss_rgb=%.4f' % (epoch, data_iter_step, len(data_loader), 
                                                                                                            loss_value, loss_occ, loss_rgb))

        if args.debug and data_iter_step == 5:
            break

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    print("Training epoch time:", time.time() - epoch_start_time)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def eval_one_epoch(
        model: torch.nn.Module,
        data_loader: Iterable,
        device: torch.device,
        args=None
    ):
    epoch_start_time = time.time()
    model.train(False)

    metric_logger = misc.MetricLogger(delimiter="  ")

    print('Eval len(data_loader):', len(data_loader))

    for data_iter_step, samples in enumerate(tqdm(data_loader, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')):
        seen_xyz, valid_seen_xyz, unseen_xyz, unseen_rgb, labels, seen_images = prepare_data(samples, device, is_train=False, args=args)

        # don't forward all at once to avoid oom
        max_n_queries_fwd = 2000
        all_loss, all_preds = [], []
        for p_idx in range(int(np.ceil(unseen_xyz.shape[1] / max_n_queries_fwd))):
            p_start = p_idx     * max_n_queries_fwd
            p_end = (p_idx + 1) * max_n_queries_fwd
            cur_unseen_xyz = unseen_xyz[:, p_start:p_end]
            cur_unseen_rgb = unseen_rgb[:, p_start:p_end]
            cur_labels = labels[:, p_start:p_end]

            with torch.no_grad():
                loss, pred = model(
                    seen_images=seen_images,
                    seen_xyz=seen_xyz,
                    unseen_xyz=cur_unseen_xyz,
                    unseen_rgb=cur_unseen_rgb,
                    unseen_occupy=cur_labels,
                    valid_seen_xyz=valid_seen_xyz,
                )
                loss = loss[0] + loss[1]
            all_loss.append(loss)
            all_preds.append(pred)

        loss = sum(all_loss) / len(all_loss)
        pred = torch.cat(all_preds, dim=1)

        B = pred.shape[0]

        gt_xyz = samples[1][0].to(device).reshape((B, -1, 3))
        gt_rgb = samples[1][1].to(device).reshape(B, -1, 3)
        #if args.use_hypersim:
        #    mesh_xyz = samples[2].to(device).reshape((B, -1, 3))

        s_thres = args.eval_score_threshold
        d_thres = args.eval_dist_threshold

        for b_idx in range(B):
            geometry_metrics = {}
            predicted_idx = torch.nn.Sigmoid()(pred[b_idx, :, 0]) > s_thres
            predicted_xyz = unseen_xyz[b_idx, predicted_idx]

            predicted_colors = pred[b_idx, predicted_idx, 1:].reshape((-1, 3, 256)).max(dim=2)[1] / 255.0

            #if args.use_hypersim:
            #    precision, recall, f1 = evaluate_points(predicted_xyz, mesh_xyz[b_idx], d_thres)
            #    geometry_metrics[f'd{d_thres}_s{s_thres}_mesh_pr'] = precision
            #    geometry_metrics[f'd{d_thres}_s{s_thres}_mesh_rc'] = recall
            #    geometry_metrics[f'd{d_thres}_s{s_thres}_mesh_f1'] = f1

            color_metrics = {}
            col_pred_gt, col_gt_pred, col = evaluate_colors(predicted_xyz, gt_xyz[b_idx], predicted_colors, gt_rgb[b_idx], d_thres)
            color_metrics[f'ColPrGt'] = col_pred_gt
            color_metrics[f'ColGtPr'] = col_gt_pred
            color_metrics[f'Col'] = col

            metric_logger.update(**color_metrics)

            geometry_metrics = {}

            precision, recall, f1 = evaluate_points(predicted_xyz, gt_xyz[b_idx], d_thres)
            geometry_metrics[f'Pr'] = precision
            geometry_metrics[f'Rc'] = recall
            geometry_metrics[f'F1'] = f1

            prec_dist, rec_dist, chamfer_dist = evaluate_distance(predicted_xyz.float(), gt_xyz[b_idx].float())

            geometry_metrics[f'Acc'] = prec_dist
            geometry_metrics[f'Com'] = rec_dist
            geometry_metrics[f'CD'] = chamfer_dist

            metric_logger.update(**geometry_metrics)

        loss_value = loss.item()

        torch.cuda.synchronize()
        metric_logger.update(loss=loss_value)

        if args.debug and data_iter_step == 5:
            break

    metric_logger.synchronize_between_processes()
    print("Validation averaged stats:", metric_logger)
    print("Val epoch time:", time.time() - epoch_start_time)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def prepare_data(samples, device, is_train, args, is_viz=False):
    # Seen
    seen_xyz, seen_rgb = samples[0][0].to(device), samples[0][1].to(device)
    valid_seen_xyz = torch.isfinite(seen_xyz.sum(axis=-1))
    seen_xyz[~valid_seen_xyz] = -100
    B = seen_xyz.shape[0]
    # Gt
    gt_xyz, gt_rgb = samples[1][0].to(device).reshape(B, -1, 3), samples[1][1].to(device).reshape(B, -1, 3)

    sampling_func = construct_uniform_semisphere if args.use_hypersim else construct_uniform_grid
    unseen_xyz, unseen_rgb, labels = sampling_func(
        gt_xyz, gt_rgb,
        args.semisphere_size if args.use_hypersim else args.co3d_world_size,
        args.n_queries,
        args.train_dist_threshold,
        is_train,
        args.viz_granularity if is_viz else args.eval_granularity,
        is_viz,
        args
    )

    if is_train:
        seen_xyz, unseen_xyz = aug_xyz(seen_xyz, unseen_xyz, args, is_train=is_train)

        # Random Flip
        if random.random() < 0.5:
            seen_xyz[..., 0] *= -1
            unseen_xyz[..., 0] *= -1
            seen_xyz = torch.flip(seen_xyz, [2])
            valid_seen_xyz = torch.flip(valid_seen_xyz, [2])
            seen_rgb = torch.flip(seen_rgb, [3])

    return seen_xyz, valid_seen_xyz, unseen_xyz, unseen_rgb, labels[0], seen_rgb