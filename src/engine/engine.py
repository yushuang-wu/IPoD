# Copyright 2023 Garena Online Private Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from typing import Iterable
import os
import random
import torch
import numpy as np
import time
import inspect

import util.misc as misc
import util.lr_sched as lr_sched

from src.fns import *
from tqdm import tqdm

from pytorch3d.ops import sample_farthest_points, knn_points
import util.misc as misc

def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, loss_fns, args=None, output_dir = None):
    epoch_start_time = time.time()
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if args.distributed:
        for param in model.module.parameters():
            param.requires_grad = True
    else:
        for param in model.parameters():
            param.requires_grad = True

    for data_iter_step, samples in enumerate(data_loader):
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        if args.geo == 'occ':
            prepare_data_func = prepare_data
        elif args.geo == 'udf':
            prepare_data_func = prepare_data_udf

        seen_xyz, valid_seen_xyz, query_xyz, query_rgb_gt, labels, seen_images, gt_fps_xyz, seen_xyz_hr, valid_seen_xyz_hr, gt_xyz, gt_rgb = prepare_data_func(samples, device, is_train=True, args=args)
        
        B = seen_xyz.shape[0]
        # gt_xyz = samples[1][0].to(device).reshape((B, -1, 3))
        # gt_rgb = samples[1][1].to(device).reshape((B, -1, 3))
        
        # print(gt_xyz.shape, gt_rgb.shape)

        out, label_out, loss_noise, centers_xyz = model(
            seen_images=seen_images,
            seen_xyz=seen_xyz,
            query_xyz=query_xyz,
            valid_seen_xyz=valid_seen_xyz,
            # seen_xyz_hr=seen_xyz_hr,
            # valid_seen_xyz_hr=valid_seen_xyz_hr,
            gt_xyz=gt_xyz,
            gt_rgb=gt_rgb,
            # mode='train'
        )
        
        # print(out.shape, label_out[0].shape)

        # center loss
        loss_centers = loss_fns['centers'](centers_xyz.float(), gt_fps_xyz, norm=args.cd_norm)[0]*0.03

        # geo loss
        if args.geo == 'occ':
            loss_geo = loss_fns[args.geo](out[:,:,:1].reshape((-1, 1)), labels[0].reshape((-1, 1)).float())
        elif args.geo == 'udf':
            max_dist = 0.5
            # labels_udf = torch.clamp(labels[1], max=max_dist).reshape((-1, 1)).float()
            labels_udf = label_out[0]
            pred_udf = F.relu(out[:,:,:1]).reshape((-1, 1))
            pred_udf = torch.clamp(pred_udf, max=max_dist)
            
            # print(pred_udf.shape, labels_udf.shape)

            loss_geo = loss_fns[args.geo](pred_udf, labels_udf.reshape((-1, 1)))

        # rgb loss
        # pred_rgb = out[:, :, 1:][labels[0].bool()].reshape((-1, 256))
        pred_rgb = out[:, :, 1:][(labels_udf<0.1).squeeze(-1)].reshape((-1, 256))
        if pred_rgb.size()[0]!= 0:
            # gt_rgb = torch.round(query_rgb_gt[labels[0].bool()] * 255).long().reshape((-1,))
            gt_rgb = torch.round(label_out[1][(labels_udf<0.1).squeeze(-1)] * 255).long().reshape((-1,))
            # print(query_rgb_gt.shape, query_rgb_gt)
            # print(label_out[1], label_out[1].shape)
            loss_rgb = loss_fns['rgb'](pred_rgb, gt_rgb)*0.01
        else:
            loss_rgb = 0

        loss = loss_centers + loss_geo + loss_rgb + loss_noise

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Warning: Loss is {}".format(loss_value))
            loss *= 0.0
            loss_value = 0
            
        if loss_value > 10:
            print("Warning: Loss is {}".format(loss_value))
            loss *= 0.0
            loss_value = 0
            print(loss_centers.item(), loss_geo.item(), loss_rgb.item(), loss_noise.item())
        
        if loss_centers > 10:
            print("Warning: Loss centers is {}".format(loss_centers))
            loss *= 0.0
            loss_value = 0

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    #clip_grad=args.clip_grad,
                    clip_grad=None,
                    update_grad=(data_iter_step + 1) % accum_iter == 0,
                    verbose=False)
                    #verbose=(data_iter_step + 1) % accum_iter == 0)

        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()
        metric_logger.update(loss=loss_value)
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        if (data_iter_step % args.print_every)== 0 and (data_iter_step > 0):
            print('[Epoch %02d] it=%03d/%03d, loss=%.4f, loss_geo=%.4f, loss_rgb=%.4f, loss_center=%.4f, loss_noise=%.4f' % (epoch, data_iter_step, len(data_loader), 
                                                                                                            loss_value, loss_geo, loss_rgb, loss_centers, loss_noise))

            logging_string = '[Epoch %02d] it=%03d/%03d, loss=%.4f, loss_geo=%.4f, loss_rgb=%.4f, loss_center=%.4f, loss_noise=%.4f' % (epoch, data_iter_step, len(data_loader), 
                                                                                                                loss_value, loss_geo, loss_rgb, loss_centers, loss_noise)
            
            if output_dir != None and misc.is_main_process():
                with open(os.path.join(output_dir, "log_screen.txt"), mode="a", encoding="utf-8") as f:
                    f.write(logging_string + "\n")

        if args.debug and data_iter_step == 5:
        # if data_iter_step == 100:
            break

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    print("Training epoch time:", time.time() - epoch_start_time)

    str1 = "Averaged stats: " + str(metric_logger)
    str2 = "Training epoch time:" + str(time.time() - epoch_start_time)

    if output_dir != None and misc.is_main_process():
        with open(os.path.join(output_dir, "log_screen.txt"), mode="a", encoding="utf-8") as f:
            f.write(str1 + "\n")
            f.write(str2 + "\n")
          
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def eval_one_epoch(
        model: torch.nn.Module,
        data_loader: Iterable,
        device: torch.device,
        loss_fns,
        args=None
    ):
    epoch_start_time = time.time()
    model.train(False)

    metric_logger = misc.MetricLogger(delimiter="  ")

    print('Eval len(data_loader):', len(data_loader))

    for data_iter_step, samples in enumerate(tqdm(data_loader, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')):
        seen_xyz, valid_seen_xyz, query_xyz, unseen_rgb, labels, seen_images, gt_fps_xyz, seen_xyz_hr, valid_seen_xyz_hr, _, _ = prepare_data(samples, device, is_train=False, args=args)

        with torch.no_grad():
            seen_images_hr = None

            if args.hr == 1:
                seen_images_hr = preprocess_img(seen_images.clone(), res=args.xyz_size)
                seen_xyz_hr = shrink_points_beyond_threshold(seen_xyz_hr, args.shrink_threshold)

            seen_images = preprocess_img(seen_images)
            query_xyz = shrink_points_beyond_threshold(query_xyz, args.shrink_threshold)
            seen_xyz = shrink_points_beyond_threshold(seen_xyz, args.shrink_threshold)
            

            if args.distributed:
                latent, up_grid_fea = model.module.encoder(seen_images, seen_xyz, valid_seen_xyz, up_grid_bypass=seen_images_hr)
                fea = model.module.decoderl1(latent)
            else:
                latent, up_grid_fea = model.encoder(seen_images, seen_xyz, valid_seen_xyz, up_grid_bypass=seen_images_hr)
                fea = model.decoderl1(latent)
            centers_xyz = fea['anchors_xyz']

            #visualize_centers(centers_xyz, data_iter_step)

            # center loss
            loss_centers = loss_fns['centers'](centers_xyz.float(), gt_fps_xyz, norm=args.cd_norm)[0]*0.03

        # don't forward all at once to avoid oom
        max_n_queries_fwd = 2000
        all_loss_occ, all_loss_rgb, all_preds = [], [], []
        for p_idx in range(int(np.ceil(query_xyz.shape[1] / max_n_queries_fwd))):
            p_start = p_idx     * max_n_queries_fwd
            p_end = (p_idx + 1) * max_n_queries_fwd
            cur_query_xyz = query_xyz[:, p_start:p_end]
            query_rgb_gt = unseen_rgb[:, p_start:p_end]
            query_occ_gt = labels[0][:, p_start:p_end]


            with torch.no_grad():
                if args.hr != 1:
                    seen_points = seen_xyz
                    valid_seen = valid_seen_xyz
                else:
                    seen_points = seen_xyz_hr
                    valid_seen = valid_seen_xyz_hr

                if args.distributed:
                    pred = model.module.decoderl2(cur_query_xyz, seen_points, valid_seen, fea, up_grid_fea)
                    pred = model.module.fc_out(pred)
                else:
                    pred = model.decoderl2(cur_query_xyz, seen_points, valid_seen, fea, up_grid_fea)
                    pred = model.fc_out(pred)
                
                # occupancy loss
                loss_occ = loss_fns['occ'](pred[:,:,:1].reshape((-1, 1)), query_occ_gt.reshape((-1, 1)).float())

                # rgb loss
                pred_rgb = pred[:, :, 1:][query_occ_gt.bool()].reshape((-1, 256))
                if pred_rgb.size()[0]!= 0:
                    gt_rgb = torch.round(query_rgb_gt[query_occ_gt.bool()] * 255).long().reshape((-1,))
                    loss_rgb = loss_fns['rgb'](pred_rgb, gt_rgb)*0.01
                else:
                    loss_rgb = 0

            all_loss_occ.append(loss_occ)
            all_loss_rgb.append(loss_rgb)
            all_preds.append(pred)

        loss_occ = sum(all_loss_occ) / len(all_loss_occ)
        loss_rgb = sum(all_loss_rgb) / len(all_loss_rgb)
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
            predicted_xyz = query_xyz[b_idx, predicted_idx]

            predicted_colors = pred[b_idx, predicted_idx, 1:].reshape((-1, 3, 256)).max(dim=2)[1] / 255.0

            color_metrics = {}
            col_pred_gt, col_gt_pred, col = evaluate_colors(predicted_xyz, gt_xyz[b_idx], predicted_colors, gt_rgb[b_idx], d_thres)
            color_metrics[f'ColPrGt'] = col_pred_gt
            color_metrics[f'ColGtPr'] = col_gt_pred
            color_metrics[f'Col'] = col

            metric_logger.update(**color_metrics)

            precision, recall, f1 = evaluate_points(predicted_xyz, gt_xyz[b_idx], d_thres)
            geometry_metrics[f'Pr'] = precision
            geometry_metrics[f'Rc'] = recall
            geometry_metrics[f'F1'] = f1

            prec_dist, rec_dist, chamfer_dist = evaluate_distance(predicted_xyz, gt_xyz[b_idx])

            geometry_metrics[f'Acc'] = prec_dist
            geometry_metrics[f'Com'] = rec_dist
            geometry_metrics[f'CD'] = chamfer_dist

            #if args.use_hypersim:
            #    precision, recall, f1 = evaluate_points(predicted_xyz, mesh_xyz[b_idx], d_thres)
            #    geometry_metrics[f'd{d_thres}_s{s_thres}_mesh_pr'] = precision
            #    geometry_metrics[f'd{d_thres}_s{s_thres}_mesh_rc'] = recall
            #    geometry_metrics[f'd{d_thres}_s{s_thres}_mesh_f1'] = f1

            metric_logger.update(**geometry_metrics)

        loss = loss_occ + loss_rgb + loss_centers
        #loss = loss_occ + loss_rgb
        loss_value = loss.item()

        torch.cuda.synchronize()
        metric_logger.update(loss=loss_value)

        if args.debug and data_iter_step == 5:
            break

    metric_logger.synchronize_between_processes()
    print("Validation averaged stats:", metric_logger)
    print("Val epoch time:", time.time() - epoch_start_time)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def eval_one_epoch_udf(
        model: torch.nn.Module,
        data_loader: Iterable,
        device: torch.device,
        loss_fns,
        args=None
    ):
    epoch_start_time = time.time()
    model.train(False)

    metric_logger = misc.MetricLogger(delimiter="  ")

    print('Eval len(data_loader):', len(data_loader))
    
    mm = model.module if args.distributed else model

    for data_iter_step, samples in enumerate(tqdm(data_loader, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')):
        seen_xyz, valid_seen_xyz, query_xyz, unseen_rgb, labels, seen_images, gt_fps_xyz, seen_xyz_hr, valid_seen_xyz_hr, _, _ = prepare_data_udf(samples, device, is_train=False, args=args)
        
        B, N = seen_images.shape[0], args.n_queries

        with torch.no_grad():
            seen_images_hr = None

            if args.hr == 1:
                seen_images_hr = preprocess_img(seen_images.clone(), res=args.xyz_size)
                seen_xyz_hr = shrink_points_beyond_threshold(seen_xyz_hr, args.shrink_threshold)

            seen_images = preprocess_img(seen_images)
            seen_xyz = shrink_points_beyond_threshold(seen_xyz, args.shrink_threshold)

            latent, up_grid_fea = mm.encoder(seen_images, seen_xyz, valid_seen_xyz, up_grid_bypass=seen_images_hr)
            fea = mm.decoderl1(latent)
            
            centers_xyz = fea['anchors_xyz']

            # center loss
            loss_centers = loss_fns['centers'](centers_xyz.float(), gt_fps_xyz, norm=args.cd_norm)[0]*0.03

        pred_points = np.empty((0,3))
        pred_colors = np.empty((0,3))
        max_n_queries_fwd = args.n_query_udf if not args.hr else int(args.n_query_udf * (args.xyz_size/args.xyz_size_hr)**2)
        all_loss_geo, all_loss_rgb = [], []

        for param in mm.parameters():
            param.requires_grad = False

        for p_idx in range(max_n_queries_fwd // N):
            scheduler = mm.scheduler
            gt_xyz = samples[1][0].to(device).reshape((B, -1, 3))
            gt_rgb = samples[1][1].to(device).reshape((B, -1, 3))
            
            # Set timesteps
            accepts_offset = "offset" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
            extra_set_kwargs = {"offset": 1} if accepts_offset else {}
            scheduler.set_timesteps(mm.num_inference_steps, **extra_set_kwargs)
            
            accepts_eta = "eta" in set(inspect.signature(scheduler.step).parameters.keys())
            extra_step_kwargs = {"eta": eta} if accepts_eta else {}
            
            cur_query_xyz = torch.randn(B, N, 3, device=device)
            x_t = cur_query_xyz
            # x_t = torch.clamp(x_t, min=-3., max=3.)
            # x_t = shrink_points_beyond_threshold(x_t, args.shrink_threshold)
            
            knn = knn_points(x_t, gt_xyz, K=1)
            knn_dists, knn_idx = knn[0]**0.5, knn[1]
            query_udf_gt = torch.clamp(knn_dists, min=0, max=0.5)
            query_occ_gt = (query_udf_gt < args.udf_threshold).squeeze(-1)
            # query_rgb_gt = mm.index_select(gt_rgb, knn_idx)
            labels_01 = (knn_dists < args.train_dist_threshold).squeeze(-1)
            query_rgb_gt = torch.zeros_like(x_t)
            query_rgb_gt[labels_01] = torch.gather(gt_rgb, 1, knn_idx.repeat(1, 1, 3))[labels_01]

            with torch.no_grad():
                if args.hr != 1:
                    seen_points = seen_xyz
                    valid_seen = valid_seen_xyz
                else:
                    seen_points = seen_xyz_hr
                    valid_seen = valid_seen_xyz_hr
                
                progress_bar = tqdm(scheduler.timesteps.to(device), desc=None, disable=1)
                for idx_t, timestep in enumerate(progress_bar):

                    pred2 = mm.decoderl2(x_t, seen_points, valid_seen, fea, up_grid_fea)
                    pred2 = mm.fc_out2(pred2)
                    
                    pred_udf = F.relu(pred2[:,:,:1])
                    pred_udf = torch.clamp(pred_udf, max=0.5)
                    self_cond = pred_udf
                    
                    pred3 = mm.decoderl3(x_t, seen_points, valid_seen, fea, up_grid_fea, timestep.reshape(1).expand(B), self_cond)
                    pred3 = mm.fc_out3(pred3)
                    pred_noise = pred3.permute(0, 2, 1)
                        
                    x_t = scheduler.step(pred_noise, timestep, x_t, **extra_step_kwargs).prev_sample
                    # x_t = torch.clamp(x_t, min=-3., max=3.)
                    # x_t = shrink_points_beyond_threshold(x_t, args.shrink_threshold)
                
                # x_0 predict udf #
                pred2 = mm.decoderl2(x_t, seen_points, valid_seen, fea, up_grid_fea)
                pred2 = mm.fc_out2(pred2)
                    
                # geometry loss
                max_dist = 0.5
                labels_udf = torch.clamp(query_udf_gt, max=max_dist).reshape((-1, 1)).float()
                pred_udf = F.relu(pred2[:, :, :1]).reshape((-1, 1)) # nQ, 1
                pred_udf = torch.clamp(pred_udf, max=max_dist) 

                loss_geo = loss_fns[args.geo](pred_udf, labels_udf)

                # rgb loss
                pred_rgb = pred2[:, :, 1:][query_occ_gt.bool()].reshape((-1, 256))
                if pred_rgb.size()[0]!= 0:
                    gt_rgb = torch.round(query_rgb_gt[query_occ_gt.bool()] * 255).long().reshape((-1,))
                    loss_rgb = loss_fns['rgb'](pred_rgb, gt_rgb)*0.01
                else:
                    loss_rgb = 0
                    
            gt_xyz = samples[1][0].to(device).reshape((B, -1, 3))
            gt_rgb = samples[1][1].to(device).reshape((B, -1, 3))
            
            all_loss_geo.append(loss_geo)
            all_loss_rgb.append(loss_rgb)

            # Move points
            # Candidate points
            t = args.udf_threshold
            pos = (pred_udf < t).squeeze(-1) # (nQ, )
            points = x_t.squeeze(0) # (nQ, 3)
            points = points[pos].unsqueeze(0) # (1, n, 3)

            if torch.sum(pos) > 0:
                points = move_points(mm, points, seen_points, valid_seen, fea, up_grid_fea, args, n_iter=args.udf_n_iter)

                # predict final color
                with torch.no_grad():
                    pred = mm.decoderl2(points, seen_points, valid_seen, fea, up_grid_fea)
                    pred = mm.fc_out2(pred)

                cur_color_out = pred[:,:,1:].reshape((-1, 3, 256)).max(dim=2)[1] / 255.0
                cur_color_out = cur_color_out.detach().squeeze(0).cpu().numpy()
                if len(cur_color_out.shape) == 1:
                    cur_color_out = cur_color_out[None,...]
                pts = points.detach().squeeze(0).cpu().numpy()
                pred_points = np.append(pred_points, pts, axis = 0)
                pred_colors = np.append(pred_colors, cur_color_out, axis = 0)
        
        loss_geo = sum(all_loss_geo) / len(all_loss_geo)
        loss_rgb = sum(all_loss_rgb) / len(all_loss_rgb)

        B = 1
        predicted_xyz = torch.from_numpy(pred_points).to('cuda')
        predicted_colors = torch.from_numpy(pred_colors).to('cuda')

        d_thres = args.eval_dist_threshold

        b_idx = 0 # batch = 1

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

        loss = loss_geo + loss_rgb + loss_centers
        loss_value = loss.item()

        torch.cuda.synchronize()
        metric_logger.update(loss=loss_value)

        # if args.debug and data_iter_step == 5:
        if data_iter_step == 5:
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

    # Hres
    seen_xyz_hr = None
    valid_seen_xyz_hr = None
    if args.hr == 1:
        seen_xyz_hr = samples[0][2].to(device)
        valid_seen_xyz_hr = torch.isfinite(seen_xyz_hr.sum(axis=-1))
        seen_xyz_hr[~valid_seen_xyz_hr] = -100

    # Gt
    gt_xyz, gt_rgb = samples[1][0].to(device).reshape(B, -1, 3), samples[1][1].to(device).reshape(B, -1, 3)
    gt_xyz = shrink_points_beyond_threshold(gt_xyz, args.shrink_threshold)
    gt_fps_xyz, _ = sample_farthest_points(gt_xyz, K=args.n_groups) # B, M, 3

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
        seen_xyz, unseen_xyz, gt_fps_xyz, seen_xyz_hr = aug_xyz_all(seen_xyz, unseen_xyz, gt_fps_xyz, args, is_train=is_train, seen_xyz_hr=seen_xyz_hr)

        # Random Flip
        if random.random() < 0.5:
            seen_xyz[..., 0] *= -1
            unseen_xyz[..., 0] *= -1
            gt_fps_xyz[...,0] *= -1
            seen_xyz = torch.flip(seen_xyz, [2])
            valid_seen_xyz = torch.flip(valid_seen_xyz, [2])
            seen_rgb = torch.flip(seen_rgb, [3])

            if seen_xyz_hr != None:
                seen_xyz_hr[..., 0] *= -1
                seen_xyz_hr = torch.flip(seen_xyz_hr, [2])
                valid_seen_xyz_hr = torch.flip(valid_seen_xyz_hr, [2])
        
    return seen_xyz, valid_seen_xyz, unseen_xyz, unseen_rgb, labels, seen_rgb, gt_fps_xyz, seen_xyz_hr, valid_seen_xyz_hr

def prepare_data_udf(samples, device, is_train, args, is_viz=False):
    # Seen
    seen_xyz, seen_rgb = samples[0][0].to(device), samples[0][1].to(device)
    valid_seen_xyz = torch.isfinite(seen_xyz.sum(axis=-1))
    seen_xyz[~valid_seen_xyz] = -100
    B = seen_xyz.shape[0]

    # Hres
    seen_xyz_hr = None
    valid_seen_xyz_hr = None
    if args.hr == 1:
        seen_xyz_hr = samples[0][2].to(device)
        valid_seen_xyz_hr = torch.isfinite(seen_xyz_hr.sum(axis=-1))
        seen_xyz_hr[~valid_seen_xyz_hr] = -100

    # Gt
    gt_xyz, gt_rgb = samples[1][0].to(device).reshape(B, -1, 3), samples[1][1].to(device).reshape(B, -1, 3)
    gt_xyz = shrink_points_beyond_threshold(gt_xyz, args.shrink_threshold)

    if is_train:
        seen_xyz, gt_xyz, seen_xyz_hr = aug_xyz_udf_train(seen_xyz, gt_xyz, args, is_train=is_train, seen_xyz_hr = seen_xyz_hr)

        if random.random() < 0.5:
            seen_xyz[..., 0] *= -1
            gt_xyz[...,0] *= -1
            seen_xyz = torch.flip(seen_xyz, [2])
            valid_seen_xyz = torch.flip(valid_seen_xyz, [2])
            seen_rgb = torch.flip(seen_rgb, [3])

            if seen_xyz_hr != None:
                seen_xyz_hr[..., 0] *= -1
                seen_xyz_hr = torch.flip(seen_xyz_hr, [2])
                valid_seen_xyz_hr = torch.flip(valid_seen_xyz_hr, [2])
    
    gt_fps_xyz, _ = sample_farthest_points(gt_xyz, K=args.n_groups) # B, M, 3
    sampling_func = construct_uniform_semisphere_udf if args.use_hypersim else construct_uniform_grid_udf
    
    # print(gt_xyz.shape, gt_rgb.shape)

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
        
    return seen_xyz, valid_seen_xyz, unseen_xyz, unseen_rgb, labels, seen_rgb, gt_fps_xyz, seen_xyz_hr, valid_seen_xyz_hr, gt_xyz, gt_rgb
