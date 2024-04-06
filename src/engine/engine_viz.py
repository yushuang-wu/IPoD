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

import os
import matplotlib.pyplot as plt
import random
import torch
import numpy as np
import time
import base64
from io import BytesIO

import util.misc as misc

from pytorch3d.structures import Pointclouds
from pytorch3d.vis.plotly_vis import plot_scene

from src.engine.engine import prepare_data, prepare_data_udf
from pathlib import Path
from tqdm import tqdm
from src.fns import *

import pandas as pd
from pyntcloud import PyntCloud

def run_viz(model, data_loader, device, args, epoch):
    epoch_start_time = time.time()
    model.eval()

    print('Visualization data_loader length:', len(data_loader))
    dataset = data_loader.dataset
    for sample_idx, samples in enumerate(tqdm(data_loader, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')):
        if sample_idx >= args.max_n_viz_obj:
            break
        seen_xyz, valid_seen_xyz, query_xyz, unseen_rgb, labels, seen_images, gt_fps_xyz, seen_xyz_hr, valid_seen_xyz_hr = prepare_data(samples, device, is_train=False, is_viz=True, args=args)

        seen_images_no_preprocess = seen_images.clone()

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
            #centers_xyz = centers_xyz*0.001

        pred_occupy = []
        pred_colors = []

        # don't forward all at once to avoid oom
        max_n_queries_fwd = 5400

        total_n_passes = int(np.ceil(query_xyz.shape[1] / max_n_queries_fwd))
        for p_idx in range(total_n_passes):
            p_start = p_idx     * max_n_queries_fwd
            p_end = (p_idx + 1) * max_n_queries_fwd
            cur_query_xyz = query_xyz[:, p_start:p_end]

            with torch.no_grad():
                if args.hr != 1:
                    seen_points = seen_xyz
                    valid_seen = valid_seen_xyz
                else:
                    seen_points = seen_xyz_hr
                    valid_seen = valid_seen_xyz_hr

                if args.distributed:
                    pred = model.module.decoderl2(cur_query_xyz, seen_points, valid_seen, fea, up_grid_fea, custom_centers = None)
                    pred = model.module.fc_out(pred)
                else:
                    pred = model.decoderl2(cur_query_xyz, seen_points, valid_seen, fea, up_grid_fea, custom_centers = None)
                    pred = model.fc_out(pred)

            cur_occupy_out = pred[..., 0]
            cur_color_out = pred[..., 1:].reshape((-1, 3, 256)).max(dim=2)[1] / 255.0
            pred_occupy.append(cur_occupy_out)
            pred_colors.append(cur_color_out)

        rank = misc.get_rank()
        out_folder = os.path.join('experiments/', f'{args.exp_name}', 'viz', 'epoch'+str(epoch).zfill(3))
        Path(out_folder).mkdir(parents= True, exist_ok=True)
        prefix = os.path.join(out_folder, dataset.dataset_split+f'_ep{epoch}_rank{rank}_i{sample_idx}')
        img = (seen_images_no_preprocess[0].permute(1, 2, 0) * 255).cpu().numpy().copy().astype(np.uint8)

        gt_xyz = samples[1][0].to(device).reshape(-1, 3)
        gt_rgb = samples[1][1].to(device).reshape(-1, 3)
        mesh_xyz = samples[2].to(device).reshape(-1, 3) if args.use_hypersim else None

        with open(prefix + '.html', 'a') as f:
            generate_html(
                img,
                seen_xyz, seen_images_no_preprocess,
                torch.cat(pred_occupy, dim=1),
                torch.cat(pred_colors, dim=0),
                query_xyz,
                f,
                gt_xyz=gt_xyz,
                gt_rgb=gt_rgb,
                mesh_xyz=mesh_xyz,
                centers = centers_xyz,
            )
    print("Visualization epoch time:", time.time() - epoch_start_time)

def generate_html(img, seen_xyz, seen_rgb, pred_occ, pred_rgb, unseen_xyz, f,
        gt_xyz=None, gt_rgb=None, mesh_xyz=None, centers = None, sampled_seen = None, score_thresholds=[0.1, 0.2, 0.3],
        pointcloud_marker_size=3,
    ):
    if img is not None:
        fig = plt.figure()
        plt.imshow(img)
        tmpfile = BytesIO()
        fig.savefig(tmpfile, format='jpg')
        encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')

        html = '<img src=\'data:image/png;base64,{}\'>'.format(encoded)
        f.write(html)
        plt.close()

    clouds = {"Output": {}}
    # Seen
    if seen_xyz is not None:
        seen_xyz = seen_xyz.reshape((-1, 3)).cpu()
        seen_rgb = torch.nn.functional.interpolate(seen_rgb, (112, 112)).permute(0, 2, 3, 1).reshape((-1, 3)).cpu()
        good_seen = seen_xyz[:, 0] != -100

        seen_pc = Pointclouds(
            points=seen_xyz[good_seen][None],
            features=seen_rgb[good_seen][None],
        )
        clouds["Output"]["seen"] = seen_pc

    # GT points
    if gt_xyz is not None:
        subset_gt = random.sample(range(gt_xyz.shape[0]), 10000)
        gt_pc = Pointclouds(
            points=gt_xyz[subset_gt][None],
            features=gt_rgb[subset_gt][None],
        )
        clouds["Output"]["GT points"] = gt_pc
    


    # GT meshes
    if mesh_xyz is not None:
        subset_mesh = random.sample(range(mesh_xyz.shape[0]), 10000)
        mesh_pc = Pointclouds(
            points=mesh_xyz[subset_mesh][None],
        )
        clouds["Output"]["GT mesh"] = mesh_pc
    
    # Centers
    if centers is not None:
        centers_pc = Pointclouds(
            points=centers
            )
        clouds["Output"]["Centers"] = centers_pc
    
    if sampled_seen is not None:
        sampled_seen_pc = Pointclouds(
            points=sampled_seen
            )
        clouds["Output"]["Sampled seen"] = sampled_seen_pc

    pred_occ = torch.nn.Sigmoid()(pred_occ).cpu()
    for t in score_thresholds:
        pos = pred_occ > t

        points = unseen_xyz[pos].reshape((-1, 3))
        features = pred_rgb[None][pos].reshape((-1, 3))
        good_points = points[:, 0] != -100

        if good_points.sum() == 0:
            continue

        pc = Pointclouds(
            points=points[good_points][None].cpu(),
            features=features[good_points][None].cpu(),
        )

        clouds["Output"][f"pred_{t}"] = pc

    plt.figure()
    try:
        fig = plot_scene(clouds, pointcloud_marker_size=pointcloud_marker_size, pointcloud_max_points=20000 * 2)
        fig.update_layout(height=1000, width=1000)
        f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))
    except Exception as e:
        print('writing failed', e)
    try:
        plt.close()
    except:
        pass


def run_viz_udf(model, data_loader, device, args, epoch):
    epoch_start_time = time.time()
    model.eval()

    print('Visualization data_loader length:', len(data_loader))
    dataset = data_loader.dataset
    for sample_idx, samples in enumerate(tqdm(data_loader, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')):
        if sample_idx >= args.max_n_viz_obj:
            break
        seen_xyz, valid_seen_xyz, query_xyz, unseen_rgb, labels, seen_images, gt_fps_xyz, seen_xyz_hr, valid_seen_xyz_hr, _, _ = prepare_data_udf(samples, device, is_train=False, is_viz=True, args=args)

        seen_images_no_preprocess = seen_images.clone()


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
        
        # don't forward all at once to avoid oom
        max_n_queries_fwd = args.n_query_udf if not args.hr else int(args.n_query_udf * (args.xyz_size/args.xyz_size_hr)**2)

        # Filter query based on centers xyz # (1, 200, 3)
        offset = 0.3
        min_xyz = torch.min(centers_xyz, dim=1)[0][0] - offset
        max_xyz = torch.max(centers_xyz, dim=1)[0][0] + offset

        mask = (torch.rand(1, query_xyz.size()[1]) >= 0).to(args.device)
        mask = mask & (query_xyz[:,:,0] > min_xyz[0]) & (query_xyz[:,:,1] > min_xyz[1]) & (query_xyz[:,:,2] > min_xyz[2])
        mask = mask & (query_xyz[:,:,0] < max_xyz[0]) & (query_xyz[:,:,1] < max_xyz[1]) & (query_xyz[:,:,2] < max_xyz[2])
        query_xyz = query_xyz[mask].unsqueeze(0)

        total_n_passes = int(np.ceil(query_xyz.shape[1] / max_n_queries_fwd))

        pred_points = np.empty((0,3))
        pred_colors = np.empty((0,3))

        if args.distributed:
            for param in model.module.parameters():
                param.requires_grad = False
        else:
            for param in model.parameters():
                param.requires_grad = False


        for p_idx in range(total_n_passes):
            p_start = p_idx     * max_n_queries_fwd
            p_end = (p_idx + 1) * max_n_queries_fwd
            cur_query_xyz = query_xyz[:, p_start:p_end]

            with torch.no_grad():
                if args.hr != 1:
                    seen_points = seen_xyz
                    valid_seen = valid_seen_xyz
                else:
                    seen_points = seen_xyz_hr
                    valid_seen = valid_seen_xyz_hr

                if args.distributed:
                    pred = model.module.decoderl2(cur_query_xyz, seen_points, valid_seen, fea, up_grid_fea, custom_centers = None)
                    pred = model.module.fc_out(pred)
                else:
                    pred = model.decoderl2(cur_query_xyz, seen_points, valid_seen, fea, up_grid_fea, custom_centers = None)
                    pred = model.fc_out(pred)

            max_dist = 0.5
            pred_udf = F.relu(pred[:,:,:1]).reshape((-1, 1)) # nQ, 1
            pred_udf = torch.clamp(pred_udf, max=max_dist) 

            # Candidate points
            t = args.udf_threshold
            pos = (pred_udf < t).squeeze(-1) # (nQ, )
            points = cur_query_xyz.squeeze(0) # (nQ, 3)
            points = points[pos].unsqueeze(0) # (1, n, 3)

            if torch.sum(pos) > 0:
                points = move_points(model, points, seen_points, valid_seen, fea, up_grid_fea, args, n_iter=args.udf_n_iter)

                # predict final color
                with torch.no_grad():
                    if args.distributed:
                        pred = model.module.decoderl2(points, seen_points, valid_seen, fea, up_grid_fea)
                        pred = model.module.fc_out(pred)
                    else:
                        pred = model.decoderl2(points, seen_points, valid_seen, fea, up_grid_fea)
                        pred = model.fc_out(pred)

                cur_color_out = pred[:,:,1:].reshape((-1, 3, 256)).max(dim=2)[1] / 255.0
                cur_color_out = cur_color_out.detach().squeeze(0).cpu().numpy()
                if len(cur_color_out.shape) == 1:
                    cur_color_out = cur_color_out[None,...]
                pts = points.detach().squeeze(0).cpu().numpy()
                pred_points = np.append(pred_points, pts, axis = 0)
                pred_colors = np.append(pred_colors, cur_color_out, axis = 0)
            
        rank = misc.get_rank()
        out_folder = os.path.join('experiments/', f'{args.exp_name}', 'viz', 'epoch'+str(epoch).zfill(3))
        Path(out_folder).mkdir(parents= True, exist_ok=True)
        prefix = os.path.join(out_folder, dataset.dataset_split+f'_ep{epoch}_rank{rank}_i{sample_idx}')
        img = (seen_images_no_preprocess[0].permute(1, 2, 0) * 255).cpu().numpy().copy().astype(np.uint8)

        gt_xyz = samples[1][0].to(device).reshape(-1, 3)
        gt_rgb = samples[1][1].to(device).reshape(-1, 3)
        #mesh_xyz = samples[2].to(device).reshape(-1, 3) if args.use_hypersim else None
        mesh_xyz = None

        fn_pc = None
        fn_pc_seen = None
        fn_pc_gt = None
        if args.save_pc:
            out_folder_ply = os.path.join('experiments/', f'{args.exp_name}', 'ply', 'epoch'+str(epoch).zfill(3))
            Path(out_folder_ply).mkdir(parents= True, exist_ok=True)
            prefix_pc = os.path.join(out_folder_ply, dataset.dataset_split+f'_ep{epoch}_rank{rank}_i{sample_idx}')
            fn_pc = prefix_pc + '.ply'

            # seen
            out_folder_ply = os.path.join('experiments/', f'{args.exp_name}', 'ply_seen', 'epoch'+str(epoch).zfill(3))
            Path(out_folder_ply).mkdir(parents= True, exist_ok=True)
            prefix_pc = os.path.join(out_folder_ply, dataset.dataset_split+f'_ep{epoch}_rank{rank}_i{sample_idx}')
            fn_pc_seen = prefix_pc +'_seen' +'.ply'

            # gt
            out_folder_ply = os.path.join('experiments/', f'{args.exp_name}', 'ply_gt', 'epoch'+str(epoch).zfill(3))
            Path(out_folder_ply).mkdir(parents= True, exist_ok=True)
            prefix_pc = os.path.join(out_folder_ply, dataset.dataset_split+f'_ep{epoch}_rank{rank}_i{sample_idx}')
            fn_pc_gt = prefix_pc +'_gt' +'.ply'

        with open(prefix + '.html', 'a') as f:
            generate_html_udf(
                img,
                seen_xyz, seen_images_no_preprocess,
                pred_points,
                pred_colors,
                query_xyz,
                f,
                gt_xyz=gt_xyz,
                gt_rgb=gt_rgb,
                mesh_xyz=mesh_xyz,
                centers = centers_xyz,
                fn_pc=fn_pc,
                fn_pc_seen = fn_pc_seen,
                fn_pc_gt=fn_pc_gt
            )
    print("Visualization epoch time:", time.time() - epoch_start_time)

def generate_html_udf(img, seen_xyz, seen_rgb, pred_points, pred_rgb, unseen_xyz, f,
        gt_xyz=None, gt_rgb=None, mesh_xyz=None, centers = None, sampled_seen = None, fn_pc = None, fn_pc_seen = None, fn_pc_gt = None,
        pointcloud_marker_size=2,
    ):
    if img is not None:
        fig = plt.figure()
        plt.imshow(img)
        tmpfile = BytesIO()
        fig.savefig(tmpfile, format='jpg')
        encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')

        html = '<img src=\'data:image/png;base64,{}\'>'.format(encoded)
        f.write(html)
        plt.close()

    clouds = {"Output": {}}
    # Seen
    if seen_xyz is not None:
        seen_xyz = seen_xyz.reshape((-1, 3)).cpu()
        seen_rgb = torch.nn.functional.interpolate(seen_rgb, (112, 112)).permute(0, 2, 3, 1).reshape((-1, 3)).cpu()
        good_seen = seen_xyz[:, 0] != -100

        seen_pc = Pointclouds(
            points=seen_xyz[good_seen][None],
            features=seen_rgb[good_seen][None],
        )
        clouds["Output"]["seen"] = seen_pc

        if fn_pc_seen != None:
            colors = (seen_rgb[good_seen][None]*255).int()
            cloud = PyntCloud(pd.DataFrame(
            data=np.hstack((seen_xyz[good_seen][None].squeeze(0).numpy(), colors.squeeze(0).numpy())),
            columns=["x", "y", "z", "red", "green", "blue"])
            )
            cloud.to_file(fn_pc_seen)

    # GT points
    if gt_xyz is not None:
        subset_gt = random.sample(range(gt_xyz.shape[0]), 20000)
        gt_pc = Pointclouds(
            points=gt_xyz[subset_gt][None],
            features=gt_rgb[subset_gt][None],
        )
        clouds["Output"]["GT points"] = gt_pc

        if fn_pc_gt != None:
            colors = (gt_rgb[subset_gt][None]*255).int().cpu()
            cloud = PyntCloud(pd.DataFrame(
            data=np.hstack((gt_xyz[subset_gt][None].squeeze(0).cpu().numpy(), colors.squeeze(0).numpy())),
            columns=["x", "y", "z", "red", "green", "blue"])
            )
            cloud.to_file(fn_pc_gt)
    

    # GT meshes
    if mesh_xyz is not None:
        subset_mesh = random.sample(range(mesh_xyz.shape[0]), 10000)
        mesh_pc = Pointclouds(
            points=mesh_xyz[subset_mesh][None],
        )
        clouds["Output"]["GT mesh"] = mesh_pc
        
    
    # Centers
    if centers is not None:
        centers_pc = Pointclouds(
            points=centers
            )
        clouds["Output"]["Centers"] = centers_pc
    
    if sampled_seen is not None:
        sampled_seen_pc = Pointclouds(
            points=sampled_seen
            )
        clouds["Output"]["Sampled seen"] = sampled_seen_pc


    # pred
    points = pred_points
    features = pred_rgb
    good_points = points[:, 0] != -100

    points = torch.from_numpy(np.expand_dims(points[good_points], axis = 0))
    features = torch.from_numpy(np.expand_dims(features[good_points], axis = 0))


    if good_points.sum() != 0:
        pc = Pointclouds(
            points= points,
            features=features,
        )

        if fn_pc != None:
            colors = (features*255).int()
            cloud = PyntCloud(pd.DataFrame(
            # same arguments that you are passing to visualize_pcl
            data=np.hstack((points.squeeze(0).numpy(), colors.squeeze(0).numpy())),
            columns=["x", "y", "z", "red", "green", "blue"])
            )
            cloud.to_file(fn_pc)

        clouds["Output"]["pred_udf"] = pc

    plt.figure()
    try:
        fig = plot_scene(clouds, pointcloud_marker_size=pointcloud_marker_size, pointcloud_max_points=20000 * 2)
        fig.update_layout(height=1000, width=1000)
        f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))
    except Exception as e:
        print('writing failed', e)
    try:
        plt.close()
    except:
        pass