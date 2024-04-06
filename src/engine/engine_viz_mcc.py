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

from src.engine.engine_mcc import prepare_data
from pathlib import Path
from tqdm import tqdm

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
        seen_xyz, valid_seen_xyz, unseen_xyz, unseen_rgb, labels, seen_images = prepare_data(samples, device, is_train=False, args=args, is_viz=True)

        pred_occupy = []
        pred_colors = []
        (model.module if hasattr(model, "module") else model).clear_cache()

        # don't forward all at once to avoid oom
        max_n_queries_fwd = 2000

        total_n_passes = int(np.ceil(unseen_xyz.shape[1] / max_n_queries_fwd))
        for p_idx in range(total_n_passes):
            p_start = p_idx     * max_n_queries_fwd
            p_end = (p_idx + 1) * max_n_queries_fwd
            cur_unseen_xyz = unseen_xyz[:, p_start:p_end]
            cur_unseen_rgb = unseen_rgb[:, p_start:p_end].zero_()
            cur_labels = labels[:, p_start:p_end].zero_()

            with torch.no_grad():
                _, pred, = model(
                    seen_images=seen_images,
                    seen_xyz=seen_xyz,
                    unseen_xyz=cur_unseen_xyz,
                    unseen_rgb=cur_unseen_rgb,
                    unseen_occupy=cur_labels,
                    cache_enc=args.run_viz,
                    valid_seen_xyz=valid_seen_xyz,
                )

            cur_occupy_out = pred[..., 0]

            if args.regress_color:
                cur_color_out = pred[..., 1:].reshape((-1, 3))
            else:
                cur_color_out = pred[..., 1:].reshape((-1, 3, 256)).max(dim=2)[1] / 255.0
            pred_occupy.append(cur_occupy_out)
            pred_colors.append(cur_color_out)

        rank = misc.get_rank()
        out_folder = os.path.join('experiments/', f'{args.exp_name}', 'viz', 'epoch'+str(epoch).zfill(3))
        Path(out_folder).mkdir(parents= True, exist_ok=True)
        prefix = os.path.join(out_folder, dataset.dataset_split+f'_ep{epoch}_rank{rank}_i{sample_idx}')
        img = (seen_images[0].permute(1, 2, 0) * 255).cpu().numpy().copy().astype(np.uint8)

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
            generate_html(
                img,
                seen_xyz, seen_images,
                torch.cat(pred_occupy, dim=1),
                torch.cat(pred_colors, dim=0),
                unseen_xyz,
                f,
                gt_xyz=gt_xyz,
                gt_rgb=gt_rgb,
                mesh_xyz=mesh_xyz,
                fn_pc=fn_pc,
                fn_pc_seen = fn_pc_seen,
                fn_pc_gt=fn_pc_gt
            )
    print("Visualization epoch time:", time.time() - epoch_start_time)

def generate_html(img, seen_xyz, seen_rgb, pred_occ, pred_rgb, unseen_xyz, f,
        gt_xyz=None, gt_rgb=None, mesh_xyz=None, score_thresholds=[0.1, 0.3, 0.5, 0.7, 0.9], fn_pc = None, fn_pc_seen = None, fn_pc_gt = None,
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

    clouds = {"MCC Output": {}}
    # Seen
    if seen_xyz is not None:
        seen_xyz = seen_xyz.reshape((-1, 3)).cpu()
        seen_rgb = torch.nn.functional.interpolate(seen_rgb, (112, 112)).permute(0, 2, 3, 1).reshape((-1, 3)).cpu()
        good_seen = seen_xyz[:, 0] != -100

        seen_pc = Pointclouds(
            points=seen_xyz[good_seen][None],
            features=seen_rgb[good_seen][None],
        )
        clouds["MCC Output"]["seen"] = seen_pc

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
        clouds["MCC Output"]["GT points"] = gt_pc

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
        clouds["MCC Output"]["GT mesh"] = mesh_pc

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

        clouds["MCC Output"][f"pred_{t}"] = pc

        if fn_pc != None and t == 0.1:
            colors = (features[good_points][None].cpu()*255).int()
            cloud = PyntCloud(pd.DataFrame(
            data=np.hstack((points[good_points][None].cpu().squeeze(0).numpy(), colors.squeeze(0).numpy())),
            columns=["x", "y", "z", "red", "green", "blue"])
            )
            cloud.to_file(fn_pc)

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