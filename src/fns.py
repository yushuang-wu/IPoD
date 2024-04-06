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


import random
import torch
import torch.nn as nn
import numpy as np
from pytorch3d.transforms import RotateAxisAngle
import torch.nn.functional as F
from src.layers import LayerNorm
from src.chamfer_loss_separate import chamfer_distance_sep

class Swish(nn.Module):
    def forward(self,x):
        return  x * torch.sigmoid(x)

class SharedMLP(nn.Module):
    def __init__(self, in_channels, out_channels, dim=1):
        super().__init__()
        conv = nn.Conv1d
        bn = nn.GroupNorm
        
        if not isinstance(out_channels, (list, tuple)):
            out_channels = [out_channels]
        layers = []
        for oc in out_channels:
            layers.extend([
                conv(in_channels, oc, 1),
                bn(8, oc),
                Swish(),
            ])
            in_channels = oc
        self.layers = nn.Sequential(*layers)

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 1)
        return self.layers(inputs)

def repulsive(points):
    pts = points.clone()
    k = min(16+1, pts.shape[1]) # plus itself
    dists = square_distance(pts, pts)
    sort_dist, sort_idx = dists.sort()
    knn_dist = sort_dist[:,:,:k] # 1, n, k
    knn_dist = torch.clamp(knn_dist, min=0.001)
    knn_idx = sort_idx[:,:,:k] # 1, n, k

    knn_points = index_points(pts, knn_idx) # 1, n, k, 3
    d = pts[:, :, None] - knn_points # 1, n, k, 3

    const = 0.001
    repulsive = d / (knn_dist[...,None])**2 # 1, n, k, 3 #^2
    repulsive = torch.sum(repulsive, dim=2) * const # 1, n, 3
    repulsive = torch.clamp(repulsive, min=-0.03, max=0.03)

    return repulsive

def move_points(model, points, seen_xyz, valid_seen_xyz, fea, up_grid_fea, args, n_iter=2):
    points.requires_grad = True

    for i in range(n_iter):
        pred = model.decoderl2(points, seen_xyz, valid_seen_xyz, fea, up_grid_fea)
        pred = model.fc_out2(pred)

        pred_udf = pred[:,:,0]
        pred_udf.sum().backward()

        gradient = points.grad.detach()
        points = points.detach() # 1, n, 3
        pred_udf = pred_udf.detach()

        points = points - F.normalize(gradient, dim=2) * pred_udf.reshape(-1, 1)
        points = points.detach()

        # repulsive force
        if args.repulsive==1:
            points += repulsive(points)

        points.requires_grad = True

    return points

def evaluate_colors(predicted_xyz, gt_xyz, predicted_rgb, gt_rgb, dist_thres=0.1):
    if predicted_xyz.shape[0] == 0:
        return 0.0, 0.0, 0.0

    nneigh = 1
    pts1, col1 = predicted_xyz.unsqueeze(0), predicted_rgb.unsqueeze(0) # B, N1, 3
    pts2, col2 = gt_xyz.unsqueeze(0), gt_rgb.unsqueeze(0) # B, N, 3

    slice_size = 1000

    metrics_sum, n = 0, 0
    for i in range(int(np.ceil(predicted_xyz.shape[0] / slice_size))):
        start = slice_size * i
        end   = slice_size * (i + 1)
     
        # knn pred -> gt
        dists = square_distance(pts1[:, start:end, :], pts2)
        sort_dist, sort_idx = dists.sort()
        knn_dist = sort_dist[:,:,:nneigh].squeeze(-1) # B, n1
        knn_idx = sort_idx[:,:,:nneigh] # B, n1, k
        col2_nn = index_points(col2,knn_idx) # B, n1, k, 3
        col2_nn = col2_nn.squeeze(-2) # B, n1, 3
        color_dist = torch.abs(col1[:, start:end, :] - col2_nn).sum(-1) # B, N1

        metrics_sum += color_dist[knn_dist < dist_thres].sum()
        n += (knn_dist < dist_thres).sum()
        
    color_1_2 = metrics_sum / n

    # knn gt -> pred
    metrics_sum, n = 0, 0
    for i in range(int(np.ceil(gt_xyz.shape[0] / slice_size))):
        start = slice_size * i
        end   = slice_size * (i + 1)

        dists = square_distance(pts2[:, start:end, :], pts1)
        sort_dist, sort_idx = dists.sort()
        knn_dist = sort_dist[:,:,:nneigh].squeeze(-1) # B, n2
        knn_idx = sort_idx[:,:,:nneigh] # B, n2, k
        col1_nn = index_points(col1,knn_idx) # B, n2, k, 3
        col1_nn = col1_nn.squeeze(-2) # B, n2, 3
        color_dist = torch.abs(col2[:, start:end] - col1_nn).sum(-1) # B, N2

        metrics_sum += color_dist[knn_dist < dist_thres].sum()
        n += (knn_dist < dist_thres).sum()
    color_2_1 = metrics_sum / n

    return [color_1_2, color_2_1, (color_1_2+color_2_1)/2]

def evaluate_points(predicted_xyz, gt_xyz, dist_thres):
    if predicted_xyz.shape[0] == 0:
        return 0.0, 0.0, 0.0
    slice_size = 1000
    precision = 0.0
    for i in range(int(np.ceil(predicted_xyz.shape[0] / slice_size))):
        start = slice_size * i
        end   = slice_size * (i + 1)
        dist = ((predicted_xyz[start:end, None] - gt_xyz[None]) ** 2.0).sum(axis=-1) ** 0.5
        precision += ((dist < dist_thres).sum(axis=1) > 0).sum()
     
    precision /= predicted_xyz.shape[0]

    recall = 0.0

    for i in range(int(np.ceil(gt_xyz.shape[0] / slice_size))):
        start = slice_size * i
        end   = slice_size * (i + 1)
        dist = ((predicted_xyz[:, None] - gt_xyz[None, start:end]) ** 2.0).sum(axis=-1) ** 0.5
        recall += ((dist < dist_thres).sum(axis=0) > 0).sum()

    recall /= gt_xyz.shape[0]
    return precision, recall, get_f1(precision, recall)

def evaluate_distance(predicted_xyz, gt_xyz):
    if predicted_xyz.shape[0] == 0:
        return 0, 0, 0
    
    with torch.no_grad():
        res = chamfer_distance_sep(predicted_xyz.unsqueeze(0), gt_xyz.unsqueeze(0), norm=1)
        prec_dist, rec_dist = res[0][0], res[0][1]
        chamfer_dist = res[1] 

    return prec_dist, rec_dist, chamfer_dist

def aug_xyz(seen_xyz, unseen_xyz, args, is_train):
    degree_x = 0
    degree_y = 0
    degree_z = 0
    if is_train:
        r_delta = args.random_scale_delta
        scale = torch.tensor([
            random.uniform(1.0 - r_delta, 1.0 + r_delta),
            random.uniform(1.0 - r_delta, 1.0 + r_delta),
            random.uniform(1.0 - r_delta, 1.0 + r_delta),
        ], device=seen_xyz.device)

        if args.use_hypersim:
            shift = 0
        else:
            degree_x = random.randrange(-args.random_rotate_degree, args.random_rotate_degree + 1)
            degree_y = random.randrange(-args.random_rotate_degree, args.random_rotate_degree + 1)
            degree_z = random.randrange(-args.random_rotate_degree, args.random_rotate_degree + 1)

            r_shift = args.random_shift
            shift = torch.tensor([[[
                random.uniform(-r_shift, r_shift),
                random.uniform(-r_shift, r_shift),
                random.uniform(-r_shift, r_shift),
            ]]], device=seen_xyz.device)
        seen_xyz = seen_xyz * scale + shift
        unseen_xyz = unseen_xyz * scale + shift

    B, H, W, _ = seen_xyz.shape
    return [
        rotate(seen_xyz.reshape((B, -1, 3)), degree_x, degree_y, degree_z).reshape((B, H, W, 3)),
        rotate(unseen_xyz, degree_x, degree_y, degree_z),
    ]

def aug_xyz_all(seen_xyz, unseen_xyz, gt_fps_xyz, args, is_train, seen_xyz_hr=None):
    degree_x = 0
    degree_y = 0
    degree_z = 0
    if is_train:
        r_delta = args.random_scale_delta
        scale = torch.tensor([
            random.uniform(1.0 - r_delta, 1.0 + r_delta),
            random.uniform(1.0 - r_delta, 1.0 + r_delta),
            random.uniform(1.0 - r_delta, 1.0 + r_delta),
        ], device=seen_xyz.device)

        if args.use_hypersim:
            shift = 0
        else:
            degree_x = random.randrange(-args.random_rotate_degree, args.random_rotate_degree + 1)
            degree_y = random.randrange(-args.random_rotate_degree, args.random_rotate_degree + 1)
            degree_z = random.randrange(-args.random_rotate_degree, args.random_rotate_degree + 1)

            r_shift = args.random_shift
            shift = torch.tensor([[[
                random.uniform(-r_shift, r_shift),
                random.uniform(-r_shift, r_shift),
                random.uniform(-r_shift, r_shift),
            ]]], device=seen_xyz.device)
        seen_xyz = seen_xyz * scale + shift
        unseen_xyz = unseen_xyz * scale + shift
        gt_fps_xyz = gt_fps_xyz * scale + shift

        if seen_xyz_hr != None:
            seen_xyz_hr = seen_xyz_hr * scale + shift

    B, H, W, _ = seen_xyz.shape

    rotated = [
        rotate(seen_xyz.reshape((B, -1, 3)), degree_x, degree_y, degree_z).reshape((B, H, W, 3)),
        rotate(unseen_xyz, degree_x, degree_y, degree_z),
        rotate(gt_fps_xyz, degree_x, degree_y, degree_z),
    ]

    if seen_xyz_hr != None:
        B, H_hr, W_hr, _ = seen_xyz_hr.shape
        seen_xyz_hr = rotate(seen_xyz_hr.reshape((B, -1, 3)), degree_x, degree_y, degree_z).reshape((B, H_hr, W_hr, 3))
    rotated.append(seen_xyz_hr)    

    return rotated
    
def aug_xyz_udf_train(seen_xyz, gt_xyz, args, is_train, seen_xyz_hr=None):
    degree_x = 0
    degree_y = 0
    degree_z = 0
    if is_train:
        r_delta = args.random_scale_delta
        scale = torch.tensor([
            random.uniform(1.0 - r_delta, 1.0 + r_delta),
            random.uniform(1.0 - r_delta, 1.0 + r_delta),
            random.uniform(1.0 - r_delta, 1.0 + r_delta),
        ], device=seen_xyz.device)

        if args.use_hypersim:
            shift = 0
        else:
            degree_x = random.randrange(-args.random_rotate_degree, args.random_rotate_degree + 1)
            degree_y = random.randrange(-args.random_rotate_degree, args.random_rotate_degree + 1)
            degree_z = random.randrange(-args.random_rotate_degree, args.random_rotate_degree + 1)

            r_shift = args.random_shift
            shift = torch.tensor([[[
                random.uniform(-r_shift, r_shift),
                random.uniform(-r_shift, r_shift),
                random.uniform(-r_shift, r_shift),
            ]]], device=seen_xyz.device)
        seen_xyz = seen_xyz * scale + shift
        gt_xyz = gt_xyz * scale + shift

        if seen_xyz_hr != None:
            seen_xyz_hr = seen_xyz_hr * scale + shift

    B, H, W, _ = seen_xyz.shape

    rotated = [
        rotate(seen_xyz.reshape((B, -1, 3)), degree_x, degree_y, degree_z).reshape((B, H, W, 3)),
        rotate(gt_xyz, degree_x, degree_y, degree_z),
    ]

    if seen_xyz_hr != None:
        B, H_hr, W_hr, _ = seen_xyz_hr.shape
        seen_xyz_hr = rotate(seen_xyz_hr.reshape((B, -1, 3)), degree_x, degree_y, degree_z).reshape((B, H_hr, W_hr, 3))
    rotated.append(seen_xyz_hr)
    
    return rotated


def rotate(sample, degree_x, degree_y, degree_z):
    for degree, axis in [(degree_x, "X"), (degree_y, "Y"), (degree_z, "Z")]:
        if degree != 0:
            sample = RotateAxisAngle(degree, axis=axis).to(sample.device).transform_points(sample)
    return sample

def get_grid(B, device, co3d_world_size, granularity):
    N = int(np.ceil(2 * co3d_world_size / granularity))
    grid_unseen_xyz = torch.zeros((N, N, N, 3), device=device)
    for i in range(N):
        grid_unseen_xyz[i, :, :, 0] = i
    for j in range(N):
        grid_unseen_xyz[:, j, :, 1] = j
    for k in range(N):
        grid_unseen_xyz[:, :, k, 2] = k
    grid_unseen_xyz -= (N / 2.0)
    grid_unseen_xyz /= (N / 2.0) / co3d_world_size
    grid_unseen_xyz = grid_unseen_xyz.reshape((1, -1, 3)).repeat(B, 1, 1)
    return grid_unseen_xyz

def get_f1(precision, recall):
    if (precision + recall) == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)

def sample_uniform_semisphere(B, N, semisphere_size, device):
    for _ in range(100):
        points = torch.empty(B * N * 3, 3, device=device).uniform_(-semisphere_size, semisphere_size)
        points[..., 2] = points[..., 2].abs()
        dist = (points ** 2.0).sum(axis=-1) ** 0.5
        if (dist < semisphere_size).sum() >= B * N:
            return points[dist < semisphere_size][:B * N].reshape((B, N, 3))
        else:
            print('resampling sphere')

def get_grid_semisphere(B, granularity, semisphere_size, device):
    n_grid_pts = int(semisphere_size / granularity) * 2 + 1
    grid_unseen_xyz = torch.zeros((n_grid_pts, n_grid_pts, n_grid_pts // 2 + 1, 3), device=device)
    for i in range(n_grid_pts):
        grid_unseen_xyz[i, :, :, 0] = i
        grid_unseen_xyz[:, i, :, 1] = i
    for i in range(n_grid_pts // 2 + 1):
        grid_unseen_xyz[:, :, i, 2] = i
    grid_unseen_xyz[..., :2] -= (n_grid_pts // 2.0)
    grid_unseen_xyz *= granularity
    dist = (grid_unseen_xyz ** 2.0).sum(axis=-1) ** 0.5
    grid_unseen_xyz = grid_unseen_xyz[dist <= semisphere_size]
    return grid_unseen_xyz[None].repeat(B, 1, 1)

def get_min_dist(a, b, slice_size=1000):
    all_min, all_idx = [], []
    for i in range(int(np.ceil(a.shape[1] / slice_size))):
        start = slice_size * i
        end   = slice_size * (i + 1)
        # B, n_queries, n_gt
        dist = ((a[:, start:end] - b) ** 2.0).sum(axis=-1) ** 0.5
        # B, n_queries
        cur_min, cur_idx = dist.min(axis=2)
        all_min.append(cur_min)
        all_idx.append(cur_idx)
    return torch.cat(all_min, dim=1), torch.cat(all_idx, dim=1)

def construct_uniform_semisphere(gt_xyz, gt_rgb, semisphere_size, n_queries, dist_threshold, is_train, granularity, is_viz, args = None):
    B = gt_xyz.shape[0]
    device = gt_xyz.device

    if is_train:
        unseen_xyz = sample_uniform_semisphere(B, n_queries, semisphere_size, device)
    else:
        unseen_xyz = get_grid_semisphere(B, granularity, semisphere_size, device)
    dist, idx_to_gt = get_min_dist(unseen_xyz[:, :, None], gt_xyz[:, None])
    labels = dist < dist_threshold
    unseen_rgb = torch.zeros_like(unseen_xyz)
    unseen_rgb[labels] = torch.gather(gt_rgb, 1, idx_to_gt.unsqueeze(-1).repeat(1, 1, 3))[labels]
    return unseen_xyz, unseen_rgb, [labels.float()]

def construct_uniform_semisphere_udf(gt_xyz, gt_rgb, semisphere_size, n_queries, dist_threshold, is_train, granularity, is_viz, args = None):
    B = gt_xyz.shape[0]
    device = gt_xyz.device

    if is_train:
        unseen_xyz = sample_uniform_semisphere(B, n_queries, semisphere_size, device)
    else:
        unseen_xyz = get_grid_semisphere(B, granularity, semisphere_size, device)
    dist, idx_to_gt = get_min_dist(unseen_xyz[:, :, None], gt_xyz[:, None])
    labels = dist
    labels_01 = dist < dist_threshold
    unseen_rgb = torch.zeros_like(unseen_xyz)
    unseen_rgb[labels_01] = torch.gather(gt_rgb, 1, idx_to_gt.unsqueeze(-1).repeat(1, 1, 3))[labels_01]
    return unseen_xyz, unseen_rgb, [labels_01.float(), labels.float()]

def construct_uniform_grid(gt_xyz, gt_rgb, co3d_world_size, n_queries, dist_threshold, is_train, granularity, is_viz, args = None):
    B = gt_xyz.shape[0]
    device = gt_xyz.device

    if is_train:
        unseen_xyz = torch.empty((B, n_queries, 3), device=device).uniform_(-co3d_world_size, co3d_world_size)
    elif is_viz:
        nq = 216000 if not args.save_pc else 5000000
        unseen_xyz = torch.empty((B, nq, 3), device=device).uniform_(-co3d_world_size, co3d_world_size)
    else:
        unseen_xyz = get_grid(B, device, co3d_world_size, granularity)
    dist, idx_to_gt = get_min_dist(unseen_xyz[:, :, None], gt_xyz[:, None])
    labels = dist < dist_threshold
    unseen_rgb = torch.zeros_like(unseen_xyz)
    unseen_rgb[labels] = torch.gather(gt_rgb, 1, idx_to_gt.unsqueeze(-1).repeat(1, 1, 3))[labels]
    return unseen_xyz, unseen_rgb, [labels.float()]

def construct_uniform_grid_udf(gt_xyz, gt_rgb, co3d_world_size, n_queries, dist_threshold, is_train, granularity, is_viz, args = None):
    B = gt_xyz.shape[0]
    device = gt_xyz.device

    if is_train:
        unseen_xyz = torch.empty((B, n_queries, 3), device=device).uniform_(-co3d_world_size, co3d_world_size)
    elif is_viz:
        nq = 216000 if not args.save_pc else 5000000
        unseen_xyz = torch.empty((B, nq, 3), device=device).uniform_(-co3d_world_size, co3d_world_size)
    else:
        unseen_xyz = get_grid(B, device, co3d_world_size, granularity)
        
    dist, idx_to_gt = get_min_dist(unseen_xyz[:, :, None], gt_xyz[:, None])
    labels = dist
    labels_01 = dist < dist_threshold
    unseen_rgb = torch.zeros_like(unseen_xyz)
    unseen_rgb[labels_01] = torch.gather(gt_rgb, 1, idx_to_gt.unsqueeze(-1).repeat(1, 1, 3))[labels_01]

    return unseen_xyz, unseen_rgb, [labels_01.float(), labels.float()]

def shrink_points_beyond_threshold(xyz, threshold):
    xyz = xyz.clone().detach()
    dist = (xyz ** 2.0).sum(axis=-1) ** 0.5
    affected = (dist > threshold) * torch.isfinite(dist)
    xyz[affected] = xyz[affected] * (
        threshold * (2.0 - threshold / dist[affected]) / dist[affected]
    )[..., None]
    return xyz

def preprocess_img(x, res=224.):
    if x.shape[2] != res:
        assert x.shape[2] == 800
        x = F.interpolate(
            x,
            scale_factor=res/800.,
            mode="bilinear",
        )
    resnet_mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).reshape((1, 3, 1, 1))
    resnet_std = torch.tensor([0.229, 0.224, 0.225], device=x.device).reshape((1, 3, 1, 1))
    imgs_normed = (x - resnet_mean) / resnet_std
    return imgs_normed

def square_distance(src, dst):
    """
    Code from: https://github.com/qq456cvb/Point-Transformers/blob/master/pointnet_util.py

    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    return torch.sum((src[:, :, None] - dst[:, None]) ** 2, dim=-1)

def index_points(points, idx):
    """
    Code from: https://github.com/qq456cvb/Point-Transformers/blob/master/pointnet_util.py
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S, [K]]
    Return:
        new_points:, indexed points data, [B, S, [K], C]
    """
    raw_size = idx.size()
    idx = idx.reshape(raw_size[0], -1)
    res = torch.gather(points, 1, idx[..., None].expand(-1, -1, points.size(-1)))
    return res.reshape(*raw_size, -1)

def get_norm(norm, out_channels):
    """
    Args:
        norm (str or callable): either one of BN, SyncBN, FrozenBN, GN;
            or a callable that takes a channel number and returns
            the normalization layer as a nn.Module.

    Returns:
        nn.Module or None: the normalization layer
    """
    if norm is None:
        return None
    if isinstance(norm, str):
        if len(norm) == 0:
            return None
        norm = {
            "LN": lambda channels: LayerNorm(channels),
        }[norm]
    return norm(out_channels)
