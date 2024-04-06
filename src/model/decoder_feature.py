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

import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.nn.functional as F
from src.fns import square_distance, index_points
import math
import numpy as np
from src.layers import positional_encoding, positional_encoding_t
    
class CrossTransformerBlock(nn.Module):
    def __init__(self, dim_inp, dim, nneigh=5, args=None):
        super().__init__()

        self.dim = dim
        self.nneigh = nneigh

        self.fc_delta = nn.Sequential(
            nn.Linear(3, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

        self.fc_gamma = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
        self.w_k_global = nn.Linear(dim_inp, dim, bias=False)
        self.w_v_global = nn.Linear(dim_inp, dim, bias=False)

        self.w_qs = nn.Linear(dim_inp, dim, bias=False)
        self.w_ks = nn.Linear(dim_inp, dim, bias=False)
        self.w_vs = nn.Linear(dim_inp, dim, bias=False)

        self.w_kc = nn.Linear(dim_inp, dim, bias=False)
        self.w_vc = nn.Linear(dim_inp, dim, bias=False)
    

    def forward(self, xyz_q, lat_rep, xyz, points, sampled_grid_feat, closest_seen):
        
        with torch.no_grad():
            dists = square_distance(xyz_q, xyz)
            
            sort_dist, sort_idx = dists.sort()
            knn_dist = sort_dist[:,:,:self.nneigh] # B, nQ, k
            knn_idx = sort_idx[:,:,:self.nneigh] # B, nQ, k

        b, nQ, _ = xyz_q.shape

        if len(lat_rep.shape) == 2:
            q_attn = self.w_qs(lat_rep).unsqueeze(1).repeat(1, nQ, 1)
            k_global = self.w_k_global(lat_rep).unsqueeze(1).repeat(1, nQ, 1).unsqueeze(2)
            v_global = self.w_v_global(lat_rep).unsqueeze(1).repeat(1, nQ, 1).unsqueeze(2)
        else:
            q_attn = self.w_qs(lat_rep)
            k_global = self.w_k_global(lat_rep).unsqueeze(2)
            v_global = self.w_v_global(lat_rep).unsqueeze(2)

        k_attn = index_points(self.w_ks(points),knn_idx)  # b, nQ, k, dim 
        k_closest = self.w_kc(sampled_grid_feat) # b, nQ, nn_seen, dim
        k_attn = torch.cat([k_attn, k_closest, k_global], dim=2)
        v_attn = index_points(self.w_vs(points), knn_idx) 
        v_closest = self.w_vc(sampled_grid_feat) # b, nQ, nn_seen, dim
        v_attn = torch.cat([v_attn, v_closest, v_global], dim=2)

        xyz = index_points(xyz, knn_idx) # B, nQ, k, 3
        xyz = torch.cat([xyz, closest_seen], dim=2) # B, nQ, k+nn_seen, 3

        d = xyz_q[:, :, None] - xyz # b, nQ, k+nn_seen, 3

        pos_encode = self.fc_delta(d)  # b x nQ x k+nn_seen x dim
        pos_encode = torch.cat([pos_encode, torch.zeros([b, nQ, 1, self.dim], device=pos_encode.device)], dim=2)  # b, nQ, k+nn_seen+1, dim

        attn = self.fc_gamma(q_attn[:, :, None] - k_attn + pos_encode)
        attn = functional.softmax(attn, dim=-2)  # b x nQ x k+nn_seen+1 x dim
        res = torch.einsum('bmnf,bmnf->bmf', attn, v_attn + pos_encode)  # b x nQ x dim
 
        return res
    
class CrossTransformerBlock_NoFine(nn.Module):
    def __init__(self, dim_inp, dim, nneigh=5, args=None):
        super().__init__()

        self.dim = dim
        self.nneigh = nneigh

        self.fc_delta = nn.Sequential(
            nn.Linear(3, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

        self.fc_gamma = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
        self.w_k_global = nn.Linear(dim_inp, dim, bias=False)
        self.w_v_global = nn.Linear(dim_inp, dim, bias=False)

        self.w_qs = nn.Linear(dim_inp, dim, bias=False)
        self.w_ks = nn.Linear(dim_inp, dim, bias=False)
        self.w_vs = nn.Linear(dim_inp, dim, bias=False)

    def forward(self, xyz_q, lat_rep, xyz, points):
        
        with torch.no_grad():
            dists = square_distance(xyz_q, xyz)
            
            sort_dist, sort_idx = dists.sort()
            knn_dist = sort_dist[:,:,:self.nneigh] # B, nQ, k
            knn_idx = sort_idx[:,:,:self.nneigh] # B, nQ, k

        b, nQ, _ = xyz_q.shape

        if len(lat_rep.shape) == 2:
            q_attn = self.w_qs(lat_rep).unsqueeze(1).repeat(1, nQ, 1)
            k_global = self.w_k_global(lat_rep).unsqueeze(1).repeat(1, nQ, 1).unsqueeze(2)
            v_global = self.w_v_global(lat_rep).unsqueeze(1).repeat(1, nQ, 1).unsqueeze(2)
        else:
            q_attn = self.w_qs(lat_rep)
            k_global = self.w_k_global(lat_rep).unsqueeze(2)
            v_global = self.w_v_global(lat_rep).unsqueeze(2)

        k_attn = index_points(self.w_ks(points),knn_idx)  # b, nQ, k, dim 
        k_attn = torch.cat([k_attn, k_global], dim=2)
        v_attn = index_points(self.w_vs(points), knn_idx) 
        v_attn = torch.cat([v_attn, v_global], dim=2)

        xyz = index_points(xyz, knn_idx) # B, nQ, k, 3

        d = xyz_q[:, :, None] - xyz # b, nQ, k+nn_seen, 3

        pos_encode = self.fc_delta(d)  # b x nQ x k+nn_seen x dim
        pos_encode = torch.cat([pos_encode, torch.zeros([b, nQ, 1, self.dim], device=pos_encode.device)], dim=2)  # b, nQ, k+nn_seen+1, dim

        attn = self.fc_gamma(q_attn[:, :, None] - k_attn + pos_encode)
        attn = functional.softmax(attn, dim=-2)  # b x nQ x k+nn_seen+1 x dim
        res = torch.einsum('bmnf,bmnf->bmf', attn, v_attn + pos_encode)  # b x nQ x dim
 
        return res


class ResnetBlockFC(nn.Module):
    ''' Fully connected ResNet Block class.
    Copied from https://github.com/autonomousvision/convolutional_occupancy_networks

    Args:
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
    '''

    def __init__(self, size_in, size_out=None, size_h=None):
        super().__init__()
        # Attributes
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)
        self.actvn = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(size_in, size_out, bias=False)
        # Initialization
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x):
        net = self.fc_0(self.actvn(x))
        dx = self.fc_1(self.actvn(net))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx

class FeatureAggregator(nn.Module):
    """
    Attributes:
        dim_inp int: dimensionality of encoding (global and local latent vectors)
        dim int: internal dimensionality
        nneigh int: number of nearest anchor points to draw information from
        hidden_dim int: hidden dimensionality of final feed-forward network
        n_blocks int: number of blocks in feed forward network
    """
    def __init__(self, dim_inp=512, dim=512, nneigh=5, hidden_dim=512, n_blocks=5, timestep=False, args = None):
        super().__init__()
        self.dim = dim
        self.n_blocks = n_blocks

        self.args = args

        if args.no_fine == 1:
            self.ct1 = CrossTransformerBlock_NoFine(dim_inp, dim, nneigh=nneigh, args = args)
        else:
            self.ct1 = CrossTransformerBlock(dim_inp, dim, nneigh=nneigh, args = args)
            self.fc_s = nn.Linear(3, dim) # RGB -> dim

        self.blocks = nn.ModuleList([
            ResnetBlockFC(hidden_dim) for i in range(n_blocks)
        ])

        self.fc_c = nn.ModuleList([
            nn.Linear(dim, hidden_dim) for i in range(n_blocks)
        ])

        self.pe = positional_encoding()
        add_channel = 1 if timestep else 0
        self.fc_p = nn.Linear(60 + add_channel, hidden_dim)

        if timestep:
            self.pe_t = positional_encoding_t()
            # self.out_norm = nn.GroupNorm(2, 512)
            self.timestep_emb = nn.Sequential(
                nn.Sigmoid(),
                nn.Linear(20, 2),
            )
    
    def sample_grid_features(self, xyz_q, seen_xyz, valid_seen_xyz, grid_feat):

        k = self.args.nn_seen
        xyz_reso = self.args.xyz_size

        # Sample from ViT features
        seen_xyz[~valid_seen_xyz] = 9999
        B, H, W, _ = seen_xyz.size()
        seen_xyz = seen_xyz.reshape(B,-1, 3)

        with torch.no_grad():
            dists = square_distance(xyz_q, seen_xyz)

            sort_dist, sort_idx = dists.sort()
            knn_dist = sort_dist[:,:,:k] # B, nQ, 1
            knn_idx = sort_idx[:,:,:k] # B, nQ, 1

            row = knn_idx.div(xyz_reso, rounding_mode="floor").unsqueeze(-1) # B, nQ, k, 1
            col = knn_idx.remainder(xyz_reso).unsqueeze(-1) # B, nQ, k, 1
            indices_xy = torch.cat([col, row], dim=-1) / 112 #[0,1] # B, nQ, k, 2
            indices_xy = (indices_xy - 0.5) * 2 #[-1, 1] # B, nQ, k, 2

        knn_xyz = index_points(seen_xyz, knn_idx) # B, nQ, k, 3

        out_rgb = F.grid_sample(grid_feat['rgb'], indices_xy, padding_mode='border', align_corners=True, mode='bilinear')
        out_rgb = out_rgb.permute(0,2,3,1) # B, nQ, k, D
        out = self.fc_s(out_rgb)

        return out, knn_xyz


    def forward(self, xyz_q, seen_xyz, valid_seen_xyz, fea, up_grid_fea, timestep=None, self_cond=None, custom_centers=None):
        """
        :param xyz_q [B x n_queries x 3]: queried 3D coordinates
        :param lat_rep [B x dim_inp]: global latent vectors
        :param xyz [B x n_anchors x 3]: anchor positions
        :param feats [B x n_anchros x dim_inp]: local latent vectors
        :return: occ [B x n_queries]: occupancy probability for each queried 3D coordinate
        """

        lat_rep = fea['global_feats']

        if custom_centers is not None:
            xyz = custom_centers
        else:
            xyz = fea['anchors_xyz']

        feats = fea['anchors_feats']

        B = xyz_q.size()[0]
        
        if self.args.no_fine == 1:
            lat_rep = self.ct1(xyz_q, lat_rep, xyz, feats)
        else:
            grid_feat = up_grid_fea # B, 512, 14*scale ,14*scale
            sampled_grid_feat, nn_seen_loc = self.sample_grid_features(xyz_q, seen_xyz.clone(), valid_seen_xyz, grid_feat)
            lat_rep = self.ct1(xyz_q, lat_rep, xyz, feats, sampled_grid_feat, nn_seen_loc)

        p = self.pe(xyz_q)
        if timestep != None:
            p = torch.cat([p, self_cond], dim=-1)
        net = self.fc_p(p)
        
        # print(p.shape, xyz_q.shape, net.shape)
        
        if timestep != None:
            timestep_emb = self.pe_t(timestep)
            # print(timestep_emb.shape)
            emb_out = self.timestep_emb(timestep_emb)
            scale, shift = torch.chunk(emb_out, 2, dim=2)
            # print(scale.shape, shift.shape, net.shape)
            net = net * (1 + scale) + shift

        for i in range(self.n_blocks):
            net = net + self.fc_c[i](lat_rep)
            net = self.blocks[i](net)

        return net