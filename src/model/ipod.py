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

import inspect
import random
import matplotlib.pyplot as plt

import torch.nn as nn
from src.fns import *
from src.model.encoder import MCCEncoder
from src.model.decoder_anchor import DecoderPredictCenters
from src.model.decoder_feature import FeatureAggregator
import torch
import torch.nn.functional as F

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
# from diffusers.schedulers.scheduling_ddim import DDIMScheduler
# from diffusers.schedulers.scheduling_pndm import PNDMScheduler

from pytorch3d.ops import knn_points
from pytorch3d.structures import Pointclouds
from pytorch3d.vis.plotly_vis import plot_scene

class IPoD_transfomer(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, args=None):
        super().__init__()

        self.args = args
        self.encoder = MCCEncoder(args=args)
        self.decoderl1 = DecoderPredictCenters(args=args)
        self.decoderl2 = FeatureAggregator(nneigh=args.nneigh, args=args)
        self.decoderl3 = FeatureAggregator(nneigh=args.nneigh, args=args, timestep=True)

        self.fc_out2 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(512, 1 + 256*3)
        )
        
        self.fc_out3 = nn.Sequential(
            SharedMLP(512, 128),
            nn.Dropout(p=0.1, inplace=False),
            nn.Conv1d(128, 3, kernel_size=(1,), stride=(1,))
        )
        
        self.scale = 3.0
        
        ddpm_dict = {
            'beta_start': 1e-5,  # 0.00085
            'beta_end': 8e-3,  # 0.012
            'beta_schedule': 'linear'  # 'custom'
        }
        self.scheduler = DDPMScheduler(**ddpm_dict, clip_sample=False)
        
        self.num_inference_steps = 1000
        
    def index_select(self, x, idx):
        num_ptx = x.shape[1]
        B, N, _ = idx.shape
        line_idx = torch.Tensor([i*num_ptx for i in range(B)]).long().to(self.args.device)
        idx = idx.reshape((-1,)) + line_idx[:, None].repeat(1, N).reshape((-1,))
        x = x.reshape((-1, 3))
        y = x[idx]
        return y.reshape((B, -1, 3))
        
    def forward(self, seen_images, seen_xyz, query_xyz, valid_seen_xyz, gt_xyz, gt_rgb):
        
        B, N, _ = gt_xyz.shape
        random_idx = random.sample(list(range(N)), self.args.n_queries)
        
        x_0 = gt_xyz[:, random_idx, :] # [B, 4096, 3]
        noise = torch.randn_like(x_0)
        timestep = torch.randint(0, self.num_inference_steps, (B,), device=self.args.device, dtype=torch.long)
        x_t = self.scheduler.add_noise(x_0, noise, timestep)
        # x_t = torch.clamp(x_t, min=-3., max=3.)
        # x_t = shrink_points_beyond_threshold(x_t, self.args.shrink_threshold)
        
        knn = knn_points(x_t, gt_xyz, K=1)
        knn_dists, knn_idx = knn[0]**0.5, knn[1]
        label_udf = torch.clamp(knn_dists, min=0, max=0.5)
        # print(gt_rgb.shape, knn_idx.shape, label_udf.shape)
        # label_rgb = self.index_select(gt_rgb, knn_idx)
        labels_01 = (knn_dists < self.args.train_dist_threshold).squeeze(-1)
        label_rgb = torch.zeros_like(x_t)
        label_rgb[labels_01] = torch.gather(gt_rgb, 1, knn_idx.repeat(1, 1, 3))[labels_01]
        label = [label_udf, label_rgb]
        
        seen_images = preprocess_img(seen_images)
        seen_xyz = shrink_points_beyond_threshold(seen_xyz, self.args.shrink_threshold)
        
        with torch.cuda.amp.autocast():
            latent, up_grid_fea = self.encoder(seen_images, seen_xyz, valid_seen_xyz, up_grid_bypass=None)
            
        fea = self.decoderl1(latent)
        
        net2 = self.decoderl2(x_t, seen_xyz, valid_seen_xyz, fea, up_grid_fea)
        out2 = self.fc_out2(net2) # udf prediction
        
        ### self condition ###
        fea_detach = {k:v.detach() for k,v in fea.items()}
        up_grid_fea_detach = {k:v.detach() for k,v in up_grid_fea.items()}
        
        pred_udf = F.relu(out2[:,:,:1])
        pred_udf = torch.clamp(pred_udf, max=0.5).detach()
        self_cond = pred_udf
        # print(pred_udf.shape, x_t.shape) # B, N, 1
        
        net3 = self.decoderl3(x_t, seen_xyz, valid_seen_xyz, fea_detach, up_grid_fea_detach, timestep, self_cond)
        out3 = self.fc_out3(net3) # noise prediction
        out3 = out3.permute(0, 2, 1)
        
        loss_noise = F.mse_loss(out3, noise)#*0.05
        
        return out2, label, loss_noise, fea['anchors_xyz']

