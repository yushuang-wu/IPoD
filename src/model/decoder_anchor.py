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
from util.pos_embed import get_2d_sincos_pos_embed
from src.layers import DecoderBlockCenters

class DecoderPredictCenters(nn.Module):
    def __init__(self,
                 img_size=224, patch_size=16,
                 embed_dim=768,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, args=None):
        super().__init__()


        self.M = 200 if args is None else args.n_groups
        self.device = args.device if args is not None else 'cuda'
        drop_path = 0 if args == None else args.drop_path
        self.num_patches = int(img_size/patch_size)**2
        self.decoder_embed = nn.Linear(
            2*embed_dim,
            decoder_embed_dim,
            bias=True
        )

        self.init_embedding = nn.Embedding(self.M, decoder_embed_dim)
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding
        self.decoder_blocks = nn.ModuleList([
            DecoderBlockCenters(
                decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer,
                drop_path=drop_path,
                args=args,
            ) for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, decoder_embed_dim+3, bias=True)
        self.initialize_weights()

    def initialize_weights(self):
        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x): 
        B = x.size()[0]
        fea = {}

        # Initialize centers
        input = torch.ones(B, self.M, device=self.device) * torch.linspace(0, self.M-1, steps=self.M, device=self.device)
        init_embedding = self.init_embedding(input.int()) # B, M, d

        # embed tokens
        y = self.decoder_embed(x)

        init_global = y[:,0,:].unsqueeze(1)
        init_embedding = init_embedding + init_global

        # Pos embed input vit tokens
        y = y + self.decoder_pos_embed

        # 3D pos embed
        y = torch.cat([y, init_embedding], dim=1)

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            y = blk(y)

        y = self.decoder_norm(y)
        center_fea = self.decoder_pred(y[:,-self.M:, :])

        fea['enc_feats'] = x
        fea['global_feats'] = y[:,0,:]
        fea['anchors_xyz'] = center_fea[:, :, :3]
        fea['anchors_feats'] = center_fea[:,:,3:]

        return fea