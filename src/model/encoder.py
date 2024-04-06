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
from timm.models.vision_transformer import PatchEmbed, Block
from util.pos_embed import get_2d_sincos_pos_embed
from src.layers import XYZPosEmbed
import torch.nn.functional as F

class MCCEncoder(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self,
                 img_size=224, patch_size=16, in_chans=3,
                 embed_dim=768, depth=12, num_heads=12,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, args=None):
        super().__init__()

        drop_path = 0 if args == None else args.drop_path
        # --------------------------------------------------------------------------
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_token_xyz = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.xyz_pos_embed = XYZPosEmbed(embed_dim)

        self.blocks = nn.ModuleList([
            Block(
                embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer,
                drop_path=drop_path
            ) for i in range(depth)])

        self.blocks_xyz = nn.ModuleList([
            Block(
                embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer,
                drop_path=drop_path
            ) for i in range(depth)])

        self.norm = norm_layer(embed_dim)
        self.norm_xyz = norm_layer(embed_dim)
        self.cached_enc_feat = None

        self.initialize_weights()

    def initialize_weights(self):

        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.cls_token_xyz, std=.02)

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

    def forward(self, seen_images, seen_xyz, valid_seen_xyz, up_grid_bypass=None):

        if up_grid_bypass is not None:
            fine_fea_rgb = up_grid_bypass # HR resolution
        else:
            fine_fea_rgb = F.interpolate(seen_images, scale_factor=0.5, mode='bilinear') # B, 3, 112, 112
        
        # get tokens
        x = self.patch_embed(seen_images)
        x = x + self.pos_embed[:, 1:, :]
        y = self.xyz_pos_embed(seen_xyz, valid_seen_xyz)

        ##### forward E_XYZ #####
        # append cls token
        cls_token_xyz = self.cls_token_xyz
        cls_tokens_xyz = cls_token_xyz.expand(y.shape[0], -1, -1)

        y = torch.cat((cls_tokens_xyz, y), dim=1)
        # apply Transformer blocks
        for blk in self.blocks_xyz:
            y = blk(y)
        y = self.norm_xyz(y)

        ##### forward E_RGB #####
        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)

        x = torch.cat((cls_tokens, x), dim=1)
        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        # combine encodings
        xy = torch.cat([x, y], dim=2) # B x (PatchPatch) x 2dim

        fine_fea = {}
        fine_fea['rgb'] = fine_fea_rgb

        return xy, fine_fea