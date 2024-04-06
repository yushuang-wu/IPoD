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

from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import PatchEmbed, Block, Mlp, DropPath
from util.pos_embed import get_2d_sincos_pos_embed
from src.fns import *
import math


class Conv2d(torch.nn.Conv2d):
    """
    A wrapper around :class:`torch.nn.Conv2d` to support empty inputs and more features.
    """

    def __init__(self, *args, **kwargs):
        """
        Extra keyword arguments supported in addition to those in `torch.nn.Conv2d`:

        Args:
            norm (nn.Module, optional): a normalization layer
            activation (callable(Tensor) -> Tensor): a callable activation function

        It assumes that norm layer is used before activation.
        """
        norm = kwargs.pop("norm", None)
        activation = kwargs.pop("activation", None)
        super().__init__(*args, **kwargs)

        self.norm = norm
        self.activation = activation

    def forward(self, x):
        # torchscript does not support SyncBatchNorm yet
        # https://github.com/pytorch/pytorch/issues/40507
        # and we skip these codes in torchscript since:
        # 1. currently we only support torchscript in evaluation mode
        # 2. features needed by exporting module to torchscript are added in PyTorch 1.6 or
        # later version, `Conv2d` in these PyTorch versions has already supported empty inputs.

        x = F.conv2d(
            x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x

class LayerNorm(nn.Module):
    """
    A LayerNorm variant, popularized by Transformers, that performs point-wise mean and
    variance normalization over the channel dimension for inputs that have shape
    (batch_size, channels, height, width).
    https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa B950
    """

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class positional_encoding(object):
    ''' Positional Encoding (presented in NeRF)
    Args:
        basis_function (str): basis function
    '''
    def __init__(self, basis_function='sin_cos'):
        super().__init__()
        self.func = basis_function

        L = 10
        freq_bands = 2.**(np.linspace(0, L-1, L))
        self.freq_bands = freq_bands * math.pi

    def __call__(self, p):
        if self.func == 'sin_cos':
            out = []
            p = p / 3.0 # change to the range [-1, 1]
            for freq in self.freq_bands:
                out.append(torch.sin(freq * p))
                out.append(torch.cos(freq * p))
            p = torch.cat(out, dim=2)
        return p
    
class positional_encoding_t(object):
    ''' Positional Encoding (presented in NeRF)
    Args:
        basis_function (str): basis function
    '''
    def __init__(self, basis_function='sin_cos'):
        super().__init__()
        self.func = basis_function

        L = 10
        freq_bands = 2.**(np.linspace(0, L-1, L))
        self.freq_bands = freq_bands * math.pi

    def __call__(self, p):
        if self.func == 'sin_cos':
            out = []
            p = p / 1000 # change to the range [-1, 1]
            p = p[:, None, None]
            for freq in self.freq_bands:
                out.append(torch.sin(freq * p))
                out.append(torch.cos(freq * p))
            p = torch.cat(out, dim=2)
        return p

class positional_encoding2(object):
    ''' Positional Encoding (presented in NeRF)
    Args:
        basis_function (str): basis function
    '''
    def __init__(self, basis_function='sin_cos'):
        super().__init__()
        self.func = basis_function

        L = 10
        freq_bands = 2.**(np.linspace(0, L-1, L))
        self.freq_bands = freq_bands * math.pi

    def __call__(self, p):
        if self.func == 'sin_cos':
            out = []
            p = p / 6.0 # change to the range [-1, 1]
            for freq in self.freq_bands:
                out.append(torch.sin(freq * p))
                out.append(torch.cos(freq * p))
            p = torch.cat(out, dim=2)
        return p

class XYZPosEmbedPE(nn.Module):

    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.pe = positional_encoding()
        self.pos_embed = nn.Linear(60, embed_dim)

    def forward(self, unseen_xyz):
        unseen_xyz = self.pe(unseen_xyz)
        return self.pos_embed(unseen_xyz)

class XYZPosEmbed(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, embed_dim):
        super().__init__()

        self.pe=None
        self.embed_dim = embed_dim
        self.two_d_pos_embed = nn.Parameter(
            torch.zeros(1, 64 + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.win_size = 8
        self.pos_embed = nn.Linear(3, embed_dim)

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads=12, mlp_ratio=2.0, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))
            for _ in range(1)
        ])

        self.invalid_xyz_token = nn.Parameter(torch.zeros(embed_dim,))
        self.initialize_weights()

    def initialize_weights(self):
        torch.nn.init.normal_(self.cls_token, std=.02)

        two_d_pos_embed = get_2d_sincos_pos_embed(self.two_d_pos_embed.shape[-1], self.win_size, cls_token=True)
        self.two_d_pos_embed.data.copy_(torch.from_numpy(two_d_pos_embed).float().unsqueeze(0))

        torch.nn.init.normal_(self.invalid_xyz_token, std=.02)

    def forward(self, seen_xyz, valid_seen_xyz):

        emb = self.pos_embed(seen_xyz)

        emb[~valid_seen_xyz] = 0.0
        emb[~valid_seen_xyz] += self.invalid_xyz_token

        B, H, W, C = emb.shape
        emb = emb.view(B, H // self.win_size, self.win_size, W // self.win_size, self.win_size, C)
        emb = emb.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, self.win_size * self.win_size, C)


        emb = emb + self.two_d_pos_embed[:, 1:, :]
        cls_token = self.cls_token + self.two_d_pos_embed[:, :1, :]

        cls_tokens = cls_token.expand(emb.shape[0], -1, -1)
        emb = torch.cat((cls_tokens, emb), dim=1)

        for _, blk in enumerate(self.blocks):
            emb = blk(emb)
        return emb[:, 0].view(B, (H // self.win_size) * (W // self.win_size), -1)
    
class DecodeXYZPosEmbed(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.pos_embed = nn.Linear(3, embed_dim)

    def forward(self, unseen_xyz):
        return self.pos_embed(unseen_xyz)
    

class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma

class MCCDecoderAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., args=None):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.args = args
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, unseen_size):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale

        mask = torch.zeros((1, 1, N, N), device=attn.device)
        mask[:, :, :, -unseen_size:] = float('-inf')
        for i in range(unseen_size):
            mask[:, :, -(i + 1), -(i + 1)] = 0
        attn = attn + mask
        attn = attn.softmax(dim=-1)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class MCCDecoderBlock(nn.Module):

    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., init_values=None,
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, args=None):
        super().__init__()
        self.args = args
        self.norm1 = norm_layer(dim)
        self.attn = MCCDecoderAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, args=args)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, unseen_size):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), unseen_size)))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x

class DecoderAttentionCenters(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., args=None):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.args = args
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class DecoderBlockCenters(nn.Module):
    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., init_values=None,
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, args=None):
        super().__init__()
        self.args = args
        self.norm1 = norm_layer(dim)
        self.attn = DecoderAttentionCenters(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, args=args)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x
    
