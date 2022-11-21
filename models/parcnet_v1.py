# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
from torch import Tensor
from typing import Optional

# channel wise attention
class CA_layer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CA_layer, self).__init__()
        # global average pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel//reduction, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(channel//reduction),
            nn.Hardswish(),
            nn.Conv2d(channel//reduction, channel, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(channel),
            nn.Hardsigmoid()
        )

    def forward(self, x):
        y = self.fc(self.gap(x))
        return x*y.expand_as(x)

# global circular conv based pure ConvNet Meta-Former block
# channel wise attention is used
class gcc_ca_mf_block(nn.Module):
    def __init__(self,
                 dim: int,
                 meta_kernel_size: int,
                 instance_kernel_method='interpolation_bilinear',
                 use_pe:Optional[bool]=True,
                 mid_mix: Optional[bool]=False,
                 bias: Optional[bool]=True,
                 mlp_ratio: Optional[int]=8,
                 ffn_dropout=0.0,
                 dropout=0.1):

        super().__init__()

        # spatial part,
        self.pre_Norm_1 = nn.BatchNorm2d(num_features=dim)
        self.pre_Norm_2 = nn.BatchNorm2d(num_features=dim)

        self.meta_kernel_1_H = nn.Conv2d(dim, dim, (meta_kernel_size, 1), groups=dim).weight
        self.meta_kernel_1_W = nn.Conv2d(dim, dim, (1, meta_kernel_size), groups=dim).weight
        self.meta_kernel_2_H = nn.Conv2d(dim, dim, (meta_kernel_size, 1), groups=dim).weight
        self.meta_kernel_2_W = nn.Conv2d(dim, dim, (1, meta_kernel_size), groups=dim).weight

        if bias:
            self.meta_1_H_bias = nn.Parameter(torch.randn(dim))
            self.meta_1_W_bias = nn.Parameter(torch.randn(dim))
            self.meta_2_H_bias = nn.Parameter(torch.randn(dim))
            self.meta_2_W_bias = nn.Parameter(torch.randn(dim))
        else:
            self.register_parameter('meta_1_H_bias', None)
            self.register_parameter('meta_1_W_bias', None)
            self.register_parameter('meta_2_H_bias', None)
            self.register_parameter('meta_2_W_bias', None)

        self.instance_kernel_method = instance_kernel_method

        if use_pe:
            self.meta_pe_1_H = nn.Parameter(torch.randn(1, dim, meta_kernel_size, 1))
            self.meta_pe_1_W = nn.Parameter(torch.randn(1, dim, 1, meta_kernel_size))
            self.meta_pe_2_H = nn.Parameter(torch.randn(1, dim, meta_kernel_size, 1))
            self.meta_pe_2_W = nn.Parameter(torch.randn(1, dim, 1, meta_kernel_size))

        if mid_mix:
            self.mixer = nn.ChannelShuffle(groups=2)

        self.mid_mix = mid_mix
        self.use_pe = use_pe
        self.dim = dim

        # channel part
        ffn_dim = mlp_ratio * dim
        self.ffn = nn.Sequential(
            nn.BatchNorm2d(num_features=2*dim),
            nn.Conv2d(2*dim, ffn_dim, kernel_size=(1, 1), bias=True),
            nn.Hardswish(),
            nn.Dropout(p=ffn_dropout),
            nn.Conv2d(ffn_dim, 2*dim, kernel_size=(1, 1), bias=True),
            nn.Dropout(p=dropout)
        )

        self.ca = CA_layer(channel=2*dim)

    def get_instance_kernel(self, instance_kernel_size):
        if self.instance_kernel_method == 'crop':
            return self.meta_kernel_1_H[:, :, : instance_kernel_size,:], \
                   self.meta_kernel_1_W[:, :, :, :instance_kernel_size], \
                   self.meta_kernel_2_H[:, :, :instance_kernel_size, :], \
                   self.meta_kernel_2_W[:, :, :, :instance_kernel_size]

        elif self.instance_kernel_method == 'interpolation_bilinear':
            H_shape = [instance_kernel_size, 1]
            W_shape = [1, instance_kernel_size]
            return F.interpolate(self.meta_kernel_1_H, H_shape, mode='bilinear', align_corners=True), \
                   F.interpolate(self.meta_kernel_1_W, W_shape, mode='bilinear', align_corners=True), \
                   F.interpolate(self.meta_kernel_2_H, H_shape, mode='bilinear', align_corners=True), \
                   F.interpolate(self.meta_kernel_2_W, W_shape, mode='bilinear', align_corners=True),

        else:
            print('{} is not supported!'.format(self.instance_kernel_method))

    def get_instance_pe(self, instance_kernel_size):
        if self.instance_kernel_method == 'crop':
            return self.meta_pe_1_H[:, :, :instance_kernel_size, :]\
                       .expand(1, self.dim, instance_kernel_size, instance_kernel_size), \
                   self.meta_pe_1_W[:, :, :, :instance_kernel_size]\
                       .expand(1, self.dim, instance_kernel_size, instance_kernel_size), \
                   self.meta_pe_2_H[:, :, :instance_kernel_size, :]\
                       .expand(1, self.dim, instance_kernel_size, instance_kernel_size), \
                   self.meta_pe_2_W[:, :, :, :instance_kernel_size]\
                       .expand(1, self.dim, instance_kernel_size, instance_kernel_size)

        elif self.instance_kernel_method == 'interpolation_bilinear':
            return F.interpolate(self.meta_pe_1_H, [instance_kernel_size, 1], mode='bilinear', align_corners=True)\
                       .expand(1, self.dim, instance_kernel_size, instance_kernel_size), \
                   F.interpolate(self.meta_pe_1_W, [1, instance_kernel_size], mode='bilinear', align_corners=True)\
                       .expand(1, self.dim, instance_kernel_size, instance_kernel_size), \
                   F.interpolate(self.meta_pe_2_H, [instance_kernel_size, 1], mode='bilinear', align_corners=True)\
                       .expand(1, self.dim, instance_kernel_size, instance_kernel_size), \
                   F.interpolate(self.meta_pe_2_W, [1, instance_kernel_size], mode='bilinear', align_corners=True)\
                       .expand(1, self.dim, instance_kernel_size, instance_kernel_size)
        else:
            print('{} is not supported!'.format(self.instance_kernel_method))

    def forward(self, x: Tensor) -> Tensor:

        x_1, x_2 = torch.chunk(x, 2, 1)
        x_1_res, x_2_res = x_1, x_2
        _, _, f_s, _ = x_1.shape

        K_1_H, K_1_W, K_2_H, K_2_W = self.get_instance_kernel(f_s)

        if self.use_pe:
            pe_1_H, pe_1_W, pe_2_H, pe_2_W = self.get_instance_pe(f_s)

        # **************************************************************************************************sptial part
        # pre norm
        if self.use_pe:
            x_1, x_2 = x_1 + pe_1_H, x_2 + pe_1_W

        x_1, x_2 = self.pre_Norm_1(x_1), self.pre_Norm_2(x_2)

        # stage 1
        x_1_1 = F.conv2d(torch.cat((x_1, x_1[:, :, :-1, :]), dim=2), weight=K_1_H, bias=self.meta_1_H_bias, padding=0,
                         groups=self.dim)
        x_2_1 = F.conv2d(torch.cat((x_2, x_2[:, :, :, :-1]), dim=3), weight=K_1_W, bias=self.meta_1_W_bias, padding=0,
                         groups=self.dim)
        if self.mid_mix:
            mid_rep = torch.cat((x_1_1, x_2_1), dim=1)
            x_1_1, x_2_1 = torch.chunk(self.mixer(mid_rep), chunks=2, dim=1)

        if self.use_pe:
            x_1_1, x_2_1 = x_1_1 + pe_2_W, x_2_1 + pe_2_H

        # stage 2
        x_1_2 = F.conv2d(torch.cat((x_1_1, x_1_1[:, :, :, :-1]), dim=3), weight=K_2_W, bias=self.meta_2_W_bias,
                         padding=0, groups=self.dim)
        x_2_2 = F.conv2d(torch.cat((x_2_1, x_2_1[:, :, :-1, :]), dim=2), weight=K_2_H, bias=self.meta_2_H_bias,
                         padding=0, groups=self.dim)

        # residual
        x_1 = x_1_res + x_1_2
        x_2 = x_2_res + x_2_2

        # *************************************************************************************************channel part
        x_ffn = torch.cat((x_1, x_2), dim=1)
        x_ffn = x_ffn + self.ca(self.ffn(x_ffn))

        return x_ffn

class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

class ConvNeXt(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, in_chans=3, num_classes=1000, 
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0., 
                 layer_scale_init_value=1e-6, head_init_scale=1.,
                 meta_kernel_sizes=[64, 32, 16, 8],
                 ):
        super().__init__()

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            if i <= 1:
                stage = nn.Sequential(
                    *[Block(dim=dims[i], drop_path=dp_rates[cur + j], 
                    layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
                )
            else:
                stage = nn.Sequential(
                    *[gcc_ca_mf_block(dim=dims[i] // 2, meta_kernel_size=meta_kernel_sizes[i]) for j in range(depths[i])]
                )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1])) # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


model_urls = {
    "convnext_tiny_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth",
    "convnext_small_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth",
    "convnext_base_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth",
    "convnext_large_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth",
    "convnext_tiny_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_224.pth",
    "convnext_small_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_22k_224.pth",
    "convnext_base_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth",
    "convnext_large_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth",
    "convnext_xlarge_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_224.pth",
}

@register_model
def parcnet_v1_xt(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 9, 3], dims=[48, 96, 192, 384], **kwargs)
    return model
