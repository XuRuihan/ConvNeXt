import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
import math


class SpatialConv2d(nn.Module):
    def __init__(self, dim, kernel_size, hidden_ratio=1.0):
        super().__init__()
        hidden_dim = int(hidden_ratio * dim)
        assert kernel_size % 2 == 1
        padding = kernel_size // 2
        self.conv_h = nn.Conv2d(dim, hidden_dim, (kernel_size, 1), padding=(padding, 0), groups=dim)
        self.conv_w = nn.Conv2d(hidden_dim, dim, (1, kernel_size), padding=(0, padding), groups=dim)
        self.kernel_size = kernel_size
        self.dim = dim

    def forward(self, x):
        x = self.conv_h(x)
        x = self.conv_w(x)
        return x

    def extra_repr(self):
        return f"dim={self.dim}, kernel_size={self.kernel_size}"


class OversizeConv2d(nn.Module):
    def __init__(self, dim, kernel_size, bias=True):
        super().__init__()
        self.conv_h = nn.Conv2d(dim, dim, (kernel_size, 1), groups=dim, bias=bias)
        self.conv_w = nn.Conv2d(dim, dim, (1, kernel_size), groups=dim, bias=bias)

        self.dim = dim
        self.kernel_size = kernel_size

    def get_instance_kernel(self, instance_kernel_size):
        h_weight = F.interpolate(self.conv_h.weight, [instance_kernel_size[0], 1], mode='bilinear', align_corners=True)
        w_weight = F.interpolate(self.conv_w.weight, [1, instance_kernel_size[1]], mode='bilinear', align_corners=True)
        return h_weight, w_weight

    def forward(self, x):
        H, W = x.shape[-2:]
        instance_kernel_size = 2 * H - 1, 2 * W - 1
        h_weight, w_weight = self.get_instance_kernel(instance_kernel_size)

        padding = H - 1, W - 1
        x = F.conv2d(x, h_weight, self.conv_h.bias, padding=(padding[0], 0), groups=self.dim)
        x = F.conv2d(x, w_weight, self.conv_w.bias, padding=(0, padding[1]), groups=self.dim)
        return x

    def extra_repr(self):
        return f"dim={self.dim}, kernel_size={self.kernel_size}"


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features // 2, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x1, x2 = torch.chunk(x, 2, 1)
        x = x1 * self.act(x2)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class LKC(nn.Module):
    def __init__(self, dim, kernel_size, ratio=0.5):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        ratio = 0 if ratio < 0 else ratio
        ratio = 1 if ratio > 1 else ratio
        global_dim = round(ratio * dim)
        local_dim = dim - global_dim

        if local_dim > 0:
            self.local_conv = SpatialConv2d(local_dim, 5)
        if global_dim > 0:
            # self.global_conv = FourierConv2d(global_dim, 2 * img_size)
            self.global_conv = OversizeConv2d(global_dim, kernel_size)
        self.conv1 = nn.Conv2d(dim, dim, 1)

        self.local_dim = local_dim
        self.global_dim = global_dim

    def forward(self, x):
        attn = self.conv0(x)

        if self.local_dim > 0 and self.global_dim > 0:
            local_attn, global_attn = torch.split(attn, [self.local_dim, self.global_dim], dim=1)
            local_attn = self.local_conv(local_attn)
            global_attn = self.global_conv(global_attn)
            attn = torch.cat([local_attn, global_attn], dim=1)
        elif self.local_dim > 0:
            attn = self.local_conv(attn)
        elif self.global_dim > 0:
            attn = self.global_conv(attn)
        attn = self.conv1(attn)

        return attn


class Attention(nn.Module):
    def __init__(self, d_model, kernel_size, ratio=0.5):
        super().__init__()

        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.lkc = LKC(d_model, kernel_size, ratio)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        shorcut = x.clone()

        attn = self.lkc(x)
        x = self.proj_1(x)
        x = self.activation(x) * attn
        x = self.proj_2(x)
        x = x + shorcut
        return x


class Block(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, drop=0.0, drop_path=0.0, act_layer=nn.GELU, kernel_size=32, ratio=0.5):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.attn = Attention(dim, kernel_size, ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = nn.BatchNorm2d(dim)
        mlp_ratio = mlp_ratio // 4 * 5
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        layer_scale_init_value = 1e-2
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True
        )
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = x + self.drop_path(
            self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(self.norm1(x))
        )
        x = x + self.drop_path(
            self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x))
        )
        return x


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=(patch_size[0] // 2, patch_size[1] // 2),
        )
        self.norm = nn.BatchNorm2d(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = self.norm(x)
        return x, H, W


class ParCNet(nn.Module):
    def __init__(
        self,
        img_size=224,
        in_chans=3,
        num_classes=1000,
        embed_dims=[64, 128, 256, 512],
        mlp_ratios=[4, 4, 4, 4],
        drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        depths=[3, 4, 6, 3],
        num_stages=4,
        flag=False,
        kernel_size=[128, 64, 32, 16],
        ratio=[0.25, 0.5, 0.75, 1],
    ):
        super().__init__()
        if flag == False:
            self.num_classes = num_classes
        self.depths = depths
        self.num_stages = num_stages

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule
        cur = 0

        for i in range(num_stages):
            patch_embed = OverlapPatchEmbed(
                img_size=img_size // (2 ** (i + 2)),
                patch_size=7 if i == 0 else 3,
                stride=4 if i == 0 else 2,
                in_chans=in_chans if i == 0 else embed_dims[i - 1],
                embed_dim=embed_dims[i],
            )

            block = nn.ModuleList(
                [
                    Block(
                        dim=embed_dims[i],
                        mlp_ratio=mlp_ratios[i],
                        drop=drop_rate,
                        drop_path=dpr[cur + j],
                        kernel_size=kernel_size[i],
                        ratio=ratio[i],
                    )
                    for j in range(depths[i])
                ]
            )
            norm = norm_layer(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

        # classification head
        self.head = (
            nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {
            "pos_embed1",
            "pos_embed2",
            "pos_embed3",
            "pos_embed4",
            "cls_token",
        }  # has pos_embed may be better

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=""):
        self.num_classes = num_classes
        self.head = (
            nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

    def forward_features(self, x):
        B = x.shape[0]

        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x, H, W = patch_embed(x)
            for blk in block:
                x = blk(x)
            x = x.flatten(2).transpose(1, 2)
            x = norm(x)
            if i != self.num_stages - 1:
                x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        return x.mean(dim=1)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)

        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x


def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if "patch_embed.proj.weight" in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v

    return out_dict


model_urls = {
    "parcnet_v2_b0": "https://huggingface.co/Visual-Attention-Network/ParCNet-Tiny-original/resolve/main/parcnet_v2_tiny_754.pth.tar",
    "parcnet_v2_b1": "https://huggingface.co/Visual-Attention-Network/ParCNet-Small-original/resolve/main/parcnet_v2_small_811.pth.tar",
    "parcnet_v2_b2": "https://huggingface.co/Visual-Attention-Network/ParCNet-Base-original/resolve/main/parcnet_v2_base_828.pth.tar",
    "parcnet_v2_b3": "https://huggingface.co/Visual-Attention-Network/ParCNet-Large-original/resolve/main/parcnet_v2_large_839.pth.tar",
}


def load_model_weights(model, arch, kwargs):
    url = model_urls[arch]
    checkpoint = torch.hub.load_state_dict_from_url(
        url=url, map_location="cpu", check_hash=True
    )
    strict = True
    if "num_classes" in kwargs and kwargs["num_classes"] != 1000:
        strict = False
        del checkpoint["state_dict"]["head.weight"]
        del checkpoint["state_dict"]["head.bias"]
    model.load_state_dict(checkpoint["state_dict"], strict=strict)
    return model


@register_model
def parcnet_v2_b0(pretrained=False, **kwargs):
    model = ParCNet(
        embed_dims=[32, 64, 160, 256],
        mlp_ratios=[8, 8, 4, 4],
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=[3, 3, 5, 2],
        **kwargs,
    )
    model.default_cfg = _cfg()
    if pretrained:
        model = load_model_weights(model, "parcnet_v2_b0", kwargs)
    return model


@register_model
def parcnet_v2_b1(pretrained=False, **kwargs):
    model = ParCNet(
        embed_dims=[64, 128, 320, 512],
        mlp_ratios=[8, 8, 4, 4],
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=[2, 2, 4, 2],
        **kwargs,
    )
    model.default_cfg = _cfg()
    if pretrained:
        model = load_model_weights(model, "parcnet_v2_b1", kwargs)
    return model


@register_model
def parcnet_v2_b2(pretrained=False, **kwargs):
    model = ParCNet(
        embed_dims=[64, 128, 320, 512],
        mlp_ratios=[8, 8, 4, 4],
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=[3, 3, 12, 3],
        **kwargs,
    )
    model.default_cfg = _cfg()
    if pretrained:
        model = load_model_weights(model, "parcnet_v2_b2", kwargs)
    return model


@register_model
def parcnet_v2_b3(pretrained=False, **kwargs):
    model = ParCNet(
        embed_dims=[64, 128, 320, 512],
        mlp_ratios=[8, 8, 4, 4],
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=[3, 5, 27, 3],
        **kwargs,
    )
    model.default_cfg = _cfg()
    if pretrained:
        model = load_model_weights(model, "parcnet_v2_b3", kwargs)
    return model


@register_model
def parcnet_v2_b4(pretrained=False, **kwargs):
    model = ParCNet(
        embed_dims=[64, 128, 320, 512],
        mlp_ratios=[8, 8, 4, 4],
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=[3, 6, 40, 3],
        **kwargs,
    )
    model.default_cfg = _cfg()
    if pretrained:
        model = load_model_weights(model, "parcnet_v2_b4", kwargs)
    return model


@register_model
def parcnet_v2_b5(pretrained=False, **kwargs):
    model = ParCNet(
        embed_dims=[96, 192, 480, 768],
        mlp_ratios=[8, 8, 4, 4],
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=[3, 3, 24, 3],
        **kwargs,
    )
    model.default_cfg = _cfg()
    if pretrained:
        model = load_model_weights(model, "parcnet_v2_b5", kwargs)
    return model


@register_model
def parcnet_v2_b6(pretrained=False, **kwargs):
    model = ParCNet(
        embed_dims=[96, 192, 384, 768],
        mlp_ratios=[8, 8, 4, 4],
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=[6, 6, 90, 6],
        **kwargs,
    )
    model.default_cfg = _cfg()
    if pretrained:
        model = load_model_weights(model, "parcnet_v2_b6", kwargs)
    return model
