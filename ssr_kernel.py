'''
Modifed from Visual Attention Network
https://github.com/Visual-Attention-Network/VAN-Classification.git
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
import math
from einops.layers.torch import Rearrange



class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 7, 1, 3, bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
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
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x




class LKA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 7, 1, 3, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, 1, 9, dilation=3, groups=dim)
        self.conv1 = nn.Conv2d(dim, dim, 1)


    def forward(self, x):
        u = x.clone()        
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)

        return u * attn


class Attention(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = LKA(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x


class Block(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0.,drop_path=0., act_layer=nn.GELU):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.attn = Attention(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = nn.BatchNorm2d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        layer_scale_init_value = 1e-2            
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
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
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x)))
        return x


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=32, stride=4, padding=14, in_chans=3, embed_dim=32):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.ConvTranspose2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=padding)
        self.norm = nn.BatchNorm2d(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
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
        return x


class SSRKernel(nn.Module):
    def __init__(self, img_size=224, in_chans=3, num_classes=1, embed_dims=[64],
                mlp_ratios=[8], drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm,
                 depths=[12], num_stages=1, flag=False):
        super().__init__()
        if flag == False:
            self.num_classes = num_classes
        self.depths = depths
        self.num_stages = num_stages

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        for i in range(num_stages):

            block = nn.ModuleList([Block(
                dim=embed_dims[i], mlp_ratio=mlp_ratios[i], drop=drop_rate, drop_path=dpr[cur + j])
                for j in range(depths[i])])
            norm = norm_layer(embed_dims[i])
            cur += depths[i]

            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
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

    def forward_features(self, x):
        B = x.shape[0]
        for i in range(self.num_stages):
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            B, C, H, W = x.shape
            for blk in block:
                x = blk(x)
            x = x.flatten(2).transpose(1, 2)
            x = norm(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        return x

    def forward(self, x):
        x = self.forward_features(x)
        return x



class SSR(nn.Module):
    def __init__(self,
                 in_channel=3,
                 norm=nn.BatchNorm2d,
                 conv=nn.Conv2d,
                 seq=nn.Sequential,
                 kernel=SSRKernel):
        super().__init__()
        r = 1
        ch1, ch2 = 32 * r, 16 * r
        ind, oud = ch1, ch2
        d1, d2, d3, d4 = [ch1, ch1, ch1, ch1]
        p1, p2, p3, p4 = [2, 2, 2, 2]
        m1, m2, m3, m4 = [4, 4, 4, 4]

        self.c1 = kernel(embed_dims=[d1], mlp_ratios=[m1], depths=[p1])
        self.c2 = kernel(embed_dims=[d2], mlp_ratios=[m2], depths=[p2])
        self.c3 = kernel(embed_dims=[d3], mlp_ratios=[m3], depths=[p3])
        self.c4 = kernel(embed_dims=[d4], mlp_ratios=[m4], depths=[p4])

        self.proj_conv = OverlapPatchEmbed(in_chans=3, embed_dim=oud)
        self.act = nn.GELU()
        self.re = Rearrange('b (g c) h w -> b (c g) h w', g=2)

        self.ds1 = seq(conv(ind, oud, 1), norm(oud))
        self.ds2 = seq(conv(ind, oud, 1), norm(oud))
        self.ds3 = seq(conv(ind, oud, 1), norm(oud))
        self.ds4 = seq(conv(ind, oud, 1), norm(oud))

        self.cl1, self.cr1 = conv(ind, oud, 1), conv(ind, oud, 1)
        self.cl2, self.cr2 = conv(ind, oud, 1), conv(ind, oud, 1)
        self.cl3, self.cr3 = conv(ind, oud, 1), conv(ind, oud, 1)
        self.cl4, self.cr4 = conv(ind, oud, 1), conv(ind, oud, 1)

        self.ln1, self.rn1 = norm(oud), norm(oud)
        self.ln2, self.rn2 = norm(oud), norm(oud)
        self.ln3, self.rn3 = norm(oud), norm(oud)
        self.ln4, self.rn4 = norm(oud), norm(oud)

        self.out = nn.ModuleList([conv(oud, 3, 3, 1, 1) for i in range(10)])

    def forward(self, lr_left, lr_right):
        out_list = []
        l0 = self.proj_conv(lr_left)
        r0 = self.proj_conv(lr_right)
        out_list.extend([self.out[0](l0), self.out[1](l0)])

        f = self.act(self.re(torch.cat([l0, r0], 1)))
        f = self.ds1(self.act(f + self.c1(f)))
        l1 = l0 + self.ln1(self.cl1(self.re(self.act(torch.cat([l0, f], 1)))))
        r1 = r0 + self.rn1(self.cr1(self.re(self.act(torch.cat([r0, f], 1)))))
        out_list.extend([self.out[2](l1), self.out[3](l1)])

        f = self.act(self.re(torch.cat([l1, r1], 1)))
        f = self.ds2(self.act(f + self.c2(f)))
        l2 = l1 + self.ln2(self.cl2(self.re(self.act(torch.cat([l1, f], 1)))))
        r2 = r1 + self.rn2(self.cr2(self.re(self.act(torch.cat([r1, f], 1)))))
        out_list.extend([self.out[4](l2), self.out[5](l2)])

        f = self.act(self.re(torch.cat([l2, r2], 1)))
        f = self.ds3(self.act(f + self.c3(f)))
        l3 = l2 + self.ln3(self.cl3(self.re(self.act(torch.cat([l2, f], 1)))))
        r3 = r2 + self.rn3(self.cr3(self.re(self.act(torch.cat([r2, f], 1)))))
        out_list.extend([self.out[6](l3), self.out[7](l3)])

        f = self.act(self.re(torch.cat([l3, r3], 1)))
        f = self.ds4(self.act(f + self.c4(f)))
        l4 = l3 + self.ln4(self.cl4(self.re(self.act(torch.cat([l3, f], 1)))))
        r4 = r3 + self.rn4(self.cr4(self.re(self.act(torch.cat([r3, f], 1)))))
        out_list.extend([self.out[8](l4), self.out[9](l4)])

        return out_list
