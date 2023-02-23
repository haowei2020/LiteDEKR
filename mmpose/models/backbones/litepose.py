# 参考litepose, https://github.com/mit-han-lab/litepose
import torch
import logging
import torch.nn as nn
import random

from mmcv.cnn import constant_init, kaiming_init
from ..builder import BACKBONES
from .utils import load_checkpoint, make_divisible
# from mmpose.models.backbones.utils import load_checkpoint, make_divisible


def rand(c):
    return random.randint(0, c - 1)


class convbnrelu(nn.Sequential):
    def __init__(self, inp, oup, ker=3, stride=1, groups=1):
        super(convbnrelu, self).__init__(
            nn.Conv2d(inp, oup, ker, stride, ker // 2, groups=groups, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU6(inplace=True)
        )
        

class densedownconvolution(nn.Sequential):
    def __init__(self, inp, oup, ker=3, stride=2, groups=1):
        
        super(densedownconvolution, self).__init__(
            nn.Conv2d(inp, oup//stride**2, ker, 1, ker // 2, groups=groups, bias=False),
            nn.BatchNorm2d(oup//stride**2),
            nn.ReLU6(inplace=True),
            nn.PixelUnshuffle(stride)
        )


class SepConv2d(nn.Module):
    def __init__(self, inp, oup, ker=3, stride=1):
        super(SepConv2d, self).__init__()
        conv = [
            nn.Conv2d(inp, inp, ker, stride, ker // 2, groups=inp, bias=False),
            nn.BatchNorm2d(inp),
            nn.ReLU(inplace=True),
            nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        ]
        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        output = self.conv(x)
        return output


class GCBlock(nn.Module):

    def __init__(self, inplanes, planes, pool, fusions):
        super(GCBlock, self).__init__()
        assert pool in ['avg', 'att']
        assert all([f in ['channel_add', 'channel_mul'] for f in fusions])
        assert len(fusions) > 0, 'at least one fusion should be used'
        self.inplanes = inplanes
        self.planes = planes
        self.pool = pool
        self.fusions = fusions
        if 'att' in pool:
            self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if 'channel_add' in fusions:
            self.channel_add_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1)
            )
        else:
            self.channel_add_conv = None
        if 'channel_mul' in fusions:
            self.channel_mul_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1)
            )
        else:
            self.channel_mul_conv = None
      
    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        if self.pool == 'att':
            input_x = x
            # [N, C, H * W]
            input_x = input_x.view(batch, channel, height * width)
            # [N, 1, C, H * W]
            input_x = input_x.unsqueeze(1)
            # [N, 1, H, W]
            context_mask = self.conv_mask(x)
            # [N, 1, H * W]
            context_mask = context_mask.view(batch, 1, height * width)
            # [N, 1, H * W]
            context_mask = self.softmax(context_mask)
            # [N, 1, H * W, 1]
            context_mask = context_mask.unsqueeze(3)
            # [N, 1, C, 1]
            context = torch.matmul(input_x, context_mask)
            # [N, C, 1, 1]
            context = context.view(batch, channel, 1, 1)
        else:
            # [N, C, 1, 1]
            context = self.avg_pool(x)

        return context

    def forward(self, x):
        # [N, C, 1, 1]
        context = self.spatial_pool(x)

        if self.channel_mul_conv is not None:
            # [N, C, 1, 1]
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
            out = x * channel_mul_term
        else:
            out = x
        if self.channel_add_conv is not None:
            # [N, C, 1, 1]
            channel_add_term = self.channel_add_conv(context)
            out = out + channel_add_term
        return out


class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super(ChannelShuffle,self).__init__()
        self.groups = groups
    
    def forward(self,x):
        batchsize, num_channels, height, width = x.size()
        channels_per_group = num_channels // self.groups
        # reshape
        x = x.view(batchsize, self.groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        # flatten
        x = x.view(batchsize, -1, height, width)
        return x


class InvBottleneck(nn.Module):

    def __init__(self, inplanes, planes, stride=1, ker=3, exp=6):
        super(InvBottleneck, self).__init__()
        feature_dim = make_divisible(round(inplanes * exp), 8)
        self.inv = nn.Sequential(
            nn.Conv2d(inplanes, feature_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU6(inplace = True)
        )
        self.depth_conv = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, ker, stride, ker // 2, groups=feature_dim, bias=False),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU6(inplace = True)
        )
        self.point_conv = nn.Sequential(
            nn.Conv2d(feature_dim, planes, 1, 1, 0, bias=False),
            nn.BatchNorm2d(planes)
        )
        self.stride = stride
        self.use_residual_connection = stride == 1 and inplanes == planes
        
    def forward(self, x):
        out = self.inv(x)
        out = self.depth_conv(out)
        out = self.point_conv(out)
        if self.use_residual_connection:
            out += x
        return out


class InvBottleneckV2(nn.Module):

    def __init__(self, inplanes, planes, stride=1, ker=3, exp=6):
        super(InvBottleneckV2, self).__init__()
        feature_dim = make_divisible(round(inplanes * exp), 8)
        self.inv = nn.Sequential(
            nn.Conv2d(inplanes, feature_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU6(inplace = True)
        )
        # self.depth_conv = nn.Sequential(
        #     nn.Conv2d(feature_dim, feature_dim, ker, stride, ker // 2, groups=feature_dim, bias=False),
        #     nn.BatchNorm2d(feature_dim),
        #     nn.ReLU6(inplace = True)
        # )
        self.depth_conv = convbnrelu(feature_dim, feature_dim, ker, stride, feature_dim) \
            if stride==1 \
            else densedownconvolution(feature_dim, feature_dim, ker, stride, feature_dim//stride**2)
        
        
        self.point_conv = nn.Sequential(
            nn.Conv2d(feature_dim, planes, 1, 1, 0, bias=False),
            nn.BatchNorm2d(planes)
        )
        self.stride = stride
        self.use_residual_connection = stride == 1 and inplanes == planes
        
    def forward(self, x):
        out = self.inv(x)
        out = self.depth_conv(out)
        out = self.point_conv(out)
        if self.use_residual_connection:
            out += x
        return out


class InvBottleneckV3(nn.Module):

    def __init__(self, inplanes, planes, stride=1, ker=3, exp=6):
        super(InvBottleneckV3, self).__init__()
        feature_dim = make_divisible(round(inplanes * exp), 8)
        self.inv = nn.Sequential(
            nn.Conv2d(inplanes, feature_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU6(inplace = True)
        )
        self.depth_conv = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, ker, stride, ker // 2, groups=feature_dim, bias=False),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU6(inplace = True)
        )
        # self.depth_conv = convbnrelu(feature_dim, feature_dim, ker, stride, feature_dim) \
        #     if stride==1 \
        #     else densedownconvolution(feature_dim, feature_dim, ker, stride, feature_dim//stride**2)
        # self.channel_shuffle = ChannelShuffle(exp)
        
        self.point_conv = nn.Sequential(
            nn.Conv2d(feature_dim, planes, 1, 1, 0, bias=False),
            nn.BatchNorm2d(planes)
        )
        self.stride = stride
        self.use_residual_connection = stride == 1 and inplanes == planes
        self.gcb = GCBlock(planes, planes, 'att', ['channel_add'])
        
    def forward(self, x):
        out = self.inv(x)
        out = self.depth_conv(out)
        # out = self.channel_shuffle(out)
        out = self.point_conv(out)
        out = self.gcb(out)
        if self.use_residual_connection:
            out += x
        return out


@BACKBONES.register_module()
class LitePose(nn.Module):
    '''LitePose backbone'''
    def __init__(self, cfg_arch, width_mult=1.0, round_nearest=8, frozen_stages=-1, norm_eval=False):
        super(LitePose, self).__init__()
        self.norm_eval = norm_eval
        self.frozen_stages = frozen_stages
        backbone_setting = cfg_arch['backbone_setting']
        input_channel = cfg_arch['input_channel']
        # building first layer
        input_channel = make_divisible(input_channel * width_mult, round_nearest)
        self.first = nn.Sequential(
            convbnrelu(3, 32, ker=3, stride=2),
            convbnrelu(32, 32, ker=3, stride=1, groups=32),
            nn.Conv2d(32, input_channel, 1, 1, 0, bias=False),
            nn.BatchNorm2d(input_channel)
        )
        self.channel = [input_channel]
        # building inverted residual blocks
        self.stage = []
        for id_stage in range(len(backbone_setting)):
            n = backbone_setting[id_stage]['num_blocks']
            s = backbone_setting[id_stage]['stride']
            c = backbone_setting[id_stage]['channel']
            c = make_divisible(c * width_mult, round_nearest)
            block_setting = backbone_setting[id_stage]['block_setting']
            layer = []
            for id_block in range(n):
                t, k = block_setting[id_block]
                stride = s if id_block == 0 else 1
                layer.append(InvBottleneck(input_channel, c, stride, ker=k, exp=t))
                input_channel = c
            layer = nn.Sequential(*layer)
            self.stage.append(layer)
            self.channel.append(c)
        self.stage = nn.ModuleList(self.stage)


    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None')


    def forward(self, x):
        x = self.first(x)
        x_list = [x]
        for i in range(len(self.stage)):
            tmp = self.stage[i](x_list[-1])
            x_list.append(tmp)
        
        return x_list
    

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            for param in self.first.parameters():
                param.requires_grad = False
        for i in range(1, self.frozen_stages + 1):
            layer = getattr(self, f'stage')[i-1]
            layer.eval()
            for param in layer.parameters():
                param.requires_grad = False


    def train(self, mode=True):
        super().train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()


@BACKBONES.register_module()
class LitePoseV2(nn.Module):
    '''LitePoseV2 backbone'''
    def __init__(self, cfg_arch, width_mult=1.0, round_nearest=8, frozen_stages=-1, norm_eval=False):
        super(LitePoseV2, self).__init__()
        self.norm_eval = norm_eval
        self.frozen_stages = frozen_stages
        backbone_setting = cfg_arch['backbone_setting']
        input_channel = cfg_arch['input_channel']
        # building first layer
        input_channel = make_divisible(input_channel * width_mult, round_nearest)
        self.first = nn.Sequential(
            # convbnrelu(3, 32, ker=3, stride=2),
            densedownconvolution(3, 32, ker=3, stride=2),
            convbnrelu(32, 32, ker=3, stride=1, groups=32),
            nn.Conv2d(32, input_channel, 1, 1, 0, bias=False),
            nn.BatchNorm2d(input_channel)
        )
        self.channel = [input_channel]
        # building inverted residual blocks
        self.stage = []
        for id_stage in range(len(backbone_setting)):
            n = backbone_setting[id_stage]['num_blocks']
            s = backbone_setting[id_stage]['stride']
            c = backbone_setting[id_stage]['channel']
            c = make_divisible(c * width_mult, round_nearest)
            block_setting = backbone_setting[id_stage]['block_setting']
            layer = []
            for id_block in range(n):
                t, k = block_setting[id_block]
                stride = s if id_block == 0 else 1
                layer.append(InvBottleneckV2(input_channel, c, stride, ker=k, exp=t))
                input_channel = c
            layer = nn.Sequential(*layer)
            self.stage.append(layer)
            self.channel.append(c)
        self.stage = nn.ModuleList(self.stage)


    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None')


    def forward(self, x):
        x = self.first(x)
        x_list = [x]
        for i in range(len(self.stage)):
            tmp = self.stage[i](x_list[-1])
            x_list.append(tmp)
        
        return x_list
    

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            for param in self.first.parameters():
                param.requires_grad = False
        for i in range(1, self.frozen_stages + 1):
            layer = getattr(self, f'stage')[i-1]
            layer.eval()
            for param in layer.parameters():
                param.requires_grad = False


    def train(self, mode=True):
        super().train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()


@BACKBONES.register_module()
class LitePoseV3(nn.Module):
    '''LitePoseV3 backbone'''
    def __init__(self, cfg_arch, width_mult=1.0, round_nearest=8, frozen_stages=-1, norm_eval=False):
        super(LitePoseV3, self).__init__()
        self.norm_eval = norm_eval
        self.frozen_stages = frozen_stages
        backbone_setting = cfg_arch['backbone_setting']
        input_channel = cfg_arch['input_channel']
        # building first layer
        input_channel = make_divisible(input_channel * width_mult, round_nearest)
        self.first = nn.Sequential(
            convbnrelu(3, 32, ker=3, stride=2),
            convbnrelu(32, 32, ker=3, stride=1, groups=32),
            nn.Conv2d(32, input_channel, 1, 1, 0, bias=False),
            nn.BatchNorm2d(input_channel)
        )
        self.channel = [input_channel]
        # building inverted residual blocks
        self.stage = []
        for id_stage in range(len(backbone_setting)):
            n = backbone_setting[id_stage]['num_blocks']
            s = backbone_setting[id_stage]['stride']
            c = backbone_setting[id_stage]['channel']
            c = make_divisible(c * width_mult, round_nearest)
            block_setting = backbone_setting[id_stage]['block_setting']
            layer = []
            for id_block in range(n):
                t, k = block_setting[id_block]
                stride = s if id_block == 0 else 1
                layer.append(InvBottleneckV3(input_channel, c, stride, ker=k, exp=t))
                input_channel = c
            layer = nn.Sequential(*layer)
            self.stage.append(layer)
            self.channel.append(c)
        self.stage = nn.ModuleList(self.stage)


    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None')


    def forward(self, x):
        x = self.first(x)
        x_list = [x]
        for i in range(len(self.stage)):
            tmp = self.stage[i](x_list[-1])
            x_list.append(tmp)
        
        return x_list
    

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            for param in self.first.parameters():
                param.requires_grad = False
        for i in range(1, self.frozen_stages + 1):
            layer = getattr(self, f'stage')[i-1]
            layer.eval()
            for param in layer.parameters():
                param.requires_grad = False


    def train(self, mode=True):
        super().train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()


# if __name__ == '__main__':
#     cfg = dict(
#         input_channel=16,
#         backbone_setting=[
#             dict(
#                 num_blocks=6, 
#                 stride=2, 
#                 channel=16, 
#                 block_setting=[
#                     [6, 7], 
#                     [6, 7], 
#                     [6, 7], 
#                     [6, 7], 
#                     [6, 7], 
#                     [6, 7],
#                 ],
#             ),
#             dict(
#                 num_blocks=8, 
#                 stride=2, 
#                 channel=32, 
#                 block_setting=[
#                     [6, 7], 
#                     [6, 7], 
#                     [6, 7], 
#                     [6, 7], 
#                     [6, 7], 
#                     [6, 7],
#                     [6, 7], 
#                     [6, 7],
#                 ],
#             ),
#             dict(
#                 num_blocks=10, 
#                 stride=2, 
#                 channel=48, 
#                 block_setting=[
#                     [6, 7], 
#                     [6, 7], 
#                     [6, 7], 
#                     [6, 7], 
#                     [6, 7], 
#                     [6, 7], 
#                     [6, 7], 
#                     [6, 7], 
#                     [6, 7], 
#                     [6, 7],
#                 ],
#             ),
#             dict(
#                 num_blocks=10, 
#                 stride=1, 
#                 channel=80, 
#                 block_setting=[
#                     [6, 7], 
#                     [6, 7], 
#                     [6, 7], 
#                     [6, 7], 
#                     [6, 7], 
#                     [6, 7], 
#                     [6, 7], 
#                     [6, 7], 
#                     [6, 7], 
#                     [6, 7],
#                 ],
#             ),
#         ]
#     )
#     model = LitePoseV3(cfg)
#     inputs = torch.randn(2,3,256,256)
#     out = model(inputs)
#     for i in range(len(out)):
#         print(f'shape of out {i}:', out[i].shape)
    # shape of out 0: torch.Size([2, 16, 112, 112])
    # shape of out 1: torch.Size([2, 32, 56, 56])
    # shape of out 2: torch.Size([2, 48, 28, 28])
    # shape of out 3: torch.Size([2, 80, 28, 28])