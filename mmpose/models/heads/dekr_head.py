# Copyright (c) OpenMMLab. All rights reserved.
import copy

import torch
import torch.nn as nn
from mmcv.cnn import (ConvModule, build_activation_layer, build_conv_layer,
                      build_norm_layer, constant_init, normal_init)

from mmpose.models.builder import build_loss
from ..backbones.resnet import BasicBlock
from ..builder import HEADS
from .deconv_head import DeconvHead

try:
    from mmcv.ops import DeformConv2d
    has_mmcv_full = True
except (ImportError, ModuleNotFoundError):
    has_mmcv_full = False


class AdaptiveActivationBlock(nn.Module):
    """Adaptive activation convolution block. "Bottom-up human pose estimation
    via disentangled keypoint regression", CVPR'2021.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        groups (int): Number of groups. Generally equal to the
            number of joints.
        norm_cfg (dict): Config for normalization layers.
        act_cfg (dict): Config for activation layers.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 groups=1,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU')):

        super(AdaptiveActivationBlock, self).__init__()

        assert in_channels % groups == 0 and out_channels % groups == 0
        self.groups = groups

        regular_matrix = torch.tensor([[-1, -1, -1, 0, 0, 0, 1, 1, 1],
                                       [-1, 0, 1, -1, 0, 1, -1, 0, 1],
                                       [1, 1, 1, 1, 1, 1, 1, 1, 1]])
        self.register_buffer('regular_matrix', regular_matrix.float())

        self.transform_matrix_conv = build_conv_layer(
            dict(type='Conv2d'),
            in_channels=in_channels,
            out_channels=6 * groups,
            kernel_size=3,
            padding=1,
            groups=groups,
            bias=True)

        if has_mmcv_full:
            self.adapt_conv = DeformConv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
                groups=groups,
                deform_groups=groups)
        else:
            raise ImportError('Please install the full version of mmcv '
                              'to use `DeformConv2d`.')

        self.norm = build_norm_layer(norm_cfg, out_channels)[1]
        self.act = build_activation_layer(act_cfg)

    def forward(self, x):
        B, _, H, W = x.size()
        residual = x

        affine_matrix = self.transform_matrix_conv(x)
        affine_matrix = affine_matrix.permute(0, 2, 3, 1).contiguous()
        affine_matrix = affine_matrix.view(B, H, W, self.groups, 2, 3)
        offset = torch.matmul(affine_matrix, self.regular_matrix)
        offset = offset.transpose(4, 5).reshape(B, H, W, self.groups * 18)
        offset = offset.permute(0, 3, 1, 2).contiguous()

        x = self.adapt_conv(x, offset)
        x = self.norm(x)
        x = self.act(x + residual)

        return x


class MSA(nn.Module):
    """multi-head self-attention (MSA) module with relative position bias.

    Args:
        embed_dims (int): Number of input channels.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to q, k, v.
            Default: True.
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        attn_drop_rate (float, optional): Dropout ratio of attention weight.
            Default: 0.0
        proj_drop_rate (float, optional): Dropout ratio of output. Default: 0.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 num_joints,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop_rate=0.,
                 proj_drop_rate=0.,
        ):

        super().__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_joints = num_joints
        head_embed_dims = embed_dims // num_heads
        self.scale = qk_scale or head_embed_dims**-0.5

        self.qkv = nn.Linear(embed_dims, embed_dims * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(proj_drop_rate)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        Args:
            x (tensor): input features with shape of (B, C, H, W)
        """
        B, C, H, W = x.shape
        _C = C//self.num_joints
        x = x.permute(0,2,3,1).reshape(B*H*W, self.num_joints, _C).contiguous()
        
        qkv = self.qkv(x).reshape(B*H*W, self.num_joints, 3, self.num_heads,
                                  _C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B*H*W, self.num_joints, _C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        x = x.reshape(B, H, W, self.num_joints*_C).permute(0, 3, 1,2).contiguous()
        return x

    @staticmethod
    def double_step_seq(step1, len1, step2, len2):
        seq1 = torch.arange(0, step1 * len1, step1)
        seq2 = torch.arange(0, step2 * len2, step2)
        return (seq1[:, None] + seq2[None, :]).reshape(1, -1)


@HEADS.register_module()
class DEKRHead(DeconvHead):
    """DisEntangled Keypoint Regression head. "Bottom-up human pose estimation
    via disentangled keypoint regression", CVPR'2021.

    Args:
        in_channels (int): Number of input channels.
        num_joints (int): Number of joints.
        num_heatmap_filters (int): Number of filters for heatmap branch.
        num_offset_filters_per_joint (int): Number of filters for each joint.
        in_index (int|Sequence[int]): Input feature index. Default: 0
        input_transform (str|None): Transformation type of input features.
            Options: 'resize_concat', 'multiple_select', None.
            Default: None.

            - 'resize_concat': Multiple feature maps will be resized to the
                same size as the first one and then concat together.
                Usually used in FCN head of HRNet.
            - 'multiple_select': Multiple feature maps will be bundle into
                a list and passed into decode head.
            - None: Only one select feature map is allowed.
        align_corners (bool): align_corners argument of F.interpolate.
            Default: False.
        heatmap_loss (dict): Config for heatmap loss. Default: None.
        offset_loss (dict): Config for offset loss. Default: None.
    """

    def __init__(self,
                 in_channels,
                 num_joints,
                 num_heatmap_filters=32,
                 num_offset_filters_per_joint=15,
                 in_index=0,
                 input_transform=None,
                 num_deconv_layers=0,
                 num_deconv_filters=None,
                 num_deconv_kernels=None,
                 extra=dict(final_conv_kernel=0),
                 align_corners=False,
                 heatmap_loss=None,
                 offset_loss=None):

        super().__init__(
            in_channels,
            out_channels=in_channels,
            num_deconv_layers=num_deconv_layers,
            num_deconv_filters=num_deconv_filters,
            num_deconv_kernels=num_deconv_kernels,
            align_corners=align_corners,
            in_index=in_index,
            input_transform=input_transform,
            extra=extra,
            loss_keypoint=heatmap_loss)

        # set up filters for heatmap
        self.heatmap_conv_layers = nn.Sequential(
            ConvModule(
                in_channels=self.in_channels,
                out_channels=num_heatmap_filters,
                kernel_size=1,
                norm_cfg=dict(type='BN')),
            BasicBlock(num_heatmap_filters, num_heatmap_filters),
            build_conv_layer(
                dict(type='Conv2d'),
                in_channels=num_heatmap_filters,
                out_channels=1 + num_joints,
                kernel_size=1))

        # set up filters for offset map
        groups = num_joints
        num_offset_filters = num_joints * num_offset_filters_per_joint

        self.offset_conv_layers = nn.Sequential(
            ConvModule(
                in_channels=self.in_channels,
                out_channels=num_offset_filters,
                kernel_size=1,
                norm_cfg=dict(type='BN')),
            AdaptiveActivationBlock(
                num_offset_filters, num_offset_filters, groups=groups),
            AdaptiveActivationBlock(
                num_offset_filters, num_offset_filters, groups=groups),
            build_conv_layer(
                dict(type='Conv2d'),
                in_channels=num_offset_filters,
                out_channels=2 * num_joints,
                kernel_size=1,
                groups=groups))

        # set up offset losses
        self.offset_loss = build_loss(copy.deepcopy(offset_loss))

    def get_loss(self, outputs, heatmaps, masks, offsets, offset_weights):
        """Calculate the dekr loss.

        Note:
            - batch_size: N
            - num_channels: C
            - num_joints: K
            - heatmaps height: H
            - heatmaps weight: W

        Args:
            outputs (List(torch.Tensor[N,C,H,W])): Multi-scale outputs.
            heatmaps (List(torch.Tensor[N,K+1,H,W])): Multi-scale heatmap
                targets.
            masks (List(torch.Tensor[N,K+1,H,W])): Weights of multi-scale
                heatmap targets.
            offsets (List(torch.Tensor[N,K*2,H,W])): Multi-scale offset
                targets.
            offset_weights (List(torch.Tensor[N,K*2,H,W])): Weights of
                multi-scale offset targets.
        """

        losses = dict()

        for idx in range(len(outputs)):
            pred_heatmap, pred_offset = outputs[idx]
            heatmap_weight = masks[idx].view(masks[idx].size(0),
                                             masks[idx].size(1), -1)
            losses['loss_hms'] = losses.get('loss_hms', 0) + self.loss(
                pred_heatmap, heatmaps[idx], heatmap_weight)
            losses['loss_ofs'] = losses.get('loss_ofs', 0) + self.offset_loss(
                pred_offset, offsets[idx], offset_weights[idx])

        return losses

    def forward(self, x):
        """Forward function."""
        x = self._transform_inputs(x)
        x = self.deconv_layers(x)
        x = self.final_layer(x)
        heatmap = self.heatmap_conv_layers(x)
        offset = self.offset_conv_layers(x)
        return [[heatmap, offset]]

    def init_weights(self):
        """Initialize model weights."""
        super().init_weights()
        for name, m in self.heatmap_conv_layers.named_modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)
        for name, m in self.offset_conv_layers.named_modules():
            if isinstance(m, nn.Conv2d):
                if 'transform_matrix_conv' in name:
                    normal_init(m, std=1e-8, bias=0)
                else:
                    normal_init(m, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)


@HEADS.register_module()
class DEKRHeadWithMSA(DeconvHead):
    """在DEKRHead基础上offset分支增加MSA, 通过use_msa和num_heads参数控制.

    Args:
        in_channels (int): Number of input channels.
        num_joints (int): Number of joints.
        num_heatmap_filters (int): Number of filters for heatmap branch.
        num_offset_filters_per_joint (int): Number of filters for each joint.
        in_index (int|Sequence[int]): Input feature index. Default: 0
        input_transform (str|None): Transformation type of input features.
            Options: 'resize_concat', 'multiple_select', None.
            Default: None.

            - 'resize_concat': Multiple feature maps will be resized to the
                same size as the first one and then concat together.
                Usually used in FCN head of HRNet.
            - 'multiple_select': Multiple feature maps will be bundle into
                a list and passed into decode head.
            - None: Only one select feature map is allowed.
        align_corners (bool): align_corners argument of F.interpolate.
            Default: False.
        heatmap_loss (dict): Config for heatmap loss. Default: None.
        offset_loss (dict): Config for offset loss. Default: None.
    """

    def __init__(self,
                 in_channels,
                 num_joints,
                 num_heatmap_filters=32,
                 num_offset_filters_per_joint=15,
                 in_index=0,
                 input_transform=None,
                 num_deconv_layers=0,
                 num_deconv_filters=None,
                 num_deconv_kernels=None,
                 extra=dict(final_conv_kernel=0),
                 align_corners=False,
                 heatmap_loss=None,
                 offset_loss=None,
                 use_msa=True,
                 num_heads=3):

        super().__init__(
            in_channels,
            out_channels=in_channels,
            num_deconv_layers=num_deconv_layers,
            num_deconv_filters=num_deconv_filters,
            num_deconv_kernels=num_deconv_kernels,
            align_corners=align_corners,
            in_index=in_index,
            input_transform=input_transform,
            extra=extra,
            loss_keypoint=heatmap_loss)

        # set up filters for heatmap
        self.heatmap_conv_layers = nn.Sequential(
            ConvModule(
                in_channels=self.in_channels,
                out_channels=num_heatmap_filters,
                kernel_size=1,
                norm_cfg=dict(type='BN')),
            BasicBlock(num_heatmap_filters, num_heatmap_filters),
            build_conv_layer(
                dict(type='Conv2d'),
                in_channels=num_heatmap_filters,
                out_channels=1 + num_joints,
                kernel_size=1))

        # set up filters for offset map
        groups = num_joints
        num_offset_filters = num_joints * num_offset_filters_per_joint

        if use_msa and num_heads>0:
            assert num_offset_filters_per_joint%num_heads == 0, 'num_offset_filters_per_joint is not divisible by num_heads'
            self.offset_conv_layers = nn.Sequential(
                ConvModule(
                    in_channels=self.in_channels,
                    out_channels=num_offset_filters,
                    kernel_size=1,
                    norm_cfg=dict(type='BN')),
                AdaptiveActivationBlock(
                    num_offset_filters, num_offset_filters, groups=groups),
                AdaptiveActivationBlock(
                    num_offset_filters, num_offset_filters, groups=groups),
                MSA(num_offset_filters_per_joint, num_heads, num_joints),
                build_conv_layer(
                    dict(type='Conv2d'),
                    in_channels=num_offset_filters,
                    out_channels=2 * num_joints,
                    kernel_size=1,
                    groups=groups
                )
            )
        else:
            self.offset_conv_layers = nn.Sequential(
                ConvModule(
                    in_channels=self.in_channels,
                    out_channels=num_offset_filters,
                    kernel_size=1,
                    norm_cfg=dict(type='BN')),
                AdaptiveActivationBlock(
                    num_offset_filters, num_offset_filters, groups=groups),
                AdaptiveActivationBlock(
                    num_offset_filters, num_offset_filters, groups=groups),
                build_conv_layer(
                    dict(type='Conv2d'),
                    in_channels=num_offset_filters,
                    out_channels=2 * num_joints,
                    kernel_size=1,
                    groups=groups))

        # set up offset losses
        self.offset_loss = build_loss(copy.deepcopy(offset_loss))

    def get_loss(self, outputs, heatmaps, masks, offsets, offset_weights):
        """Calculate the dekr loss.

        Note:
            - batch_size: N
            - num_channels: C
            - num_joints: K
            - heatmaps height: H
            - heatmaps weight: W

        Args:
            outputs (List(torch.Tensor[N,C,H,W])): Multi-scale outputs.
            heatmaps (List(torch.Tensor[N,K+1,H,W])): Multi-scale heatmap
                targets.
            masks (List(torch.Tensor[N,K+1,H,W])): Weights of multi-scale
                heatmap targets.
            offsets (List(torch.Tensor[N,K*2,H,W])): Multi-scale offset
                targets.
            offset_weights (List(torch.Tensor[N,K*2,H,W])): Weights of
                multi-scale offset targets.
        """

        losses = dict()

        for idx in range(len(outputs)):
            pred_heatmap, pred_offset = outputs[idx]
            heatmap_weight = masks[idx].view(masks[idx].size(0),
                                             masks[idx].size(1), -1)
            losses['loss_hms'] = losses.get('loss_hms', 0) + self.loss(
                pred_heatmap, heatmaps[idx], heatmap_weight)
            losses['loss_ofs'] = losses.get('loss_ofs', 0) + self.offset_loss(
                pred_offset, offsets[idx], offset_weights[idx])

        return losses

    def forward(self, x):
        """Forward function."""
        x = self._transform_inputs(x)
        x = self.deconv_layers(x)
        x = self.final_layer(x)
        heatmap = self.heatmap_conv_layers(x)
        offset = self.offset_conv_layers(x)
        return [[heatmap, offset]]

    def init_weights(self):
        """Initialize model weights."""
        super().init_weights()
        for name, m in self.heatmap_conv_layers.named_modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)
        for name, m in self.offset_conv_layers.named_modules():
            if isinstance(m, nn.Conv2d):
                if 'transform_matrix_conv' in name:
                    normal_init(m, std=1e-8, bias=0)
                else:
                    normal_init(m, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)


@HEADS.register_module()
class DEKRHeadWithRLELoss(DeconvHead):
    """在DEKRHead基础上修改, 
        1、为适配offset部分使用RLE loss, offset部分每个关键点需增加两个维度的输出, 代表方差.
        2、关键点热力图可选, use_keypoint_heatmaps=False时, test_cfg[refine_keypoint_scores_by_keypoint_heatmaps]需为False
        3、offset分支增加MSA操作, 增加关键点特征间的信息交换, 通过use_msa和num_heads参数决定是否使用

    Args:
        in_channels (int): Number of input channels.
        num_joints (int): Number of joints.
        num_heatmap_filters (int): Number of filters for heatmap branch.
        num_offset_filters_per_joint (int): Number of filters for each joint.
        in_index (int|Sequence[int]): Input feature index. Default: 0
        input_transform (str|None): Transformation type of input features.
            Options: 'resize_concat', 'multiple_select', None.
            Default: None.

            - 'resize_concat': Multiple feature maps will be resized to the
                same size as the first one and then concat together.
                Usually used in FCN head of HRNet.
            - 'multiple_select': Multiple feature maps will be bundle into
                a list and passed into decode head.
            - None: Only one select feature map is allowed.
        align_corners (bool): align_corners argument of F.interpolate.
            Default: False.
        heatmap_loss (dict): Config for heatmap loss. Default: None.
        offset_loss (dict): Config for offset loss. Default: None.
        use_keypoint_heatmaps (bool): 是否生成关键点热力图
        use_msa (bool): 是否在offset分支使用多头注意力
        num_heads (int): 当use_mas=True时, 设置该参数, 多头注意力头数, 需能整除num_offset_filters_per_joint
    """

    def __init__(self,
                 in_channels,
                 num_joints,
                 num_heatmap_filters=32,
                 num_offset_filters_per_joint=15,
                 in_index=0,
                 input_transform=None,
                 num_deconv_layers=0,
                 num_deconv_filters=None,
                 num_deconv_kernels=None,
                 extra=dict(final_conv_kernel=0),
                 align_corners=False,
                 heatmap_loss=None,
                 offset_loss=None,
                 use_keypoint_heatmaps=True,
                 use_msa=False,
                 num_heads=None,
    ):

        super().__init__(
            in_channels,
            out_channels=in_channels,
            num_deconv_layers=num_deconv_layers,
            num_deconv_filters=num_deconv_filters,
            num_deconv_kernels=num_deconv_kernels,
            align_corners=align_corners,
            in_index=in_index,
            input_transform=input_transform,
            extra=extra,
            loss_keypoint=heatmap_loss)
        self.use_keypoint_heatmaps = use_keypoint_heatmaps

        # set up filters for heatmap
        self.heatmap_conv_layers = nn.Sequential(
            ConvModule(
                in_channels=self.in_channels,
                out_channels=num_heatmap_filters,
                kernel_size=1,
                norm_cfg=dict(type='BN')),
            BasicBlock(num_heatmap_filters, num_heatmap_filters),
            build_conv_layer(
                dict(type='Conv2d'),
                in_channels=num_heatmap_filters,
                out_channels=1 + num_joints if use_keypoint_heatmaps else 1,
                kernel_size=1))

        # set up filters for offset map
        groups = num_joints
        num_offset_filters = num_joints * num_offset_filters_per_joint

        if use_msa and num_heads>0:
            assert num_offset_filters_per_joint%num_heads == 0, 'num_offset_filters_per_joint is not divisible by num_heads'
            self.offset_conv_layers = nn.Sequential(
                ConvModule(
                    in_channels=self.in_channels,
                    out_channels=num_offset_filters,
                    kernel_size=1,
                    norm_cfg=dict(type='BN')),
                AdaptiveActivationBlock(
                    num_offset_filters, num_offset_filters, groups=groups),
                AdaptiveActivationBlock(
                    num_offset_filters, num_offset_filters, groups=groups),
                MSA(num_offset_filters_per_joint, num_heads, num_joints),
                build_conv_layer(
                    dict(type='Conv2d'),
                    in_channels=num_offset_filters,
                    out_channels=4 * num_joints,
                    kernel_size=1,
                    groups=groups
                )
            )
        else:
            self.offset_conv_layers = nn.Sequential(
                ConvModule(
                    in_channels=self.in_channels,
                    out_channels=num_offset_filters,
                    kernel_size=1,
                    norm_cfg=dict(type='BN')),
                AdaptiveActivationBlock(
                    num_offset_filters, num_offset_filters, groups=groups),
                AdaptiveActivationBlock(
                    num_offset_filters, num_offset_filters, groups=groups),
                build_conv_layer(
                    dict(type='Conv2d'),
                    in_channels=num_offset_filters,
                    out_channels=4 * num_joints,
                    kernel_size=1,
                    groups=groups
                )
            )

        # set up offset losses
        self.offset_loss = build_loss(copy.deepcopy(offset_loss))

    def get_loss(self, outputs, heatmaps, masks, offsets, offset_weights):
        """Calculate the dekr loss.

        Note:
            - batch_size: N
            - num_channels: C
            - num_joints: K
            - heatmaps height: H
            - heatmaps weight: W

        Args:
            outputs (List(torch.Tensor[N,C,H,W])): Multi-scale outputs.
            heatmaps (List(torch.Tensor[N,K+1,H,W])): Multi-scale heatmap
                targets.
            masks (List(torch.Tensor[N,K+1,H,W])): Weights of multi-scale
                heatmap targets.
            offsets (List(torch.Tensor[N,K*2,H,W])): Multi-scale offset
                targets.
            offset_weights (List(torch.Tensor[N,K*2,H,W])): Weights of
                multi-scale offset targets.
        """
        if not self.use_keypoint_heatmaps:
            heatmaps = [x[:,0,:,:] for x in heatmaps]   # BottomUpGenerateHeatmapTarget生成的热力图第一层为中心点热力图，其余为关键点热力图
        losses = dict()

        for idx in range(len(outputs)):
            pred_heatmap, pred_offset = outputs[idx]
            heatmap_weight = masks[idx].view(masks[idx].size(0),
                                             masks[idx].size(1), -1)
            losses['loss_hms'] = losses.get('loss_hms', 0) + self.loss(
                pred_heatmap, heatmaps[idx], heatmap_weight)
            losses['loss_ofs'] = losses.get('loss_ofs', 0) + self.offset_loss(
                pred_offset, offsets[idx], offset_weights[idx])

        return losses

    def forward(self, x):
        """Forward function."""
        x = self._transform_inputs(x)
        x = self.deconv_layers(x)
        x = self.final_layer(x)
        heatmap = self.heatmap_conv_layers(x)
        offset = self.offset_conv_layers(x)
        return [[heatmap, offset]]

    def init_weights(self):
        """Initialize model weights."""
        super().init_weights()
        for name, m in self.heatmap_conv_layers.named_modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)
        for name, m in self.offset_conv_layers.named_modules():
            if isinstance(m, nn.Conv2d):
                if 'transform_matrix_conv' in name:
                    normal_init(m, std=1e-8, bias=0)
                else:
                    normal_init(m, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)


@HEADS.register_module()
class DEKRHeadWithRLELoss_new(DeconvHead):
    """在DEKRHead基础上修改, 
        1、为适配offset部分使用RLE loss, offset部分每个关键点需增加两个维度的输出, 代表方差.
        2、关键点热力图可选, use_keypoint_heatmaps=False时, test_cfg[refine_keypoint_scores_by_keypoint_heatmaps]需为False
        3、增加OKS损失

    Args:
        in_channels (int): Number of input channels.
        num_joints (int): Number of joints.
        num_heatmap_filters (int): Number of filters for heatmap branch.
        num_offset_filters_per_joint (int): Number of filters for each joint.
        in_index (int|Sequence[int]): Input feature index. Default: 0
        input_transform (str|None): Transformation type of input features.
            Options: 'resize_concat', 'multiple_select', None.
            Default: None.

            - 'resize_concat': Multiple feature maps will be resized to the
                same size as the first one and then concat together.
                Usually used in FCN head of HRNet.
            - 'multiple_select': Multiple feature maps will be bundle into
                a list and passed into decode head.
            - None: Only one select feature map is allowed.
        align_corners (bool): align_corners argument of F.interpolate.
            Default: False.
        heatmap_loss (dict): Config for heatmap loss. Default: None.
        offset_loss (dict): Config for offset loss. Default: None.
        oks_loss (dict): Config for oks loss. Default: None.
        use_keypoint_heatmaps (bool): 是否生成关键点热力图
    """

    def __init__(self,
                 in_channels,
                 num_joints,
                 num_heatmap_filters=32,
                 num_offset_filters_per_joint=15,
                 in_index=0,
                 input_transform=None,
                 num_deconv_layers=0,
                 num_deconv_filters=None,
                 num_deconv_kernels=None,
                 extra=dict(final_conv_kernel=0),
                 align_corners=False,
                 heatmap_loss=None,
                 offset_loss=None,
                 oks_loss=None,
                 use_keypoint_heatmaps=True):

        super().__init__(
            in_channels,
            out_channels=in_channels,
            num_deconv_layers=num_deconv_layers,
            num_deconv_filters=num_deconv_filters,
            num_deconv_kernels=num_deconv_kernels,
            align_corners=align_corners,
            in_index=in_index,
            input_transform=input_transform,
            extra=extra,
            loss_keypoint=heatmap_loss)
        self.use_keypoint_heatmaps = use_keypoint_heatmaps

        # set up filters for heatmap
        self.heatmap_conv_layers = nn.Sequential(
            ConvModule(
                in_channels=self.in_channels,
                out_channels=num_heatmap_filters,
                kernel_size=1,
                norm_cfg=dict(type='BN')),
            BasicBlock(num_heatmap_filters, num_heatmap_filters),
            build_conv_layer(
                dict(type='Conv2d'),
                in_channels=num_heatmap_filters,
                out_channels=1 + num_joints if use_keypoint_heatmaps else 1,
                kernel_size=1))

        # set up filters for offset map
        groups = num_joints
        num_offset_filters = num_joints * num_offset_filters_per_joint

        self.offset_conv_layers = nn.Sequential(
            ConvModule(
                in_channels=self.in_channels,
                out_channels=num_offset_filters,
                kernel_size=1,
                norm_cfg=dict(type='BN')),
            AdaptiveActivationBlock(
                num_offset_filters, num_offset_filters, groups=groups),
            AdaptiveActivationBlock(
                num_offset_filters, num_offset_filters, groups=groups),
            build_conv_layer(
                dict(type='Conv2d'),
                in_channels=num_offset_filters,
                out_channels=4 * num_joints,
                kernel_size=1,
                groups=groups
            )
        )

        # set up offset losses
        self.offset_loss = build_loss(copy.deepcopy(offset_loss))
        
        # set up OKS losses
        self.oks_loss = build_loss(copy.deepcopy(oks_loss))

    def get_loss(self, outputs, heatmaps, masks, offsets, offset_weights):
        """Calculate the dekr loss.

        Note:
            - batch_size: N
            - num_channels: C
            - num_joints: K
            - heatmaps height: H
            - heatmaps weight: W

        Args:
            outputs (List(torch.Tensor[N,C,H,W])): Multi-scale outputs.
            heatmaps (List(torch.Tensor[N,K+1,H,W])): Multi-scale heatmap
                targets.
            masks (List(torch.Tensor[N,K+1,H,W])): Weights of multi-scale
                heatmap targets.
            offsets (List(torch.Tensor[N,K*2,H,W])): Multi-scale offset
                targets.
            offset_weights (List(torch.Tensor[N,K*2,H,W])): Weights of
                multi-scale offset targets.
        """
        if not self.use_keypoint_heatmaps:
            heatmaps = [x[:,0,:,:] for x in heatmaps]   # BottomUpGenerateHeatmapTarget生成的热力图第一层为中心点热力图，其余为关键点热力图
        losses = dict()

        for idx in range(len(outputs)):
            pred_heatmap, pred_offset = outputs[idx]
            heatmap_weight = masks[idx].view(masks[idx].size(0),
                                             masks[idx].size(1), -1)
            losses['loss_hms'] = losses.get('loss_hms', 0) + self.loss(
                pred_heatmap, heatmaps[idx], heatmap_weight)
            losses['loss_ofs'] = losses.get('loss_ofs', 0) + self.offset_loss(
                pred_offset, offsets[idx], offset_weights[idx])
            losses['loss_oks'] = losses.get('loss_oks', 0) + self.oks_loss(
                pred_offset, offsets[idx], pred_heatmap, offset_weights[idx])
        return losses

    def forward(self, x):
        """Forward function."""
        x = self._transform_inputs(x)
        x = self.deconv_layers(x)
        x = self.final_layer(x)
        heatmap = self.heatmap_conv_layers(x)
        offset = self.offset_conv_layers(x)
        return [[heatmap, offset]]

    def init_weights(self):
        """Initialize model weights."""
        super().init_weights()
        for name, m in self.heatmap_conv_layers.named_modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)
        for name, m in self.offset_conv_layers.named_modules():
            if isinstance(m, nn.Conv2d):
                if 'transform_matrix_conv' in name:
                    normal_init(m, std=1e-8, bias=0)
                else:
                    normal_init(m, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)
