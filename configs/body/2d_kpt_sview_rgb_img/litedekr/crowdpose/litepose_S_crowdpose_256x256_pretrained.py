_base_ = [
    '../../../../_base_/default_runtime.py',
    '../../../../_base_/datasets/crowdpose.py'
]
checkpoint_config = dict(interval=20)
evaluation = dict(interval=20, metric='mAP', save_best='AP')
# fp16 = dict(loss_scale=512.)
optimizer = dict(
    type='Adam',
    lr=0.001,
)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[140, 200, 260])
total_epochs = 300
channel_cfg = dict(
    num_output_channels=14,
    dataset_joints=14,
    dataset_channel=[
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
    ],
    inference_channel=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])

data_cfg = dict(
    image_size=256,
    base_size=256,
    base_sigma=2,
    heatmap_size=[64, 128],
    num_joints=channel_cfg['dataset_joints'],
    dataset_channel=channel_cfg['dataset_channel'],
    inference_channel=channel_cfg['inference_channel'],
    num_scales=2,
    scale_aware_sigma=False,
)

# model settings
model = dict(
    type='DisentangledKeypointRegressor',
    pretrained='/workspace/mmpose/LitePose-Auto-S-CrowdPose.pth.tar',
    backbone=dict(
        type='LitePose',
        cfg_arch=dict(
            input_channel=16,
            backbone_setting=[
                dict(
                    num_blocks=6, 
                    stride=2, 
                    channel=16, 
                    block_setting=[
                        [6, 7], 
                        [6, 7], 
                        [6, 7], 
                        [6, 7], 
                        [6, 7], 
                        [6, 7],
                    ],
                ),
                dict(
                    num_blocks=8, 
                    stride=2, 
                    channel=32, 
                    block_setting=[
                        [6, 7], 
                        [6, 7], 
                        [6, 7], 
                        [6, 7], 
                        [6, 7], 
                        [6, 7],
                        [6, 7], 
                        [6, 7],
                    ],
                ),
                dict(
                    num_blocks=10, 
                    stride=2, 
                    channel=48, 
                    block_setting=[
                        [6, 7], 
                        [6, 7], 
                        [6, 7], 
                        [6, 7], 
                        [6, 7], 
                        [6, 7], 
                        [6, 7], 
                        [6, 7], 
                        [6, 7], 
                        [6, 7],
                    ],
                ),
                dict(
                    num_blocks=10, 
                    stride=1, 
                    channel=120, 
                    block_setting=[
                        [6, 7], 
                        [6, 7], 
                        [6, 7], 
                        [6, 7], 
                        [6, 7], 
                        [6, 7], 
                        [6, 7], 
                        [6, 7], 
                        [6, 7], 
                        [6, 7],
                    ],
                ),
            ]
        ),
        width_mult=1.0, 
        round_nearest=8, 
        frozen_stages=-1, 
        norm_eval=False,
    ),
    keypoint_head=dict(
        type='DEKRHead',
        in_channels=(16, 32, 48, 120),
        in_index=(1, 2, 3, 4),   # stem, stage1, stage2, stage3, stage4
        num_heatmap_filters=32,
        num_joints=channel_cfg['dataset_joints'],
        input_transform='resize_concat',
        heatmap_loss=dict(
            type='JointsMSELoss',
            use_target_weight=True,
            loss_weight=1.0,
        ),
        offset_loss=dict(
            type='SoftWeightSmoothL1Loss',
            use_target_weight=True,
            supervise_empty=False,
            loss_weight=0.004,
            beta=1 / 9.0,
        )),
    train_cfg=dict(),
    test_cfg=dict(
        num_joints=channel_cfg['dataset_joints'],
        max_num_people=30,
        project2image=False,
        align_corners=False,
        max_pool_kernel=5,
        use_nms=True,
        nms_dist_thr=0.05,
        nms_joints_thr=7,
        keypoint_threshold=0.5,
        rescore_cfg=dict(
            in_channels=59,
            norm_indexes=(0, 1),
            pretrained='https://download.openmmlab.com/mmpose/'
            'pretrain_models/kpt_rescore_crowdpose-300c7efe.pth'),
        flip_test=True,
    )
)

train_pipeline = [
    dict(type='LoadImageFromFile'),# 读取图像前，关键点坐标已根据多尺度书训练进行备份
    dict(
        type='BottomUpRandomAffine',    # 坐标和图像都进行了仿射变换
        rot_factor=30,
        scale_factor=[0.75, 1.5],
        scale_type='short',
        trans_factor=40),
    dict(type='BottomUpRandomFlip', flip_prob=0.5),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(type='GetKeypointCenterArea'),
    dict(
        type='BottomUpGenerateHeatmapTarget',
        sigma=(2, 4),
        bg_weight=0.1,
        gen_center_heatmap=True,
    ),
    dict(
        type='BottomUpGenerateOffsetTarget',
        radius=4,
    ),
    dict(
        type='Collect',
        keys=['img', 'heatmaps', 'masks', 'offsets', 'offset_weights'],
        meta_keys=[]),
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='BottomUpGetImgSize', test_scale_factor=[1]),
    dict(
        type='BottomUpResizeAlign',
        transforms=[
            dict(type='ToTensor'),
            dict(
                type='NormalizeTensor',
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
        ]),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=[
            'image_file', 'aug_data', 'test_scale_factor', 'base_size',
            'center', 'scale', 'flip_index', 'num_joints', 'skeleton',
            'image_size', 'heatmap_size'
        ]),
]

test_pipeline = val_pipeline

data_root = '/workspace/mmpose/data/crowdpose'
data = dict(
    workers_per_gpu=0,
    train_dataloader=dict(samples_per_gpu=32),
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1),
    train=dict(
        type='BottomUpCrowdPoseDataset',
        ann_file=f'{data_root}/annotations/mmpose_crowdpose_trainval.json',
        img_prefix=f'{data_root}/images/',
        data_cfg=data_cfg,
        pipeline=train_pipeline,
        dataset_info={{_base_.dataset_info}}),
    val=dict(
        type='BottomUpCrowdPoseDataset',
        ann_file=f'{data_root}/annotations/mmpose_crowdpose_test.json',
        img_prefix=f'{data_root}/images/',
        data_cfg=data_cfg,
        pipeline=val_pipeline,
        dataset_info={{_base_.dataset_info}}),
    test=dict(
        type='BottomUpCrowdPoseDataset',
        ann_file=f'{data_root}/annotations/mmpose_crowdpose_test.json',
        img_prefix=f'{data_root}/images/',
        data_cfg=data_cfg,
        pipeline=test_pipeline,
        dataset_info={{_base_.dataset_info}}),
)
