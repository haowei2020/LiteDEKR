_base_ = [
    '../../../../_base_/default_runtime.py',
    '../../../../_base_/datasets/coco.py'
]

log_config = dict(
    interval=500,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
        # dict(type='PaviLoggerHook') # for internal services
    ])

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
    step=[40, 60, 90])
total_epochs = 120
channel_cfg = dict(
    dataset_joints=17,
    dataset_channel=[
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    ],
    inference_channel=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])

data_cfg = dict(
    image_size=448,
    base_size=256,
    base_sigma=2,
    heatmap_size=[112, 224],
    num_joints=channel_cfg['dataset_joints'],
    dataset_channel=channel_cfg['dataset_channel'],
    inference_channel=channel_cfg['inference_channel'],
    num_scales=2,
    scale_aware_sigma=False,
)

# model settings
model = dict(
    type='DisentangledKeypointRegressorWithRLE',
    pretrained='/workspace/mmpose/LitePose-Auto-S-COCO.pth.tar',
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
        type='DEKRHeadWithRLELoss_new',
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
            type='RLELossForBottomUp',
            use_target_weight=True,
            supervise_empty=False,
            residual=True,
            q_dis='laplace',
            loss_weight=0.004,
        ),
        oks_loss=dict(
            type='OKSLoss',
            num_joints=17,
            loss_weight=1.0,
        ),
        use_keypoint_heatmaps=True,
    ),
    train_cfg=dict(),
    test_cfg=dict(
        num_joints=channel_cfg['dataset_joints'],
        max_num_people=30,
        project2image=False,
        align_corners=False,
        max_pool_kernel=5,
        use_sigma_as_score=False,
        use_nms=True,
        nms_dist_thr=0.05,
        nms_joints_thr=8,
        keypoint_threshold=0.1,
        rescore_cfg=dict(
            in_channels=74,
            norm_indexes=(5, 6),
            pretrained='https://download.openmmlab.com/mmpose/'
            'pretrain_models/kpt_rescore_coco-33d58c5c.pth'),
        flip_test=True,
        use_udp=False,
        refine_keypoint_scores_by_keypoint_heatmaps=True,
    )
)

train_pipeline = [
    dict(type='LoadImageFromFile'),# 读取图像前，关键点坐标已根据多尺度书训练进行备份
    dict(
        type='BottomUpRandomAffine',    # 坐标和图像都进行了仿射变换
        rot_factor=30,
        scale_factor=[0.75, 1.5],
        scale_type='short',
        trans_factor=40,
        use_udp=False),
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
        use_udp=False,
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
    dict(type='BottomUpGetImgSize', test_scale_factor=[1], use_udp=False),
    dict(
        type='BottomUpResizeAlign',
        transforms=[
            dict(type='ToTensor'),
            dict(
                type='NormalizeTensor',
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
        ],
        use_udp=False),
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

data_root = '/workspace/mmpose/data/coco'
data = dict(
    workers_per_gpu=4,
    train_dataloader=dict(samples_per_gpu=3),
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1),
    train=dict(
        type='BottomUpCocoDataset',
        ann_file=f'{data_root}/annotations/person_keypoints_train2017.json',
        img_prefix=f'{data_root}/train2017/',
        data_cfg=data_cfg,
        pipeline=train_pipeline,
        dataset_info={{_base_.dataset_info}}),
    val=dict(
        type='BottomUpCocoDataset',
        ann_file=f'{data_root}/annotations/person_keypoints_val2017.json',
        img_prefix=f'{data_root}/val2017/',
        data_cfg=data_cfg,
        pipeline=val_pipeline,
        dataset_info={{_base_.dataset_info}}),
    test=dict(
        type='BottomUpCocoDataset',
        ann_file=f'{data_root}/annotations/person_keypoints_val2017.json',
        img_prefix=f'{data_root}/val2017/',
        data_cfg=data_cfg,
        pipeline=test_pipeline,
        dataset_info={{_base_.dataset_info}}),
)
