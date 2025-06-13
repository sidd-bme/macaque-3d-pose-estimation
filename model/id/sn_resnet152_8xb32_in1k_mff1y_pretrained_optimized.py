auto_scale_lr = dict(base_batch_size=256)
classes = [
    'b',
    'd',
    'g',
    'r',
    'unknown',
    'w',
]
data_preprocessor = dict(
    mean=[
        123.675,
        116.28,
        103.53,
    ],
    num_classes=6,
    std=[
        58.395,
        57.12,
        57.375,
    ],
    to_rgb=True)
dataset_type = 'ImageNet'
default_hooks = dict(
    checkpoint=dict(interval=1, type='CheckpointHook'),
    logger=dict(interval=100, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(enable=False, type='VisualizationHook'))
default_scope = 'mmpretrain'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
launcher = 'pytorch'
load_from = '/mnt/nas_siddharth/code_final/model/id/id.pth'
log_level = 'INFO'
model = dict(
    backbone=dict(
        depth=152,
        init_cfg=dict(
            checkpoint=
            'https://download.openmmlab.com/mmclassification/v0/resnet/resnet152_8xb32_in1k_20210901-4d7582fa.pth',
            prefix='backbone',
            type='Pretrained'),
        num_stages=4,
        out_indices=(3, ),
        style='pytorch',
        type='ResNet'),
    head=dict(
        in_channels=2048,
        loss=dict(
            class_weight=[
                1.803030303030303,
                6.346666666666667,
                2.6225895316804406,
                1.9232323232323232,
                0.23681592039800994,
                6.102564102564102,
            ],
            label_smooth_val=0.1,
            loss_weight=1.0,
            mode='original',
            type='LabelSmoothLoss'),
        num_classes=6,
        topk=(
            1,
            5,
        ),
        type='LinearClsHead'),
    neck=dict(type='GlobalAveragePooling'),
    type='ImageClassifier')
optim_wrapper = dict(
    optimizer=dict(lr=0.0005, momentum=0.9, type='SGD', weight_decay=0.0001))
param_scheduler = [
    dict(T_max=100, by_epoch=True, eta_min=1e-05, type='CosineAnnealingLR'),
]
randomness = dict(deterministic=False, seed=None)
resume = False
test_cfg = dict()
test_dataloader = dict(
    batch_size=32,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        ann_file=
        '/mnt/nas_siddharth/dataset/id_macaque_mff1y/test/annotations.txt',
        classes=[
            'b',
            'd',
            'g',
            'r',
            'unknown',
            'w',
        ],
        data_root='/mnt/nas_siddharth/dataset/id_macaque_mff1y/test',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(edge='short', scale=256, type='ResizeEdge'),
            dict(crop_size=224, type='CenterCrop'),
            dict(type='PackInputs'),
        ],
        type='ImageNet'),
    num_workers=5,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = [
    dict(type='Accuracy'),
    dict(type='SingleLabelMetric'),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(edge='short', scale=256, type='ResizeEdge'),
    dict(crop_size=224, type='CenterCrop'),
    dict(type='PackInputs'),
    dict(
        transforms=[
            [
                dict(type='Identity'),
            ],
            [
                dict(prob=1.0, type='RandomHorizontalFlip'),
            ],
        ],
        type='TestTimeAug'),
]
train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=1)
train_dataloader = dict(
    batch_size=32,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        classes=[
            'b',
            'd',
            'g',
            'r',
            'unknown',
            'w',
        ],
        data_root='/mnt/nas_siddharth/dataset/id_macaque_mff1y/train',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(scale=224, type='RandomResizedCrop'),
            dict(direction='horizontal', prob=0.5, type='RandomFlip'),
            dict(
                brightness=0.2,
                contrast=0.2,
                hue=0.1,
                saturation=0.2,
                type='ColorJitter'),
            dict(erase_prob=0.25, type='RandomErasing'),
            dict(type='PackInputs'),
        ],
        type='ImageNet'),
    num_workers=5,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(scale=224, type='RandomResizedCrop'),
    dict(prob=0.5, type='RandomHorizontalFlip'),
    dict(degrees=20, type='RandomRotation'),
    dict(
        brightness=0.3,
        contrast=0.3,
        hue=0.1,
        saturation=0.3,
        type='ColorJitter'),
    dict(prob=0.15, type='RandomGrayscale'),
    dict(
        erase_prob=0.35,
        max_area_ratio=0.33,
        min_area_ratio=0.02,
        mode='rand',
        type='RandomErasing'),
    dict(type='PackInputs'),
]
val_cfg = dict()
val_dataloader = dict(
    batch_size=32,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        ann_file=
        '/mnt/nas_siddharth/dataset/id_macaque_mff1y/test/annotations.txt',
        classes=[
            'b',
            'd',
            'g',
            'r',
            'unknown',
            'w',
        ],
        data_root='/mnt/nas_siddharth/dataset/id_macaque_mff1y/test',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(edge='short', scale=256, type='ResizeEdge'),
            dict(crop_size=224, type='CenterCrop'),
            dict(type='PackInputs'),
        ],
        type='ImageNet'),
    num_workers=5,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = [
    dict(type='Accuracy'),
    dict(type='SingleLabelMetric'),
]
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    type='UniversalVisualizer', vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = '/mnt/nas_siddharth/collar_id_classifier_mff1y/training'
