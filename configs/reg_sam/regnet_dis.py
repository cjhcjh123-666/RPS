from mmengine.config import read_base
from mmdet.models import BatchFixedSizePad, CrossEntropyLoss, DiceLoss, MaskFormerFusionHead
from mmdet.models.task_modules.assigners import HungarianAssigner, ClassificationCost, CrossEntropyLossCost, DiceCost
from mmdet.models.task_modules.samplers import MaskPseudoSampler
from mmdet.models.backbones import RegNet
from seg.models.backbones import RegNetReparam
from seg.models.heads.rapsam_head import RapSAMVideoHead
from seg.models.heads.rapsam_head_static import RapSAMVideoHeadStatic
from seg.models.detectors.rapsam import RapSAM
from seg.models.data_preprocessor.vid_sam_preprocessor import VideoPromptDataPreprocessor
from seg.models.hooks import ReparamHook

with read_base():
    from .._base_.default_runtime import *
    from .._base_.datasets.coco_panoptic_lsj import *
    from .._base_.schedules.schedule_12e import *

load_from = 'input your pretrain model path'

image_size = (640, 640)

from mmcv.transforms import LoadImageFromFile, RandomResize
from mmengine.dataset import DefaultSampler
from mmdet.datasets import AspectRatioBatchSampler
from mmdet.datasets.transforms import LoadPanopticAnnotations, RandomFlip, RandomCrop, PackDetInputs, Resize
from seg.datasets.coco_ov import CocoPanopticOVDataset
from seg.datasets.pipeliens.loading import FilterAnnotationsHB, LoadPanopticAnnotationsHB

backend_args = None
data_root = 'input your data root'

train_pipeline = [
    dict(type=LoadImageFromFile, to_float32=True, backend_args=backend_args),
    dict(type=LoadPanopticAnnotationsHB, with_bbox=True, with_mask=True,
         with_seg=True, backend_args=backend_args),
    dict(type=RandomFlip, prob=0.5),
    dict(type=RandomResize, resize_type=Resize, scale=image_size,
         ratio_range=(0.8, 1.3), keep_ratio=True),
    dict(type=RandomCrop, crop_size=image_size, crop_type='absolute',
         recompute_bbox=True, allow_negative_crop=False),
    dict(type=FilterAnnotationsHB, by_box=False, by_mask=True, min_gt_mask_area=32),
    dict(type=PackDetInputs)
]

train_dataloader = dict(
    batch_size=24,
    num_workers=24,
    persistent_workers=True,
    sampler=dict(type=DefaultSampler, shuffle=True),
    batch_sampler=dict(type=AspectRatioBatchSampler),
    dataset=dict(
        type=CocoPanopticOVDataset,
        data_root=data_root,
        ann_file='annotations/panoptic_train2017.json',
        data_prefix=dict(img='train2017/', seg='annotations/panoptic_train2017/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=16),
        pipeline=train_pipeline,
        backend_args=backend_args
    )
)

batch_augments = [
    dict(
        type=BatchFixedSizePad,
        size=(image_size[1], image_size[0]),
        img_pad_value=0,
        pad_mask=True,
        mask_pad_value=0,
        pad_seg=True,
        seg_pad_value=255
    )
]

data_preprocessor = dict(
    type=VideoPromptDataPreprocessor,
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_size_divisor=32,
    pad_mask=True,
    mask_pad_value=0,
    pad_seg=True,
    seg_pad_value=255,
    batch_augments=batch_augments
)

num_things_classes = 80
num_stuff_classes = 53
num_classes = num_things_classes + num_stuff_classes

class_weights = [1.0] * 134
class_weights[-1] = 0.1

difficult_classes = [10, 11, 12, 21, 67, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90]
for idx in difficult_classes:
    if idx < len(class_weights) - 1:
        class_weights[idx] = 1.8

small_classes = [0, 2, 8, 9, 13, 24, 25, 38, 39, 43, 44, 55, 57, 59, 65, 66, 71, 72, 73, 74]
for idx in small_classes:
    if idx < len(class_weights) - 1:
        class_weights[idx] = 1.2

model = dict(
    type=RapSAM,
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='RegNet',
        arch='regnetx_800mf',
        out_indices=( 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        init_cfg=dict(type='Pretrained', checkpoint='open-mmlab://regnetx_800mf')
    ),
    neck=dict(
        type='FPN',
        in_channels=[ 128, 288, 672],
        out_channels=128,
        num_outs=3,
        start_level=0,
        end_level=-1,
        add_extra_convs='on_output',
        relu_before_extra_convs=True
    ),
    panoptic_head=dict(
        type=RapSAMVideoHeadStatic,
        prompt_with_kernel_updator=False,
        panoptic_with_kernel_updator=True,
        use_adaptor=False,
        use_kernel_updator=False,
        use_self_attn=False,
        sphere_cls=True,
        ov_classifier_name='/data/chenjiahui/RMP-SAM-yolo/convnext_large_d_320_CocoPanopticOVDataset',
        num_stages=3,
        feat_channels=128,
        num_things_classes=num_things_classes,
        num_stuff_classes=num_stuff_classes,
        num_queries=80,
        loss_cls=dict(
            type=CrossEntropyLoss,
            use_sigmoid=False,
            loss_weight=2.0,
            reduction='mean',
            class_weight=class_weights),
        loss_mask=dict(
            type=CrossEntropyLoss,
            use_sigmoid=True,
            reduction='mean',
            loss_weight=3.0),
        loss_dice=dict(
            type=DiceLoss,
            use_sigmoid=True,
            activate=True,
            reduction='mean',
            naive_dice=True,
            eps=1.0,
            loss_weight=2.0),
        distill_cfg=dict(
            type='ResidualLogit',
            name='cls_logit_distill',
            loss_distill=dict(
                type='KnowledgeDistillationKLDivLoss',
                reduction='mean',
                loss_weight=25.0,
                T=4.0
            ),
            teacher_module=dict(
                type=RapSAMVideoHead,
                prompt_with_kernel_updator=False,
                panoptic_with_kernel_updator=True,
                use_adaptor=False,
                use_kernel_updator=False,
                use_self_attn=False,
                sphere_cls=True,
                ov_classifier_name='/data/chenjiahui/RMP-SAM-yolo/convnext_large_d_320_CocoPanopticOVDataset',
                num_stages=3,
                feat_channels=128,
                num_things_classes=num_things_classes,
                num_stuff_classes=num_stuff_classes,
                num_queries=80
            )
        )
    ),
    panoptic_fusion_head=dict(
        type=MaskFormerFusionHead,
        num_things_classes=num_things_classes,
        num_stuff_classes=num_stuff_classes,
        loss_panoptic=None,
        init_cfg=None),
    train_cfg=dict(
        num_points=12000,
        oversample_ratio=3.0,
        importance_sample_ratio=0.6,
        assigner=dict(
            type='HungarianAssigner',
            match_costs=[
                dict(type='ClassificationCost', weight=2.0),
                dict(type='CrossEntropyLossCost', weight=5.0, use_sigmoid=True),
                dict(type='DiceCost', weight=2.0, pred_act=True, eps=1.0)
            ]),
        sampler=dict(type='MaskPseudoSampler'),
    ),
    test_cfg=dict(
        panoptic_on=True,
        semantic_on=False,
        instance_on=True,
        max_per_image=200,
        iou_thr=0.55,
        filter_low_score=True,
        score_thr=0.1),
)

val_dataloader = dict(
    batch_size=4,
    num_workers=8,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoPanopticDataset',
        data_root='/data/zhangyafei/RMP-SAM/data/coco/',
        ann_file='annotations/panoptic_val2017.json',
        data_prefix=dict(img='val2017/', seg='annotations/panoptic_val2017/'),
        test_mode=True,
        pipeline=[
            dict(type='LoadImageFromFile', backend_args=None),
            dict(type='Resize', scale=(640, 480), keep_ratio=False),
            dict(type='LoadPanopticAnnotations', backend_args=None),
            dict(
                type='PackDetInputs',
                meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor')
            )
        ],
        backend_args=None
    )
)

val_evaluator = [
    dict(
        type='CocoPanopticMetric',
        ann_file='/data/zhangyafei/RMP-SAM/data/coco/annotations/panoptic_val2017.json',
        seg_prefix='/data/zhangyafei/RMP-SAM/data/coco/annotations/panoptic_val2017/',
        backend_args=None
    ),
    dict(
        type='CocoMetric',
        ann_file='/data/zhangyafei/RMP-SAM/data/coco/annotations/instances_val2017.json',
        metric=['bbox', 'segm'],
        backend_args=None
    )
]

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=1e-4,
        weight_decay=0.05,
        betas=(0.9, 0.999)
    ),
    clip_grad=dict(max_norm=0.5, norm_type=2),
)

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.01,
        by_epoch=False,
        begin=0,
        end=500
    ),
    dict(
        type='CosineAnnealingLR',
        by_epoch=True,
        begin=0,
        end=100,
        T_max=100,
        eta_min=1e-5
    )
]

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=100, val_interval=1)

default_hooks = dict(
    checkpoint=dict(
        type='mmengine.hooks.CheckpointHook',
        interval=1,
        max_keep_ckpts=3,
        save_best='coco_panoptic/PQ',
        rule='greater',
        save_last=True
    ),
    early_stopping=dict(
        type='mmengine.hooks.EarlyStoppingHook',
        monitor='coco_panoptic/PQ',
        patience=15,
        min_delta=0.001,
        rule='greater'
    ),
    reparam=dict(
        type=ReparamHook,
        auto_fuse_on_test=True,
        auto_fuse_on_val=True,
        verbose=True
    ),
    logger=dict(type='mmengine.hooks.LoggerHook', interval=50),
    param_scheduler=dict(type='mmengine.hooks.ParamSchedulerHook'),
    sampler_seed=dict(type='mmengine.hooks.DistSamplerSeedHook'),
    timer=dict(type='mmengine.hooks.IterTimerHook'),
    visualization=dict(type='mmdet.engine.hooks.DetVisualizationHook')
)

auto_scale_lr = dict(base_batch_size=1, enable=False)
find_unused_parameters = True
