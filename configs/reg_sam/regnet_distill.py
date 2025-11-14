"""
实时全景分割配置文件 - 整合分辨率蒸馏和SAM蒸馏
对标MaskConver，实现全卷积、实时、高性能的全景分割
创新点：
1. Backbone改进（RegNet + 可学习特征融合）
2. SAM蒸馏学习
3. 分辨率蒸馏学习（参考DSRL）
"""
from mmengine.config import read_base
from mmdet.models import BatchFixedSizePad, CrossEntropyLoss, DiceLoss, MaskFormerFusionHead, FocalLoss
from mmdet.models.task_modules.assigners import HungarianAssigner, ClassificationCost, CrossEntropyLossCost, DiceCost
from mmdet.models.task_modules.samplers import MaskPseudoSampler
from mmdet.datasets.coco import CocoDataset
from mmdet.models.backbones import ResNet
from seg.models.heads.rapsam_head import RapSAMVideoHead
from seg.models.detectors.rapsam import RapSAM
from seg.models.data_preprocessor.vid_sam_preprocessor import VideoPromptDataPreprocessor
from seg.models.necks.ramsam_neck import YOSONeck
with read_base():
    from .._base_.default_runtime import *
    from .._base_.datasets.coco_panoptic_video_lsj import *
    from .._base_.schedules.schedule_12e import *

backend_args = None

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
num_classes = num_things_classes + 1

class_weights = [2.0]*num_things_classes + [1.0]*num_stuff_classes + [0.1]

model = dict(
    type=RapSAM,
    use_learnable_fusion=True,  # 创新点1：可学习特征融合
    # 分辨率蒸馏配置（创新点3）
    use_resolution_distill=True,
    resolution_distill=dict(
        feat_channels=256,
        sr_loss_weight=0.5,  # 超分辨率重建损失权重
        fa_loss_weight=0.5,  # 特征对齐损失权重
        fa_subscale=0.0625,  # FA Loss下采样比例
        # 显存优化
        use_low_res_sr=True,  # 使用较低分辨率的超分辨率重建
        sr_target_scale=1.5,  # 超分辨率目标倍数（降低以节省显存，默认2.0）
    ),
    # SAM蒸馏配置（创新点2）
    use_sam_distill=True,
    sam_distill=dict(
        # 配置SAM教师模型（使用完整的SAM模型）
        teacher_model=dict(
            type='SAMTeacherModel',  # 使用SAM教师模型包装器
            model_type='vit_h',  # SAM模型类型：'vit_h', 'vit_l', 'vit_b'（可改为'vit_l'或'vit_b'以节省显存）
            checkpoint='/data/chenjiahui/RPS/checkpoint/sam_vit_h_4b8939.pth',  # SAM checkpoint路径
            freeze=True  # 冻结教师模型参数
        ),
        teacher_checkpoint=None,  # 如果teacher_model中已配置checkpoint，这里可以为None
        feat_distill_weight=1.0,  # 特征蒸馏损失权重
        output_distill_weight=1.0,  # 输出蒸馏损失权重（如果SAM结构支持）
        temperature=4.0,  # 蒸馏温度
        distill_feat_layers=[0],  # SAM只输出单尺度特征，所以只蒸馏第0层
        # 显存优化参数
        use_low_res_sam=True,  # 使用低分辨率SAM输入（512而不是1024）以节省显存
        sam_input_size=512,  # SAM输入尺寸（降低以节省显存，默认1024）
        distill_interval=2,  # 每2个迭代进行一次SAM蒸馏（1=每次都蒸馏，增大可节省显存）
    ),
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='RegNet',
        arch='regnetx_12gf',
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        init_cfg=dict(type='Pretrained', checkpoint='open-mmlab://regnetx_12gf')
    ),
    neck=dict(
        type=YOSONeck,
        agg_dim=128,
        hidden_dim=256,
        backbone_shape=[224, 448, 896, 2240],
    ),
    panoptic_head=dict(
        type=RapSAMVideoHead,
        prompt_with_kernel_updator=False,
        panoptic_with_kernel_updator=True,  
        use_adaptor=False,
        use_kernel_updator=False,
        use_self_attn=False,
        sphere_cls=True,
        ov_classifier_name='convnext_large_d_320_CocoPanopticOVDataset',
        num_stages=3,
        feat_channels=256,
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
            type='CrossEntropyLoss',
            use_sigmoid=True,
            reduction='mean',
            loss_weight=3.0),
        loss_dice=dict(
            type='DiceLoss',
            use_sigmoid=True,
            activate=True,
            reduction='mean',
            naive_dice=True,
            eps=1.0,
            loss_weight=2.0)),
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
        iou_thr=0.8,
        filter_low_score=True,
        score_thr=0.1),
)


optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=2e-4,
        weight_decay=0.05,
        betas=(0.9, 0.999)
    ),
    clip_grad=dict(max_norm=0.5, norm_type=2),
)

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.1,
        by_epoch=False,
        begin=0,
        end=300
    ),
    dict(
        type='StepLR',
        step_size=50,
        gamma=0.9,
        by_epoch=True
    )
]

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=1)

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
        patience=20,
        min_delta=0.001,
        rule='greater'
    ),
    logger=dict(type='mmengine.hooks.LoggerHook', interval=50),
    param_scheduler=dict(type='mmengine.hooks.ParamSchedulerHook'),
    sampler_seed=dict(type='mmengine.hooks.DistSamplerSeedHook'),
    timer=dict(type='mmengine.hooks.IterTimerHook'),
    visualization=dict(type='mmdet.engine.hooks.DetVisualizationHook')
)

auto_scale_lr = dict(base_batch_size=1, enable=False)
find_unused_parameters = True

