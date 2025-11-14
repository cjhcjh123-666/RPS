import torch
import torch.nn.functional as F
from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from typing import Dict, List, Optional

from mmdet.models.detectors import SingleStageDetector
from .mask2former_vid import Mask2formerVideo
from seg.models.utils import ResolutionDistillModule, SAMDistillModule

@MODELS.register_module()
class RapSAM(Mask2formerVideo):
    OVERLAPPING = None

    def __init__(self,
                 backbone: ConfigType,
                 neck: OptConfigType = None,
                 panoptic_head: OptConfigType = None,
                 panoptic_fusion_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 inference_sam: bool = False,
                 init_cfg: OptMultiConfig = None,
                 use_learnable_fusion: bool = True,  # ★ 是否使用可学习加权融合
                 # 分辨率蒸馏配置
                 use_resolution_distill: bool = False,
                 resolution_distill: OptConfigType = None,
                 # SAM蒸馏配置
                 use_sam_distill: bool = False,
                 sam_distill: OptConfigType = None,
                 ):
        super(SingleStageDetector, self).__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.backbone = MODELS.build(backbone)
        if neck is not None:
            self.neck = MODELS.build(neck)

        panoptic_head_ = panoptic_head.deepcopy()
        panoptic_head_.update(train_cfg=train_cfg)
        panoptic_head_.update(test_cfg=test_cfg)
        self.panoptic_head = MODELS.build(panoptic_head_)

        panoptic_fusion_head_ = panoptic_fusion_head.deepcopy()
        panoptic_fusion_head_.update(test_cfg=test_cfg)
        self.panoptic_fusion_head = MODELS.build(panoptic_fusion_head_)

        self.num_things_classes = self.panoptic_head.num_things_classes
        self.num_stuff_classes = self.panoptic_head.num_stuff_classes
        self.num_classes = self.panoptic_head.num_classes

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.alpha = 0.4
        self.beta = 0.8

        self.inference_sam = inference_sam
        
        # ★ 初始化可学习融合权重（如果启用）
        self.use_learnable_fusion = use_learnable_fusion
        if use_learnable_fusion and neck is not None:
            import torch.nn as nn
            # FPN通常输出4个尺度的特征，初始化4个可学习权重
            # 初始值都设为1.0，经过softmax后每个尺度权重为0.25
            self.fusion_weights = nn.Parameter(torch.ones(4))
        
        # ★ 分辨率蒸馏模块
        self.use_resolution_distill = use_resolution_distill
        if use_resolution_distill and resolution_distill is not None:
            self.resolution_distill_module = ResolutionDistillModule(**resolution_distill)
        else:
            self.resolution_distill_module = None
        
        # ★ SAM蒸馏模块
        self.use_sam_distill = use_sam_distill
        if use_sam_distill and sam_distill is not None:
            self.sam_distill_module = SAMDistillModule(**sam_distill)
        else:
            self.sam_distill_module = None
    
    def extract_feat(self, batch_inputs):
        """Extract features from batch_inputs and adapt FPN output to single feature map."""
        # 使用父类的骨干网络提取特征
        x = self.backbone(batch_inputs)
        
        # 如果有neck（FPN），处理多尺度特征
        if self.with_neck:
            x = self.neck(x)
            # ★ 优雅的多尺度特征融合策略
            if isinstance(x, (list, tuple)):
                x = self._fuse_multi_scale_features(x)
        
        return x
    
    def _fuse_multi_scale_features(self, features):
        """
        多尺度特征融合 - FPN特征金字塔融合
        将所有尺度的特征上采样到最高分辨率，然后融合
        
        Args:
            features: FPN输出的多尺度特征 [(B,C,H1,W1), (B,C,H2,W2), ...]
        
        Returns:
            fused_feature: 融合后的特征 (B,C,H1,W1)
        """
        import torch.nn.functional as F
        
        # 使用最高分辨率（第一个）作为目标尺寸
        target_size = features[0].shape[-2:]  # (H, W)
        
        # 将所有特征上采样到相同尺寸
        upsampled_features = []
        for i, feat in enumerate(features):
            if feat.shape[-2:] != target_size:
                # 双线性插值上采样
                feat_up = F.interpolate(
                    feat,
                    size=target_size,
                    mode='bilinear',
                    align_corners=False
                )
            else:
                feat_up = feat
            upsampled_features.append(feat_up)
        
        # ★ 根据配置选择融合策略
        if self.use_learnable_fusion and hasattr(self, 'fusion_weights'):
            # 可学习加权融合
            # 确保权重数量与特征数量匹配
            num_features = len(upsampled_features)
            if self.fusion_weights.shape[0] != num_features:
                # 如果数量不匹配，动态调整（只在第一次forward时发生）
                import torch.nn as nn
                self.fusion_weights = nn.Parameter(torch.ones(num_features).to(self.fusion_weights.device))
            
            weights = F.softmax(self.fusion_weights, dim=0)
            fused = sum(w * f for w, f in zip(weights, upsampled_features))
            
            # 可选：打印权重（用于调试）
            # print(f"Fusion weights: {weights.detach().cpu().numpy()}")
        else:
            # 简单平均融合（稳定的baseline）
            fused = sum(upsampled_features) / len(upsampled_features)
        
        return fused
    
    def loss(self, batch_inputs: torch.Tensor,
             batch_data_samples: List) -> Dict[str, torch.Tensor]:
        """
        重写loss方法以支持分辨率蒸馏和SAM蒸馏。
        
        Args:
            batch_inputs: 输入图像，形状为 (N, C, H, W)
            batch_data_samples: 数据样本列表
        
        Returns:
            dict: 包含所有损失的字典
        """
        from mmdet.structures import TrackDataSample
        
        # 处理视频数据
        if isinstance(batch_data_samples[0], TrackDataSample):
            bs, num_frames, three, h, w = batch_inputs.shape
            assert three == 3, "Only supporting images with 3 channels."
            x = batch_inputs.reshape((bs * num_frames, three, h, w))
        else:
            x = batch_inputs
        
        # 分辨率蒸馏：如果启用，需要处理低分辨率和高分辨率两个路径
        if self.use_resolution_distill and self.training:
            # 创建低分辨率输入（尺寸减半）
            # 使用scale_factor而不是固定尺寸，避免尺寸不匹配问题
            x_low = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False, recompute_scale_factor=True)
            
            # 提取低分辨率特征
            feats_low = self.extract_feat(x_low)
            if isinstance(feats_low, (list, tuple)):
                feats_low = feats_low[0] if len(feats_low) > 0 else feats_low
            
            # 提取高分辨率特征
            feats_high = self.extract_feat(x)
            if isinstance(feats_high, (list, tuple)):
                feats_high = feats_high[0] if len(feats_high) > 0 else feats_high
            
            # 计算分辨率蒸馏损失
            resolution_distill_losses = self.resolution_distill_module(
                seg_feat=feats_high,
                sr_feat=feats_low,
                target_image=x
            )
        else:
            resolution_distill_losses = {}
            feats_high = self.extract_feat(x)
        
        # 标准损失计算
        if isinstance(batch_data_samples[0], TrackDataSample):
            feats = feats_high.reshape((bs, num_frames, *feats_high.shape[1:]))
            feats = feats.flatten(0, 1)
        else:
            feats = feats_high
        
        # 计算标准损失（这会内部调用forward）
        losses = self.panoptic_head.loss(feats, batch_data_samples)
        
        # 获取head输出用于SAM蒸馏（需要重新forward，但可以优化）
        # 注意：这里可能会重复计算，但为了获取输出用于蒸馏是必要的
        # 在实际使用中可以考虑缓存forward结果
        if self.use_sam_distill and self.training and self.sam_distill_module is not None:
            all_cls_scores, all_mask_preds = self.panoptic_head.forward(feats, batch_data_samples)
        else:
            all_cls_scores, all_mask_preds = None, None
        
        # SAM蒸馏损失
        if self.use_sam_distill and self.training and self.sam_distill_module is not None:
            # 获取多尺度特征（用于特征蒸馏）
            if self.with_neck:
                # 如果使用neck，获取多尺度特征
                backbone_feats = self.backbone(x)
                if isinstance(backbone_feats, (list, tuple)):
                    neck_feats = self.neck(backbone_feats)
                    if isinstance(neck_feats, (list, tuple)):
                        student_feats = neck_feats
                    else:
                        student_feats = [neck_feats]
                else:
                    student_feats = [backbone_feats]
            else:
                backbone_feats = self.backbone(x)
                student_feats = [backbone_feats] if not isinstance(backbone_feats, (list, tuple)) else backbone_feats
            
            # 计算SAM蒸馏损失
            sam_distill_losses = self.sam_distill_module(
                student_feats=student_feats,
                student_cls_scores=all_cls_scores[-1] if len(all_cls_scores) > 0 else None,
                student_mask_preds=all_mask_preds[-1] if len(all_mask_preds) > 0 else None,
                batch_inputs=x
            )
            losses.update(sam_distill_losses)
        
        # 添加分辨率蒸馏损失
        losses.update(resolution_distill_losses)
        
        return losses