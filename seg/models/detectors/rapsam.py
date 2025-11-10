import torch
from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig


from mmdet.models.detectors import SingleStageDetector
from .mask2former_vid import Mask2formerVideo

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