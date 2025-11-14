"""
SAM模型包装器，用于在MMDetection框架中使用SAM作为教师模型
"""
import sys
import os
from pathlib import Path
import torch
import torch.nn as nn
from typing import List, Tuple, Optional

# 添加SAM项目路径到sys.path
sam_path = Path(__file__).parent.parent.parent.parent / 'segment-anything-main'
if str(sam_path) not in sys.path:
    sys.path.insert(0, str(sam_path))

try:
    from segment_anything import sam_model_registry
    from segment_anything.modeling import Sam
except ImportError:
    sam_model_registry = None
    Sam = None

from mmdet.registry import MODELS
from mmdet.utils import ConfigType


@MODELS.register_module()
class SAMBackbone(nn.Module):
    """SAM模型包装器，用于知识蒸馏。
    
    将SAM模型包装为可以在MMDetection框架中使用的backbone。
    主要用于提取图像特征用于知识蒸馏。
    
    Args:
        model_type (str): SAM模型类型，可选 'vit_h', 'vit_l', 'vit_b'
        checkpoint (str): SAM模型checkpoint路径
        freeze (bool): 是否冻结模型参数，默认True（用于教师模型）
    """
    
    def __init__(
        self,
        model_type: str = 'vit_h',
        checkpoint: Optional[str] = None,
        freeze: bool = True,
        **kwargs
    ):
        super(SAMBackbone, self).__init__()
        
        if sam_model_registry is None:
            raise ImportError(
                "Cannot import segment_anything. Please ensure segment-anything-main "
                "is in the project root directory."
            )
        
        # 构建SAM模型
        if checkpoint is None:
            raise ValueError("checkpoint must be provided for SAM model")
        
        self.sam_model = sam_model_registry[model_type](checkpoint=checkpoint)
        
        # 冻结模型参数（用于教师模型）
        if freeze:
            for param in self.sam_model.parameters():
                param.requires_grad = False
            self.sam_model.eval()
        
        self.model_type = model_type
        self.checkpoint = checkpoint
    
    def extract_feat(self, img: torch.Tensor) -> List[torch.Tensor]:
        """
        提取图像特征（仅使用image_encoder部分）。
        
        Args:
            img: 输入图像，形状为 (B, C, H, W)，已经归一化
        
        Returns:
            List[torch.Tensor]: 多尺度特征列表
                注意：SAM的image_encoder只输出单尺度特征，这里返回列表以兼容接口
        """
        # SAM的image_encoder期望输入是1024x1024，且需要正确的归一化
        # 显存优化：如果输入已经是较小尺寸，直接使用；否则resize到较小尺寸
        import torch.nn.functional as F
        
        with torch.no_grad() if not self.training else torch.enable_grad():
            # 获取SAM期望的输入尺寸（通常是1024）
            target_size = self.sam_model.image_encoder.img_size
            
            # 显存优化：如果输入尺寸较大，先resize到较小尺寸（如512）再padding到1024
            # 这样可以减少中间特征图的大小
            h, w = img.shape[-2:]
            if h > 512 or w > 512:
                # 先resize到512以节省显存
                img_resized = F.interpolate(
                    img,
                    size=(512, 512),
                    mode='bilinear',
                    align_corners=False
                )
            else:
                img_resized = img
            
            # 由于data_preprocessor和SAM使用相同的归一化参数，我们可以直接padding
            # 但为了安全，我们使用SAM的preprocess方法（它会处理归一化和padding）
            # 由于输入已经归一化，我们需要先反归一化
            pixel_mean = self.sam_model.pixel_mean.view(1, -1, 1, 1)
            pixel_std = self.sam_model.pixel_std.view(1, -1, 1, 1)
            
            # 反归一化：img_unnormalized = img * std + mean
            img_unnormalized = img_resized * pixel_std + pixel_mean
            
            # 使用SAM的preprocess方法（归一化+padding到1024x1024）
            img_preprocessed = self.sam_model.preprocess(img_unnormalized)
            
            # 使用SAM的image_encoder提取特征
            image_embeddings = self.sam_model.image_encoder(img_preprocessed)
            # SAM的image_encoder输出形状为 (B, C, H, W)
            # 为了兼容多尺度特征接口，返回列表
            return [image_embeddings]
    
    def forward(self, img: torch.Tensor) -> List[torch.Tensor]:
        """前向传播，提取特征"""
        return self.extract_feat(img)
    
    def train(self, mode: bool = True):
        """设置训练/评估模式"""
        if hasattr(self, 'sam_model'):
            # 如果是冻结的教师模型，强制保持eval模式
            if not any(p.requires_grad for p in self.sam_model.parameters()):
                self.sam_model.eval()
            else:
                self.sam_model.train(mode)
        return super().train(mode)


@MODELS.register_module()
class SAMTeacherModel(nn.Module):
    """完整的SAM教师模型包装器。
    
    用于知识蒸馏，可以提取特征和输出预测。
    
    Args:
        model_type (str): SAM模型类型
        checkpoint (str): SAM模型checkpoint路径
        freeze (bool): 是否冻结模型参数
    """
    
    def __init__(
        self,
        model_type: str = 'vit_h',
        checkpoint: Optional[str] = None,
        freeze: bool = True,
        **kwargs
    ):
        super(SAMTeacherModel, self).__init__()
        
        if sam_model_registry is None:
            raise ImportError(
                "Cannot import segment_anything. Please ensure segment-anything-main "
                "is in the project root directory."
            )
        
        # 构建SAM模型
        if checkpoint is None:
            raise ValueError("checkpoint must be provided for SAM model")
        
        self.sam_model = sam_model_registry[model_type](checkpoint=checkpoint)
        
        # 冻结模型参数
        if freeze:
            for param in self.sam_model.parameters():
                param.requires_grad = False
            self.sam_model.eval()
        
        self.model_type = model_type
        self.checkpoint = checkpoint
    
    def extract_feat(self, img: torch.Tensor) -> List[torch.Tensor]:
        """
        提取图像特征（仅使用image_encoder部分）。
        
        Args:
            img: 输入图像，形状为 (B, C, H, W)，已经归一化
        
        Returns:
            List[torch.Tensor]: 多尺度特征列表
                注意：SAM的image_encoder只输出单尺度特征，这里返回列表以兼容接口
        """
        # SAM的image_encoder期望输入是1024x1024，且需要正确的归一化
        # 显存优化：如果输入已经是较小尺寸，直接使用；否则resize到较小尺寸
        import torch.nn.functional as F
        
        with torch.no_grad() if not self.training else torch.enable_grad():
            # 获取SAM期望的输入尺寸（通常是1024）
            target_size = self.sam_model.image_encoder.img_size
            
            # 显存优化：如果输入尺寸较大，先resize到较小尺寸（如512）再padding到1024
            # 这样可以减少中间特征图的大小
            h, w = img.shape[-2:]
            if h > 512 or w > 512:
                # 先resize到512以节省显存
                img_resized = F.interpolate(
                    img,
                    size=(512, 512),
                    mode='bilinear',
                    align_corners=False
                )
            else:
                img_resized = img
            
            # 由于data_preprocessor和SAM使用相同的归一化参数，我们可以直接padding
            # 但为了安全，我们使用SAM的preprocess方法（它会处理归一化和padding）
            # 由于输入已经归一化，我们需要先反归一化
            pixel_mean = self.sam_model.pixel_mean.view(1, -1, 1, 1)
            pixel_std = self.sam_model.pixel_std.view(1, -1, 1, 1)
            
            # 反归一化：img_unnormalized = img * std + mean
            img_unnormalized = img_resized * pixel_std + pixel_mean
            
            # 使用SAM的preprocess方法（归一化+padding到1024x1024）
            img_preprocessed = self.sam_model.preprocess(img_unnormalized)
            
            # 使用SAM的image_encoder提取特征
            image_embeddings = self.sam_model.image_encoder(img_preprocessed)
            # SAM的image_encoder输出形状为 (B, C, H, W)
            # 为了兼容多尺度特征接口，返回列表
            return [image_embeddings]
    
    def forward(self, img: torch.Tensor) -> List[torch.Tensor]:
        """前向传播"""
        return self.extract_feat(img)
    
    def train(self, mode: bool = True):
        """设置训练/评估模式"""
        if hasattr(self, 'sam_model'):
            if not any(p.requires_grad for p in self.sam_model.parameters()):
                self.sam_model.eval()
            else:
                self.sam_model.train(mode)
        return super().train(mode)

