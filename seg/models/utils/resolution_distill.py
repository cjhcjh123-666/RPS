"""
分辨率蒸馏学习模块
参考DSRL-main，实现双路径训练：低分辨率路径（用于快速推理）和高分辨率路径（用于学习）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional
from .fa_loss import FALoss


class SuperResolutionDecoder(nn.Module):
    """超分辨率解码器，用于从低分辨率特征重建高分辨率图像。
    
    该模块将分割特征转换为RGB图像，用于分辨率蒸馏学习。
    """
    
    def __init__(self, in_channels: int = 64, out_channels: int = 3):
        super(SuperResolutionDecoder, self).__init__()
        
        # 使用转置卷积和EDSR风格的残差块进行上采样
        self.up_sr_1 = nn.ConvTranspose2d(in_channels, 64, kernel_size=2, stride=2)
        self.up_edsr_1 = self._make_edsr_block(64, 64)
        
        self.up_sr_2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.up_edsr_2 = self._make_edsr_block(32, 32)
        
        self.up_sr_3 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.up_edsr_3 = self._make_edsr_block(16, 16)
        
        self.up_conv_last = nn.Conv2d(16, out_channels, kernel_size=1)
        
    def _make_edsr_block(self, in_ch, out_ch):
        """创建EDSR风格的残差块"""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
        )
    
    def forward(self, x):
        """
        Args:
            x: 输入特征，形状为 (B, C, H, W)
        
        Returns:
            Tensor: 重建的RGB图像，形状为 (B, 3, 2*H, 2*W)
        """
        x = self.up_sr_1(x)
        x = x + self.up_edsr_1(x)  # 残差连接
        
        x = self.up_sr_2(x)
        x = x + self.up_edsr_2(x)
        
        x = self.up_sr_3(x)
        x = x + self.up_edsr_3(x)
        
        x = self.up_conv_last(x)
        return x


class ResolutionDistillModule(nn.Module):
    """分辨率蒸馏学习模块。
    
    该模块实现双路径训练策略：
    1. 低分辨率路径：使用下采样的输入图像进行快速推理
    2. 高分辨率路径：使用原始分辨率图像进行精确学习
    
    通过特征对齐损失和超分辨率重建损失来连接两个路径。
    """
    
    def __init__(
        self,
        feat_channels: int = 128,
        sr_loss_weight: float = 0.5,
        fa_loss_weight: float = 0.5,
        fa_subscale: float = 0.0625
    ):
        super(ResolutionDistillModule, self).__init__()
        
        self.sr_loss_weight = sr_loss_weight
        self.fa_loss_weight = fa_loss_weight
        
        # 超分辨率解码器
        self.sr_decoder = SuperResolutionDecoder(in_channels=feat_channels, out_channels=3)
        
        # 特征对齐损失
        self.fa_loss = FALoss(subscale=fa_subscale)
        
        # MSE损失用于超分辨率重建
        self.mse_loss = nn.MSELoss()
        
        # 将分割特征转换为RGB特征的转换层
        self.pointwise = nn.Sequential(
            nn.Conv2d(feat_channels, 3, kernel_size=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )
    
    def forward(
        self,
        seg_feat: torch.Tensor,
        sr_feat: torch.Tensor,
        target_image: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播，计算分辨率蒸馏损失。
        
        Args:
            seg_feat: 语义分割路径的特征，形状为 (B, C, H, W)
            sr_feat: 超分辨率路径的特征，形状为 (B, C, H, W)
            target_image: 目标高分辨率图像，形状为 (B, 3, 2*H, 2*W)
        
        Returns:
            dict: 包含各种损失的字典
                - 'loss_sr': 超分辨率重建损失
                - 'loss_fa': 特征对齐损失
                - 'loss_resolution_distill': 总的分辨率蒸馏损失
        """
        # 上采样分割特征到目标分辨率
        target_size = target_image.shape[-2:]
        seg_feat_up = F.interpolate(
            seg_feat,
            size=target_size,
            mode='bilinear',
            align_corners=False
        )
        
        # 超分辨率重建
        sr_reconstructed = self.sr_decoder(sr_feat)
        
        # 计算超分辨率重建损失（MSE）
        loss_sr = self.mse_loss(sr_reconstructed, target_image)
        
        # 将分割特征转换为RGB特征用于特征对齐
        # 使用中间特征进行对齐，而不是最终输出
        seg_rgb_feat = self.pointwise(seg_feat)
        
        # 获取超分辨率路径的中间特征（在第一个上采样层之后）
        sr_intermediate = self.sr_decoder.up_sr_1(sr_feat)
        sr_intermediate = self.sr_decoder.up_edsr_1(sr_intermediate)
        
        # 计算特征对齐损失
        # 需要将特征对齐到相同尺寸
        min_h = min(seg_rgb_feat.shape[2], sr_intermediate.shape[2])
        min_w = min(seg_rgb_feat.shape[3], sr_intermediate.shape[3])
        seg_rgb_feat = F.interpolate(seg_rgb_feat, size=(min_h, min_w), mode='bilinear', align_corners=False)
        sr_intermediate = F.interpolate(sr_intermediate, size=(min_h, min_w), mode='bilinear', align_corners=False)
        
        loss_fa = self.fa_loss(seg_rgb_feat, sr_intermediate)
        
        # 总损失
        loss_resolution_distill = (
            self.sr_loss_weight * loss_sr +
            self.fa_loss_weight * loss_fa
        )
        
        return {
            'loss_sr': loss_sr,
            'loss_fa': loss_fa,
            'loss_resolution_distill': loss_resolution_distill
        }

