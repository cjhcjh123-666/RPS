"""
Feature Alignment Loss (FA Loss) for Resolution Distillation Learning
参考DSRL-main的实现，用于对齐低分辨率和高分辨率路径的特征
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FALoss(nn.Module):
    """Feature Alignment Loss for dual super-resolution learning.
    
    该损失函数用于对齐语义分割路径和超分辨率路径的特征表示。
    通过计算两个特征图之间的Gram矩阵差异来对齐特征分布。
    
    Args:
        subscale (float): 下采样比例，用于降低计算复杂度。默认0.0625 (1/16)
    """
    
    def __init__(self, subscale=0.0625):
        super(FALoss, self).__init__()
        self.subscale = int(1 / subscale)
    
    def forward(self, feature1, feature2):
        """
        计算两个特征图之间的对齐损失。
        
        Args:
            feature1 (Tensor): 第一个特征图，形状为 (B, C, H, W)
            feature2 (Tensor): 第二个特征图，形状为 (B, C, H, W)
        
        Returns:
            Tensor: 标量损失值
        """
        # 使用平均池化下采样以降低计算复杂度
        feature1 = F.avg_pool2d(feature1, self.subscale)
        feature2 = F.avg_pool2d(feature2, self.subscale)
        
        m_batchsize, C, height, width = feature1.size()
        
        # 将特征图重塑为 (B, C, H*W)
        feature1 = feature1.view(m_batchsize, -1, width * height)  # [N, C, W*H]
        feature2 = feature2.view(m_batchsize, -1, width * height)  # [N, C, W*H]
        
        # 计算Gram矩阵: G = F^T * F，形状为 (B, H*W, H*W)
        mat1 = torch.bmm(feature1.permute(0, 2, 1), feature1)  # [N, W*H, W*H]
        mat2 = torch.bmm(feature2.permute(0, 2, 1), feature2)  # [N, W*H, W*H]
        
        # 计算L1范数损失
        L1norm = torch.norm(mat2 - mat1, p=1)
        
        # 归一化损失
        return L1norm / ((height * width) ** 2)

