"""
LSK (Large Selective Kernel) Module
大选择核注意力模块 - 纯卷积实现

Paper: Large Selective Kernel Network for Remote Sensing Object Detection (ICCV 2023)
Github: https://github.com/zcablii/Large-Selective-Kernel-Network

特点：
- 纯卷积实现，无Self-Attention
- 增加感受野，适合多尺度物体
- 自适应选择不同尺度的特征
- 适合全景分割任务
"""
import torch
import torch.nn as nn


class LSKblock(nn.Module):
    """
    大选择核模块
    
    Args:
        dim (int): 输入输出通道数
    
    工作原理:
        1. 使用两个不同尺度的卷积捕获多尺度特征
        2. 通过空间注意力自适应选择不同尺度
        3. 输出特征 = 输入 * 注意力权重（残差连接）
    """
    def __init__(self, dim):
        super().__init__()
        # 深度可分离卷积，卷积核大小为9
        self.conv0 = nn.Conv2d(dim, dim, 9, padding=4, groups=dim)
        
        # 空间卷积，卷积核大小为11，膨胀率为3，增加感受野
        self.conv_spatial = nn.Conv2d(dim, dim, 11, stride=1, padding=15, groups=dim, dilation=3)
        
        # 1x1卷积，用于降维
        self.conv1 = nn.Conv2d(dim, dim // 2, 1)
        self.conv2 = nn.Conv2d(dim, dim // 2, 1)
        
        # 结合平均和最大注意力的卷积
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        
        # 最后的1x1卷积，将通道数恢复到原始维度
        self.conv = nn.Conv2d(dim // 2, dim, 1)

    def forward(self, x):
        # 对输入进行两种不同的卷积操作以生成注意力特征
        attn1 = self.conv0(x)  # 第一个卷积特征 (9x9)
        attn2 = self.conv_spatial(attn1)  # 空间卷积特征 (11x11, dilation=3)

        # 对卷积特征进行1x1卷积以降维
        attn1 = self.conv1(attn1)
        attn2 = self.conv2(attn2)

        # 将两个特征在通道维度上拼接
        attn = torch.cat([attn1, attn2], dim=1)
        
        # 计算平均注意力特征
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        
        # 计算最大注意力特征
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        
        # 拼接平均和最大注意力特征
        agg = torch.cat([avg_attn, max_attn], dim=1)
        
        # 通过卷积生成注意力权重，并应用sigmoid激活函数
        sig = self.conv_squeeze(agg).sigmoid()
        
        # 根据注意力权重调整特征
        attn = attn1 * sig[:, 0, :, :].unsqueeze(1) + \
               attn2 * sig[:, 1, :, :].unsqueeze(1)
        
        # 最终卷积恢复到原始通道数
        attn = self.conv(attn)
        
        # 通过注意力特征加权原输入（残差连接）
        return x * attn


if __name__ == '__main__':
    # 测试LSK
    print("Testing LSKblock...")
    lsk = LSKblock(256).cuda()
    x = torch.rand(2, 256, 64, 128).cuda()
    y = lsk(x)
    print(f"Input: {x.shape}, Output: {y.shape}")
    print(f"✓ LSKblock works correctly!")

