# author Jiahui Chen <chenjiahui2025@ia.ac.cn>
"""
支持 BN 融合（重参数化）的 RegNet 实现
在推理时将 Conv+BN 融合为单个 Conv，提升推理速度和部署效率
"""

import torch
import torch.nn as nn
from mmdet.models.backbones import RegNet
from mmdet.registry import MODELS


def fuse_conv_bn(conv: nn.Module, bn: nn.Module) -> nn.Conv2d:
    """
    将 Conv2d + BatchNorm2d 融合为单个 Conv2d
    
    Args:
        conv: Conv2d 层（bias=False）
        bn: BatchNorm2d 层
        
    Returns:
        融合后的 Conv2d 层（带 bias）
    """
    assert isinstance(conv, nn.Conv2d), f"Expected Conv2d, got {type(conv)}"
    assert isinstance(bn, nn.modules.batchnorm._BatchNorm), f"Expected BatchNorm, got {type(bn)}"
    assert conv.bias is None, "Conv should not have bias before fusion"
    
    # 获取 BN 参数
    bn_weight = bn.weight.data
    bn_bias = bn.bias.data
    bn_running_mean = bn.running_mean.data
    bn_running_var = bn.running_var.data
    bn_eps = bn.eps
    
    # 计算融合后的权重和偏置
    # w_fused = conv_weight * (bn_weight / sqrt(bn_running_var + eps))
    # b_fused = bn_bias - bn_running_mean * bn_weight / sqrt(bn_running_var + eps)
    bn_std = (bn_running_var + bn_eps).sqrt()
    bn_weight_normalized = bn_weight / bn_std
    
    # 扩展维度以匹配 conv weight 的形状 [out_channels, in_channels, H, W]
    bn_weight_normalized = bn_weight_normalized.view(-1, 1, 1, 1)
    
    # 融合权重
    fused_weight = conv.weight.data * bn_weight_normalized
    
    # 融合偏置
    fused_bias = bn_bias - bn_running_mean * bn_weight / bn_std
    
    # 创建新的融合卷积层
    fused_conv = nn.Conv2d(
        conv.in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=True,  # 融合后需要 bias
        padding_mode=conv.padding_mode,
        device=conv.weight.device,
        dtype=conv.weight.dtype
    )
    
    fused_conv.weight.data.copy_(fused_weight)
    fused_conv.bias.data.copy_(fused_bias)
    
    return fused_conv


def fuse_sequential_conv_bn(seq: nn.Sequential) -> nn.Module:
    """
    融合 Sequential(Conv2d, BatchNorm2d) 为单个 Conv2d
    
    Args:
        seq: nn.Sequential 包含 Conv2d 和 BatchNorm2d
        
    Returns:
        融合后的 Conv2d 或原始模块（如果无法融合）
    """
    if len(seq) == 2:
        if isinstance(seq[0], nn.Conv2d) and isinstance(seq[1], nn.modules.batchnorm._BatchNorm):
            return fuse_conv_bn(seq[0], seq[1])
    return seq




@MODELS.register_module()
class RegNetReparam(RegNet):
    """
    支持 BN 融合（重参数化）的 RegNet
    
    使用方法：
        1. 训练时：正常训练，无需特殊处理
        2. 推理前：调用 model.backbone.switch_to_deploy() 进行 BN 融合
        3. 或者在配置文件中使用 ReparamHook 自动融合
    
    示例：
        model = build_model(cfg)
        model.backbone.switch_to_deploy()
        model.eval()
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_reparam = False  # 标记是否已完成重参数化
    
    def switch_to_deploy(self):
        """
        切换到部署模式：融合所有 BN 层
        
        这个方法应该在模型训练完成后、推理前调用
        融合后可以显著提升推理速度，特别是在 NPU/移动端设备上
        """
        if self.is_reparam:
            print("RegNetReparam: 模型已经完成重参数化")
            return
        
        print("RegNetReparam: 开始 BN 融合...")
        
        # 融合 stem 层的 BN
        if hasattr(self, 'norm1'):
            if isinstance(self.norm1, nn.modules.batchnorm._BatchNorm):
                fused_conv1 = fuse_conv_bn(self.conv1, self.norm1)
                self.conv1 = fused_conv1
                # 替换 norm1 为 Identity（保留属性名以避免访问错误）
                identity = nn.Identity()
                setattr(self, self.norm1_name, identity)
                self._modules[self.norm1_name] = identity
                print(f"RegNetReparam: 融合 stem 层 BN ({self.norm1_name})")
        
        # 融合所有 ResLayer 中的 Bottleneck blocks
        for layer_name in self.res_layers:
            res_layer = getattr(self, layer_name)
            self._fuse_res_layer(res_layer)
            print(f"RegNetReparam: 融合 {layer_name} 中的所有 BN")
        
        self.is_reparam = True
        print("RegNetReparam: BN 融合完成！推理速度将显著提升。")
    
    def _fuse_res_layer(self, res_layer):
        """
        融合 ResLayer 中所有 Bottleneck 的 BN
        
        Args:
            res_layer: ResLayer 模块
        """
        for block in res_layer:
            self._fuse_bottleneck(block)
    
    def _fuse_bottleneck(self, bottleneck):
        """
        融合单个 Bottleneck 中的所有 BN
        
        Args:
            bottleneck: Bottleneck 模块
        """
        # 融合 conv1 + norm1
        if hasattr(bottleneck, 'conv1') and hasattr(bottleneck, 'norm1'):
            conv1 = bottleneck.conv1
            norm1 = bottleneck.norm1
            if isinstance(conv1, nn.Conv2d) and isinstance(norm1, nn.modules.batchnorm._BatchNorm):
                fused_conv1 = fuse_conv_bn(conv1, norm1)
                bottleneck.conv1 = fused_conv1
                # 替换 norm1 为 Identity（保留属性名以避免访问错误）
                identity = nn.Identity()
                setattr(bottleneck, bottleneck.norm1_name, identity)
                bottleneck._modules[bottleneck.norm1_name] = identity
        
        # 融合 conv2 + norm2
        if hasattr(bottleneck, 'conv2') and hasattr(bottleneck, 'norm2'):
            conv2 = bottleneck.conv2
            norm2 = bottleneck.norm2
            if isinstance(conv2, nn.Conv2d) and isinstance(norm2, nn.modules.batchnorm._BatchNorm):
                fused_conv2 = fuse_conv_bn(conv2, norm2)
                bottleneck.conv2 = fused_conv2
                # 替换 norm2 为 Identity
                identity = nn.Identity()
                setattr(bottleneck, bottleneck.norm2_name, identity)
                bottleneck._modules[bottleneck.norm2_name] = identity
        
        # 融合 conv3 + norm3
        if hasattr(bottleneck, 'conv3') and hasattr(bottleneck, 'norm3'):
            conv3 = bottleneck.conv3
            norm3 = bottleneck.norm3
            if isinstance(conv3, nn.Conv2d) and isinstance(norm3, nn.modules.batchnorm._BatchNorm):
                fused_conv3 = fuse_conv_bn(conv3, norm3)
                bottleneck.conv3 = fused_conv3
                # 替换 norm3 为 Identity
                identity = nn.Identity()
                setattr(bottleneck, bottleneck.norm3_name, identity)
                bottleneck._modules[bottleneck.norm3_name] = identity
        
        # 融合 downsample
        if hasattr(bottleneck, 'downsample') and bottleneck.downsample is not None:
            downsample = bottleneck.downsample
            if isinstance(downsample, nn.Sequential) and len(downsample) >= 2:
                # 查找 Conv2d 和 BatchNorm
                conv_idx = None
                bn_idx = None
                for i, module in enumerate(downsample):
                    if isinstance(module, nn.Conv2d) and conv_idx is None:
                        conv_idx = i
                    elif isinstance(module, nn.modules.batchnorm._BatchNorm) and bn_idx is None:
                        bn_idx = i
                
                if conv_idx is not None and bn_idx is not None:
                    # 融合 downsample 中的 BN
                    fused_downsample_conv = fuse_conv_bn(
                        downsample[conv_idx], downsample[bn_idx]
                    )
                    # 替换 Sequential
                    new_downsample = nn.Sequential()
                    for i, module in enumerate(downsample):
                        if i == conv_idx:
                            new_downsample.add_module(str(i), fused_downsample_conv)
                        elif i != bn_idx:  # 跳过 BN 层
                            new_downsample.add_module(str(i), module)
                    bottleneck.downsample = new_downsample
    


# 为了兼容性，提供与原 RegNet 相同的接口
def regnet_reparam(**kwargs):
    """创建支持重参数化的 RegNet"""
    return RegNetReparam(**kwargs)

