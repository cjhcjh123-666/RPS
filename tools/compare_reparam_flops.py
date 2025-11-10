#!/usr/bin/env python
"""
对比 BN 重参数化前后的 FLOPs 和参数量
"""
import argparse
import sys
import os
from pathlib import Path

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
from mmengine.config import Config
from mmengine.logging import MMLogger
from mmengine.model import revert_sync_batchnorm
from mmengine.registry import init_default_scope
from mmdet.registry import MODELS

#  关键：导入自定义模块以确保注册到 MODELS
import seg.models.backbones  # 注册 RegNetReparam
import seg.models.hooks  # 注册 ReparamHook
import seg.models.detectors  # 注册自定义 detectors

try:
    from mmengine.analysis import get_model_complexity_info
    from mmengine.analysis.print_helper import _format_size
except ImportError:
    raise ImportError('Please upgrade mmengine >= 0.6.0')


def count_parameters(model):
    """统计模型参数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def count_bn_layers(model, prefix=''):
    """统计 BN 层数量"""
    bn_count = 0
    bn_params = 0
    
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d, torch.nn.SyncBatchNorm)):
            bn_count += 1
            # BN 层参数: weight + bias + running_mean + running_var
            if hasattr(module, 'weight') and module.weight is not None:
                bn_params += module.weight.numel()  # gamma
            if hasattr(module, 'bias') and module.bias is not None:
                bn_params += module.bias.numel()  # beta
            if hasattr(module, 'running_mean') and module.running_mean is not None:
                bn_params += module.running_mean.numel()
            if hasattr(module, 'running_var') and module.running_var is not None:
                bn_params += module.running_var.numel()
    
    return bn_count, bn_params


def analyze_model(config_path, use_reparam=False):
    """分析模型的 FLOPs 和参数量"""
    logger = MMLogger.get_instance(name='MMLogger')
    
    cfg = Config.fromfile(config_path)
    init_default_scope(cfg.get('default_scope', 'mmdet'))
    
    # 构建模型
    model = MODELS.build(cfg.model)
    if torch.cuda.is_available():
        model = model.cuda()
    model = revert_sync_batchnorm(model)
    model.eval()
    
    # 统计融合前的信息
    total_params_before, trainable_params_before = count_parameters(model)
    bn_count_before, bn_params_before = count_bn_layers(model)
    
    print(f"\n{'='*60}")
    print(f"{'融合后 (Reparameterized)' if use_reparam else '融合前 (Original)'}")
    print(f"{'='*60}")
    
    # 如果需要重参数化，手动调用融合
    if use_reparam:
        if hasattr(model, 'backbone'):
            backbone_type = type(model.backbone).__name__
            print(f"\n Backbone 类型: {backbone_type}")
            print(f" Backbone 类: {type(model.backbone)}")
            print(f" 可用方法: {[m for m in dir(model.backbone) if 'switch' in m.lower() or 'deploy' in m.lower() or 'reparam' in m.lower()]}")
            
            if hasattr(model.backbone, 'switch_to_deploy'):
                print("\n 执行 BN 融合...")
                model.backbone.switch_to_deploy()
                print(" BN 融合完成！\n")
            else:
                print(f"\n Backbone ({backbone_type}) 不支持重参数化\n")
        else:
            print("\n 模型没有 backbone 属性\n")
    
    # 统计融合后的信息
    total_params_after, trainable_params_after = count_parameters(model)
    bn_count_after, bn_params_after = count_bn_layers(model)
    
    # 计算 FLOPs (使用固定输入大小)
    input_shape = (640, 640)  # 从配置文件中的 image_size 读取
    inputs = torch.randn(1, 3, *input_shape)
    if torch.cuda.is_available():
        inputs = inputs.cuda()
    
    try:
        outputs = get_model_complexity_info(
            model,
            input_shape=input_shape,
            inputs=inputs,
            show_table=False,
            show_arch=False
        )
        flops = outputs['flops']
        params = outputs['params']
        flops_str = _format_size(flops)
        params_str = _format_size(params)
    except Exception as e:
        print(f" FLOPs 计算失败: {e}")
        flops_str = "N/A"
        params_str = f"{total_params_after / 1e6:.3f}M"
    
    # 打印结果
    print(f"📊 参数统计:")
    print(f"  - 总参数量: {params_str} ({total_params_after:,} 个)")
    print(f"  - 可训练参数: {trainable_params_after / 1e6:.3f}M ({trainable_params_after:,} 个)")
    print(f"  - BN 层数量: {bn_count_after} 个")
    print(f"  - BN 参数量: {bn_params_after / 1e3:.2f}K ({bn_params_after:,} 个)")
    print(f"\n📈 计算量:")
    print(f"  - FLOPs: {flops_str}")
    print(f"  - 输入尺寸: {input_shape}")
    
    return {
        'total_params': total_params_after,
        'trainable_params': trainable_params_after,
        'bn_count': bn_count_after,
        'bn_params': bn_params_after,
        'flops': flops_str,
    }


def main():
    parser = argparse.ArgumentParser(description='对比 BN 重参数化前后的模型性能')
    parser.add_argument('config', help='配置文件路径')
    args = parser.parse_args()
    
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"❌ 配置文件不存在: {config_path}")
        return
    
    # 分析融合前的模型
    print("\n" + "🔍 正在分析原始模型（未融合 BN）...")
    results_before = analyze_model(args.config, use_reparam=False)
    
    # 分析融合后的模型
    print("\n" + "🔍 正在分析重参数化模型（融合 BN）...")
    results_after = analyze_model(args.config, use_reparam=True)
    
    # 对比结果
    print(f"\n{'='*60}")
    print("📊 重参数化效果对比")
    print(f"{'='*60}")
    
    param_reduction = results_before['total_params'] - results_after['total_params']
    bn_count_reduction = results_before['bn_count'] - results_after['bn_count']
    bn_param_reduction = results_before['bn_params'] - results_after['bn_params']
    
    print(f"\n 参数量减少:")
    print(f"  - 总参数减少: {param_reduction / 1e3:.2f}K ({param_reduction:,} 个)")
    print(f"  - BN 层减少: {bn_count_reduction} 个")
    print(f"  - BN 参数减少: {bn_param_reduction / 1e3:.2f}K ({bn_param_reduction:,} 个)")
    
    if param_reduction > 0:
        reduction_rate = (param_reduction / results_before['total_params']) * 100
        print(f"  - 参数量减少比例: {reduction_rate:.2f}%")
    
    print(f"\n💡 结论:")
    print(f"  - BN 融合后，参数量从 {results_before['total_params']/1e6:.3f}M 降至 {results_after['total_params']/1e6:.3f}M")
    print(f"  - 推理速度将提升（减少了 {bn_count_reduction} 个 BN 层的计算）")
    print(f"  - 更适合部署到 NPU/移动端设备")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()

