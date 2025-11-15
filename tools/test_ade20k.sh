#!/bin/bash
# ADE20K全景分割测试/评估脚本

cd /9950backfile/zhangyafei/RPS

CONFIG="configs/reg_sam/regnet_ade20k.py"

# 检查参数
if [ -z "$1" ]; then
    echo "用法: $0 <checkpoint_path> [gpu_num]"
    echo ""
    echo "示例:"
    echo "  单GPU测试: $0 work_dirs/regnet_ade20k/epoch_50.pth"
    echo "  多GPU测试: $0 work_dirs/regnet_ade20k/epoch_50.pth 8"
    echo ""
    echo "可用的checkpoints:"
    if [ -d "work_dirs/regnet_ade20k" ]; then
        ls -lh work_dirs/regnet_ade20k/*.pth 2>/dev/null || echo "  未找到checkpoint文件"
    else
        echo "  work_dirs/regnet_ade20k/ 目录不存在"
    fi
    exit 1
fi

CHECKPOINT=$1
GPU_NUM=${2:-1}

# 检查checkpoint是否存在
if [ ! -f "$CHECKPOINT" ]; then
    echo "错误: Checkpoint文件不存在: $CHECKPOINT"
    exit 1
fi

echo "=========================================="
echo "ADE20K全景分割测试"
echo "=========================================="
echo "配置文件: $CONFIG"
echo "Checkpoint: $CHECKPOINT"
echo "GPU数量: $GPU_NUM"
echo "环境: mmdet_2"
echo "=========================================="
echo ""

if [ "$GPU_NUM" -eq 1 ]; then
    # 单GPU测试
    echo "开始单GPU测试..."
    python tools/test.py $CONFIG $CHECKPOINT
else
    # 多GPU测试
    echo "开始${GPU_NUM}个GPU测试..."
    bash tools/dist_test.sh $CONFIG $CHECKPOINT $GPU_NUM
fi

echo ""
echo "=========================================="
echo "测试完成!"
echo "=========================================="

