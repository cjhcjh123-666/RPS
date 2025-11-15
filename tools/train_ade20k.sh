#!/bin/bash
# ADE20K全景分割训练脚本

# 设置工作目录
cd /9950backfile/zhangyafei/RPS

# 配置文件
CONFIG="configs/reg_sam/regnet_ade20k.py"

# 检查配置文件是否存在
if [ ! -f "$CONFIG" ]; then
    echo "错误: 配置文件不存在: $CONFIG"
    exit 1
fi

# 解析命令行参数
GPU_NUM=${1:-1}  # 默认使用1个GPU
EXTRA_ARGS="${@:2}"  # 其他额外参数

echo "=========================================="
echo "ADE20K全景分割训练"
echo "=========================================="
echo "配置文件: $CONFIG"
echo "GPU数量: $GPU_NUM"
echo "环境: mmdet_2"
echo "额外参数: $EXTRA_ARGS"
echo "=========================================="

# 验证数据集
echo "验证数据集..."
python3 tools/check_ade_files.py
if [ $? -ne 0 ]; then
    echo "错误: 数据集验证失败"
    exit 1
fi
echo ""

# 开始训练
if [ "$GPU_NUM" -eq 1 ]; then
    # 单GPU训练
    echo "开始单GPU训练..."
    python tools/train.py $CONFIG $EXTRA_ARGS
else
    # 多GPU分布式训练
    echo "开始${GPU_NUM}个GPU分布式训练..."
    bash tools/dist_train.sh $CONFIG $GPU_NUM $EXTRA_ARGS
fi

echo ""
echo "=========================================="
echo "训练完成!"
echo "检查点保存在: work_dirs/regnet_ade20k/"
echo "=========================================="

