#!/bin/bash
# 测试导入是否正常

cd /9950backfile/zhangyafei/RPS

echo "=========================================="
echo "测试环境和导入"
echo "=========================================="

# 激活conda环境
echo "1. 激活conda环境 mmdet_2..."
eval "$(conda shell.bash hook)"
conda activate mmdet_2

# 检查Python版本
echo ""
echo "2. Python版本:"
python --version

# 检查conda环境
echo ""
echo "3. 当前conda环境:"
echo $CONDA_DEFAULT_ENV

# 测试导入ext.templates
echo ""
echo "4. 测试导入 ext.templates..."
python -c "from ext.templates import VILD_PROMPT; print(f'✓ 成功导入 VILD_PROMPT, 包含 {len(VILD_PROMPT)} 个模板')"

# 测试导入mmdet
echo ""
echo "5. 测试导入 mmdet..."
python -c "import mmdet; print(f'✓ mmdet 版本: {mmdet.__version__}')"

# 测试导入数据集
echo ""
echo "6. 测试导入 ADEPanopticOVDataset..."
python -c "from seg.datasets.ade_ov import ADEPanopticOVDataset; print(f'✓ 成功导入 ADEPanopticOVDataset'); print(f'  Dataset name: {ADEPanopticOVDataset.dataset_name}')"

# 测试导入OpenCLIPBackbone
echo ""
echo "7. 测试导入 OpenCLIPBackbone..."
python -c "from seg.models.backbones.openclip_backbone import OpenCLIPBackbone; print('✓ 成功导入 OpenCLIPBackbone')" 2>&1 | head -3

echo ""
echo "=========================================="
echo "✓ 所有测试完成!"
echo "=========================================="
echo ""
echo "现在可以运行:"
echo "  ./gen_ade20k_classifier.sh"

