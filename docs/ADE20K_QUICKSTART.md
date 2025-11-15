# ADE20K全景分割 - 快速开始

## 🚀 一键开始

### 步骤1: 验证数据集
```bash
cd /9950backfile/zhangyafei/RPS
python3 tools/check_ade_files.py
```

### 步骤2: 分类器已就绪 ✅
**好消息**: 分类器文件已经存在并复制到正确位置！
- 文件: `~/.cache/embd/convnext_large_d_320_ADEPanopticOVDataset.pth`
- 大小: 5.3 MB (150类别, 768维特征)
- 无需重新生成，可以直接训练！

验证分类器（可选）：
```bash
python -c "import torch; cls=torch.load('${HOME}/.cache/embd/convnext_large_d_320_ADEPanopticOVDataset.pth'); print(f'✓ Shape: {cls.shape}')"
```

### 步骤3: 开始训练

**单GPU训练**:
```bash
./train_ade20k.sh 1
```

**4个GPU训练**:
```bash
./train_ade20k.sh 4
```

**8个GPU训练**:
```bash
./train_ade20k.sh 8
```

### 步骤4: 测试模型
```bash
# 单GPU测试
./test_ade20k.sh work_dirs/regnet_ade20k/best_ade_panoptic_PQ_epoch_XX.pth

# 8个GPU测试
./test_ade20k.sh work_dirs/regnet_ade20k/best_ade_panoptic_PQ_epoch_XX.pth 8
```

## 📋 已完成的配置

✅ **数据集配置**
- 数据路径: `/9950backfile/zhangyafei/ade/`
- 训练集: 20,210 张图像
- 验证集: 2,000 张图像
- 类别: 100 things + 50 stuff = 150 类

✅ **模型配置**
- Backbone: RegNet-X 12GF (预训练)
- Neck: FPN (4层特征金字塔)
- Head: RapSAMVideoHead
- 创新点:
  - ✅ 可学习特征融合 (已启用)
  - ✅ 分辨率蒸馏 (已启用)
  - ⏸️ SAM蒸馏 (默认关闭，可选启用)

✅ **训练配置**
- Batch size: 2 per GPU
- Learning rate: 2e-4 (AdamW)
- Epochs: 50
- 自动学习率缩放: 启用
- Checkpoint保存: 每个epoch + 最佳模型

## 📂 项目文件

### 核心配置文件
- `configs/reg_sam/regnet_ade20k.py` - ADE20K训练配置
- `configs/reg_sam/gen_ade20k_classifier.py` - 生成分类器配置 ⭐
- `configs/_base_/datasets/ade_panoptic_ov.py` - 数据集配置

### 脚本工具
- `gen_ade20k_classifier.sh` - 生成分类器脚本 ⭐ **首次必须运行**
- `train_ade20k.sh` - 训练启动脚本
- `test_ade20k.sh` - 测试评估脚本
- `tools/check_ade_files.py` - 数据集验证工具
- `tools/gen_cls.py` - 分类器生成工具

### 文档
- `ADE20K_QUICKSTART.md` - 本文档
- `ADE20K_TRAINING_GUIDE.md` - 详细训练指南
- `GENERATE_CLASSIFIER_GUIDE.md` - 分类器生成指南 ⭐
- `README.md` - 项目主文档
- `MEMORY_OPTIMIZATION.md` - 显存优化指南

## ⚙️ 高级配置

### 从检查点恢复训练
```bash
python tools/train.py configs/reg_sam/regnet_ade20k.py \
    --resume work_dirs/regnet_ade20k/latest.pth
```

### 使用COCO预训练模型
```bash
python tools/train.py configs/reg_sam/regnet_ade20k.py \
    --cfg-options load_from=work_dirs/regnet_coco/latest.pth
```

### 调整Batch Size
```bash
./train_ade20k.sh 8 --cfg-options train_dataloader.batch_size=1
```

### 启用SAM蒸馏

1. 修改 `configs/reg_sam/regnet_ade20k.py`:
```python
use_sam_distill = True
```

2. 确保SAM checkpoint存在:
```bash
ls -lh /9950backfile/zhangyafei/checkpoint/sam_vit_h_4b8939.pth
```

## 📊 监控训练

### 查看日志
```bash
tail -f work_dirs/regnet_ade20k/$(date +%Y%m%d)_*/vis_data/scalars.json
```

### TensorBoard
```bash
tensorboard --logdir work_dirs/regnet_ade20k/
```

### 检查Checkpoint
```bash
ls -lht work_dirs/regnet_ade20k/*.pth
```

## 🎯 预期训练时间

基于V100/A100 GPU的估算：

| GPU配置 | Batch Size | 每个Epoch时间 | 50 Epochs总时间 |
|---------|-----------|--------------|----------------|
| 1x GPU  | 2         | ~2-3小时      | ~5-6天         |
| 4x GPU  | 8         | ~30-45分钟    | ~1-1.5天       |
| 8x GPU  | 16        | ~15-25分钟    | ~12-20小时     |

*注：实际时间取决于GPU型号和配置*

## 🔧 常见问题快速解决

### OOM (显存不足)
```bash
# 方案1: 减小batch size
./train_ade20k.sh 8 --cfg-options train_dataloader.batch_size=1

# 方案2: 禁用分辨率蒸馏
# 修改配置文件: use_resolution_distill = False
```

### 训练速度慢
```bash
# 增加workers数量
./train_ade20k.sh 8 --cfg-options train_dataloader.num_workers=8
```

### 找不到模块
```bash
cd /9950backfile/zhangyafei/RPS
pip install -e .
```

## 📈 评估指标

训练完成后，关注以下指标：

- **PQ (Panoptic Quality)**: 全景分割总体质量 ⬆️
- **SQ (Segmentation Quality)**: 分割质量 ⬆️  
- **RQ (Recognition Quality)**: 识别质量 ⬆️
- **mIoU**: 平均交并比 ⬆️

## 🎨 可视化结果

```bash
python demo/demo.py \
    configs/reg_sam/regnet_ade20k.py \
    work_dirs/regnet_ade20k/best_ade_panoptic_PQ_epoch_XX.pth \
    /path/to/test/images \
    --out-dir results/ade20k/
```

## 📞 获取帮助

- 详细训练指南: `ADE20K_TRAINING_GUIDE.md`
- 显存优化: `MEMORY_OPTIMIZATION.md`
- SAM蒸馏: `SAM_DISTILL_SETUP.md`
- 项目主页: `README.md`

## ✅ 检查清单

开始训练前，请确认：

- [ ] 数据集验证通过 (`python3 tools/check_ade_files.py`)
- [x] **分类器已就绪** (已复制到 `~/.cache/embd/`) ✅
- [ ] 配置文件存在 (`configs/reg_sam/regnet_ade20k.py`)
- [ ] 有足够的磁盘空间 (建议 >100GB)
- [ ] GPU显存充足 (建议 >16GB per GPU)
- [ ] 已安装所有依赖 (`pip install -r requirements.txt`)

全部确认后，运行:
```bash
# 直接开始训练（分类器已就绪）
./train_ade20k.sh 8  # 使用8个GPU
```

祝训练顺利! 🚀

