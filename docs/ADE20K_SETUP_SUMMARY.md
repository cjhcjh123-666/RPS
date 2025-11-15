# ADE20K全景分割配置总结

## 📋 任务完成概览

✅ **已完成所有ADE20K数据集的配置和准备工作**

## 🆕 新增/修改的文件

### 1. 核心配置文件

#### `configs/reg_sam/regnet_ade20k.py` ⭐ (新增)
- ADE20K数据集的主要训练配置
- 基于regnet_coco.py改造
- 关键配置：
  - 100 things + 50 stuff classes
  - 数据路径: `/9950backfile/zhangyafei/ade/`
  - 使用RegNet backbone + FPN neck
  - 启用可学习特征融合和分辨率蒸馏
  - SAM蒸馏默认关闭（可选启用）

#### `configs/_base_/datasets/ade_panoptic_ov.py` (修改)
- 更新数据路径为: `/9950backfile/zhangyafei/ade/`
- 原路径: `/data/chenjiahui/regnet_sam_l/datasetss/ade/`

### 2. 训练和测试脚本

#### `train_ade20k.sh` (新增)
- 一键启动ADE20K训练
- 自动验证数据集
- 支持单GPU和多GPU训练
- 用法:
  ```bash
  ./train_ade20k.sh 1    # 单GPU
  ./train_ade20k.sh 4    # 4 GPUs
  ./train_ade20k.sh 8    # 8 GPUs
  ```

#### `test_ade20k.sh` (新增)
- 一键测试/评估模型
- 支持单GPU和多GPU测试
- 用法:
  ```bash
  ./test_ade20k.sh work_dirs/regnet_ade20k/epoch_50.pth
  ./test_ade20k.sh work_dirs/regnet_ade20k/epoch_50.pth 8
  ```

### 3. 验证工具

#### `tools/check_ade_files.py` (新增)
- 验证ADE20K数据集文件完整性
- 检查所有必需的目录和文件
- 显示数据集统计信息
- 用法:
  ```bash
  python3 tools/check_ade_files.py
  ```

#### `tools/test_ade_dataset.py` (新增)
- 高级数据集验证脚本（需要mmengine）
- 验证配置文件和数据集加载
- 用法:
  ```bash
  python tools/test_ade_dataset.py
  ```

### 4. 文档

#### `ADE20K_QUICKSTART.md` ⭐ (新增)
- 快速开始指南
- 一键命令汇总
- 常见问题快速解决

#### `ADE20K_TRAINING_GUIDE.md` (新增)
- 详细的训练指南
- 配置说明和调优建议
- 故障排除和性能优化

#### `DATASET_COMPARISON.md` (新增)
- COCO vs ADE20K对比
- 迁移学习策略
- 联合训练方法

#### `ADE20K_SETUP_SUMMARY.md` (本文档)
- 配置总结
- 文件清单
- 使用流程

## 📊 数据集验证结果

```
✓ 数据根目录存在: /9950backfile/zhangyafei/ade/
✓ 训练集: 20,210 张图像
✓ 验证集: 2,000 张图像
✓ 类别: 150 (100 things + 50 stuff)
✓ 全景分割标注完整
✓ 配置文件已创建
```

## 🎯 模型配置

### Backbone
- **类型**: RegNet-X 12GF
- **预训练**: ImageNet (open-mmlab)
- **输出层**: 4个特征层

### Neck
- **类型**: FPN (Feature Pyramid Network)
- **通道数**: 256
- **输出层数**: 4

### Head
- **类型**: RapSAMVideoHead
- **Things类别**: 100
- **Stuff类别**: 50
- **Query数量**: 100

### 创新点
1. ✅ **可学习特征融合**: 启用
2. ✅ **分辨率蒸馏**: 启用 (sr_target_scale=1.5)
3. ⏸️ **SAM蒸馏**: 默认关闭（可选）

## 🚀 快速开始

### 最简单的方式（推荐）

```bash
# 1. 进入项目目录
cd /9950backfile/zhangyafei/RPS

# 2. 验证数据集
python3 tools/check_ade_files.py

# 3. 开始训练（8个GPU）
./train_ade20k.sh 8

# 4. 测试模型
./test_ade20k.sh work_dirs/regnet_ade20k/best_ade_panoptic_PQ_epoch_XX.pth 8
```

### 使用Python命令

```bash
# 训练
python tools/train.py configs/reg_sam/regnet_ade20k.py

# 多GPU训练
bash tools/dist_train.sh configs/reg_sam/regnet_ade20k.py 8

# 测试
python tools/test.py \
    configs/reg_sam/regnet_ade20k.py \
    work_dirs/regnet_ade20k/epoch_50.pth
```

## 📁 项目文件结构

```
RPS/
├── configs/
│   ├── _base_/
│   │   └── datasets/
│   │       └── ade_panoptic_ov.py         # ✏️ 已修改
│   └── reg_sam/
│       ├── regnet_coco.py                 # COCO配置
│       └── regnet_ade20k.py               # 🆕 ADE20K配置
│
├── tools/
│   ├── train.py
│   ├── test.py
│   ├── check_ade_files.py                 # 🆕 数据集验证
│   └── test_ade_dataset.py                # 🆕 高级验证
│
├── train_ade20k.sh                        # 🆕 训练脚本
├── test_ade20k.sh                         # 🆕 测试脚本
│
├── ADE20K_QUICKSTART.md                   # 🆕 快速开始
├── ADE20K_TRAINING_GUIDE.md               # 🆕 训练指南
├── DATASET_COMPARISON.md                  # 🆕 数据集对比
└── ADE20K_SETUP_SUMMARY.md                # 🆕 本文档
```

## ⚙️ 关键配置参数

### 数据配置
```python
data_root = '/9950backfile/zhangyafei/ade/'
image_size = (1024, 1024)
batch_size = 2  # per GPU
num_workers = 2
```

### 训练配置
```python
max_epochs = 50
learning_rate = 2e-4
optimizer = 'AdamW'
weight_decay = 0.05
```

### 损失权重
```python
loss_cls_weight = 2.0
loss_mask_weight = 3.0
loss_dice_weight = 2.0
sr_loss_weight = 0.5   # 分辨率蒸馏
fa_loss_weight = 0.5   # 特征对齐
```

## 🔧 可选配置

### 1. 启用SAM蒸馏

编辑 `configs/reg_sam/regnet_ade20k.py`:
```python
use_sam_distill = True
```

### 2. 调整显存使用

**减小batch size**:
```bash
./train_ade20k.sh 8 --cfg-options train_dataloader.batch_size=1
```

**禁用分辨率蒸馏**:
```python
use_resolution_distill = False
```

### 3. 从COCO预训练模型开始

```bash
python tools/train.py configs/reg_sam/regnet_ade20k.py \
    --cfg-options load_from=work_dirs/regnet_coco/epoch_12.pth
```

## 📈 预期训练时间和性能

### 训练时间（基于A100 GPU）

| GPU配置 | Batch Size | 单Epoch | 50 Epochs |
|---------|-----------|---------|-----------|
| 1x GPU  | 2         | ~2小时   | ~4天      |
| 4x GPU  | 8         | ~35分钟  | ~1.2天    |
| 8x GPU  | 16        | ~20分钟  | ~17小时   |

### 预期性能（ADE20K验证集）

| 配置 | PQ | 说明 |
|------|----|----|
| Baseline | ~35 | 基础模型 |
| + 分辨率蒸馏 | ~37 | 当前配置 |
| + SAM蒸馏 | ~39 | 完整配置 |

*注：实际性能需要训练后确认*

## 📝 训练监控

### 输出目录
```
work_dirs/regnet_ade20k/
├── regnet_ade20k.py              # 配置备份
├── tf_logs/                      # TensorBoard日志
├── vis_data/                     # 可视化
├── epoch_*.pth                   # 每个epoch的checkpoint
├── best_ade_panoptic_PQ_*.pth    # 最佳模型
└── latest.pth                    # 最新checkpoint
```

### 查看日志
```bash
# 实时日志
tail -f work_dirs/regnet_ade20k/*/log.txt

# TensorBoard
tensorboard --logdir work_dirs/regnet_ade20k/
```

## ✅ 检查清单

开始训练前确认：

- [x] 数据集已验证 (`python3 tools/check_ade_files.py`)
- [x] 配置文件已创建 (`configs/reg_sam/regnet_ade20k.py`)
- [x] 训练脚本已准备 (`train_ade20k.sh`)
- [ ] 安装所有依赖 (`pip install -r requirements.txt`)
- [ ] GPU可用且显存充足 (>16GB per GPU)
- [ ] 磁盘空间充足 (>100GB)

## 🎓 推荐使用流程

### 初学者
1. 阅读 `ADE20K_QUICKSTART.md`
2. 运行 `python3 tools/check_ade_files.py`
3. 执行 `./train_ade20k.sh 1` (单GPU测试)
4. 查看日志和checkpoint

### 进阶用户
1. 阅读 `ADE20K_TRAINING_GUIDE.md`
2. 根据显存调整配置
3. 启用SAM蒸馏（可选）
4. 使用多GPU训练: `./train_ade20k.sh 8`
5. 调优超参数

### 研究者
1. 阅读所有文档
2. 研究 `DATASET_COMPARISON.md`
3. 实验不同的训练策略
4. 进行消融实验
5. 与COCO结果对比

## 🆘 获取帮助

- **快速问题**: 查看 `ADE20K_QUICKSTART.md`
- **详细指南**: 查看 `ADE20K_TRAINING_GUIDE.md`
- **显存问题**: 查看 `MEMORY_OPTIMIZATION.md`
- **SAM蒸馏**: 查看 `SAM_DISTILL_SETUP.md`
- **项目概览**: 查看 `README.md`

## 🎉 总结

✅ **ADE20K全景分割配置已完全就绪！**

所有必需的配置文件、脚本和文档都已准备完毕。您现在可以：

1. 直接开始训练
2. 根据需要调整配置
3. 监控训练过程
4. 评估和测试模型

祝训练顺利！🚀

---

**配置日期**: 2025-11-15  
**数据集**: ADE20K Panoptic Segmentation  
**项目**: RPS (Real-time Panoptic Segmentation)

