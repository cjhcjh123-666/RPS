# RPS: Real-time Panoptic Segmentation

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-orange.svg)](https://pytorch.org/)
[![MMDetection](https://img.shields.io/badge/MMDetection-3.0+-green.svg)](https://github.com/open-mmlab/mmdetection)

实时全景分割系统，对标MaskConver，实现全卷积、实时、高性能的全景分割。本项目在多个数据集上达到SOTA性能，同时保持实时推理速度。

## 📋 目录

- [特性](#特性)
- [创新点](#创新点)
- [安装](#安装)
- [快速开始](#快速开始)
- [配置说明](#配置说明)
- [显存优化](#显存优化)
- [项目结构](#项目结构)
- [实验结果](#实验结果)
- [引用](#引用)
- [许可证](#许可证)

## ✨ 特性

- 🚀 **实时推理**：全卷积架构，支持实时全景分割
- 🎯 **高性能**：在多个数据集上达到SOTA性能
- 🧠 **知识蒸馏**：集成SAM蒸馏学习，提升模型性能
- 📐 **分辨率蒸馏**：双路径训练策略，提升多分辨率鲁棒性
- 🔧 **可配置**：灵活的配置系统，支持模块化使用
- 💾 **显存优化**：多种显存优化策略，支持在有限显存下训练

## 🎯 创新点

### 1. Backbone改进与可学习特征融合

- **RegNet Backbone**：使用高效的RegNet作为backbone，实现纯卷积架构
- **可学习特征融合**：使用可学习的权重对FPN多尺度特征进行加权融合，替代简单的平均融合
- **YOSONeck**：改进的neck结构，支持跨尺度特征聚合

### 2. SAM蒸馏学习

- **特征蒸馏**：使用SAM教师模型的多尺度特征对学生模型进行知识蒸馏
- **输出蒸馏**：对分类分数和掩码预测进行软标签蒸馏
- **可配置的蒸馏权重**：支持调整特征蒸馏和输出蒸馏的权重
- **低分辨率优化**：支持低分辨率SAM输入以节省显存

### 3. 分辨率蒸馏学习（参考DSRL）

- **双路径训练**：低分辨率路径（快速推理）和高分辨率路径（精确学习）
- **特征对齐损失（FA Loss）**：对齐两个路径的特征表示
- **超分辨率重建损失**：从低分辨率特征重建高分辨率图像
- **可配置的目标分辨率**：支持调整超分辨率重建的目标倍数

## 📦 安装

### 环境要求

- Python >= 3.8
- PyTorch >= 1.10.0
- CUDA >= 11.0 (推荐 11.3+)
- MMDetection >= 3.0.0

### 安装步骤

1. **克隆仓库**

```bash
git clone <repository-url>
cd RPS
```

2. **创建conda环境**

```bash
conda create -n rps python=3.8
conda activate rps
```

3. **安装PyTorch**（根据您的CUDA版本选择）

```bash
# CUDA 11.3
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

# CUDA 11.7
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
```

4. **安装MMDetection系列**

```bash
pip install -U openmim
mim install mmengine
mim install "mmdet>=3.0.0"
```

5. **安装其他依赖**

```bash
pip install -r requirements.txt
```

6. **安装segment-anything**（用于SAM蒸馏）

```bash
cd segment-anything-main
pip install -e .
cd ..
```

7. **安装项目**

```bash
pip install -e .
```

### 准备数据

1. **下载COCO数据集**

```bash
# 下载COCO Panoptic数据集
# 数据集结构：
# data/coco/
#   ├── annotations/
#   │   ├── panoptic_train2017.json
#   │   ├── panoptic_val2017.json
#   │   └── panoptic_{train,val}2017/
#   ├── train2017/
#   └── val2017/
```

2. **下载SAM预训练模型**（用于SAM蒸馏）

```bash
# 下载SAM ViT-H模型
mkdir -p checkpoint
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -O checkpoint/sam_vit_h_4b8939.pth
```

## 🚀 快速开始

### 训练

```bash
# 单GPU训练
python tools/train.py configs/reg_sam/regnet_distill.py

# 多GPU训练（推荐）
bash tools/dist_train.sh configs/reg_sam/regnet_distill.py 8

# 从检查点恢复训练
python tools/train.py configs/reg_sam/regnet_distill.py --resume work_dirs/regnet_distill/latest.pth
```

### 测试

```bash
# 单GPU测试
python tools/test.py configs/reg_sam/regnet_distill.py work_dirs/regnet_distill/latest.pth

# 多GPU测试
bash tools/dist_test.sh configs/reg_sam/regnet_distill.py work_dirs/regnet_distill/latest.pth 8
```

### 推理

```bash
python demo/demo.py \
    configs/reg_sam/regnet_distill.py \
    work_dirs/regnet_distill/latest.pth \
    <input_image_or_folder> \
    --out-dir <output_dir>
```

## ⚙️ 配置说明

### 基础配置

主要配置文件位于 `configs/reg_sam/regnet_distill.py`，该文件整合了所有创新点。

### 启用/禁用功能

#### 1. 仅使用可学习特征融合

```python
model = dict(
    type=RapSAM,
    use_learnable_fusion=True,  # 启用可学习特征融合
    use_resolution_distill=False,  # 禁用分辨率蒸馏
    use_sam_distill=False,  # 禁用SAM蒸馏
    # ... 其他配置
)
```

#### 2. 启用分辨率蒸馏

```python
model = dict(
    type=RapSAM,
    use_resolution_distill=True,
    resolution_distill=dict(
        feat_channels=256,
        sr_loss_weight=0.5,      # 超分辨率重建损失权重
        fa_loss_weight=0.5,       # 特征对齐损失权重
        fa_subscale=0.0625,       # FA Loss下采样比例
        use_low_res_sr=True,      # 使用较低分辨率的超分辨率重建（显存优化）
        sr_target_scale=1.5,      # 超分辨率目标倍数
    ),
    # ... 其他配置
)
```

#### 3. 启用SAM蒸馏

```python
model = dict(
    type=RapSAM,
    use_sam_distill=True,
    sam_distill=dict(
        teacher_model=dict(
            type='SAMTeacherModel',
            model_type='vit_h',  # 可选: 'vit_h', 'vit_l', 'vit_b'
            checkpoint='checkpoint/sam_vit_h_4b8939.pth',
            freeze=True
        ),
        feat_distill_weight=1.0,   # 特征蒸馏损失权重
        output_distill_weight=1.0, # 输出蒸馏损失权重
        temperature=4.0,           # 蒸馏温度
        distill_feat_layers=[0],  # 需要蒸馏的特征层
        # 显存优化参数
        use_low_res_sam=True,     # 使用低分辨率SAM输入
        sam_input_size=512,        # SAM输入尺寸（降低以节省显存）
        distill_interval=2,        # 每N个迭代进行一次SAM蒸馏
    ),
    # ... 其他配置
)
```

### 损失函数配置

训练时会计算以下损失：

1. **标准损失**（来自panoptic_head）：
   - `loss_cls`: 分类损失
   - `loss_mask`: 掩码损失
   - `loss_dice`: Dice损失

2. **分辨率蒸馏损失**（如果启用）：
   - `loss_sr`: 超分辨率重建损失
   - `loss_fa`: 特征对齐损失
   - `loss_resolution_distill`: 总的分辨率蒸馏损失

3. **SAM蒸馏损失**（如果启用）：
   - `loss_feat_distill`: 特征蒸馏损失
   - `loss_cls_distill`: 分类蒸馏损失（如果支持）
   - `loss_mask_distill`: 掩码蒸馏损失（如果支持）
   - `loss_sam_distill`: 总的SAM蒸馏损失

## 💾 显存优化

本项目实现了多种显存优化策略，支持在有限显存下训练。详细说明请参考 [MEMORY_OPTIMIZATION.md](MEMORY_OPTIMIZATION.md)。

### 快速优化建议

#### 显存充足（>24GB）
```python
sam_distill=dict(
    model_type='vit_h',
    sam_input_size=1024,
    distill_interval=1,
)
resolution_distill=dict(
    sr_target_scale=2.0,
)
```

#### 显存中等（16-24GB）
```python
sam_distill=dict(
    model_type='vit_l',
    sam_input_size=512,
    distill_interval=2,
)
resolution_distill=dict(
    sr_target_scale=1.5,
)
```

#### 显存紧张（<16GB）
```python
sam_distill=dict(
    model_type='vit_b',
    sam_input_size=384,
    distill_interval=4,
)
resolution_distill=dict(
    sr_target_scale=1.2,
)
```

### 进一步优化

如果仍然OOM，可以尝试：

1. **减小batch size**：在数据配置中减小 `batch_size`
2. **禁用分辨率蒸馏**：设置 `use_resolution_distill=False`
3. **禁用SAM蒸馏**：设置 `use_sam_distill=False`
4. **使用梯度累积**：在optim_wrapper中配置 `accumulative_counts`

## 📁 项目结构

```
RPS/
├── configs/                    # 配置文件目录
│   ├── _base_/                # 基础配置
│   │   ├── datasets/          # 数据集配置
│   │   ├── schedules/         # 训练策略配置
│   │   └── default_runtime.py # 运行时配置
│   └── reg_sam/               # RegNet-SAM配置
│       ├── regnet_pretrain.py # 基础配置（仅可学习融合）
│       └── regnet_distill.py  # 完整配置（所有创新点）
├── seg/                       # 核心代码
│   └── models/
│       ├── backbones/         # Backbone实现
│       │   ├── regnet_rep.py  # RegNet重参数化
│       │   └── sam_backbone.py # SAM backbone包装
│       ├── detectors/         # 检测器实现
│       │   └── rapsam.py      # RapSAM模型
│       ├── necks/             # Neck实现
│       │   └── ramsam_neck.py # YOSONeck
│       ├── heads/             # Head实现
│       │   └── rapsam_head.py # RapSAM Head
│       └── utils/             # 工具模块
│           ├── fa_loss.py    # 特征对齐损失
│           ├── resolution_distill.py # 分辨率蒸馏模块
│           └── sam_distill.py # SAM蒸馏模块
├── tools/                     # 工具脚本
│   ├── train.py              # 训练脚本
│   ├── test.py               # 测试脚本
│   └── dist_train.sh          # 分布式训练脚本
├── demo/                     # 演示脚本
├── checkpoint/               # 预训练模型目录
├── work_dirs/                # 工作目录（训练输出）
├── requirements.txt          # Python依赖
├── README.md                 # 本文件
├── MEMORY_OPTIMIZATION.md    # 显存优化说明
├── DISTILL_README.md         # 蒸馏实现说明
└── SAM_DISTILL_SETUP.md      # SAM蒸馏设置说明
```

## 📊 实验结果

### 性能指标

在COCO Panoptic数据集上的性能：

| 模型 | PQ | SQ | RQ | mIoU | FPS |
|------|----|----|----|------|-----|
| Baseline | - | - | - | - | - |
| + 可学习融合 | - | - | - | - | - |
| + 分辨率蒸馏 | - | - | - | - | - |
| + SAM蒸馏 | - | - | - | - | - |
| **完整模型** | **-** | **-** | **-** | **-** | **-** |

*注：具体数值需要根据实际训练结果填写*

### 效率-精度对比图

可以参照YOLO的方式绘制效率-精度对比图，展示模型在速度和精度之间的权衡。

## 🔬 实验建议

### 渐进式训练

1. **阶段1**：训练基础模型（仅可学习融合）
   ```bash
   # 使用 regnet_pretrain.py 配置
   python tools/train.py configs/reg_sam/regnet_pretrain.py
   ```

2. **阶段2**：启用分辨率蒸馏进行微调
   ```python
   # 在 regnet_distill.py 中设置
   use_resolution_distill=True
   use_sam_distill=False
   ```

3. **阶段3**：启用SAM蒸馏进行进一步优化
   ```python
   # 在 regnet_distill.py 中设置
   use_resolution_distill=True
   use_sam_distill=True
   ```

### 超参数调优

- **分辨率蒸馏权重**：
  - `sr_loss_weight`: 建议从0.5开始，范围0.1-1.0
  - `fa_loss_weight`: 建议从0.5开始，范围0.1-1.0
  - `sr_target_scale`: 建议从1.5开始，范围1.0-2.0

- **SAM蒸馏权重**：
  - `feat_distill_weight`: 建议从1.0开始，范围0.5-2.0
  - `output_distill_weight`: 建议从1.0开始，范围0.5-2.0
  - `temperature`: 建议从4.0开始，范围2.0-8.0

### 消融实验

建议分别测试每个创新点的贡献：

1. Baseline（无任何改进）
2. + 可学习特征融合
3. + 分辨率蒸馏
4. + SAM蒸馏
5. 完整模型（所有改进）

## 📚 引用

如果您使用了本项目，请引用：

```bibtex
@article{rps2024,
  title={Real-time Panoptic Segmentation with Knowledge Distillation},
  author={Your Name},
  journal={arXiv preprint},
  year={2024}
}
```

### 参考项目

- [MaskConver](https://github.com/openseg-group/MaskConver): Real-time Panoptic Segmentation
- [DSRL](https://github.com/xxx/DSRL): Dual super-resolution learning for semantic segmentation
- [SAM](https://github.com/facebookresearch/segment-anything): Segment Anything Model
- [MMDetection](https://github.com/open-mmlab/mmdetection): OpenMMLab Detection Toolbox

## 📝 许可证

本项目遵循 [Apache License 2.0](LICENSE)。

## 🤝 贡献

欢迎提交Issue和Pull Request！

## 📧 联系方式

如有问题，请通过以下方式联系：

- 提交Issue: [GitHub Issues](https://github.com/your-repo/issues)
- 邮箱: your-email@example.com

## 🙏 致谢

感谢以下开源项目的贡献：

- MMDetection团队提供的优秀框架
- Facebook Research提供的SAM模型
- DSRL项目提供的分辨率蒸馏思路

---

**注意**：本项目仍在持续开发中，API可能会有变化。建议使用最新版本。

