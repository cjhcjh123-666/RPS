# ADE20K全景分割训练指南

## 数据集信息

- **数据集**: ADE20K Panoptic Segmentation
- **数据路径**: `/9950backfile/zhangyafei/ade/`
- **训练图像**: 20,210张
- **验证图像**: 2,000张
- **类别数**: 150 (100 things + 50 stuff)

## 配置文件

主配置文件: `configs/reg_sam/regnet_ade20k.py`

### 关键配置

```python
# 数据集
data_root = '/9950backfile/zhangyafei/ade/'
num_things_classes = 100
num_stuff_classes = 50

# 模型创新点
use_learnable_fusion = True      # 可学习特征融合
use_resolution_distill = True    # 分辨率蒸馏（已启用）
use_sam_distill = False          # SAM蒸馏（默认关闭）

# 训练参数
batch_size = 2
learning_rate = 2e-4
max_epochs = 50
```

## 快速开始

### 1. 验证数据集

```bash
cd /9950backfile/zhangyafei/RPS
python3 tools/check_ade_files.py
```

### 2. 单GPU训练

```bash
cd /9950backfile/zhangyafei/RPS
python tools/train.py configs/reg_sam/regnet_ade20k.py
```

### 3. 多GPU训练（推荐）

```bash
cd /9950backfile/zhangyafei/RPS

# 4 GPUs
bash tools/dist_train.sh configs/reg_sam/regnet_ade20k.py 4

# 8 GPUs
bash tools/dist_train.sh configs/reg_sam/regnet_ade20k.py 8
```

### 4. 从检查点恢复训练

```bash
python tools/train.py configs/reg_sam/regnet_ade20k.py \
    --resume work_dirs/regnet_ade20k/latest.pth
```

### 5. 使用预训练模型

如果有COCO预训练的模型，可以加载来加速训练：

```bash
python tools/train.py configs/reg_sam/regnet_ade20k.py \
    --cfg-options load_from=work_dirs/regnet_coco/latest.pth
```

或者在配置文件中添加：
```python
load_from = 'work_dirs/regnet_coco/latest.pth'
```

## 测试/评估

### 单GPU测试

```bash
python tools/test.py \
    configs/reg_sam/regnet_ade20k.py \
    work_dirs/regnet_ade20k/epoch_50.pth
```

### 多GPU测试

```bash
bash tools/dist_test.sh \
    configs/reg_sam/regnet_ade20k.py \
    work_dirs/regnet_ade20k/epoch_50.pth \
    8
```

## 可选配置

### 启用SAM蒸馏

如果想使用SAM蒸馏来提升性能（需要更多显存）：

1. 准备SAM checkpoint：
```bash
mkdir -p /9950backfile/zhangyafei/checkpoint
# 下载或复制 sam_vit_h_4b8939.pth 到该目录
```

2. 修改配置文件 `configs/reg_sam/regnet_ade20k.py`：
```python
use_sam_distill = True
```

3. 根据显存调整SAM配置：

**显存充足（>24GB）**：
```python
sam_distill=dict(
    teacher_model=dict(model_type='vit_h'),
    sam_input_size=1024,
    distill_interval=1,
)
```

**显存中等（16-24GB）**：
```python
sam_distill=dict(
    teacher_model=dict(model_type='vit_l'),
    sam_input_size=512,
    distill_interval=2,
)
```

**显存紧张（<16GB）**：
```python
sam_distill=dict(
    teacher_model=dict(model_type='vit_b'),
    sam_input_size=384,
    distill_interval=4,
)
```

### 调整Batch Size

如果遇到OOM（内存不足），可以减小batch size：

```bash
python tools/train.py configs/reg_sam/regnet_ade20k.py \
    --cfg-options train_dataloader.batch_size=1
```

### 禁用分辨率蒸馏

如果显存紧张，可以关闭分辨率蒸馏：

```python
use_resolution_distill = False
```

## 训练监控

### TensorBoard

训练日志会保存在 `work_dirs/regnet_ade20k/` 目录下，可以使用TensorBoard查看：

```bash
tensorboard --logdir work_dirs/regnet_ade20k/
```

### 关键指标

- **PQ (Panoptic Quality)**: 全景分割质量，越高越好
- **SQ (Segmentation Quality)**: 分割质量
- **RQ (Recognition Quality)**: 识别质量
- **mIoU**: 平均IoU

## 输出目录结构

```
work_dirs/regnet_ade20k/
├── regnet_ade20k.py          # 配置文件备份
├── tf_logs/                  # TensorBoard日志
├── vis_data/                 # 可视化结果
├── epoch_1.pth              # 每个epoch的checkpoint
├── epoch_2.pth
├── ...
├── best_ade_panoptic_PQ_epoch_XX.pth  # 最佳模型
└── latest.pth               # 最新checkpoint（用于恢复训练）
```

## 常见问题

### 1. CUDA Out of Memory

**解决方案**：
- 减小batch size
- 禁用SAM蒸馏或使用更小的SAM模型
- 禁用分辨率蒸馏
- 使用梯度累积

### 2. 找不到模块

**解决方案**：
```bash
cd /9950backfile/zhangyafei/RPS
pip install -e .
```

### 3. 数据加载错误

**解决方案**：
- 运行 `python3 tools/check_ade_files.py` 检查数据集
- 确认数据路径正确
- 检查annotation文件格式

### 4. 训练速度慢

**解决方案**：
- 增加 `num_workers`
- 使用更多GPU进行分布式训练
- 禁用SAM蒸馏或增加 `distill_interval`

## 性能优化建议

### 渐进式训练策略

1. **第一阶段（快速验证）**：
   - 使用较少epoch（10-20）
   - 禁用SAM蒸馏
   - 验证pipeline是否正常工作

2. **第二阶段（完整训练）**：
   - 使用完整epoch（50）
   - 启用分辨率蒸馏
   - 调整超参数

3. **第三阶段（精调）**：
   - 根据需要启用SAM蒸馏
   - 使用更小的学习率微调
   - 使用最佳checkpoint继续训练

### 超参数调优

- **学习率**: 2e-4 (默认), 可尝试 1e-4 到 5e-4
- **Batch size**: 2 (默认), 根据GPU显存调整
- **分辨率蒸馏权重**: sr_loss_weight=0.5, fa_loss_weight=0.5
- **训练epoch**: 50 (默认), 可根据验证集性能调整

## 预期结果

在ADE20K验证集上，预期性能（参考值）：

| 配置 | PQ | SQ | RQ | mIoU |
|------|----|----|----|----|
| Baseline | ~35 | ~75 | ~45 | ~40 |
| + 分辨率蒸馏 | ~37 | ~76 | ~47 | ~42 |
| + SAM蒸馏 | ~39 | ~77 | ~49 | ~44 |

*注：具体数值需要根据实际训练结果更新*

## 下一步

训练完成后，可以：

1. **可视化结果**：
```bash
python demo/demo.py \
    configs/reg_sam/regnet_ade20k.py \
    work_dirs/regnet_ade20k/best_ade_panoptic_PQ_epoch_XX.pth \
    /path/to/test/images \
    --out-dir results/
```

2. **评估测试集**：
```bash
python tools/test.py \
    configs/reg_sam/regnet_ade20k.py \
    work_dirs/regnet_ade20k/best_ade_panoptic_PQ_epoch_XX.pth \
    --show-dir results/
```

3. **模型导出**：
   - 导出为ONNX格式用于部署
   - 量化模型以提升推理速度

## 联系方式

如有问题，请查看：
- 主README: `README.md`
- 显存优化指南: `MEMORY_OPTIMIZATION.md`
- SAM蒸馏设置: `SAM_DISTILL_SETUP.md`

