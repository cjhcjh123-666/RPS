# 数据集配置对比: COCO vs ADE20K

## 数据集统计

| 项目 | COCO Panoptic | ADE20K Panoptic |
|------|---------------|-----------------|
| **训练图像** | 118,287 | 20,210 |
| **验证图像** | 5,000 | 2,000 |
| **Things类别** | 80 | 100 |
| **Stuff类别** | 53 | 50 |
| **总类别数** | 133 | 150 |
| **数据路径** | `/data/coco/` | `/9950backfile/zhangyafei/ade/` |

## 配置文件对比

### COCO配置
- **文件**: `configs/reg_sam/regnet_coco.py`
- **数据配置**: `configs/_base_/datasets/coco_panoptic_video_lsj.py`
- **特点**: 
  - 大规模数据集
  - 适合预训练
  - 视频序列支持

### ADE20K配置
- **文件**: `configs/reg_sam/regnet_ade20k.py`
- **数据配置**: `configs/_base_/datasets/ade_panoptic_ov.py`
- **特点**:
  - 场景更丰富
  - 类别更多
  - 开放词汇支持

## 模型配置差异

### 类别数量
```python
# COCO
num_things_classes = 80
num_stuff_classes = 53
num_classes = 81  # things + 1

# ADE20K
num_things_classes = 100
num_stuff_classes = 50
num_classes = 101  # things + 1
```

### 类别权重
```python
# COCO
class_weights = [2.0]*80 + [1.0]*53 + [0.1]

# ADE20K
class_weights = [2.0]*100 + [1.0]*50 + [0.1]
```

### OV分类器
```python
# COCO
ov_classifier_name = 'convnext_large_d_320_CocoPanopticOVDataset'

# ADE20K
ov_classifier_name = 'convnext_large_d_320_ADE20KPanopticDataset'
```

## 训练配置建议

### COCO → ADE20K 迁移学习

如果已有COCO预训练模型，可以用于ADE20K微调：

```bash
python tools/train.py configs/reg_sam/regnet_ade20k.py \
    --cfg-options load_from=work_dirs/regnet_coco/epoch_12.pth
```

**注意事项**:
- Head需要重新初始化（类别数不同）
- Backbone和Neck可以直接迁移
- 建议使用较小的学习率

### 联合训练

可以同时在COCO和ADE20K上训练（参考`coco_panoptic_ade_city.py`）：

```python
train_dataloader = dict(
    dataset=dict(
        type=ConcatOVDataset,
        datasets=[
            coco_dataset,
            ade_dataset,
        ]
    )
)
```

## 推荐训练策略

### 策略1: ADE20K从头训练
```bash
# 适合只需要ADE20K性能的场景
./train_ade20k.sh 8
```

### 策略2: COCO预训练 + ADE20K微调
```bash
# 第一阶段: COCO预训练
python tools/train.py configs/reg_sam/regnet_coco.py

# 第二阶段: ADE20K微调
python tools/train.py configs/reg_sam/regnet_ade20k.py \
    --cfg-options load_from=work_dirs/regnet_coco/epoch_12.pth \
    --cfg-options optim_wrapper.optimizer.lr=1e-4
```

### 策略3: 联合训练
```bash
# 同时在两个数据集上训练
python tools/train.py configs/reg_sam/coco_ade_joint.py
```

## 性能对比（预期）

| 模型 | COCO PQ | ADE20K PQ | 训练时间 |
|------|---------|-----------|----------|
| COCO only | ~45 | - | ~2天(8 GPU) |
| ADE20K only | - | ~37 | ~1天(8 GPU) |
| COCO→ADE20K | ~45 | ~39 | ~3天(8 GPU) |
| Joint | ~43 | ~38 | ~3天(8 GPU) |

*注：数值为估算，需实际训练验证*

## 评估指标对比

### COCO评估
```bash
python tools/test.py \
    configs/reg_sam/regnet_coco.py \
    work_dirs/regnet_coco/latest.pth
```

输出指标:
- PQ, SQ, RQ (全景指标)
- PQ_th, PQ_st (things/stuff分开)
- 每个类别的详细指标

### ADE20K评估
```bash
python tools/test.py \
    configs/reg_sam/regnet_ade20k.py \
    work_dirs/regnet_ade20k/latest.pth
```

输出指标:
- PQ, SQ, RQ (全景指标)
- mIoU (语义分割指标)
- 150个类别的详细指标

## 数据集文件结构

### COCO
```
data/coco/
├── annotations/
│   ├── panoptic_train2017.json
│   ├── panoptic_val2017.json
│   └── panoptic_{train,val}2017/
├── train2017/
└── val2017/
```

### ADE20K
```
/9950backfile/zhangyafei/ade/
└── ADEChallengeData2016/
    ├── ade20k_panoptic_train.json
    ├── ade20k_panoptic_val.json
    ├── ade20k_panoptic_train/
    ├── ade20k_panoptic_val/
    └── images/
        ├── training/
        └── validation/
```

## 总结

- **COCO**: 更大、更标准、适合预训练
- **ADE20K**: 更多类别、更丰富场景、适合室内/室外综合场景

根据实际需求选择合适的训练策略！

