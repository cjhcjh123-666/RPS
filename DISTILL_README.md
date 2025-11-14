# 实时全景分割 - 分辨率蒸馏与SAM蒸馏实现

本实现对标MaskConver，实现了全卷积、实时、高性能的全景分割系统，包含以下创新点：

## 创新点

### 1. Backbone改进
- **可学习特征融合**：使用可学习的权重对FPN多尺度特征进行加权融合，替代简单的平均融合
- **RegNet Backbone**：使用高效的RegNet作为backbone，实现纯卷积架构

### 2. SAM蒸馏学习
- **特征蒸馏**：使用SAM教师模型的多尺度特征对学生模型进行知识蒸馏
- **输出蒸馏**：对分类分数和掩码预测进行软标签蒸馏
- **可配置的蒸馏权重**：支持调整特征蒸馏和输出蒸馏的权重

### 3. 分辨率蒸馏学习（参考DSRL）
- **双路径训练**：低分辨率路径（快速推理）和高分辨率路径（精确学习）
- **特征对齐损失（FA Loss）**：对齐两个路径的特征表示
- **超分辨率重建损失**：从低分辨率特征重建高分辨率图像

## 文件结构

```
seg/models/
├── detectors/
│   └── rapsam.py              # 改进的RapSAM模型，支持蒸馏
├── utils/
│   ├── fa_loss.py              # 特征对齐损失
│   ├── resolution_distill.py   # 分辨率蒸馏模块
│   └── sam_distill.py          # SAM蒸馏模块

configs/reg_sam/
└── regnet_distill.py           # 整合所有改进的配置文件
```

## 使用方法

### 1. 基础配置（仅使用可学习特征融合）

```python
model = dict(
    type=RapSAM,
    use_learnable_fusion=True,  # 启用可学习特征融合
    # ... 其他配置
)
```

### 2. 启用分辨率蒸馏

```python
model = dict(
    type=RapSAM,
    use_resolution_distill=True,
    resolution_distill=dict(
        feat_channels=128,
        sr_loss_weight=0.5,      # 超分辨率重建损失权重
        fa_loss_weight=0.5,      # 特征对齐损失权重
        fa_subscale=0.0625       # FA Loss下采样比例
    ),
    # ... 其他配置
)
```

### 3. 启用SAM蒸馏

```python
model = dict(
    type=RapSAM,
    use_sam_distill=True,
    sam_distill=dict(
        teacher_model=None,      # SAM教师模型配置（可选）
        teacher_checkpoint='path/to/sam/checkpoint.pth',  # SAM检查点路径
        feat_distill_weight=1.0,  # 特征蒸馏损失权重
        output_distill_weight=1.0, # 输出蒸馏损失权重
        temperature=4.0,          # 蒸馏温度
        distill_feat_layers=[0, 1, 2, 3]  # 需要蒸馏的特征层
    ),
    # ... 其他配置
)
```

### 4. 完整配置（所有创新点）

使用 `configs/reg_sam/regnet_distill.py` 配置文件，该文件整合了所有改进：

```bash
# 训练
python tools/train.py configs/reg_sam/regnet_distill.py

# 测试
python tools/test.py configs/reg_sam/regnet_distill.py work_dirs/regnet_distill/latest.pth
```

## 损失函数

训练时会计算以下损失：

1. **标准损失**（来自panoptic_head）：
   - `loss_cls`: 分类损失
   - `loss_mask`: 掩码损失
   - `loss_dice`: Dice损失
   - `loss_iou`: IoU损失

2. **分辨率蒸馏损失**（如果启用）：
   - `loss_sr`: 超分辨率重建损失
   - `loss_fa`: 特征对齐损失
   - `loss_resolution_distill`: 总的分辨率蒸馏损失

3. **SAM蒸馏损失**（如果启用）：
   - `loss_feat_distill`: 特征蒸馏损失
   - `loss_cls_distill`: 分类蒸馏损失
   - `loss_mask_distill`: 掩码蒸馏损失
   - `loss_sam_distill`: 总的SAM蒸馏损失

## 性能优化建议

1. **分辨率蒸馏**：
   - 在训练时启用，可以提升模型对多分辨率的鲁棒性
   - 推理时自动使用高分辨率路径，无需额外配置

2. **SAM蒸馏**：
   - 需要预训练的SAM模型作为教师
   - 如果SAM模型较大，建议在推理时禁用（仅用于训练）

3. **可学习特征融合**：
   - 对性能影响很小，建议始终启用
   - 可以自动学习最优的多尺度特征融合权重

## 注意事项

1. **SAM教师模型**：
   - 如果`teacher_model`为None，需要在训练前手动加载SAM模型
   - 确保SAM模型的输出格式与当前模型兼容

2. **内存使用**：
   - 分辨率蒸馏会增加约30%的内存使用（需要同时处理低分辨率和高分辨率输入）
   - SAM蒸馏会增加内存使用（取决于教师模型大小）

3. **训练速度**：
   - 分辨率蒸馏会略微降低训练速度（约10-15%）
   - SAM蒸馏会显著降低训练速度（取决于教师模型）

## 实验建议

1. **渐进式训练**：
   - 先训练基础模型（不使用蒸馏）
   - 然后启用分辨率蒸馏进行微调
   - 最后启用SAM蒸馏进行进一步优化

2. **超参数调优**：
   - `sr_loss_weight`和`fa_loss_weight`：建议从0.5开始，根据验证集性能调整
   - `feat_distill_weight`和`output_distill_weight`：建议从1.0开始，根据验证集性能调整
   - `temperature`：SAM蒸馏温度，建议在2.0-8.0之间

3. **消融实验**：
   - 分别测试每个创新点的贡献
   - 对比不同权重配置的效果

## 参考

- DSRL: Dual super-resolution learning for semantic segmentation
- SAM: Segment Anything Model
- MaskConver: Real-time Panoptic Segmentation

