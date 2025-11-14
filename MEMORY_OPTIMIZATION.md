# 显存优化说明

## 已实现的显存优化策略

### 1. SAM蒸馏优化

#### 低分辨率SAM输入
- **参数**: `use_low_res_sam=True`, `sam_input_size=512`
- **效果**: 将SAM输入从1024x1024降低到512x512，显存减少约75%
- **配置位置**: `configs/reg_sam/regnet_distill.py` 中的 `sam_distill` 配置

#### 间隔蒸馏
- **参数**: `distill_interval=2`
- **效果**: 每2个迭代进行一次SAM蒸馏，显存使用减少约50%
- **说明**: 设置为1表示每次都蒸馏，增大可进一步节省显存

#### 使用更小的SAM模型
- **参数**: `model_type='vit_l'` 或 `'vit_b'`
- **效果**: 
  - `vit_h`: 最大模型，显存占用最高
  - `vit_l`: 中等模型，显存占用约为vit_h的60%
  - `vit_b`: 最小模型，显存占用约为vit_h的30%
- **配置**: 修改 `sam_distill.teacher_model.model_type`

### 2. 分辨率蒸馏优化

#### 降低超分辨率目标分辨率
- **参数**: `use_low_res_sr=True`, `sr_target_scale=1.5`
- **效果**: 将超分辨率重建的目标分辨率从2倍降低到1.5倍，显存减少约44%
- **配置位置**: `resolution_distill` 配置

#### 低分辨率特征提取优化
- **实现**: 在 `torch.no_grad()` 中提取低分辨率特征
- **效果**: 避免保存低分辨率路径的梯度，节省显存

### 3. 计算图优化

#### 及时释放中间变量
- **实现**: 在关键位置使用 `del` 释放不需要的变量
- **位置**: 
  - 分辨率蒸馏后释放 `x_low`, `feats_low`
  - SAM蒸馏后释放 `student_feats`, `backbone_feats`

#### 使用detach分离计算图
- **实现**: 低分辨率特征使用 `.detach()` 分离计算图
- **效果**: 避免保存不必要的梯度信息

### 4. SAM模型预处理优化

#### 智能resize
- **实现**: 如果输入大于512，先resize到512再padding到1024
- **效果**: 减少SAM内部中间特征图的大小

## 配置建议

### 显存充足（>24GB）
```python
sam_distill=dict(
    model_type='vit_h',
    sam_input_size=1024,
    distill_interval=1,  # 每次都蒸馏
)
resolution_distill=dict(
    sr_target_scale=2.0,  # 完整超分辨率
)
```

### 显存中等（16-24GB）
```python
sam_distill=dict(
    model_type='vit_l',  # 使用中等模型
    sam_input_size=512,
    distill_interval=2,  # 每2次蒸馏一次
)
resolution_distill=dict(
    sr_target_scale=1.5,
)
```

### 显存紧张（<16GB）
```python
sam_distill=dict(
    model_type='vit_b',  # 使用小模型
    sam_input_size=384,  # 进一步降低
    distill_interval=4,  # 每4次蒸馏一次
)
resolution_distill=dict(
    sr_target_scale=1.2,  # 最小超分辨率
)
```

## 进一步优化建议

如果仍然OOM，可以尝试：

1. **减小batch size**: 在数据配置中减小 `batch_size`
2. **禁用分辨率蒸馏**: 设置 `use_resolution_distill=False`
3. **禁用SAM蒸馏**: 设置 `use_sam_distill=False`
4. **使用梯度累积**: 在optim_wrapper中配置 `accumulative_counts`
5. **使用CPU卸载**: 将SAM模型移到CPU（会降低速度）

## 性能权衡

- **低分辨率SAM输入**: 可能略微降低蒸馏效果，但显存节省显著
- **间隔蒸馏**: 训练可能略微变慢，但显存节省明显
- **降低超分辨率目标**: 可能略微影响分辨率蒸馏效果

建议根据显存情况逐步调整参数，找到性能和显存的平衡点。

