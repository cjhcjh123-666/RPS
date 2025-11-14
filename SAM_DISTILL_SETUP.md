# SAM完整模型蒸馏配置说明

## 概述

现在项目支持使用完整的SAM模型作为教师模型进行知识蒸馏。SAM模型来自 `segment-anything-main` 项目。

## 项目结构要求

确保项目结构如下：
```
RPS/
├── segment-anything-main/    # SAM原始项目
│   ├── segment_anything/
│   └── ...
├── checkpoint/
│   └── sam_vit_h_4b8939.pth  # SAM checkpoint
├── seg/
│   └── models/
│       ├── backbones/
│       │   └── sam_backbone.py  # SAM模型包装器
│       └── utils/
│           └── sam_distill.py   # SAM蒸馏模块
└── configs/
    └── reg_sam/
        └── regnet_distill.py    # 配置文件
```

## 配置说明

在 `configs/reg_sam/regnet_distill.py` 中，SAM蒸馏配置如下：

```python
sam_distill=dict(
    # 配置SAM教师模型（使用完整的SAM模型）
    teacher_model=dict(
        type='SAMTeacherModel',  # 使用SAM教师模型包装器
        model_type='vit_h',  # SAM模型类型：'vit_h', 'vit_l', 'vit_b'
        checkpoint='/data/chenjiahui/RPS/checkpoint/sam_vit_h_4b8939.pth',  # SAM checkpoint路径
        freeze=True  # 冻结教师模型参数
    ),
    teacher_checkpoint=None,  # 如果teacher_model中已配置checkpoint，这里可以为None
    feat_distill_weight=1.0,  # 特征蒸馏损失权重
    output_distill_weight=1.0,  # 输出蒸馏损失权重（如果SAM结构支持）
    temperature=4.0,  # 蒸馏温度
    distill_feat_layers=[0]  # SAM只输出单尺度特征，所以只蒸馏第0层
),
```

## SAM模型类型

支持三种SAM模型：
- `vit_h`: ViT-Huge (默认，最大模型)
- `vit_l`: ViT-Large
- `vit_b`: ViT-Base

根据你的checkpoint文件选择对应的类型：
- `sam_vit_h_4b8939.pth` → `model_type='vit_h'`
- `sam_vit_l_0b3195.pth` → `model_type='vit_l'`
- `sam_vit_b_01ec64.pth` → `model_type='vit_b'`

## 工作原理

1. **SAM模型加载**：
   - `SAMTeacherModel` 会自动从 `segment-anything-main` 导入SAM模型
   - 使用 `sam_model_registry` 构建对应类型的SAM模型
   - 加载checkpoint权重

2. **特征提取**：
   - SAM的 `image_encoder` (ViT backbone) 提取图像特征
   - 特征形状为 `(B, C, H, W)`，其中H=W=64（对于1024x1024输入）

3. **知识蒸馏**：
   - 使用SAM的特征对学生模型进行特征蒸馏
   - 通过MSE损失对齐SAM和学生的特征表示

## 注意事项

1. **输入预处理**：
   - SAM期望输入图像已经归一化（使用SAM的pixel_mean和pixel_std）
   - 当前实现假设输入已经预处理，如果需要，可以在data_preprocessor中添加SAM的预处理

2. **特征对齐**：
   - SAM输出单尺度特征（64x64 for 1024x1024输入）
   - 学生模型（RegNet+FPN）输出多尺度特征
   - 蒸馏时会将SAM特征与学生模型的第一个特征层对齐

3. **内存使用**：
   - SAM ViT-H模型较大，会增加显存使用
   - 如果显存不足，可以：
     - 使用较小的SAM模型（vit_l或vit_b）
     - 减小batch size
     - 只在部分epoch启用SAM蒸馏

4. **训练速度**：
   - SAM模型会增加训练时间（约20-30%）
   - 教师模型已冻结，不会更新梯度

## 验证配置

运行训练命令验证配置：

```bash
python tools/train.py configs/reg_sam/regnet_distill.py
```

如果配置正确，日志中会显示：
```
Using SAM teacher model: vit_h
Loaded SAM checkpoint from: /data/chenjiahui/RPS/checkpoint/sam_vit_h_4b8939.pth
```

如果出现导入错误，检查：
1. `segment-anything-main` 是否在项目根目录
2. SAM checkpoint路径是否正确
3. Python环境是否安装了必要的依赖

## 故障排除

### 问题1: ImportError: Cannot import segment_anything

**解决方案**：
- 确保 `segment-anything-main` 在项目根目录
- 检查 `segment-anything-main/segment_anything/__init__.py` 是否存在

### 问题2: 特征尺寸不匹配

**解决方案**：
- SAM输出特征尺寸固定（64x64 for 1024输入）
- 确保学生模型的输入尺寸与SAM兼容
- 可以在蒸馏时使用插值对齐特征尺寸

### 问题3: 显存不足

**解决方案**：
- 使用较小的SAM模型（vit_l或vit_b）
- 减小batch size
- 使用梯度累积

## 进阶配置

如果需要更精细的控制，可以修改 `seg/models/backbones/sam_backbone.py`：

1. **多尺度特征提取**：修改 `extract_feat` 方法返回多尺度特征
2. **特征后处理**：添加特征归一化或变换
3. **输入预处理**：集成SAM的预处理逻辑

## 参考

- SAM项目: https://github.com/facebookresearch/segment-anything
- SAM论文: https://arxiv.org/abs/2304.02643

