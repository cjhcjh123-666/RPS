# ADE20K分类器生成指南

## 问题描述

训练时遇到错误：
```
FileNotFoundError: [Errno 2] No such file or directory: 
'/home/zhangyafei/.cache/embd/convnext_large_d_320_ADE20KPanopticDataset.pth'
```

这是因为模型使用了开放词汇（Open Vocabulary）架构，需要预先生成文本分类器。

## 解决方案

### 快速方法（推荐）

```bash
cd /9950backfile/zhangyafei/RPS
./gen_ade20k_classifier.sh
```

### 手动方法

```bash
cd /9950backfile/zhangyafei/RPS
python tools/gen_cls.py configs/reg_sam/gen_ade20k_classifier.py
```

## 工作原理

1. **配置文件**: `configs/reg_sam/gen_ade20k_classifier.py`
   - 使用OpenCLIP的ConvNeXt-Large backbone
   - 加载LAION预训练权重
   - 使用ADE20K数据集的类别名称

2. **生成过程**:
   - 读取ADE20K的150个类别名称
   - 对每个类别使用多个文本模板（如"a photo of {}"）
   - 使用CLIP文本编码器提取特征向量
   - 平均所有模板的特征，得到每个类别的最终表示
   - 保存到 `~/.cache/embd/convnext_large_d_320_ADE20KPanopticDataset.pth`

3. **输出文件**:
   - 路径: `~/.cache/embd/convnext_large_d_320_ADE20KPanopticDataset.pth`
   - 内容: Tensor shape [150, max_candidates, 768]
   - 用途: 在推理时作为文本分类器权重

## 验证生成结果

```bash
# 检查文件是否存在
ls -lh ~/.cache/embd/convnext_large_d_320_ADE20KPanopticDataset.pth

# 查看文件信息
python -c "import torch; cls=torch.load('~/.cache/embd/convnext_large_d_320_ADE20KPanopticDataset.pth'); print(f'Shape: {cls.shape}')"
```

预期输出：
```
Shape: torch.Size([150, X, 768])
```
其中X是最大候选数（取决于类别名称的同义词数量）。

## 相关文件

### 新增文件
- `configs/reg_sam/gen_ade20k_classifier.py` - 生成分类器的配置
- `gen_ade20k_classifier.sh` - 便捷生成脚本

### 修改文件
- `seg/datasets/ade_ov.py` - 添加了 `dataset_name = 'ADE20KPanopticDataset'`

## 技术细节

### OpenCLIP Backbone配置

```python
model = dict(
    backbone=dict(
        type=OpenCLIPBackbone,
        model_name='convnext_large_d_320',  # ConvNeXt-Large with 320x320 input
        fix=True,  # 冻结参数
        init_cfg=dict(
            type='clip_pretrain',
            checkpoint='laion2b_s29b_b131k_ft_soup'  # LAION-2B预训练
        )
    )
)
```

### 文本模板（来自VILD）

生成使用多个模板来增强鲁棒性：
- "a photo of a {}"
- "a photo of the {}"
- "a photo of one {}"
- ... (更多模板在 `ext/templates.py` 中定义)

### 类别名称处理

ADE20K类别名称包含同义词，例如：
- `'bed,beds'` → 提取 ['bed', 'beds']
- `'person,child,girl,boy,woman,man,...'` → 提取多个同义词

每个同义词都会使用所有模板生成特征，然后平均。

## 多GPU加速（可选）

如果生成速度慢，可以使用多GPU加速：

```bash
# 4个GPU
bash tools/dist_train.sh configs/reg_sam/gen_ade20k_classifier.py 4 \
    --launcher pytorch

# 8个GPU  
bash tools/dist_train.sh configs/reg_sam/gen_ade20k_classifier.py 8 \
    --launcher pytorch
```

注意：这会在rank 0上保存结果。

## 常见问题

### Q1: 生成失败，提示找不到OpenCLIPBackbone

**原因**: 缺少依赖或代码未正确安装

**解决**:
```bash
cd /9950backfile/zhangyafei/RPS
pip install -e .
pip install open_clip_torch
```

### Q2: 下载预训练权重失败

**原因**: 网络问题或访问限制

**解决**:
1. 检查网络连接
2. 使用代理或镜像源
3. 手动下载权重并指定本地路径

### Q3: 显存不足

**原因**: ConvNeXt-Large模型较大

**解决**:
- 使用更小的模型（如 `convnext_base` 或 `vit_b_16`）
- 减小批处理大小（修改 `gen_cls.py` 中的 `NUM_BATCH`）

### Q4: 生成的文件名不匹配

**确认**:
- 数据集类的 `dataset_name` 属性
- 配置文件中的 `ov_classifier_name`
- 生成的文件名

这三者必须一致！

## 为COCO生成分类器（参考）

如果你也需要COCO的分类器：

```bash
python tools/gen_cls.py configs/reg_sam/regnet_coco.py
```

会生成：
```
~/.cache/embd/convnext_large_d_320_CocoPanopticOVDataset.pth
```

## 完整训练流程

```bash
# 1. 生成分类器
./gen_ade20k_classifier.sh

# 2. 验证数据集
python3 tools/check_ade_files.py

# 3. 开始训练
./train_ade20k.sh 8
```

## 性能影响

- **生成时间**: 约5-10分钟（单GPU）
- **文件大小**: 约10-50MB
- **推理影响**: 无（仅在加载时读取一次）
- **精度影响**: 使用高质量的CLIP文本编码器，对开放词汇任务很重要

## 参考

- CLIP论文: [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)
- OpenCLIP: [open_clip](https://github.com/mlfoundations/open_clip)
- VILD论文: [Open-vocabulary Object Detection via Vision and Language Knowledge Distillation](https://arxiv.org/abs/2104.13921)

## 总结

生成分类器是使用开放词汇模型的必要步骤。完成后，训练过程会自动加载这个文件，实现zero-shot或开放词汇的目标检测和分割能力。

