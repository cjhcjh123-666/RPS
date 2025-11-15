# 🚀 ADE20K训练快速参考

## 一键命令

```bash
cd /9950backfile/zhangyafei/RPS

# 1️⃣ 验证数据集
python3 tools/check_ade_files.py

# 2️⃣ 生成分类器（首次必须！）
./gen_ade20k_classifier.sh

# 3️⃣ 开始训练
./train_ade20k.sh 8

# 4️⃣ 测试模型
./test_ade20k.sh work_dirs/regnet_ade20k/best_*.pth 8
```

## 常见问题速查

| 问题 | 解决方案 | 文档 |
|------|---------|------|
| 找不到分类器文件 | `./gen_ade20k_classifier.sh` | `CLASSIFIER_SOLUTION.md` |
| 显存不足(OOM) | 减小batch_size或禁用蒸馏 | `MEMORY_OPTIMIZATION.md` |
| 数据集加载错误 | `python3 tools/check_ade_files.py` | `ADE20K_QUICKSTART.md` |
| 如何从COCO预训练 | 使用`--cfg-options load_from=...` | `ADE20K_TRAINING_GUIDE.md` |
| 如何启用SAM蒸馏 | 修改配置`use_sam_distill=True` | `SAM_DISTILL_SETUP.md` |

## 关键文件位置

```
配置: configs/reg_sam/regnet_ade20k.py
数据: /9950backfile/zhangyafei/ade/
分类器: ~/.cache/embd/convnext_large_d_320_ADE20KPanopticDataset.pth
输出: work_dirs/regnet_ade20k/
```

## 监控训练

```bash
# 查看日志
tail -f work_dirs/regnet_ade20k/*/log.txt

# TensorBoard
tensorboard --logdir work_dirs/regnet_ade20k/

# 检查checkpoint
ls -lht work_dirs/regnet_ade20k/*.pth
```

## 配置调整

```bash
# 修改batch size
./train_ade20k.sh 8 --cfg-options train_dataloader.batch_size=1

# 从检查点恢复
python tools/train.py configs/reg_sam/regnet_ade20k.py \
    --resume work_dirs/regnet_ade20k/latest.pth

# 使用COCO预训练
python tools/train.py configs/reg_sam/regnet_ade20k.py \
    --cfg-options load_from=work_dirs/regnet_coco/epoch_12.pth
```

## 文档索引

| 文档 | 用途 |
|------|------|
| `ADE20K_QUICKSTART.md` | ⭐ 5分钟快速上手 |
| `CLASSIFIER_SOLUTION.md` | ⭐ 解决分类器错误 |
| `ADE20K_TRAINING_GUIDE.md` | 详细训练配置 |
| `GENERATE_CLASSIFIER_GUIDE.md` | 分类器技术细节 |
| `DATASET_COMPARISON.md` | COCO vs ADE20K |
| `MEMORY_OPTIMIZATION.md` | 显存优化策略 |

## 脚本说明

| 脚本 | 功能 |
|------|------|
| `gen_ade20k_classifier.sh` | 生成文本分类器 |
| `train_ade20k.sh` | 启动训练 |
| `test_ade20k.sh` | 评估模型 |
| `tools/check_ade_files.py` | 验证数据集 |

## 预期时间

| 任务 | 时间 (8 GPUs) |
|------|---------------|
| 生成分类器 | 5-10分钟 |
| 训练50 epochs | 17-20小时 |
| 验证 | 约10分钟/epoch |

## 帮助命令

```bash
# 查看快速解决方案
cat CLASSIFIER_SOLUTION.md

# 查看快速开始
cat ADE20K_QUICKSTART.md

# 列出所有文档
ls -lh *.md
```

