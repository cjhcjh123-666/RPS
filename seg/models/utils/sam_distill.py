"""
SAM知识蒸馏模块
使用SAM（Segment Anything Model）作为教师模型，对学生模型进行知识蒸馏
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
from mmdet.registry import MODELS
from mmdet.utils import ConfigType


class SAMDistillModule(nn.Module):
    """SAM知识蒸馏模块。
    
    该模块使用预训练的SAM模型作为教师，对学生模型进行知识蒸馏。
    支持特征蒸馏和输出蒸馏两种方式。
    """
    
    def __init__(
        self,
        teacher_model: Optional[ConfigType] = None,
        teacher_checkpoint: Optional[str] = None,
        feat_distill_weight: float = 1.0,
        output_distill_weight: float = 1.0,
        temperature: float = 4.0,
        distill_feat_layers: Optional[List[int]] = None,
        # 显存优化参数
        use_low_res_sam: bool = True,  # 使用低分辨率SAM输入（512而不是1024）
        sam_input_size: int = 512,  # SAM输入尺寸（降低以节省显存）
        distill_interval: int = 1,  # 每N个迭代进行一次SAM蒸馏（1=每次都蒸馏）
    ):
        super(SAMDistillModule, self).__init__()
        
        self.feat_distill_weight = feat_distill_weight
        self.output_distill_weight = output_distill_weight
        self.temperature = temperature
        
        # 显存优化参数
        self.use_low_res_sam = use_low_res_sam
        self.sam_input_size = sam_input_size
        self.distill_interval = distill_interval
        self._iter_count = 0  # 迭代计数器
        
        # 构建教师模型（如果提供配置）
        self.teacher_model = None
        if teacher_model is not None:
            # 如果teacher_model是dict配置，构建模型
            if isinstance(teacher_model, dict):
                # 如果配置中指定了checkpoint，先加载checkpoint
                if teacher_checkpoint is not None:
                    teacher_model['checkpoint'] = teacher_checkpoint
                self.teacher_model = MODELS.build(teacher_model)
            else:
                # 如果是已经构建的模型
                self.teacher_model = teacher_model
                if teacher_checkpoint is not None:
                    self._load_teacher_checkpoint(teacher_checkpoint)
            
            # 冻结教师模型参数
            for param in self.teacher_model.parameters():
                param.requires_grad = False
            self.teacher_model.eval()
        elif teacher_checkpoint is not None:
            # 如果没有提供teacher_model配置，但提供了checkpoint，尝试使用默认配置
            import logging
            logger = logging.getLogger()
            logger.warning(
                "teacher_model is None but teacher_checkpoint is provided. "
                "Trying to use SAMTeacherModel with default config."
            )
            try:
                # 尝试使用SAMTeacherModel（需要segment-anything-main在项目中）
                self.teacher_model = MODELS.build(dict(
                    type='SAMTeacherModel',
                    model_type='vit_h',  # 默认使用vit_h
                    checkpoint=teacher_checkpoint,
                    freeze=True
                ))
            except Exception as e:
                logger.error(f"Failed to build SAM teacher model: {e}")
                logger.info("SAM distillation will be disabled.")
                self.teacher_model = None
        
        # 需要蒸馏的特征层索引
        self.distill_feat_layers = distill_feat_layers or [0, 1, 2, 3]
        
        # KL散度损失用于输出蒸馏
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        
        # MSE损失用于特征蒸馏
        self.mse_loss = nn.MSELoss()
    
    def _load_teacher_checkpoint(self, checkpoint_path: str):
        """加载教师模型检查点"""
        import torch
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        # 移除可能的前缀
        if self.teacher_model is not None:
            self.teacher_model.load_state_dict(state_dict, strict=False)
    
    def forward_teacher(
        self, 
        batch_inputs: torch.Tensor, 
        use_low_res: bool = True,
        student_backbone: Optional[torch.nn.Module] = None
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播教师模型，获取教师特征和输出。
        
        Args:
            batch_inputs: 输入图像，形状为 (B, C, H, W)
            use_low_res: 是否使用低分辨率输入
            student_backbone: 学生模型的backbone（用于checkpoint模式）
        
        Returns:
            dict: 包含教师模型的特征和输出
                - 'features': 多尺度特征列表
                - 'cls_scores': 分类分数
                - 'mask_preds': 掩码预测
        """
        if self.teacher_model is None:
            return {}
        
        # 显存优化：使用低分辨率输入
        if use_low_res and self.use_low_res_sam:
            # 将输入resize到较小尺寸以节省显存
            import torch.nn.functional as F
            original_size = batch_inputs.shape[-2:]
            batch_inputs_low = F.interpolate(
                batch_inputs,
                size=(self.sam_input_size, self.sam_input_size),
                mode='bilinear',
                align_corners=False
            )
        else:
            batch_inputs_low = batch_inputs
        
        with torch.no_grad():
            # 提取特征（使用低分辨率输入）
            teacher_feats = self.teacher_model.extract_feat(batch_inputs_low)
            
            # 获取head输出（需要根据实际模型结构调整）
            if hasattr(self.teacher_model, 'panoptic_head'):
                # 创建虚拟的data_samples用于前向传播
                from mmengine.structures import InstanceData
                from mmdet.structures import DetDataSample
                
                batch_size = batch_inputs.shape[0]
                dummy_data_samples = []
                for i in range(batch_size):
                    data_sample = DetDataSample()
                    data_sample.set_metainfo({
                        'img_shape': batch_inputs.shape[-2:],
                        'ori_shape': batch_inputs.shape[-2:],
                        'scale_factor': [1.0, 1.0, 1.0, 1.0]
                    })
                    dummy_data_samples.append(data_sample)
                
                teacher_outputs = self.teacher_model.panoptic_head.forward(teacher_feats, dummy_data_samples)
                
                return {
                    'features': teacher_feats if isinstance(teacher_feats, (list, tuple)) else [teacher_feats],
                    'cls_scores': teacher_outputs[0] if len(teacher_outputs) > 0 else None,
                    'mask_preds': teacher_outputs[1] if len(teacher_outputs) > 1 else None,
                }
            else:
                return {
                    'features': teacher_feats if isinstance(teacher_feats, (list, tuple)) else [teacher_feats],
                }
    
    def compute_feature_distill_loss(
        self,
        student_feats: List[torch.Tensor],
        teacher_feats: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        计算特征蒸馏损失。
        
        Args:
            student_feats: 学生模型的多尺度特征列表
            teacher_feats: 教师模型的多尺度特征列表
        
        Returns:
            Tensor: 特征蒸馏损失
        """
        if len(teacher_feats) == 0:
            return torch.tensor(0.0, device=student_feats[0].device)
        
        losses = []
        
        # 对每个需要蒸馏的层计算损失
        for layer_idx in self.distill_feat_layers:
            if layer_idx < len(student_feats) and layer_idx < len(teacher_feats):
                student_feat = student_feats[layer_idx]
                teacher_feat = teacher_feats[layer_idx]
                
                # 如果尺寸不匹配，进行插值对齐
                if student_feat.shape != teacher_feat.shape:
                    teacher_feat = F.interpolate(
                        teacher_feat,
                        size=student_feat.shape[-2:],
                        mode='bilinear',
                        align_corners=False
                    )
                
                # 计算MSE损失
                loss = self.mse_loss(student_feat, teacher_feat)
                losses.append(loss)
        
        if len(losses) == 0:
            return torch.tensor(0.0, device=student_feats[0].device)
        
        return sum(losses) / len(losses)
    
    def compute_output_distill_loss(
        self,
        student_cls_scores: torch.Tensor,
        student_mask_preds: torch.Tensor,
        teacher_cls_scores: Optional[torch.Tensor] = None,
        teacher_mask_preds: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        计算输出蒸馏损失。
        
        Args:
            student_cls_scores: 学生模型的分类分数，形状为 (B, N, C)
            student_mask_preds: 学生模型的掩码预测，形状为 (B, N, H, W)
            teacher_cls_scores: 教师模型的分类分数
            teacher_mask_preds: 教师模型的掩码预测
        
        Returns:
            dict: 包含各种输出蒸馏损失
                - 'loss_cls_distill': 分类蒸馏损失
                - 'loss_mask_distill': 掩码蒸馏损失
        """
        losses = {}
        
        # 分类蒸馏损失
        if teacher_cls_scores is not None:
            # 使用KL散度进行软标签蒸馏
            student_log_probs = F.log_softmax(student_cls_scores / self.temperature, dim=-1)
            teacher_probs = F.softmax(teacher_cls_scores / self.temperature, dim=-1)
            
            loss_cls_distill = self.kl_loss(student_log_probs, teacher_probs) * (self.temperature ** 2)
            losses['loss_cls_distill'] = loss_cls_distill
        
        # 掩码蒸馏损失
        if teacher_mask_preds is not None:
            # 如果尺寸不匹配，进行插值对齐
            if student_mask_preds.shape != teacher_mask_preds.shape:
                teacher_mask_preds = F.interpolate(
                    teacher_mask_preds,
                    size=student_mask_preds.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                )
            
            # 使用MSE损失
            loss_mask_distill = self.mse_loss(
                torch.sigmoid(student_mask_preds),
                torch.sigmoid(teacher_mask_preds)
            )
            losses['loss_mask_distill'] = loss_mask_distill
        
        return losses
    
    def forward(
        self,
        student_feats: List[torch.Tensor],
        student_cls_scores: Optional[torch.Tensor] = None,
        student_mask_preds: Optional[torch.Tensor] = None,
        batch_inputs: Optional[torch.Tensor] = None,
        student_backbone: Optional[torch.nn.Module] = None
    ) -> Dict[str, torch.Tensor]:
        """
        计算SAM蒸馏损失。
        
        Args:
            student_feats: 学生模型的多尺度特征
            student_cls_scores: 学生模型的分类分数
            student_mask_preds: 学生模型的掩码预测
            batch_inputs: 输入图像（用于前向传播教师模型）
        
        Returns:
            dict: 包含所有蒸馏损失的字典
        """
        losses = {}
        
        if self.teacher_model is None or batch_inputs is None:
            return losses
        
        # 显存优化：只在指定间隔进行SAM蒸馏
        self._iter_count += 1
        if self.distill_interval > 1 and self._iter_count % self.distill_interval != 0:
            return losses
        
        # 获取教师模型输出（传递student_backbone用于checkpoint模式，使用低分辨率）
        teacher_outputs = self.forward_teacher(batch_inputs, student_backbone=student_backbone, use_low_res=self.use_low_res_sam)
        
        # 特征蒸馏损失
        if 'features' in teacher_outputs and len(teacher_outputs['features']) > 0:
            loss_feat_distill = self.compute_feature_distill_loss(
                student_feats,
                teacher_outputs['features']
            )
            losses['loss_feat_distill'] = self.feat_distill_weight * loss_feat_distill
        
        # 输出蒸馏损失
        if student_cls_scores is not None or student_mask_preds is not None:
            output_losses = self.compute_output_distill_loss(
                student_cls_scores,
                student_mask_preds,
                teacher_outputs.get('cls_scores'),
                teacher_outputs.get('mask_preds')
            )
            for key, value in output_losses.items():
                losses[key] = self.output_distill_weight * value
        
        # 总蒸馏损失
        if len(losses) > 0:
            losses['loss_sam_distill'] = sum(losses.values())
        
        return losses

