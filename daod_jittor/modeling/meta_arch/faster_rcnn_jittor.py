"""
Faster R-CNN 的 Jittor 实现

Faster R-CNN 是一个经典的两阶段目标检测算法：
1. 第一阶段：Region Proposal Network (RPN) 生成候选区域
2. 第二阶段：ROI Head 对候选区域进行分类和边界框回归

本模块实现了：
- GeneralizedRCNN: R-CNN 系列模型的通用基类
- FasterRCNN: 标准 Faster R-CNN 模型
- SourceFreeAdaptiveTeacherRCNN: 源无关域自适应版本（用于 SFOD）
"""

import jittor as jt
import jittor.nn as nn
from jittor.nn import functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from daod_jittor.jittor_utils import (
    ImageList, Instances, Boxes, JittorConfigurable,
    convert_image_to_rgb
)
from daod_jittor.modeling.meta_arch.vgg_jittor import VGGBackbone, build_vgg_backbone
from daod_jittor.modeling.proposal_generator.rpn_jittor import RPN
from daod_jittor.modeling.roi_heads.roi_heads_jittor import StandardROIHeads


class GeneralizedRCNN(nn.Module, JittorConfigurable):
    """R-CNN 系列模型的通用基类
    
    这个类实现了 R-CNN 风格检测模型的基础前向传播流程：
    1. 图像预处理和归一化
    2. 骨干网络提取特征
    3. RPN 生成候选区域
    4. ROI Head 进行分类和回归
    
    支持训练和推理两种模式。
    """
    
    def __init__(self,
                 backbone: nn.Module,
                 proposal_generator: nn.Module,
                 roi_heads: nn.Module,
                 pixel_mean: Tuple[float, ...] = (103.53, 116.28, 123.675),
                 pixel_std: Tuple[float, ...] = (1.0, 1.0, 1.0),
                 input_format: str = "RGB",
                 vis_period: int = 0):
        """初始化 R-CNN 模型
        
        Args:
            backbone: 骨干网络，用于提取图像特征（如 VGG、ResNet）
            proposal_generator: 候选区域生成器（RPN）
            roi_heads: ROI 头部，用于对候选区域进行分类和回归
            pixel_mean: 图像像素均值，用于归一化（BGR 格式）
                       默认值来自 ImageNet 数据集
            pixel_std: 图像像素标准差，用于归一化
            input_format: 输入图像格式，"RGB" 或 "BGR"
            vis_period: 可视化周期，0 表示禁用可视化
        """
        super().__init__()
        
        self.backbone = backbone
        self.proposal_generator = proposal_generator
        self.roi_heads = roi_heads
        
        self.input_format = input_format
        self.vis_period = vis_period
        
        # 注册像素归一化的缓冲区（不参与梯度更新）
        # 将均值和标准差转换为 [C, 1, 1] 形状，便于广播
        self.register_buffer(
            "pixel_mean",
            jt.array(pixel_mean, dtype=jt.float32).reshape(-1, 1, 1)
        )
        self.register_buffer(
            "pixel_std",
            jt.array(pixel_std, dtype=jt.float32).reshape(-1, 1, 1)
        )
    
    def register_buffer(self, name: str, tensor: jt.Var):
        """注册缓冲区张量
        
        缓冲区是模型的一部分，会被保存到检查点，但不会被优化器更新。
        通常用于存储固定的参数，如归一化的均值和标准差。
        
        Args:
            name: 缓冲区名称
            tensor: 要注册的张量
        """
        setattr(self, name, tensor)
    
    def forward(self, batched_inputs: List[Dict]) -> Union[Dict, List[Instances]]:
        """前向传播（训练或推理）
        
        Args:
            batched_inputs: 批处理输入列表，每个元素是一个字典，包含：
                - 'image': 输入图像张量，shape: (C, H, W)
                - 'instances': （仅训练时）真实标注实例
                - 'height', 'width': 图像的原始尺寸
        
        Returns:
            训练模式：返回损失字典，如：
                {
                    'loss_cls': 分类损失,
                    'loss_box_reg': 边界框回归损失,
                    'loss_rpn_cls': RPN 分类损失,
                    'loss_rpn_loc': RPN 定位损失
                }
            
            推理模式：返回检测结果列表，每个元素是一个 Instances 对象，包含：
                - pred_boxes: 预测的边界框
                - scores: 置信度分数
                - pred_classes: 预测的类别
        """
        # 预处理图像
        images = self._preprocess_images(batched_inputs)
        
        # Extract features from backbone
        features = self.backbone(images.tensor)
        
        # Generate proposals with RPN
        proposals, proposal_losses = self.proposal_generator(
            features,
            [(h, w) for h, w in zip(batched_inputs, batched_inputs)]
        )
        
        # ROI heads - detection
        if self.training:
            # Training mode
            instances = [x["instances"] for x in batched_inputs]
            detection_losses = self.roi_heads(features, proposals, instances)
            losses = {**proposal_losses, **detection_losses}
            return losses
        else:
            # Inference mode
            detected_instances = self.roi_heads(features, proposals)
            return detected_instances
    
    def _preprocess_images(self, batched_inputs: List[Dict]) -> ImageList:
        """预处理输入图像
        
        执行以下操作：
        1. 格式转换（如果需要 BGR 到 RGB）
        2. 像素归一化（减均值除标准差）
        3. 批处理对齐（填充到相同尺寸）
        
        Args:
            batched_inputs: 输入字典列表
        
        Returns:
            ImageList 对象，包含处理后的图像张量和原始尺寸
        """
            
        images = []
        sizes = []
        
        for input_dict in batched_inputs:
            img = input_dict["image"]
            
            # Convert to RGB if needed
            if self.input_format == "BGR":
                img = img[[2, 1, 0], :, :]  # BGR to RGB
            
            # Normalize
            img = img.float()
            img = (img - self.pixel_mean) / self.pixel_std
            
            images.append(img)
            sizes.append((img.shape[1], img.shape[2]))
        
        # Stack images (with zero-padding if needed)
        max_h = max(s[0] for s in sizes)
        max_w = max(s[1] for s in sizes)
        
        stacked_images = []
        for img in images:
            padded = jt.zeros((3, max_h, max_w), dtype=img.dtype)
            h, w = img.shape[1], img.shape[2]
            padded[:, :h, :w] = img
            stacked_images.append(padded)
        
        batched_images = jt.stack(stacked_images, dim=0)
        return ImageList(batched_images, sizes)
    
    @classmethod
    def from_config(cls, cfg):
        """从配置对象创建模型
        
        Args:
            cfg: 配置对象，包含模型的所有超参数
        
        Returns:
            包含模型组件的字典
        """
        # 构建骨干网络
        backbone = build_vgg_backbone(cfg)
        # 构建 RPN
        proposal_generator = RPN(cfg, backbone.output_shape)
        # 构建 ROI 头部
        roi_heads = StandardROIHeads(cfg, backbone.output_shape)
        
        return {
            "backbone": backbone,
            "proposal_generator": proposal_generator,
            "roi_heads": roi_heads,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN if hasattr(cfg.MODEL, 'PIXEL_MEAN') else (103.53, 116.28, 123.675),
            "pixel_std": cfg.MODEL.PIXEL_STD if hasattr(cfg.MODEL, 'PIXEL_STD') else (1.0, 1.0, 1.0),
            "input_format": cfg.INPUT.FORMAT if hasattr(cfg.INPUT, 'FORMAT') else "RGB",
            "vis_period": cfg.VIS_PERIOD if hasattr(cfg, 'VIS_PERIOD') else 0,
        }


class FasterRCNN(GeneralizedRCNN):
    """标准 Faster R-CNN 检测器
    
    Faster R-CNN 是一个经典的两阶段目标检测算法：
    1. RPN 生成候选区域（约 2000 个）
    2. ROI Head 对每个候选区域进行分类和边界框精修
    
    相比单阶段检测器（如 YOLO），Faster R-CNN 通常有更高的精度但速度较慢。
    """
    
    def __init__(self,
                 backbone: nn.Module,
                 proposal_generator: nn.Module,
                 roi_heads: nn.Module,
                 pixel_mean: Tuple[float, ...] = (103.53, 116.28, 123.675),
                 pixel_std: Tuple[float, ...] = (1.0, 1.0, 1.0),
                 input_format: str = "RGB",
                 vis_period: int = 0):
        """初始化 Faster R-CNN 模型
        
        Args:
            backbone: 骨干网络（如 VGG-16）
            proposal_generator: RPN 网络
            roi_heads: ROI 头部网络
            pixel_mean: 像素均值
            pixel_std: 像素标准差
            input_format: 输入格式 ("RGB" 或 "BGR")
            vis_period: 可视化周期
        """
        super().__init__(
            backbone=backbone,
            proposal_generator=proposal_generator,
            roi_heads=roi_heads,
            pixel_mean=pixel_mean,
            pixel_std=pixel_std,
            input_format=input_format,
            vis_period=vis_period,
        )
    
    @classmethod
    def from_config(cls, cfg):
        """Create Faster R-CNN from config."""
        return GeneralizedRCNN.from_config(cfg)


class SourceFreeAdaptiveTeacherRCNN(GeneralizedRCNN):
    """Faster R-CNN with source-free domain adaptation using teacher-student framework."""
    
    def __init__(self,
                 backbone: nn.Module,
                 proposal_generator: nn.Module,
                 roi_heads: nn.Module,
                 pixel_mean: Tuple[float, ...] = (103.53, 116.28, 123.675),
                 pixel_std: Tuple[float, ...] = (1.0, 1.0, 1.0),
                 input_format: str = "RGB",
                 vis_period: int = 0,
                 dis_type: str = "img",
                 ins_dc: bool = False):
        """
        Args:
            backbone: Backbone network
            proposal_generator: RPN for generating proposals
            roi_heads: ROI head for object detection
            pixel_mean: Pixel mean for normalization
            pixel_std: Pixel std for normalization
            input_format: Input image format
            vis_period: Visualization period
            dis_type: Type of discriminator ("img" or "ins")
            ins_dc: Whether to use instance-level domain classifier
        """
        super().__init__(
            backbone=backbone,
            proposal_generator=proposal_generator,
            roi_heads=roi_heads,
            pixel_mean=pixel_mean,
            pixel_std=pixel_std,
            input_format=input_format,
            vis_period=vis_period,
        )
        
        self.dis_type = dis_type
        self.ins_dc = ins_dc
        
        # Discriminator for domain adaptation
        self.discriminator = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )
    
    def forward(self, batched_inputs: List[Dict], 
                batched_inputs_t: Optional[List[Dict]] = None) -> Union[Dict, List[Instances]]:
        """Forward pass for source-free domain adaptation.
        
        Args:
            batched_inputs: Source batch inputs (training only)
            batched_inputs_t: Target batch inputs
        
        Returns:
            Dict of losses during training, or List of instances during inference
        """
        if self.training and batched_inputs_t is not None:
            # Training with both source and target
            return self._forward_train(batched_inputs, batched_inputs_t)
        elif batched_inputs_t is not None:
            # Inference on target
            return self._forward_inference(batched_inputs_t)
        else:
            # Standard forward
            return super().forward(batched_inputs)
    
    def _forward_train(self, batched_inputs, batched_inputs_t):
        """Forward pass during training with domain adaptation."""
        # Source forward pass
        images = self._preprocess_images(batched_inputs)
        features = self.backbone(images.tensor)
        
        # Target forward pass
        images_t = self._preprocess_images(batched_inputs_t)
        features_t = self.backbone(images_t.tensor)
        
        # Generate proposals
        proposals, proposal_losses = self.proposal_generator(
            features,
            [(h, w) for _, (h, w) in enumerate(batched_inputs)]
        )
        
        # ROI heads
        instances = [x["instances"] for x in batched_inputs]
        detection_losses = self.roi_heads(features, proposals, instances)
        
        # Domain adaptation losses
        da_losses = self._compute_da_losses(features, features_t)
        
        # Combine all losses
        all_losses = {**proposal_losses, **detection_losses, **da_losses}
        return all_losses
    
    def _forward_inference(self, batched_inputs_t):
        """Forward pass during inference on target domain."""
        images_t = self._preprocess_images(batched_inputs_t)
        features_t = self.backbone(images_t.tensor)
        
        proposals, _ = self.proposal_generator(
            features_t,
            [(h, w) for _, (h, w) in enumerate(batched_inputs_t)]
        )
        
        detected_instances = self.roi_heads(features_t, proposals)
        return detected_instances
    
    def _compute_da_losses(self, features_s, features_t):
        """Compute domain adaptation losses."""
        losses = {}
        
        if self.dis_type == "img":
            # Image-level domain classification loss
            # This is a placeholder - real implementation would use GRL
            loss_da_img = F.mse_loss(features_s, features_t)
            losses["loss_da_img"] = loss_da_img * 0.01
        
        return losses


def build_model(cfg):
    """Build a detection model from config."""
    if cfg.MODEL.META_ARCHITECTURE == "GeneralizedRCNN":
        return GeneralizedRCNN.from_config(cfg)
    elif cfg.MODEL.META_ARCHITECTURE == "FasterRCNN":
        return FasterRCNN.from_config(cfg)
    elif cfg.MODEL.META_ARCHITECTURE == "SourceFreeAdaptiveTeacherRCNN":
        return SourceFreeAdaptiveTeacherRCNN.from_config(cfg)
    else:
        raise ValueError(f"Unknown model architecture: {cfg.MODEL.META_ARCHITECTURE}")
