"""
Jittor 适配工具函数模块

该模块提供了从 PyTorch 迁移到 Jittor 所需的兼容性包装器和辅助函数。
主要包含：
1. 数据结构容器：ImageList（图像列表）、Instances（实例容器）、Boxes（边界框容器）
2. 模型管理：DetectionCheckpointer（检查点管理器）、JittorConfigurable（可配置基类）
3. 工具函数：图像转换、参数获取、IoU计算等
"""

import jittor as jt
import jittor.nn as nn
from jittor.nn import functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union


class JittorConfigurable:
    """Jittor 模型的可配置基类
    
    这个基类为所有需要从配置文件初始化的模型提供统一接口。
    子类需要重写 from_config 方法以实现具体的配置逻辑。
    """
    
    @classmethod
    def from_config(cls, cfg):
        """从配置对象创建模型实例
        
        Args:
            cfg: 配置对象，通常包含模型的所有超参数
            
        Returns:
            模型实例
            
        注意：子类必须重写此方法
        """
        raise NotImplementedError


def convert_image_to_rgb(image):
    """将图像张量转换为 RGB 格式
    
    如果输入是单通道灰度图（shape: [1, H, W]），则将其复制为三通道。
    
    Args:
        image: 输入图像张量，shape 可以是 [C, H, W]
    
    Returns:
        RGB 格式的图像张量，shape: [3, H, W]
    """
    # 如果是单通道图像，复制为三通道
    if len(image.shape) == 3 and image.shape[0] == 1:
        image = jt.repeat(image, 3, axis=0)
    return image


def get_model_parameters(model):
    """获取模型的所有可训练参数
    
    Args:
        model: Jittor 模型
        
    Returns:
        参数列表
    """
    return list(model.parameters())


class DetectionCheckpointer:
    """检测模型的检查点管理器
    
    负责保存和加载模型检查点，支持同时保存模型、优化器和学习率调度器的状态。
    这对于长时间训练和断点续训非常重要。
    """
    
    def __init__(self, model, save_dir=None, optimizer=None, scheduler=None):
        """初始化检查点管理器
        
        Args:
            model: 要保存的模型
            save_dir: 保存检查点的目录路径
            optimizer: 优化器（可选）
            scheduler: 学习率调度器（可选）
        """
        self.model = model
        self.save_dir = save_dir
        self.optimizer = optimizer
        self.scheduler = scheduler
        
    def save(self, name: str):
        """保存检查点到磁盘
        
        Args:
            name: 检查点文件名（不含扩展名）
        """
        if self.save_dir is None:
            return
        
        import os
        os.makedirs(self.save_dir, exist_ok=True)
        
        state = {
            'model': self.model.state_dict() if hasattr(self.model, 'state_dict') else self.model,
        }
        
        if self.optimizer is not None:
            state['optimizer'] = self.optimizer.state_dict() if hasattr(self.optimizer, 'state_dict') else self.optimizer
        
        if self.scheduler is not None:
            state['scheduler'] = self.scheduler.state_dict() if hasattr(self.scheduler, 'state_dict') else self.scheduler
        
        save_path = os.path.join(self.save_dir, f"{name}.pkl")
        jt.save(state, save_path)
        print(f"Checkpoint saved to {save_path}")
    
    def load(self, checkpoint_path: str):
        """Load checkpoint."""
        state = jt.load(checkpoint_path)
        
        if 'model' in state:
            if hasattr(self.model, 'load_state_dict'):
                self.model.load_state_dict(state['model'])
            else:
                for k, v in state['model'].items():
                    setattr(self.model, k, v)
        
        if self.optimizer is not None and 'optimizer' in state:
            if hasattr(self.optimizer, 'load_state_dict'):
                self.optimizer.load_state_dict(state['optimizer'])
        
        if self.scheduler is not None and 'scheduler' in state:
            if hasattr(self.scheduler, 'load_state_dict'):
                self.scheduler.load_state_dict(state['scheduler'])
    
    def resume_or_load(self, checkpoint_path: str, resume=True):
        """Resume from checkpoint or load weights only."""
        if checkpoint_path and checkpoint_path != '':
            self.load(checkpoint_path)


class ImageList:
    """图像列表容器
    
    在批处理中，不同图像可能有不同尺寸。ImageList 将它们填充到统一大小，
    同时记录每张图像的原始尺寸，以便后续处理时恢复。
    """
    
    def __init__(self, tensor, image_sizes):
        """初始化图像列表
        
        Args:
            tensor: 批处理后的图像张量，shape: (N, C, H, W)
                   N: batch size（批大小）
                   C: 通道数（通常为3，RGB）
                   H, W: 填充后的统一高度和宽度
            image_sizes: 每张图像填充前的原始尺寸列表，格式: [(H1, W1), (H2, W2), ...]
        """
        self.tensor = tensor
        self.image_sizes = image_sizes
    
    def __len__(self):
        """返回批中的图像数量"""
        return len(self.image_sizes)
    
    @property
    def device(self):
        """返回图像所在的设备"""
        return self.tensor.device


class Instances:
    """检测实例容器
    
    用于存储检测结果或标注信息，包括边界框、类别标签、置信度分数等。
    这是一个灵活的容器，可以动态添加各种字段（如 pred_boxes, scores, pred_classes 等）。
    
    常用字段：
        - gt_boxes: 真实边界框（训练时）
        - gt_classes: 真实类别标签（训练时）
        - pred_boxes: 预测边界框（推理时）
        - scores: 置信度分数（推理时）
        - pred_classes: 预测类别（推理时）
    """
    
    def __init__(self, image_size: Tuple[int, int]):
        """初始化实例容器
        
        Args:
            image_size: 图像尺寸 (高度, 宽度)
        """
        self.image_size = image_size
        self._fields = {}  # 存储所有字段的字典
    
    def __setattr__(self, name: str, value):
        if name.startswith('_'):
            super().__setattr__(name, value)
        else:
            self._fields[name] = value
    
    def __getattr__(self, name: str):
        if name.startswith('_'):
            return super().__getattribute__(name)
        if name in self._fields:
            return self._fields[name]
        raise AttributeError(f"Instances has no field {name}")
    
    def __len__(self):
        field_names = list(self._fields.keys())
        if len(field_names) == 0:
            return 0
        return len(self._fields[field_names[0]])
    
    def get(self, name: str, default=None):
        """获取指定字段的值
        
        Args:
            name: 字段名
            default: 如果字段不存在，返回的默认值
            
        Returns:
            字段值或默认值
        """
        return self._fields.get(name, default)
    
    def has(self, name: str):
        """检查字段是否存在
        
        Args:
            name: 字段名
            
        Returns:
            布尔值，表示字段是否存在
        """
        return name in self._fields
    
    def remove(self, name: str):
        """删除指定字段
        
        Args:
            name: 要删除的字段名
        """
        del self._fields[name]
    
    def边界框容器
    
    封装边界框的存储和操作，提供面积计算、IoU计算、裁剪等常用功能。
    边界框格式统一使用 (x1, y1, x2, y2)，即左上角和右下角坐标。
    """
    
    def __init__(self, tensor):
        """初始化边界框容器
        
        Args:
            tensor: 边界框张量，shape: (N, 4)
                   每行格式为 (x1, y1, x2, y2):
                   - x1, y1: 左上角坐标
                   - x2, y2: 右下角坐标
                   支持 numpy 数组或 Jittor 张量
        """
        if isinstance(tensor, np.ndarray):
            tensor = jt.array(tensor)  # 转换 numpy 数组为 Jittor 张量
        self.tensor = tensor
    
    @property
    def device(self):
        """返回边界框所在的设备"""
        return self.tensor.device
    
    def __len__(self):
        """返回边界框的数量"""
        return len(self.tensor)
    
    def __getitem__(self, idx):
        """支持索引访问
        
        Args:
            idx: 索引，可以是整数、切片或布尔掩码
            
        Returns:
            Boxes 对象，包含选中的边界框
        """
        return Boxes(self.tensor[idx])
    
    def __repr__(self):
        """字符串表示"""trs.append(f"{name}: {type(value)}")
        return f"Instances(image_size={self.image_size}, num_instances={len(self)}, " + ", ".join(field_strs) + ")"


class Boxes:
    """Container for bounding boxes."""
    
    def __init__(self, tensor):
        """
        Args:
            tensor: Tensor of shape (N, 4) in format (x1, y1, x2, y2)
        """
        if isinstance(tensor, np.ndarray):
            tensor = jt.array(tensor)
        self.tensor = tensor
    
    @property
    def device(self):
        return self.tensor.device
    
    def __len__(self):
        return len(self.tensor)
    
    def __getitem__(self, idx):
        return Boxes(self.tensor[idx])
    
    def __repr__(self):
        return f"Boxes({len(self)} boxes)"
    
    def to(self, device):
        """Move boxes to device."""
        if hasattr(self.tensor, 'to'):
            self.tensor = self.tensor.to(device)
        return self
    
    def area(self):
        """Compute area of boxes."""
        x1, y1, x2, y2 = self.tensor.unbind(dim=1)
        return (x2 - x1) * (y2 - y1)
    
    def clip(self, box_size: Tuple[int, int]):
        """Clip boxes to image boundaries."""
        h, w = box_size
        x1 = jt.clamp(self.tensor[:, 0], min=0, max=w)
        y1 = jt.clamp(self.tensor[:, 1], min=0, max=h)
        x2 = jt.clamp(self.tensor[:, 2], min=0, max=w)
        y2 = jt.clamp(self.tensor[:, 3], min=0, max=h)
        clipped = jt.stack([x1, y1, x2, y2], dim=1)
        return Boxes(clipped)
    
    @staticmethod
    def cat(boxes_list: List["Boxes"]):
        """Concatenate multiple box tensors."""
        if not boxes_list:
            return Boxes(jt.zeros((0, 4)))
        cat_tensor = jt.concat([b.tensor for b in boxes_list], dim=0)
        return Boxes(cat_tensor)


def pairwise_iou(boxes1: Boxes, boxes2: Boxes) -> jt.Var:
    """Compute pairwise IoU between two sets of boxes.
    
    Args:
        boxes1: N boxes in format (x1, y1, x2, y2)
        boxes2: M boxes in format (x1, y1, x2, y2)
    
    Returns:
        IoU matrix of shape (N, M)
    """
    x1_1, y1_1, x2_1, y2_1 = boxes1.tensor.unbind(dim=1)
    x1_2, y1_2, x2_2, y2_2 = boxes2.tensor.unbind(dim=1)
    
    # Compute intersection
    inter_x1 = jt.maximum(x1_1[:, None], x1_2[None, :])
    inter_y1 = jt.maximum(y1_1[:, None], y1_2[None, :])
    inter_x2 = jt.minimum(x2_1[:, None], x2_2[None, :])
    inter_y2 = jt.minimum(y2_1[:, None], y2_2[None, :])
    
    inter_w = jt.clamp(inter_x2 - inter_x1, min=0)
    inter_h = jt.clamp(inter_y2 - inter_y1, min=0)
    inter_area = inter_w * inter_h
    
    # Compute union
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = area1[:, None] + area2[None, :] - inter_area
    
    # Compute IoU
    iou = inter_area / (union_area + 1e-8)
    return iou


def get_event_storage():
    """Get event storage for logging. Simple stub implementation."""
    return SimpleEventStorage()


class SimpleEventStorage:
    """Simple event storage for logging."""
    
    def __init__(self):
        self.events = {}
    
    def put_scalar(self, name: str, value: float):
        if name not in self.events:
            self.events[name] = []
        self.events[name].append(value)
    
    def put_scalars(self, **kwargs):
        for name, value in kwargs.items():
            self.put_scalar(name, value)


def build_backbone(cfg):
    """Build backbone network from config."""
    from daod_jittor.modeling.meta_arch.vgg_jittor import VGGBackbone
    
    if cfg.MODEL.BACKBONE.NAME == "build_vgg_backbone":
        return VGGBackbone(cfg)
    else:
        raise NotImplementedError(f"Backbone {cfg.MODEL.BACKBONE.NAME} not implemented")


def build_proposal_generator(cfg, backbone_output_shape):
    """Build RPN from config."""
    from daod_jittor.modeling.proposal_generator.rpn_jittor import RPN
    
    return RPN(cfg, backbone_output_shape)


def build_roi_heads(cfg, backbone_output_shape):
    """Build ROI heads from config."""
    from daod_jittor.modeling.roi_heads.roi_heads_jittor import StandardROIHeads
    
    return StandardROIHeads(cfg, backbone_output_shape)
