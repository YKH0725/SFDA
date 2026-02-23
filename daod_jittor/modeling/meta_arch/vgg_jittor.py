"""
VGG 骨干网络的 Jittor 实现

VGG（Visual Geometry Group）是一个经典的卷积神经网络架构，以其简单而有效的设计著称。
本模块实现了用于目标检测的 VGG16 和 VGG19 骨干网络，包括：
1. VGGBackbone: 基础 VGG 特征提取器，输出多尺度特征
2. FPN: 特征金字塔网络，用于融合不同尺度的特征
3. LastLevelMaxPool/LastLevelP6P7: 额外的特征层级
"""

import jittor as jt
import jittor.nn as nn
from jittor.nn import functional as F
from typing import Dict, List, Optional, Tuple, Any
import math


class VGGBackbone(nn.Module):
    """VGG 骨干网络，用于特征提取
    
    VGG 网络通过堆叠 3x3 卷积和 2x2 最大池化层来提取图像特征。
    在目标检测中，我们需要不同尺度的特征图（res2, res3, res4, res5），
    对应不同的感受野，用于检测不同大小的物体。
    
    支持 VGG16 和 VGG19 两种深度，以及可选的批归一化。
    """
    
    def __init__(self, cfg, depth: int = 16, norm_layer=None, freeze_at=0):
        """初始化 VGG 骨干网络
        
        Args:
            cfg: 配置对象
            depth: VGG 深度，16 表示 VGG16，19 表示 VGG19
            norm_layer: 归一化层类型
                       None 或 nn.Identity: 不使用归一化
                       nn.BatchNorm2d: 使用批归一化（VGG-BN）
            freeze_at: 冻结的层级（0-5）
                      0: 不冻结任何层
                      1: 冻结 res2 之前的层
                      2: 冻结 res2 和 res3 之前的层
                      以此类推
        """
        super().__init__()
        
        self.depth = depth
        self.freeze_at = freeze_at
        self.norm_layer = norm_layer or nn.Identity
        
        # 定义 VGG 网络结构
        # 数字表示卷积层的输出通道数，'M' 表示最大池化层
        if depth == 16:
            # VGG16: 13 个卷积层 + 5 个池化层
            layers = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
        elif depth == 19:
            # VGG19: 16 个卷积层 + 5 个池化层
            layers = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
        else:
            raise ValueError(f"不支持的 VGG 深度: {depth}，仅支持 16 或 19")
        
        # 构建特征提取层
        self.features = self._build_layers(layers)
        
        # 初始化权重
        self._init_weights()
        
        # 输出特征名称列表
        # res2, res3, res4, res5 对应不同下采样倍数的特征图
        self._out_features = ["res2", "res3", "res4", "res5"]
        
        # 每个特征层的通道数
        self._out_feature_channels = {
            "res2": 128,   # 1/4 分辨率，128 通道
            "res3": 256,   # 1/8 分辨率，256 通道
            "res4": 512,   # 1/16 分辨率，512 通道
            "res5": 512,   # 1/32 分辨率，512 通道
        }
        
        # 输出形状信息（包含通道数和步长）
        self.output_shape = {
            "res2": {"channels": 128, "stride": 4},    # 步长 4 = 下采样 2 次（2^2）
            "res3": {"channels": 256, "stride": 8},    # 步长 8 = 下采样 3 次（2^3）
            "res4": {"channels": 512, "stride": 16},   # 步长 16 = 下采样 4 次（2^4）
            "res5": {"channels": 512, "stride": 32},   # 步长 32 = 下采样 5 次（2^5）
        }
    
    def _build_layers(self, cfg):
        """根据配置构建 VGG 网络层
        
        Args:
            cfg: 层配置列表，数字表示卷积层通道数，'M' 表示池化层
            
        Returns:
            nn.Sequential 模块
        """
        layers = []
        in_channels = 3  # 输入图像为 RGB，3 通道
        
        for v in cfg:
            if v == 'M':
                # 添加 2x2 最大池化层，步长为 2（下采样 2 倍）
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                v = int(v)
                # 添加 3x3 卷积层
                if self.norm_layer is not None and self.norm_layer != nn.Identity:
                    # 使用批归一化：Conv -> BN -> ReLU
                    layers.append(nn.Conv2d(in_channels, v, kernel_size=3, padding=1))
                    layers.append(self.norm_layer(v))
                    layers.append(nn.ReLU(inplace=True))
                else:
                    # 不使用归一化：Conv -> ReLU
                    layers.append(nn.Conv2d(in_channels, v, kernel_size=3, padding=1))
                    layers.append(nn.ReLU(inplace=True))
                in_channels = v
        
        return nn.Sequential(*layers)
    
    def _init_weights(self):
        """初始化网络权重
        
        使用 Kaiming 正态初始化（He initialization）来初始化卷积层权重，
        这对于使用 ReLU 激活函数的网络特别有效。
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Kaiming 初始化：根据输入神经元数量调整初始化方差
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                # BN 层：权重初始化为 1，偏置初始化为 0
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    
    def forward(self, x):
        """提取多尺度特征
        
        Args:
            x: 输入张量，shape: (B, 3, H, W)
               B: batch size
               H, W: 图像高度和宽度
        
        Returns:
            字典，包含不同尺度的特征图：
            {
                "res2": 1/4 分辨率特征图 (B, 128, H/4, W/4)
                "res3": 1/8 分辨率特征图 (B, 256, H/8, W/8)
                "res4": 1/16 分辨率特征图 (B, 512, H/16, W/16)
                "res5": 1/32 分辨率特征图 (B, 512, H/32, W/32)
            }
        """
        outputs = {}
        
        # res2 (1/4 分辨率)：经过前两个池化层
        x = self.features[0:6](x)
        outputs["res2"] = x
        
        # res3 (1/8 分辨率)：再经过一个池化层
        x = self.features[6:13](x)
        outputs["res3"] = x
        
        # res4 (1/16 分辨率)：再经过一个池化层
        x = self.features[13:23](x)
        outputs["res4"] = x
        
        # res5 (1/32 分辨率)：最后一个池化层
        x = self.features[23:](x)
        outputs["res5"] = x
        
        return outputs
    
    @property
    def out_features(self) -> List[str]:
        """Return output feature names."""
        return self._out_features
    
    @property
    def out_feature_channels(self) -> Dict[str, int]:
        """Return output feature channel counts."""
        return self._out_feature_channels


class FPN(nn.Module):
    """Feature Pyramid Network for multi-scale feature fusion.
    
    Builds a feature pyramid from higher-resolution features by lateral and top-down connections.
    """
    
    def __init__(self, in_channels_list: List[int], out_channels: int, 
                 num_levels: int = 4, top_blocks: Optional[nn.Module] = None):
        """
        Args:
            in_channels_list: List of input channel counts
            out_channels: Output channel count for all pyramid levels
            num_levels: Number of pyramid levels
            top_blocks: Optional additional blocks for highest level
        """
        super().__init__()
        
        self.in_channels_list = in_channels_list
        self.out_channels = out_channels
        self.num_levels = num_levels
        
        # Lateral convolutions
        self.lateral_convs = nn.ModuleList()
        for in_channels in in_channels_list:
            self.lateral_convs.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=1)
            )
        
        # Output convolutions
        self.fpn_convs = nn.ModuleList()
        for _ in range(num_levels):
            self.fpn_convs.append(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            )
        
        self.top_blocks = top_blocks
        
        # Initialize weights
        for conv in self.lateral_convs + self.fpn_convs:
            nn.init.xavier_uniform_(conv.weight)
            if conv.bias is not None:
                nn.init.constant_(conv.bias, 0)
    
    def forward(self, features: Dict[str, jt.Var]) -> Dict[str, jt.Var]:
        """Build FPN from backbone features.
        
        Args:
            features: Dictionary of backbone features with keys like "res2", "res3", etc.
        
        Returns:
            Dictionary of FPN features with keys "p2", "p3", etc.
        """
        # Get features in order (lowest to highest resolution)
        feature_keys = ["res2", "res3", "res4", "res5"]
        backbone_features = [features[k] for k in feature_keys]
        
        # Apply lateral convolutions
        laterals = [self.lateral_convs[i](backbone_features[i]) 
                   for i in range(len(backbone_features))]
        
        # Top-down path
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=laterals[i - 1].shape[-2:], mode='nearest'
            )
        
        # Apply FPN convolutions
        fpn_features = [self.fpn_convs[i](laterals[i]) 
                       for i in range(len(laterals))]
        
        # Create output dictionary
        result = {}
        for i, key in enumerate(["p2", "p3", "p4", "p5"]):
            result[key] = fpn_features[i]
        
        # Add additional levels if needed
        if self.top_blocks is not None:
            p6 = self.top_blocks(fpn_features[-1])
            result["p6"] = p6
        
        return result


class LastLevelMaxPool(nn.Module):
    """Builds additional level P6 level from P5."""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, x: jt.Var) -> jt.Var:
        """Apply max pooling to create P6."""
        return F.max_pool2d(x, kernel_size=2, stride=2)


class LastLevelP6P7(nn.Module):
    """Builds P6 and P7 levels from P5."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.p6 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.p7 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
        )
    
    def forward(self, x: jt.Var) -> Tuple[jt.Var, jt.Var]:
        """Generate P6 and P7 levels."""
        p6 = self.p6(x)
        p7 = self.p7(p6)
        return p6, p7


def build_vgg_backbone(cfg):
    """Build VGG backbone with optional FPN."""
    depth = cfg.MODEL.BACKBONE.DEPTH if hasattr(cfg.MODEL.BACKBONE, 'DEPTH') else 16
    
    backbone = VGGBackbone(cfg, depth=depth)
    
    # Optionally add FPN
    if cfg.MODEL.BACKBONE.USE_FPN:
        fpn = FPN(
            in_channels_list=[128, 256, 512, 512],
            out_channels=cfg.MODEL.FPN.OUT_CHANNELS,
            num_levels=4
        )
        
        class VGGWithFPN(nn.Module):
            def __init__(self, backbone, fpn):
                super().__init__()
                self.backbone = backbone
                self.fpn = fpn
                self.out_features = fpn.num_levels
                self._out_feature_channels = {
                    f"p{i+2}": cfg.MODEL.FPN.OUT_CHANNELS for i in range(4)
                }
            
            def forward(self, x):
                backbone_features = self.backbone(x)
                return self.fpn(backbone_features)
        
        return VGGWithFPN(backbone, fpn)
    
    return backbone
