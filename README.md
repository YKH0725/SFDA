# Jittor 代码复现 SFOD 尝试

本文档描述了将原始的PyTorch/Detectron2实现迁移到Jittor框架。

## 概述

该项目已创建 `daod_jittor/` 目录作为Jittor实现的主目录。原始PyTorch代码保持不变，新的Jittor实现与其并行存在。

## 项目结构

```
simple-SFOD/
├── daod/                      # 原始PyTorch/Detectron2实现
│   ├── modeling/
│   ├── engine/
│   ├── data/
│   └── ...
├── daod_jittor/               # 新的Jittor实现 
│   ├── jittor_utils.py        # Jittor工具函数和兼容性层
│   ├── __init__.py            # 模块初始化
│   ├── train_net_jittor.py    # 主训练脚本
│   ├── modeling/
│   │   ├── __init__.py
│   │   ├── meta_arch/
│   │   │   ├── vgg_jittor.py         # VGG骨干网络
│   │   │   └── faster_rcnn_jittor.py # Faster R-CNN模型
│   │   ├── proposal_generator/
│   │   │   └── rpn_jittor.py         # RPN实现
│   │   └── roi_heads/
│   │       └── roi_heads_jittor.py   # ROI头部
│   └── data/
│       ├── __init__.py
│       └── data_loader.py    # 数据加载和处理
└── README_JITTOR.md           # 本指南
```

## 已实现的关键组件

### 1. 工具模块 (`daod_jittor/jittor_utils.py`)
- **ImageList**: 图像容器类
- **Instances**: 实例容器类（边界框、类标签等）
- **Boxes**: 边界框容器类
- **DetectionCheckpointer**: 检查点管理器
- **兼容性函数**: 与PyTorch API兼容的辅助函数

### 2. 骨干网络 (`daod_jittor/modeling/meta_arch/vgg_jittor.py`)
- **VGGBackbone**: 完整的VGG16/VGG19实现
  - 支持可选的批归一化
  - 多尺度特征提取 (res2, res3, res4, res5)
  - 权重初始化
- **FPN**: 特征金字塔网络
- **LastLevelMaxPool/LastLevelP6P7**: 附加特征级

### 3. RPN实现 (`daod_jittor/modeling/proposal_generator/rpn_jittor.py`)
- **RPNHead**: RPN头部（分类和回归）
- **AnchorGenerator**: 锚点生成器
- **RPN**: 完整的区域提议网络
  - 锚点生成
  - 提议解码
  - NMS处理

### 4. ROI头部 (`daod_jittor/modeling/roi_heads/roi_heads_jittor.py`)
- **ROIPooler**: ROI池化层
- **ROIBoxHead**: 分类和回归头
- **StandardROIHeads**: 标准ROI头部
- **FastRCNNOutputLayers**: Fast R-CNN输出层

### 5. 模型架构 (`daod_jittor/modeling/meta_arch/faster_rcnn_jittor.py`)
- **GeneralizedRCNN**: 基础R-CNN框架
- **FasterRCNN**: 完整的Faster R-CNN模型
- **SourceFreeAdaptiveTeacherRCNN**: 源无域适应模型（部分实现）

### 6. 数据加载 (`daod_jittor/data/data_loader.py`)
- **COCODataset**: COCO数据集加载器
- **DatasetMapper**: 数据预处理和转换
- **collate_fn**: 批处理函数

### 7. 训练脚本 (`daod_jittor/train_net_jittor.py`)
- **SimpleTrainer**: 基础训练器类
  - 模型构建
  - 优化器配置
  - 训练循环
  - 检查点管理
  - 日志记录

## 关键转换规则

### 导入转换

| PyTorch | Jittor |
|---------|--------|
| `import torch` | `import jittor as jt` |
| `import torch.nn as nn` | `import jittor.nn as nn` |
| `from torch.nn import functional as F` | `from jittor.nn import functional as F` |
| `torch.tensor()` | `jt.array()` |
| `torch.zeros()` | `jt.zeros()` |
| `torch.ones()` | `jt.ones()` |
| `torch.randn()` | `jt.randn()` |
| `torch.stack()` | `jt.stack()` |
| `torch.cat()` / `torch.concat()` | `jt.concat()` |

### 常见操作转换

```python
# 张量创建
torch.tensor([1, 2, 3])        → jt.array([1, 2, 3])
torch.zeros((3, 4))            → jt.zeros((3, 4))
torch.ones((3, 4))             → jt.ones((3, 4))

# 张量操作
x.shape[0]                      → x.shape[0]  (相同)
x.reshape(-1)                   → x.reshape(-1)  (相同)
x.permute(0, 2, 1)             → x.permute(0, 2, 1)  (相同)
x.to(device)                    → 在Jittor中不需要

# 激活函数
F.relu(x)                       → F.relu(x)  (相同)
F.softmax(x, dim=1)            → F.softmax(x, dim=1)  (相同)
F.max_pool2d(x, 2)             → F.max_pool2d(x, 2)  (相同)

# 反向传播
loss.backward()                 → optimizer.step(loss)
optimizer.step()                → optimizer.zero_grad()
optimizer.zero_grad()           → (包含在step()中)
```

## 使用方法

### 基础训练

```bash
# 使用默认配置
python daod_jittor/train_net_jittor.py

# 使用自定义输出目录
python daod_jittor/train_net_jittor.py --output-dir ./my_output

# 从检查点恢复
python daod_jittor/train_net_jittor.py --resume

# 评估模式
python daod_jittor/train_net_jittor.py --eval-only
```

### Python代码中使用

```python
from daod_jittor import FasterRCNN, build_model
from daod_jittor import SimpleTrainer

# 构建模型
model = build_model(cfg)

# 或直接创建
from daod_jittor.modeling.meta_arch.faster_rcnn_jittor import FasterRCNN
from daod_jittor.modeling.meta_arch.vgg_jittor import build_vgg_backbone
from daod_jittor.modeling.proposal_generator.rpn_jittor import RPN
from daod_jittor.modeling.roi_heads.roi_heads_jittor import StandardROIHeads

backbone = build_vgg_backbone(cfg)
rpn = RPN(cfg, backbone.output_shape)
roi_heads = StandardROIHeads(cfg, backbone.output_shape)

model = FasterRCNN(backbone, rpn, roi_heads)

# 使用训练器
trainer = SimpleTrainer(cfg)
trainer.train()
```

## 仍需实现的组件

### 1. 高级功能
- [ ] 多GPU分布式训练 (DistributedDataParallel)
- [ ] 混合精度训练 (AMP)
- [ ] 学习率调度器
- [ ] 梯度累积

### 2. 域适应模块
- [ ] 完整的SourceFreeAdaptiveTeacherRCNN实现
- [ ] 域判别器（Gradient Reversal Layer）
- [ ] 域适应损失函数
- [ ] 一致性正则化

### 3. 优化和性能
- [ ] 模型编译优化
- [ ] 内存优化
- [ ] 推理加速


## 参考资源

- Jittor文档: https://cg.cs.tsinghua.edu.cn/jittor/
- 原始SFOD论文: https://arxiv.org/abs/2407.07586
- Detectron2文档: https://detectron2.readthedocs.io/


