"""
Jittor DAOD - Jittor implementation of Source-Free Domain Adaptation for Object Detection.

This module provides a Jittor-based implementation of the SFOD (Simplifying Source-Free Domain Adaptation
for Object Detection) paper, which was originally implemented in PyTorch/Detectron2.

Key components:
- VGG backbone with optional FPN
- Region Proposal Network (RPN)
- ROI pooling and detection heads
- Faster R-CNN detector
- Domain adaptation modules for source-free adaptation
- Data loading and training utilities

Usage:
    python train_net_jittor.py --config-file configs/config.yaml --output-dir ./output
"""

__version__ = "0.1.0"

# Import main components
from .jittor_utils import (
    ImageList,
    Instances,
    Boxes,
    DetectionCheckpointer,
)

from .modeling.meta_arch.faster_rcnn_jittor import (
    GeneralizedRCNN,
    FasterRCNN,
    SourceFreeAdaptiveTeacherRCNN,
    build_model,
)

from .data.data_loader import (
    COCODataset,
    DatasetMapper,
)

from .train_net_jittor import SimpleTrainer

__all__ = [
    "ImageList",
    "Instances", 
    "Boxes",
    "DetectionCheckpointer",
    "GeneralizedRCNN",
    "FasterRCNN",
    "SourceFreeAdaptiveTeacherRCNN",
    "build_model",
    "COCODataset",
    "DatasetMapper",
    "SimpleTrainer",
]
