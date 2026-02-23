"""
ROI Pooling and ROI Heads for object detection in Jittor.
"""

import jittor as jt
import jittor.nn as nn
from jittor.nn import functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from daod_jittor.jittor_utils import Boxes, Instances


class ROIPooler(nn.Module):
    """ROI pooling layer that extracts fixed-size features from regions."""
    
    def __init__(self, output_size: int = 7, spatial_scale: float = 1.0):
        """
        Args:
            output_size: Output feature map size (output_size x output_size)
            spatial_scale: Scaling factor from image to feature map
        """
        super().__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale
    
    def forward(self, features: jt.Var, boxes_list: List[Boxes]) -> jt.Var:
        """Extract ROI features.
        
        Args:
            features: Feature map (B, C, H, W)
            boxes_list: List of Boxes objects for each image in batch
        
        Returns:
            ROI features (total_rois, C, output_size, output_size)
        """
        batch_size, channels, height, width = features.shape
        
        # Concatenate boxes from all images
        all_boxes = []
        all_batch_indices = []
        
        for batch_idx, boxes in enumerate(boxes_list):
            all_boxes.append(boxes.tensor)
            all_batch_indices.append(
                jt.full((len(boxes),), batch_idx, dtype=jt.int32)
            )
        
        all_boxes = jt.concat(all_boxes, dim=0)
        all_batch_indices = jt.concat(all_batch_indices, dim=0)
        
        # Scale boxes to feature map coordinates
        scaled_boxes = all_boxes * self.spatial_scale
        
        # Use adaptive average pooling for each ROI
        roi_features = []
        
        for i in range(len(scaled_boxes)):
            batch_idx = all_batch_indices[i]
            x1, y1, x2, y2 = scaled_boxes[i].tolist()
            
            # Ensure valid region
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(width, int(x2)), min(height, int(y2))
            
            if x2 > x1 and y2 > y1:
                # Extract region
                region = features[batch_idx:batch_idx+1, :, y1:y2, x1:x2]
                
                # Adaptive average pooling to output size
                roi_feature = F.adaptive_avg_pool2d(region, self.output_size)
                roi_features.append(roi_feature)
            else:
                # Invalid region, use zero features
                roi_feature = jt.zeros(
                    (1, channels, self.output_size, self.output_size),
                    dtype=features.dtype
                )
                roi_features.append(roi_feature)
        
        # Concatenate all ROI features
        return jt.concat(roi_features, dim=0)


class ROIBoxHead(nn.Module):
    """Head for predicting class scores and bbox regression."""
    
    def __init__(self, in_channels: int, num_classes: int, pooler_resolution: int = 7):
        """
        Args:
            in_channels: Input channel dimension
            num_classes: Number of classes
            pooler_resolution: ROI pool output size
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.pooler_resolution = pooler_resolution
        
        # FC layers for classification and regression
        fc_dim = in_channels * pooler_resolution * pooler_resolution
        
        self.fc1 = nn.Linear(fc_dim, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        
        self.cls_score = nn.Linear(1024, num_classes)
        self.bbox_pred = nn.Linear(1024, num_classes * 4)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for m in [self.fc1, self.fc2, self.cls_score, self.bbox_pred]:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, features: jt.Var) -> Tuple[jt.Var, jt.Var]:
        """
        Args:
            features: ROI pooled features (num_rois, C, H, W)
        
        Returns:
            cls_scores: Classification scores (num_rois, num_classes)
            bbox_preds: Bounding box predictions (num_rois, num_classes*4)
        """
        # Flatten
        x = features.reshape(features.shape[0], -1)
        
        # FC layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        # Predictions
        cls_scores = self.cls_score(x)
        bbox_preds = self.bbox_pred(x)
        
        return cls_scores, bbox_preds


class StandardROIHeads(nn.Module):
    """Standard ROI heads for object detection."""
    
    def __init__(self, cfg, input_shape: Dict[str, Dict]):
        """
        Args:
            cfg: Configuration object
            input_shape: Dictionary describing input shape from backbone
        """
        super().__init__()
        
        self.num_classes = cfg.MODEL.ROI_HEADS.NUM_CLASSES if hasattr(cfg.MODEL.ROI_HEADS, 'NUM_CLASSES') else 80
        
        # ROI pooler
        self.pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION if hasattr(cfg.MODEL.ROI_BOX_HEAD, 'POOLER_RESOLUTION') else 7
        self.box_pooler = ROIPooler(output_size=self.pooler_resolution, spatial_scale=1.0/16)
        
        # Get input channels
        self.in_channels = 512  # Assuming VGG16 with last layer having 512 channels
        
        # Box head
        self.box_head = ROIBoxHead(self.in_channels, self.num_classes, self.pooler_resolution)
        
        self.box_predictor = self.box_head
    
    def forward(self, features: Dict[str, jt.Var], 
                proposals: List[Instances]) -> Dict[str, jt.Var]:
        """Forward pass for ROI heads.
        
        Args:
            features: Feature maps from backbone
            proposals: List of Instances containing proposal boxes
        
        Returns:
            Dictionary of predictions
        """
        # Extract ROI features
        if isinstance(features, dict):
            # Use first feature level
            feature_key = list(features.keys())[0]
            features = features[feature_key]
        
        boxes_list = [prop.proposal_boxes for prop in proposals]
        roi_features = self.box_pooler(features, boxes_list)
        
        # Predict class scores and bbox regression
        cls_scores, bbox_preds = self.box_head(roi_features)
        
        return {
            "cls_scores": cls_scores,
            "bbox_preds": bbox_preds,
        }


def assign_boxes_to_levels(boxes: List[Boxes], 
                           lvl_min: int = 2,
                           lvl_max: int = 5,
                           canonical_scale: int = 224,
                           canonical_level: int = 4) -> List[int]:
    """Assign boxes to FPN levels based on their sizes.
    
    Args:
        boxes: List of Boxes objects
        lvl_min: Minimum level
        lvl_max: Maximum level
        canonical_scale: Scale at canonical level
        canonical_level: Canonical level
    
    Returns:
        List of level assignments for each box
    """
    levels = []
    for box_list in boxes:
        areas = box_list.area()
        level = jt.log2(jt.sqrt(areas) / canonical_scale + 1e-6)
        level = jt.clamp(level + canonical_level, min=lvl_min, max=lvl_max)
        levels.append(level.int())
    
    return levels


class FastRCNNOutputLayers(nn.Module):
    """Output layers for Fast R-CNN head."""
    
    def __init__(self, in_channels: int, num_classes: int):
        """
        Args:
            in_channels: Input channel dimension
            num_classes: Number of classes
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)
        
        # Initialize weights
        for m in [self.cls_score, self.bbox_pred]:
            nn.init.normal_(m.weight, std=0.01)
            nn.init.constant_(m.bias, 0)
    
    def forward(self, x: jt.Var) -> Tuple[jt.Var, jt.Var]:
        """
        Args:
            x: Input features (num_boxes, in_channels)
        
        Returns:
            cls_logits: Classification logits (num_boxes, num_classes)
            bbox_pred: Bounding box predictions (num_boxes, num_classes*4)
        """
        cls_logits = self.cls_score(x)
        bbox_pred = self.bbox_pred(x)
        
        return cls_logits, bbox_pred
    
    def losses(self, predictions: Dict[str, jt.Var], 
               instances: List[Instances]) -> Dict[str, jt.Var]:
        """Compute losses.
        
        Args:
            predictions: Dictionary of predictions from forward pass
            instances: Ground truth instances
        
        Returns:
            Dictionary of losses
        """
        # This is a placeholder implementation
        # Actual loss computation would go here
        
        cls_logits = predictions["cls_logits"]
        bbox_pred = predictions["bbox_pred"]
        
        # Dummy loss for now
        cls_loss = F.cross_entropy(cls_logits, jt.zeros(len(cls_logits), dtype=jt.int32))
        bbox_loss = F.smooth_l1_loss(bbox_pred, jt.zeros_like(bbox_pred))
        
        return {
            "loss_classifier": cls_loss,
            "loss_box_reg": bbox_loss,
        }
    
    def inference(self, predictions: Dict[str, jt.Var],
                  instances: List[Instances]) -> Tuple[List[Instances], Dict]:
        """Inference with predictions.
        
        Args:
            predictions: Dictionary of predictions
            instances: Input instances with proposal boxes
        
        Returns:
            Tuple of (predicted instances, additional info)
        """
        # Get top class prediction
        cls_logits = predictions["cls_logits"]
        bbox_pred = predictions["bbox_pred"]
        
        class_preds = jt.argmax(cls_logits, dim=1)
        scores = F.softmax(cls_logits, dim=1)
        
        # Decode bounding boxes
        result_instances = []
        for i, inst in enumerate(instances):
            new_inst = Instances(inst.image_size)
            new_inst.pred_boxes = inst.proposal_boxes
            new_inst.scores = scores[i]
            new_inst.pred_classes = class_preds[i]
            result_instances.append(new_inst)
        
        return result_instances, {}
