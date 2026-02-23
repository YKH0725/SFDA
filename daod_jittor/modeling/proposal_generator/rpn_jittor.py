"""
Region Proposal Network (RPN) for object detection in Jittor.
"""

import jittor as jt
import jittor.nn as nn
from jittor.nn import functional as F
import numpy as np
import math
from typing import Dict, List, Tuple, Optional
from daod_jittor.jittor_utils import Boxes, Instances


class RPNHead(nn.Module):
    """RPN head that outputs objectness and bounding box regression scores."""
    
    def __init__(self, in_channels: int, num_anchors: int = 9):
        """
        Args:
            in_channels: Input channel dimension
            num_anchors: Number of anchors per location (default 3x3=9)
        """
        super().__init__()
        self.num_anchors = num_anchors
        
        # 3x3 conv for feature reduction
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        
        # Objectness classification (num_anchors * 2 classes)
        self.cls_logits = nn.Conv2d(
            in_channels, num_anchors * 2, kernel_size=1
        )
        
        # Bounding box regression (num_anchors * 4 coordinates)
        self.bbox_pred = nn.Conv2d(
            in_channels, num_anchors * 4, kernel_size=1
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with proper initialization."""
        for m in [self.conv, self.cls_logits, self.bbox_pred]:
            nn.init.normal_(m.weight, std=0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, features: jt.Var) -> Tuple[jt.Var, jt.Var]:
        """
        Args:
            features: Feature map of shape (B, C, H, W)
        
        Returns:
            cls_logits: Classification logits (B, num_anchors*2, H, W)
            bbox_pred: Bounding box predictions (B, num_anchors*4, H, W)
        """
        x = F.relu(self.conv(features))
        cls_logits = self.cls_logits(x)
        bbox_pred = self.bbox_pred(x)
        
        return cls_logits, bbox_pred


class AnchorGenerator:
    """Generate anchor boxes for RPN."""
    
    def __init__(self, scales: List[float] = None, aspect_ratios: List[float] = None):
        """
        Args:
            scales: List of anchor scales (relative to feature stride)
            aspect_ratios: List of aspect ratios
        """
        if scales is None:
            scales = [8.0, 16.0, 32.0]
        if aspect_ratios is None:
            aspect_ratios = [0.5, 1.0, 2.0]
        
        self.scales = scales
        self.aspect_ratios = aspect_ratios
        self.num_anchors = len(scales) * len(aspect_ratios)
    
    def generate_anchors(self, stride: int = 16) -> jt.Var:
        """Generate anchor templates.
        
        Args:
            stride: Feature stride
        
        Returns:
            Anchor boxes of shape (num_anchors, 4) in (x1, y1, x2, y2) format
        """
        anchors = []
        
        for scale in self.scales:
            for ratio in self.aspect_ratios:
                h = stride * scale / math.sqrt(ratio)
                w = stride * scale * math.sqrt(ratio)
                
                # Anchor box centered at (0, 0)
                x1 = -w / 2
                y1 = -h / 2
                x2 = w / 2
                y2 = h / 2
                
                anchors.append([x1, y1, x2, y2])
        
        return jt.array(anchors, dtype=jt.float32)
    
    def grid_anchors(self, feature_map_size: Tuple[int, int], 
                     stride: int = 16) -> jt.Var:
        """Generate grid of anchors for entire feature map.
        
        Args:
            feature_map_size: (H, W) of feature map
            stride: Feature stride
        
        Returns:
            Anchors of shape (H*W*num_anchors, 4)
        """
        h, w = feature_map_size
        
        # Get base anchors
        base_anchors = self.generate_anchors(stride)
        
        # Generate shifts
        shift_x = jt.arange(0, w, dtype=jt.float32) * stride
        shift_y = jt.arange(0, h, dtype=jt.float32) * stride
        
        shift_y, shift_x = jt.meshgrid(shift_y, shift_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        
        shifts = jt.stack([shift_x, shift_y, shift_x, shift_y], axis=1)
        
        # Generate all anchors
        all_anchors = base_anchors[None, :, :] + shifts[:, None, :]
        all_anchors = all_anchors.reshape(-1, 4)
        
        return all_anchors


class RPN(nn.Module):
    """Region Proposal Network for generating region proposals."""
    
    def __init__(self, cfg, input_shape: Dict[str, Dict]):
        """
        Args:
            cfg: Configuration object
            input_shape: Dictionary describing input shape from backbone
        """
        super().__init__()
        
        self.in_channels = list(input_shape.values())[0]["channels"]
        self.scales = [8.0, 16.0, 32.0]
        self.aspect_ratios = [0.5, 1.0, 2.0]
        self.nms_thresh = cfg.MODEL.RPN.NMS_THRESH if hasattr(cfg.MODEL.RPN, 'NMS_THRESH') else 0.7
        self.post_nms_topk = cfg.MODEL.RPN.POST_NMS_TOPK_TEST if hasattr(cfg.MODEL.RPN, 'POST_NMS_TOPK_TEST') else 1000
        
        # RPN head
        self.rpn_head = RPNHead(self.in_channels, num_anchors=9)
        
        # Anchor generator
        self.anchor_generator = AnchorGenerator(self.scales, self.aspect_ratios)
    
    def forward(self, features: Dict[str, jt.Var], 
                image_sizes: List[Tuple[int, int]]) -> List[Instances]:
        """Generate region proposals.
        
        Args:
            features: Feature maps from backbone
            image_sizes: List of (H, W) for each image
        
        Returns:
            List of Instances containing proposal boxes
        """
        if isinstance(features, dict):
            # Use first feature level (e.g., "res4" or "p4")
            feature_key = list(features.keys())[0]
            features = features[feature_key]
        
        batch_size, _, feat_h, feat_w = features.shape
        
        # Get RPN outputs
        cls_logits, bbox_pred = self.rpn_head(features)
        
        # Generate anchors
        anchors = self.anchor_generator.grid_anchors((feat_h, feat_w), stride=16)
        
        # Process predictions
        cls_logits = cls_logits.permute(0, 2, 3, 1).reshape(batch_size, -1, 2)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(batch_size, -1, 4)
        
        proposals = []
        for i in range(batch_size):
            # Get objectness scores (background class)
            objectness = F.softmax(cls_logits[i], dim=-1)[:, 1]
            
            # Get top-k proposals
            topk = min(self.post_nms_topk, len(objectness))
            topk_indices = jt.argsort(objectness, descending=True)[:topk]
            
            # Apply bbox regression
            deltas = bbox_pred[i][topk_indices]
            proposal_boxes = self._decode_boxes(anchors[topk_indices], deltas)
            
            # Clip to image size
            h, w = image_sizes[i]
            proposal_boxes = self._clip_boxes(proposal_boxes, (h, w))
            
            # Create instance
            proposal = Instances(image_sizes[i])
            proposal.proposal_boxes = Boxes(proposal_boxes)
            proposal.objectness_logits = objectness[topk_indices]
            
            proposals.append(proposal)
        
        return proposals
    
    def _decode_boxes(self, anchors: jt.Var, deltas: jt.Var) -> jt.Var:
        """Decode bbox predictions from deltas.
        
        Args:
            anchors: Anchor boxes (N, 4)
            deltas: Bbox deltas (N, 4) in format (dx, dy, dw, dh)
        
        Returns:
            Decoded boxes (N, 4)
        """
        x1, y1, x2, y2 = anchors.unbind(dim=1)
        
        w = x2 - x1
        h = y2 - y1
        cx = x1 + w / 2
        cy = y1 + h / 2
        
        dx, dy, dw, dh = deltas.unbind(dim=1)
        
        # Decode
        pred_cx = cx + w * dx
        pred_cy = cy + h * dy
        pred_w = w * jt.exp(dw)
        pred_h = h * jt.exp(dh)
        
        # Convert to x1, y1, x2, y2
        pred_x1 = pred_cx - pred_w / 2
        pred_y1 = pred_cy - pred_h / 2
        pred_x2 = pred_cx + pred_w / 2
        pred_y2 = pred_cy + pred_h / 2
        
        return jt.stack([pred_x1, pred_y1, pred_x2, pred_y2], dim=1)
    
    def _clip_boxes(self, boxes: jt.Var, image_size: Tuple[int, int]) -> jt.Var:
        """Clip boxes to image boundaries.
        
        Args:
            boxes: Bounding boxes (N, 4)
            image_size: (H, W)
        
        Returns:
            Clipped boxes (N, 4)
        """
        h, w = image_size
        
        x1 = jt.clamp(boxes[:, 0], min=0, max=w)
        y1 = jt.clamp(boxes[:, 1], min=0, max=h)
        x2 = jt.clamp(boxes[:, 2], min=0, max=w)
        y2 = jt.clamp(boxes[:, 3], min=0, max=h)
        
        return jt.stack([x1, y1, x2, y2], dim=1)


def build_rpn(cfg, input_shape):
    """Build RPN from config."""
    return RPN(cfg, input_shape)
