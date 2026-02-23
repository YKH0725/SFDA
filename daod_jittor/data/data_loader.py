"""
Data loading utilities for Jittor-based training.
"""

import jittor as jt
import jittor.dataset as dataset
import os
import json
import numpy as np
from typing import Dict, List, Optional, Tuple
from daod_jittor.jittor_utils import Instances, Boxes


class COCODataset(dataset.Dataset):
    """COCO dataset for object detection."""
    
    def __init__(self, annotation_file: str, image_dir: str, 
                 transforms=None, is_train: bool = True):
    
        super().__init__() # 调用Jittor Dataset的初始化
        
        self.annotation_file = annotation_file
        self.image_dir = image_dir
        self.transforms = transforms
        self.is_train = is_train
        
        # 加载COCO JSON文件
        with open(annotation_file, 'r') as f:
            self.coco_data = json.load(f)
        
        # Create image id to annotations mapping
        self.image_ids = [img['id'] for img in self.coco_data['images']]
        self.annotations_by_image = {}
        
        for ann in self.coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.annotations_by_image:
                self.annotations_by_image[img_id] = []
            self.annotations_by_image[img_id].append(ann)
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        """Get a single sample.
        
        Returns:
            Dict containing:
                - image: Image tensor (C, H, W)
                - instances: Instances with boxes and labels
                - image_id: Image ID
                - height: Image height
                - width: Image width
        """
        image_id = self.image_ids[idx]
        
        # Find image info
        image_info = None
        for img in self.coco_data['images']:
            if img['id'] == image_id:
                image_info = img
                break
        
        if image_info is None:
            raise ValueError(f"Image {image_id} not found")
        
        # Load image
        image_path = os.path.join(self.image_dir, image_info['file_name'])
        
        try:
            from PIL import Image
            image = Image.open(image_path).convert('RGB')
            image = np.array(image, dtype=np.float32)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return dummy image
            image = np.zeros((256, 256, 3), dtype=np.float32)
        
        # Convert to CHW format
        image = image.transpose(2, 0, 1)
        
        # Load annotations
        height, width = image_info['height'], image_info['width']
        instances = self._load_annotations(image_id, height, width)
        
        # Apply transforms
        if self.transforms is not None:
            image, instances = self.transforms(image, instances)
        
        # Convert to Jittor tensor
        image = jt.array(image, dtype=jt.float32)
        
        return {
            "image": image,
            "instances": instances,
            "image_id": image_id,
            "height": height,
            "width": width,
        }
    
    def _load_annotations(self, image_id: int, height: int, width: int) -> Instances:
        """Load annotations for an image.
        
        Args:
            image_id: Image ID
            height: Image height
            width: Image width
        
        Returns:
            Instances object with boxes and labels
        """
        instances = Instances((height, width))
        
        if image_id not in self.annotations_by_image:
            # No annotations for this image
            instances.gt_boxes = Boxes(np.zeros((0, 4), dtype=np.float32))
            instances.gt_classes = np.zeros((0,), dtype=np.int32)
            return instances
        
        boxes = []
        classes = []
        
        for ann in self.annotations_by_image[image_id]:
            if ann.get('iscrowd', False):
                continue
            
            # Get category ID
            cat_id = ann['category_id']
            
            # Get bounding box
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])
            classes.append(cat_id - 1)  # Convert to 0-indexed
        
        if len(boxes) > 0:
            instances.gt_boxes = Boxes(np.array(boxes, dtype=np.float32))
            instances.gt_classes = np.array(classes, dtype=np.int32)
        else:
            instances.gt_boxes = Boxes(np.zeros((0, 4), dtype=np.float32))
            instances.gt_classes = np.array([], dtype=np.int32)
        
        return instances


class DatasetMapper:
    """Maps dataset samples to model inputs."""
    
    def __init__(self, cfg, is_train: bool = True):
        """
        Args:
            cfg: Configuration object
            is_train: Whether mapping for training or inference
        """
        self.cfg = cfg
        self.is_train = is_train
    
    def __call__(self, dataset_dict: Dict) -> Dict:
        """Apply transforms to a sample.
        
        Args:
            dataset_dict: Sample from dataset
        
        Returns:
            Processed sample
        """
        # Image is already preprocessed
        # Just ensure it's in the right format
        
        image = dataset_dict["image"]
        
        # Ensure it's float32
        if isinstance(image, np.ndarray):
            image = image.astype(np.float32)
        
        # Normalize if needed
        # (pixel_mean and pixel_std will be applied in model)
        
        return dataset_dict


def build_detection_train_loader(cfg, mapper=None):
    """Build training data loader.
    
    Args:
        cfg: Configuration object
        mapper: Custom dataset mapper
    
    Returns:
        Data loader
    """
    if mapper is None:
        mapper = DatasetMapper(cfg, is_train=True)
    
    # Get dataset
    dataset_names = cfg.DATASETS.TRAIN
    dataset_list = []
    
    for dataset_name in dataset_names:
        # This is a placeholder - in real implementation,
        # would load COCO or other datasets
        pass
    
    # Create batch loader
    def collate_fn(batch):
        """Collate function for batching."""
        batch_dict = {}
        
        for key in batch[0].keys():
            if key == "instances":
                # Handle instances separately
                batch_dict[key] = [item[key] for item in batch]
            elif key in ["image_id", "height", "width"]:
                batch_dict[key] = [item[key] for item in batch]
            elif key == "image":
                # Stack images with padding
                images = [item[key] for item in batch]
                max_h = max(img.shape[1] for img in images)
                max_w = max(img.shape[2] for img in images)
                
                padded_images = []
                for img in images:
                    padded = jt.zeros((3, max_h, max_w), dtype=img.dtype)
                    h, w = img.shape[1], img.shape[2]
                    padded[:, :h, :w] = img
                    padded_images.append(padded)
                
                batch_dict[key] = jt.stack(padded_images, dim=0)
            else:
                batch_dict[key] = [item[key] for item in batch]
        
        return batch_dict
    
    # Return a simple list-based loader for now
    # In real implementation, use jt.dataset.DataLoader
    return None


def build_detection_test_loader(cfg, dataset_name: str):
    """Build test/evaluation data loader.
    
    Args:
        cfg: Configuration object
        dataset_name: Name of test dataset
    
    Returns:
        Data loader
    """
    # Placeholder
    return None
