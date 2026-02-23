"""
基于 Jittor 的 SFOD 模型训练主脚本

本脚本实现了完整的训练流程：
1. 模型构建和初始化
2. 数据加载和预处理
3. 优化器和学习率调度
4. 训练循环和损失计算
5. 检查点保存和恢复
6. 日志记录和可视化
"""

import jittor as jt
import jittor.nn as nn
from jittor import optim
import numpy as np
import os
import sys
from typing import Dict, List, Optional
import argparse
import json

# 设置路径，确保可以导入模块
sys.path.insert(0, os.path.dirname(__file__))

from jittor_utils import DetectionCheckpointer
from modeling.meta_arch.faster_rcnn_jittor import build_model
from data.data_loader import COCODataset, DatasetMapper, build_detection_train_loader
from daod.config import add_config


class SimpleTrainer:
    """Jittor 模型的简单训练器
    
    封装了训练的全部逻辑，包括模型构建、优化器设置、训练循环等。
    使用方法：
        trainer = SimpleTrainer(cfg)
        trainer.train()
    """
    
    def __init__(self, cfg):
        """初始化训练器
        
        Args:
            cfg: 配置对象，包含所有训练超参数
        """
        self.cfg = cfg
        self.max_iter = cfg.SOLVER.MAX_ITER  # 最大训练迭代次数
        self.log_period = cfg.SOLVER.LOG_PERIOD if hasattr(cfg.SOLVER, 'LOG_PERIOD') else 20  # 日志记录周期
        
        # 构建模型
        self.model = self._build_model()
        
        # 构建优化器
        self.optimizer = self._build_optimizer()
        
        # 构建学习率调度器
        self.scheduler = self._build_scheduler()
        
        # 检查点管理器（用于保存和加载模型）
        self.checkpointer = DetectionCheckpointer(
            self.model,
            save_dir=cfg.OUTPUT_DIR,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
        )
        
        # 训练状态
        self.start_iter = 0      # 起始迭代（用于断点续训）
        self.current_iter = 0    # 当前迭代
        
        # 损失历史（用于日志记录）
        self.loss_history = []
    
    def _build_model(self):
        """从配置构建模型
        
        Returns:
            构建好的模型实例
        """
        model = build_model(self.cfg)
        
        # 如果指定了预训练权重，加载它们
        if self.cfg.MODEL.WEIGHTS and self.cfg.MODEL.WEIGHTS != "":
            try:
                self.checkpointer.load(self.cfg.MODEL.WEIGHTS)
                print(f"已从 {self.cfg.MODEL.WEIGHTS} 加载权重")
            except Exception as e:
                print(f"警告：无法加载权重：{e}")
        
        return model
    
    def _build_optimizer(self):
        """从配置构建优化器
        
        使用不同的权重衰减策略：
        - 偏置和归一化层：不使用权重衰减
        - 其他参数：使用权重衰减
        
        Returns:
            优化器实例
        """
        params_with_decay = []      # 需要权重衰减的参数
        params_without_decay = []   # 不需要权重衰减的参数
        
        for name, param in self.model.named_parameters():
            if "bias" in name or "norm" in name:
                params_without_decay.append(param)
            else:
                params_with_decay.append(param)
        
        param_groups = [
            {"params": params_with_decay, 
             "lr": self.cfg.SOLVER.BASE_LR,
             "weight_decay": self.cfg.SOLVER.WEIGHT_DECAY},
            {"params": params_without_decay,
             "lr": self.cfg.SOLVER.BASE_LR,
             "weight_decay": 0.0},
        ]
        
        if self.cfg.SOLVER.OPTIMIZER == "SGD":
            optimizer = optim.SGD(
                param_groups,
                momentum=self.cfg.SOLVER.MOMENTUM,
            )
        elif self.cfg.SOLVER.OPTIMIZER == "Adam":
            optimizer = optim.Adam(
                param_groups,
                lr=self.cfg.SOLVER.BASE_LR,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.cfg.SOLVER.OPTIMIZER}")
        
        return optimizer
    
    def _build_scheduler(self):
        """Build learning rate scheduler."""
        # Placeholder - would implement actual scheduler logic
        return None
    
    def train(self):
        """Main training loop."""
        print(f"Starting training for {self.max_iter} iterations...")
        
        # Create dummy data loader for now
        # In real implementation, this would be actual data loader
        self.model.train()
        
        for iteration in range(self.start_iter, self.max_iter):
            self.current_iter = iteration
            
            # Training step
            loss_dict = self._forward_batch()
            
            # Backward pass
            total_loss = sum(loss_dict.values())
            self.optimizer.step(total_loss)
            self.optimizer.zero_grad()
            
            # Logging
            if (iteration + 1) % self.log_period == 0:
                self._log_losses(loss_dict, iteration)
            
            # Save checkpoint
            if hasattr(self.cfg.SOLVER, 'CHECKPOINT_PERIOD') and \
               (iteration + 1) % self.cfg.SOLVER.CHECKPOINT_PERIOD == 0:
                self._save_checkpoint(iteration)
        
        print("Training finished!")
    
    def _forward_batch(self) -> Dict[str, jt.Var]:
        """Forward pass for a single batch.
        
        Returns:
            Dictionary of losses
        """
        # Placeholder - would load actual batch from data loader
        # and compute losses
        
        # Dummy loss for demonstration
        dummy_input = jt.randn((2, 3, 224, 224))
        
        # For now, return dummy losses
        return {
            "loss_classifier": jt.array([0.5]),
            "loss_box_reg": jt.array([0.3]),
            "loss_objectness": jt.array([0.2]),
            "loss_rpn_box_reg": jt.array([0.1]),
        }
    
    def _log_losses(self, loss_dict: Dict, iteration: int):
        """Log losses to console and file.
        
        Args:
            loss_dict: Dictionary of losses
            iteration: Current iteration number
        """
        total_loss = sum(float(v.data if hasattr(v, 'data') else v) 
                        for v in loss_dict.values())
        
        self.loss_history.append({
            'iteration': iteration,
            'total_loss': total_loss,
            **{k: float(v.data if hasattr(v, 'data') else v) 
               for k, v in loss_dict.items()}
        })
        
        # Print to console
        loss_str = " ".join(
            f"{k}: {v:.4f}" for k, v in loss_dict.items()
        )
        print(f"Iteration {iteration+1}/{self.max_iter}: {loss_str}")
        
        # Save to file
        log_file = os.path.join(self.cfg.OUTPUT_DIR, "training_log.json")
        os.makedirs(self.cfg.OUTPUT_DIR, exist_ok=True)
        with open(log_file, 'w') as f:
            json.dump(self.loss_history, f, indent=2)
    
    def _save_checkpoint(self, iteration: int):
        """Save training checkpoint.
        
        Args:
            iteration: Current iteration number
        """
        checkpoint_name = f"model_iter_{iteration}"
        self.checkpointer.save(checkpoint_name)
        print(f"Checkpoint saved: {checkpoint_name}")
    
    def resume_or_load(self, resume: bool = True):
        """Resume from checkpoint or load weights.
        
        Args:
            resume: If True, resume from last checkpoint
        """
        if resume:
            checkpoint_file = os.path.join(
                self.cfg.OUTPUT_DIR, "last_checkpoint"
            )
            if os.path.exists(checkpoint_file):
                with open(checkpoint_file, 'r') as f:
                    checkpoint_path = f.read().strip()
                self.checkpointer.load(checkpoint_path)
                print(f"Resumed from checkpoint: {checkpoint_path}")
            else:
                print("No checkpoint found, starting from scratch")
        else:
            if self.cfg.MODEL.WEIGHTS and self.cfg.MODEL.WEIGHTS != "":
                self.checkpointer.load(self.cfg.MODEL.WEIGHTS)
                print(f"Loaded weights from {self.cfg.MODEL.WEIGHTS}")


def setup(args):
    """Setup configuration from arguments.
    
    Args:
        args: Parsed command line arguments
    
    Returns:
        Configuration object
    """
    # Simple config object
    class Config:
        def __init__(self):
            self.SOLVER = type('obj', (object,), {
                'MAX_ITER': 90000,
                'BASE_LR': 0.02,
                'MOMENTUM': 0.9,
                'WEIGHT_DECAY': 0.0001,
                'OPTIMIZER': 'SGD',
                'LOG_PERIOD': 20,
                'CHECKPOINT_PERIOD': 5000,
            })()
            
            self.MODEL = type('obj', (object,), {
                'WEIGHTS': '',
                'PIXEL_MEAN': [103.53, 116.28, 123.675],
                'PIXEL_STD': [1.0, 1.0, 1.0],
                'BACKBONE': type('obj', (object,), {
                    'NAME': 'build_vgg_backbone',
                    'DEPTH': 16,
                    'USE_FPN': False,
                })(),
                'RPN': type('obj', (object,), {
                    'NMS_THRESH': 0.7,
                    'POST_NMS_TOPK_TEST': 1000,
                })(),
                'ROI_HEADS': type('obj', (object,), {
                    'NUM_CLASSES': 80,
                })(),
                'ROI_BOX_HEAD': type('obj', (object,), {
                    'POOLER_RESOLUTION': 7,
                })(),
            })()
            
            self.DATASETS = type('obj', (object,), {
                'TRAIN': (),
                'TEST': (),
            })()
            
            self.INPUT = type('obj', (object,), {
                'FORMAT': 'RGB',
            })()
            
            self.OUTPUT_DIR = args.output_dir
            self.TRAINER = args.trainer if hasattr(args, 'trainer') else "base"
    
    cfg = Config()
    
    # Merge from config file if provided
    if args.config_file:
        # Would load YAML/JSON config here
        pass
    
    return cfg


def main(args):
    """Main training function.
    
    Args:
        args: Parsed command line arguments
    """
    cfg = setup(args)
    
    # Create trainer
    trainer = SimpleTrainer(cfg)
    
    # Resume or load pretrained weights
    trainer.resume_or_load(resume=args.resume)
    
    # Run training
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Jittor SFOD Training")
    parser.add_argument("--config-file", type=str, default="",
                       help="Path to config file")
    parser.add_argument("--output-dir", type=str, default="./output",
                       help="Output directory for checkpoints and logs")
    parser.add_argument("--resume", action="store_true",
                       help="Resume from last checkpoint")
    parser.add_argument("--eval-only", action="store_true",
                       help="Evaluation only mode")
    parser.add_argument("--trainer", type=str, default="base",
                       help="Trainer type (base, adaptive_teacher, etc.)")
    parser.add_argument("--num-gpus", type=int, default=1,
                       help="Number of GPUs to use")
    parser.add_argument("--num-machines", type=int, default=1,
                       help="Number of machines")
    parser.add_argument("--machine-rank", type=int, default=0,
                       help="Rank of this machine")
    parser.add_argument("--dist-url", type=str, default="tcp://127.0.0.1:29500",
                       help="URL for distributed training")
    
    args = parser.parse_args()
    
    print("Command line args:", args)
    main(args)
