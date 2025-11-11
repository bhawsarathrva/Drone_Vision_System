"""
Training script for YOLOv8 fire detection model
"""
import os
import yaml
from pathlib import Path
from ultralytics import YOLO
import torch
from loguru import logger

class FireDetectionTrainer:
    def __init__(self, config_path: str = "configs/training_config.yaml"):
        """
        Initialize trainer with configuration
        
        Args:
            config_path: Path to training configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.model = None
        self.results = None
        
    def _load_config(self) -> dict:
        """Load training configuration"""
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        else:
            # Default configuration
            return {
                'model': 'yolov8n.pt',
                'data': 'Dataset/data.yaml',
                'epochs': 100,
                'imgsz': 640,
                'batch': 16,
                'lr0': 0.01,
                'lrf': 0.01,
                'momentum': 0.937,
                'weight_decay': 0.0005,
                'warmup_epochs': 3,
                'warmup_momentum': 0.8,
                'warmup_bias_lr': 0.1,
                'box': 7.5,
                'cls': 0.5,
                'dfl': 1.5,
                'pose': 12.0,
                'kobj': 1.0,
                'label_smoothing': 0.0,
                'nbs': 64,
                'hsv_h': 0.015,
                'hsv_s': 0.7,
                'hsv_v': 0.4,
                'degrees': 0.0,
                'translate': 0.1,
                'scale': 0.5,
                'shear': 0.0,
                'perspective': 0.0,
                'flipud': 0.0,
                'fliplr': 0.5,
                'mosaic': 1.0,
                'mixup': 0.0,
                'copy_paste': 0.0,
                'project': 'runs/detect',
                'name': 'fire_detection',
                'exist_ok': False,
                'pretrained': True,
                'optimizer': 'SGD',
                'verbose': True,
                'seed': 0,
                'deterministic': True,
                'single_cls': False,
                'rect': False,
                'cos_lr': False,
                'close_mosaic': 10,
                'resume': False,
                'amp': True,
                'fraction': 1.0,
                'profile': False,
                'freeze': None,
                'lr0': 0.01,
                'lrf': 0.01,
                'momentum': 0.937,
                'weight_decay': 0.0005,
                'warmup_epochs': 3.0,
                'warmup_momentum': 0.8,
                'warmup_bias_lr': 0.1,
                'box': 7.5,
                'cls': 0.5,
                'dfl': 1.5,
                'pose': 12.0,
                'kobj': 1.0,
                'label_smoothing': 0.0,
                'nbs': 64,
                'overlap_mask': True,
                'mask_ratio': 4,
                'dropout': 0.0,
                'val': True
            }
    
    def load_model(self, model_path: str = None):
        """
        Load YOLOv8 model
        
        Args:
            model_path: Path to model weights (uses config if None)
        """
        model_path = model_path or self.config.get('model', 'yolov8n.pt')
        logger.info(f"Loading model: {model_path}")
        
        # Check if model exists locally, otherwise use pretrained
        if not os.path.exists(model_path):
            logger.info(f"Model not found locally, using pretrained: {model_path}")
        
        self.model = YOLO(model_path)
        logger.info("Model loaded successfully")
    
    def train(self):
        """Train the model"""
        if self.model is None:
            self.load_model()
        
        logger.info("Starting training...")
        logger.info(f"Configuration: {self.config}")
        
        # Prepare data path
        data_path = self.config.get('data', 'Dataset/data.yaml')
        if not os.path.exists(data_path):
            logger.error(f"Data file not found: {data_path}")
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        # Training parameters
        train_params = {
            'data': data_path,
            'epochs': self.config.get('epochs', 100),
            'imgsz': self.config.get('imgsz', 640),
            'batch': self.config.get('batch', 16),
            'lr0': self.config.get('lr0', 0.01),
            'lrf': self.config.get('lrf', 0.01),
            'momentum': self.config.get('momentum', 0.937),
            'weight_decay': self.config.get('weight_decay', 0.0005),
            'warmup_epochs': self.config.get('warmup_epochs', 3),
            'warmup_momentum': self.config.get('warmup_momentum', 0.8),
            'warmup_bias_lr': self.config.get('warmup_bias_lr', 0.1),
            'box': self.config.get('box', 7.5),
            'cls': self.config.get('cls', 0.5),
            'dfl': self.config.get('dfl', 1.5),
            'pose': self.config.get('pose', 12.0),
            'kobj': self.config.get('kobj', 1.0),
            'label_smoothing': self.config.get('label_smoothing', 0.0),
            'nbs': self.config.get('nbs', 64),
            'hsv_h': self.config.get('hsv_h', 0.015),
            'hsv_s': self.config.get('hsv_s', 0.7),
            'hsv_v': self.config.get('hsv_v', 0.4),
            'degrees': self.config.get('degrees', 0.0),
            'translate': self.config.get('translate', 0.1),
            'scale': self.config.get('scale', 0.5),
            'shear': self.config.get('shear', 0.0),
            'perspective': self.config.get('perspective', 0.0),
            'flipud': self.config.get('flipud', 0.0),
            'fliplr': self.config.get('fliplr', 0.5),
            'mosaic': self.config.get('mosaic', 1.0),
            'mixup': self.config.get('mixup', 0.0),
            'copy_paste': self.config.get('copy_paste', 0.0),
            'project': self.config.get('project', 'runs/detect'),
            'name': self.config.get('name', 'fire_detection'),
            'exist_ok': self.config.get('exist_ok', False),
            'pretrained': self.config.get('pretrained', True),
            'optimizer': self.config.get('optimizer', 'SGD'),
            'verbose': self.config.get('verbose', True),
            'seed': self.config.get('seed', 0),
            'deterministic': self.config.get('deterministic', True),
            'single_cls': self.config.get('single_cls', False),
            'rect': self.config.get('rect', False),
            'cos_lr': self.config.get('cos_lr', False),
            'close_mosaic': self.config.get('close_mosaic', 10),
            'resume': self.config.get('resume', False),
            'amp': self.config.get('amp', True),
            'fraction': self.config.get('fraction', 1.0),
            'profile': self.config.get('profile', False),
            'freeze': self.config.get('freeze', None),
            'val': self.config.get('val', True)
        }
        
        # Remove None values
        train_params = {k: v for k, v in train_params.items() if v is not None}
        
        # Train the model
        try:
            self.results = self.model.train(**train_params)
            logger.info("Training completed successfully")
            return self.results
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    def save_model(self, path: str = "Model/best.pt"):
        """Save trained model"""
        if self.model is None:
            logger.error("No model to save")
            return
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.export(format='pt')  # Export as PyTorch
        logger.info(f"Model saved to: {path}")
    
    def validate(self, model_path: str = None):
        """Validate the model"""
        if model_path:
            self.load_model(model_path)
        elif self.model is None:
            self.load_model()
        
        data_path = self.config.get('data', 'Dataset/data.yaml')
        results = self.model.val(data=data_path)
        return results

def main():
    """Main training function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train fire detection model')
    parser.add_argument('--config', type=str, default='configs/training_config.yaml',
                       help='Path to training config file')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to model weights')
    parser.add_argument('--data', type=str, default=None,
                       help='Path to data.yaml file')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of epochs')
    parser.add_argument('--batch', type=int, default=None,
                       help='Batch size')
    parser.add_argument('--imgsz', type=int, default=None,
                       help='Image size')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = FireDetectionTrainer(config_path=args.config)
    
    # Override config with command line arguments
    if args.model:
        trainer.config['model'] = args.model
    if args.data:
        trainer.config['data'] = args.data
    if args.epochs:
        trainer.config['epochs'] = args.epochs
    if args.batch:
        trainer.config['batch'] = args.batch
    if args.imgsz:
        trainer.config['imgsz'] = args.imgsz
    
    # Train
    try:
        trainer.train()
        logger.info("Training completed successfully")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main()

