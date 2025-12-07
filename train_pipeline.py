import os
import sys
import yaml
from pathlib import Path
from loguru import logger

# Disable MLflow integration BEFORE importing ultralytics
os.environ['MLFLOW_TRACKING_URI'] = ''
os.environ['MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING'] = 'false'
os.environ['MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR'] = 'false'

from ultralytics import YOLO
from ultralytics import settings
import torch

# Disable MLflow in ultralytics settings
settings.update({'mlflow': False})

class FireDetectionTrainingPipeline:    
    def __init__(
        self,
        data_yaml: str = "Dataset/data.yaml",
        model_name: str = "yolov8s.pt",         
        project_name: str = "fire_detection_training",
        experiment_name: str = "yolov8s_fire",
        epochs: int = 100,
        batch_size: int = 16,
        img_size: int = 640,
        device: str = "auto"
    ):
        self.data_yaml = data_yaml
        self.model_name = model_name
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.epochs = epochs
        self.batch_size = batch_size
        self.img_size = img_size
        
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        elif device == "cuda" and not torch.cuda.is_available():
            logger.warning("⚠️  CUDA requested but not available. Falling back to CPU.")
            logger.warning("   Training will be slower. Consider using a smaller model (yolov8n.pt)")
            self.device = "cpu"
        else:
            self.device = device
        
        self.model = None
        self.results = None
        
        logger.info("="*70)
        logger.info("Fire Detection Training Pipeline Initialized")
        logger.info("="*70)
        logger.info(f"Model: {self.model_name}")
        logger.info(f"Dataset: {self.data_yaml}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Epochs: {self.epochs}")
        logger.info(f"Batch Size: {self.batch_size}")
        logger.info(f"Image Size: {self.img_size}")
        logger.info("="*70)
    
    def verify_dataset(self):
        logger.info("Verifying dataset...")
        
        if not os.path.exists(self.data_yaml):
            raise FileNotFoundError(f"Dataset YAML not found: {self.data_yaml}")
        
        with open(self.data_yaml, 'r') as f:
            data_config = yaml.safe_load(f)
        
        logger.info(f"Dataset configuration loaded:")
        logger.info(f"  Classes: {data_config.get('nc', 'N/A')}")
        logger.info(f"  Names: {data_config.get('names', 'N/A')}")
        
        base_path = Path(self.data_yaml).parent
        for split in ['train', 'val', 'test']:
            if split in data_config:
                split_path = base_path / data_config[split]
                if split_path.exists():
                    images = list(split_path.glob('*.jpg')) + list(split_path.glob('*.png'))
                    logger.info(f"  {split.capitalize()}: {len(images)} images found")
                else:
                    logger.warning(f"  {split.capitalize()} path not found: {split_path}")
        
        logger.info("✓ Dataset verification complete")
        return data_config
    
    def load_pretrained_model(self):
        logger.info(f"Loading pretrained model: {self.model_name}")
        
        try:
            try:
                from ultralytics.nn import tasks
                original_torch_safe_load = tasks.torch_safe_load
                
                def patched_torch_safe_load(file, *args, **kwargs):
                    import torch
                    try:
                        return torch.load(file, map_location='cpu', weights_only=False), file
                    except Exception:
                        return original_torch_safe_load(file, *args, **kwargs)
                
                tasks.torch_safe_load = patched_torch_safe_load
            except Exception as patch_error:
                logger.warning(f"Could not patch torch_safe_load (non-critical): {patch_error}")
            
            self.model = YOLO(self.model_name)
            logger.info(f"✓ Model loaded successfully: {self.model_name}")
            
            logger.info(f"Model parameters: {sum(p.numel() for p in self.model.model.parameters()):,}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def train(self):
        logger.info("="*70)
        logger.info("Starting Training")
        logger.info("="*70)
        
        try:
            train_args = {
                'data': self.data_yaml,
                'epochs': self.epochs,
                'batch': self.batch_size,
                'imgsz': self.img_size,
                'device': self.device,
                'project': self.project_name,
                'name': self.experiment_name,
                'exist_ok': True,
                'pretrained': True,
                'optimizer': 'AdamW',
                'lr0': 0.001,
                'lrf': 0.01,
                'momentum': 0.937,
                'weight_decay': 0.0005,
                'warmup_epochs': 3,
                'warmup_momentum': 0.8,
                'warmup_bias_lr': 0.1,
                'box': 7.5,
                'cls': 0.5,
                'dfl': 1.5,
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
                'patience': 50,
                'save': True,
                'save_period': -1,
                'val': True,
                'plots': True,
                'verbose': True,
                'amp': True,
            }
            
            logger.info("Training configuration:")
            for key, value in train_args.items():
                logger.info(f"  {key}: {value}")
            
            self.results = self.model.train(**train_args)
            
            logger.info("="*70)
            logger.info("✓ Training Complete!")
            logger.info("="*70)
            
            return self.results
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    def validate(self):
        logger.info("="*70)
        logger.info("Running Validation")
        logger.info("="*70)
        
        try:
            val_results = self.model.val(
                data=self.data_yaml,
                batch=self.batch_size,
                imgsz=self.img_size,
                device=self.device,
                plots=True,
                verbose=True
            )
            
            logger.info("Validation Results:")
            logger.info(f"  mAP50: {val_results.box.map50:.4f}")
            logger.info(f"  mAP50-95: {val_results.box.map:.4f}")
            logger.info(f"  Precision: {val_results.box.mp:.4f}")
            logger.info(f"  Recall: {val_results.box.mr:.4f}")
            
            return val_results
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            raise
    
    def save_model(self, output_path: str = "Model/best.pt"):
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            best_model_path = f"{self.project_name}/{self.experiment_name}/weights/best.pt"
            
            if os.path.exists(best_model_path):
                import shutil
                shutil.copy(best_model_path, output_path)
                logger.info(f"✓ Best model saved to: {output_path}")
                
                # Also save last model
                last_model_path = f"{self.project_name}/{self.experiment_name}/weights/last.pt"
                if os.path.exists(last_model_path):
                    last_output = output_path.replace('best.pt', 'last.pt')
                    shutil.copy(last_model_path, last_output)
                    logger.info(f"✓ Last model saved to: {last_output}")
            else:
                logger.warning(f"Best model not found at: {best_model_path}")
                
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise
    
    def export_model(self, format: str = 'onnx'):
        try:
            logger.info(f"Exporting model to {format} format...")
            
            # Fix for PyTorch 2.6+ weights_only=True default behavior
            try:
                from ultralytics.nn import tasks
                original_torch_safe_load = tasks.torch_safe_load
                
                def patched_torch_safe_load(file, *args, **kwargs):
                    """Patched version that uses weights_only=False for PyTorch 2.6+ compatibility"""
                    try:
                        return torch.load(file, map_location='cpu', weights_only=False), file
                    except Exception:
                        return original_torch_safe_load(file, *args, **kwargs)
                
                tasks.torch_safe_load = patched_torch_safe_load
            except Exception:
                pass  # Already patched in load_pretrained_model
            
            best_model_path = f"{self.project_name}/{self.experiment_name}/weights/best.pt"
            model = YOLO(best_model_path)
            
            export_path = model.export(format=format)
            logger.info(f"✓ Model exported to: {export_path}")
            
            return export_path
            
        except Exception as e:
            logger.error(f"Export failed: {e}")
            raise
    
    def run_complete_pipeline(self):
        try:
            self.verify_dataset()
            
            self.load_pretrained_model()
            
            self.train()
            
            self.validate()
            
            self.save_model()
            
            logger.info("="*70)
            logger.info("🎉 Complete Pipeline Finished Successfully!")
            logger.info("="*70)
            logger.info(f"Results saved in: {self.project_name}/{self.experiment_name}")
            logger.info(f"Best model: Model/best.pt")
            logger.info("="*70)
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Fire Detection Training Pipeline')
    parser.add_argument('--data', type=str, default='Dataset/data.yaml',
                       help='Path to dataset YAML')
    parser.add_argument('--model', type=str, default='yolov8s.pt',
                       choices=['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt'],
                       help='YOLOv8 model variant')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--img-size', type=int, default=640,
                       help='Image size')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device (auto/cpu/cuda)')
    parser.add_argument('--project', type=str, default='fire_detection_training',
                       help='Project name')
    parser.add_argument('--name', type=str, default='yolov8s_fire',
                       help='Experiment name')
    
    args = parser.parse_args()
    
    pipeline = FireDetectionTrainingPipeline(
        data_yaml=args.data,
        model_name=args.model,
        project_name=args.project,
        experiment_name=args.name,
        epochs=args.epochs,
        batch_size=args.batch,
        img_size=args.img_size,
        device=args.device
    )
    
    pipeline.run_complete_pipeline()


if __name__ == "__main__":
    main()
