"""
Model validation script
"""
import os
import yaml
from pathlib import Path
from ultralytics import YOLO
from loguru import logger
import torch

class ModelValidator:
    def __init__(self, model_path: str, data_path: str):
        """
        Initialize model validator
        
        Args:
            model_path: Path to model weights
            data_path: Path to data.yaml file
        """
        self.model_path = model_path
        self.data_path = data_path
        self.model = None
        self.results = None
        
    def load_model(self):
        """Load model"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        self.model = YOLO(self.model_path)
        logger.info(f"Model loaded from {self.model_path}")
    
    def validate(self) -> dict:
        """Validate model"""
        if self.model is None:
            self.load_model()
        
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        logger.info("Starting validation...")
        
        # Run validation
        self.results = self.model.val(data=self.data_path)
        
        # Extract metrics
        metrics = {
            'mAP50': self.results.box.map50,
            'mAP50-95': self.results.box.map,
            'precision': self.results.box.p,
            'recall': self.results.box.r,
            'f1_score': 2 * (self.results.box.p * self.results.box.r) / (self.results.box.p + self.results.box.r) if (self.results.box.p + self.results.box.r) > 0 else 0
        }
        
        logger.info("Validation completed")
        logger.info(f"mAP50: {metrics['mAP50']:.4f}")
        logger.info(f"mAP50-95: {metrics['mAP50-95']:.4f}")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall: {metrics['recall']:.4f}")
        logger.info(f"F1 Score: {metrics['f1_score']:.4f}")
        
        return metrics
    
    def save_results(self, output_path: str):
        """Save validation results"""
        import json
        
        results_dict = {
            'model_path': self.model_path,
            'data_path': self.data_path,
            'metrics': {
                'mAP50': float(self.results.box.map50),
                'mAP50-95': float(self.results.box.map),
                'precision': float(self.results.box.p),
                'recall': float(self.results.box.r),
                'f1_score': float(2 * (self.results.box.p * self.results.box.r) / (self.results.box.p + self.results.box.r) if (self.results.box.p + self.results.box.r) > 0 else 0)
            }
        }
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")

def main():
    """Main validation function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate fire detection model')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to model weights')
    parser.add_argument('--data', type=str, default='Dataset/data.yaml',
                       help='Path to data.yaml file')
    parser.add_argument('--output', type=str, default='outputs/validation_results.json',
                       help='Output results file')
    
    args = parser.parse_args()
    
    validator = ModelValidator(args.model, args.data)
    metrics = validator.validate()
    validator.save_results(args.output)

if __name__ == "__main__":
    main()