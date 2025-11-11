"""
Hyperparameter tuning for fire detection model
"""
import os
import yaml
from pathlib import Path
from ultralytics import YOLO
from loguru import logger
import itertools

class HyperparameterTuner:
    def __init__(self, base_config: dict):
        """
        Initialize hyperparameter tuner
        
        Args:
            base_config: Base configuration dictionary
        """
        self.base_config = base_config
        self.results = []
    
    def grid_search(
        self,
        param_grid: dict,
        data_path: str,
        epochs: int = 50,
        max_trials: int = None
    ):
        """
        Perform grid search over hyperparameters
        
        Args:
            param_grid: Dictionary of parameter ranges
            data_path: Path to data.yaml file
            epochs: Number of epochs per trial
            max_trials: Maximum number of trials (None for all)
        """
        # Generate parameter combinations
        param_names = list(param_grid.keys())
        param_values = [param_grid[name] for name in param_names]
        combinations = list(itertools.product(*param_values))
        
        if max_trials:
            combinations = combinations[:max_trials]
        
        logger.info(f"Running grid search with {len(combinations)} combinations")
        
        best_score = 0
        best_params = None
        
        for i, combination in enumerate(combinations):
            params = dict(zip(param_names, combination))
            logger.info(f"Trial {i+1}/{len(combinations)}: {params}")
            
            # Update config
            config = self.base_config.copy()
            config.update(params)
            config['epochs'] = epochs
            config['data'] = data_path
            config['name'] = f'trial_{i+1}'
            
            # Train model
            try:
                model = YOLO(config.get('model', 'yolov8n.pt'))
                results = model.train(**config)
                
                # Get validation score
                score = results.box.map50
                
                # Save results
                self.results.append({
                    'params': params,
                    'score': score,
                    'results': results
                })
                
                # Update best
                if score > best_score:
                    best_score = score
                    best_params = params
                
                logger.info(f"Trial {i+1} completed: mAP50 = {score:.4f}")
                
            except Exception as e:
                logger.error(f"Trial {i+1} failed: {e}")
                continue
        
        logger.info(f"Grid search completed. Best score: {best_score:.4f}")
        logger.info(f"Best parameters: {best_params}")
        
        return best_params, best_score
    
    def save_results(self, output_path: str):
        """Save tuning results"""
        import json
        
        results_dict = {
            'results': [
                {
                    'params': r['params'],
                    'score': float(r['score'])
                }
                for r in self.results
            ]
        }
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")

def main():
    """Main hyperparameter tuning function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Hyperparameter tuning for fire detection')
    parser.add_argument('--config', type=str, default='configs/training_config.yaml',
                       help='Path to training config file')
    parser.add_argument('--data', type=str, default='Dataset/data.yaml',
                       help='Path to data.yaml file')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs per trial')
    parser.add_argument('--max-trials', type=int, default=None,
                       help='Maximum number of trials')
    
    args = parser.parse_args()
    
    # Load base config
    with open(args.config, 'r') as f:
        base_config = yaml.safe_load(f)
    
    # Define parameter grid
    param_grid = {
        'lr0': [0.01, 0.001, 0.0001],
        'batch': [8, 16, 32],
        'imgsz': [640, 1280]
    }
    
    # Run tuning
    tuner = HyperparameterTuner(base_config)
    best_params, best_score = tuner.grid_search(
        param_grid=param_grid,
        data_path=args.data,
        epochs=args.epochs,
        max_trials=args.max_trials
    )
    
    # Save results
    tuner.save_results('outputs/hyperparameter_tuning_results.json')
    
    logger.info(f"Best parameters: {best_params}")
    logger.info(f"Best score: {best_score:.4f}")

if __name__ == "__main__":
    main()

