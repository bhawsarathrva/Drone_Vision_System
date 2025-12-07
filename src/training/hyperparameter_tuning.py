import os
import yaml
from pathlib import Path
import itertools

# Disable MLflow integration BEFORE importing ultralytics
os.environ['MLFLOW_TRACKING_URI'] = ''
os.environ['MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING'] = 'false'
os.environ['MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR'] = 'false'

from ultralytics import YOLO
from ultralytics import settings
import torch
from loguru import logger

# Disable MLflow in ultralytics settings
settings.update({'mlflow': False})

# Determine project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

class HyperparameterTuner:
    def __init__(self, base_config: dict):
        self.base_config = base_config
        self.results = []
    
    def grid_search(
        self,
        param_grid: dict,
        data_path: str,
        epochs: int = 50,
        max_trials: int = None
    ):
        # Generate parameter combinations
        param_names = list(param_grid.keys())
        param_values = [param_grid[name] for name in param_names]
        combinations = list(itertools.product(*param_values))
        
        if max_trials:
            combinations = combinations[:max_trials]
        
        logger.info(f"Running grid search with {len(combinations)} combinations")
        
        best_score = 0
        best_params = None
        
        # Monkey-patch ultralytics' torch_safe_load for PyTorch 2.6+ compatibility
        try:
            from ultralytics.nn import tasks
            
            # Only patch if not already patched (check attribute)
            if not hasattr(tasks, '_is_patched_by_tuner'):
                if hasattr(tasks, 'torch_safe_load'):
                    original_torch_safe_load = tasks.torch_safe_load
                    
                    def patched_torch_safe_load(file, *args, **kwargs):
                        try:
                            # Verify if torch is imported
                            import torch
                            return torch.load(file, map_location='cpu', weights_only=False), file
                        except Exception:
                            return original_torch_safe_load(file, *args, **kwargs)
                    
                    tasks.torch_safe_load = patched_torch_safe_load
                    tasks._is_patched_by_tuner = True
                    logger.info("Patched ultralytics.nn.tasks.torch_safe_load for PyTorch 2.6 compatibility")
        except Exception as e:
            logger.warning(f"Failed to patch torch_safe_load: {e}")
        
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
                model_name = config.get('model', 'yolov8n.pt')
                model = YOLO(model_name)
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
    
    # Default paths handling
    default_config = PROJECT_ROOT / 'configs' / 'training_config.yaml'
    default_data = PROJECT_ROOT / 'Dataset' / 'data.yaml'
    
    parser = argparse.ArgumentParser(description='Hyperparameter tuning for fire detection')
    parser.add_argument('--config', type=str, default=str(default_config),
                       help='Path to training config file')
    parser.add_argument('--data', type=str, default=str(default_data),
                       help='Path to data.yaml file')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs per trial')
    parser.add_argument('--max-trials', type=int, default=None,
                       help='Maximum number of trials')
    
    args = parser.parse_args()
    
    # Load base config
    if not os.path.exists(args.config):
        # Try relative path if absolute fail
        if os.path.exists(os.path.join(PROJECT_ROOT, args.config)):
             args.config = os.path.join(PROJECT_ROOT, args.config)
        else:
             logger.error(f"Config file not found: {args.config}")
             return

    with open(args.config, 'r') as f:
        base_config = yaml.safe_load(f)
    
    # Ensure data path is correct
    if not os.path.exists(args.data):
        if os.path.exists(os.path.join(PROJECT_ROOT, args.data)):
            args.data = os.path.join(PROJECT_ROOT, args.data)
    
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
    output_json = PROJECT_ROOT / 'outputs' / 'hyperparameter_tuning_results.json'
    tuner.save_results(str(output_json))
    
    logger.info(f"Best parameters: {best_params}")
    logger.info(f"Best score: {best_score:.4f}")

if __name__ == "__main__":
    main()

