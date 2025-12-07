"""
PyTorch compatibility utilities for the Fire Detection System
"""
import torch
from loguru import logger


def setup_pytorch_compatibility():
    """
    Setup PyTorch compatibility for loading YOLO models.
    
    PyTorch 2.6+ changed the default value of weights_only from False to True
    for security reasons. This function monkey-patches ultralytics' torch_safe_load
    to use weights_only=False, which is safe for YOLO models from trusted sources.
    
    Returns:
        bool: True if patch was applied, False otherwise
    """
    try:
        # Check PyTorch version
        pytorch_version = torch.__version__
        version_parts = pytorch_version.split('.')
        major = int(version_parts[0])
        minor = int(version_parts[1].split('+')[0]) if '+' in version_parts[1] else int(version_parts[1])
        
        if major > 2 or (major == 2 and minor >= 6):
            logger.info(f"Detected PyTorch {pytorch_version} - Applying compatibility fix")
            
            # Monkey-patch ultralytics' torch_safe_load function
            try:
                from ultralytics.nn import tasks
                
                # Store original function
                if not hasattr(tasks, '_original_torch_safe_load'):
                    tasks._original_torch_safe_load = tasks.torch_safe_load
                
                def patched_torch_safe_load(file, *args, **kwargs):
                    """
                    Patched version that uses weights_only=False for PyTorch 2.6+ compatibility.
                    This is safe for YOLO models from trusted sources (official ultralytics models).
                    """
                    try:
                        # Try loading with weights_only=False for compatibility
                        return torch.load(file, map_location='cpu', weights_only=False), file
                    except Exception:
                        # Fallback to original implementation
                        return tasks._original_torch_safe_load(file, *args, **kwargs)
                
                # Apply the patch
                tasks.torch_safe_load = patched_torch_safe_load
                logger.info("✓ Patched ultralytics.nn.tasks.torch_safe_load for PyTorch 2.6 compatibility")
                return True
                
            except Exception as patch_error:
                logger.warning(f"Could not patch torch_safe_load: {patch_error}")
                return False
        else:
            logger.debug(f"PyTorch {pytorch_version} - No compatibility fix needed")
            return False
            
    except Exception as e:
        logger.warning(f"Could not apply PyTorch compatibility fix: {e}")
        return False


def load_yolo_model_safe(model_path: str):
    """
    Safely load a YOLO model with PyTorch 2.6+ compatibility.
    
    Args:
        model_path: Path to YOLO model file or model name
        
    Returns:
        Loaded YOLO model
    """
    from ultralytics import YOLO
    
    # Apply compatibility fix
    setup_pytorch_compatibility()
    
    # Load model
    try:
        model = YOLO(model_path)
        logger.info(f"✓ Model loaded successfully: {model_path}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model {model_path}: {e}")
        raise
