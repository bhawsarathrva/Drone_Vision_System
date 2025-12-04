"""
PyTorch compatibility utilities for the Fire Detection System
"""
import torch
from loguru import logger


def setup_pytorch_compatibility():
    """
    Setup PyTorch compatibility for loading YOLO models.
    
    PyTorch 2.6+ changed the default value of weights_only from False to True
    for security reasons. This function adds Ultralytics classes to the safe
    globals list to allow loading YOLO models.
    """
    try:
        # Check PyTorch version
        pytorch_version = torch.__version__
        major, minor = map(int, pytorch_version.split('.')[:2])
        
        if major > 2 or (major == 2 and minor >= 6):
            logger.info(f"Detected PyTorch {pytorch_version} - Applying compatibility fix")
            
            # Add Ultralytics classes to safe globals
            import torch.serialization
            
            # Import all necessary Ultralytics classes
            try:
                from ultralytics.nn.tasks import DetectionModel, SegmentationModel, ClassificationModel, PoseModel
                from ultralytics.nn.modules import (
                    Conv, C2f, SPPF, Detect, Segment, Classify, Pose,
                    C3, C3TR, SPP, DWConv, Focus, BottleneckCSP, C3Ghost
                )
                
                # Add all model-related classes to safe globals
                safe_classes = [
                    DetectionModel, SegmentationModel, ClassificationModel, PoseModel,
                    Conv, C2f, SPPF, Detect, Segment, Classify, Pose,
                    C3, C3TR, SPP, DWConv, Focus, BottleneckCSP, C3Ghost
                ]
                
                torch.serialization.add_safe_globals(safe_classes)
                logger.info(f"✓ Added {len(safe_classes)} Ultralytics classes to PyTorch safe globals")
                
            except ImportError as ie:
                # Fallback: just add DetectionModel if other imports fail
                logger.warning(f"Could not import all Ultralytics classes: {ie}")
                from ultralytics.nn.tasks import DetectionModel
                torch.serialization.add_safe_globals([DetectionModel])
                logger.info("✓ Added DetectionModel to PyTorch safe globals (minimal)")
            
            return True
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
