"""
Data preprocessing utilities for fire detection dataset
"""
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import json
from loguru import logger

class DataPreprocessor:
    def __init__(
        self,
        target_size: Tuple[int, int] = (640, 640),
        normalize: bool = True
    ):
        """
        Initialize data preprocessor
        
        Args:
            target_size: Target image size (width, height)
            normalize: Normalize pixel values
        """
        self.target_size = target_size
        self.normalize = normalize
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Preprocess single image
        
        Args:
            image_path: Path to image file
            
        Returns:
            Preprocessed image
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")
        
        # Resize
        image = cv2.resize(image, self.target_size)
        
        # Normalize
        if self.normalize:
            image = image.astype(np.float32) / 255.0
        
        return image
    
    def preprocess_batch(self, image_paths: List[str]) -> np.ndarray:
        """
        Preprocess batch of images
        
        Args:
            image_paths: List of image paths
            
        Returns:
            Batch of preprocessed images
        """
        images = []
        for image_path in image_paths:
            try:
                image = self.preprocess_image(image_path)
                images.append(image)
            except Exception as e:
                logger.error(f"Failed to preprocess {image_path}: {e}")
        
        return np.array(images)
    
    def augment_image(
        self,
        image: np.ndarray,
        bboxes: List[List[float]] = None,
        augmentation_types: List[str] = None
    ) -> Tuple[np.ndarray, List[List[float]]]:
        """
        Augment image with bounding boxes
        
        Args:
            image: Input image
            bboxes: List of bounding boxes [x1, y1, x2, y2]
            augmentation_types: List of augmentation types
            
        Returns:
            (augmented_image, augmented_bboxes)
        """
        if augmentation_types is None:
            augmentation_types = ['flip', 'brightness', 'contrast']
        
        augmented_image = image.copy()
        augmented_bboxes = bboxes.copy() if bboxes else []
        
        # Horizontal flip
        if 'flip' in augmentation_types:
            augmented_image = cv2.flip(augmented_image, 1)
            if augmented_bboxes:
                h, w = augmented_image.shape[:2]
                for bbox in augmented_bboxes:
                    x1, y1, x2, y2 = bbox
                    bbox[0] = w - x2
                    bbox[2] = w - x1
        
        # Brightness adjustment
        if 'brightness' in augmentation_types:
            alpha = np.random.uniform(0.8, 1.2)
            augmented_image = cv2.convertScaleAbs(augmented_image, alpha=alpha, beta=0)
        
        # Contrast adjustment
        if 'contrast' in augmentation_types:
            alpha = np.random.uniform(0.8, 1.2)
            augmented_image = cv2.convertScaleAbs(augmented_image, alpha=alpha, beta=0)
        
        return augmented_image, augmented_bboxes
    
    def convert_yolo_to_xyxy(
        self,
        yolo_bbox: List[float],
        image_width: int,
        image_height: int
    ) -> List[float]:
        """
        Convert YOLO format (center_x, center_y, width, height) to xyxy
        
        Args:
            yolo_bbox: YOLO format bbox [center_x, center_y, width, height] (normalized)
            image_width: Image width
            image_height: Image height
            
        Returns:
            xyxy format bbox [x1, y1, x2, y2]
        """
        center_x, center_y, width, height = yolo_bbox
        
        # Convert to pixel coordinates
        center_x *= image_width
        center_y *= image_height
        width *= image_width
        height *= image_height
        
        # Convert to xyxy
        x1 = center_x - width / 2
        y1 = center_y - height / 2
        x2 = center_x + width / 2
        y2 = center_y + height / 2
        
        return [x1, y1, x2, y2]
    
    def convert_xyxy_to_yolo(
        self,
        xyxy_bbox: List[float],
        image_width: int,
        image_height: int
    ) -> List[float]:
        """
        Convert xyxy format to YOLO format
        
        Args:
            xyxy_bbox: xyxy format bbox [x1, y1, x2, y2]
            image_width: Image width
            image_height: Image height
            
        Returns:
            YOLO format bbox [center_x, center_y, width, height] (normalized)
        """
        x1, y1, x2, y2 = xyxy_bbox
        
        # Calculate center and size
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        width = x2 - x1
        height = y2 - y1
        
        # Normalize
        center_x /= image_width
        center_y /= image_height
        width /= image_width
        height /= image_height
        
        return [center_x, center_y, width, height]

