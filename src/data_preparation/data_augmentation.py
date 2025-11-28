"""
Data augmentation for fire detection dataset
"""
import cv2
import numpy as np
from typing import List, Tuple, Optional
import random
from loguru import logger

class DataAugmentation:
    def __init__(self):
        """Initialize data augmentation"""
        pass
    
    def augment(
        self,
        image: np.ndarray,
        bboxes: List[List[float]] = None,
        augmentation_types: List[str] = None
    ) -> Tuple[np.ndarray, List[List[float]]]:
        """
        Apply data augmentation
        
        Args:
            image: Input image
            bboxes: List of bounding boxes [x1, y1, x2, y2]
            augmentation_types: List of augmentation types to apply
            
        Returns:
            (augmented_image, augmented_bboxes)
        """
        if augmentation_types is None:
            augmentation_types = ['flip', 'brightness', 'contrast', 'noise']
        
        augmented_image = image.copy()
        augmented_bboxes = bboxes.copy() if bboxes else []
        
        for aug_type in augmentation_types:
            if aug_type == 'flip' and random.random() > 0.5:
                augmented_image, augmented_bboxes = self._flip(
                    augmented_image, augmented_bboxes
                )
            elif aug_type == 'brightness' and random.random() > 0.5:
                augmented_image = self._adjust_brightness(augmented_image)
            elif aug_type == 'contrast' and random.random() > 0.5:
                augmented_image = self._adjust_contrast(augmented_image)
            elif aug_type == 'noise' and random.random() > 0.5:
                augmented_image = self._add_noise(augmented_image)
            elif aug_type == 'blur' and random.random() > 0.5:
                augmented_image = self._apply_blur(augmented_image)
            elif aug_type == 'rotate' and random.random() > 0.5:
                augmented_image, augmented_bboxes = self._rotate(
                    augmented_image, augmented_bboxes
                )
        
        return augmented_image, augmented_bboxes
    
    def _flip(
        self,
        image: np.ndarray,
        bboxes: List[List[float]]
    ) -> Tuple[np.ndarray, List[List[float]]]:
        """Horizontal flip"""
        flipped_image = cv2.flip(image, 1)
        flipped_bboxes = []
        
        h, w = image.shape[:2]
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            flipped_bboxes.append([w - x2, y1, w - x1, y2])
        
        return flipped_image, flipped_bboxes
    
    def _adjust_brightness(self, image: np.ndarray) -> np.ndarray:
        """Adjust brightness"""
        alpha = random.uniform(0.7, 1.3)
        return cv2.convertScaleAbs(image, alpha=alpha, beta=0)
    
    def _adjust_contrast(self, image: np.ndarray) -> np.ndarray:
        """Adjust contrast"""
        alpha = random.uniform(0.8, 1.2)
        beta = random.randint(-20, 20)
        return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    
    def _add_noise(self, image: np.ndarray) -> np.ndarray:
        """Add Gaussian noise"""
        noise = np.random.normal(0, 10, image.shape).astype(np.uint8)
        noisy_image = cv2.add(image, noise)
        return noisy_image
    
    def _apply_blur(self, image: np.ndarray) -> np.ndarray:
        """Apply Gaussian blur"""
        kernel_size = random.choice([3, 5])
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    
    def _rotate(
        self,
        image: np.ndarray,
        bboxes: List[List[float]],
        angle: float = None
    ) -> Tuple[np.ndarray, List[List[float]]]:
        """Rotate image and bounding boxes"""
        if angle is None:
            angle = random.uniform(-15, 15)
        
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        # Get rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Rotate image
        rotated_image = cv2.warpAffine(image, M, (w, h))
        
        # Rotate bounding boxes
        rotated_bboxes = []
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            corners = np.array([
                [x1, y1],
                [x2, y1],
                [x2, y2],
                [x1, y2]
            ], dtype=np.float32)
            
            # Rotate corners
            rotated_corners = cv2.transform(
                corners.reshape(1, -1, 2), M
            ).reshape(-1, 2)
            
            # Get new bounding box
            x_min = int(np.min(rotated_corners[:, 0]))
            y_min = int(np.min(rotated_corners[:, 1]))
            x_max = int(np.max(rotated_corners[:, 0]))
            y_max = int(np.max(rotated_corners[:, 1]))
            
            # Clip to image bounds
            x_min = max(0, min(w, x_min))
            y_min = max(0, min(h, y_min))
            x_max = max(0, min(w, x_max))
            y_max = max(0, min(h, y_max))
            
            rotated_bboxes.append([x_min, y_min, x_max, y_max])
        
        return rotated_image, rotated_bboxes

