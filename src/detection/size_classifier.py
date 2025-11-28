"""
Size classification for fire detections
"""
from typing import List, Dict, Tuple
import numpy as np
from enum import Enum
from loguru import logger

class FireSize(Enum):
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    VERY_LARGE = "very_large"

class SizeClassifier:
    def __init__(
        self,
        image_width: int = 1920,
        image_height: int = 1080,
        small_threshold: float = 0.01,  # 1% of image area
        medium_threshold: float = 0.05,  # 5% of image area
        large_threshold: float = 0.15,   # 15% of image area
    ):
        """
        Initialize size classifier
        
        Args:
            image_width: Expected image width
            image_height: Expected image height
            small_threshold: Threshold for small fires (fraction of image area)
            medium_threshold: Threshold for medium fires
            large_threshold: Threshold for large fires
        """
        self.image_width = image_width
        self.image_height = image_height
        self.image_area = image_width * image_height
        
        self.small_threshold = small_threshold
        self.medium_threshold = medium_threshold
        self.large_threshold = large_threshold
        
    def classify(self, detection: Dict, image_width: int = None, image_height: int = None) -> FireSize:
        """
        Classify fire size based on bounding box area
        
        Args:
            detection: Detection dictionary with 'bbox' key
            image_width: Actual image width (uses default if None)
            image_height: Actual image height (uses default if None)
            
        Returns:
            FireSize enum
        """
        image_width = image_width or self.image_width
        image_height = image_height or self.image_height
        image_area = image_width * image_height
        
        # Get bounding box
        bbox = detection['bbox']
        x1, y1, x2, y2 = bbox
        
        # Calculate area
        width = x2 - x1
        height = y2 - y1
        area = width * height
        area_fraction = area / image_area
        
        # Classify
        if area_fraction < self.small_threshold:
            return FireSize.SMALL
        elif area_fraction < self.medium_threshold:
            return FireSize.MEDIUM
        elif area_fraction < self.large_threshold:
            return FireSize.LARGE
        else:
            return FireSize.VERY_LARGE
    
    def classify_batch(
        self,
        detections: List[Dict],
        image_width: int = None,
        image_height: int = None
    ) -> List[Tuple[Dict, FireSize]]:
        """
        Classify multiple detections
        
        Args:
            detections: List of detection dictionaries
            image_width: Actual image width
            image_height: Actual image height
            
        Returns:
            List of (detection, size) tuples
        """
        return [
            (det, self.classify(det, image_width, image_height))
            for det in detections
        ]
    
    def get_size_stats(self, detections: List[Dict], image_width: int = None, image_height: int = None) -> Dict:
        """
        Get size statistics for detections
        
        Args:
            detections: List of detection dictionaries
            image_width: Actual image width
            image_height: Actual image height
            
        Returns:
            Dictionary with size statistics
        """
        if not detections:
            return {
                'small': 0,
                'medium': 0,
                'large': 0,
                'very_large': 0,
                'total': 0
            }
        
        sizes = [self.classify(det, image_width, image_height) for det in detections]
        
        stats = {
            'small': sum(1 for s in sizes if s == FireSize.SMALL),
            'medium': sum(1 for s in sizes if s == FireSize.MEDIUM),
            'large': sum(1 for s in sizes if s == FireSize.LARGE),
            'very_large': sum(1 for s in sizes if s == FireSize.VERY_LARGE),
            'total': len(detections)
        }
        
        return stats
    
    def update_thresholds(
        self,
        small: float = None,
        medium: float = None,
        large: float = None
    ):
        """Update size thresholds"""
        if small is not None:
            self.small_threshold = small
        if medium is not None:
            self.medium_threshold = medium
        if large is not None:
            self.large_threshold = large
        
        logger.info(f"Size thresholds updated: small={self.small_threshold}, "
                   f"medium={self.medium_threshold}, large={self.large_threshold}")

