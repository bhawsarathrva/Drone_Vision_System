"""
Confidence filtering for fire detections
"""
from typing import List, Dict
import numpy as np
from loguru import logger

class ConfidenceFilter:
    def __init__(
        self,
        min_confidence: float = 0.5,
        max_confidence: float = 1.0,
        use_adaptive: bool = False,
        adaptive_window: int = 10
    ):
        """
        Initialize confidence filter
        
        Args:
            min_confidence: Minimum confidence threshold
            max_confidence: Maximum confidence threshold
            use_adaptive: Use adaptive thresholding
            adaptive_window: Window size for adaptive threshold
        """
        self.min_confidence = min_confidence
        self.max_confidence = max_confidence
        self.use_adaptive = use_adaptive
        self.adaptive_window = adaptive_window
        
        # Adaptive threshold tracking
        self.confidence_history = []
        self.adaptive_threshold = min_confidence
        
    def filter(self, detections: List[Dict]) -> List[Dict]:
        """
        Filter detections by confidence
        
        Args:
            detections: List of detection dictionaries
            
        Returns:
            Filtered list of detections
        """
        if not detections:
            return []
        
        # Update adaptive threshold if enabled
        if self.use_adaptive:
            self._update_adaptive_threshold(detections)
        
        # Determine threshold to use
        threshold = self.adaptive_threshold if self.use_adaptive else self.min_confidence
        
        # Filter detections
        filtered = [
            det for det in detections
            if threshold <= det['confidence'] <= self.max_confidence
        ]
        
        return filtered
    
    def _update_adaptive_threshold(self, detections: List[Dict]):
        """Update adaptive threshold based on confidence history"""
        if detections:
            # Get average confidence
            avg_confidence = np.mean([det['confidence'] for det in detections])
            self.confidence_history.append(avg_confidence)
            
            # Keep only recent history
            if len(self.confidence_history) > self.adaptive_window:
                self.confidence_history.pop(0)
            
            # Calculate adaptive threshold (mean - std)
            if len(self.confidence_history) >= 3:
                mean_conf = np.mean(self.confidence_history)
                std_conf = np.std(self.confidence_history)
                self.adaptive_threshold = max(
                    self.min_confidence,
                    mean_conf - std_conf
                )
            else:
                self.adaptive_threshold = self.min_confidence
    
    def update_threshold(self, threshold: float):
        """Update minimum confidence threshold"""
        self.min_confidence = max(0.0, min(1.0, threshold))
        logger.info(f"Confidence threshold updated to {self.min_confidence}")
    
    def reset(self):
        """Reset filter state"""
        self.confidence_history = []
        self.adaptive_threshold = self.min_confidence
    
    def get_statistics(self) -> Dict:
        """Get filter statistics"""
        return {
            'min_confidence': self.min_confidence,
            'max_confidence': self.max_confidence,
            'adaptive_threshold': self.adaptive_threshold if self.use_adaptive else None,
            'use_adaptive': self.use_adaptive,
            'history_size': len(self.confidence_history)
        }

