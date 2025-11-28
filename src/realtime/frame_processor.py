"""
Frame processing utilities for fire detection
"""
import cv2
import numpy as np
from typing import Tuple, Optional, List
from loguru import logger

class FrameProcessor:
    def __init__(
        self,
        target_size: Tuple[int, int] = (640, 640),
        normalize: bool = True,
        enhance: bool = False
    ):
        """
        Initialize frame processor
        
        Args:
            target_size: Target frame size (width, height)
            normalize: Normalize pixel values to [0, 1]
            enhance: Apply image enhancement
        """
        self.target_size = target_size
        self.normalize = normalize
        self.enhance = enhance
    
    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess frame for detection
        
        Args:
            frame: Input frame (BGR)
            
        Returns:
            Preprocessed frame
        """
        # Resize
        frame = cv2.resize(frame, self.target_size)
        
        # Enhance if enabled
        if self.enhance:
            frame = self._enhance_image(frame)
        
        # Normalize if enabled
        if self.normalize:
            frame = frame.astype(np.float32) / 255.0
        
        return frame
    
    def _enhance_image(self, frame: np.ndarray) -> np.ndarray:
        """Apply image enhancement"""
        # Convert to LAB color space
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Merge channels
        lab = cv2.merge([l, a, b])
        frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # Apply slight sharpening
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        frame = cv2.filter2D(frame, -1, kernel)
        
        return frame
    
    def draw_detections(
        self,
        frame: np.ndarray,
        detections: List[dict],
        show_labels: bool = True,
        show_confidence: bool = True
    ) -> np.ndarray:
        """
        Draw detections on frame
        
        Args:
            frame: Input frame
            detections: List of detection dictionaries
            show_labels: Show class labels
            show_confidence: Show confidence scores
            
        Returns:
            Frame with detections drawn
        """
        frame = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            confidence = det['confidence']
            class_name = det.get('class_name', 'fire')
            size = det.get('size', 'unknown')
            
            # Choose color based on size
            color = self._get_size_color(size)
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            if show_labels or show_confidence:
                label_parts = []
                if show_labels:
                    label_parts.append(class_name)
                if show_confidence:
                    label_parts.append(f"{confidence:.2f}")
                if 'size' in det:
                    label_parts.append(f"({size})")
                
                label = " | ".join(label_parts)
                
                # Calculate text size
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
                )
                
                # Draw label background
                cv2.rectangle(
                    frame,
                    (x1, y1 - text_height - baseline - 5),
                    (x1 + text_width, y1),
                    color,
                    -1
                )
                
                # Draw label text
                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - baseline - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2
                )
        
        return frame
    
    def _get_size_color(self, size: str) -> Tuple[int, int, int]:
        """Get color for fire size"""
        color_map = {
            'small': (0, 255, 255),      # Yellow
            'medium': (0, 165, 255),     # Orange
            'large': (0, 0, 255),        # Red
            'very_large': (0, 0, 139)    # Dark red
        }
        return color_map.get(size.lower(), (0, 0, 255))
    
    def draw_statistics(
        self,
        frame: np.ndarray,
        statistics: dict,
        position: Tuple[int, int] = (10, 30)
    ) -> np.ndarray:
        """
        Draw statistics on frame
        
        Args:
            frame: Input frame
            statistics: Statistics dictionary
            position: Text position (x, y)
            
        Returns:
            Frame with statistics drawn
        """
        frame = frame.copy()
        x, y = position
        
        # Statistics text
        stats_lines = []
        if 'fps' in statistics:
            stats_lines.append(f"FPS: {statistics['fps']:.1f}")
        if 'detection_count' in statistics:
            stats_lines.append(f"Detections: {statistics['detection_count']}")
        if 'frame_count' in statistics:
            stats_lines.append(f"Frames: {statistics['frame_count']}")
        
        # Draw each line
        for i, line in enumerate(stats_lines):
            cv2.putText(
                frame,
                line,
                (x, y + i * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
        
        return frame
    
    def resize_frame(self, frame: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        """Resize frame"""
        return cv2.resize(frame, size)
    
    def crop_frame(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """Crop frame to bounding box"""
        x1, y1, x2, y2 = bbox
        return frame[y1:y2, x1:x2]

