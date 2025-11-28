"""
Visualization utilities for fire detection
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import json

def draw_detections(
    image: np.ndarray,
    detections: List[Dict],
    show_labels: bool = True,
    show_confidence: bool = True,
    show_size: bool = True
) -> np.ndarray:
    """
    Draw detections on image
    
    Args:
        image: Input image (BGR)
        detections: List of detection dictionaries
        show_labels: Show class labels
        show_confidence: Show confidence scores
        show_size: Show fire size
        
    Returns:
        Image with detections drawn
    """
    image = image.copy()
    
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        confidence = det['confidence']
        class_name = det.get('class_name', 'fire')
        size = det.get('size', 'unknown')
        
        # Choose color based on confidence
        color = get_confidence_color(confidence)
        
        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        label_parts = []
        if show_labels:
            label_parts.append(class_name)
        if show_confidence:
            label_parts.append(f"{confidence:.2f}")
        if show_size and 'size' in det:
            label_parts.append(f"({size})")
        
        if label_parts:
            label = " | ".join(label_parts)
            
            # Calculate text size
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            
            # Draw label background
            cv2.rectangle(
                image,
                (x1, y1 - text_height - baseline - 5),
                (x1 + text_width, y1),
                color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                image,
                label,
                (x1, y1 - baseline - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
    
    return image

def get_confidence_color(confidence: float) -> Tuple[int, int, int]:
    """Get color based on confidence level"""
    if confidence >= 0.8:
        return (0, 0, 255)      # Red (high confidence)
    elif confidence >= 0.5:
        return (0, 165, 255)    # Orange (medium confidence)
    else:
        return (0, 255, 255)    # Yellow (low confidence)

def get_size_color(size: str) -> Tuple[int, int, int]:
    """Get color based on fire size"""
    color_map = {
        'small': (0, 255, 255),      # Yellow
        'medium': (0, 165, 255),     # Orange
        'large': (0, 0, 255),        # Red
        'very_large': (0, 0, 139)    # Dark red
    }
    return color_map.get(size.lower(), (0, 0, 255))

def plot_detection_statistics(
    statistics: Dict,
    save_path: Optional[str] = None
):
    """
    Plot detection statistics
    
    Args:
        statistics: Statistics dictionary
        save_path: Path to save plot (optional)
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Detection count over time
    if 'detection_history' in statistics:
        axes[0, 0].plot(statistics['detection_history'])
        axes[0, 0].set_title('Detection Count Over Time')
        axes[0, 0].set_xlabel('Frame')
        axes[0, 0].set_ylabel('Detections')
        axes[0, 0].grid(True)
    
    # Confidence distribution
    if 'confidence_distribution' in statistics:
        axes[0, 1].hist(statistics['confidence_distribution'], bins=20)
        axes[0, 1].set_title('Confidence Distribution')
        axes[0, 1].set_xlabel('Confidence')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True)
    
    # Size distribution
    if 'size_distribution' in statistics:
        sizes = list(statistics['size_distribution'].keys())
        counts = list(statistics['size_distribution'].values())
        axes[1, 0].bar(sizes, counts)
        axes[1, 0].set_title('Size Distribution')
        axes[1, 0].set_xlabel('Size')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].grid(True)
    
    # FPS over time
    if 'fps_history' in statistics:
        axes[1, 1].plot(statistics['fps_history'])
        axes[1, 1].set_title('FPS Over Time')
        axes[1, 1].set_xlabel('Frame')
        axes[1, 1].set_ylabel('FPS')
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def create_detection_video(
    input_video: str,
    output_video: str,
    detections_list: List[List[Dict]],
    fps: int = 30
):
    """
    Create video with detections drawn
    
    Args:
        input_video: Input video path
        output_video: Output video path
        detections_list: List of detections for each frame
        fps: Output video FPS
    """
    cap = cv2.VideoCapture(input_video)
    
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {input_video}")
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Draw detections if available
        if frame_idx < len(detections_list):
            frame = draw_detections(frame, detections_list[frame_idx])
        
        out.write(frame)
        frame_idx += 1
    
    cap.release()
    out.release()

def save_detections_json(
    detections: List[Dict],
    output_path: str
):
    """Save detections to JSON file"""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(detections, f, indent=2)

def load_detections_json(input_path: str) -> List[Dict]:
    """Load detections from JSON file"""
    with open(input_path, 'r') as f:
        return json.load(f)

