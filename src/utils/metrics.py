"""
Metrics and evaluation utilities for fire detection
"""
import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict
import json
from pathlib import Path

class DetectionMetrics:
    def __init__(self):
        """Initialize metrics collector"""
        self.metrics = {
            'total_frames': 0,
            'total_detections': 0,
            'detection_history': [],
            'confidence_history': [],
            'size_distribution': defaultdict(int),
            'fps_history': [],
            'false_positives': 0,
            'false_negatives': 0,
            'true_positives': 0
        }
    
    def update(self, detections: List[Dict], fps: float = None):
        """
        Update metrics with new detections
        
        Args:
            detections: List of detection dictionaries
            fps: Current FPS
        """
        self.metrics['total_frames'] += 1
        self.metrics['total_detections'] += len(detections)
        self.metrics['detection_history'].append(len(detections))
        
        # Collect confidence scores
        for det in detections:
            self.metrics['confidence_history'].append(det['confidence'])
            if 'size' in det:
                self.metrics['size_distribution'][det['size']] += 1
        
        # Update FPS
        if fps is not None:
            self.metrics['fps_history'].append(fps)
    
    def get_statistics(self) -> Dict:
        """Get computed statistics"""
        stats = self.metrics.copy()
        
        # Compute averages
        if self.metrics['confidence_history']:
            stats['avg_confidence'] = np.mean(self.metrics['confidence_history'])
            stats['min_confidence'] = np.min(self.metrics['confidence_history'])
            stats['max_confidence'] = np.max(self.metrics['confidence_history'])
            stats['std_confidence'] = np.std(self.metrics['confidence_history'])
        
        if self.metrics['fps_history']:
            stats['avg_fps'] = np.mean(self.metrics['fps_history'])
            stats['min_fps'] = np.min(self.metrics['fps_history'])
            stats['max_fps'] = np.max(self.metrics['fps_history'])
        
        if self.metrics['total_frames'] > 0:
            stats['detection_rate'] = (
                self.metrics['total_detections'] / self.metrics['total_frames'] * 100
            )
        
        # Compute precision, recall, F1
        if self.metrics['true_positives'] + self.metrics['false_positives'] > 0:
            stats['precision'] = (
                self.metrics['true_positives'] /
                (self.metrics['true_positives'] + self.metrics['false_positives'])
            )
        else:
            stats['precision'] = 0.0
        
        if self.metrics['true_positives'] + self.metrics['false_negatives'] > 0:
            stats['recall'] = (
                self.metrics['true_positives'] /
                (self.metrics['true_positives'] + self.metrics['false_negatives'])
            )
        else:
            stats['recall'] = 0.0
        
        if stats['precision'] + stats['recall'] > 0:
            stats['f1_score'] = (
                2 * stats['precision'] * stats['recall'] /
                (stats['precision'] + stats['recall'])
            )
        else:
            stats['f1_score'] = 0.0
        
        return stats
    
    def reset(self):
        """Reset metrics"""
        self.metrics = {
            'total_frames': 0,
            'total_detections': 0,
            'detection_history': [],
            'confidence_history': [],
            'size_distribution': defaultdict(int),
            'fps_history': [],
            'false_positives': 0,
            'false_negatives': 0,
            'true_positives': 0
        }
    
    def save(self, path: str):
        """Save metrics to file"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.get_statistics(), f, indent=2)
    
    def load(self, path: str):
        """Load metrics from file"""
        with open(path, 'r') as f:
            self.metrics = json.load(f)

def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """
    Calculate Intersection over Union (IoU) of two bounding boxes
    
    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]
        
    Returns:
        IoU value
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 < x1 or y2 < y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    if union == 0:
        return 0.0
    
    return intersection / union

def match_detections(
    pred_detections: List[Dict],
    gt_detections: List[Dict],
    iou_threshold: float = 0.5
) -> Tuple[List[int], List[int], List[int]]:
    """
    Match predicted detections with ground truth
    
    Args:
        pred_detections: Predicted detections
        gt_detections: Ground truth detections
        iou_threshold: IoU threshold for matching
        
    Returns:
        (true_positives, false_positives, false_negatives)
    """
    tp = []
    fp = []
    fn = []
    
    matched_gt = set()
    
    for pred in pred_detections:
        best_iou = 0
        best_gt_idx = -1
        
        for gt_idx, gt in enumerate(gt_detections):
            if gt_idx in matched_gt:
                continue
            
            iou = calculate_iou(pred['bbox'], gt['bbox'])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        
        if best_iou >= iou_threshold:
            tp.append(pred)
            matched_gt.add(best_gt_idx)
        else:
            fp.append(pred)
    
    # Unmatched ground truth are false negatives
    for gt_idx, gt in enumerate(gt_detections):
        if gt_idx not in matched_gt:
            fn.append(gt)
    
    return tp, fp, fn

