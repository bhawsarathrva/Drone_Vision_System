"""
Enhanced real-time fire detection optimized for drone operations
Features: Adaptive FPS, frame skipping, temporal filtering, tracking
"""
import cv2
import numpy as np
from typing import Optional, Callable, List, Dict
from threading import Thread, Lock
import time
from collections import deque
from loguru import logger

from src.detection.fire_detector import FireDetector
from src.detection.confidence_filter import ConfidenceFilter
from src.detection.size_classifier import SizeClassifier, FireSize
from src.drone.camera_interface import CameraInterface
from src.drone.drone_controller import DroneController


class TemporalFilter:
    """Temporal filtering to reduce false positives"""
    
    def __init__(self, window_size: int = 5, threshold: float = 0.6):
        """
        Args:
            window_size: Number of frames to consider
            threshold: Minimum ratio of frames with detection to confirm
        """
        self.window_size = window_size
        self.threshold = threshold
        self.detection_history = deque(maxlen=window_size)
    
    def update(self, has_detection: bool) -> bool:
        """
        Update history and check if detection is stable
        
        Returns:
            True if detection is confirmed across multiple frames
        """
        self.detection_history.append(has_detection)
        
        if len(self.detection_history) < self.window_size:
            return False
        
        detection_ratio = sum(self.detection_history) / len(self.detection_history)
        return detection_ratio >= self.threshold
    
    def reset(self):
        """Reset history"""
        self.detection_history.clear()


class AdaptiveFPSController:
    """Adaptive FPS control based on detection load"""
    
    def __init__(self, base_fps: int = 30, min_fps: int = 10, max_fps: int = 60):
        self.base_fps = base_fps
        self.min_fps = min_fps
        self.max_fps = max_fps
        self.current_fps = base_fps
        self.detection_load = 0.0
    
    def update(self, inference_time: float, has_detections: bool):
        """Adjust FPS based on processing load"""
        # If detections found, prioritize accuracy (lower FPS)
        if has_detections:
            self.current_fps = max(self.min_fps, self.current_fps - 2)
        else:
            # No detections, can increase FPS for faster scanning
            self.current_fps = min(self.max_fps, self.current_fps + 1)
        
        return self.current_fps
    
    def get_frame_time(self) -> float:
        """Get target time per frame"""
        return 1.0 / self.current_fps


class EnhancedRealtimeDetector:
    """Enhanced real-time detector optimized for drone-based fire detection"""
    
    def __init__(
        self,
        model_path: str = "Model/best.pt",
        camera_source: str = "0",
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        fps: int = 30,
        img_size: int = 640,
        use_half: bool = True,
        use_tta: bool = False,
        callback: Optional[Callable] = None,
        enable_temporal_filter: bool = True,
        enable_adaptive_fps: bool = True,
        frame_skip: int = 0
    ):
        """
        Initialize enhanced real-time detector
        
        Args:
            model_path: Path to YOLOv8 model or Hugging Face model ID
            camera_source: Camera source (index, RTSP URL, or file path)
            confidence_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            fps: Target FPS
            img_size: Input image size for model
            use_half: Use FP16 half-precision
            use_tta: Use Test Time Augmentation
            callback: Callback function for detections (detections, frame, telemetry)
            enable_temporal_filter: Enable temporal filtering
            enable_adaptive_fps: Enable adaptive FPS control
            frame_skip: Number of frames to skip (0 = process all frames)
        """
        self.model_path = model_path
        self.camera_source = camera_source
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.target_fps = fps
        self.img_size = img_size
        self.use_half = use_half
        self.use_tta = use_tta
        self.callback = callback
        self.frame_skip = frame_skip
        
        # Initialize components
        logger.info("Initializing enhanced fire detector...")
        self.detector = FireDetector(
            model_path=model_path,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            img_size=img_size,
            use_half=use_half,
            use_tta=use_tta,
            device="auto"
        )
        
        self.confidence_filter = ConfidenceFilter(min_confidence=confidence_threshold)
        self.size_classifier = SizeClassifier()
        
        # Temporal filtering
        self.enable_temporal_filter = enable_temporal_filter
        self.temporal_filter = TemporalFilter(window_size=5, threshold=0.6) if enable_temporal_filter else None
        
        # Adaptive FPS
        self.enable_adaptive_fps = enable_adaptive_fps
        self.fps_controller = AdaptiveFPSController(base_fps=fps) if enable_adaptive_fps else None
        
        # Camera and drone
        self.camera = CameraInterface(source=camera_source, fps=fps)
        self.drone = None
        
        # Threading
        self.running = False
        self.detection_thread = None
        self.frame_lock = Lock()
        self.current_frame = None
        self.current_detections = []
        
        # Statistics
        self.frame_count = 0
        self.processed_frame_count = 0
        self.detection_count = 0
        self.start_time = None
        self.skipped_frames = 0
        
        # Performance tracking
        self.avg_inference_time = 0.0
        self.total_inference_time = 0.0
        
        logger.info(f"Enhanced detector initialized - Temporal Filter: {enable_temporal_filter}, Adaptive FPS: {enable_adaptive_fps}")
        
    def connect_drone(self, drone_type=None, connection_string=None):
        """Connect to drone (optional)"""
        try:
            from src.drone.drone_controller import DroneController, DroneType
            
            if drone_type is None:
                drone_type = DroneType.SIMULATOR
            
            self.drone = DroneController(
                drone_type=drone_type,
                connection_string=connection_string or "udp:127.0.0.1:14550"
            )
            if self.drone.connect():
                logger.info("Drone connected successfully")
                return True
            else:
                logger.warning("Failed to connect drone")
                return False
        except Exception as e:
            logger.warning(f"Failed to connect drone: {e}")
            return False
    
    def start(self):
        """Start real-time detection"""
        if self.running:
            logger.warning("Detector already running")
            return False
        
        # Connect camera
        if not self.camera.connect():
            logger.error("Failed to connect to camera")
            return False
        
        # Start detection thread
        self.running = True
        self.start_time = time.time()
        self.detection_thread = Thread(target=self._detection_loop, daemon=True)
        self.detection_thread.start()
        
        logger.info("Enhanced real-time detection started")
        return True
    
    def stop(self):
        """Stop real-time detection"""
        self.running = False
        
        if self.detection_thread:
            self.detection_thread.join(timeout=2.0)
        
        self.camera.disconnect()
        
        if self.drone:
            self.drone.disconnect()
        
        logger.info("Real-time detection stopped")
    
    def _should_process_frame(self) -> bool:
        """Determine if current frame should be processed (frame skipping logic)"""
        if self.frame_skip == 0:
            return True
        
        return (self.frame_count % (self.frame_skip + 1)) == 0
    
    def _detection_loop(self):
        """Main detection loop with optimizations"""
        frame_time = 1.0 / self.target_fps
        
        while self.running:
            loop_start = time.time()
            
            try:
                # Read frame
                ret, frame = self.camera.read()
                if not ret or frame is None:
                    time.sleep(0.01)
                    continue
                
                self.frame_count += 1
                
                # Frame skipping logic
                if not self._should_process_frame():
                    self.skipped_frames += 1
                    # Still update current frame for display
                    with self.frame_lock:
                        self.current_frame = frame.copy()
                    continue
                
                self.processed_frame_count += 1
                
                # Detect fire
                inference_start = time.time()
                detections = self.detector.detect(frame, preprocess=True)
                inference_time = time.time() - inference_start
                
                self.total_inference_time += inference_time
                self.avg_inference_time = self.total_inference_time / self.processed_frame_count
                
                # Filter by confidence
                filtered_detections = self.confidence_filter.filter(detections)
                
                # Classify sizes
                classified_detections = []
                for det in filtered_detections:
                    size = self.size_classifier.classify(det, frame.shape[1], frame.shape[0])
                    det['size'] = size.value
                    classified_detections.append(det)
                
                # Temporal filtering
                has_detections = len(classified_detections) > 0
                confirmed_detection = has_detections
                
                if self.enable_temporal_filter and self.temporal_filter:
                    confirmed_detection = self.temporal_filter.update(has_detections)
                
                # Update detection count
                if confirmed_detection and classified_detections:
                    self.detection_count += len(classified_detections)
                
                # Update current frame and detections
                with self.frame_lock:
                    self.current_frame = frame.copy()
                    # Only update detections if confirmed
                    self.current_detections = classified_detections if confirmed_detection else []
                
                # Get telemetry if drone connected
                telemetry = None
                if self.drone and self.drone.is_connected():
                    telemetry = self.drone.get_telemetry()
                
                # Call callback if provided and detections confirmed
                if self.callback and confirmed_detection and classified_detections:
                    try:
                        self.callback(classified_detections, frame, telemetry)
                    except Exception as e:
                        logger.error(f"Callback error: {e}")
                
                # Adaptive FPS control
                if self.enable_adaptive_fps and self.fps_controller:
                    self.fps_controller.update(inference_time, has_detections)
                    frame_time = self.fps_controller.get_frame_time()
                
                # Maintain FPS
                elapsed = time.time() - loop_start
                sleep_time = max(0, frame_time - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    
            except Exception as e:
                logger.error(f"Detection loop error: {e}")
                time.sleep(0.1)
    
    def get_frame(self) -> Optional[np.ndarray]:
        """Get current frame"""
        with self.frame_lock:
            return self.current_frame.copy() if self.current_frame is not None else None
    
    def get_detections(self) -> List[Dict]:
        """Get current detections"""
        with self.frame_lock:
            return self.current_detections.copy()
    
    def get_statistics(self) -> Dict:
        """Get comprehensive detection statistics"""
        elapsed = time.time() - self.start_time if self.start_time else 0
        actual_fps = self.frame_count / elapsed if elapsed > 0 else 0
        processing_fps = self.processed_frame_count / elapsed if elapsed > 0 else 0
        
        stats = {
            'frame_count': self.frame_count,
            'processed_frames': self.processed_frame_count,
            'skipped_frames': self.skipped_frames,
            'detection_count': self.detection_count,
            'actual_fps': actual_fps,
            'processing_fps': processing_fps,
            'avg_inference_time': self.avg_inference_time,
            'elapsed_time': elapsed,
            'detection_rate': (self.detection_count / self.processed_frame_count * 100) if self.processed_frame_count > 0 else 0
        }
        
        # Add adaptive FPS info
        if self.enable_adaptive_fps and self.fps_controller:
            stats['current_target_fps'] = self.fps_controller.current_fps
        
        # Add detector stats
        detector_stats = self.detector.get_statistics()
        stats.update({
            'model_path': detector_stats.get('model_path'),
            'model_type': detector_stats.get('model_type'),
            'device': detector_stats.get('device')
        })
        
        return stats
    
    def update_confidence_threshold(self, threshold: float):
        """Update confidence threshold"""
        self.confidence_threshold = threshold
        self.detector.update_confidence_threshold(threshold)
        self.confidence_filter.update_threshold(threshold)
        logger.info(f"Confidence threshold updated to {threshold}")
    
    def enable_preprocessing(self, enable: bool = True):
        """Enable/disable preprocessing"""
        self.detector.enable_preprocessing(enable)
    
    def enable_denoising(self, enable: bool = True):
        """Enable/disable denoising"""
        self.detector.enable_denoising(enable)


def main():
    """Test enhanced real-time detector"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced Real-time Fire Detection')
    parser.add_argument('--model', type=str, default='Model/best.pt',
                       help='Path to model weights or Hugging Face model ID')
    parser.add_argument('--source', type=str, default='0',
                       help='Camera source')
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='Confidence threshold')
    parser.add_argument('--fps', type=int, default=30,
                       help='Target FPS')
    parser.add_argument('--img-size', type=int, default=640,
                       help='Input image size')
    parser.add_argument('--half', action='store_true',
                       help='Use FP16 half-precision')
    parser.add_argument('--tta', action='store_true',
                       help='Use Test Time Augmentation')
    parser.add_argument('--no-temporal', action='store_true',
                       help='Disable temporal filtering')
    parser.add_argument('--no-adaptive-fps', action='store_true',
                       help='Disable adaptive FPS')
    parser.add_argument('--frame-skip', type=int, default=0,
                       help='Number of frames to skip (0 = process all)')
    
    args = parser.parse_args()
    
    # Detection callback
    def on_detection(detections, frame, telemetry):
        logger.info(f"🔥 Fire detected! Count: {len(detections)}")
        for det in detections:
            logger.info(f"  - {det['class_name']}: {det['confidence']:.2f}, Size: {det['size']}, Area: {det.get('area', 0)}")
    
    # Create detector
    detector = EnhancedRealtimeDetector(
        model_path=args.model,
        camera_source=args.source,
        confidence_threshold=args.confidence,
        fps=args.fps,
        img_size=args.img_size,
        use_half=args.half,
        use_tta=args.tta,
        callback=on_detection,
        enable_temporal_filter=not args.no_temporal,
        enable_adaptive_fps=not args.no_adaptive_fps,
        frame_skip=args.frame_skip
    )
    
    # Start detection
    if detector.start():
        logger.info("Press Ctrl+C to stop or 'q' to quit")
        try:
            while True:
                # Get and display frame
                frame = detector.get_frame()
                if frame is not None:
                    detections = detector.get_detections()
                    
                    # Draw detections
                    for det in detections:
                        x1, y1, x2, y2 = det['bbox']
                        confidence = det['confidence']
                        
                        # Color based on confidence
                        if confidence > 0.8:
                            color = (0, 0, 255)  # Red - high confidence
                        elif confidence > 0.6:
                            color = (0, 165, 255)  # Orange - medium
                        else:
                            color = (0, 255, 255)  # Yellow - low
                        
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        label = f"{det['class_name']}: {confidence:.2f} ({det['size']})"
                        cv2.putText(frame, label, (x1, y1 - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    # Show statistics
                    stats = detector.get_statistics()
                    y_offset = 30
                    cv2.putText(frame, f"FPS: {stats['actual_fps']:.1f} (Processing: {stats['processing_fps']:.1f})", 
                               (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    y_offset += 25
                    cv2.putText(frame, f"Detections: {stats['detection_count']} | Frames: {stats['frame_count']}", 
                               (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    y_offset += 25
                    cv2.putText(frame, f"Inference: {stats['avg_inference_time']*1000:.1f}ms", 
                               (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    cv2.imshow('Enhanced Fire Detection', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                time.sleep(0.01)
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            detector.stop()
            cv2.destroyAllWindows()
            
            # Print final statistics
            stats = detector.get_statistics()
            logger.info("="*60)
            logger.info("Final Statistics:")
            for key, value in stats.items():
                logger.info(f"  {key}: {value}")
            logger.info("="*60)
    else:
        logger.error("Failed to start detector")


if __name__ == "__main__":
    main()
