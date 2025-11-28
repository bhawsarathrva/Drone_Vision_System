"""
Real-time fire detection using camera stream
"""
import cv2
import numpy as np
from typing import Optional, Callable
from threading import Thread, Lock
import time
from loguru import logger

from src.detection.fire_detector import FireDetector
from src.detection.confidence_filter import ConfidenceFilter
from src.detection.size_classifier import SizeClassifier, FireSize
from src.drone.camera_interface import CameraInterface
from src.drone.drone_controller import DroneController

class RealtimeDetector:
    def __init__(
        self,
        model_path: str = "Model/best.pt",
        camera_source: str = "0",
        confidence_threshold: float = 0.5,
        fps: int = 30,
        callback: Optional[Callable] = None
    ):
        """
        Initialize real-time detector
        
        Args:
            model_path: Path to YOLOv8 model
            camera_source: Camera source (index, RTSP URL, or file path)
            confidence_threshold: Confidence threshold for detections
            fps: Target FPS
            callback: Callback function for detections (detections, frame, telemetry)
        """
        self.model_path = model_path
        self.camera_source = camera_source
        self.confidence_threshold = confidence_threshold
        self.fps = fps
        self.callback = callback
        
        # Initialize components
        self.detector = FireDetector(
            model_path=model_path,
            confidence_threshold=confidence_threshold
        )
        self.confidence_filter = ConfidenceFilter(min_confidence=confidence_threshold)
        self.size_classifier = SizeClassifier()
        
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
        self.detection_count = 0
        self.start_time = None
        
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
                logger.info("Drone connected")
            else:
                logger.warning("Failed to connect drone")
        except Exception as e:
            logger.warning(f"Failed to connect drone: {e}")
    
    def start(self):
        """Start real-time detection"""
        if self.running:
            logger.warning("Detector already running")
            return
        
        # Connect camera
        if not self.camera.connect():
            logger.error("Failed to connect to camera")
            return False
        
        # Start detection thread
        self.running = True
        self.start_time = time.time()
        self.detection_thread = Thread(target=self._detection_loop, daemon=True)
        self.detection_thread.start()
        
        logger.info("Real-time detection started")
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
    
    def _detection_loop(self):
        """Main detection loop"""
        frame_time = 1.0 / self.fps
        
        while self.running:
            loop_start = time.time()
            
            try:
                # Read frame
                ret, frame = self.camera.read()
                if not ret or frame is None:
                    time.sleep(0.01)
                    continue
                
                self.frame_count += 1
                
                # Detect fire
                detections = self.detector.detect(frame)
                
                # Filter by confidence
                filtered_detections = self.confidence_filter.filter(detections)
                
                # Classify sizes
                classified_detections = []
                for det in filtered_detections:
                    size = self.size_classifier.classify(det, frame.shape[1], frame.shape[0])
                    det['size'] = size.value
                    classified_detections.append(det)
                
                if classified_detections:
                    self.detection_count += len(classified_detections)
                
                # Update current frame and detections
                with self.frame_lock:
                    self.current_frame = frame.copy()
                    self.current_detections = classified_detections
                
                # Get telemetry if drone connected
                telemetry = None
                if self.drone and self.drone.is_connected():
                    telemetry = self.drone.get_telemetry()
                
                # Call callback if provided and detections found
                if self.callback and classified_detections:
                    try:
                        self.callback(classified_detections, frame, telemetry)
                    except Exception as e:
                        logger.error(f"Callback error: {e}")
                
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
    
    def get_detections(self) -> list:
        """Get current detections"""
        with self.frame_lock:
            return self.current_detections.copy()
    
    def get_statistics(self) -> dict:
        """Get detection statistics"""
        elapsed = time.time() - self.start_time if self.start_time else 0
        fps = self.frame_count / elapsed if elapsed > 0 else 0
        
        return {
            'frame_count': self.frame_count,
            'detection_count': self.detection_count,
            'fps': fps,
            'elapsed_time': elapsed,
            'detection_rate': (self.detection_count / self.frame_count * 100) if self.frame_count > 0 else 0
        }
    
    def update_confidence_threshold(self, threshold: float):
        """Update confidence threshold"""
        self.confidence_threshold = threshold
        self.detector.update_confidence_threshold(threshold)
        self.confidence_filter.update_threshold(threshold)
        logger.info(f"Confidence threshold updated to {threshold}")

def main():
    """Test real-time detector"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Real-time fire detection')
    parser.add_argument('--model', type=str, default='Model/best.pt',
                       help='Path to model weights')
    parser.add_argument('--source', type=str, default='0',
                       help='Camera source')
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='Confidence threshold')
    parser.add_argument('--fps', type=int, default=30,
                       help='Target FPS')
    
    args = parser.parse_args()
    
    # Detection callback
    def on_detection(detections, frame, telemetry):
        logger.info(f"Fire detected! Count: {len(detections)}")
        for det in detections:
            logger.info(f"  - {det['class_name']}: {det['confidence']:.2f}, Size: {det['size']}")
    
    # Create detector
    detector = RealtimeDetector(
        model_path=args.model,
        camera_source=args.source,
        confidence_threshold=args.confidence,
        fps=args.fps,
        callback=on_detection
    )
    
    # Start detection
    if detector.start():
        logger.info("Press Ctrl+C to stop")
        try:
            while True:
                # Get and display frame
                frame = detector.get_frame()
                if frame is not None:
                    detections = detector.get_detections()
                    
                    # Draw detections
                    for det in detections:
                        x1, y1, x2, y2 = det['bbox']
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        label = f"{det['class_name']}: {det['confidence']:.2f} ({det['size']})"
                        cv2.putText(frame, label, (x1, y1 - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    
                    # Show statistics
                    stats = detector.get_statistics()
                    stats_text = f"FPS: {stats['fps']:.1f} | Detections: {stats['detection_count']}"
                    cv2.putText(frame, stats_text, (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    cv2.imshow('Real-time Fire Detection', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                time.sleep(0.01)
        except KeyboardInterrupt:
            pass
        finally:
            detector.stop()
            cv2.destroyAllWindows()
    else:
        logger.error("Failed to start detector")

if __name__ == "__main__":
    main()

