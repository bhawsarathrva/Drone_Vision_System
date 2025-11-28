"""
Camera interface for various video sources
Supports webcam, RTSP streams, HTTP streams, and video files
"""
import cv2
import numpy as np
import threading
from queue import Queue
from typing import Optional, Tuple, Dict
import time
from loguru import logger

class CameraInterface:
    def __init__(
        self,
        source: str = "0",
        resolution: Tuple[int, int] = (1920, 1080),
        fps: int = 30,
        buffer_size: int = 5
    ):
        """
        Initialize camera interface
        
        Args:
            source: Video source (camera index, RTSP URL, file path)
            resolution: Desired resolution (width, height)
            fps: Target frames per second
            buffer_size: Frame buffer size
        """
        self.source = source
        self.resolution = resolution
        self.target_fps = fps
        self.buffer_size = buffer_size
        
        # Video capture
        self.cap = None
        self.is_camera = False
        self.is_stream = False
        self.is_file = False
        
        # Frame buffer
        self.frame_queue = Queue(maxsize=buffer_size)
        
        # Threading
        self.running = False
        self.capture_thread = None
        
        # Statistics
        self.frame_count = 0
        self.dropped_frames = 0
        self.actual_fps = 0
        
    def connect(self) -> bool:
        """
        Connect to video source
        
        Returns:
            True if connected successfully
        """
        try:
            # Determine source type
            source_value = self.source
            if str(self.source).isdigit():
                # Camera index
                source_value = int(self.source)
                self.is_camera = True
                logger.info(f"Opening camera {source_value}...")
            elif str(self.source).startswith('rtsp://') or str(self.source).startswith('http://') or str(self.source).startswith('https://'):
                # Network stream
                self.is_stream = True
                source_value = self.source
                logger.info(f"Connecting to stream: {self.source}...")
            else:
                # Video file
                self.is_file = True
                source_value = self.source
                logger.info(f"Opening video file: {self.source}...")
            
            # Open video capture
            self.cap = cv2.VideoCapture(source_value)
            
            if not self.cap.isOpened():
                logger.error(f"Failed to open video source: {self.source}")
                return False
            
            # Set camera properties if it's a camera
            if self.is_camera:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
                self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
                
                # Try to set buffer size to reduce latency
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Get actual properties
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            logger.info(f"Video source opened successfully")
            logger.info(f"Resolution: {actual_width}x{actual_height}, FPS: {actual_fps}")
            
            # Start capture thread
            self.running = True
            self.capture_thread = threading.Thread(
                target=self._capture_loop,
                daemon=True
            )
            self.capture_thread.start()
            
            return True
            
        except Exception as e:
            logger.error(f"Connection error: {e}")
            return False
    
    def _capture_loop(self):
        """Continuous frame capture loop"""
        fps_start = time.time()
        fps_count = 0
        
        while self.running:
            try:
                ret, frame = self.cap.read()
                
                if not ret:
                    if self.is_file:
                        # End of video file
                        logger.info("End of video file reached")
                        break
                    else:
                        logger.warning("Failed to read frame from stream")
                        time.sleep(0.01)
                    continue
                
                self.frame_count += 1
                fps_count += 1
                
                # Calculate FPS
                if time.time() - fps_start >= 1.0:
                    self.actual_fps = fps_count / (time.time() - fps_start)
                    fps_start = time.time()
                    fps_count = 0
                
                # Add to queue
                if self.frame_queue.full():
                    # Remove oldest frame if buffer is full
                    try:
                        self.frame_queue.get_nowait()
                        self.dropped_frames += 1
                    except:
                        pass
                
                self.frame_queue.put(frame)
                
            except Exception as e:
                logger.error(f"Capture error: {e}")
                time.sleep(0.1)
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read next frame from buffer
        
        Returns:
            (success, frame)
        """
        try:
            frame = self.frame_queue.get(timeout=1.0)
            return True, frame
        except:
            return False, None
    
    def get_frame(self) -> Optional[np.ndarray]:
        """
        Get latest frame (non-blocking)
        
        Returns:
            Frame or None
        """
        try:
            return self.frame_queue.get_nowait()
        except:
            return None
    
    def get_properties(self) -> Dict:
        """Get video source properties"""
        if self.cap is None:
            return {}
        
        try:
            return {
                'width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'fps': self.cap.get(cv2.CAP_PROP_FPS),
                'frame_count': self.frame_count,
                'dropped_frames': self.dropped_frames,
                'actual_fps': self.actual_fps,
                'buffer_size': self.frame_queue.qsize(),
                'is_camera': self.is_camera,
                'is_stream': self.is_stream,
                'is_file': self.is_file,
                'source': str(self.source)
            }
        except Exception as e:
            logger.error(f"Error getting properties: {e}")
            return {}
    
    def is_opened(self) -> bool:
        """Check if video source is open"""
        return self.cap is not None and self.cap.isOpened()
    
    def disconnect(self):
        """Disconnect from video source"""
        self.running = False
        
        if self.capture_thread:
            self.capture_thread.join(timeout=2.0)
        
        if self.cap:
            self.cap.release()
        
        # Clear queue
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except:
                break
        
        logger.info("Camera disconnected")
    
    def __del__(self):
        """Cleanup on deletion"""
        self.disconnect()

def main():
    """Test camera interface"""
    # Test with webcam (change to your source)
    camera = CameraInterface(
        source="0",  # or "rtsp://..." or "video.mp4"
        resolution=(640, 480),
        fps=30
    )
    
    if camera.connect():
        logger.info("Camera connected successfully")
        logger.info("Press 'q' to quit")
        
        frame_num = 0
        try:
            while True:
                ret, frame = camera.read()
                
                if ret and frame is not None:
                    frame_num += 1
                    
                    # Get properties
                    props = camera.get_properties()
                    
                    # Display info on frame
                    info_text = f"Frame: {frame_num} | FPS: {props.get('actual_fps', 0):.1f}"
                    cv2.putText(frame, info_text, (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    # Show frame
                    cv2.imshow('Camera Test', frame)
                    
                    # Log stats every 30 frames
                    if frame_num % 30 == 0:
                        logger.info(f"Captured: {props.get('frame_count', 0)} | "
                                  f"Dropped: {props.get('dropped_frames', 0)} | "
                                  f"FPS: {props.get('actual_fps', 0):.1f}")
                
                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            camera.disconnect()
            cv2.destroyAllWindows()
            logger.info("Test completed")
    else:
        logger.error("Failed to connect to camera")

if __name__ == "__main__":
    main()