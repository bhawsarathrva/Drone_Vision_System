"""
Video streaming utilities for fire detection
"""
import cv2
import numpy as np
from typing import Optional, Callable
from threading import Thread
import time
from loguru import logger

class VideoStream:
    def __init__(
        self,
        source: str = "0",
        fps: int = 30,
        resolution: tuple = (1920, 1080)
    ):
        """
        Initialize video stream
        
        Args:
            source: Video source (camera index, RTSP URL, or file path)
            fps: Target FPS
            resolution: Target resolution (width, height)
        """
        self.source = source
        self.fps = fps
        self.resolution = resolution
        
        self.cap = None
        self.running = False
        self.frame = None
        
    def start(self) -> bool:
        """Start video stream"""
        try:
            # Open video source
            if self.source.isdigit():
                self.cap = cv2.VideoCapture(int(self.source))
            else:
                self.cap = cv2.VideoCapture(self.source)
            
            if not self.cap.isOpened():
                logger.error(f"Failed to open video source: {self.source}")
                return False
            
            # Set properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            self.running = True
            logger.info(f"Video stream started: {self.source}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start video stream: {e}")
            return False
    
    def read(self) -> Optional[np.ndarray]:
        """Read frame from stream"""
        if not self.running or self.cap is None:
            return None
        
        ret, frame = self.cap.read()
        if ret:
            return frame
        return None
    
    def stop(self):
        """Stop video stream"""
        self.running = False
        if self.cap:
            self.cap.release()
        logger.info("Video stream stopped")
    
    def get_properties(self) -> dict:
        """Get stream properties"""
        if self.cap is None:
            return {}
        
        return {
            'width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': self.cap.get(cv2.CAP_PROP_FPS),
            'frame_count': int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        }
    
    def is_opened(self) -> bool:
        """Check if stream is open"""
        return self.cap is not None and self.cap.isOpened()

