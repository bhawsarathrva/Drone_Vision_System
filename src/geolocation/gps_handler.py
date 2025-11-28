"""
GPS handling utilities for fire detection
"""
import time
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from loguru import logger

@dataclass
class GPSData:
    """GPS data structure"""
    latitude: float
    longitude: float
    altitude: float
    heading: float
    speed: float
    satellite_count: int
    fix_type: int
    timestamp: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'latitude': self.latitude,
            'longitude': self.longitude,
            'altitude': self.altitude,
            'heading': self.heading,
            'speed': self.speed,
            'satellite_count': self.satellite_count,
            'fix_type': self.fix_type,
            'timestamp': self.timestamp
        }

class GPSHandler:
    def __init__(self, drone_controller=None):
        """
        Initialize GPS handler
        
        Args:
            drone_controller: Drone controller instance (optional)
        """
        self.drone_controller = drone_controller
        self.current_gps = None
        self.gps_history = []
        self.max_history = 1000
    
    def get_current_gps(self) -> Optional[GPSData]:
        """Get current GPS data"""
        if self.drone_controller and self.drone_controller.is_connected():
            telemetry = self.drone_controller.get_telemetry()
            
            self.current_gps = GPSData(
                latitude=telemetry.get('latitude', 0.0),
                longitude=telemetry.get('longitude', 0.0),
                altitude=telemetry.get('altitude', 0.0),
                heading=telemetry.get('heading', 0.0),
                speed=telemetry.get('ground_speed', 0.0),
                satellite_count=telemetry.get('satellite_count', 0),
                fix_type=telemetry.get('gps_fix_type', 0),
                timestamp=time.time()
            )
            
            # Add to history
            self.gps_history.append(self.current_gps)
            if len(self.gps_history) > self.max_history:
                self.gps_history.pop(0)
        
        return self.current_gps
    
    def update_gps(
        self,
        latitude: float,
        longitude: float,
        altitude: float = 0.0,
        heading: float = 0.0,
        speed: float = 0.0,
        satellite_count: int = 0,
        fix_type: int = 0
    ):
        """Update GPS data manually"""
        self.current_gps = GPSData(
            latitude=latitude,
            longitude=longitude,
            altitude=altitude,
            heading=heading,
            speed=speed,
            satellite_count=satellite_count,
            fix_type=fix_type,
            timestamp=time.time()
        )
        
        # Add to history
        self.gps_history.append(self.current_gps)
        if len(self.gps_history) > self.max_history:
            self.gps_history.pop(0)
    
    def get_gps_history(self, limit: int = None) -> list:
        """Get GPS history"""
        if limit:
            return self.gps_history[-limit:]
        return self.gps_history
    
    def is_valid(self) -> bool:
        """Check if GPS data is valid"""
        if self.current_gps is None:
            return False
        
        # Check if coordinates are valid
        if not (-90 <= self.current_gps.latitude <= 90):
            return False
        if not (-180 <= self.current_gps.longitude <= 180):
            return False
        
        # Check if fix is valid (fix_type >= 2 for 2D/3D fix)
        if self.current_gps.fix_type < 2:
            return False
        
        return True
    
    def get_position(self) -> Optional[Tuple[float, float, float]]:
        """Get current position (lat, lon, alt)"""
        if self.current_gps:
            return (
                self.current_gps.latitude,
                self.current_gps.longitude,
                self.current_gps.altitude
            )
        return None
    
    def clear_history(self):
        """Clear GPS history"""
        self.gps_history = []

