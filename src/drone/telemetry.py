"""
Telemetry utilities for drone fire detection
"""
import time
from typing import Dict, Optional
from dataclasses import dataclass, asdict
from loguru import logger

@dataclass
class TelemetryData:
    """Telemetry data structure"""
    latitude: float
    longitude: float
    altitude: float
    heading: float
    ground_speed: float
    vertical_speed: float
    battery_voltage: float
    battery_percent: float
    armed: bool
    mode: str
    satellite_count: int
    gps_fix_type: int
    timestamp: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)

class TelemetryHandler:
    def __init__(self, drone_controller=None):
        """
        Initialize telemetry handler
        
        Args:
            drone_controller: Drone controller instance
        """
        self.drone_controller = drone_controller
        self.current_telemetry = None
        self.telemetry_history = []
        self.max_history = 1000
    
    def get_telemetry(self) -> Optional[TelemetryData]:
        """Get current telemetry data"""
        if self.drone_controller and self.drone_controller.is_connected():
            telemetry_dict = self.drone_controller.get_telemetry()
            
            self.current_telemetry = TelemetryData(
                latitude=telemetry_dict.get('latitude', 0.0),
                longitude=telemetry_dict.get('longitude', 0.0),
                altitude=telemetry_dict.get('altitude', 0.0),
                heading=telemetry_dict.get('heading', 0.0),
                ground_speed=telemetry_dict.get('ground_speed', 0.0),
                vertical_speed=telemetry_dict.get('vertical_speed', 0.0),
                battery_voltage=telemetry_dict.get('battery_voltage', 0.0),
                battery_percent=telemetry_dict.get('battery_percent', 100.0),
                armed=telemetry_dict.get('armed', False),
                mode=telemetry_dict.get('mode', 'UNKNOWN'),
                satellite_count=telemetry_dict.get('satellite_count', 0),
                gps_fix_type=telemetry_dict.get('gps_fix_type', 0),
                timestamp=time.time()
            )
            
            # Add to history
            self.telemetry_history.append(self.current_telemetry)
            if len(self.telemetry_history) > self.max_history:
                self.telemetry_history.pop(0)
        
        return self.current_telemetry
    
    def get_telemetry_history(self, limit: int = None) -> list:
        """Get telemetry history"""
        if limit:
            return self.telemetry_history[-limit:]
        return self.telemetry_history
    
    def clear_history(self):
        """Clear telemetry history"""
        self.telemetry_history = []
    
    def is_valid(self) -> bool:
        """Check if telemetry data is valid"""
        if self.current_telemetry is None:
            return False
        
        # Check GPS validity
        if not (-90 <= self.current_telemetry.latitude <= 90):
            return False
        if not (-180 <= self.current_telemetry.longitude <= 180):
            return False
        
        # Check GPS fix
        if self.current_telemetry.gps_fix_type < 2:
            return False
        
        return True

