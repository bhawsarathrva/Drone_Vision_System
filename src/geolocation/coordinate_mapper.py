"""
Coordinate mapping utilities for fire detection
"""
import math
from typing import Tuple, List, Dict
from geopy.distance import geodesic
from loguru import logger

class CoordinateMapper:
    def __init__(self):
        """Initialize coordinate mapper"""
        pass
    
    def pixel_to_gps(
        self,
        pixel_x: int,
        pixel_y: int,
        image_width: int,
        image_height: int,
        drone_lat: float,
        drone_lon: float,
        drone_altitude: float,
        drone_heading: float,
        camera_fov_horizontal: float = 60.0,
        camera_fov_vertical: float = 45.0
    ) -> Tuple[float, float]:
        """
        Convert pixel coordinates to GPS coordinates
        
        Args:
            pixel_x: Pixel X coordinate
            pixel_y: Pixel Y coordinate
            image_width: Image width in pixels
            image_height: Image height in pixels
            drone_lat: Drone latitude
            drone_lon: Drone longitude
            drone_altitude: Drone altitude in meters
            drone_heading: Drone heading in degrees
            camera_fov_horizontal: Camera horizontal FOV in degrees
            camera_fov_vertical: Camera vertical FOV in degrees
            
        Returns:
            (latitude, longitude) of the point
        """
        # Normalize pixel coordinates to [-1, 1]
        norm_x = (pixel_x / image_width) * 2 - 1
        norm_y = (pixel_y / image_height) * 2 - 1
        
        # Calculate angles from center
        angle_x = norm_x * (camera_fov_horizontal / 2)
        angle_y = norm_y * (camera_fov_vertical / 2)
        
        # Calculate distance from drone to point
        # Using simple trigonometry (assuming flat ground)
        distance_horizontal = drone_altitude * math.tan(math.radians(angle_y))
        distance_forward = distance_horizontal / math.cos(math.radians(angle_x))
        
        # Calculate bearing
        bearing = (drone_heading + angle_x) % 360
        
        # Calculate GPS coordinates
        point = geodesic(meters=distance_forward).destination(
            (drone_lat, drone_lon),
            bearing
        )
        
        return (point.latitude, point.longitude)
    
    def bbox_to_gps(
        self,
        bbox: List[int],
        image_width: int,
        image_height: int,
        drone_lat: float,
        drone_lon: float,
        drone_altitude: float,
        drone_heading: float,
        camera_fov_horizontal: float = 60.0,
        camera_fov_vertical: float = 45.0
    ) -> Dict[str, Tuple[float, float]]:
        """
        Convert bounding box to GPS coordinates
        
        Args:
            bbox: Bounding box [x1, y1, x2, y2]
            image_width: Image width
            image_height: Image height
            drone_lat: Drone latitude
            drone_lon: Drone longitude
            drone_altitude: Drone altitude
            drone_heading: Drone heading
            camera_fov_horizontal: Camera horizontal FOV
            camera_fov_vertical: Camera vertical FOV
            
        Returns:
            Dictionary with center, top_left, top_right, bottom_left, bottom_right GPS coordinates
        """
        x1, y1, x2, y2 = bbox
        
        # Calculate center
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # Calculate corner coordinates
        corners = {
            'center': self.pixel_to_gps(
                center_x, center_y, image_width, image_height,
                drone_lat, drone_lon, drone_altitude, drone_heading,
                camera_fov_horizontal, camera_fov_vertical
            ),
            'top_left': self.pixel_to_gps(
                x1, y1, image_width, image_height,
                drone_lat, drone_lon, drone_altitude, drone_heading,
                camera_fov_horizontal, camera_fov_vertical
            ),
            'top_right': self.pixel_to_gps(
                x2, y1, image_width, image_height,
                drone_lat, drone_lon, drone_altitude, drone_heading,
                camera_fov_horizontal, camera_fov_vertical
            ),
            'bottom_left': self.pixel_to_gps(
                x1, y2, image_width, image_height,
                drone_lat, drone_lon, drone_altitude, drone_heading,
                camera_fov_horizontal, camera_fov_vertical
            ),
            'bottom_right': self.pixel_to_gps(
                x2, y2, image_width, image_height,
                drone_lat, drone_lon, drone_altitude, drone_heading,
                camera_fov_horizontal, camera_fov_vertical
            )
        }
        
        return corners
    
    def calculate_distance(
        self,
        lat1: float,
        lon1: float,
        lat2: float,
        lon2: float
    ) -> float:
        """
        Calculate distance between two GPS points in meters
        
        Args:
            lat1: First point latitude
            lon1: First point longitude
            lat2: Second point latitude
            lon2: Second point longitude
            
        Returns:
            Distance in meters
        """
        return geodesic((lat1, lon1), (lat2, lon2)).meters
    
    def calculate_bearing(
        self,
        lat1: float,
        lon1: float,
        lat2: float,
        lon2: float
    ) -> float:
        """
        Calculate bearing from point 1 to point 2
        
        Args:
            lat1: First point latitude
            lon1: First point longitude
            lat2: Second point latitude
            lon2: Second point longitude
            
        Returns:
            Bearing in degrees
        """
        point1 = (lat1, lon1)
        point2 = (lat2, lon2)
        
        # Calculate initial bearing
        bearing = geodesic(point1, point2).initial_bearing
        
        return bearing

